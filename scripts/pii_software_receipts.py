#!/usr/bin/env python3
"""Text-free score receipts for the paper, and a verifier that recomputes it.

The paper's comparisons are pooled from per-document counts
`[true positives, predicted regions, gold regions]`. `build` (maintainer
side) copies those counts out of the saved score archives into a columnar
receipt that keeps document ids, languages and source groups but drops every
text preview, prediction and absolute path; it then proves the receipt
expands back to the original counts and that no receipt string occurs in any
reference text. `verify` (reader side) recomputes every reported maximum,
fixed-bias score and paired bootstrap interval from the receipts with the
paper's own pooling and resampling code and compares them with the shipped
summaries.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import acli

EVIDENCE = ROOT / "research/pii/frontier/evidence"
RECEIPTS = ROOT / "research/pii/frontier/software/records/receipts"
# name: (score archive, summary, what the paper reports from it)
SOURCES = {
    "o4-comparison": (
        "paper-o4-v1/scores.json.gz",
        "paper-o4-v1/summary.json",
        "Main comparison: 13 systems on human gold and the pooled Ont3 development set; O3-O4 paired intervals",
    ),
    "o4-boundary": (
        "paper-o4-boundary-v2/scores.json.gz",
        "paper-o4-boundary-v2/summary.json",
        "Character-boundary refinement on and off for O4 at zero bias",
    ),
    "gliner-trajectory": (
        "paper-o4-v1/gliner-paper-trajectory-scores.json.gz",
        None,
        "GLiNER2 adaptation checkpoints on the paper populations (development trajectory)",
    ),
}
# Already text-free evidence behind other paper claims, shipped as sanitized
# copies (paths reduced to basenames). Recorded, not recomputed by verify.
EXTRAS = {
    "local-llm-summary.json": "paper-o4-v1/local-llm-summary.json",
    "local-llm-per-input.json.gz": "paper-o4-v1/local-llm-per-input.json.gz",
    "tab-agreement.json": "paper-tab-agreement-v1.json",
    "annotation-volume.json": "paper-o4-v1/annotation-volume.json",
    "annotation-cost.json": "paper-o4-v1/annotation-cost.json",
    "coverage-mixtures-summary.json": "four-corpus-v1/coverage-v2/summary.json",
    "gliner-full-inventory.json": "paper-o4-v1/gliner-full-inventory.json",
    "fresh-fit-summary.json": "software-fresh-fit-v1/summary.json",
}
POLICY_KEYS = ("status", "title_policy", "coverage_policy", "reference_policy", "selection")
DROPPED_KEYS = {"preview", "example", "text", "predictions", "gold_regions", "predicted_regions"}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write_json(path: Path, value) -> None:
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def identity(value: dict) -> dict:
    """A file reference reduced to its basename and hash."""
    return {"name": Path(value["path"]).name, "sha256": value["sha256"]}


def columnar(report: dict) -> dict:
    """Per-document counts, with document identity stored once per population."""
    rows: dict[str, dict] = {}
    systems = {}
    for model, populations in report["systems"].items():
        systems[model] = {}
        for population, entry in populations.items():
            points = []
            for point in entry["points"]:
                views = {}
                for view, items in [
                    *((f"regions/{key}", value["per_input"]) for key, value in point["regions"].items()),
                    *((("typed", point["typed"]["per_input"]),) if "typed" in point else ()),
                ]:
                    order = rows.setdefault(
                        population,
                        {
                            "id": [item["id"] for item in items],
                            "lang": [item["lang"] for item in items],
                            "group": [item["group"] for item in items],
                        },
                    )
                    if [item["id"] for item in items] != order["id"] or [
                        item["group"] for item in items
                    ] != order["group"]:
                        raise ValueError(f"row order differs: {model}/{population}/{view}")
                    views[view] = {"counts": [value for item in items for value in item["counts"]]}
                    if "neutral" in items[0]:
                        views[view]["neutral"] = [item["neutral"] for item in items]
                        views[view]["masked"] = [item["masked"] for item in items]
                points.append({"threshold": point["threshold"], "views": views})
            systems[model][population] = {
                "prediction": {"name": Path(entry["path"]).name, "sha256": entry["sha256"]},
                "checkpoint": Path(str(entry["checkpoint"])).name if entry.get("checkpoint") else None,
                "context": entry.get("context"),
                "expressible": entry.get("expressible"),
                "points": points,
            }
    return {
        "schema": "pii-software-score-receipt-v1",
        "policies": {key: report.get(key) for key in POLICY_KEYS},
        "references": {name: identity(value) for name, value in report["references"].items()},
        "title_sidecar": identity(report["title_sidecar"]) if report.get("title_sidecar") else None,
        "populations": {
            name: {**{k: v for k, v in value.items()}, "rows": rows.get(name)}
            for name, value in report["populations"].items()
        },
        "systems": systems,
    }


def expand(receipt: dict) -> dict:
    """The original score-archive shape, with metrics recomputed from counts."""
    from pii_paper_pooled_eval import metrics

    report = {key: value for key, value in receipt["policies"].items()}
    report["populations"] = {
        name: {k: v for k, v in value.items() if k != "rows"}
        for name, value in receipt["populations"].items()
    }
    report["systems"] = {}
    for model, populations in receipt["systems"].items():
        report["systems"][model] = {}
        for population, entry in populations.items():
            rows = receipt["populations"][population]["rows"]
            points = []
            for point in entry["points"]:
                expanded = {"threshold": point["threshold"], "regions": {}}
                for view, value in point["views"].items():
                    counts = value["counts"]
                    items = []
                    for index, key in enumerate(rows["id"]):
                        item = {
                            "id": key,
                            "lang": rows["lang"][index],
                            "group": rows["group"][index],
                            "counts": counts[3 * index : 3 * index + 3],
                        }
                        if "neutral" in value:
                            item["neutral"] = value["neutral"][index]
                            item["masked"] = value["masked"][index]
                        items.append(item)
                    block = {
                        "metrics": metrics([sum(item["counts"][i] for item in items) for i in range(3)]),
                        "per_input": items,
                    }
                    if view == "typed":
                        expanded["typed"] = block
                    else:
                        expanded["regions"][view.split("/", 1)[1]] = block
                points.append(expanded)
            report["systems"][model][population] = {"points": points}
    return report


def text_free(value, *, path="") -> None:
    """Reject keys that carry text or predictions and absolute path strings."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in DROPPED_KEYS:
                raise ValueError(f"receipt keeps text-bearing key {path}/{key}")
            if key.startswith(("/", "~")):
                raise ValueError(f"receipt keeps an absolute path as a key at {path}: {key[:40]}")
            text_free(item, path=f"{path}/{key}")
    elif isinstance(value, list):
        for item in value[:50]:
            text_free(item, path=path)
    elif isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
        raise ValueError(f"receipt keeps an absolute path at {path}: {value[:40]}")


def sanitized(value):
    """A summary without text-bearing fields; paths reduced to basenames."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in DROPPED_KEYS:
                continue
            if key == "path" and isinstance(item, str):
                out["name"] = Path(item).name
            else:
                out[Path(key).name if key.startswith("/") else key] = sanitized(item)
        return out
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def strings(value, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            strings(item, found)
    elif isinstance(value, list):
        for item in value:
            strings(item, found)
    elif isinstance(value, str):
        found.add(value)


IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+")


def content_like(value: str) -> bool:
    """Strings that could carry document content, not schema vocabulary or ids."""
    return len(value) >= 16 or (len(value) >= 8 and not IDENTIFIER.fullmatch(value))


def reference_texts(report: dict) -> str:
    texts = []
    for value in report["references"].values():
        path = Path(value["path"])
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])
    return "\n\x00\n".join(texts)


def build(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "pii-software-receipts-v1", "receipts": {}}
    for name, (scores, summary, purpose) in SOURCES.items():
        source = EVIDENCE / scores
        report = read_json(source)
        receipt = columnar(report)
        text_free(receipt)
        # Proof the receipt carries exactly the original per-document counts.
        expanded = expand(receipt)
        for model, populations in report["systems"].items():
            for population, entry in populations.items():
                for original, rebuilt in zip(
                    entry["points"], expanded["systems"][model][population]["points"], strict=True
                ):
                    for key, value in original["regions"].items():
                        if value != rebuilt["regions"][key]:
                            raise ValueError(f"expansion differs: {name}/{model}/{population}/{key}")
                    if "typed" in original and original["typed"] != rebuilt["typed"]:
                        raise ValueError(f"typed expansion differs: {name}/{model}/{population}")
        # No receipt string may appear inside any reference document text.
        corpus = reference_texts(report)
        found: set[str] = set()
        strings(receipt, found)
        leaked = sorted(value for value in found if content_like(value) and value in corpus)
        if leaked:
            raise ValueError(f"{name}: receipt strings occur in reference text: {leaked[:5]}")
        target = args.out / f"{name}.json.gz"
        write_json(target, receipt)
        entry = {
            "purpose": purpose,
            "receipt": target.name,
            "receipt_sha256": sha256_bytes(target.read_bytes()),
            "source": {"evidence": scores, "sha256": sha256_bytes(source.read_bytes())},
            "leak_scan": {"strings_checked": len(found), "reference_documents": corpus.count("\n\x00\n") + 1},
        }
        if summary:
            clean = sanitized(read_json(EVIDENCE / summary))
            text_free(clean)
            found = set()
            strings(clean, found)
            leaked = sorted(value for value in found if content_like(value) and value in corpus)
            if leaked:
                raise ValueError(f"{name} summary strings occur in reference text: {leaked[:5]}")
            summary_target = args.out / f"{name}-summary.json"
            write_json(summary_target, clean)
            entry["summary"] = summary_target.name
            entry["summary_source"] = {
                "evidence": summary,
                "sha256": sha256_bytes((EVIDENCE / summary).read_bytes()),
            }
        manifest["receipts"][name] = entry
    manifest["recorded"] = {}
    for name, evidence in EXTRAS.items():
        source = EVIDENCE / evidence
        clean = sanitized(read_json(source))
        text_free(clean)
        target = args.out / name
        write_json(target, clean)
        manifest["recorded"][name] = {
            "source": {"evidence": evidence, "sha256": sha256_bytes(source.read_bytes())},
            "sha256": sha256_bytes(target.read_bytes()),
        }
    (args.out / "receipts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"ok": True, "out": str(args.out.resolve()), "receipts": sorted(manifest["receipts"])}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12


def verify(args) -> dict:
    """Recompute every reported number from the receipts and compare."""
    import copy

    from pii_paper_o4_figures import fixed, paired, pool_ont3
    from pii_paper_pooled_eval import metrics

    manifest = json.loads((args.receipts / "receipts.json").read_text())
    checks, failures = [], []
    for name, entry in manifest["receipts"].items():
        path = args.receipts / entry["receipt"]
        if sha256_bytes(path.read_bytes()) != entry["receipt_sha256"]:
            failures.append(f"{name}: receipt hash differs from manifest")
            continue
        report = expand(read_json(path))
        if "summary" not in entry:
            checks.append({"receipt": name, "check": "expanded", "systems": len(report["systems"])})
            continue
        summary = read_json(args.receipts / entry["summary"])
        pooled = pool_ont3(copy.deepcopy(report))
        for model, populations in summary.get("systems", {}).items():
            for population, views in populations.items():
                points = pooled["systems"][model][population]["points"]
                for coverage in ("80", "100"):
                    best = max(points, key=lambda p: p["regions"][coverage]["metrics"]["F1"])
                    stated = views[coverage]["maximum"]
                    got = best["regions"][coverage]["metrics"]
                    ok = close(got["F1"], stated["F1"]) and best["threshold"] == stated["threshold"]
                    fixed_point = fixed(pooled, model, population)
                    fixed_got = metrics(
                        [
                            sum(r["counts"][i] for r in fixed_point["regions"][coverage]["per_input"])
                            for i in range(3)
                        ]
                    )
                    ok = ok and close(fixed_got["F1"], views[coverage]["fixed"]["F1"])
                    checks.append(
                        {
                            "receipt": name,
                            "system": model,
                            "population": population,
                            "overlap": coverage,
                            "maximum_f1": got["F1"],
                            "fixed_f1": fixed_got["F1"],
                            "ok": ok,
                        }
                    )
                    if not ok:
                        failures.append(f"{name}: {model}/{population}/{coverage}")
        for stated in summary.get("paired", []) + summary.get("comparisons", []):
            got = paired(
                pooled,
                stated["candidate"],
                stated["control"],
                stated["populations"],
                stated["metric"] == "exact typed",
            )
            ok = all(close(got[key], stated[key]) for key in ("candidate_f1", "control_f1", "delta")) and all(
                close(a, b) for a, b in zip(got["ci95"], stated["ci95"], strict=True)
            )
            checks.append(
                {
                    "receipt": name,
                    "paired": f"{stated['candidate']} vs {stated['control']}",
                    "populations": stated["populations"],
                    "metric": stated["metric"],
                    "delta": got["delta"],
                    "ci95": got["ci95"],
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(f"{name}: paired {stated['candidate']}/{stated['control']}")
    result = {
        "ok": not failures,
        "checks": len(checks),
        "failures": failures,
        "details": checks if args.details else None,
    }
    if failures:
        raise ValueError(f"{len(failures)} reported numbers did not recompute: {failures[:5]}")
    return result


def build_parser():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build", help="Derive text-free receipts from saved score archives.")
    build_command.add_argument("--out", type=Path, default=RECEIPTS)
    build_command.set_defaults(action=build)
    verify_command = commands.add_parser("verify", help="Recompute reported numbers from the receipts.")
    verify_command.add_argument("--receipts", type=Path, default=RECEIPTS)
    verify_command.add_argument("--details", action="store_true", help="List every recomputed number")
    verify_command.set_defaults(action=verify)
    acli.add_standard_args(parser)
    for command in (build_command, verify_command):
        acli.add_standard_args(command)
    return parser


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        result = args.action(args)
    except (OSError, ValueError, KeyError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
