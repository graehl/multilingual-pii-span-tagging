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
summaries. It also re-runs the paper's Silver-dev operating-point selection
on the receipts and checks each system's chosen threshold and its F1 on
Gold-7 and Silver-test against the paper.
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
        "gl4-shuffled-o4-v1/trajectory/gliner-paper-trajectory-scores.json.gz",
        None,
        "GL4 checkpoints (shuffled type order) on Gold-7 and Silver-dev (development trajectory)",
    ),
    "gliner-trajectory-unshuffled": (
        "paper-o4-v1/gliner-paper-trajectory-scores.json.gz",
        None,
        "The first, unshuffled GL4 run's checkpoints on Gold-7 and Silver-dev (development trajectory)",
    ),
    # Finer threshold grids behind the Silver-dev operating points, merged
    # over o4-comparison in this order, as the paper's figure script does.
    "operating-point-presidio": (
        "paper-o4-v1/presidio-threshold-v1/scores.json.gz",
        None,
        "Presidio score-threshold sweep (step 0.05) filtered from its saved threshold-0 outputs",
    ),
    "operating-point-refined-grid": (
        "paper-o4-v1/refined-grid-v1/scores.json.gz",
        None,
        "GLiNER2 and GL4 confidence steps of 0.05, Presidio score steps of 0.025, from saved outputs",
    ),
    "operating-point-fine-bias": (
        "paper-o4-v1/fine-bias-v1/scores.json.gz",
        None,
        "Quarter-step O biases over -4..4 for O3, O4 and OpenMed Privacy Filter; the rerun "
        "reproduced every saved bias row by row",
    ),
}
# Silver-dev operating points (the paper's operating-point appendix): the
# shipped summary is recomputed in full from o4-comparison merged with the
# finer grids above.
OPERATING_POINTS = {
    "summary": "operating-points-trust-region.json",
    "source": "paper-o4-v1/operating-points-trust-region.json",
    "rule": "trust-region",
    "comparison": "o4-comparison",
    "grids": ("operating-point-presidio", "operating-point-refined-grid", "operating-point-fine-bias"),
}
# What the paper states at those points: the threshold fixed on Silver-dev
# and 80%-overlap redaction-region F1 (percent, one decimal) on Gold-7
# (`human`) and Silver-test (`heldout`).
PAPER_OPERATING_POINTS = {
    "o4": (0, {"gold7": 88.2, "silver_test": 79.9}),
    "o3": (0.25, {}),
    "ont2": (-2, {}),
    "gliner2": (0.6, {"gold7": 69.1, "silver_test": 72.7}),
    "gliner2-o4": (0.6, {"gold7": 65.4, "silver_test": 64.9}),
    "presidio": (0.2, {"gold7": 57.3}),
    # Silver-test is 43.8496: 43.8, not the 43.9 a two-decimal 43.85 rounds to.
    "opf-openmed-multi2": (-1.5, {"gold7": 34.1, "silver_test": 43.8}),
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
    manifest["operating_points"] = build_operating_points(args.out, manifest)
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


def curve_copy(report: dict) -> dict:
    """A copy whose population and curve entries can be replaced without touching `report`.

    Pooling and grid merging replace whole populations, curves and points but
    never edit a point, so the per-row counts are shared rather than deep-copied.
    """
    return {
        **report,
        "populations": dict(report["populations"]),
        "systems": {
            model: {population: dict(entry) for population, entry in populations.items()}
            for model, populations in report["systems"].items()
        },
    }


def operating_point_curves(reports: dict[str, dict]) -> dict:
    """The main comparison with the finer selection grids merged in, from expanded receipts.

    Uses the figure script's own merge: each finer grid must reproduce every
    coarser point row by row.
    """
    from pii_paper_o4_figures import merge_refined_system

    report = curve_copy(reports[OPERATING_POINTS["comparison"]])
    for name in OPERATING_POINTS["grids"]:
        grid = reports[name]
        if grid.get("status") != "complete":
            raise ValueError(f"{name}: incomplete grid receipt")
        for model in grid["systems"]:
            merge_refined_system(report, grid, model)
    return report


def recompute_operating_points(reports: dict[str, dict]) -> dict:
    """The paper's Silver-dev operating points from expanded receipts, by its selection rule."""
    import contextlib

    from pii_paper_o4_figures import operating_points

    report = operating_point_curves(reports)
    # The figure script narrates each selection on stdout, which carries this tool's result.
    with contextlib.redirect_stdout(sys.stderr):
        return operating_points(report, OPERATING_POINTS["rule"])


def differences(stated, got, path: str = "") -> list[str]:
    """Paths where a recomputed value differs from the stated one (numbers within 1e-12)."""
    if isinstance(stated, dict):
        if not isinstance(got, dict):
            return [path]
        return [
            difference
            for key, value in stated.items()
            for difference in (
                differences(value, got[key], f"{path}/{key}") if key in got else [f"{path}/{key}"]
            )
        ]
    if isinstance(stated, list):
        if not isinstance(got, list) or len(stated) != len(got):
            return [path]
        return [
            difference
            for i, (a, b) in enumerate(zip(stated, got))
            for difference in differences(a, b, f"{path}/{i}")
        ]
    numeric = (int, float)
    if isinstance(stated, numeric) and not isinstance(stated, bool):
        ok = isinstance(got, numeric) and not isinstance(got, bool) and close(stated, got)
        return [] if ok else [path]
    return [] if stated == got else [path]


def operating_point_checks(stated: dict, got: dict) -> tuple[list[dict], list[str]]:
    """Compare a recomputed operating-point summary with the shipped one and the paper."""
    # Sources name the maintainer's score archives; the receipts manifest binds their hashes.
    failures = [
        f"operating points: {path} differs"
        for path in differences({k: v for k, v in stated.items() if k != "sources"}, got)
    ]
    checks = []
    for model, (threshold, f1s) in PAPER_OPERATING_POINTS.items():
        entry = got["systems"][model]
        f1 = {
            name: entry["evaluations"][name]["80"]["selected"]["F1"] * 100
            for name in ("silver_dev", "gold7", "silver_test")
        }
        ok = close(entry["selected_threshold"], threshold) and all(
            f"{f1[name]:.1f}" == f"{value:.1f}" for name, value in f1s.items()
        )
        checks.append(
            {
                "system": model,
                "selected": entry["selected_threshold"],
                "near_optimal": entry["trust_region"]["bounds"],
                **{f"{name}_f1": round(value, 2) for name, value in f1.items()},
                "paper": {"threshold": threshold, **f1s},
                "ok": ok,
            }
        )
        if not ok:
            failures.append(f"operating points: {model} does not give the paper's {threshold} / {f1s}")
    return checks, failures


def build_operating_points(out: Path, manifest: dict) -> dict:
    """Ship the operating-point summary and prove the receipts recompute it."""
    from pii_paper_o4_figures import PRESIDIO_SWEEP, REFINED_SOURCES

    # The grid receipts must be the figure script's own grids, merged in its order.
    expected = [(PRESIDIO_SWEEP, ("presidio",)), *REFINED_SOURCES.items()]
    reports = {}
    for name, (path, models) in zip(OPERATING_POINTS["grids"], expected, strict=True):
        if EVIDENCE / SOURCES[name][0] != path:
            raise ValueError(f"{name} is not the figure script's grid {path}")
        reports[name] = expand(read_json(out / manifest["receipts"][name]["receipt"]))
        if set(reports[name]["systems"]) != set(models):
            raise ValueError(f"{name} systems differ from the figure script's {models}")
    comparison = OPERATING_POINTS["comparison"]
    reports[comparison] = expand(read_json(out / manifest["receipts"][comparison]["receipt"]))
    source = EVIDENCE / OPERATING_POINTS["source"]
    clean = sanitized(read_json(source))
    text_free(clean)
    found: set[str] = set()
    strings(clean, found)
    corpus = reference_texts(read_json(EVIDENCE / SOURCES[comparison][0]))
    if leaked := sorted(value for value in found if content_like(value) and value in corpus):
        raise ValueError(f"operating-point summary strings occur in reference text: {leaked[:5]}")
    _checks, failures = operating_point_checks(clean, recompute_operating_points(reports))
    if failures:
        raise ValueError(f"receipts do not recompute the operating points: {failures[:5]}")
    target = out / OPERATING_POINTS["summary"]
    write_json(target, clean)
    return {
        "purpose": "Silver-dev trust-region operating point of every main-comparison system, "
        "scored on Gold-7 and Silver-test",
        "summary": target.name,
        "summary_sha256": sha256_bytes(target.read_bytes()),
        "summary_source": {
            "evidence": OPERATING_POINTS["source"],
            "sha256": sha256_bytes(source.read_bytes()),
        },
        "rule": OPERATING_POINTS["rule"],
        "comparison": comparison,
        "grids": list(OPERATING_POINTS["grids"]),
    }


def verify(args) -> dict:
    """Recompute every reported number from the receipts and compare."""
    from pii_paper_o4_figures import fixed, paired, pool_ont3
    from pii_paper_pooled_eval import metrics

    manifest = json.loads((args.receipts / "receipts.json").read_text())
    checks, failures = [], []
    reports = {}
    for name, entry in manifest["receipts"].items():
        path = args.receipts / entry["receipt"]
        if sha256_bytes(path.read_bytes()) != entry["receipt_sha256"]:
            failures.append(f"{name}: receipt hash differs from manifest")
            continue
        report = reports[name] = expand(read_json(path))
        if "summary" not in entry:
            checks.append({"receipt": name, "check": "expanded", "systems": len(report["systems"])})
            continue
        summary = read_json(args.receipts / entry["summary"])
        pooled = pool_ont3(curve_copy(report))
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
    operating = manifest["operating_points"]
    summary_path = args.receipts / operating["summary"]
    operating_rows = []
    if sha256_bytes(summary_path.read_bytes()) != operating["summary_sha256"]:
        failures.append("operating points: summary hash differs from manifest")
    elif missing := [name for name in (operating["comparison"], *operating["grids"]) if name not in reports]:
        failures.append(f"operating points: unverified receipts {missing}")
    else:
        operating_rows, operating_failures = operating_point_checks(
            read_json(summary_path), recompute_operating_points(reports)
        )
        failures.extend(operating_failures)
        checks.extend({"receipt": "operating-points", **row} for row in operating_rows)
    o4 = next((row for row in operating_rows if row["system"] == "o4"), None)
    result = {
        "ok": not failures,
        "checks": len(checks),
        "failures": failures,
        "operating_points": (
            f"{len(operating_rows)} systems' Silver-dev trust-region points and their Gold-7/Silver-test "
            f"F1 match the paper; O4 bias {o4['selected']:g}: Gold-7 {o4['gold7_f1']:.1f}, "
            f"Silver-test {o4['silver_test_f1']:.1f}"
            if o4 and not failures
            else "not verified"
        ),
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
