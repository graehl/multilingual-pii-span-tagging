#!/usr/bin/env python3
"""Rebuild the paper's public human-gold evaluation rows from onboarded sources.

The paper's human-gold population is 1,283 publisher test rows from four
public corpora. `selected-ids.json` names each row by its onboarded source id
and pins its text SHA-256, so a reader who fetches and prepares the same
pinned sources can rebuild the exact rows without receiving any text from us.
The 30 Ont3 supplement rows in that list are private and are skipped.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acli

EVIDENCE = ROOT / "research/pii/frontier/evidence"
SELECTED = EVIDENCE / "human-gold-v1/selected-ids.json"
HUMAN_ROWS = 1283


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def onboarded_shard(entry: dict) -> tuple[str, int]:
    """Corpus-relative shard path and 1-based line recorded at selection time."""
    locator = entry["source_locator"]
    relative = locator["path"].split("pii-onboarded/", 1)[1]
    return relative, locator["line_1based"]


def rebuild(selected: Path, onboarded: Path, sources: set[str] | None = None) -> list[dict]:
    """Rebuild every public row, or with `sources` only those corpora's rows (a declared subset)."""
    entries = [entry for entry in json.loads(selected.read_text()) if entry["source"] != "ont3"]
    if len(entries) != HUMAN_ROWS:
        raise ValueError(f"expected {HUMAN_ROWS} public selections, found {len(entries)}")
    shards: dict[str, dict[str, tuple[int, dict]]] = {}
    rows = []
    for entry in entries:
        if sources is not None and entry["source"] not in sources:
            continue
        relative, line = onboarded_shard(entry)
        if relative not in shards:
            path = onboarded / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing onboarded shard {path}; run fetch and prepare for {entry['source']}"
                )
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                shards[relative] = {
                    row["id"]: (number, row) for number, row in enumerate(map(json.loads, stream), start=1)
                }
        found = shards[relative].get(entry["source_id"])
        if found is None:
            raise ValueError(f"{entry['id']}: source id absent from {relative}")
        number, source = found
        if sha256_text(source["text"]) != entry["text_sha256"]:
            raise ValueError(f"{entry['id']}: text differs from the pinned hash")
        if number != line:
            raise ValueError(f"{entry['id']}: source row moved from line {line} to {number}")
        rows.append(
            {
                **source,
                "id": entry["id"],
                "source_id": entry["source_id"],
                "evaluation_source": entry["source"],
                "source_locator": {"path": relative, "line_1based": number},
            }
        )
    expected = HUMAN_ROWS if sources is None else len(rows)
    if (
        not rows
        or len({row["id"] for row in rows}) != expected
        or len({row["text"] for row in rows}) != expected
    ):
        raise ValueError("rebuilt rows are not unique by id and text")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_rebuild(args) -> dict:
    rows = rebuild(args.selected, args.onboarded, set(args.source) if args.source else None)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "evaluation.jsonl", rows)
    # Prediction inputs carry no gold; context is empty because O4 serves without it.
    write_jsonl(
        args.out / "inputs.jsonl",
        [{"id": row["id"], "text": row["text"], "lang": row["lang"], "spans": []} for row in rows],
    )
    receipt = {
        "schema": "pii-public-human-gold-v1",
        "selected": {"path": args.selected.name, "sha256": sha256_file(args.selected)},
        "rows": len(rows),
        "sources": dict(Counter(row["evaluation_source"] for row in rows)),
        "languages": dict(Counter(row["lang"] for row in rows)),
        "outputs": {name: sha256_file(args.out / name) for name in ("evaluation.jsonl", "inputs.jsonl")},
        "scope": "Paper human-gold population; the 30 private Ont3 supplement rows are not rebuilt"
        if not args.source
        else f"SUBSET of the paper human-gold population: {sorted(args.source)} only; not paper-comparable",
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"ok": True, "out": str(args.out.resolve()), **receipt}


def score(args) -> dict:
    """Score one prediction sweep with the paper's human-gold region policy."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from pii_ontology_v2 import load_ontology
    from pii_paper_o4_eval import score_points
    from pii_paper_pooled_eval import load_module, metrics, read_json, write

    human = load_module("public_human_score", EVIDENCE / "human-gold-v1/score.py")
    rows = [json.loads(line) for line in args.evaluation.open(encoding="utf-8")]
    for row in rows:
        row["scoring_language"] = row.get("scoring_language", row["lang"])
    sweep = read_json(args.sweep)
    identity = {row["id"]: row["text"] for row in rows}
    sidecar = {}
    if args.title_sidecar:
        sidecar = {entry["id"]: entry["extents"] for entry in map(json.loads, args.title_sidecar.open())}
    points = score_points(
        sweep["points"],
        rows,
        identity,
        human=human,
        ontology=load_ontology(),
        sidecar=sidecar,
        projection="ont3",
        source=args.sweep.name,
    )
    languages = dict(Counter(row["scoring_language"] for row in rows))
    result = {
        "schema": "pii-paper-o4-evaluation/v1",
        "status": "public human-gold rescore",
        "references": {"human": {"path": args.evaluation.name, "sha256": sha256_file(args.evaluation)}},
        "title_sidecar": {"path": args.title_sidecar.name, "sha256": sha256_file(args.title_sidecar)}
        if args.title_sidecar
        else None,
        "title_policy": human.TITLE_POLICY,
        "coverage_policy": human.COVERAGE_POLICY,
        "reference_policy": human.OPTIONAL_REFERENCE_POLICY,
        "selection": "Curve maxima are descriptive; the fixed zero-bias point is the predeclared operating point",
        "populations": {"human": {"N": len(rows), "languages": languages, "split": "publisher test subset"}},
        "systems": {
            args.model: {
                "human": {
                    "path": args.sweep.name,
                    "sha256": sha256_file(args.sweep),
                    "context": sweep.get("context_side"),
                    "checkpoint": Path(str(sweep.get("model_path"))).name,
                    "expressible": None,
                    "points": points,
                }
            }
        },
    }
    write(args.out, result)
    best = max(points, key=lambda point: point["regions"]["80"]["metrics"]["F1"])
    fixed = next((point for point in points if point["threshold"] == 0), None)
    versus_o4 = None
    if args.compare_receipt and len(rows) == HUMAN_ROWS and args.title_sidecar and fixed:
        # Same documents, groups and scoring as O4's shipped counts: pair them.
        from pii_paper_o4_figures import paired
        from pii_software_receipts import expand
        from pii_software_receipts import read_json as read_receipt

        if args.model == "o4":
            raise ValueError("--model o4 would collide with the O4 receipt in the paired comparison")
        o4 = expand(read_receipt(args.compare_receipt))["systems"]["o4"]["human"]
        report = {"systems": {args.model: {"human": {"points": points}}, "o4": {"human": o4}}}
        versus_o4 = paired(report, args.model, "o4", ["human"], False)
    return {
        "ok": True,
        "out": str(args.out.resolve()),
        "rows": len(rows),
        "title_sidecar": bool(args.title_sidecar),
        "paired_versus_o4": versus_o4,
        "maximum": {"threshold": best["threshold"], **best["regions"]["80"]["metrics"]},
        "fixed_zero_bias": metrics(
            [sum(row["counts"][i] for row in fixed["regions"]["80"]["per_input"]) for i in range(3)]
        )
        if fixed
        else None,
    }


def build_parser():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild_command = commands.add_parser(
        "eval-rebuild", help="Write evaluation.jsonl and inputs.jsonl for the 1,283 public rows."
    )
    rebuild_command.add_argument(
        "--onboarded",
        type=Path,
        required=True,
        help="Prepare output root holding <corpus>/test/<lang>.jsonl.gz",
    )
    rebuild_command.add_argument("--selected", type=Path, default=SELECTED)
    rebuild_command.add_argument(
        "--source",
        action="append",
        help="Rebuild only this corpus's rows (repeatable); the result is a declared subset",
    )
    rebuild_command.add_argument("--out", type=Path, required=True)
    rebuild_command.set_defaults(action=eval_rebuild)
    score_command = commands.add_parser(
        "score", help="Score a prediction sweep on rebuilt rows in the paper's per-input schema."
    )
    score_command.add_argument("--evaluation", type=Path, required=True, help="Rebuilt evaluation.jsonl")
    score_command.add_argument(
        "--sweep", type=Path, required=True, help="Sweep JSON: points {bias: [{id, preds}]}, model_path"
    )
    score_command.add_argument("--model", required=True, help="System name recorded in the score file")
    score_command.add_argument(
        "--title-sidecar", type=Path, help="Neutral title extents JSONL used by the paper's title policy"
    )
    score_command.add_argument("--out", type=Path, required=True, help="Score JSON (.gz compresses)")
    score_command.add_argument(
        "--compare-receipt",
        type=Path,
        help="O4 score receipt (records/receipts/o4-comparison.json.gz): paired bootstrap versus O4 "
        "at zero bias, exact regions, when all 1,283 rows and the title sidecar are scored",
    )
    score_command.set_defaults(action=score)
    acli.add_standard_args(parser)
    for command in (rebuild_command, score_command):
        acli.add_standard_args(command)
    return parser


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        result = args.action(args)
    except (OSError, ValueError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
