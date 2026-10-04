#!/usr/bin/env python3
# acli: 1 complete
"""Score prediction sweeps on the paper's Ont3 evaluation, exactly as the paper does.

The paper's 31-type development pool has two collections, shipped verbatim in
data/ont3-evaluation/:

- selection (659 rows): selection-inputs.jsonl is the prediction input;
  selection-references.jsonl holds the references (Sol-adjudicated two-pass
  annotation, manually revised r4), scored under the language corrections of
  selection-language-review.json. O4 was selected on these rows.
- heldout (542 rows): heldout.jsonl is both input and reference
  (single-teacher Luna, needle-selected, training-disjoint), never used for
  selection.

Scoring is the paper's score_points (exact typed spans and redaction regions
at 80% and exact overlap over the O-bias sweep, optional references neutral)
and its paired source-group bootstrap against O4's text-free receipts.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import acli  # noqa: E402

DATA = ROOT / "data/ont3-evaluation"
RECEIPTS = ROOT / "research/pii/frontier/software/records/receipts"
POPULATIONS = {
    "ont3": ("selection-inputs.jsonl", "selection-references.jsonl"),
    "heldout": ("heldout.jsonl", "heldout.jsonl"),
}
SPLITS = {
    "ont3": "reused development, manually revised r4",
    "heldout": "training-disjoint development-bucket extension; single-teacher Luna; needle-selected",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inputs(population: str) -> Path:
    return DATA / POPULATIONS[population][0]


def score(sweeps: dict[str, Path], model: str) -> dict:
    """Per-input counts for each population's sweep, in the paper's score format."""
    from pii_ontology_v2 import load_ontology
    from pii_paper_o4_eval import score_points
    from pii_paper_pooled_eval import EVIDENCE, load_module

    human = load_module("o4_human_score", EVIDENCE / "human-gold-v1/score.py")
    ontology = load_ontology()
    changes = json.loads((DATA / "selection-language-review.json").read_text())["changes_by_id_prefix"]
    report = {"populations": {}, "systems": {model: {}}}
    for population, path in sweeps.items():
        input_path, reference_path = (DATA / name for name in POPULATIONS[population])
        sweep = json.loads(path.read_text())
        if sweep["input_sha256"] != sha(input_path):
            raise ValueError(f"{path} was not predicted on {input_path.name}")
        records = rows(reference_path)
        for row in records:
            row["evaluation_source"] = "ont3"
            row["scoring_language"] = (
                changes.get(row["id"].split(":")[0], row["lang"]) if population == "ont3" else row["lang"]
            )
        identity = {row["id"]: row["text"] for row in rows(input_path)}
        if identity != {row["id"]: row["text"] for row in records}:
            raise ValueError(f"{population} inputs and references differ in membership or text")
        report["populations"][population] = {
            "N": len(records),
            "languages": dict(Counter(row["scoring_language"] for row in records)),
            "split": SPLITS[population],
        }
        report["systems"][model][population] = {
            "path": str(path),
            "sha256": sha(path),
            "points": score_points(
                sweep["points"],
                records,
                identity,
                human=human,
                ontology=ontology,
                sidecar={},
                projection="ont3",
                typed_view=True,
                preview=[],
                source=path,
            ),
        }
    return report


def summarize(report: dict, model: str, control: str) -> dict:
    """Zero-bias scores per collection and pooled, paired against an O4 receipt system."""
    from pii_paper_o4_figures import fixed, paired, pool_ont3
    from pii_paper_pooled_eval import metrics
    from pii_software_receipts import expand, read_json

    receipt = "o4-comparison.json.gz" if control == "o4" else "o4-boundary.json.gz"
    reference = expand(read_json(RECEIPTS / receipt))["systems"][control]
    joined = copy.deepcopy(report)
    joined["systems"][control] = {population: reference[population] for population in POPULATIONS}
    views = {population: (joined, population) for population in report["systems"][model]}
    if set(views) == set(POPULATIONS):
        views["pooled"] = (pool_ont3(copy.deepcopy(joined)), "ont3")
    summary = {}
    for view, (data, population) in views.items():
        entry = {}
        for name in (model, control):
            point = fixed(data, name, population)
            entry[name] = {
                "regions_80": 100 * metrics(sum_counts(point["regions"]["80"]["per_input"]))["F1"],
                "regions_exact": 100 * metrics(sum_counts(point["regions"]["100"]["per_input"]))["F1"],
                "typed_exact": 100 * metrics(sum_counts(point["typed"]["per_input"]))["F1"],
            }
        for typed in (False, True):
            result = paired(data, model, control, [population], typed)
            entry[f"paired {result['metric']}"] = {
                "delta": 100 * result["delta"],
                "ci95": [100 * bound for bound in result["ci95"]],
                "source_groups": result["source_groups"],
            }
        summary[view] = entry
    return summary


def sum_counts(items: list[dict]) -> list[int]:
    return [sum(item["counts"][i] for item in items) for i in range(3)]


def build_parser():
    parser = acli.argument_parser(description=__doc__, exit_codes={0: "complete", 2: "invalid input"})
    parser.add_argument("--selection-sweep", type=Path, help="Sweep predicted on selection-inputs.jsonl")
    parser.add_argument("--heldout-sweep", type=Path, help="Sweep predicted on heldout.jsonl")
    parser.add_argument("--model", required=True, help="System name in the score file")
    parser.add_argument(
        "--control",
        choices=("o4", "o4-unrefined"),
        default="o4",
        help="O4 as served (default) or without boundary refinement, to pair raw output",
    )
    parser.add_argument("--out", type=Path, required=True, help="New .json.gz score file")
    acli.add_standard_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    sweeps = {
        population: path
        for population, path in (("ont3", args.selection_sweep), ("heldout", args.heldout_sweep))
        if path is not None
    }
    if not sweeps:
        raise ValueError("give --selection-sweep, --heldout-sweep or both")
    if args.out.exists():
        raise FileExistsError(args.out)
    report = score(sweeps, args.model)
    summary = summarize(report, args.model, args.control)
    with gzip.open(args.out, "wt", encoding="utf-8") as sink:
        json.dump({**report, "summary": summary}, sink)
    acli.emit(
        {"ok": True, "out": str(args.out), "control": args.control, "summary": summary},
        acli.resolve_format(args),
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        acli.die(str(error), 2)
