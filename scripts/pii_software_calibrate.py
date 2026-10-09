#!/usr/bin/env python3
# acli: 1 complete
"""Fix a model's operating point on a development set by the paper's rule, then report it.

Reads the score files `pii-reproduce.py evaluate` wrote (`scores-human.json.gz`,
`scores-ont3.json.gz`): per-row counts at every bias or confidence setting.
On the development population it applies the paper's trust-region rule
(`select_threshold` in `scripts/pii_paper_o4_figures.py`, the code that fixed
every system's operating point in the paper): find the setting with the best
pooled 80%-overlap region F1, resample source documents 10,000 times (seed
20261009), keep the contiguous range of settings whose F1 difference from the
best has a 95% interval including zero, and take its median setting. That
setting is then scored unchanged on every other population, and paired
against O4 at its own paper operating point.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import acli  # noqa: E402

RECEIPTS = ROOT / "research/pii/frontier/software/records/receipts"
# The paper's population names, and the score-file populations they denote.
POPULATIONS = {"silver-dev": "ont3", "gold-7": "human", "silver-test": "heldout"}
NAMES = {
    "silver-dev": "Silver-dev: the 659 Ont3 selection rows, which also selected O4",
    "gold-7": "Gold-7: the 1,283 public human-gold rows in seven languages",
    "silver-test": "Silver-test: the 542 Ont3 held-out rows, never used for any selection",
    "silver-test-seven": "Silver-test rows in Gold-7's seven languages",
}
# Their keys in the paper's operating-points-trust-region.json.
PAPER_KEYS = {
    "silver-dev": "silver_dev",
    "gold-7": "gold7",
    "silver-test": "silver_test",
    "silver-test-seven": "silver_test_seven",
}


def read_scores(evaluation: Path) -> tuple[dict, str, str]:
    """Join an evaluate directory's score files into one report for its model."""
    summary = json.loads((evaluation / "summary.json").read_text())
    report = {"populations": {}, "systems": {}}
    for name in ("scores-human.json.gz", "scores-ont3.json.gz"):
        path = evaluation / name
        if not path.is_file():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            scores = json.load(stream)
        report["populations"].update(scores["populations"])
        for model, populations in scores["systems"].items():
            report["systems"].setdefault(model, {}).update(populations)
    if len(report["systems"]) != 1:
        raise ValueError(f"{evaluation}: expected one scored model, found {sorted(report['systems'])}")
    (model,) = report["systems"]
    return report, model, summary["control"]


def scored_view(points: list[dict], languages: set[str] | None) -> list[dict]:
    """Points restricted to rows in `languages` (all rows when None)."""
    if languages is None:
        return points
    restricted = []
    for point in points:
        view = {"threshold": point["threshold"], "regions": {}}
        for coverage, region in point["regions"].items():
            view["regions"][coverage] = {
                "per_input": [row for row in region["per_input"] if row["lang"] in languages]
            }
        if "typed" in point:
            view["typed"] = {
                "per_input": [row for row in point["typed"]["per_input"] if row["lang"] in languages]
            }
        restricted.append(view)
    return restricted


def point_metrics(point: dict) -> dict:
    from pii_paper_o4_figures import items, total

    values = {
        "regions_80": total(items(point, 80)),
        "regions_exact": total(items(point, 100)),
    }
    if "typed" in point:
        values["typed_exact"] = total(items(point, typed=True))
    return {
        name: {
            key: round(100 * value, 2) if key in ("P", "R", "F1") else value for key, value in metric.items()
        }
        for name, metric in values.items()
    }


def calibrate(evaluation: Path, development: str) -> dict:
    from pii_paper_o4_figures import LANGS, default_threshold, paired, resample_weights, select_threshold
    from pii_software_receipts import expand, read_json

    report, model, control = read_scores(evaluation)
    curves = report["systems"][model]
    population = POPULATIONS[development]
    if population not in curves:
        raise ValueError(
            f"{evaluation} has no {development} scores; run evaluate with --population all or one that "
            "includes it"
        )
    points = sorted(curves[population]["points"], key=lambda point: point["threshold"])
    groups = sorted({row["group"] for row in points[0]["regions"]["80"]["per_input"]})
    index = {group: i for i, group in enumerate(groups)}
    default = default_threshold(model)
    chosen, argmax, region, _curve = select_threshold(points, default, resample_weights(len(groups)), index)

    views = {name: (POPULATIONS[name], None) for name in POPULATIONS if POPULATIONS[name] in curves}
    if "heldout" in curves:
        views["silver-test-seven"] = ("heldout", set(LANGS))
    paper = json.loads((RECEIPTS / "operating-points-trust-region.json").read_text())["systems"]
    control_threshold = paper["o4"]["selected_threshold"]
    receipt = "o4-comparison.json.gz" if control == "o4" else "o4-boundary.json.gz"
    reference = expand(read_json(RECEIPTS / receipt))["systems"][control]
    evaluations = {}
    for name, (source, languages) in views.items():
        by_threshold = {p["threshold"]: p for p in scored_view(curves[source]["points"], languages)}
        entry = {
            "role": "development" if name == development else "held out",
            "rows": len(by_threshold[chosen]["regions"]["80"]["per_input"]),
            "selected": point_metrics(by_threshold[chosen]),
            "default": point_metrics(by_threshold[default]) if default in by_threshold else None,
            "argmax_on_development": point_metrics(by_threshold[argmax]),
            "paper": paper_reference(paper, name, model),
        }
        if source in reference and languages is None and not same_rows(curves[source], reference[source]):
            entry["paired_versus_" + control] = None
            entry["unpaired"] = f"rows differ from {control}'s receipt (a subset, such as --demo human gold)"
        elif source in reference and languages is None:
            joined = {
                "systems": {model: {source: curves[source]}, control: {source: reference[source]}},
            }
            comparison = paired(
                joined,
                model,
                control,
                [source],
                coverage=80,
                thresholds={model: chosen, control: control_threshold},
            )
            entry["paired_versus_" + control] = {
                "metric": comparison["metric"],
                "delta": round(100 * comparison["delta"], 2),
                "ci95": [round(100 * bound, 2) for bound in comparison["ci95"]],
                "source_groups": comparison["source_groups"],
                "thresholds": comparison["thresholds"],
            }
        evaluations[name] = entry
    return {
        "schema": "pii-software-calibration/v1",
        "model": model,
        "evaluation": str(evaluation.resolve()),
        "development": development,
        "rule": "trust-region median of the development argmax of pooled 80%-overlap region F1; "
        "10,000 source-document resamples, seed 20261009 (the paper's operating-point rule)",
        "default_threshold": default,
        "selected_threshold": chosen,
        "trust_region": region,
        "names": {name: NAMES[name] for name in evaluations},
        "evaluations": evaluations,
        "note": "F1, P and R in percent from counts pooled over rows. Held-out populations never "
        "influence the setting. The paper's numbers (`paper`) are its systems at their own Silver-dev "
        "points, finer grids included.",
    }


def same_rows(curve: dict, reference: dict) -> bool:
    """Whether a curve scores exactly the rows of a receipt curve, so the two can be paired."""

    def ids(entry: dict) -> set[str]:
        return {row["id"] for row in entry["points"][0]["regions"]["80"]["per_input"]}

    return ids(curve) == ids(reference)


def paper_reference(paper: dict, name: str, model: str) -> dict:
    """The paper's O4 (and GL4, for a GLiNER2 model) at their Silver-dev points, 80% regions."""
    key = PAPER_KEYS[name]
    systems = {"o4": "O4"} | ({"gliner2-o4": "GL4"} if model.startswith("gliner2") else {})
    return {
        label: {
            "threshold": paper[system]["selected_threshold"],
            "regions_80_f1": round(100 * paper[system]["evaluations"][key]["80"]["selected"]["F1"], 2),
        }
        for system, label in systems.items()
    }


def build_parser():
    parser = acli.argument_parser(description=__doc__, exit_codes={0: "complete", 2: "invalid input"})
    parser.add_argument("--evaluation", type=Path, required=True, help="An evaluate output directory")
    parser.add_argument(
        "--development",
        choices=sorted(POPULATIONS),
        default="silver-dev",
        help="Population that fixes the operating point (default silver-dev, as in the paper)",
    )
    parser.add_argument("--out", type=Path, required=True, help="New calibration JSON")
    acli.add_standard_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = calibrate(args.evaluation, args.development)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    acli.emit(
        {
            "ok": True,
            "out": str(args.out),
            "selected_threshold": result["selected_threshold"],
            "trust_region": result["trust_region"]["bounds"],
            "regions_80_f1": {
                name: entry["selected"]["regions_80"]["F1"] for name, entry in result["evaluations"].items()
            },
        },
        acli.resolve_format(args),
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        acli.die(str(error), 2)
