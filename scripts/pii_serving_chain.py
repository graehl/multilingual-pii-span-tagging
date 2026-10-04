#!/usr/bin/env python3
# acli: 1 complete
"""Apply the O2/O3 serving stages that follow boundary adjustment to a saved sweep.

the production toolkit serving refines decoded model spans, runs the name postprocessor on them,
then adds regex spans that do not overlap a model span (`skip`). Regex spans
receive neither earlier stage. The input sweep must already carry the
`pii_character_boundary_refiner.py apply-sweep` receipt, so this step cannot
run ahead of boundary adjustment. The explicit --without-boundary ablation
instead accepts an unrefined sweep and runs the same remaining stages.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "agents"))
import acli

NAME_CONFIG = ROOT / "research/pii/frontier/models/name-components/name-postprocessor-name-kind.json"
NAME_KIND_BUNDLE = ROOT / "research/pii/frontier/models/name-components/name-kind-v4/name-kind.config.json"
REGEX_RULES = ROOT / "scripts/pii_regex_tags_v1.json"
REGEX_POLICY = ROOT / "scripts/pii_regex_policy_ont3_v1.json"
ORDER = ["boundary_adjustment", "name_kind", "regex_skip"]


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def primary_key(spans: list[dict]) -> list[tuple]:
    return sorted((s["start"], s["end"], s["label"]) for s in spans)


def run(args) -> dict:
    from scripts.pii_name_annotation_qc import NameAnnotationQc, load_config
    from scripts.pii_name_role_publish import NameKindOnnxDeployment
    from scripts.pii_regex_tag import load_policy, load_rules, tag_rows

    if args.output.exists():
        raise FileExistsError(args.output)
    sweep = json.loads(args.sweep.read_text())
    without_boundary = args.without_boundary
    if without_boundary and "boundary_refinement" in sweep:
        raise ValueError("Boundary ablation requires an unrefined sweep")
    if not without_boundary and "boundary_refinement" not in sweep:
        raise ValueError("Sweep lacks a boundary-adjustment receipt; refine it first")
    if "serving_chain" in sweep:
        raise ValueError("Sweep already carries a serving-chain receipt")
    if identity(args.inputs)["sha256"] != sweep["input_sha256"]:
        raise ValueError("Inputs differ from the sweep's recorded input hash")
    rows = {row["id"]: row for row in map(json.loads, args.inputs.open())}
    rules, families, rules_receipt = load_rules(REGEX_RULES, args.regex_ontology)
    policy, arbitration, policy_receipt = load_policy(REGEX_POLICY)
    if policy.mode != "skip":
        raise ValueError(f"Regex arbitration must be skip; policy says {policy.mode!r}")
    bundle = None if args.no_name_kind else args.name_kind_bundle
    engine = None
    if bundle is not None:
        engine = NameAnnotationQc(
            load_config(NAME_CONFIG),
            mode="insert-name-kinds",
            override_name_kinds=False,
            name_kind_deployment=NameKindOnnxDeployment.load(bundle),
        )
    points = {}
    stages = {}
    for bias, predictions in sweep["points"].items():
        served = []
        name_changed = 0
        for prediction in predictions:
            if engine is None:
                served.append(prediction)
                continue
            row = rows[prediction["id"]]
            analysis = engine.analyze(
                row_id=row["id"], text=row["text"], language=row["lang"], annotations=prediction["preds"]
            )
            preds = engine.inference_fields(analysis)["preds"]
            name_changed += primary_key(preds) != primary_key(prediction["preds"])
            served.append({**prediction, "preds": preds})
        regex = tag_rows(
            served,
            rules,
            span_keys=["preds"],
            policy=policy,
            families=families,
            scores=arbitration["scores"],
            gates=arbitration["gates"],
            texts=[rows[p["id"]]["text"] for p in served],
            languages=[rows[p["id"]]["lang"] for p in served],
        )
        points[bias] = served
        stages[bias] = {"name_kind_changed_primary_rows": name_changed, "regex": regex}
    sweep["points"] = points
    order = [stage for stage in ORDER if stage != "name_kind" or engine is not None]
    sweep["serving_chain"] = {
        "order": order[1:] if without_boundary else order,
        "boundary_ablation": without_boundary,
        "source_sweep": identity(args.sweep),
        "name_kind": (
            {"config": identity(NAME_CONFIG), "bundle": identity(bundle)} if engine is not None else None
        ),
        "regex": {
            "rules": {**identity(REGEX_RULES), **rules_receipt},
            "policy": {**identity(REGEX_POLICY), **policy_receipt},
            "mode": policy.mode,
        },
        "stages": stages,
    }
    args.output.write_text(json.dumps(sweep) + "\n")
    zero = stages.get("0.0") or next(iter(stages.values()))
    return {
        "output": str(args.output),
        "points": len(points),
        "rows": len(predictions),
        "bias_zero": zero,
    }


def main() -> None:
    parser = acli.argument_parser(description=__doc__, exit_codes={0: "complete", 2: "invalid input"})
    parser.add_argument("--sweep", type=Path, required=True, help="boundary-refined sweep JSON")
    parser.add_argument("--inputs", type=Path, required=True, help="JSONL rows with id, text and lang")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regex-ontology", choices=("ont3", "ont2"), default="ont3")
    parser.add_argument(
        "--without-boundary", action="store_true", help="Ablate boundary adjustment on a raw sweep"
    )
    name_kind = parser.add_mutually_exclusive_group()
    name_kind.add_argument(
        "--name-kind-bundle",
        type=Path,
        default=NAME_KIND_BUNDLE,
        help="name-kind.config.json of the name-kind model (default: the paper's bundle)",
    )
    name_kind.add_argument(
        "--no-name-kind", action="store_true", help="Skip name-kind postprocessing (no bundle built)"
    )
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    acli.emit(run(args), acli.resolve_format(args))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        acli.die(str(error), 2)
