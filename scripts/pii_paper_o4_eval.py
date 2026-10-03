#!/usr/bin/env python3
"""Rescore paper systems on revised Ont3 and human-gold references."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.home() / "agents"))
from pii_paper_pooled_eval import (
    EVIDENCE,
    FILTERS,
    HUMAN,
    PRESIDIO_SCHEMA,
    load_module,
    metrics,
    read_json,
    rows,
    sha,
    write,
)

import acli

DEST = EVIDENCE / "paper-o4-v1"
# The authors' local run artifacts; only the paper's own rescoring reads them.
ARTIFACTS = Path("/local") / Path.home().name / "artifacts/pii-redaction-frontier"
SERVED = ARTIFACTS / "tag-status-replay-v2/served"
O4_ARM = "gs30-titles-v6-g50-seed2-4000-none"
O3_ARM = "o3-2000-both"
GOLD = EVIDENCE / "ont3-five-model-v1/gold-manual-r4.jsonl"
HELDOUT = Path.home() / "artifacts/pii-redaction-frontier/needle-heldout-v1/inputs.jsonl"
SIDECAR = Path.home() / "artifacts/pii-redaction-frontier/title-extents-v1/neutral-extents-fulltest-v2.jsonl"


def paper_sweeps():
    """Frozen system identities; O3 retains its original both-neighbor policy."""
    yield "o4", "human", ARTIFACTS / "paper-o4-v1/o4-human-served.json", "ont3"
    yield "o3", "human", SERVED / f"four-corpus-{O3_ARM}-human-sweep-v1.json", "ont3"
    for population in ("ont3", "heldout"):
        yield "o4", population, SERVED / f"four-corpus-{O4_ARM}-{population}-sweep-v1.json", "ont3"
    yield "o3", "ont3", SERVED / f"four-corpus-{O3_ARM}-ont3-sweep-v1.json", "ont3"
    # The extension reserves only the preceding sentence; both-neighbor
    # inference therefore equals previous-only inference on this population.
    yield "o3", "heldout", SERVED / "four-corpus-o3-2000-previous-heldout-sweep-v1.json", "ont3"
    yield "gliner2", "heldout", ARTIFACTS / "paper-o4-v1/gliner2-heldout.json", "gliner2"
    for model, filename, projection in (
        ("ont1", "ont1-heldout.json", "ont1"),
        ("ont2", "ont2-heldout-served.json", "ont3"),
        ("gliner2-tuned", "gl3-heldout-raw.json", "ont3"),
        ("gliner2-o4", "gliner-o4-heldout.json", "ont3"),
        ("presidio", "presidio-extension/presidio.json.gz", PRESIDIO_SCHEMA),
    ):
        yield model, "heldout", ARTIFACTS / "paper-o4-v1" / filename, projection
    for key, (_, _, _, schema, _) in FILTERS.items():
        filename = (
            f"filter-{key}-heldout-refresh.json.gz"
            if key == "opf-openmed-multi2"
            else f"filter-{key}-heldout.json.gz"
        )
        yield key, "heldout", ARTIFACTS / "paper-o4-v1" / filename, schema
    for population in ("human", "ont3"):
        yield (
            "gliner2-o4",
            population,
            ARTIFACTS / f"paper-o4-v1/gliner-trajectory-{population}/checkpoint-600.json",
            "ont3",
        )
        yield (
            "ont2",
            population,
            ARTIFACTS / "paper-o4-v1" / f"ont2-{population}-served.json",
            "ont3",
        )
        filename = "ont1-dense.json" if population == "human" else "ont1.json"
        folder = "human-gold-v1" if population == "human" else "ont3-five-model-v1"
        yield "ont1", population, EVIDENCE / folder / filename, "ont1"
        yield "presidio", population, EVIDENCE / "pooled-gold-languages-v1/presidio.json.gz", PRESIDIO_SCHEMA
        for key, (_, _, _, schema, _) in FILTERS.items():
            path = (
                ARTIFACTS / f"paper-o4-v1/filter-{key}-pooled-refresh.json.gz"
                if key == "opf-openmed-multi2"
                else EVIDENCE / "pooled-gold-languages-v1" / f"filter-{key}.json.gz"
            )
            yield key, population, path, schema
        folder = "human-gold-v1" if population == "human" else "ont3-five-model-v1"
        filename = "gliner2-extended.json" if population == "human" else "gliner2-organizations.json"
        yield "gliner2", population, EVIDENCE / folder / filename, "gliner2"
        filename = f"checkpoint-100-{'human-gold' if population == 'human' else 'ont3'}-sweep.json.gz"
        yield "gliner2-tuned", population, EVIDENCE / "gliner2-ont3-transfer-v1/lowrate" / filename, "ont3"


def gliner_trajectory_sweeps(full_human=False):
    """Development-only checkpoint comparisons; never select on other sets."""
    for name in (
        "initial",
        "checkpoint-100",
        "checkpoint-300",
        "checkpoint-400",
        "checkpoint-600",
        "checkpoint-1000",
        "checkpoint-2000",
    ):
        for population in ("ont3", "human"):
            directory = "fullhuman" if full_human and population == "human" else population
            yield (
                name,
                population,
                ARTIFACTS / f"paper-o4-v1/gliner-trajectory-{directory}" / f"{name}.json",
                "ont3",
            )
    yield "checkpoint-200", "ont3", ARTIFACTS / "paper-o4-v1/gliner-200-ont3.json", "ont3"
    yield (
        "checkpoint-200",
        "human",
        ARTIFACTS
        / f"paper-o4-v1/gliner-trajectory-{'fullhuman' if full_human else 'human'}/checkpoint-200.json",
        "ont3",
    )


def score_paper_o4(
    limit=0,
    gliner_trajectory=False,
    full_human=False,
    *,
    sweeps=None,
    destination=DEST,
    selection_note=None,
):
    """Score aligned saved predictions with the shared title and coverage policy."""
    human = load_module("o4_human_score", EVIDENCE / "human-gold-v1/score.py")
    from pii_ontology_v2 import load_ontology

    ontology = load_ontology()
    human_root = HUMAN.parent / "comparison-fulltest-prevctx-v1" if full_human else HUMAN
    paths = {"human": human_root / "evaluation.jsonl", "ont3": GOLD, "heldout": HELDOUT}
    populations = {name: rows(path) for name, path in paths.items()}
    changes = read_json(EVIDENCE / "ont3-five-model-v1/language-review.json")["changes_by_id_prefix"]
    for population, records in populations.items():
        for row in records:
            if population != "human":
                row["evaluation_source"] = "ont3"
            row["scoring_language"] = (
                changes.get(row["id"].split(":")[0], row["lang"])
                if population == "ont3"
                else row.get("scoring_language", row["lang"])
            )
    sidecar = {row["id"]: row["extents"] for row in rows(SIDECAR)}
    result = {
        "schema": "pii-paper-o4-evaluation/v1",
        "status": "smoke" if limit else "development trajectory" if gliner_trajectory else "complete",
        "references": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "title_sidecar": {"path": str(SIDECAR), "sha256": sha(SIDECAR)},
        "title_policy": human.TITLE_POLICY,
        "coverage_policy": human.COVERAGE_POLICY,
        "reference_policy": human.OPTIONAL_REFERENCE_POLICY,
        "selection": "Diagnostic trajectory on paper populations, not the full campaign selection population"
        if gliner_trajectory
        else "O4 selected before this paper rescore; curve maxima are descriptive only",
        "populations": {},
        "systems": {},
        "preview": [],
    }
    for population, records in populations.items():
        selected = [r for r in records if population != "human" or r["evaluation_source"] != "ont3"]
        result["populations"][population] = {
            "N": len(selected),
            "languages": dict(Counter(r["scoring_language"] for r in selected)),
            "split": {
                "human": "publisher full test, repeatedly evaluated during development"
                if full_human
                else "publisher test subset, repeatedly evaluated during development",
                "ont3": "reused development, manually revised r4",
                "heldout": "training-disjoint development-bucket extension; single-teacher Luna; needle-selected",
            }[population],
        }
    if sweeps is None:
        sweeps = gliner_trajectory_sweeps(full_human) if gliner_trajectory else paper_sweeps()
    for model, population, path, projection in sweeps:
        sweep = read_json(path)
        input_path = (
            human_root / "inputs.jsonl"
            if population == "human"
            else HELDOUT
            if population == "heldout"
            else EVIDENCE / "ont3-five-model-v1/inputs.jsonl"
        )
        mapped_source = model == "presidio" or model in FILTERS
        pooled = mapped_source and population != "heldout"
        if pooled:
            input_path = EVIDENCE / "pooled-gold-languages-v1/evaluation.jsonl"
        if sweep["input_sha256"] != sha(input_path):
            raise ValueError(f"Prediction input hash mismatch: {path}")
        inputs = rows(input_path)
        records = populations[population]
        if model == "presidio" and not pooled:
            sweep["points"] = {"0": sweep["rows"]}
        if pooled:
            inputs = [{**r, "id": r["original_id"]} for r in inputs if r["population"] == population]
            records = [r for r in records if population != "human" or r["evaluation_source"] != "ont3"]
            raw_points = {"0": sweep["rows"]} if model == "presidio" else sweep["points"]
            sweep["points"] = {
                threshold: [
                    {**p, "id": p["id"].removeprefix(population + "::")}
                    for p in predictions
                    if p["id"].startswith(population + "::")
                ]
                for threshold, predictions in raw_points.items()
            }
        identity = {r["id"]: r["text"] for r in inputs}
        if identity != {r["id"]: r["text"] for r in records}:
            raise ValueError(f"Prediction input membership/text mismatch: {path}")
        expressible = None
        if projection == "gliner2":
            expressible = {
                tag
                for label in sweep["requested_labels"]
                for tag in (
                    (label,) if label == "organization" else ontology.source_acceptable("fastino_42", label)
                )
            } - {"O"}
        selected = [r for r in records if population != "human" or r["evaluation_source"] != "ont3"]
        if limit:
            selected = selected[:limit]
        points = score_points(
            sweep["points"],
            selected,
            identity,
            human=human,
            ontology=ontology,
            sidecar=sidecar,
            projection=projection,
            expressible=expressible,
            mapped_source=mapped_source,
            typed_view=population != "human" and projection == "ont3",
            full_human=full_human,
            preview=result["preview"],
            preview_labels={"model": model, "population": population},
            source=path,
        )
        result["systems"].setdefault(model, {})[population] = {
            "path": str(path),
            "sha256": sha(path),
            "context": sweep.get("context_side"),
            "checkpoint": sweep.get("model_path") or sweep.get("hf_id") or sweep.get("source"),
            "expressible": sorted(expressible) if expressible else None,
            "points": points,
        }
        best = max(points, key=lambda p: p["regions"]["80"]["metrics"]["F1"])
        print(
            f"[score] {model} {population} N={len(selected)} max-overlap-F1={best['regions']['80']['metrics']['F1']:.6f}",
            flush=True,
        )
    if full_human:
        import yaml

        language_path = Path(__file__).with_name("pii_language_round.yaml")
        importance = yaml.safe_load(language_path.read_text())["language_importance"]
        ranking = []
        for model, populations in result["systems"].items():
            combined = [0.0, 0.0, 0.0]
            for population, family, share in (
                ("human", "regions", 0.70),
                ("ont3", "regions", 0.12),
                ("ont3", "typed", 0.18),
            ):
                point = populations[population]["points"][0]
                items = (
                    point[family]["100"]["per_input"] if family == "regions" else point[family]["per_input"]
                )
                weights = [
                    float(importance["weights"].get(row["lang"], importance["unlisted_language_weight"]))
                    for row in items
                ]
                total = sum(weights)
                for row, weight in zip(items, weights, strict=True):
                    for i, value in enumerate(row["counts"]):
                        combined[i] += share * weight / total * value
            ranking.append({"checkpoint": model, "metrics": metrics(combined)})
        ranking.sort(key=lambda row: row["metrics"]["F1"], reverse=True)
        result["selection"] = {
            "criterion": "language-weighted pooled counts: 70% human exact regions, 12% Ont3 exact regions, 18% Ont3 exact typed spans; fixed confidence 0.5",
            "language_weights_sha256": sha(language_path),
            "ranking": ranking,
            "selected": ranking[0]["checkpoint"] if not limit else None,
        }
    if selection_note is not None and not full_human:
        result["selection"] = selection_note
    destination.mkdir(parents=True, exist_ok=True)
    filename = "smoke.json.gz" if limit else "scores.json.gz"
    if gliner_trajectory:
        filename = "gliner-paper-trajectory-" + filename
    if full_human:
        filename = "gliner-fullhuman-" + ("smoke.json.gz" if limit else "scores.json.gz")
    write(destination / filename, result)


def score_points(
    sweep_points,
    selected,
    identity,
    *,
    human,
    ontology,
    sidecar,
    projection,
    expressible=None,
    mapped_source=False,
    typed_view=False,
    full_human=False,
    preview=None,
    preview_labels=None,
    source="sweep",
):
    """Per-input region (and optionally typed) counts for every sweep threshold.

    `human` is the loaded `human-gold-v1/score.py` module; `identity` maps each
    prediction input id to its text; `selected` holds the scored gold rows.
    """
    points = []
    for threshold, predicted in sweep_points.items():
        if full_human and float(threshold) != 0.5:
            continue
        by_id = {p["id"]: p["preds"] for p in predicted}
        if len(by_id) != len(predicted) or set(by_id) != set(identity):
            raise ValueError(f"Prediction identities differ: {source}, {threshold}")
        counts = {str(overlap): [] for overlap in (80, 100)}
        typed = []
        for row in selected:
            preds = by_id[row["id"]]
            if mapped_source:
                preds = [
                    {**span, "label": label}
                    for span in preds
                    for label in ontology.source_acceptable(projection, span["label"])
                    if label != "O"
                ]
            expected, actual, neutral, masked = human.project(
                row,
                preds,
                "ont3" if mapped_source else projection,
                ontology,
                titles=human.title_extents(row, sidecar),
                expressible=expressible,
            )
            for overlap in (80, 100):
                tp = human.plot._maximum_matches(
                    expected, actual, lambda a, b: human.plot.compatible(a, b, overlap)
                )
                counts[str(overlap)].append(
                    {
                        "id": row["id"],
                        "lang": row["scoring_language"],
                        "group": source_group(row),
                        "counts": [tp, len(actual), len(expected)],
                        "neutral": neutral,
                        "masked": masked,
                    }
                )
            if typed_view:
                gold_set, pred_set, _ = human.optional_reference_spans(
                    {tuple(s) for s in row["spans"]},
                    {(s["start"], s["end"], s["label"]) for s in preds},
                )
                if full_human:
                    # Preserve the campaign's predeclared selection task:
                    # references excluded, without named-on-reference neutrality.
                    gold_set = {tuple(s) for s in row["spans"] if s[2] not in human.plot.REFERENCES}
                    pred_set = {
                        (s["start"], s["end"], s["label"])
                        for s in preds
                        if s["label"] not in human.plot.REFERENCES
                    }
                typed.append(
                    {
                        "id": row["id"],
                        "lang": row["scoring_language"],
                        "group": source_group(row),
                        "counts": [len(gold_set & pred_set), len(pred_set), len(gold_set)],
                    }
                )
            if preview is not None and row is selected[0] and float(threshold) in (0, 0.5):
                preview.append(
                    {
                        **(preview_labels or {}),
                        "threshold": float(threshold),
                        "id": row["id"],
                        "text": row["text"],
                        "predictions": preds,
                        "gold_regions": expected,
                        "predicted_regions": actual,
                    }
                )
        point = {"threshold": float(threshold), "regions": {}}
        for overlap, items in counts.items():
            point["regions"][overlap] = {
                "metrics": metrics([sum(r["counts"][i] for r in items) for i in range(3)]),
                "per_input": items,
            }
        if typed:
            point["typed"] = {
                "metrics": metrics([sum(r["counts"][i] for r in typed) for i in range(3)]),
                "per_input": typed,
            }
        points.append(point)
    return points


def source_group(row):
    provenance = row.get("context_provenance", {})
    if row.get("source_group_id"):
        return row["source_group_id"]
    if provenance.get("document"):
        return str((provenance.get("file"), provenance["document"]))
    return row["id"]


def main():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument(
        "--limit", type=int, default=0, help="Smoke row limit per population; zero scores every row"
    )
    parser.add_argument(
        "--gliner-trajectory",
        action="store_true",
        help="Score the new GLiNER2 checkpoints on development only",
    )
    parser.add_argument(
        "--gliner-fullhuman",
        action="store_true",
        help="Score fixed-confidence GLiNER2 checkpoints on the full selection population",
    )
    parser.add_argument(
        "--sweep",
        nargs=4,
        action="append",
        metavar=("MODEL", "POPULATION", "FILE", "PROJECTION"),
        help="Score this saved sweep instead of the default systems; repeat for each model/population",
    )
    parser.add_argument("--destination", type=Path, default=DEST, help="Directory for score artifacts")
    parser.add_argument("--selection-note", help="Describe the selection status of explicit comparisons")
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    sweeps = None
    if args.sweep:
        if any(population not in {"human", "ont3", "heldout"} for _, population, _, _ in args.sweep):
            parser.error("--sweep POPULATION must be human, ont3 or heldout")
        keys = [(model, population) for model, population, _, _ in args.sweep]
        if len(set(keys)) != len(keys):
            parser.error("--sweep model/population pairs must be unique")
        sweeps = [
            (model, population, Path(path), projection) for model, population, path, projection in args.sweep
        ]
    score_paper_o4(
        args.limit,
        args.gliner_trajectory or args.gliner_fullhuman,
        args.gliner_fullhuman,
        sweeps=sweeps,
        destination=args.destination,
        selection_note=args.selection_note,
    )


if __name__ == "__main__":
    main()
