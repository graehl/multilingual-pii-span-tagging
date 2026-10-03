#!/usr/bin/env python3
# acli: 1 complete
"""Pool current paper comparisons by counts and add a fixed Presidio point."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import distribution, version
from pathlib import Path

sys.path.insert(0, str(Path.home() / "agents"))
import acli

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/pii/frontier/evidence"
DEST = EVIDENCE / "pooled-gold-languages-v1"
HUMAN = Path.home() / "artifacts/pii-redaction-frontier/human-gold-overlap-v1/comparison-v1"
ONT3 = EVIDENCE / "ont3-five-model-v1"
BOUNDARY = EVIDENCE / "boundary-serving-v2"
PRESIDIO_SCHEMA = "presidio_fork_2_2_364_multilingual"
PRESIDIO_COMMIT = "a7b17c75f3098b92b369f0b01855519f1cd5e8cc"
# OpenAI Privacy Filter and its published OpenMed fine-tunes, swept by
# `pii_eval.py predict-bias-grid` over `paper-pooled-opf-v1` (this
# evaluation's id/text rows). Key: (label, Hugging Face id, revision,
# ontology source schema, prediction prefix).
FILTERS = {
    "opf": (
        "OpenAI Privacy Filter",
        "openai/privacy-filter",
        "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
        "openai_8",
        "paper-opf-openai-v1",
    ),
    "opf-openmed-multi": (
        "OpenMed multilingual",
        "OpenMed/privacy-filter-multilingual",
        "f914f18d909d541d288dfda44a4d0b6bdb638d57",
        "openmed_54",
        "paper-opf-openmed-multi-v1",
    ),
    "opf-openmed-multi2": (
        "OpenMed multilingual v2",
        "OpenMed/privacy-filter-multilingual-v2",
        "0d0c0430fa386435ace1d9f842f0f966903a9ac8",
        "openmed_54",
        "paper-opf-openmed-multi2-v1",
    ),
    "opf-openmed-nemo": (
        "OpenMed Nemotron",
        "OpenMed/privacy-filter-nemotron",
        "f6f3a633dc6fc96a644d20218633a98ac43f6c71",
        "openmed_nemotron_55",
        "paper-opf-openmed-nemo-v1",
    ),
    "opf-openmed-nemo2": (
        "OpenMed Nemotron v2",
        "OpenMed/privacy-filter-nemotron-v2",
        "968247329d18998cc7d5338b941b52f1b2a9abd9",
        "openmed_nemotron_55",
        "paper-opf-openmed-nemo2-v1",
    ),
}
FILTER_DATASET = "paper-pooled-opf-v1"
FILTER_BIASES = (8, 6, 4, 2, 1, 0, -1, -2, -3, -4, -6, -8, -12)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    """Retain existing split membership; check duplicate text before pooling."""
    DEST.mkdir(exist_ok=True)
    human_path = HUMAN / "evaluation.jsonl"
    ont3_path = ONT3 / "gold-manual-r3.jsonl"
    human = [row for row in rows(human_path) if row["evaluation_source"] != "ont3"]
    ont3 = rows(ont3_path)
    changes = read_json(ONT3 / "language-review.json")["changes_by_id_prefix"]
    languages = sorted({row.get("scoring_language", row["lang"]) for row in human})
    output = []
    for population, source in (("human", human), ("ont3", ont3)):
        for original in source:
            row = dict(original)
            row["original_id"] = row["id"]
            row["id"] = population + "::" + row["id"]
            row["population"] = population
            row["scoring_language"] = (
                changes.get(original["id"].split(":")[0], row["lang"])
                if population == "ont3"
                else row.get("scoring_language", row["lang"])
            )
            if population == "ont3":
                row["evaluation_source"] = "ont3"
            output.append(row)
    if len({r["id"] for r in output}) != len(output):
        raise ValueError("Duplicate qualified input identity")
    pooled = [r for r in output if r["scoring_language"] in languages]
    seen = {}
    for row in pooled:
        normalized = " ".join(unicodedata.normalize("NFC", row["text"]).split())
        if normalized in seen:
            raise ValueError(
                f"Pooled duplicate requires explicit resolution: {seen[normalized]}, {row['id']}"
            )
        seen[normalized] = row["id"]
    path = DEST / "evaluation.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    sources = [
        human_path,
        HUMAN / "inputs.jsonl",
        ont3_path,
        ONT3 / "inputs.jsonl",
        ONT3 / "language-review.json",
    ]
    manifest = {
        "schema": "pii-paper-pooled-membership/v1",
        "gold_languages": languages,
        "sources": {str(p): sha(p) for p in sources},
        "evaluation_sha256": sha(path),
        "N": {"human": len(human), "ont3": len(ont3), "pooled": len(pooled)},
        "pooled_languages": dict(Counter(r["scoring_language"] for r in pooled)),
        "pooled_components": dict(Counter(r["population"] for r in pooled)),
        "duplicates": {"exact_or_NFC_whitespace": 0, "checked": len(pooled)},
        "scope": "Aggregation of existing evaluation memberships; no new split, annotation or unseen-data certification",
    }
    write(DEST / "membership.json", manifest)
    print(json.dumps({"phase": "prepare", **manifest["N"], "languages": languages}), flush=True)


def predict(limit, input_path=None, output_dir=None):
    """Run the previously used multilingual Presidio configuration on CPU."""
    import spacy
    from pii_presidio_fresh20 import MULTILINGUAL, SPACY_LG, build_engine
    from presidio_analyzer import RecognizerResult

    spacy.require_cpu()
    installed = distribution("presidio-analyzer")
    source = json.loads(installed.read_text("direct_url.json"))
    if version("presidio-analyzer") != "2.2.364" or source["vcs_info"]["commit_id"] != PRESIDIO_COMMIT:
        raise ValueError("Presidio environment differs from the pinned paper baseline")
    input_path = input_path or DEST / "evaluation.jsonl"
    output_dir = output_dir or DEST
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = rows(input_path)
    grouped = defaultdict(list)
    for row in evaluation:
        grouped[row.get("scoring_language", row["lang"])].append(row)
    result, configurations, preview = {}, {}, []
    for language, group in sorted(grouped.items()):
        model = SPACY_LG.get(language, MULTILINGUAL)
        if not spacy.util.is_package(model):
            raise ValueError(f"Missing pinned spaCy model: {model}")
        # This fork validates two-letter engine language codes. The shared
        # multilingual model still handles Filipino and language-free inputs.
        engine_language = {"fil": "tl", "und": "xx"}.get(language, language)
        engine = build_engine(engine_language, model, None)
        if engine.default_score_threshold != 0:
            raise ValueError("The fixed Presidio point requires its recorded default threshold of zero")
        started = time.monotonic()
        for row in group[:limit] if limit else group:
            predictions = [
                {
                    "start": s.start,
                    "end": s.end,
                    "label": s.entity_type,
                    "score": s.score,
                    "recognizer": (s.recognition_metadata or {}).get(RecognizerResult.RECOGNIZER_NAME_KEY),
                }
                for s in engine.analyze(text=row["text"], language=engine_language)
            ]
            if any(not 0 <= p["start"] < p["end"] <= len(row["text"]) for p in predictions):
                raise ValueError(f"Invalid prediction offsets: {row['id']}")
            result[row["id"]] = {"id": row["id"], "preds": predictions}
            if len(preview) < 2 or row == group[0]:
                preview.append(
                    {
                        "id": row["id"],
                        "language": language,
                        "text": row["text"],
                        "predictions": [
                            {**p, "surface": row["text"][p["start"] : p["end"]]} for p in predictions
                        ],
                    }
                )
        configurations[language] = {
            "engine_language": engine_language,
            "model": model,
            "model_version": version(model),
            "route": "native-lg" if language in SPACY_LG else "multilingual",
            "threshold": engine.default_score_threshold,
            "entities": sorted(engine.get_supported_entities(engine_language)),
            "seconds": time.monotonic() - started,
        }
        print(
            json.dumps(
                {
                    "phase": "presidio",
                    "language": language,
                    "rows": min(limit, len(group)) if limit else len(group),
                    **configurations[language],
                }
            ),
            flush=True,
        )
        del engine
    output = {
        "schema": "pii-presidio-paper-point/v1",
        "input_sha256": sha(input_path),
        "package_version": version("presidio-analyzer"),
        "source": source,
        "spacy_version": version("spacy"),
        "configurations": configurations,
        "rows": list(result.values()),
        "smoke": bool(limit),
    }
    stem = "presidio-smoke" if limit else "presidio"
    write(output_dir / f"{stem}.json.gz", output)
    write(output_dir / f"{stem}-preview.json", preview)


def bias_slug(value):
    """Match `pii_eval.o_logit_bias_slug` for the integer grid used here."""
    return f"{'m' if value < 0 else 'p'}{abs(value) * 100:03d}"


def collect_filters():
    """Bundle each filter's raw bias-grid predictions with its identity."""
    import os

    evaluation = rows(DEST / "evaluation.jsonl")
    home = Path(os.environ.get("PII_EVAL_HOME") or ROOT / "untracked/pii-eval")
    dataset = home / "gold" / f"{FILTER_DATASET}.jsonl"
    inputs = rows(dataset)
    if [(r["id"], r["text"]) for r in inputs] != [(r["id"], r["text"]) for r in evaluation]:
        raise ValueError(f"{dataset} does not match evaluation.jsonl ids and texts in order")
    for key, (label, model_id, revision, schema, prefix) in FILTERS.items():
        points = {}
        for bias in FILTER_BIASES:
            path = home / "pred" / f"{prefix}-obias-{bias_slug(bias)}.{FILTER_DATASET}.jsonl"
            predicted = rows(path)
            if [p["id"] for p in predicted] != [r["id"] for r in evaluation]:
                raise ValueError(f"{path} does not align with the evaluation rows")
            for row, p in zip(evaluation, predicted, strict=True):
                if any(not 0 <= s["start"] < s["end"] <= len(row["text"]) for s in p["preds"]):
                    raise ValueError(f"Invalid prediction offsets in {path}: {p['id']}")
            points[str(bias)] = [{"id": p["id"], "preds": p["preds"]} for p in predicted]
        output = {
            "schema": "pii-paper-filter-sweep/v1",
            "model": key,
            "label": label,
            "hf_id": model_id,
            "hf_revision": revision,
            "ontology_schema": schema,
            "decoder": "pii_eval.py predict-bias-grid; character-conservative windowing; O-logit bias grid",
            "input_sha256": sha(DEST / "evaluation.jsonl"),
            "dataset_sha256": sha(dataset),
            "o_logit_biases": list(FILTER_BIASES),
            "points": points,
        }
        write(DEST / f"filter-{key}.json.gz", output)
        spans = {bias: sum(len(p["preds"]) for p in value) for bias, value in points.items()}
        print(json.dumps({"phase": "collect-filters", "model": key, "spans_by_bias": spans}), flush=True)


def filter_counts(key, evaluation, ontology, human_score):
    """Per-bias, per-input counts for one filter under the shared projection."""
    bundle = read_json(DEST / f"filter-{key}.json.gz")
    if bundle["input_sha256"] != sha(DEST / "evaluation.jsonl"):
        raise ValueError(f"filter-{key} predictions are for a different evaluation")
    schema = bundle["ontology_schema"]
    counts = {}
    for bias, predicted in bundle["points"].items():
        by_id = {p["id"]: p for p in predicted}
        counts[float(bias)] = {}
        for row in evaluation:
            canonical = []
            for span in by_id[row["id"]]["preds"]:
                allowed = ontology.source_acceptable(schema, span["label"])
                if not allowed:
                    raise ValueError(f"Unmapped {key} label {span['label']}")
                canonical.extend({**span, "label": label} for label in allowed if label != "O")
            expected, actual, neutral, masked = human_score.project(row, canonical, "ont3", ontology)
            counts[float(bias)][row["id"]] = {}
            for overlap in (80, 90, 100):
                tp = human_score.plot._maximum_matches(
                    expected, actual, lambda a, b: human_score.plot.compatible(a, b, overlap)
                )
                counts[float(bias)][row["id"]][str(overlap)] = {
                    "id": row["id"],
                    "counts": [tp, len(actual), len(expected)],
                    "neutral_predictions": neutral,
                    "masked_predictions": masked,
                }
    return bundle, counts


def fixed_points(counts_by_threshold, selected, overlap):
    """Pool per-input counts at each measured threshold for one view."""
    scored = []
    for threshold, by_id in sorted(counts_by_threshold.items()):
        per_input = [by_id[r["id"]][str(overlap)] for r in selected]
        totals = defaultdict(lambda: [0, 0, 0])
        for row, item in zip(selected, per_input, strict=True):
            for key in (row["scoring_language"], "overall"):
                totals[key] = [a + b for a, b in zip(totals[key], item["counts"], strict=True)]
        scored.append(
            {
                "threshold": threshold,
                "per_input": per_input,
                "panels": {key: metrics(value) for key, value in totals.items()},
            }
        )
    return scored


def pool_points(sources, membership, metric):
    """Join identical thresholds, then sum counts from the selected unique inputs."""
    by_population = defaultdict(dict)
    for row in membership:
        by_population[row["population"]][row["original_id"]] = row
    result = {}
    models = set.intersection(*(set(sources[p]["scores"][metric]) for p in by_population))
    for model in sorted(models):
        points = {p: {s["threshold"]: s for s in sources[p]["scores"][metric][model]} for p in by_population}
        thresholds = set.intersection(*(set(value) for value in points.values()))
        if not thresholds:
            raise ValueError(f"No common measured operating points for {model}")
        scored = []
        for threshold in sorted(thresholds):
            per_input, totals = [], defaultdict(lambda: [0, 0, 0])
            for population, selected in by_population.items():
                available = {r["id"]: r for r in points[population][threshold]["per_input"]}
                if not selected.keys() <= available.keys():
                    raise ValueError(f"Missing {model} predictions on {population}")
                for identifier, row in selected.items():
                    item = {**available[identifier], "id": row["id"]}
                    per_input.append(item)
                    for key in (row["scoring_language"], "overall"):
                        totals[key] = [a + b for a, b in zip(totals[key], item["counts"], strict=True)]
            scored.append(
                {
                    "threshold": threshold,
                    "per_input": per_input,
                    "panels": {key: metrics(value) for key, value in totals.items()},
                }
            )
        result[model] = scored
    return result


def metrics(counts):
    tp, predicted, gold = counts
    return {
        "tp": tp,
        "predicted": predicted,
        "gold": gold,
        "P": tp / predicted if predicted else 0,
        "R": tp / gold if gold else 0,
        "F1": 2 * tp / (predicted + gold) if predicted + gold else 0,
    }


def score():
    human_score = load_module("paper_human_score", EVIDENCE / "human-gold-v1/score.py")
    from pii_ontology_v2 import load_ontology

    ontology = load_ontology()
    membership = read_json(DEST / "membership.json")
    evaluation = rows(DEST / "evaluation.jsonl")
    if membership["evaluation_sha256"] != sha(DEST / "evaluation.jsonl"):
        raise ValueError("Membership changed after preparation")
    sources = {
        p: read_json(BOUNDARY / f"boundary-{name}-scores.json.gz")
        for p, name in (("human", "human"), ("ont3", "all35"))
    }
    if sources["human"]["inputs"]["evaluation.jsonl"] != sha(HUMAN / "evaluation.jsonl"):
        raise ValueError("Human reference differs from saved current-model counts")
    if sources["ont3"]["inputs"]["gold-manual-r3.jsonl"] != sha(ONT3 / "gold-manual-r3.jsonl"):
        raise ValueError("Ont3 reference differs from saved current-model counts")
    if sources["human"]["mapping_sha256"] != sha(ROOT / "scripts/pii_tagset_v2.yaml"):
        raise ValueError("Ontology mapping changed; rescore all models first")
    if sources["human"]["negative_coverage_sha256"] != sha(human_score.NEGATIVE_COVERAGE_PATH):
        raise ValueError("Negative coverage changed; rescore all models first")
    presidio = read_json(DEST / "presidio.json.gz")
    if presidio["smoke"] or presidio["input_sha256"] != membership["evaluation_sha256"]:
        raise ValueError("Presidio predictions are not the complete current evaluation")
    predictions = {p["id"]: p for p in presidio["rows"]}
    if len(predictions) != len(presidio["rows"]) or set(predictions) != {r["id"] for r in evaluation}:
        raise ValueError("Presidio input identities do not align")
    preview, presidio_counts = [], {}
    for row in evaluation:
        canonical = []
        for span in predictions[row["id"]]["preds"]:
            allowed = ontology.source_acceptable(PRESIDIO_SCHEMA, span["label"])
            if not allowed:
                raise ValueError(f"Unmapped Presidio label {span['label']}")
            canonical.extend({**span, "label": label} for label in allowed if label != "O")
        expected, actual, neutral, masked = human_score.project(row, canonical, "ont3", ontology)
        presidio_counts[row["id"]] = {}
        for overlap in (80, 90, 100):
            tp = human_score.plot._maximum_matches(
                expected, actual, lambda a, b: human_score.plot.compatible(a, b, overlap)
            )
            presidio_counts[row["id"]][str(overlap)] = {
                "id": row["id"],
                "counts": [tp, len(actual), len(expected)],
                "neutral_predictions": neutral,
                "masked_predictions": masked,
            }
        if len(preview) < 7 or (row["population"] == "ont3" and len(preview) < 10):
            preview.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "gold_regions": expected,
                    "predicted_regions": actual,
                    "counts": presidio_counts[row["id"]],
                }
            )
    # Fixed-decoder baselines scored directly on every evaluation row:
    # Presidio's single point and each privacy filter's O-bias grid.
    fixed = {"presidio": {0.0: presidio_counts}}
    filter_models = {}
    for key in FILTERS:
        bundle, fixed[key] = filter_counts(key, evaluation, ontology, human_score)
        filter_models[key] = {
            name: bundle[name] for name in ("label", "hf_id", "hf_revision", "ontology_schema", "decoder")
        }
    views = {
        "pooled": [r for r in evaluation if r["scoring_language"] in membership["gold_languages"]],
        "human": [r for r in evaluation if r["population"] == "human"],
        "ont3": [r for r in evaluation if r["population"] == "ont3"],
    }
    output = {
        "schema": "pii-paper-pooled-curves/v1",
        "membership": membership,
        "inputs": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                BOUNDARY / "boundary-human-scores.json.gz",
                BOUNDARY / "boundary-all35-scores.json.gz",
                DEST / "presidio.json.gz",
                *(DEST / f"filter-{key}.json.gz" for key in FILTERS),
            ]
        },
        "models_by_population": {p: s["models"] for p, s in sources.items()},
        "filter_models": filter_models,
        "reference_policy": sources["human"]["reference_policy"],
        "coverage_policy": sources["human"]["coverage_policy"],
        "views": {},
    }
    for view, selected in views.items():
        evidence = {
            "N": dict(Counter(r["scoring_language"] for r in selected)),
            "aggregate_label": {
                "human": "Human gold · 7 languages",
                "ont3": "Ont3 · 35 languages",
                "pooled": "Human gold + Ont3 · 7 languages",
            }[view]
            + f" · N = {len(selected):,}",
            "scores": {},
        }
        for overlap in (80, 90, 100):
            values = pool_points(sources, selected, str(overlap))
            reference_gold = {r["id"]: r["counts"][2] for r in values["ont3"][0]["per_input"]}
            for model, counts in fixed.items():
                values[model] = fixed_points(counts, selected, overlap)
                for point in values[model]:
                    if any(item["counts"][2] != reference_gold[item["id"]] for item in point["per_input"]):
                        raise ValueError(f"{model} scorer's gold differs from shared scorer on {view}")
            evidence["scores"][str(overlap)] = values
        output["views"][view] = evidence
    write(DEST / "scores.json.gz", output)
    write(DEST / "score-preview.json", preview)
    summary = {}
    for view, evidence in output["views"].items():
        summary[view] = {
            model: {
                **max(points, key=lambda p: p["panels"]["overall"]["F1"])["panels"]["overall"],
                "threshold": max(points, key=lambda p: p["panels"]["overall"]["F1"])["threshold"],
            }
            for model, points in evidence["scores"]["80"].items()
        }
    write(DEST / "summary.json", summary)
    print(json.dumps({"phase": "score", "summary": summary}), flush=True)


def render():
    plot = load_module("paper_plot", EVIDENCE / "priority9-shared-v1/render-comparison.py")
    result = read_json(DEST / "scores.json.gz")
    models = {**plot.MODELS, "presidio": ("Presidio", "#B22222", "None", "X")}
    languages = {
        key: name for key, name in plot.LANGUAGES.items() if key in result["membership"]["gold_languages"]
    }
    for view, evidence in result["views"].items():
        plot.render(
            evidence,
            f"{view}-gold-languages-six-model",
            models=models,
            aggregate_only=view == "ont3",
            languages=languages,
            precision_floor=0.4,
        )
    # Main-body figure: human gold by language, then Ont3 pooled over all 35
    # languages, over the same seven, and English alone, with one legend.
    ont3 = result["views"]["ont3"]
    seven = [lang for lang in languages if lang in ont3["N"]]
    row = {"scores": {}}
    for coverage, by_model in ont3["scores"].items():
        row["scores"][coverage] = {
            model: [
                {
                    "threshold": point["threshold"],
                    "panels": {
                        "all": point["panels"]["overall"],
                        "seven": metrics(
                            [
                                sum(point["panels"][lang][key] for lang in seven)
                                for key in ("tp", "predicted", "gold")
                            ]
                        ),
                        "en": point["panels"]["en"],
                    },
                }
                for point in points
            ]
            for model, points in by_model.items()
        }
    total = sum(ont3["N"].values())
    plot.render(
        result["views"]["human"],
        "human-gold-ont3-rows-six-model",
        models=models,
        languages=languages,
        precision_floor=0.4,
        recall_floor=0.4,
        extra_row=(
            row,
            [
                # "Ont3 annotations", not bare "Ont3", which also names a model.
                ("all", f"Ont3 annotations\n35 languages · N = {total}"),
                (
                    "seven",
                    f"Ont3 annotations\nsame {len(seven)} · N = {sum(ont3['N'][lang] for lang in seven)}",
                ),
                ("en", f"Ont3 annotations\nEnglish · N = {ont3['N']['en']}"),
            ],
        ),
    )
    plot.render(read_json(BOUNDARY / "boundary-ont3-scores.json.gz"), "ont3-priority9-five-model")


def main():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument("stage", choices=("prepare", "predict", "collect-filters", "score", "render"))
    parser.add_argument("--limit-per-language", type=int, default=0, help="Smoke only; zero runs all rows")
    parser.add_argument(
        "--input", type=Path, help="Prediction input; defaults to the pooled paper population"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Prediction output directory; defaults to the pooled evidence directory",
    )
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args.limit_per_language, args.input, args.output_dir)
    else:
        {
            "prepare": prepare,
            "collect-filters": collect_filters,
            "score": score,
            "render": render,
        }[args.stage]()


if __name__ == "__main__":
    main()
