"""Score frozen human-gold targets; keep Ont3 language supplements separate."""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from pii_ontology_v2 import load_ontology
from pii_reference_projection import OPTIONAL_REFERENCE_POLICY, optional_reference_spans

spec = importlib.util.spec_from_file_location(
    "paper_plot", HERE.parent / "priority9-shared-v1/render-comparison.py"
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
DATA = Path.home() / "artifacts/pii-redaction-frontier/human-gold-overlap-v1/comparison-v1"
SCHEMAS = {
    "mapa": "mapa_coarse",
    "aqmar-openner": "aqmar_core",
    "openner-commercial-core": "openner_core",
    "wojood-sample": "wojood_nested",
}
NEGATIVE_COVERAGE_PATH = HERE.parent / "four-corpus-v1/negative-coverage-v1.json"
NEGATIVE_COVERAGE = json.loads(NEGATIVE_COVERAGE_PATH.read_text())["negative_types_by_corpus"]
COVERAGE_POLICY = "corpus-negative-coverage-with-annotated-overlap/v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Title extents (user-directed, 2026-09-27). A job title, office or role beside or
# inside a gold person span is its own region: scored for a model that can emit
# demographic_attribute (covered by any predicted span, so redacting it inside a
# name costs nothing) and unscored for one that cannot. An honorific extent is
# neutral either way. The name's non-title part is always required.
TITLE_POLICY = "title-extents/v1"
MAPA_FINE_KINDS = {"TITLE": "honorific", "ROLE": "title", "PROFESSION": "title"}


def _adjacent(text, a, b):
    """True when spans a and b touch or are separated by whitespace only."""
    left, right = sorted((a, b))
    return left[1] <= right[0] and not text[left[1] : right[0]].strip()


def title_extents(row, sidecar=None):
    """Title and honorific extents beside or inside a gold person span of one row."""
    text, source, found = row["text"], row["evaluation_source"], []
    if source == "mapa":
        tokens, fine = row["metadata"]["tokens"], row["metadata"]["fine_grained"]
        if " ".join(tokens) != text:
            raise ValueError(f"{row['id']}: MAPA text is not the single-space token join")
        persons = [(s["start"], s["end"]) for s in row["spans"] if s["source_label"] == "PERSON"]
        position, current = 0, None
        for token, tag in zip(tokens, fine, strict=True):
            start, end = position, position + len(token)
            position = end + 1
            kind = MAPA_FINE_KINDS.get(tag[2:])
            if kind and any(a <= start and end <= b for a, b in persons):
                if current and tag.startswith("I-") and current["kind"] == kind:
                    current["end"] = end
                else:
                    current = {"start": start, "end": end, "kind": kind, "source": "mapa-fine-layer"}
                    found.append(current)
            else:
                current = None
    elif source == "wojood-sample":
        persons = [(s["start"], s["end"]) for s in row["spans"] if s["source_label"] == "PERS"]
        for span in row["spans"]:
            extent = (span["start"], span["end"])
            if span["source_label"] == "OCC" and any(_adjacent(text, extent, p) for p in persons):
                found.append({"start": extent[0], "end": extent[1], "kind": "title", "source": "wojood-occ"})
    elif source == "ont3":
        persons = [(s, e) for s, e, t in row["spans"] if t == "person_name"]
        for s, e, t in row["spans"]:
            if t == "demographic_attribute" and any(_adjacent(text, (s, e), p) for p in persons):
                found.append({"start": s, "end": e, "kind": "title", "source": "ont3-gold"})
    return found + list((sidecar or {}).get(row["id"], []))


def _outside(start, end, extents):
    """Pieces of [start, end) outside every extent."""
    pieces = [(start, end)]
    for extent in extents:
        pieces = [
            piece
            for a, b in pieces
            for piece in ((a, min(b, extent["start"])), (max(a, extent["end"]), b))
            if piece[0] < piece[1]
        ]
    return pieces


def _apply_titles(text, required, retained, titles, capable):
    """Split gold and predictions at title extents; titles become their own regions."""
    scored = [t for t in titles if t["kind"] == "title"] if capable else []
    expected = plot.regions(
        [dict(start=a, end=b, type=t) for s, e, t in required for a, b in _outside(s, e, titles)], text
    )
    actual = plot.regions(
        [dict(start=a, end=b, type=t) for s, e, t in retained for a, b in _outside(s, e, titles)], text
    )
    for title in scored:
        region = plot.regions([dict(start=title["start"], end=title["end"], type="title")], text)
        covered = plot.regions(
            [
                dict(start=max(s, title["start"]), end=min(e, title["end"]), type=t)
                for s, e, t in retained
                if s < title["end"] and title["start"] < e
            ],
            text,
        )
        expected += region
        actual += covered
    return expected, actual


def project(row, prediction, model, ontology, titles=None, expressible=None):
    """Mask unannotated categories before untyped region matching.

    titles: title_extents() of the row to apply TITLE_POLICY, else None.
    expressible: the primary types the model can emit; gold spans of other types
    are unscored (the shared-primary-category precedent), and title regions are
    scored only when demographic_attribute is expressible. None keeps every type.
    """
    source = row["evaluation_source"]
    if source == "ont3":
        gold = {tuple(span) for span in row["spans"]}
        supported = set(ontology.primary_types)
    else:
        schema = SCHEMAS[source]
        supported = {
            tag
            for label in ontology.source_labels(schema)
            for tag in ontology.source_acceptable(schema, label)
        } - {"O"}
        gold = {
            (span["start"], span["end"], tag)
            for span in row["spans"]
            for tag in ontology.source_acceptable(schema, span["source_label"])
            if tag != "O"
        }
    if expressible is not None:
        gold = {span for span in gold if span[2] in expressible}
    projected = set()
    masked = 0
    neutral_count = 0
    required, _, _ = optional_reference_spans(gold, set())
    for span in prediction:
        label = span["label"]
        if label in plot.REFERENCES:
            continue
        accepted = (
            ontology.canonical_acceptable(label)
            if model == "ont1"
            else ontology.source_acceptable("fastino_42", label)
            if model == "gliner2" and label != "organization"
            else (label,)
        )
        # A title extent is annotated even where the corpus has no
        # demographic_attribute category (MAPA's fine layer).
        on_title = titles is not None and any(
            t["start"] < span["end"] and span["start"] < t["end"] for t in titles
        )
        allowed = set(accepted) & (supported | ({"demographic_attribute"} if on_title else set()))
        if source != "ont3":
            # Positive label mappings do not establish exhaustive negative
            # coverage. Keep an overlapping annotation's full predicted extent
            # so incomplete coverage cannot forgive a boundary error.
            allowed = {
                tag
                for tag in allowed
                if tag in NEGATIVE_COVERAGE[source]
                or any(
                    gold_tag == tag and span["start"] < end and start < span["end"]
                    for start, end, gold_tag in gold
                )
                or (on_title and tag == "demographic_attribute")
            }
        masked += not allowed
        possible = {(span["start"], span["end"], tag) for tag in allowed}
        _, _, neutral = optional_reference_spans(gold, possible)
        if neutral and not possible & required:
            neutral_count += 1
            continue
        projected.update(possible)
    required, retained, ignored = optional_reference_spans(gold, projected)
    if titles is not None:
        capable = expressible is None or "demographic_attribute" in expressible
        expected, actual = _apply_titles(row["text"], required, retained, titles, capable)
        return expected, actual, neutral_count + len(ignored), masked
    expected = plot.regions([dict(start=a, end=b, type=t) for a, b, t in required], row["text"])
    actual = plot.regions([dict(start=a, end=b, type=t) for a, b, t in retained], row["text"])
    return expected, actual, neutral_count + len(ignored), masked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned-sweep", type=Path)
    parser.add_argument("--ont3-sweep", type=Path, help="selected Ont3 XLM-R predictions")
    parser.add_argument("--ont2-sweep", type=Path, help="selected Ont2 predictions with serving refinement")
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--title-policy",
        action="store_true",
        help=f"apply {TITLE_POLICY} and score only the primary types each model can express",
    )
    parser.add_argument(
        "--title-sidecar",
        type=Path,
        help="JSONL of {id, extents} neutral title/descriptor extents for --title-policy (finder output)",
    )
    args = parser.parse_args()
    ontology = load_ontology()
    sidecar = {}
    if args.title_sidecar:
        for line in args.title_sidecar.open(encoding="utf-8"):
            entry = json.loads(line)
            sidecar[entry["id"]] = entry["extents"]
    # Primary types each model can emit, filled per sweep below: base GLiNER2's
    # requested labels mapped exactly as project() maps its predictions.
    expressible = {}
    gold = [json.loads(line) for line in (DATA / "evaluation.jsonl").open()]
    assert len(gold) == len({r["id"] for r in gold}) == len({r["text"] for r in gold}) == 1313
    human_N = sum(r["evaluation_source"] != "ont3" for r in gold)
    assert human_N == 1283
    evidence = dict(
        N=dict(Counter(r.get("scoring_language", r["lang"]) for r in gold)),
        aggregate_label=f"Human gold · 7 languages · N = {human_N:,}",
        language_labels={**plot.LANGUAGES, "ko": "Korean (Ont3)", "vi": "Vietnamese (Ont3)"},
        split="Official human-gold test rows; Ont3 reused development supplements",
        aggregate="Human-gold-only pooled counts; Korean/Vietnamese panels are separate Ont3 supplements",
        reference_policy=OPTIONAL_REFERENCE_POLICY,
        coverage_policy=COVERAGE_POLICY,
        title_policy=TITLE_POLICY if args.title_policy else None,
        title_sidecar=dict(path=str(args.title_sidecar), sha256=digest(args.title_sidecar))
        if args.title_sidecar
        else None,
        expressible=None,
        negative_coverage_sha256=digest(NEGATIVE_COVERAGE_PATH),
        source_schemas=SCHEMAS,
        projection="Corpus-specific many:many supported-type mask, then untyped maximal redaction regions",
        inputs={name: digest(DATA / name) for name in ["evaluation.jsonl", "inputs.jsonl"]},
        mapping_sha256=digest(ROOT / "scripts/pii_tagset_v2.yaml"),
        models={},
        scores={},
    )
    for model in plot.MODELS:
        grid = "dense" if model.startswith("ont") else "extended"
        path = HERE / f"{model}-{grid}.json"
        if model == "gliner2-tuned" and args.tuned_sweep:
            path = args.tuned_sweep
        if model == "ont3" and args.ont3_sweep:
            path = args.ont3_sweep
        if model == "ont2" and args.ont2_sweep:
            path = args.ont2_sweep
        sweep = json.loads(path.read_text())
        assert sweep["rows"] == len(gold) and sweep["input_sha256"] == evidence["inputs"]["inputs.jsonl"]
        if model == "gliner2":
            assert "organization" in sweep["requested_labels"]
            expressible[model] = {
                tag
                for label in sweep["requested_labels"]
                for tag in (
                    (label,) if label == "organization" else ontology.source_acceptable("fastino_42", label)
                )
            } - {"O"}
        evidence["models"][model] = dict(path=path.name, sha256=digest(path), checkpoint=sweep["model_path"])
        for coverage in [80, 90, 100]:
            scored = []
            for threshold, predictions in sorted(sweep["points"].items(), key=lambda item: float(item[0])):
                totals = {lang: [0, 0, 0] for lang in plot.LANGUAGES}
                human, supplement, per_input = [0, 0, 0], [0, 0, 0], []
                assert len(predictions) == len(gold)
                for row, prediction in zip(gold, predictions, strict=True):
                    assert row["id"] == prediction["id"]
                    expected, actual, ignored, masked = project(
                        row,
                        prediction["preds"],
                        model,
                        ontology,
                        titles=title_extents(row, sidecar) if args.title_policy else None,
                        expressible=expressible.get(model) if args.title_policy else None,
                    )
                    tp = plot._maximum_matches(expected, actual, lambda a, b: plot.compatible(a, b, coverage))
                    triple = [tp, len(actual), len(expected)]
                    lang = row.get("scoring_language", row["lang"])
                    totals[lang] = [a + b for a, b in zip(totals[lang], triple, strict=True)]
                    bucket = supplement if row["evaluation_source"] == "ont3" else human
                    for i, value in enumerate(triple):
                        bucket[i] += value
                    per_input.append(
                        dict(
                            id=row["id"],
                            counts=triple,
                            neutral_predictions=ignored,
                            masked_predictions=masked,
                        )
                    )
                scored.append(
                    dict(
                        threshold=float(threshold),
                        per_input=per_input,
                        panels={
                            **{k: plot.metrics(v) for k, v in totals.items()},
                            "overall": plot.metrics(human),
                            "ont3_supplement": plot.metrics(supplement),
                        },
                    )
                )
            evidence["scores"].setdefault(str(coverage), {})[model] = scored
    if args.title_policy:
        evidence["expressible"] = {model: sorted(types) for model, types in expressible.items()}
    (args.score_output or HERE / "scores.json").write_text(json.dumps(evidence, separators=(",", ":")) + "\n")
    if not args.no_render:
        plot.render(evidence, "human-gold-priority9-five-model")


if __name__ == "__main__":
    main()
