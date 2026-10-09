#!/usr/bin/env python
"""Source-level multilingual scoring for calibration trajectory selection."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pii_gliner2_exact import canonical_p9_spans  # noqa: E402
from pii_projector import Tagset  # noqa: E402

SCORE_VERSION = "pii-calibration-source-score-v1"
P9_LANGUAGES = ("ar", "de", "en", "es", "fr", "ko", "pt", "vi", "zh")
METRIC_VIEWS = ("exact_typed_p9", "overlap_typed_p9", "overlap_p1", "character_p1")
_TAGSET = Tagset()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_language_weights(path: Path) -> dict[str, float]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    languages = [item["code"] for item in spec["languages"]]
    configured = spec["language_importance"]["weights"]
    default = float(spec["language_importance"]["unlisted_language_weight"])
    weights = {language: float(configured.get(language, default)) for language in languages}
    if weights.get("en") != 4 or any(
        weights.get(language) != 2 for language in P9_LANGUAGES if language != "en"
    ):
        raise ValueError(f"language importance is not the frozen 4:2:1 vector: {weights}")
    if any(weights[language] != 1 for language in set(languages) - set(P9_LANGUAGES)):
        raise ValueError(f"language importance is not the frozen 4:2:1 vector: {weights}")
    return weights


def project_prediction_label(label: str, schema: str) -> str:
    return _TAGSET.project_cut(schema, label, "redaction_9_v1")


def canonical_source_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(
            {
                "id": str(row["id"]),
                "lang": str(row["lang"]),
                "text": str(row["text"]),
                "gold": [
                    {"start": int(start), "end": int(end), "label": str(label)}
                    for start, end, label in canonical_p9_spans(row)
                ],
            }
        )
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("source rows contain duplicate IDs")
    return result


def aggregate_window_candidates(
    source_rows: Sequence[dict[str, Any]],
    window_rows: Sequence[dict[str, Any]],
    window_predictions: Sequence[Sequence[dict[str, Any]]],
    *,
    prediction_schema: str = "redaction_9_v1",
    boundary_receipt: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if len(window_rows) != len(window_predictions):
        raise ValueError("window rows and predictions must align one-to-one")
    sources = {str(row["id"]): row for row in source_rows}
    best: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = {source_id: {} for source_id in sources}
    for row, predictions in zip(window_rows, window_predictions):
        provenance = row["provenance"]
        source_id = str(provenance["source_id"])
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"window refers to unknown source {source_id!r}")
        offset = int(provenance["window_start"])
        for prediction in predictions:
            local_start = int(prediction["start"])
            local_end = int(prediction["end"])
            window_text = row.get("input")
            if (
                isinstance(window_text, str)
                and not window_text.endswith((".", "!", "?"))
                and local_end == len(window_text) + 1
            ):
                normalized_end = len(window_text)
                if boundary_receipt is not None:
                    boundary_receipt.append(
                        {
                            "source_id": source_id,
                            "window_index": int(provenance.get("window_index", -1)),
                            "label": str(prediction["label"]),
                            "original": [local_start, local_end],
                            "normalized": (
                                [local_start, normalized_end] if local_start < normalized_end else None
                            ),
                            "reason": "trim_gliner2_synthetic_terminal_period",
                        }
                    )
                local_end = normalized_end
                if local_start >= local_end:
                    continue
            if isinstance(window_text, str) and not (0 <= local_start < local_end <= len(window_text)):
                raise ValueError(
                    f"prediction [{local_start}, {local_end}) is outside window "
                    f"{provenance.get('window_index', '?')} for source {source_id}"
                )
            start = local_start + offset
            end = local_end + offset
            if not 0 <= start < end <= len(source["text"]):
                raise ValueError(f"prediction [{start}, {end}) is outside source {source_id}")
            label = project_prediction_label(str(prediction["label"]), prediction_schema)
            candidate = {
                "start": start,
                "end": end,
                "label": label,
                "confidence": float(prediction.get("confidence", 1.0)),
            }
            if not math.isfinite(candidate["confidence"]):
                raise ValueError(f"non-finite candidate confidence for {source_id}")
            key = (start, end, label)
            prior = best[source_id].get(key)
            if prior is None or candidate["confidence"] > prior["confidence"]:
                best[source_id][key] = candidate
    return {
        source_id: sorted(values.values(), key=lambda item: (item["start"], item["end"], item["label"]))
        for source_id, values in best.items()
    }


def symmetric_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    intersection = min(left["end"], right["end"]) - max(left["start"], right["start"])
    if intersection <= 0:
        return False
    return (
        intersection * 5 >= (left["end"] - left["start"]) * 4
        and intersection * 5 >= (right["end"] - right["start"]) * 4
    )


def maximum_matches(
    gold: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    compatible: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> int:
    edges = [
        [index for index, prediction in enumerate(predictions) if compatible(item, prediction)]
        for item in gold
    ]
    prediction_to_gold: dict[int, int] = {}

    def augment(gold_index: int, seen: set[int]) -> bool:
        for prediction_index in edges[gold_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            if prediction_index not in prediction_to_gold or augment(
                prediction_to_gold[prediction_index], seen
            ):
                prediction_to_gold[prediction_index] = gold_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(gold)))


def _span_counts(
    gold: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    compatible: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> Counter[str]:
    true_positives = maximum_matches(gold, predictions, compatible)
    return Counter(tp=true_positives, predicted=len(predictions), gold=len(gold))


def _character_counts(
    gold: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], text_length: int
) -> Counter[str]:
    gold_mask = bytearray(text_length)
    predicted_mask = bytearray(text_length)
    for item in gold:
        gold_mask[item["start"] : item["end"]] = b"\x01" * (item["end"] - item["start"])
    for item in predictions:
        predicted_mask[item["start"] : item["end"]] = b"\x01" * (item["end"] - item["start"])
    counts: Counter[str] = Counter()
    for expected, observed in zip(gold_mask, predicted_mask):
        if expected and observed:
            counts["tp"] += 1
        elif observed:
            counts["fp"] += 1
        elif expected:
            counts["fn"] += 1
    counts["predicted"] = counts["tp"] + counts["fp"]
    counts["gold"] = counts["tp"] + counts["fn"]
    return counts


def prf(counts: Counter[str]) -> dict[str, float | int]:
    true_positives = int(counts["tp"])
    predicted = int(counts["predicted"])
    gold = int(counts["gold"])
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positives,
        "predicted": predicted,
        "gold": gold,
        "P": precision,
        "R": recall,
        "F1": f1,
    }


def _weighted_macro(
    per_language: dict[str, dict[str, dict[str, float | int]]],
    languages: Iterable[str],
    weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    selected = list(languages)
    mass = sum(weights[language] for language in selected)
    if not selected or mass <= 0:
        raise ValueError("weighted macro requires a nonempty positive-weight language slice")
    return {
        view: {
            metric: sum(
                weights[language] * float(per_language[language][view][metric]) for language in selected
            )
            / mass
            for metric in ("P", "R", "F1")
        }
        for view in METRIC_VIEWS
    }


def score_sources(
    source_rows: Sequence[dict[str, Any]],
    candidates: dict[str, Sequence[dict[str, Any]]],
    *,
    threshold: float,
    language_weights: dict[str, float],
) -> dict[str, Any]:
    document_counts = score_document_counts(source_rows, candidates, threshold=threshold)
    return aggregate_document_counts(
        document_counts,
        threshold=threshold,
        language_weights=language_weights,
    )


def aggregate_document_counts(
    document_counts: Sequence[dict[str, Any]],
    *,
    threshold: float,
    language_weights: dict[str, float],
) -> dict[str, Any]:
    """Aggregate saved integer document statistics without rescoring spans."""
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {view: Counter() for view in METRIC_VIEWS}
    )
    for row in document_counts:
        for view in METRIC_VIEWS:
            counts[row["lang"]][view].update(row["counts"][view])
    per_language = {
        language: {view: prf(view_counts) for view, view_counts in views.items()}
        for language, views in sorted(counts.items())
    }
    languages = sorted(per_language)
    missing = set(languages) - set(language_weights)
    if missing:
        raise ValueError(f"score languages lack frozen weights: {sorted(missing)}")
    p9 = [language for language in P9_LANGUAGES if language in per_language]
    unit_weights = {language: 1.0 for language in languages}
    result = {
        "version": SCORE_VERSION,
        "threshold": threshold,
        "documents": len(document_counts),
        "per_language": per_language,
        "unweighted_language_macro": _weighted_macro(per_language, languages, unit_weights),
        "importance_weighted_language_macro": _weighted_macro(per_language, languages, language_weights),
        "priority9_unweighted_language_macro": None,
        "priority9_importance_weighted_language_macro": None,
    }
    if p9:
        result["priority9_unweighted_language_macro"] = _weighted_macro(
            per_language, p9, unit_weights
        )
        result["priority9_importance_weighted_language_macro"] = _weighted_macro(
            per_language, p9, language_weights
        )
    return result


def score_document_counts(
    source_rows: Sequence[dict[str, Any]],
    candidates: dict[str, Sequence[dict[str, Any]]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Return integer sufficient statistics for paired document bootstrap."""
    sources = canonical_source_rows(source_rows)
    result = []
    for source in sources:
        predicted_typed = {
            (int(item["start"]), int(item["end"]), str(item["label"]))
            for item in candidates.get(source["id"], [])
            if float(item["confidence"]) >= threshold
        }
        typed = [
            {"start": start, "end": end, "label": label} for start, end, label in sorted(predicted_typed)
        ]
        p1 = [
            {"start": start, "end": end}
            for start, end in sorted({(item["start"], item["end"]) for item in typed})
        ]
        gold = source["gold"]
        view_counts = {
            "exact_typed_p9": _span_counts(
                gold,
                typed,
                lambda left, right: (
                    (left["start"], left["end"], left["label"])
                    == (right["start"], right["end"], right["label"])
                ),
            ),
            "overlap_typed_p9": _span_counts(
                gold,
                typed,
                lambda left, right: left["label"] == right["label"] and symmetric_overlap(left, right),
            ),
            "overlap_p1": _span_counts(gold, p1, symmetric_overlap),
            "character_p1": _character_counts(gold, p1, len(source["text"])),
        }
        result.append(
            {
                "id": source["id"],
                "lang": source["lang"],
                "counts": {
                    view: {
                        "tp": int(counts["tp"]),
                        "predicted": int(counts["predicted"]),
                        "gold": int(counts["gold"]),
                    }
                    for view, counts in view_counts.items()
                },
            }
        )
    return result


def threshold_curve(
    source_rows: Sequence[dict[str, Any]],
    candidates: dict[str, Sequence[dict[str, Any]]],
    *,
    thresholds: Sequence[float],
    language_weights: dict[str, float],
) -> list[dict[str, Any]]:
    if list(thresholds) != sorted(set(thresholds)):
        raise ValueError("thresholds must be unique and ascending")
    return [
        score_sources(
            source_rows,
            candidates,
            threshold=float(threshold),
            language_weights=language_weights,
        )
        for threshold in thresholds
    ]


def select_curve_point(curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose checkpoint evidence without treating its threshold as deployment calibration."""
    if not curve:
        raise ValueError("cannot select from an empty curve")
    return max(
        curve,
        key=lambda point: (
            point["importance_weighted_language_macro"]["overlap_p1"]["F1"],
            point["importance_weighted_language_macro"]["overlap_typed_p9"]["F1"],
            point["importance_weighted_language_macro"]["character_p1"]["F1"],
            -float(point["threshold"]),
        ),
    )
