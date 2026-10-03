"""Derive flat primary targets while preserving reference-overlap provenance.

The input row uses character triples in ``spans``. Raw annotations stay with
their source; only the returned view changes. Conflicts are decided against
the complete deduplicated input, never a progressively shortened list.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

REFERENCE_TYPES = frozenset({"person_reference", "organization_reference"})
REFERENCE_PROJECTION_POLICY = "reference-overlap-flat-v1"
PROJECTION_FIELD = "reference_overlap_projection"
OPTIONAL_REFERENCE_POLICY = "optional-reference-neutral-exact-v1"
REFERENCE_BASE_TYPES = {"person_reference": "person_name", "organization_reference": "organization"}


def optional_reference_spans(gold, predicted):
    """Exclude references and exact same-family base predictions, without credit.

    Wrong families and shifted boundaries remain errors. Required gold wins
    over an overlapping optional carrier. This changes scoring, never decoding.
    """
    required = {span for span in gold if span[2] not in REFERENCE_TYPES}
    neutral = {
        (start, end, REFERENCE_BASE_TYPES[label]) for start, end, label in gold if label in REFERENCE_TYPES
    } - required
    primary = {span for span in predicted if span[2] not in REFERENCE_TYPES}
    ignored = primary & neutral
    return required, primary - ignored, ignored


class NonReferenceOverlap(ValueError):
    """The flat view cannot preserve all non-reference primary targets."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def project_reference_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop conflicting references; reject non-reference conflicts and bad weights.

    Exact typed duplicates collapse only when their objective weights agree.
    Receipts use source-row character coordinates and survive idempotent replay.
    """
    text = row["text"]
    if not isinstance(text, str):
        raise ValueError("reference projection requires text")
    spans = []
    for span in row["spans"]:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            raise ValueError(f"invalid primary span: {span!r}")
        start, end, label = span
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or not isinstance(label, str)
            or not label
        ):
            raise ValueError(f"invalid primary span: {span!r}")
        spans.append((start, end, label))
    weights = row.get("primary_span_objective_weights")
    if weights is not None and (
        not isinstance(weights, list)
        or len(weights) != len(spans)
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights
        )
    ):
        raise ValueError("primary span objective weights must be finite, nonnegative and align with spans")
    unique: dict[tuple[int, int, str], int] = {}
    removed = []
    for index, span in enumerate(spans):
        if span in unique:
            if weights is not None and weights[index] != weights[unique[span]]:
                raise ValueError(f"duplicate primary span has conflicting objective weights: {span}")
            removed.append({"index": index, "span": list(span), "reason": "duplicate"})
        else:
            unique[span] = index
    excluded = set()
    for span, index in unique.items():
        if span[2] not in REFERENCE_TYPES:
            continue
        conflicts = [other for other in unique if other != span and span[0] < other[1] and other[0] < span[1]]
        if conflicts:
            excluded.add(span)
            removed.append(
                {
                    "index": index,
                    "span": list(span),
                    "reason": "reference_overlap",
                    "conflicts": [list(other) for other in sorted(conflicts)],
                }
            )
    retained = sorted(set(unique) - excluded)
    for previous, current in zip(retained, retained[1:]):
        if current[0] < previous[1]:
            raise NonReferenceOverlap(f"overlapping non-reference primary spans: {previous}, {current}")
    if not removed:
        return row
    kept = [unique[span] for span in retained]
    projected = {**row, "spans": [row["spans"][index] for index in kept]}
    if weights is not None:
        projected["primary_span_objective_weights"] = [weights[index] for index in kept]
    for field, start_key, end_key in (
        ("predicate_spans", "start", "end"),
        ("subclass_spans", "carrier_start", "carrier_end"),
        ("reference_form_spans", "carrier_start", "carrier_end"),
    ):
        if field in row:
            projected[field] = [
                carrier
                for carrier in row[field]
                if (carrier[start_key], carrier[end_key], carrier["type"]) not in excluded
            ]
    projected[PROJECTION_FIELD] = {
        "policy": REFERENCE_PROJECTION_POLICY,
        "coordinate_space": "source_row",
        "source_row_id": row.get("id"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "input_spans_sha256": _digest(row["spans"]),
        "output_spans_sha256": _digest(projected["spans"]),
        "spans_before": len(spans),
        "spans_after": len(kept),
        "removed": sorted(removed, key=lambda item: item["index"]),
    }
    if PROJECTION_FIELD in row:
        projected[PROJECTION_FIELD]["parent"] = row[PROJECTION_FIELD]
    return projected


def excluded_reference_targets(row: dict[str, Any]) -> set[tuple[int, int, str]]:
    """Read exclusions from a projected row, including a preceding projection."""
    receipt = row.get(PROJECTION_FIELD)
    excluded = set()
    if receipt is not None and "spans" in row and receipt["output_spans_sha256"] != _digest(row["spans"]):
        raise ValueError("reference projection receipt does not match output spans")
    while receipt is not None:
        if receipt["policy"] != REFERENCE_PROJECTION_POLICY:
            raise ValueError(f"unsupported reference projection policy: {receipt['policy']}")
        if receipt["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("reference projection receipt does not match source text")
        excluded.update(
            tuple(item["span"]) for item in receipt["removed"] if item["reason"] == "reference_overlap"
        )
        receipt = receipt.get("parent")
    if any(tuple(span) in excluded for span in row.get("spans", [])):
        raise ValueError("an excluded reference target is still present in the projected row")
    return excluded
