"""Deterministic disjoint coverage strata for ontology-v3 training.

Each required primary type and predicate channel reserves one distinct training
window.  A sampling configuration can then assign a small, explicit mass to
every reserved row while leaving all other predicate rows in language-balanced
background strata.  Distinct reservation prevents one unusually rich row from
silently satisfying several nominal sampling pools.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

SCHEMA = "pii-predicate-exposure-strata"
SCHEMA_VERSION = 1


def _nonempty_unique_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"predicate exposure {field} must be a non-empty list")
    strings = tuple(value)
    if any(not isinstance(item, str) or not item for item in strings):
        raise ValueError(f"predicate exposure {field} must contain non-empty strings")
    if len(set(strings)) != len(strings):
        raise ValueError(f"predicate exposure {field} must not contain duplicates")
    return strings


def _canonical_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_row_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_row_sha256(
        {key: value for key, value in row.items() if not key.startswith("sampling_")}
    )


def _row_identity(row: Mapping[str, Any], index: int) -> str:
    for field in ("seed_id", "id", "document_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return f"{field}:{value}"
    return f"row-sha256:{_semantic_row_sha256(row)}:{index}"


def _positive_predicates(row: Mapping[str, Any], allowed: frozenset[str]) -> frozenset[str]:
    positive: set[str] = set()
    predicate_spans = row.get("predicate_spans", [])
    if not isinstance(predicate_spans, list):
        raise ValueError("predicate_spans must be a list")
    for predicate_span in predicate_spans:
        if not isinstance(predicate_span, dict):
            raise ValueError("predicate_spans entries must be objects")
        attrs = predicate_span.get("attrs")
        if not isinstance(attrs, dict):
            raise ValueError("predicate_spans attrs must be an object")
        for channel, intervals in attrs.items():
            if channel not in allowed:
                continue
            if not isinstance(intervals, list):
                raise ValueError(f"predicate channel {channel!r} intervals must be a list")
            if intervals:
                positive.add(channel)
    return frozenset(positive)


def _positive_primary_types(row: Mapping[str, Any], allowed: frozenset[str]) -> frozenset[str]:
    spans = row.get("spans")
    if not isinstance(spans, list):
        raise ValueError("training row spans must be a list")
    return frozenset(
        span[2]
        for span in spans
        if isinstance(span, (list, tuple))
        and len(span) == 3
        and isinstance(span[2], str)
        and span[2] in allowed
    )


def _target_candidates(
    rows: Sequence[Mapping[str, Any]],
    candidate_indices: Sequence[int],
    primary_types: tuple[str, ...],
    predicate_channels: tuple[str, ...],
) -> tuple[dict[str, list[int]], dict[int, dict[str, frozenset[str]]]]:
    targets = {
        **{f"primary:{name}": [] for name in primary_types},
        **{f"predicate:{name}": [] for name in predicate_channels},
    }
    row_targets: dict[int, dict[str, frozenset[str]]] = {}
    primary_set = frozenset(primary_types)
    predicate_set = frozenset(predicate_channels)
    for index in candidate_indices:
        row = rows[index]
        primary = _positive_primary_types(row, primary_set)
        predicates = _positive_predicates(row, predicate_set)
        row_targets[index] = {"primary": primary, "predicate": predicates}
        for name in primary:
            targets[f"primary:{name}"].append(index)
        for name in predicates:
            targets[f"predicate:{name}"].append(index)
    for target, indices in targets.items():
        indices.sort(key=lambda index: (_row_identity(rows[index], index), index))
        if not indices:
            raise ValueError(f"predicate exposure target {target!r} has no positive candidate row")
    return targets, row_targets


def distinct_target_matching(candidates: Mapping[str, Sequence[int]]) -> dict[str, int]:
    """Return a deterministic maximum matching from targets to distinct rows."""
    row_owner: dict[int, str] = {}
    target_row: dict[str, int] = {}

    def augment(target: str, seen_targets: set[str], seen_rows: set[int]) -> bool:
        seen_targets.add(target)
        for row_index in candidates[target]:
            if row_index in seen_rows:
                continue
            seen_rows.add(row_index)
            owner = row_owner.get(row_index)
            if owner is None or (owner not in seen_targets and augment(owner, seen_targets, seen_rows)):
                row_owner[row_index] = target
                target_row[target] = row_index
                return True
        return False

    order = sorted(candidates, key=lambda target: (len(candidates[target]), target))
    for target in order:
        if not augment(target, set(), set()):
            matched = sorted(target_row)
            raise ValueError(
                "predicate exposure targets cannot be assigned distinct rows: "
                f"failed={target!r} matched={len(matched)}/{len(candidates)}"
            )
    return target_row


def assign_predicate_exposure_strata(
    rows: Sequence[MutableMapping[str, Any]],
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign one reserved row per required target and background to the rest.

    The operation is idempotent: a second compilation accepts exactly the same
    derived values and rejects partial or divergent pre-existing assignments.
    """
    if declaration.get("schema") != SCHEMA:
        raise ValueError(f"predicate exposure schema must be {SCHEMA!r}")
    if declaration.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"predicate exposure schema_version must be {SCHEMA_VERSION}")
    sampling_pool = declaration.get("sampling_pool")
    field = declaration.get("field")
    background = declaration.get("background")
    if not isinstance(sampling_pool, str) or not sampling_pool:
        raise ValueError("predicate exposure sampling_pool must be a non-empty string")
    if not isinstance(field, str) or not field.startswith("sampling_"):
        raise ValueError("predicate exposure field must be a sampling_* string")
    if not isinstance(background, str) or not background:
        raise ValueError("predicate exposure background must be a non-empty string")
    primary_types = _nonempty_unique_strings(declaration.get("primary_types"), field="primary_types")
    predicate_channels = _nonempty_unique_strings(
        declaration.get("predicate_channels"), field="predicate_channels"
    )

    candidate_indices = [index for index, row in enumerate(rows) if row.get("sampling_pool") == sampling_pool]
    if not candidate_indices:
        raise ValueError(f"predicate exposure sampling_pool {sampling_pool!r} has no rows")
    outsiders = [
        index for index, row in enumerate(rows) if row.get("sampling_pool") != sampling_pool and field in row
    ]
    if outsiders:
        raise ValueError(
            f"predicate exposure field {field!r} appears outside {sampling_pool!r} at row {outsiders[0]}"
        )

    candidates, row_targets = _target_candidates(
        rows,
        candidate_indices,
        primary_types,
        predicate_channels,
    )
    target_rows = distinct_target_matching(candidates)
    row_target = {row_index: target for target, row_index in target_rows.items()}
    assignments = {index: row_target.get(index, background) for index in candidate_indices}
    existing = [index for index in candidate_indices if field in rows[index]]
    if existing and len(existing) != len(candidate_indices):
        raise ValueError(f"predicate exposure field {field!r} is only partially assigned")
    for index, value in assignments.items():
        if field in rows[index] and rows[index][field] != value:
            raise ValueError(
                f"predicate exposure field {field!r} differs at row {index}: "
                f"stored={rows[index][field]!r} derived={value!r}"
            )
        rows[index][field] = value

    reserved = {}
    for target in sorted(target_rows):
        index = target_rows[target]
        row = rows[index]
        reserved[target] = {
            "row_identity": _row_identity(row, index),
            "row_sha256": _canonical_row_sha256(row),
            "language": row.get("lang") or row.get("language"),
            "positive_primary_types": sorted(row_targets[index]["primary"]),
            "positive_predicates": sorted(row_targets[index]["predicate"]),
        }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sampling_pool": sampling_pool,
        "field": field,
        "background": background,
        "candidate_rows": len(candidate_indices),
        "reserved_rows": len(target_rows),
        "background_rows": len(candidate_indices) - len(target_rows),
        "primary_targets": list(primary_types),
        "predicate_targets": list(predicate_channels),
        "targets": reserved,
    }
