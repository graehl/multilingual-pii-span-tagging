#!/usr/bin/env python3
"""Frozen-threshold ont3 reference and token-predicate evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from scripts.pii_bcp47 import canonical_bcp47_tag
    from scripts.pii_document_context import encode_document_context, resolve_context_side
    from scripts.pii_reference_projection import (
        OPTIONAL_REFERENCE_POLICY,
        PROJECTION_FIELD,
        REFERENCE_PROJECTION_POLICY,
        excluded_reference_targets,
        optional_reference_spans,
        project_reference_row,
    )
    from scripts.pii_subclass import (
        SubclassSpec,
        decode_subclass_logits,
        load_subclass_spec,
        permissive_name_components,
    )
else:
    from pii_bcp47 import canonical_bcp47_tag
    from pii_document_context import encode_document_context, resolve_context_side
    from pii_reference_projection import (
        OPTIONAL_REFERENCE_POLICY,
        PROJECTION_FIELD,
        REFERENCE_PROJECTION_POLICY,
        excluded_reference_targets,
        optional_reference_spans,
        project_reference_row,
    )
    from pii_subclass import (
        SubclassSpec,
        decode_subclass_logits,
        load_subclass_spec,
        permissive_name_components,
    )

REFERENCE_TYPES = ("organization_reference", "person_reference")
REFERENCE_FORM_FAMILY = "reference_form"
ROUTINE_EXCLUDED_REFERENCE_FORMS = frozenset({"bare_pronoun"})
PREDICATE_THRESHOLD = 0.5


def _normalize_reference_logit_biases(
    value: dict[str, float] | None,
) -> dict[str, float]:
    biases = {reference_type: 0.0 for reference_type in REFERENCE_TYPES}
    if value is None:
        return biases
    unknown = sorted(set(value) - set(REFERENCE_TYPES))
    if unknown:
        raise ValueError(f"unknown reference-logit bias types: {unknown}")
    for reference_type, bias in value.items():
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
            raise ValueError(f"{reference_type} reference-logit bias must be finite")
        biases[reference_type] = float(bias)
    return biases


def _parse_reference_logit_bias(value: str) -> tuple[str, float]:
    reference_type, separator, raw_bias = value.partition("=")
    if separator != "=" or reference_type not in REFERENCE_TYPES:
        raise argparse.ArgumentTypeError(
            "reference-logit bias must be TYPE=FLOAT where TYPE is one of " + ", ".join(REFERENCE_TYPES)
        )
    try:
        bias = float(raw_bias)
    except ValueError as error:
        raise argparse.ArgumentTypeError("reference-logit bias must be finite") from error
    if not math.isfinite(bias):
        raise argparse.ArgumentTypeError("reference-logit bias must be finite")
    return reference_type, bias


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise ValueError(f"{path}: every row needs a nonempty string id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return rows


def _load_id_list(path: str | Path) -> list[str]:
    ids = Path(path).read_text(encoding="utf-8").splitlines()
    if not ids or any(not row_id for row_id in ids):
        raise ValueError(f"{path}: id list must contain nonempty lines")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return ids


def _select_rows_by_ids(
    rows: list[dict[str, Any]],
    ids: list[str],
    *,
    source: str | Path,
) -> list[dict[str, Any]]:
    selected = set(ids)
    available = {row["id"] for row in rows}
    missing = selected - available
    if missing:
        raise ValueError(f"{source}: missing selected ids {sorted(missing)[:3]}")
    return [row for row in rows if row["id"] in selected]


def _load_bcp47_sidecar(path: str | Path, expected_rows: int) -> list[str]:
    tags = Path(path).read_text(encoding="utf-8").splitlines()
    if len(tags) != expected_rows:
        raise ValueError(f"{path}: BCP 47 sidecar has {len(tags)} rows; expected {expected_rows}")
    result = []
    for line_number, tag in enumerate(tags, start=1):
        try:
            result.append(canonical_bcp47_tag(tag))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return result


def _load_predicate_spec(path: str | Path) -> tuple[list[str], dict[str, frozenset[str]]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    channels = spec.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{path}: predicate spec has no channels")
    names = [channel.get("name") for channel in channels]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{path}: predicate channel names must be nonempty and unique")
    applicable = {}
    for channel in channels:
        types = channel.get("applicable_types")
        if not isinstance(types, list) or not types or any(not isinstance(value, str) for value in types):
            raise ValueError(f"{path}: {channel['name']} has invalid applicable_types")
        applicable[channel["name"]] = frozenset(types)
    return names, applicable


def _span_tuple(span: Any, label_key: str) -> tuple[int, int, str]:
    if isinstance(span, dict):
        start = span.get("start")
        end = span.get("end")
        label = span.get(label_key)
    elif isinstance(span, (list, tuple)) and len(span) == 3:
        start, end, label = span
    else:
        raise ValueError(f"invalid span shape: {span}")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise ValueError(f"invalid span bounds: {span}")
    if not isinstance(label, str) or not label:
        raise ValueError(f"invalid span label: {span}")
    return start, end, label


def gold_views(
    row: dict[str, Any],
    predicate_names: Iterable[str],
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    """Split reviewed gold into reference-aware, named-only, and predicate spans."""
    reference_aware, named_only, predicates, _projected = _gold_view_data(row, predicate_names)
    return reference_aware, named_only, predicates


def _gold_view_data(
    row: dict[str, Any], predicate_names: Iterable[str]
) -> tuple[list, list, list, dict[str, Any]]:
    """Build the scored view and retain its projected carrier data and receipt."""
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{row.get('id')}: missing text")
    predicates = frozenset(predicate_names)
    if "spans" in row:
        row = project_reference_row({**row, "spans": [_span_tuple(span, "type") for span in row["spans"]]})
        reference_aware = row["spans"]
        named_only = [span for span in reference_aware if span[2] not in REFERENCE_TYPES]
        predicate_spans = []
        for carrier in row.get("predicate_spans", []):
            attrs = carrier.get("attrs")
            if not isinstance(attrs, dict):
                raise ValueError(f"{row['id']}: predicate carrier lacks attrs")
            unknown = sorted(set(attrs) - predicates)
            if unknown:
                raise ValueError(f"{row['id']}: unknown predicate labels {unknown}")
            for label, regions in attrs.items():
                if not isinstance(regions, list):
                    raise ValueError(f"{row['id']}: invalid {label} predicate regions")
                for region in regions:
                    if not isinstance(region, (list, tuple)) or len(region) != 2:
                        raise ValueError(f"{row['id']}: invalid {label} predicate region")
                    predicate_spans.append((region[0], region[1], label))
    else:
        base = [_span_tuple(span, "type") for span in row.get("base_spans", [])]
        successor = [_span_tuple(span, "label") for span in row.get("preds", [])]
        unknown = sorted({label for _, _, label in successor} - predicates - set(REFERENCE_TYPES))
        if unknown:
            raise ValueError(f"{row['id']}: unknown successor labels {unknown}")
        references = [span for span in successor if span[2] in REFERENCE_TYPES]
        predicate_spans = [span for span in successor if span[2] in predicates]
        reference_bounds = {(start, end) for start, end, _ in references}
        reference_aware = [span for span in base if span[:2] not in reference_bounds] + references
        named_only = base
        projected = project_reference_row({**row, "spans": reference_aware})
        reference_aware = projected.pop("spans")
        if excluded_reference_targets(projected):
            # Legacy predicates have no explicit carrier field. Retain an
            # interval only when a surviving primary can still activate it;
            # score_rows additionally checks channel/type applicability.
            predicate_spans = [
                span
                for span in predicate_spans
                if any(start <= span[0] and span[1] <= end for start, end, _ in reference_aware)
            ]
        row = projected
    for family_name, spans in (
        ("named-only", named_only),
        ("reference-aware", reference_aware),
        ("predicate", predicate_spans),
    ):
        if len(set(spans)) != len(spans):
            raise ValueError(f"{row['id']}: duplicate {family_name} spans")
        for start, end, _label in spans:
            if end > len(text):
                raise ValueError(f"{row['id']}: {family_name} span outside text")
    return sorted(reference_aware), sorted(named_only), sorted(predicate_spans), row


def _known_predicate_carriers(
    row: dict[str, Any],
    channel: str,
    compatible_carriers: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Return carriers with an explicit known cell for one Bernoulli channel."""
    if "spans" not in row:
        return compatible_carriers
    compatible = set(compatible_carriers)
    known = set()
    for item in row.get("predicate_spans", []):
        carrier = _span_tuple(item, "type")
        if carrier not in compatible:
            continue
        weights = item.get("objective_weights")
        if not isinstance(weights, dict):
            raise ValueError(f"{row['id']}: predicate carrier lacks objective_weights")
        weight = weights.get(channel, 0.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"{row['id']}: invalid {channel} objective weight")
        if weight > 0:
            known.add(carrier)
    return sorted(known)


def _subclass_tuple(
    item: dict[str, Any],
    spec: SubclassSpec,
) -> tuple[int, int, str, int, int, str, str]:
    carrier_start = item.get("carrier_start")
    carrier_end = item.get("carrier_end")
    primary_type = item.get("type")
    start = item.get("start")
    end = item.get("end")
    family_name = item.get("family")
    value = item.get("value")
    family = spec.family_by_name.get(family_name)
    if (
        not isinstance(carrier_start, int)
        or not isinstance(carrier_end, int)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= carrier_start <= start < end <= carrier_end
        or not isinstance(primary_type, str)
        or family is None
        or primary_type not in family.applicable_types
        or value not in family.outcomes
    ):
        raise ValueError(f"invalid subclass span: {item}")
    if family.scope == "full_primary_span" and (start, end) != (carrier_start, carrier_end):
        raise ValueError(f"partial full-span subclass value: {item}")
    return carrier_start, carrier_end, primary_type, start, end, family_name, value


def _subclass_spans(
    row: dict[str, Any],
    field: str,
    spec: SubclassSpec,
    *,
    missing_is_unknown: bool = False,
) -> set[tuple[int, int, str, int, int, str, str]]:
    items = row.get(field)
    if items is None and missing_is_unknown:
        return set()
    if not isinstance(items, list):
        raise ValueError(f"{row.get('id')}: missing {field}")
    spans = {_subclass_tuple(item, spec) for item in items}
    if len(spans) != len(items):
        raise ValueError(f"{row.get('id')}: duplicate {field} spans")
    return spans


def _reviewed_reference_forms(
    row: dict[str, Any],
    reference_spans: Iterable[tuple[int, int, str]],
    spec: SubclassSpec | None,
) -> dict[tuple[int, int, str], str]:
    """Require one reviewed reference-form value for every scored reference carrier."""
    if spec is None or REFERENCE_FORM_FAMILY not in spec.sidecar_family_names:
        raise ValueError("routine reference projection requires a reference_form sidecar family")
    expected = {span for span in reference_spans if span[2] in REFERENCE_TYPES}
    raw_items = row.get("reference_form_spans", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"{row.get('id')}: reference_form_spans must be a list")
    forms = [_subclass_tuple(item, spec) for item in raw_items]
    if any(item[-2] != REFERENCE_FORM_FAMILY for item in forms):
        raise ValueError(f"{row.get('id')}: reference_form_spans contains another family")
    carriers = [(item[0], item[1], item[2]) for item in forms]
    if len(carriers) != len(set(carriers)):
        raise ValueError(f"{row.get('id')}: duplicate reference_form carrier")
    observed = set(carriers)
    if observed != expected:
        raise ValueError(
            f"{row.get('id')}: incomplete reference_form coverage; "
            f"missing={sorted(expected - observed)!r}, extra={sorted(observed - expected)!r}"
        )
    return {carrier: item[-1] for carrier, item in zip(carriers, forms, strict=True)}


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gold: set[Any], predicted: set[Any]) -> None:
        self.tp += len(gold & predicted)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)

    def report(self) -> dict[str, int | float]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "gold": self.tp + self.fn,
            "predicted": self.tp + self.fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _prediction_spans(row: dict[str, Any], field: str) -> set[tuple[int, int, str]]:
    spans = {_span_tuple(span, "label") for span in row.get(field, [])}
    if len(spans) != len(row.get(field, [])):
        raise ValueError(f"{row.get('id')}: duplicate {field} spans")
    return spans


def _predicate_tokens(row: dict[str, Any], predicate_names: frozenset[str]) -> list[dict[str, Any]]:
    tokens = row.get("predicate_tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"{row.get('id')}: missing predicate_tokens")
    normalized = []
    seen = set()
    for token in tokens:
        start = token.get("start")
        end = token.get("end")
        token_id = token.get("token_id")
        active = token.get("active")
        key = (start, end, token_id)
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end
            or not isinstance(token_id, int)
            or not isinstance(active, list)
            or any(channel not in predicate_names for channel in active)
            or len(set(active)) != len(active)
        ):
            raise ValueError(f"{row.get('id')}: invalid predicate token {token}")
        if key in seen:
            raise ValueError(f"{row.get('id')}: duplicate predicate token {key}")
        seen.add(key)
        normalized.append({"start": start, "end": end, "token_id": token_id, "active": frozenset(active)})
    return sorted(normalized, key=lambda token: (token["start"], token["end"], token["token_id"]))


def _active_intervals(
    text: str,
    tokens: list[dict[str, Any]],
    channel: str,
    carrier: tuple[int, int, str],
) -> list[tuple[int, int, str]]:
    start, end, _primary_type = carrier
    carrier_tokens = [token for token in tokens if _overlaps((token["start"], token["end"]), (start, end))]
    intervals = []
    open_start = None
    open_end = None
    previous_was_active = False
    for token in carrier_tokens:
        token_start = max(start, token["start"])
        token_end = min(end, token["end"])
        active = channel in token["active"]
        if active and open_start is None:
            open_start, open_end = token_start, token_end
        elif active and previous_was_active and text[open_end:token_start].isspace():
            open_end = token_end
        elif active and previous_was_active and token_start <= open_end:
            open_end = max(open_end, token_end)
        elif active:
            intervals.append((open_start, open_end, channel))
            open_start, open_end = token_start, token_end
        elif open_start is not None:
            intervals.append((open_start, open_end, channel))
            open_start = open_end = None
        previous_was_active = active
    if open_start is not None:
        intervals.append((open_start, open_end, channel))
    return intervals


def _permissive_name_items(text: str, spans: set[tuple]) -> set[tuple]:
    carriers = defaultdict(list)
    for a, b, primary, start, end, family, value in spans:
        if family == "name_component":
            carriers[a, b, primary].append((start, end, value))
    return {
        (*carrier, start, end, "name_component", value)
        for carrier, components in carriers.items()
        for start, end, value in permissive_name_components(text, components)
    }


def score_rows(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    predicate_names: list[str],
    applicable_types: dict[str, frozenset[str]],
    subclass_spec: SubclassSpec | None = None,
    *,
    routine_reference_projection: bool = False,
) -> dict[str, Any]:
    """Score fixed-threshold outputs against complete reviewed successor gold."""
    gold_by_id = {row["id"]: row for row in gold_rows}
    prediction_by_id = {row["id"]: row for row in prediction_rows}
    if set(gold_by_id) != set(prediction_by_id):
        raise ValueError(
            "prediction/gold identity mismatch: "
            f"missing={sorted(set(gold_by_id) - set(prediction_by_id))[:3]}, "
            f"extra={sorted(set(prediction_by_id) - set(gold_by_id))[:3]}"
        )
    predicate_set = frozenset(predicate_names)
    reference_penalties = {row.get("reference_logit_penalty", 0.0) for row in prediction_rows}
    if len(reference_penalties) != 1 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in reference_penalties
    ):
        raise ValueError("predictions must declare one finite nonnegative reference-logit penalty")
    reference_logit_penalty = float(next(iter(reference_penalties)))
    reference_bias_configs = {
        tuple(sorted(_normalize_reference_logit_biases(row.get("reference_logit_biases")).items()))
        for row in prediction_rows
    }
    if len(reference_bias_configs) != 1:
        raise ValueError("predictions must declare one reference-logit bias configuration")
    reference_logit_biases = dict(next(iter(reference_bias_configs)))
    prepared_gold = {}
    projected_gold = {}
    excluded_reference_forms = Counter()
    for row_id, row in gold_by_id.items():
        reference_aware, named_only, predicate_spans, row = _gold_view_data(row, predicate_names)
        if "spans" not in row and excluded_reference_targets(row):
            # Legacy predicate rows omit explicit carriers. A removed reference
            # cannot activate its channel through an incompatible retained name.
            predicate_spans = [
                span
                for span in predicate_spans
                if any(
                    label in applicable_types[span[2]] and start <= span[0] and span[1] <= end
                    for start, end, label in reference_aware
                )
            ]
        projected_gold[row_id] = row
        excluded_carriers = set()
        if routine_reference_projection:
            reference_forms = _reviewed_reference_forms(row, reference_aware, subclass_spec)
            excluded_carriers = {
                carrier
                for carrier, value in reference_forms.items()
                if value in ROUTINE_EXCLUDED_REFERENCE_FORMS
            }
            excluded_reference_forms.update(reference_forms[carrier] for carrier in excluded_carriers)
            excluded_bounds = {carrier[:2] for carrier in excluded_carriers}
            reference_aware = [span for span in reference_aware if span not in excluded_carriers]
            predicate_spans = [span for span in predicate_spans if span[:2] not in excluded_bounds]
        prepared_gold[row_id] = (
            reference_aware,
            named_only,
            predicate_spans,
            excluded_carriers,
        )
    supported_channels = sorted(
        channel
        for channel in predicate_names
        if any(
            _known_predicate_carriers(
                row,
                channel,
                [
                    carrier
                    for carrier in prepared_gold[row["id"]][0]
                    if carrier[2] in applicable_types[channel]
                ],
            )
            for row in projected_gold.values()
        )
    )

    name_permissive_end_to_end = Counts()
    name_permissive_oracle = Counts()
    primary_reference_aware = Counts()
    primary_named_only = Counts()
    primary_optional = Counts()
    neutral_named = neutral_optional = 0
    references = Counts()
    references_by_type = {reference_type: Counts() for reference_type in REFERENCE_TYPES}
    # A single micro F1 cannot say whether an arm closed a deficit on one type,
    # which is what the data-admission and surface-realization arms are aimed at.
    # Keyed by every label either side produces, so a type the model invents shows
    # up as its own row rather than vanishing into the aggregate.
    primary_by_type: dict[str, Counts] = defaultdict(Counts)
    predicate_cells = Counts()
    predicate_cells_by_channel = {channel: Counts() for channel in supported_channels}
    predicate_exact = Counts()
    predicate_exact_by_channel = {channel: Counts() for channel in supported_channels}
    known_predicate_cells = 0
    subclass_end_to_end = Counts()
    subclass_oracle_carrier = Counts()
    subclass_end_to_end_by_family = (
        {family.name: Counts() for family in subclass_spec.families} if subclass_spec else {}
    )
    subclass_oracle_by_family = (
        {family.name: Counts() for family in subclass_spec.families} if subclass_spec else {}
    )
    known_subclass_carrier_families = 0

    for row_id in [row["id"] for row in gold_rows]:
        gold = projected_gold[row_id]
        prediction = prediction_by_id[row_id]
        reference_aware, named_only, gold_predicates, excluded_carriers = prepared_gold[row_id]
        predicted_reference_aware = _prediction_spans(prediction, "reference_aware_preds")
        predicted_reference_aware -= excluded_reference_targets(gold)
        if excluded_carriers:
            excluded_bounds = [carrier[:2] for carrier in excluded_carriers]
            predicted_reference_aware = {
                span
                for span in predicted_reference_aware
                if span[2] not in REFERENCE_TYPES
                or not any(_overlaps(span[:2], bounds) for bounds in excluded_bounds)
            }
        predicted_named_only = _prediction_spans(prediction, "named_only_preds")
        gold_reference_set = set(reference_aware)
        gold_named_set = set(named_only)
        # Include references removed by either overlap or bare-pronoun policy:
        # their corresponding base labels remain optional, never negatives.
        optional_gold = (
            gold_reference_set
            | excluded_carriers
            | excluded_reference_targets(gold)
            | {tuple(span[:3]) for span in gold.get("deferred_reference_spans", [])}
        )
        _, predicted_named_only, ignored = optional_reference_spans(
            optional_gold | gold_named_set, predicted_named_only
        )
        neutral_named += len(ignored)
        optional_gold_set, optional_predicted, ignored = optional_reference_spans(
            optional_gold, predicted_reference_aware
        )
        neutral_optional += len(ignored)
        primary_optional.add(optional_gold_set, optional_predicted)
        primary_reference_aware.add(gold_reference_set, predicted_reference_aware)
        primary_named_only.add(gold_named_set, predicted_named_only)
        for label in {span[2] for span in gold_reference_set} | {
            span[2] for span in predicted_reference_aware
        }:
            primary_by_type[label].add(
                {span for span in gold_reference_set if span[2] == label},
                {span for span in predicted_reference_aware if span[2] == label},
            )
        reference_gold = {span for span in gold_reference_set if span[2] in REFERENCE_TYPES}
        reference_predicted = {span for span in predicted_reference_aware if span[2] in REFERENCE_TYPES}
        references.add(reference_gold, reference_predicted)
        for reference_type in REFERENCE_TYPES:
            references_by_type[reference_type].add(
                {span for span in reference_gold if span[2] == reference_type},
                {span for span in reference_predicted if span[2] == reference_type},
            )

        tokens = _predicate_tokens(prediction, predicate_set)
        text = gold["text"]
        if any(token["end"] > len(text) for token in tokens):
            raise ValueError(f"{row_id}: predicate token outside text")
        predicates_by_channel = {
            channel: [(start, end) for start, end, label in gold_predicates if label == channel]
            for channel in supported_channels
        }
        for channel in supported_channels:
            compatible_carriers = [
                carrier for carrier in reference_aware if carrier[2] in applicable_types[channel]
            ]
            compatible_carriers = _known_predicate_carriers(gold, channel, compatible_carriers)
            for gold_interval in predicates_by_channel[channel]:
                activators = [
                    carrier
                    for carrier in compatible_carriers
                    if carrier[0] <= gold_interval[0] and gold_interval[1] <= carrier[1]
                ]
                if not activators:
                    raise ValueError(
                        f"{row_id}: predicate {channel} interval {gold_interval} has no compatible activator"
                    )
            channel_gold_cells = set()
            channel_predicted_cells = set()
            for token_index, token in enumerate(tokens):
                token_interval = (token["start"], token["end"])
                if not any(_overlaps(token_interval, carrier[:2]) for carrier in compatible_carriers):
                    continue
                cell = (row_id, token_index, channel)
                known_predicate_cells += 1
                if any(_overlaps(token_interval, interval) for interval in predicates_by_channel[channel]):
                    channel_gold_cells.add(cell)
                if channel in token["active"]:
                    channel_predicted_cells.add(cell)
            predicate_cells.add(channel_gold_cells, channel_predicted_cells)
            predicate_cells_by_channel[channel].add(channel_gold_cells, channel_predicted_cells)

            gold_exact = {(row_id, start, end, channel) for start, end in predicates_by_channel[channel]}
            predicted_exact = {
                (row_id, start, end, label)
                for carrier in compatible_carriers
                for start, end, label in _active_intervals(text, tokens, channel, carrier)
            }
            predicate_exact.add(gold_exact, predicted_exact)
            predicate_exact_by_channel[channel].add(gold_exact, predicted_exact)

        if subclass_spec is not None:
            gold_subclasses = _subclass_spans(
                gold,
                "subclass_spans",
                subclass_spec,
                missing_is_unknown=True,
            )
            known_keys = {
                (carrier_start, carrier_end, primary_type, family)
                for carrier_start, carrier_end, primary_type, _start, _end, family, _value in gold_subclasses
            }
            known_subclass_carrier_families += len(known_keys)
            predicted_end_to_end = {
                item
                for item in _subclass_spans(prediction, "subclass_spans", subclass_spec)
                if item[:3] + (item[5],) in known_keys
            }
            predicted_oracle = {
                item
                for item in _subclass_spans(
                    prediction,
                    "subclass_spans_oracle_carriers",
                    subclass_spec,
                )
                if item[:3] + (item[5],) in known_keys
            }
            name_permissive_end_to_end.add(
                _permissive_name_items(text, gold_subclasses),
                _permissive_name_items(text, predicted_end_to_end),
            )
            name_permissive_oracle.add(
                _permissive_name_items(text, gold_subclasses),
                _permissive_name_items(text, predicted_oracle),
            )
            gold_items = {(row_id, *item) for item in gold_subclasses}
            end_to_end_items = {(row_id, *item) for item in predicted_end_to_end}
            oracle_items = {(row_id, *item) for item in predicted_oracle}
            subclass_end_to_end.add(gold_items, end_to_end_items)
            subclass_oracle_carrier.add(gold_items, oracle_items)
            for family in subclass_spec.families:
                family_gold = {item for item in gold_items if item[-2] == family.name}
                subclass_end_to_end_by_family[family.name].add(
                    family_gold,
                    {item for item in end_to_end_items if item[-2] == family.name},
                )
                subclass_oracle_by_family[family.name].add(
                    family_gold,
                    {item for item in oracle_items if item[-2] == family.name},
                )

    result = {
        "rows": len(gold_rows),
        "threshold": PREDICATE_THRESHOLD,
        PROJECTION_FIELD: {
            "policy": REFERENCE_PROJECTION_POLICY,
            "excluded_targets": sum(len(excluded_reference_targets(row)) for row in projected_gold.values()),
            "rows": {
                row_id: row[PROJECTION_FIELD]
                for row_id, row in projected_gold.items()
                if PROJECTION_FIELD in row
            },
        },
        "reference_form_projection": {
            "mode": "routine_non_bare" if routine_reference_projection else "all_references",
            "excluded_values": (
                sorted(ROUTINE_EXCLUDED_REFERENCE_FORMS) if routine_reference_projection else []
            ),
            "excluded_carriers": sum(excluded_reference_forms.values()),
            "excluded_carriers_by_form": dict(sorted(excluded_reference_forms.items())),
        },
        "reference_logit_penalty": reference_logit_penalty,
        "reference_logit_biases": reference_logit_biases,
        "reference_aware_primary_exact": primary_reference_aware.report(),
        "named_only_primary_exact": primary_named_only.report(),
        "optional_reference_primary_exact": primary_optional.report(),
        "optional_reference_policy": {
            "policy": OPTIONAL_REFERENCE_POLICY,
            "neutral_predictions": neutral_optional,
            "neutral_reference_suppressed_predictions": neutral_named,
        },
        "primary_exact_by_type": {
            label: primary_by_type[label].report() for label in sorted(primary_by_type)
        },
        "reference_exact": {
            "micro": references.report(),
            "by_type": {
                reference_type: references_by_type[reference_type].report()
                for reference_type in REFERENCE_TYPES
            },
        },
        "predicate_token_cells": {
            "micro": predicate_cells.report(),
            "known_cells": known_predicate_cells,
            "supported_channels": supported_channels,
            "by_channel": {
                channel: predicate_cells_by_channel[channel].report() for channel in supported_channels
            },
        },
        "predicate_exact": {
            "micro": predicate_exact.report(),
            "by_channel": {
                channel: predicate_exact_by_channel[channel].report() for channel in supported_channels
            },
        },
    }
    if subclass_spec is not None:
        supported_families = [
            family.name
            for family in subclass_spec.families
            if subclass_oracle_by_family[family.name].report()["gold"]
        ]
        result["name_component_permissive"] = {
            "policy": "adjacent-same-kind-given-family-whitespace-v1",
            "end_to_end": name_permissive_end_to_end.report(),
            "oracle_carrier": name_permissive_oracle.report(),
            "scope": "Known name-component carriers only; carrier boundaries, kinds and Q remain distinct. Strict subclass_exact is unchanged.",
        }
        result["subclass_exact"] = {
            "known_carrier_families": known_subclass_carrier_families,
            "supported_families": supported_families,
            "end_to_end": {
                "micro": subclass_end_to_end.report(),
                "by_family": {
                    family: subclass_end_to_end_by_family[family].report() for family in supported_families
                },
            },
            "oracle_carrier": {
                "micro": subclass_oracle_carrier.report(),
                "by_family": {
                    family: subclass_oracle_by_family[family].report() for family in supported_families
                },
            },
        }
    return result


def merge_adjacent_same_type(
    spans: list[dict[str, Any]], text: str, types: frozenset[str]
) -> tuple[list[dict[str, Any]], int]:
    """Merge predicted spans of one type separated only by whitespace into one span.

    Output contract of the encoder route: a ``person_name`` or ``organization``
    span is maximal. The ontology-v1 training pool labels names as adjacent
    given/family/prefix components that the label map sends to ``person_name``
    separately, so a mapped head can emit "Robert" + "McQuinn" as two spans;
    this rule restores the maximal span. Only same-type neighbours whose gap is
    entirely whitespace merge; punctuation or any other character between them
    keeps them apart. Returns the merged spans and the number of merges.
    """
    ordered = sorted(spans, key=lambda span: (span["start"], span["end"]))
    merged: list[dict[str, Any]] = []
    merges = 0
    for span in ordered:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and span["label"] in types
            and previous["label"] == span["label"]
            and previous["end"] <= span["start"]
            and text[previous["end"] : span["start"]].strip() == ""
            and span["start"] > previous["end"]
        ):
            merged[-1] = {**previous, "end": span["end"]}
            merges += 1
            continue
        merged.append(dict(span))
    return merged, merges


ENCLOSING_PUNCTUATION = "()[]{}«»\"'“”„‘’‹›,;:!?"
TRAILING_PERIOD_TYPES = frozenset(
    {
        "date",
        "date_of_birth",
        "time",
        "age",
        "quantity",
        "monetary_amount",
        "postal_code",
        "record_identifier",
        "government_id",
        "phone_number",
        "gps_coordinates",
    }
)
KEEP_EDGE_TYPES = frozenset({"url", "username", "email", "credential", "ip_address"})


def trim_whitespace_offset(text: str, start: int, end: int) -> tuple[int, int]:
    """A token offset never begins or ends on whitespace.

    SentencePiece tokenizers of the DeBERTa family report the leading "▁" space
    inside the word token's offset, so every decoded span would start one
    character early and miss exact character scoring. XLM-R's offsets are
    already trimmed, so this is a no-op there. A token that is all whitespace
    keeps its offset; the decoder never labels it.
    """
    trimmed_start, trimmed_end = start, end
    while trimmed_start < trimmed_end and text[trimmed_start].isspace():
        trimmed_start += 1
    while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
        trimmed_end -= 1
    return (trimmed_start, trimmed_end) if trimmed_start < trimmed_end else (start, end)


def trim_edge_punctuation(spans: list[dict[str, Any]], text: str) -> tuple[list[dict[str, Any]], int]:
    """Drop enclosing punctuation a token-level span absorbed from a glued token.

    Output contract of the encoder route: a span excludes the brackets, quotes,
    commas, and clause punctuation around it. The tokenizer often glues such
    characters to the neighbouring word ("2017.", "(2000)"), so a token-correct
    prediction covers them; the gold never does. Whitespace at the edges is
    trimmed too. A trailing period is removed only for numeric-like types where
    it can never be part of the entity; names and organizations keep a final
    period ("Inc.", "Mons."). Types whose surfaces legitimately start or end with
    symbols (url, username, email, credential, ip_address) are left untouched.
    Returns the trimmed spans and the number changed; a span that would become
    empty is kept as is.
    """
    trimmed: list[dict[str, Any]] = []
    changed = 0
    for span in spans:
        label = span["label"]
        if label in KEEP_EDGE_TYPES:
            trimmed.append(dict(span))
            continue
        start, end = span["start"], span["end"]
        strip = ENCLOSING_PUNCTUATION + (". " if label in TRAILING_PERIOD_TYPES else " ")
        while start < end and (text[start] in ENCLOSING_PUNCTUATION or text[start].isspace()):
            start += 1
        while end > start and (text[end - 1] in strip or text[end - 1].isspace()):
            end -= 1
        if start >= end:
            trimmed.append(dict(span))
            continue
        if (start, end) != (span["start"], span["end"]):
            changed += 1
        trimmed.append({**span, "start": start, "end": end})
    return trimmed, changed


def _decode_window(
    logits,
    offsets,
    id2label,
    suppressed_columns=(),
    penalized_columns=(),
    column_penalty=0.0,
    column_biases=None,
):
    import numpy as np

    if __package__:
        from scripts.pii_bioes import constrained_bioes_decode, count_bioes_violations
        from scripts.pii_eval import decode_bioes_labels
    else:
        from pii_bioes import constrained_bioes_decode, count_bioes_violations
        from pii_eval import decode_bioes_labels

    decision = np.asarray(logits, dtype=np.float32).copy()
    if penalized_columns and column_penalty:
        decision[:, list(penalized_columns)] -= column_penalty
    if column_biases:
        for column, bias in column_biases.items():
            decision[:, column] += bias
    if suppressed_columns:
        decision[:, list(suppressed_columns)] = -np.inf
    greedy_ids = decision.argmax(axis=-1)
    labels = [id2label[int(label_id)] for label_id in greedy_ids]
    invalid, constraints = count_bioes_violations(labels)
    changed = 0
    if invalid:
        projected_ids = constrained_bioes_decode(decision, id2label)
        changed = int((projected_ids != greedy_ids).sum())
        labels = [id2label[int(label_id)] for label_id in projected_ids]
    return (
        decode_bioes_labels(labels, offsets),
        {
            "constraints": constraints,
            "invalid_constraints": invalid,
            "changed_tokens": changed,
            "affected_window": int(bool(invalid)),
        },
        labels,
    )


def select_decoded_conditioned_predicate_logits(
    predicate_logits,
    decoded_labels,
    condition_types,
):
    """Select a semantic-type block after primary BIOES projection.

    ``O`` and predicted primary types without a predicate block return
    ``None`` so they emit no predicates. Boundary variants share one semantic
    type block.
    """
    import numpy as np

    values = np.asarray(predicate_logits)
    if values.ndim != 3:
        raise ValueError(
            f"conditioned predicate logits must have shape [token, type, channel], got {values.shape}"
        )
    if len(decoded_labels) != values.shape[0]:
        raise ValueError("decoded primary labels do not align with predicate tokens")
    if len(condition_types) != values.shape[1] or len(set(condition_types)) != len(condition_types):
        raise ValueError("predicate condition inventory does not align with logits")
    type2id = {primary_type: index for index, primary_type in enumerate(condition_types)}
    selected = []
    for token_logits, label in zip(values, decoded_labels, strict=True):
        if label == "O":
            selected.append(None)
            continue
        boundary, separator, primary_type = label.partition("-")
        if separator != "-" or boundary not in "BIES":
            raise ValueError(f"invalid decoded primary label {label!r}")
        condition_id = type2id.get(primary_type)
        selected.append(None if condition_id is None else token_logits[condition_id])
    return selected


def logit_diagnostic_token(
    primary_logits,
    predicate_logits,
    *,
    id2label: dict[int, str],
    predicate_names: list[str],
    condition_types: list[str],
) -> dict[str, Any]:
    """Keep the small logit slice needed to diagnose added ont3 channels."""
    import numpy as np

    primary = np.asarray(primary_logits)
    predicate = np.asarray(predicate_logits)
    if primary.shape != (len(id2label),):
        raise ValueError("primary diagnostic logits do not align with label inventory")
    reference_label_ids = {
        label_id: label for label_id, label in id2label.items() if label.partition("-")[2] in REFERENCE_TYPES
    }
    nonreference_label_ids = sorted(set(id2label) - set(reference_label_ids))
    if not reference_label_ids or not nonreference_label_ids:
        raise ValueError("diagnostic requires both reference and non-reference primary labels")
    expected_predicate_shape = (
        (len(condition_types), len(predicate_names)) if condition_types else (len(predicate_names),)
    )
    if predicate.shape != expected_predicate_shape:
        raise ValueError(
            "predicate diagnostic logits do not align with condition/channel inventories: "
            f"{predicate.shape} != {expected_predicate_shape}"
        )
    best_nonreference_id = max(
        nonreference_label_ids,
        key=lambda label_id: float(primary[label_id]),
    )
    conditioned = (
        {
            condition_type: {
                channel: float(predicate[condition_id, channel_id])
                for channel_id, channel in enumerate(predicate_names)
            }
            for condition_id, condition_type in enumerate(condition_types)
        }
        if condition_types
        else {
            "unconditioned": {
                channel: float(predicate[channel_id]) for channel_id, channel in enumerate(predicate_names)
            }
        }
    )
    ranked_nonreference = sorted(
        nonreference_label_ids, key=lambda label_id: float(primary[label_id]), reverse=True
    )
    return {
        "reference_label_logits": {
            label: float(primary[label_id]) for label_id, label in reference_label_ids.items()
        },
        "best_nonreference": {
            "label": id2label[best_nonreference_id],
            "logit": float(primary[best_nonreference_id]),
        },
        # the top non-reference alternatives, for near-miss analysis of BIOES decisions
        "nonreference_topk": [
            {"label": id2label[label_id], "logit": float(primary[label_id])}
            for label_id in ranked_nonreference[:5]
        ],
        "conditioned_predicate_logits": conditioned,
    }


def predict_rows(
    model_path: str | Path,
    rows: list[dict[str, Any]],
    predicate_names: list[str],
    subclass_spec: SubclassSpec | None = None,
    reference_logit_penalty: float = 0.0,
    reference_logit_biases: dict[str, float] | None = None,
    outside_logit_bias: float = 0.0,
    emit_logit_diagnostics: bool = False,
    merge_adjacent_types: frozenset[str] = frozenset(),
    trim_edges: bool = False,
    register_language: str = "row",
    domain_posteriors: str | Path | None = None,
    register_domain: str = "available",
    context_field: str | None = None,
    window_length: int | None = None,
    context_side: str | None = None,
    primary_window_sink=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one model forward and derive both primary views plus predicates.

    ``merge_adjacent_types`` applies the route's maximal-span output contract
    (see ``merge_adjacent_same_type``) to both primary views before subclass
    carriers are derived; ``trim_edges`` then drops absorbed enclosing
    punctuation (see ``trim_edge_punctuation``). Both counts are reported in
    the run summary.
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if __package__:
        from scripts.pii_domain_model import load_posteriors as load_domain_posteriors
        from scripts.pii_domain_model import text_key as domain_text_key
        from scripts.pii_eval import (
            _resolve_suppressed_primary_columns,
            _tokenizer_capacity_windows,
            dedupe_window_preds,
        )
        from scripts.pii_layered_head_model import LayerConcatForTokenClassification
        from scripts.pii_prompt_slots import target_bounds as prompt_target_bounds
        from scripts.pii_soft_registers import (
            REGISTER_CONFIG_KEY,
            conditions_from_config,
            placeholder_id,
            reserve_window,
        )
        from scripts.pii_tag_status_prompt import CONFIG_KEY as TAG_STATUS_PROMPT_KEY
        from scripts.pii_tag_status_prompt import TagStatusPrompt
    else:
        from pii_domain_model import load_posteriors as load_domain_posteriors
        from pii_domain_model import text_key as domain_text_key
        from pii_eval import (
            _resolve_suppressed_primary_columns,
            _tokenizer_capacity_windows,
            dedupe_window_preds,
        )
        from pii_layered_head_model import LayerConcatForTokenClassification
        from pii_prompt_slots import target_bounds as prompt_target_bounds
        from pii_soft_registers import (
            REGISTER_CONFIG_KEY,
            conditions_from_config,
            placeholder_id,
            reserve_window,
        )
        from pii_tag_status_prompt import CONFIG_KEY as TAG_STATUS_PROMPT_KEY
        from pii_tag_status_prompt import TagStatusPrompt

    if (
        isinstance(reference_logit_penalty, bool)
        or not isinstance(reference_logit_penalty, (int, float))
        or not math.isfinite(reference_logit_penalty)
        or reference_logit_penalty < 0
    ):
        raise ValueError("reference-logit penalty must be finite and nonnegative")
    reference_logit_biases = _normalize_reference_logit_biases(reference_logit_biases)
    config = AutoConfig.from_pretrained(model_path)
    checkpoint_channels = list(getattr(config, "pii_predicate_channels", ()) or ())
    if checkpoint_channels != predicate_names:
        raise ValueError(
            "checkpoint/spec predicate channel mismatch: "
            f"checkpoint={checkpoint_channels}, spec={predicate_names}"
        )
    condition_types = list(getattr(config, "pii_predicate_condition_types", ()) or ())
    reference_type_residual_types = list(getattr(config, "pii_reference_type_residual_types", ()) or ())
    if subclass_spec is not None:
        checkpoint_spec_hash = getattr(config, "pii_subclass_spec_sha256", None)
        checkpoint_blocks = list(getattr(config, "pii_subclass_blocks", ()) or ())
        if checkpoint_spec_hash != subclass_spec.sha256:
            raise ValueError(
                "checkpoint/spec subclass hash mismatch: "
                f"checkpoint={checkpoint_spec_hash}, spec={subclass_spec.sha256}"
            )
        if checkpoint_blocks != subclass_spec.config_blocks():
            raise ValueError("checkpoint/spec subclass block mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = (
        LayerConcatForTokenClassification.from_local_checkpoint(
            model_path,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    id2label = {int(label_id): label for label_id, label in model.config.id2label.items()}
    suppressed = _resolve_suppressed_primary_columns(id2label, REFERENCE_TYPES)
    reference_column_biases = {
        label_id: reference_logit_biases[primary_type]
        for label_id, label in id2label.items()
        if (primary_type := label.partition("-")[2]) in reference_logit_biases
        and reference_logit_biases[primary_type]
    }
    # One number moves the whole precision/recall operating point without retraining.
    # A negative bias on the outside column makes every entity column easier to win,
    # which is what a pipeline wants when a later stage prunes better than it adds.
    outside_column_biases: dict[int, float] = {}
    if outside_logit_bias:
        outside_columns = [label_id for label_id, label in id2label.items() if label == "O"]
        if not outside_columns:
            raise ValueError("checkpoint has no O column to bias")
        outside_column_biases = {label_id: outside_logit_bias for label_id in outside_columns}
        for label_id, bias in outside_column_biases.items():
            reference_column_biases[label_id] = reference_column_biases.get(label_id, 0.0) + bias
    max_length = min(
        getattr(model.config, "max_position_embeddings", 1 << 20) - 2,
        tokenizer.model_max_length,
        16384,
    )
    # A register checkpoint expects its reserved slots to be present at the same offsets
    # the trainer used, so the sentence budget shrinks by exactly what they occupy.
    register_count = int(getattr(model.config, REGISTER_CONFIG_KEY, 0) or 0)
    register_filler = placeholder_id(tokenizer) if register_count else 0
    max_length -= register_count
    # A tag-status prompt checkpoint prepends learned slots that also occupy positions;
    # its training-only slots stay masked here because no gold status is supplied.
    prompt = None
    prompt_width = 0
    if getattr(model.config, TAG_STATUS_PROMPT_KEY, None):
        if register_count:
            raise ValueError("tag-status prompts and soft registers cannot be combined")
        prompt = TagStatusPrompt.load_for(model, model_path).cuda().eval()
        prompt_width = prompt.prefix_width
        max_length -= prompt_width
    # Prompt slots installed in the checkpoint itself are spliced in by the model at
    # each window's target bounds; their positions also come out of the budget.
    installed_prompt = getattr(model, "pii_prompt_slots", None)
    if installed_prompt is not None:
        if register_count or prompt is not None:
            raise ValueError("installed prompt slots cannot be combined with registers or a prompt file")
        prompt_width = installed_prompt.width
        max_length -= prompt_width
    if context_field is None:
        context_field = getattr(config, "pii_context_field", None)
    context_side = resolve_context_side(context_side, getattr(config, "pii_context_side", "both"))
    # A model trained with the document-start marker gets it on flagged rows;
    # any other model sees flagged rows exactly as it always did.
    document_start_marker = bool(getattr(config, "pii_document_start_marker", False))
    context_capacity = getattr(config, "pii_context_max_length", 512 if context_field else None)
    if window_length is not None:
        if window_length <= register_count + tokenizer.num_special_tokens_to_add():
            raise ValueError("window length must leave room for target tokens")
        context_capacity = window_length
    if context_capacity is not None:
        max_length = min(max_length, context_capacity - register_count - prompt_width)
    context_rows = context_tokens = 0
    # Conditions differ in what production can supply, so they are fed differently.
    # `source` is pinned to unknown: nothing at inference reveals which annotation pipeline
    # would have labelled a row, so any gain there is nuisance absorption for the shared
    # weights. `language` is the row's own, because that is available in deployment; pass
    # `--register-language unknown` to measure the unspecified-language fallback instead.
    # A per-language output bias is a second, independent way a checkpoint can be
    # conditioned on language, and it is mandatory rather than optional: the head refuses
    # to run without a route for every row. Index -1 selects the shared head, which is what
    # a language outside the trained inventory gets.
    bias_languages = list(getattr(model.config, "pii_language_bias_languages", ()) or ())
    bias_index = {value: position for position, value in enumerate(bias_languages)}
    register_conditions = conditions_from_config(model.config)
    language_index = {
        value: position
        for name, values in register_conditions
        if name == "language"
        for position, value in enumerate(values)
    }
    # A learned domain is the third kind: unlike the annotation source it is computable
    # from the text at inference, so it is supplied rather than pinned. `register_domain`
    # names the deployment scenario being measured -- classify the sentence, classify the
    # document, both, or neither -- which is the comparison the dropout during training
    # exists to make possible.
    domain_names = tuple(name for name, _values in register_conditions if name.startswith("domain_"))
    domain_table = load_domain_posteriors(domain_posteriors) if domain_posteriors else {}
    if domain_names and not domain_table and register_domain != "none":
        raise ValueError(
            f"checkpoint conditions on {list(domain_names)} but no posteriors were supplied; "
            "pass the sidecar, or --register-domain none to measure the unspecified fallback"
        )
    supplied = {
        name
        for name in domain_names
        if domain_table and register_domain in ("available", name.partition("_")[2])
    }
    if register_language == "row" and language_index:
        supplied.add("language")
    pinned_conditions = {
        f"{name}_id": torch.tensor([0], device="cuda")
        for name, _values in register_conditions
        if name not in supplied
    }
    domain_widths = {name: len(values) for name, values in register_conditions if name in supplied}
    domain_supplied_rows = dict.fromkeys(sorted(supplied - {"language"}), 0)
    projections = {
        "reference_aware": defaultdict(int),
        "named_only": defaultdict(int),
    }
    predictions = []
    total_windows = 0
    total_input_tokens = 0
    bias_routed_rows = 0
    # Peak memory is reported for this scoring pass, so it must not inherit whatever the
    # checkpoint load happened to allocate.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for row_index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{row.get('id')}: missing text")
        row_conditions = dict(pinned_conditions)
        if bias_index:
            route = bias_index.get(row.get("lang"), -1) if register_language == "row" else -1
            row_conditions["language_ids"] = torch.tensor([route], dtype=torch.long, device="cuda")
            bias_routed_rows += route >= 0
        if language_index and register_language == "row":
            # A language absent from the frozen vocabulary falls back to index 0, the same
            # unspecified condition training kept alive through dropout.
            row_conditions["language_id"] = torch.tensor(
                [language_index.get(row.get("lang"), 0)], device="cuda"
            )
        domain_record = domain_table.get(domain_text_key(text)) if domain_supplied_rows else None
        for name in domain_supplied_rows:
            values = domain_record.get(name.partition("_")[2]) if domain_record else None
            if values is None:
                # No posterior for this row at this scope: the unspecified condition, which
                # is exactly what training's dropout kept trained.
                row_conditions[f"{name}_id"] = torch.tensor([0], device="cuda")
                continue
            if len(values) + 1 != domain_widths[name]:
                raise ValueError(
                    f"{name}: sidecar has {len(values)} domains, checkpoint trained on "
                    f"{domain_widths[name] - 1}"
                )
            row_conditions[f"{name}_id"] = torch.tensor([[0.0, *values]], device="cuda")
            domain_supplied_rows[name] += 1
        context = row.get(context_field) if context_field else None
        contextual = (
            encode_document_context(
                tokenizer,
                text,
                context,
                max_length,
                side=context_side,
                document_start_marker=document_start_marker,
            )
            if context is not None
            else None
        )
        if contextual is None:
            _stride, windows = _tokenizer_capacity_windows(tokenizer, text, max_length)
        else:
            windows = [contextual]
            context_rows += 1
            context_tokens += sum(a == b for a, b in contextual[1]) - tokenizer.num_special_tokens_to_add()
        reference_spans = []
        named_spans = []
        predicate_sums: dict[tuple[int, int, int], Any] = {}
        predicate_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        predicate_token_keys = set()
        diagnostic_primary_sums: dict[tuple[int, int, int], Any] = {}
        diagnostic_predicate_sums: dict[tuple[int, int, int], Any] = {}
        diagnostic_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        subclass_sums: dict[tuple[int, int, int], Any] = {}
        subclass_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        for model_inputs, offsets in windows:
            model_inputs, offsets = reserve_window(model_inputs, offsets, register_count, register_filler)
            encoded = {name: torch.tensor([values], device="cuda") for name, values in model_inputs.items()}
            if installed_prompt is not None:
                start, end = prompt_target_bounds(offsets)
                encoded["prompt_target_start"] = torch.tensor([start], device="cuda")
                encoded["prompt_target_end"] = torch.tensor([end], device="cuda")
                if installed_prompt.languages:
                    language = row.get("lang")
                    index = (
                        installed_prompt.languages.index(language)
                        if language in installed_prompt.languages
                        else -1
                    )
                    encoded["prompt_language_id"] = torch.tensor([index], device="cuda")
            with torch.no_grad():
                if prompt is None:
                    outputs = model(**encoded, **row_conditions)
                else:
                    if set(row_conditions) - {"language_ids"} or set(encoded) != {
                        "input_ids",
                        "attention_mask",
                    }:
                        raise ValueError(
                            "tag-status prompts accept only token inputs and a language-bias route"
                        )
                    language_ids = None
                    if prompt.languages:
                        language = row.get("lang")
                        index = prompt.languages.index(language) if language in prompt.languages else -1
                        language_ids = torch.tensor([index], device="cuda")
                    outputs = prompt(
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        language_ids=language_ids,
                        bias_language_ids=row_conditions.get("language_ids"),
                    )
            if predicate_names and getattr(outputs, "predicate_logits", None) is None:
                raise ValueError("checkpoint has no predicate logits")
            if subclass_spec is not None and outputs.subclass_logits is None:
                raise ValueError("checkpoint has no subclass logits")
            if reference_type_residual_types and outputs.reference_type_logits is None:
                raise ValueError("checkpoint declares but does not emit reference-type residual logits")
            valid_indices = [index for index, (start, end) in enumerate(offsets) if start != end]
            valid_offsets = [trim_whitespace_offset(row["text"], *offsets[index]) for index in valid_indices]
            primary_logits = outputs.logits[0, valid_indices].float().cpu().numpy()
            if primary_window_sink is not None:
                primary_window_sink(row["id"], primary_logits.copy(), valid_offsets)
            predicate_logits = (
                outputs.predicate_logits[0, valid_indices].float().cpu().numpy()
                if predicate_names
                else primary_logits[:, :0]
            )
            subclass_logits = (
                outputs.subclass_logits[0, valid_indices].float().cpu().numpy()
                if subclass_spec is not None
                else None
            )
            full_spans, full_stats, full_labels = _decode_window(
                primary_logits,
                valid_offsets,
                id2label,
                penalized_columns=suppressed,
                column_penalty=reference_logit_penalty,
                column_biases=reference_column_biases,
            )
            masked_spans, masked_stats, _masked_labels = _decode_window(
                primary_logits,
                valid_offsets,
                id2label,
                suppressed,
                column_biases=outside_column_biases,
            )
            reference_spans.extend(full_spans)
            named_spans.extend(masked_spans)
            for view, stats in (
                ("reference_aware", full_stats),
                ("named_only", masked_stats),
            ):
                projections[view]["windows"] += 1
                for key, value in stats.items():
                    projections[view][key] += value
            input_ids = model_inputs["input_ids"]
            selected_predicate_logits = (
                select_decoded_conditioned_predicate_logits(
                    predicate_logits,
                    full_labels,
                    condition_types,
                )
                if condition_types
                else predicate_logits
            )
            if emit_logit_diagnostics:
                for index, offset, primary, conditioned in zip(
                    valid_indices,
                    valid_offsets,
                    primary_logits,
                    predicate_logits,
                    strict=True,
                ):
                    key = (int(offset[0]), int(offset[1]), int(input_ids[index]))
                    if key not in diagnostic_primary_sums:
                        diagnostic_primary_sums[key] = primary.copy()
                        diagnostic_predicate_sums[key] = conditioned.copy()
                    else:
                        diagnostic_primary_sums[key] += primary
                        diagnostic_predicate_sums[key] += conditioned
                    diagnostic_counts[key] += 1
            for index, offset, logits in zip(
                valid_indices,
                valid_offsets,
                selected_predicate_logits,
                strict=True,
            ):
                key = (int(offset[0]), int(offset[1]), int(input_ids[index]))
                predicate_token_keys.add(key)
                if logits is None:
                    continue
                if key not in predicate_sums:
                    predicate_sums[key] = logits.copy()
                else:
                    predicate_sums[key] += logits
                predicate_counts[key] += 1
            if subclass_logits is not None:
                for index, offset, logits in zip(
                    valid_indices,
                    valid_offsets,
                    subclass_logits,
                    strict=True,
                ):
                    key = (int(offset[0]), int(offset[1]), int(input_ids[index]))
                    if key not in subclass_sums:
                        subclass_sums[key] = logits.copy()
                    else:
                        subclass_sums[key] += logits
                    subclass_counts[key] += 1
            total_windows += 1
            total_input_tokens += len(input_ids)
        predicate_tokens = []
        for start, end, token_id in sorted(predicate_token_keys):
            key = (start, end, token_id)
            average = None if predicate_counts[key] == 0 else predicate_sums[key] / predicate_counts[key]
            predicate_tokens.append(
                {
                    "start": start,
                    "end": end,
                    "token_id": token_id,
                    "active": [
                        channel
                        for channel, logit in (
                            zip(predicate_names, average, strict=True) if average is not None else ()
                        )
                        if float(logit) > 0.0
                    ],
                }
            )
        reference_aware_preds = dedupe_window_preds(reference_spans)
        named_only_preds = dedupe_window_preds(named_spans)
        if merge_adjacent_types:
            reference_aware_preds, merged_reference = merge_adjacent_same_type(
                reference_aware_preds, row["text"], merge_adjacent_types
            )
            named_only_preds, merged_named = merge_adjacent_same_type(
                named_only_preds, row["text"], merge_adjacent_types
            )
            projections["reference_aware"]["merged_adjacent_spans"] += merged_reference
            projections["named_only"]["merged_adjacent_spans"] += merged_named
        if trim_edges:
            reference_aware_preds, trimmed_reference = trim_edge_punctuation(
                reference_aware_preds, row["text"]
            )
            named_only_preds, trimmed_named = trim_edge_punctuation(named_only_preds, row["text"])
            projections["reference_aware"]["trimmed_edge_spans"] += trimmed_reference
            projections["named_only"]["trimmed_edge_spans"] += trimmed_named
        prediction = {
            "id": row["id"],
            "reference_aware_preds": reference_aware_preds,
            "named_only_preds": named_only_preds,
            "predicate_tokens": predicate_tokens,
            "reference_logit_penalty": reference_logit_penalty,
            "reference_logit_biases": reference_logit_biases,
            "outside_logit_bias": outside_logit_bias,
            "window_count": len(windows),
        }
        if emit_logit_diagnostics:
            diagnostic_tokens = []
            for start, end, token_id in sorted(diagnostic_primary_sums):
                key = (start, end, token_id)
                divisor = diagnostic_counts[key]
                primary = diagnostic_primary_sums[key] / divisor
                conditioned = diagnostic_predicate_sums[key] / divisor
                diagnostic_tokens.append(
                    {
                        "start": start,
                        "end": end,
                        "token_id": token_id,
                        **logit_diagnostic_token(
                            primary,
                            conditioned,
                            id2label=id2label,
                            predicate_names=predicate_names,
                            condition_types=condition_types,
                        ),
                    }
                )
            prediction["logit_diagnostic_tokens"] = diagnostic_tokens
        if subclass_spec is not None:
            subclass_keys = sorted(subclass_sums)
            subclass_offsets = [(start, end) for start, end, _token_id in subclass_keys]
            averaged_subclasses = [subclass_sums[key] / subclass_counts[key] for key in subclass_keys]
            predicted_carriers = [
                [span["start"], span["end"], span["label"]] for span in reference_aware_preds
            ]
            oracle_carriers = [list(span) for span in gold_views(row, predicate_names)[0]]
            prediction["subclass_spans"] = decode_subclass_logits(
                averaged_subclasses,
                subclass_offsets,
                predicted_carriers,
                subclass_spec,
            )
            prediction["subclass_spans_oracle_carriers"] = decode_subclass_logits(
                averaged_subclasses,
                subclass_offsets,
                oracle_carriers,
                subclass_spec,
            )
        predictions.append(prediction)
        if (row_index + 1) % 100 == 0:
            print(f"ont3-eval predict {row_index + 1}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return predictions, {
        "rows": len(rows),
        "windows": total_windows,
        "input_tokens_including_special_and_overlap": total_input_tokens,
        "elapsed_seconds": elapsed,
        "max_input_tokens": max_length,
        "execution": {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "disable_addmm_cuda_lt": os.environ.get("DISABLE_ADDMM_CUDA_LT"),
        },
        # What one deployment of this checkpoint costs, measured rather than estimated, so
        # a conditioning arm's benefit can be put beside its price on the same run. The
        # register positions are counted separately because they are the whole added cost:
        # every window carries them whether or not a condition is supplied.
        "inference_cost": {
            "rows_per_second": len(rows) / elapsed if elapsed > 0 else None,
            "windows_per_second": total_windows / elapsed if elapsed > 0 else None,
            "input_tokens_per_second": total_input_tokens / elapsed if elapsed > 0 else None,
            "milliseconds_per_row": 1000.0 * elapsed / len(rows) if rows else None,
            "register_positions_per_window": register_count,
            "register_tokens_total": register_count * total_windows,
            "register_share_of_input_tokens": (
                register_count * total_windows / total_input_tokens if total_input_tokens else 0.0
            ),
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "batch_rows": 1,
            "dtype": "bfloat16",
        },
        "register_conditions": {
            "declared": [name for name, _values in register_conditions],
            "supplied": sorted(supplied),
            "pinned_unknown": sorted(name for name, _values in register_conditions if name not in supplied),
            "language": register_language,
            "domain_scope": register_domain if domain_names else "not-conditioned",
            "domain_posterior_rows": dict(domain_supplied_rows),
            "domain_posterior_source": str(domain_posteriors) if domain_posteriors else None,
        },
        "language_bias": {
            "languages": len(bias_languages),
            "strength": float(getattr(config, "pii_language_bias_strength", 0.0) or 0.0),
            "rows_routed_to_a_language": bias_routed_rows,
            "rows_on_the_shared_head": len(rows) - bias_routed_rows,
        }
        if bias_languages
        else None,
        "windowing": "token-capacity-overlap",
        "document_context": {
            "field": context_field,
            "side": context_side,
            "max_length": max_length + register_count,
            "rows_with_context": context_rows,
            "context_tokens": context_tokens,
        },
        "predicate_threshold": PREDICATE_THRESHOLD,
        "reference_logit_penalty": reference_logit_penalty,
        "reference_logit_biases": reference_logit_biases,
        "predicate_conditioning": "primary_type" if condition_types else "none",
        "predicate_condition_types": condition_types,
        "reference_type_residual": {
            "types": reference_type_residual_types,
            "loss_weight": float(getattr(config, "pii_reference_type_residual_loss_weight", 0.0) or 0.0),
            "application": "one semantic score added to each type's B/I/E/S primary logits",
        },
        "subclass_decoding": (
            "predicted_and_oracle_primary_carriers_with_whole-span_pooling_and_component_grammar"
            if subclass_spec is not None
            else "disabled"
        ),
        "overlap_predicate_logit_reduction": (
            "decoded-entity-condition arithmetic mean over entity windows then strict logit_gt_0; "
            "no selected entity condition emits no predicates"
            if condition_types
            else "arithmetic_mean_then_strict_logit_gt_0"
        ),
        "named_only_suppressed_types": list(REFERENCE_TYPES),
        "logit_diagnostics": emit_logit_diagnostics,
        "projection": {view: dict(values) for view, values in projections.items()},
    }


def _write_new_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, sort_keys=True)
        output.write("\n")


def _write_new_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cmd_predict(args: argparse.Namespace) -> None:
    for destination in (args.output, args.receipt):
        if Path(destination).exists():
            raise FileExistsError(destination)
    all_input_rows = _load_jsonl(args.input)
    all_bcp47 = _load_bcp47_sidecar(args.bcp47, len(all_input_rows))
    input_rows = all_input_rows
    input_bcp47 = all_bcp47
    if args.limit:
        input_rows = input_rows[: args.limit]
        input_bcp47 = input_bcp47[: args.limit]
    predicate_names, _applicable = _load_predicate_spec(args.predicate_spec)
    subclass_spec = load_subclass_spec(Path(args.subclass_spec)) if args.subclass_spec else None
    reference_logit_biases = dict(args.reference_logit_bias)
    if len(reference_logit_biases) != len(args.reference_logit_bias):
        raise ValueError("duplicate reference-logit bias type")
    merge_adjacent_types = frozenset(
        item.strip() for item in args.merge_adjacent_types.split(",") if item.strip()
    )
    predictions, run = predict_rows(
        args.model,
        input_rows,
        predicate_names,
        subclass_spec,
        args.reference_logit_penalty,
        reference_logit_biases,
        args.outside_logit_bias,
        args.emit_logit_diagnostics,
        merge_adjacent_types,
        args.trim_edge_punctuation,
        args.register_language,
        args.domain_posteriors,
        args.register_domain,
        args.context_field,
        args.max_length,
        args.context_side,
    )
    run["merge_adjacent_types"] = sorted(merge_adjacent_types)
    run["trim_edge_punctuation"] = bool(args.trim_edge_punctuation)
    if args.regex_config:
        # Structured identifiers the model may never have seen are better found by
        # a pattern. This runs after both projections exist so the model keeps the
        # spans it already has; see scripts/pii_regex_tag.py for the overlap policy.
        if __package__:
            from scripts.pii_regex_tag import load_policy, load_rules, tag_rows
        else:
            from pii_regex_tag import load_policy, load_rules, tag_rows

        rules, families, regex_receipt = load_rules(Path(args.regex_config), args.regex_ontology)
        policy, arbitration, policy_receipt = load_policy(
            Path(args.regex_policy) if args.regex_policy else None
        )
        if args.regex_mode:
            policy = dataclasses.replace(policy, mode=args.regex_mode)
        run["regex_tagging"] = {
            **regex_receipt,
            **policy_receipt,
            "sha256": _sha256(args.regex_config),
            "policy_sha256": _sha256(args.regex_policy) if args.regex_policy else None,
            **tag_rows(
                predictions,
                rules,
                span_keys=["reference_aware_preds", "named_only_preds"],
                policy=policy,
                families=families,
                scores=arbitration["scores"],
                gates=arbitration["gates"],
                texts=[row["text"] for row in input_rows],
            ),
        }
    for prediction, tag in zip(predictions, input_bcp47, strict=True):
        prediction["bcp47"] = tag
    _write_new_jsonl(args.output, predictions)
    model_path = Path(args.model)
    weights_path = model_path / "model.safetensors"
    receipt = {
        "schema": "pii-ont3-fixed-threshold-predictions",
        "version": 1,
        "status": "complete",
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "weights_sha256": _sha256(weights_path),
        },
        "input": {
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "rows_available": len(all_input_rows),
            "rows_predicted": len(input_rows),
            "prefix_limit": args.limit,
        },
        "bcp47": {
            "path": str(args.bcp47),
            "sha256": _sha256(args.bcp47),
            "rows_available": len(all_bcp47),
            "rows_predicted": len(input_bcp47),
            "tag_counts": {tag: input_bcp47.count(tag) for tag in sorted(set(input_bcp47))},
        },
        "predicate_spec": {
            "path": str(args.predicate_spec),
            "sha256": _sha256(args.predicate_spec),
            "channels": predicate_names,
        },
        "subclass_spec": (
            {
                "path": str(args.subclass_spec),
                "sha256": subclass_spec.sha256,
                "head_rows": subclass_spec.head_rows,
                "families": [family.name for family in subclass_spec.families],
            }
            if subclass_spec is not None
            else None
        ),
        "output": {"path": str(args.output), "sha256": _sha256(args.output)},
        "run": run,
    }
    _write_new_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def cmd_score(args: argparse.Namespace) -> None:
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    all_gold = _load_jsonl(args.gold)
    all_gold_bcp47 = _load_bcp47_sidecar(args.bcp47, len(all_gold))
    if args.limit and args.id_list:
        raise ValueError("--limit and --id-list are mutually exclusive")
    selected_ids = _load_id_list(args.id_list) if args.id_list else None
    if selected_ids is not None:
        selected = set(selected_ids)
        gold_bcp47 = [tag for row, tag in zip(all_gold, all_gold_bcp47, strict=True) if row["id"] in selected]
        gold = _select_rows_by_ids(all_gold, selected_ids, source=args.gold)
    else:
        gold = all_gold[: args.limit] if args.limit else all_gold
        gold_bcp47 = all_gold_bcp47[: args.limit] if args.limit else all_gold_bcp47
    all_predictions = _load_jsonl(args.predictions)
    predictions = (
        _select_rows_by_ids(all_predictions, selected_ids, source=args.predictions)
        if selected_ids is not None
        else all_predictions
    )
    expected_bcp47 = {row["id"]: tag for row, tag in zip(gold, gold_bcp47, strict=True)}
    for prediction in predictions:
        observed = prediction.get("bcp47")
        expected = expected_bcp47.get(prediction["id"])
        if observed != expected:
            raise ValueError(f"{prediction['id']}: prediction bcp47 {observed!r} does not match {expected!r}")
    predicate_names, applicable = _load_predicate_spec(args.predicate_spec)
    subclass_spec = load_subclass_spec(Path(args.subclass_spec)) if args.subclass_spec else None
    metrics = score_rows(
        gold,
        predictions,
        predicate_names,
        applicable,
        subclass_spec,
        routine_reference_projection=args.reference_form_projection == "routine",
    )
    result = {
        "schema": "pii-ont3-fixed-threshold-score",
        "version": 1,
        "status": "complete",
        "gold": {
            "path": str(args.gold),
            "sha256": _sha256(args.gold),
            "rows_available": len(all_gold),
            "rows_scored": len(gold),
            "prefix_limit": args.limit,
            "id_list": (
                {
                    "path": str(args.id_list),
                    "sha256": _sha256(args.id_list),
                    "rows": len(selected_ids),
                }
                if selected_ids is not None
                else None
            ),
        },
        "bcp47": {
            "path": str(args.bcp47),
            "sha256": _sha256(args.bcp47),
            "tag_counts": {tag: gold_bcp47.count(tag) for tag in sorted(set(gold_bcp47))},
        },
        "predictions": {
            "path": str(args.predictions),
            "sha256": _sha256(args.predictions),
        },
        "predicate_spec": {
            "path": str(args.predicate_spec),
            "sha256": _sha256(args.predicate_spec),
        },
        "subclass_spec": (
            {"path": str(args.subclass_spec), "sha256": subclass_spec.sha256}
            if subclass_spec is not None
            else None
        ),
        "metrics": metrics,
    }
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    predict = subcommands.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--bcp47", required=True)
    predict.add_argument("--predicate-spec", required=True)
    predict.add_argument("--subclass-spec")
    predict.add_argument("--output", required=True)
    predict.add_argument("--receipt", required=True)
    predict.add_argument("--limit", type=int, default=0)
    predict.add_argument("--reference-logit-penalty", type=float, default=0.0)
    predict.add_argument(
        "--reference-logit-bias",
        action="append",
        default=[],
        type=_parse_reference_logit_bias,
        metavar="TYPE=FLOAT",
        help="add a type-specific bias to all BIOES logits for one reference type",
    )
    predict.add_argument(
        "--outside-logit-bias",
        type=float,
        default=0.0,
        help="add this to the O column before decoding, moving the whole "
        "precision/recall operating point without retraining; negative recalls more",
    )
    predict.add_argument("--emit-logit-diagnostics", action="store_true")
    predict.add_argument(
        "--context-field",
        help="row field containing {before: text, after: text}; defaults to checkpoint field, "
        "or pass an empty string to disable context for a matched control",
    )
    predict.add_argument(
        "--context-side",
        choices=("both", "previous"),
        help="neighbor sides to encode; defaults to checkpoint policy, or both for older checkpoints",
    )
    predict.add_argument(
        "--max-length",
        type=int,
        help="total encoder window cap including special/register tokens; defaults to the "
        "saved context capacity, or the encoder capacity for older checkpoints",
    )
    predict.add_argument(
        "--regex-config",
        help="pii-regex-tag/v1 rules to apply after prediction "
        "(scripts/pii_regex_tags_v1.json); omitted means no regex step",
    )
    predict.add_argument(
        "--regex-ontology",
        default="ont3",
        help="label set the regex rules tag with (default ont3)",
    )
    predict.add_argument(
        "--regex-policy",
        help="pii-regex-policy/v1 arbitration for this model "
        "(scripts/pii_regex_policy_ont3_v1.json); without one the model keeps its spans",
    )
    predict.add_argument(
        "--regex-mode",
        choices=("skip", "add", "detail", "score"),
        help="override the policy's mode: skip, add, detail (name a model span "
        "rather than assert one) or score",
    )
    predict.add_argument(
        "--merge-adjacent-types",
        default="",
        metavar="TYPE[,TYPE...]",
        help=(
            "output contract: merge predicted spans of these types that are separated only by "
            "whitespace into one maximal span (person_name,organization); empty disables"
        ),
    )
    predict.add_argument(
        "--register-language",
        choices=("row", "unknown"),
        default="row",
        help="value for a language conditioning register: the row's own language, which is "
        "what deployment has, or unspecified to measure the fallback (default: row)",
    )
    predict.add_argument(
        "--domain-posteriors",
        help="sidecar written by pii_domain_model.py assign, joined to rows by text hash; "
        "required when the checkpoint conditions on a learned domain",
    )
    predict.add_argument(
        "--register-domain",
        choices=("available", "segment", "document", "none"),
        default="available",
        help="which learned-domain inputs the scored deployment supplies: whatever the "
        "sidecar carries, only the sentence classifier, only the document classifier, or "
        "neither (default: available)",
    )
    parser.add_argument(
        "--trim-edge-punctuation",
        action="store_true",
        help=(
            "output contract: drop enclosing punctuation and whitespace a span absorbed from a glued "
            "token (trailing period only for numeric-like types; url/username/email/credential/"
            "ip_address untouched)"
        ),
    )
    score = subcommands.add_parser("score")
    score.add_argument("--gold", required=True)
    score.add_argument("--bcp47", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--predicate-spec", required=True)
    score.add_argument("--subclass-spec")
    score.add_argument(
        "--reference-form-projection",
        choices=("all", "routine"),
        default="all",
        help="routine requires complete reviewed reference_form sidecars and excludes bare pronouns",
    )
    score.add_argument("--output", required=True)
    score.add_argument("--limit", type=int, default=0)
    score.add_argument(
        "--id-list",
        help="score only these newline-delimited row ids, preserving gold order",
    )
    args = parser.parse_args()
    {"predict": cmd_predict, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
