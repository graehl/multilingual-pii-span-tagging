"""Re-placeholderize exact-span rows and deterministically realize new surfaces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

try:
    from pii_instantiate_transport import (
        Filler,
        entity_realization_seed,
        name_mode_groups,
    )
    from pii_locale_render import LocaleRenderer
    from pii_seed_tree import SeedTree
    from pii_surface_audit import normalize_value, parse_span
    from pii_surface_pool import SurfacePool

    from pii_surface.mix_policy import SurfaceMixPolicy
    from pii_surface.predicate_pool import (
        PredicateSurfaceDraw,
        PredicateSurfacePolicy,
        PredicateSurfacePool,
    )
except ModuleNotFoundError:  # Imported as scripts.pii_surface.rerealize in tests.
    from scripts.pii_instantiate_transport import (
        Filler,
        entity_realization_seed,
        name_mode_groups,
    )
    from scripts.pii_locale_render import LocaleRenderer
    from scripts.pii_seed_tree import SeedTree
    from scripts.pii_surface.mix_policy import SurfaceMixPolicy
    from scripts.pii_surface.predicate_pool import (
        PredicateSurfaceDraw,
        PredicateSurfacePolicy,
        PredicateSurfacePool,
    )
    from scripts.pii_surface_audit import normalize_value, parse_span
    from scripts.pii_surface_pool import SurfacePool

PLACEHOLDER = re.compile(r"\[([A-Z0-9_]+)_(\d+)\]")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def canonical_tag(label: str) -> str:
    tag = re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")
    if not tag:
        raise ValueError(f"cannot form a placeholder tag from label {label!r}")
    return tag


def surface_generator_receipt(surface_generator: Any | None, attempt_rate: float) -> dict[str, Any] | None:
    if surface_generator is None:
        return None
    return {
        **surface_generator.receipt(),
        "configured_attempt_rate": attempt_rate,
        "gate": "deterministic-per-linked-entity-v1",
    }


def open_jsonl(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _predicate_sidecars(
    row: Mapping[str, Any],
    text: str,
    primary_spans: set[tuple[int, int, str]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    raw_predicates = row.get("predicate_spans", ())
    if not isinstance(raw_predicates, (list, tuple)):
        raise ValueError("row predicate_spans must be a list when present")
    result = {}
    for predicate_index, predicate_span in enumerate(raw_predicates):
        where = f"predicate_spans[{predicate_index}]"
        if not isinstance(predicate_span, dict) or set(predicate_span) not in (
            {"start", "end", "type", "attrs"},
            {"start", "end", "type", "attrs", "objective_weights"},
        ):
            raise ValueError(f"{where} has invalid fields")
        start, end, primary_type = (
            predicate_span["start"],
            predicate_span["end"],
            predicate_span["type"],
        )
        identity = (start, end, primary_type)
        if identity not in primary_spans:
            raise ValueError(f"{where} does not match an activating primary span")
        if identity in result:
            raise ValueError(f"duplicate predicate sidecar for primary span {identity!r}")
        attrs = predicate_span["attrs"]
        if not isinstance(attrs, dict) or not attrs:
            raise ValueError(f"{where}.attrs must be a nonempty object")
        relative_attrs = {}
        for predicate, intervals in attrs.items():
            if not isinstance(predicate, str) or not predicate or not isinstance(intervals, list):
                raise ValueError(f"{where}.attrs must map predicate names to interval lists")
            normalized = []
            for interval_index, interval in enumerate(intervals):
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or isinstance(interval[0], bool)
                    or not isinstance(interval[0], int)
                    or isinstance(interval[1], bool)
                    or not isinstance(interval[1], int)
                ):
                    raise ValueError(
                        f"{where}.attrs.{predicate}[{interval_index}] must be an integer interval"
                    )
                interval_start, interval_end = interval
                if not start <= interval_start < interval_end <= end:
                    raise ValueError(f"{where}.attrs.{predicate}[{interval_index}] escapes the primary span")
                if normalized and normalized[-1][1] > interval_start - start:
                    raise ValueError(f"{where}.attrs.{predicate} intervals overlap or are unsorted")
                normalized.append([interval_start - start, interval_end - start])
            relative_attrs[predicate] = normalized
        weights = predicate_span.get("objective_weights")
        if weights is not None:
            if not isinstance(weights, dict) or not {"O", "other"} <= set(weights):
                raise ValueError(f"{where}.objective_weights must contain O and other")
            if unknown := set(weights) - {"O", "other", *attrs}:
                raise ValueError(f"{where}.objective_weights names unknown attrs: {sorted(unknown)}")
            for name, value in weights.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{where}.objective_weights.{name} must be nonnegative")
        result[identity] = {
            "attrs": relative_attrs,
            "objective_weights": dict(weights) if weights is not None else None,
        }
    return result


def _slot_predicate_children(
    slot_id: str,
    attrs: Mapping[str, list[list[int]]],
    span_length: int,
) -> list[dict[str, Any]]:
    intervals = sorted(
        (start, end, predicate)
        for predicate, predicate_intervals in attrs.items()
        for start, end in predicate_intervals
        if (start, end) != (0, span_length)
    )
    return [
        {
            "slot_id": f"{slot_id}_{position}",
            "predicate": predicate,
            "source_interval": [start, end],
        }
        for position, (start, end, predicate) in enumerate(intervals, 1)
    ]


def _remap_subclass_spans(
    subclass_spans: Any,
    carrier_remap: Mapping[tuple[int, int], tuple[int, int, bool]],
    row_id: str,
) -> list[dict[str, Any]]:
    """Move carrier-anchored subclass supervision onto the realized text.

    A subclass span names an interval inside a primary carrier: which part of a
    person_name is the given name, how coarse a location is. Realization rewrites
    the text, so the carrier moves and every absolute offset in the sidecar goes
    stale. Each carrier must therefore be looked up in the remap and its interval
    re-expressed relative to the carrier's new position.

    When the carrier's own surface was replaced, the interval is not stale but
    meaningless: a freshly drawn organization has no relationship to the old
    string's internal structure. That supervision is dropped rather than moved,
    because relocating it would teach a component boundary that is not there.
    """
    if not subclass_spans:
        return []
    remapped = []
    for index, span in enumerate(subclass_spans):
        where = f"subclass_spans[{index}] of row {row_id!r}"
        try:
            carrier = (int(span["carrier_start"]), int(span["carrier_end"]))
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{where} is not a carrier-anchored interval: {error}") from error
        placed = carrier_remap.get(carrier)
        if placed is None:
            raise ValueError(f"{where} names carrier {carrier} that is not a primary span")
        if not carrier[0] <= start < end <= carrier[1]:
            raise ValueError(f"{where} escapes its carrier {carrier}")
        new_carrier_start, new_carrier_end, surface_changed = placed
        if surface_changed:
            continue
        remapped.append(
            {
                **span,
                "carrier_start": new_carrier_start,
                "carrier_end": new_carrier_end,
                "start": new_carrier_start + (start - carrier[0]),
                "end": new_carrier_start + (end - carrier[0]),
            }
        )
    return remapped


def _remap_ignored_spans(ignored_spans, carrier_remap):
    """Keep ignored annotation extents aligned after primary-span edits."""
    carriers = sorted(carrier_remap.items())

    def position(offset):
        shift = 0
        for (old_start, old_end), (new_start, new_end, changed) in carriers:
            if offset <= old_start:
                return offset + shift
            if offset < old_end:
                if changed:
                    raise ValueError("ignored-span boundary falls inside a replaced primary span")
                return new_start + offset - old_start
            shift = new_end - old_end
        return offset + shift

    return [[position(start), position(end), label] for start, end, label in ignored_spans]


def placeholderize_row(
    row: dict[str, Any],
    *,
    language_hint: str | None = None,
    link_repeated_entities: bool = True,
    opaque_entity_slots: bool = False,
) -> dict[str, Any]:
    text = row.get("text")
    spans = row.get("spans")
    if not isinstance(text, str) or not isinstance(spans, list):
        raise ValueError("row must contain string text and list spans")
    declared_language = row.get("lang")
    if declared_language is not None and not isinstance(declared_language, str):
        raise ValueError("row lang must be a string when present")
    if declared_language and language_hint and declared_language != language_hint:
        raise ValueError(f"row language {declared_language!r} conflicts with input hint {language_hint!r}")
    language = declared_language or language_hint
    if not language:
        raise ValueError("row has no language and its input has no language hint")

    parsed = []
    for raw in spans:
        span = parse_span(raw)
        if span is None or span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError(f"invalid span in row {row.get('id')!r}: {raw!r}")
        parsed.append(span)
    parsed.sort(key=lambda span: (span.start, span.end, span.label))
    for left, right in zip(parsed, parsed[1:]):
        if left.end > right.start:
            raise ValueError(f"overlapping spans in row {row.get('id')!r}: {left!r}, {right!r}")
    primary_spans = {(span.start, span.end, span.label) for span in parsed}
    predicate_sidecars = _predicate_sidecars(row, text, primary_spans)

    parts = []
    position = 0
    next_index = 1
    linked_indices: dict[tuple[str, str, str], int] = {}
    placeholder_map: dict[str, dict[str, Any]] = {}
    for span in parsed:
        value = text[span.start : span.end]
        tag = canonical_tag(span.label)
        predicate_sidecar = predicate_sidecars.get((span.start, span.end, span.label))
        identity = (
            tag,
            normalize_value(value),
            canonical_hash(predicate_sidecar) if predicate_sidecar is not None else "",
        )
        index = linked_indices.get(identity) if link_repeated_entities else None
        if index is None:
            placeholder_tag = "T" if opaque_entity_slots else tag
            while f"[{placeholder_tag}_{next_index}]" in text:
                next_index += 1
            index = next_index
            next_index += 1
            if link_repeated_entities:
                linked_indices[identity] = index
        placeholder_tag = "T" if opaque_entity_slots else tag
        placeholder = f"[{placeholder_tag}_{index}]"
        parts.extend((text[position : span.start], placeholder))
        slot_id = f"T{index}"
        predicate_attrs = predicate_sidecar["attrs"] if predicate_sidecar is not None else {}
        metadata = placeholder_map.setdefault(
            placeholder,
            {
                "tag": tag,
                "label": span.label,
                "primary_type": span.label,
                "slot_id": slot_id,
                "slot_index": index,
                "source_value": value,
                "source_value_sha256": sha256_text(value),
                "source_spans": [],
                "occurrence_count": 0,
                "predicate_attrs": predicate_attrs,
                "predicate_children": _slot_predicate_children(
                    slot_id,
                    predicate_attrs,
                    span.end - span.start,
                ),
                "objective_weights": (
                    predicate_sidecar["objective_weights"] if predicate_sidecar is not None else None
                ),
                "semantic_seed_provenance": row.get("predicate_seed"),
                "surface_origin": row.get("surface_origin"),
                "opaque_slot": opaque_entity_slots,
            },
        )
        metadata["source_spans"].append([span.start, span.end])
        metadata["occurrence_count"] += 1
        position = span.end
    parts.append(text[position:])
    placeholder_text = "".join(parts)
    return {
        "id": str(row.get("id", "")),
        "lang": language,
        "text": placeholder_text,
        "ph_map": placeholder_map,
        "source_row_sha256": canonical_hash(row),
        "source_text_sha256": sha256_text(text),
        "placeholder_text_sha256": sha256_text(placeholder_text),
    }


def _profile_weight(
    profile: Mapping[str, Any] | None,
    predicate: str,
    *,
    positive: bool,
) -> float:
    if profile is None:
        return 1.0
    fallback = "other" if positive else "O"
    return float(profile.get(predicate, profile[fallback]))


def _realized_predicate_span(
    info: Mapping[str, Any],
    draw: PredicateSurfaceDraw | None,
    policy: PredicateSurfacePolicy,
    start: int,
    end: int,
) -> dict[str, Any] | None:
    carrier_attrs = info.get("predicate_attrs") or {}
    carrier_weights = info.get("objective_weights")
    attrs: dict[str, list[list[int]]] = {}
    channel_weights: dict[str, float] = {}
    if draw is None:
        for predicate, intervals in carrier_attrs.items():
            if predicate in policy.contextual_predicates:
                realized_intervals = [[start, end]] if intervals else []
            else:
                realized_intervals = [
                    [start + interval_start, start + interval_end]
                    for interval_start, interval_end in intervals
                ]
            attrs[predicate] = realized_intervals
            channel_weights[predicate] = _profile_weight(
                carrier_weights,
                predicate,
                positive=bool(intervals),
            )
    else:
        entry = draw.entry
        for predicate, intervals in entry.intrinsic_attrs.items():
            attrs[predicate] = [
                [start + interval_start, start + interval_end] for interval_start, interval_end in intervals
            ]
            profile = carrier_weights if predicate in carrier_attrs else entry.objective_weights
            channel_weights[predicate] = _profile_weight(
                profile,
                predicate,
                positive=bool(intervals),
            )
        for predicate, intervals in carrier_attrs.items():
            if predicate not in policy.contextual_predicates:
                continue
            attrs[predicate] = [[start, end]] if intervals else []
            channel_weights[predicate] = _profile_weight(
                carrier_weights,
                predicate,
                positive=bool(intervals),
            )
    if not attrs:
        return None
    return {
        "start": start,
        "end": end,
        "type": info["primary_type"],
        "attrs": attrs,
        "objective_weights": {"O": 0.0, "other": 0.0, **channel_weights},
    }


def _reoffset_predicate_span(
    info: Mapping[str, Any],
    start: int,
    end: int,
) -> dict[str, Any] | None:
    """Move a carrier's predicate sidecar onto its realized position.

    This is the no-predicate-pool path. With a pool the draw supplies attributes
    matched to the surface it chose, and `_realized_predicate_span` handles it.
    Without one the carrier's own attribute intervals are all there is, so they
    are re-expressed against the realized offsets and nothing else changes.
    """
    attrs = info.get("predicate_attrs") or {}
    if not attrs:
        return None
    span = {
        "start": start,
        "end": end,
        "type": info["primary_type"],
        "attrs": {
            predicate: [
                [start + interval_start, start + interval_end] for interval_start, interval_end in intervals
            ]
            for predicate, intervals in attrs.items()
        },
    }
    weights = info.get("objective_weights")
    if weights is not None:
        span["objective_weights"] = dict(weights)
    return span


def realize_placeholder_row(
    source_row: dict[str, Any],
    placeholder_row: dict[str, Any],
    *,
    filler: Filler,
    seed_tree: SeedTree,
    realization_index: int,
    materialization_version: str,
    partition_role: str,
    surface_generator: Any | None = None,
    surface_generator_rate: float = 1.0,
    predicate_surface_pool: PredicateSurfacePool | None = None,
    predicate_surface_policy: PredicateSurfacePolicy | None = None,
) -> dict[str, Any]:
    if not 0.0 <= surface_generator_rate <= 1.0:
        raise ValueError("surface_generator_rate must be in [0, 1]")
    if surface_generator is None and surface_generator_rate != 1.0:
        raise ValueError("surface_generator_rate requires a surface_generator")
    if (predicate_surface_pool is None) != (predicate_surface_policy is None):
        raise ValueError("predicate surface pool and policy must be supplied together")
    language = placeholder_row["lang"]
    carrier_id = placeholder_row["id"] or placeholder_row["source_row_sha256"][:20]
    ph_map = placeholder_row["ph_map"]
    document_seed = seed_tree.fork(
        "surface-rerealizer",
        language,
        carrier_id,
        realization_index,
    )
    filler.begin_document(ph_map, seed=document_seed)
    if filler.document_rejection_reasons:
        raise ValueError(
            f"row {carrier_id!r} rejected by surface policy: " + ", ".join(filler.document_rejection_reasons)
        )
    name_groups = name_mode_groups(ph_map)
    text = placeholder_row["text"]
    parts = []
    spans = []
    span_provenance = []
    realized_predicates = []
    position = 0
    output_length = 0
    generated_values: dict[str, tuple[str, str] | None] = {}
    predicate_draws: dict[str, PredicateSurfaceDraw | None] = {}
    predicate_receipts: dict[str, dict[str, Any]] = {}
    generator_attempts: dict[str, bool] = {}
    generator_attempt_rates: dict[str, float] = {}
    surface_mix: dict[str, dict[str, Any]] = {}
    # Where each original primary span ended up, and whether its surface actually
    # changed. Subclass sidecars are anchored to a carrier by absolute offset, so
    # rewriting the text invalidates them unless they are moved with the carrier.
    carrier_remap: dict[tuple[int, int], tuple[int, int, bool]] = {}
    occurrence_cursor: dict[str, int] = {}
    for match in PLACEHOLDER.finditer(text):
        placeholder = match.group(0)
        info = ph_map.get(placeholder)
        if info is None:
            continue
        tag = info["tag"]
        source_value = info["source_value"]
        entity_seed = entity_realization_seed(
            seed_tree,
            language,
            carrier_id,
            realization_index,
            tag,
            source_value,
            entity_key=placeholder,
        )
        cell = filler.surface_policy.for_tag(language, tag) if filler.surface_policy else None
        name_group = name_groups.get(placeholder)
        name_mode = None
        if name_group is not None:
            mode_seed = seed_tree.fork(
                "surface-rerealizer",
                language,
                carrier_id,
                realization_index,
                "name-script-mode",
                *name_group,
            )
            native_rate = cell.native_name_rate if cell else 0.80
            name_mode = "native" if random.Random(mode_seed).random() < native_rate else "latin-kept"
        conditioned = generated_values.get(placeholder)
        if placeholder not in generated_values:
            if predicate_surface_pool is not None:
                draw, draw_receipt = predicate_surface_pool.draw(
                    language,
                    info["primary_type"],
                    info.get("predicate_attrs") or {},
                    random.Random(entity_seed),
                    policy=predicate_surface_policy,
                )
                predicate_draws[placeholder] = draw
                predicate_receipts[placeholder] = draw_receipt
                if draw is None:
                    conditioned = (source_value, "orig:predicate-surface-no-compatible-v1")
                else:
                    conditioned = (draw.entry.value, draw.entry.provenance)
                surface_mix[placeholder] = {
                    "mode": "predicate-aware-primary-pool-v1",
                    **draw_receipt,
                }
            elif cell is not None and cell.fresh_faker_beta is not None:
                if surface_generator_rate != 1.0:
                    raise ValueError(
                        "hierarchical alpha/beta surface mixing requires global generator rate 1.0"
                    )
                beta = cell.fresh_faker_beta
                alpha = cell.replacement_char_alpha
                if alpha is None:
                    raise ValueError(f"{language}/{tag}: hierarchical mix is missing replacement_char_alpha")
                if beta < 1 and alpha > 0 and surface_generator is None:
                    raise ValueError(
                        f"{language}/{tag}: positive character alpha requires a surface generator"
                    )
                if (
                    beta < 1
                    and alpha < 1
                    and (filler.surface_pool is None or not filler.surface_pool.distinct_count(language, tag))
                ):
                    raise ValueError(
                        f"{language}/{tag}: positive empirical replacement share requires pool entries"
                    )
                gate_parts = (
                    "surface-rerealizer",
                    language,
                    carrier_id,
                    realization_index,
                    tag,
                    source_value,
                    placeholder,
                )
                faker_draw = random.Random(seed_tree.fork(*gate_parts, "outer-faker-beta")).random()
                if faker_draw < beta:
                    branch = "fresh_faker"
                    conditioned = None
                else:
                    char_draw = random.Random(seed_tree.fork(*gate_parts, "replacement-char-alpha")).random()
                    if char_draw < alpha:
                        branch = "character_generator"
                        if surface_generator is None:
                            raise ValueError(f"{language}/{tag}: selected character branch has no generator")
                        conditioned = surface_generator.generate(
                            language,
                            tag,
                            text[: match.start()],
                            text[match.end() :],
                            seed=entity_seed,
                        )
                        if conditioned is None:
                            raise ValueError(
                                f"{language}/{tag}: selected character generator has no supported draw"
                            )
                    else:
                        branch = "exact_empirical"
                        conditioned = None
                surface_mix[placeholder] = {
                    "mode": "hierarchical-faker-char-empirical-v1",
                    "fresh_faker_beta": beta,
                    "replacement_char_alpha": alpha,
                    "configured_branch_probabilities": {
                        "fresh_faker": beta,
                        "character_generator": (1 - beta) * alpha,
                        "exact_empirical": (1 - beta) * (1 - alpha),
                    },
                    "selected_branch": branch,
                }
            elif surface_generator is not None:
                generator_gate_seed = seed_tree.fork(
                    "surface-rerealizer",
                    language,
                    carrier_id,
                    realization_index,
                    tag,
                    source_value,
                    placeholder,
                    "context-generator-gate",
                )
                attempted = random.Random(generator_gate_seed).random() < surface_generator_rate
                if attempted:
                    conditioned = surface_generator.generate(
                        language,
                        tag,
                        text[: match.start()],
                        text[match.end() :],
                        seed=entity_seed,
                    )
                generator_attempts[placeholder] = attempted
                generator_attempt_rates[placeholder] = surface_generator_rate
            generated_values[placeholder] = conditioned
        if conditioned is None:
            selected_branch = surface_mix.get(placeholder, {}).get("selected_branch")
            surface_route = {
                "fresh_faker": "fresh-faker",
                "exact_empirical": "exact-empirical",
            }.get(selected_branch)
            filler_kwargs = {
                "seed": entity_seed,
                "name_mode": name_mode,
                "entity_key": placeholder,
            }
            if surface_route is not None:
                filler_kwargs["surface_route"] = surface_route
            value, generator = filler.fill(tag, source_value, **filler_kwargs)
        else:
            value, generator = conditioned
        occurrence = occurrence_cursor.get(placeholder, 0)
        occurrence_cursor[placeholder] = occurrence + 1
        source_start, source_end = info["source_spans"][occurrence]
        # Linking normalizes mention identity, not the spelling to retain when
        # no replacement is drawn. Preserve each original occurrence verbatim.
        if generator.startswith("orig") and value == source_value:
            value = source_row["text"][source_start:source_end]
        prefix = text[position : match.start()]
        parts.extend((prefix, value))
        output_length += len(prefix)
        start = output_length
        output_length += len(value)
        end = output_length
        spans.append([start, end, info["label"]])
        carrier_remap[(source_start, source_end)] = (
            start,
            end,
            value != source_row["text"][source_start:source_end],
        )
        span_provenance.append(
            {
                "start": start,
                "end": end,
                "label": info["label"],
                "placeholder": placeholder,
                "generator": generator,
                "source_value_sha256": info["source_value_sha256"],
                "entity_seed": entity_seed,
                "slot_id": info["slot_id"],
                "primary_type": info["primary_type"],
                "surface_origin": (
                    predicate_draws[placeholder].entry.surface_origin
                    if predicate_draws.get(placeholder) is not None
                    else info.get("surface_origin")
                ),
                "predicate_sidecar": {
                    "attrs": info.get("predicate_attrs") or {},
                    "children": info.get("predicate_children") or [],
                    "objective_weights": info.get("objective_weights"),
                    "semantic_seed_provenance": info.get("semantic_seed_provenance"),
                    "source_surface_origin": info.get("surface_origin"),
                },
                **(
                    {"predicate_surface_draw": predicate_receipts[placeholder]}
                    if placeholder in predicate_receipts
                    else {}
                ),
                **({"surface_mix": surface_mix[placeholder]} if placeholder in surface_mix else {}),
                **(
                    {
                        "context_generator_attempted": generator_attempts[placeholder],
                        "context_generator_effective_attempt_rate": generator_attempt_rates[placeholder],
                    }
                    if surface_generator is not None and placeholder in generator_attempts
                    else {}
                ),
                **filler.surface_metadata(tag, generator),
            }
        )
        if predicate_surface_policy is not None:
            realized_predicate = _realized_predicate_span(
                info,
                predicate_draws.get(placeholder),
                predicate_surface_policy,
                start,
                end,
            )
        elif value == source_row["text"][source_start:source_end]:
            # No predicate pool, and this carrier kept its surface, so its own
            # attribute intervals still describe the text and only need moving.
            # A replaced surface has no relationship to the old string's internal
            # structure, so its sidecar is dropped below rather than relocated.
            realized_predicate = _reoffset_predicate_span(info, start, end)
        else:
            realized_predicate = None
        if realized_predicate is not None:
            realized_predicates.append(realized_predicate)
        position = match.end()
    parts.append(text[position:])
    realized_text = "".join(parts)
    if len(spans) != sum(item["occurrence_count"] for item in ph_map.values()):
        raise ValueError(f"row {carrier_id!r} did not realize every placeholder occurrence")
    for start, end, _label in spans:
        if not 0 <= start < end <= len(realized_text):
            raise AssertionError(f"row {carrier_id!r} emitted an invalid span")

    prior_materialization = source_row.get("materialization")
    realized_subclasses = _remap_subclass_spans(source_row.get("subclass_spans"), carrier_remap, carrier_id)
    result = {
        **source_row,
        "lang": language,
        "text": realized_text,
        "spans": spans,
        "span_provenance": span_provenance,
        "materialization": {
            "version": materialization_version,
            "partition_role": partition_role,
            "root_seed": seed_tree.root_seed,
            "document_seed": document_seed,
            "realization_index": realization_index,
            "surface_pool_version": filler.surface_pool.version if filler.surface_pool else None,
            "surface_recipe": (
                filler.surface_policy.receipt() if filler.surface_policy is not None else None
            ),
            "context_surface_generator": (
                surface_generator_receipt(surface_generator, surface_generator_rate)
            ),
            "predicate_surface_pool": (
                predicate_surface_pool.receipt() if predicate_surface_pool is not None else None
            ),
            "predicate_surface_policy": (
                {
                    "policy_version": predicate_surface_policy.version,
                    "sha256": predicate_surface_policy.sha256,
                }
                if predicate_surface_policy is not None
                else None
            ),
        },
        "surface_reboot": {
            "schema_version": 1,
            "source_row_sha256": placeholder_row["source_row_sha256"],
            "source_text_sha256": placeholder_row["source_text_sha256"],
            "placeholder_text_sha256": placeholder_row["placeholder_text_sha256"],
            "placeholder_map_sha256": canonical_hash(ph_map),
            "realized_text_sha256": sha256_text(realized_text),
            "prior_materialization": prior_materialization,
            "link_repeated_entities": any(item["occurrence_count"] > 1 for item in ph_map.values()),
        },
    }
    if predicate_surface_policy is not None or source_row.get("predicate_spans"):
        if realized_predicates:
            result["predicate_spans"] = realized_predicates
        else:
            result.pop("predicate_spans", None)
    # The spread above carried the source row's subclass spans in with their
    # original offsets, which the rewritten text no longer supports.
    if realized_subclasses:
        result["subclass_spans"] = realized_subclasses
    else:
        result.pop("subclass_spans", None)
    if "ignored_spans" in source_row:
        result["ignored_spans"] = _remap_ignored_spans(source_row["ignored_spans"], carrier_remap)
    return result


def parse_input(specification: str) -> tuple[str | None, Path]:
    if "=" not in specification:
        return None, Path(specification)
    language, raw_path = specification.split("=", 1)
    if not language or not raw_path:
        raise ValueError(f"invalid input specification: {specification!r}")
    return language, Path(raw_path)


def parse_string_equalities(specifications: Iterable[str]) -> dict[str, str]:
    equalities = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"expected FIELD=VALUE, got {specification!r}")
        field, expected = specification.split("=", 1)
        if not field or not expected:
            raise ValueError(f"expected nonempty FIELD=VALUE, got {specification!r}")
        if field in equalities:
            raise ValueError(f"duplicate string-equality field: {field}")
        equalities[field] = expected
    return equalities


def iter_selected_rows(
    inputs: Iterable[tuple[str | None, Path]],
    *,
    languages: set[str] | None,
    head_per_language: int | None,
    row_ids: set[str] | None,
    row_string_equals: Mapping[str, str] | None = None,
) -> Iterable[tuple[Path, str | None, int, dict[str, Any]]]:
    selected = Counter()
    for language_hint, path in inputs:
        with open_jsonl(path) as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                language = row.get("lang") or language_hint
                if row_string_equals is not None and any(
                    not isinstance(row.get(field), str) or row[field] != expected
                    for field, expected in row_string_equals.items()
                ):
                    continue
                if languages is not None and language not in languages:
                    continue
                if row_ids is not None and str(row.get("id", "")) not in row_ids:
                    continue
                if head_per_language is not None and selected[language] >= head_per_language:
                    continue
                selected[language] += 1
                yield path, language_hint, line_number, row


def rerealize(
    *,
    inputs: list[tuple[str | None, Path]],
    out: Path,
    report_path: Path,
    surface_pool_path: Path,
    surface_pool_split: str,
    surface_pool_report_path: Path,
    surface_recipe_path: Path,
    locale_profile_path: Path,
    materialization_version: str,
    partition_role: str,
    seed: int,
    realization_index: int = 0,
    languages: set[str] | None = None,
    head_per_language: int | None = None,
    row_ids: set[str] | None = None,
    row_string_equals: Mapping[str, str] | None = None,
    link_repeated_entities: bool = True,
    reject_unlocalized_categorical: bool = False,
) -> dict[str, Any]:
    if out.exists() or report_path.exists():
        raise FileExistsError(f"output already exists: {out if out.exists() else report_path}")
    if realization_index < 0:
        raise ValueError("realization_index must be nonnegative")
    if head_per_language is not None and head_per_language <= 0:
        raise ValueError("head_per_language must be positive")
    if row_ids is not None and not row_ids:
        raise ValueError("row_ids must contain at least one ID when provided")
    if row_ids is not None and "" in row_ids:
        raise ValueError("row_ids cannot contain an empty ID")
    if row_string_equals is not None and not row_string_equals:
        raise ValueError("row_string_equals must contain at least one field when provided")
    pool = SurfacePool.load(surface_pool_path, split=surface_pool_split)
    policy = SurfaceMixPolicy.load(surface_recipe_path)
    seed_tree = SeedTree(seed)
    fillers: dict[str, Filler] = {}
    counts = Counter()
    changed_spans = Counter()
    cell_counts: dict[tuple[str, str], Counter[str]] = {}
    cell_generators: dict[tuple[str, str], Counter[str]] = {}
    matched_row_ids = Counter()
    input_files = {
        str(path): {"sha256": file_sha256(path), "language_hint": language_hint}
        for language_hint, path in inputs
    }
    pool_report = json.loads(surface_pool_report_path.read_text(encoding="utf-8"))
    temporary = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for path, language_hint, line_number, source_row in iter_selected_rows(
                inputs,
                languages=languages,
                head_per_language=head_per_language,
                row_ids=row_ids,
                row_string_equals=row_string_equals,
            ):
                placeholder_row = placeholderize_row(
                    source_row,
                    language_hint=language_hint,
                    link_repeated_entities=link_repeated_entities,
                )
                language = placeholder_row["lang"]
                filler = fillers.get(language)
                if filler is None:
                    filler = Filler(
                        language,
                        seed,
                        surface_pool=pool,
                        reject_unlocalized_categorical=reject_unlocalized_categorical,
                        locale_renderer=LocaleRenderer(language, locale_profile_path),
                        surface_policy=policy,
                    )
                    fillers[language] = filler
                realized = realize_placeholder_row(
                    source_row,
                    placeholder_row,
                    filler=filler,
                    seed_tree=seed_tree,
                    realization_index=realization_index,
                    materialization_version=materialization_version,
                    partition_role=partition_role,
                )
                realized["surface_reboot"]["source_locator"] = {
                    "path": str(path),
                    "line_1based": line_number,
                }
                output.write(json.dumps(realized, ensure_ascii=False, sort_keys=True) + "\n")
                counts[language] += 1
                matched_row_ids[placeholder_row["id"]] += 1
                for span, provenance in zip(realized["spans"], realized["span_provenance"], strict=True):
                    source_sha = provenance["source_value_sha256"]
                    realized_sha = sha256_text(realized["text"][span[0] : span[1]])
                    change = "changed" if source_sha != realized_sha else "unchanged"
                    changed_spans[change] += 1
                    key = (language, canonical_tag(provenance["label"]))
                    cell = cell_counts.setdefault(key, Counter())
                    cell["spans"] += 1
                    cell[change] += 1
                    draw_kind = "empirical" if provenance["pool_entries"] else "non_empirical"
                    cell[f"{draw_kind}_draws"] += 1
                    cell_generators.setdefault(key, Counter())[provenance["generator"]] += 1
            if row_ids is not None:
                missing = sorted(row_ids - matched_row_ids.keys())
                duplicates = sorted(row_id for row_id, count in matched_row_ids.items() if count != 1)
                if missing or duplicates:
                    details = []
                    if missing:
                        details.append(f"missing={','.join(missing)}")
                    if duplicates:
                        details.append(f"non_unique={','.join(duplicates)}")
                    raise ValueError(
                        "row-id selection must match each ID exactly once: " + "; ".join(details)
                    )
            if not counts:
                raise ValueError("selection matched no input rows")
        os.replace(temporary, out)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    cells = []
    for (language, tag), cell in sorted(cell_counts.items()):
        policy_cell = policy.for_tag(language, tag)
        distinct = pool.distinct_count(language, tag)
        cells.append(
            {
                "language": language,
                "tag": tag,
                "spans": cell["spans"],
                "changed": cell["changed"],
                "unchanged": cell["unchanged"],
                "empirical_draws": cell["empirical_draws"],
                "non_empirical_draws": cell["non_empirical_draws"],
                "empirical_draw_rate": cell["empirical_draws"] / cell["spans"],
                "direct_pool_distinct_count": distinct,
                "configured_natural_surface_rate": policy_cell.natural_surface_rate,
                "configured_natural_min_distinct_full_rate": (policy_cell.natural_min_distinct_full_rate),
                "direct_effective_natural_surface_rate": (
                    policy_cell.natural_surface_rate
                    * min(1.0, distinct / policy_cell.natural_min_distinct_full_rate)
                ),
                "natural_count_temperature": policy_cell.natural_count_temperature,
                "generators": dict(sorted(cell_generators[(language, tag)].items())),
            }
        )
    report = {
        "schema_version": 1,
        "materialization_version": materialization_version,
        "partition_role": partition_role,
        "seed": seed,
        "realization_index": realization_index,
        "link_repeated_entities": link_repeated_entities,
        "inputs": input_files,
        "surface_pool": {
            "path": str(surface_pool_path),
            "sha256": file_sha256(surface_pool_path),
            "split": surface_pool_split,
            "version": pool.version,
            "report_path": str(surface_pool_report_path),
            "report_sha256": file_sha256(surface_pool_report_path),
            "source_manifests": pool_report.get("source_manifests"),
            "missing_source_manifests": pool_report.get("missing_source_manifests"),
        },
        "surface_recipe": policy.receipt(),
        "locale_profile": {
            "path": str(locale_profile_path),
            "sha256": file_sha256(locale_profile_path),
        },
        "configuration": {
            "languages": sorted(languages) if languages is not None else None,
            "head_per_language": head_per_language,
            "row_ids": sorted(row_ids) if row_ids is not None else None,
            "row_string_equals": (
                dict(sorted(row_string_equals.items())) if row_string_equals is not None else None
            ),
            "reject_unlocalized_categorical": reject_unlocalized_categorical,
        },
        "rows_by_language": dict(sorted(counts.items())),
        "span_changes": dict(sorted(changed_spans.items())),
        "realized_cells": cells,
        "output": {
            "path": str(out),
            "bytes": out.stat().st_size,
            "sha256": file_sha256(out),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="PATH or LANG=PATH exact-span JSONL input; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--surface-pool", type=Path, required=True)
    parser.add_argument("--surface-pool-split", choices=("train", "audit"), required=True)
    parser.add_argument("--surface-pool-report", type=Path, required=True)
    parser.add_argument("--surface-recipe", type=Path, required=True)
    parser.add_argument("--locale-profile", type=Path, required=True)
    parser.add_argument("--materialization-version", required=True)
    parser.add_argument("--partition-role", choices=("training", "evaluation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--realization-index", type=int, default=0)
    parser.add_argument("--language", action="append")
    parser.add_argument("--head-per-language", type=int)
    parser.add_argument(
        "--row-id",
        action="append",
        help="select this exact input row ID; repeatable and fail-closed",
    )
    parser.add_argument(
        "--row-string-equals",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="select rows whose top-level string field exactly equals VALUE; repeatable with AND semantics",
    )
    parser.add_argument(
        "--link-repeated-entities",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--unlocalized-categorical-policy",
        choices=("keep", "drop"),
        default="keep",
    )
    args = parser.parse_args()
    try:
        report = rerealize(
            inputs=[parse_input(item) for item in args.input],
            out=args.out,
            report_path=args.report,
            surface_pool_path=args.surface_pool,
            surface_pool_split=args.surface_pool_split,
            surface_pool_report_path=args.surface_pool_report,
            surface_recipe_path=args.surface_recipe,
            locale_profile_path=args.locale_profile,
            materialization_version=args.materialization_version,
            partition_role=args.partition_role,
            seed=args.seed,
            realization_index=args.realization_index,
            languages=set(args.language) if args.language else None,
            head_per_language=args.head_per_language,
            row_ids=set(args.row_id) if args.row_id else None,
            row_string_equals=(
                parse_string_equalities(args.row_string_equals) if args.row_string_equals else None
            ),
            link_repeated_entities=args.link_repeated_entities,
            reject_unlocalized_categorical=args.unlocalized_categorical_policy == "drop",
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps({"rows_by_language": report["rows_by_language"], "span_changes": report["span_changes"]})
    )
    print(args.out)
    print(args.report)
