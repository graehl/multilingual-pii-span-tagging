#!/usr/bin/env python
"""Two-ontology token supervision: one encoder, two heads, one correctness map.

A checkpoint trained under the old ontology already knows where spans are and
roughly what they are; only its label vocabulary is obsolete. Dual-head
training keeps that head and adds a head over the new vocabulary, so every
training row supervises both:

* a row annotated in the head's own vocabulary gets ordinary cross-entropy;
* the same row supervises the *other* head through the many:many correctness
  map -- ``-log sum_{k in allowed} p(k)``, the honest statement that one of
  those labels is right without guessing which.

``O`` is exact in the frozen transition map. A versioned successor extension
may instead declare new-only mention types that legacy ``O`` never annotated;
an explicitly marked legacy row then supervises the successor through an
``O``-or-new-type marginal rather than teaching those types false.

A weight schedule moves authority from the old head to the new one. At weight
1 the old head trains as it always did and the new head is a passenger fed only
by the map; at weight 0 the old head is frozen out of the objective entirely
and only the new head trains -- still learning from old-ontology rows, because
their supervision reaches it through the map. Anything between is a convex
blend, so a linear fade is a smooth transfer of authority rather than a cut.
The fade may be compressed into any leading fraction of the run
(``linear:1.0:0.0:0.25``), which is different from shortening the run: the
handover finishes early and the rest of the budget goes to the new head alone.
Once the weight reaches zero it never rises again, so the trainer retires the
old head at that step -- see ``transition_completion_step``.

The map (scripts/pii_v1_v2_correctness_map.py) is stated over label *names*.
This module turns it into the two boolean matrices the loss needs, over the
BIOES inventories the two heads actually carry, and refuses a map that leaves
any entity label without a target -- an empty allowed set would silently
contribute an infinite loss term.

Run this file directly to exercise the matrix construction and the loss on a
small synthetic case.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pii_annotation_conventions import PREFIX_BITS, AnnotationConventions  # noqa: E402
from pii_projector import TAGSET_PATH, Tagset  # noqa: E402
from pii_v1_v2_correctness_map import head_support  # noqa: E402

MAP_SCHEMA = "pii-ontology-v1-v2-correctness-map"
MAP_EXTENSION_SCHEMA = "pii-ontology-v1-v2-correctness-map-extension"
ONTOLOGY_EXTENSION_SCHEMA = "pii-ontology-v2-successor-extension"
COUNT_SCHEMA = "pii-ontology-v1-v2-joint-counts"
COUNT_SCHEMA_VERSION = 1
AFFINE_PROJECTION_SCHEMA = "pii-ontology-cross-tagset-affine-projection"
AFFINE_PROJECTION_SCHEMA_VERSION = 1
OUTSIDE_LABEL = "O"
BIOES_PREFIXES = ("B", "I", "E", "S")
V1_TO_V2 = "v1_to_v2"
V2_TO_V1 = "v2_to_v1"
DEFAULT_P20_CUT = "redaction_20_v1"

# The old head's own vocabulary and the new head's, named as rows carry them.
OLD_LABEL_SPACE = "v1"
NEW_LABEL_SPACE = "v2"


class DualHeadError(ValueError):
    """A map, inventory, or schedule that cannot define the objective."""


def split_bioes(label: str) -> tuple[str, str] | None:
    """``('B', 'person_name')`` for an entity label, ``None`` for ``O``."""
    if label == OUTSIDE_LABEL:
        return None
    prefix, _, node = label.partition("-")
    if prefix not in BIOES_PREFIXES or not node:
        raise DualHeadError(f"{label!r} is neither {OUTSIDE_LABEL} nor a BIOES-prefixed label")
    return prefix, node


def new_space_labels(primary_types: Iterable[str]) -> list[str]:
    """The new head's inventory: ``O`` first, then B/I/E/S per class.

    This is the ordering ``pii_ontology_v2.OntologyV2.bioes_labels`` publishes,
    so a head promoted out of this trainer is row-compatible with every v2
    checkpoint the project already decodes.
    """
    labels = [OUTSIDE_LABEL]
    for primary_type in primary_types:
        labels.extend(f"{prefix}-{primary_type}" for prefix in BIOES_PREFIXES)
    return labels


@dataclass(frozen=True)
class CorrectnessMap:
    """Per-label allowed sets between two BIOES inventories.

    ``old_to_new[i]`` is the boolean mask over new-head rows that an old-head
    row ``i`` permits, and ``new_to_old[j]`` the converse. Both include the
    ``O`` row where the map allows a span to become nothing.
    """

    old_labels: tuple[str, ...]
    new_labels: tuple[str, ...]
    old_to_new: torch.Tensor
    new_to_old: torch.Tensor
    map_sha256: str
    ontology_sha256: str
    map_path: str
    empirical_targets: int
    legacy_outside_unknown_primary_types: tuple[str, ...] = ()
    parent_map_sha256: str | None = None
    parent_ontology_sha256: str | None = None
    parent_new_labels: tuple[str, ...] = ()

    @property
    def old_outside_id(self) -> int:
        return self.old_labels.index(OUTSIDE_LABEL)

    @property
    def new_outside_id(self) -> int:
        return self.new_labels.index(OUTSIDE_LABEL)


@dataclass(frozen=True)
class AffineProjection:
    """One source-normalized transcription between complete BIOES heads."""

    direction: str
    source_labels: tuple[str, ...]
    target_labels: tuple[str, ...]
    direct: torch.Tensor
    p20: torch.Tensor
    coefficients: torch.Tensor
    receipt: dict[str, Any]


def semantic_sha256(value: Any) -> str:
    """Stable hash for a JSON-valued semantic contract."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bioes_classes(labels: tuple[str, ...], description: str) -> tuple[str, ...]:
    if not labels or labels[0] != OUTSIDE_LABEL or labels.count(OUTSIDE_LABEL) != 1:
        raise DualHeadError(f"{description} must start with one {OUTSIDE_LABEL!r} row")
    entity_labels = labels[1:]
    if len(entity_labels) % len(BIOES_PREFIXES):
        raise DualHeadError(f"{description} does not contain complete BIOES groups")
    classes = []
    for offset in range(0, len(entity_labels), len(BIOES_PREFIXES)):
        group = entity_labels[offset : offset + len(BIOES_PREFIXES)]
        prefix, separator, entity_class = group[0].partition("-")
        expected = tuple(f"{boundary}-{entity_class}" for boundary in BIOES_PREFIXES)
        if prefix != "B" or not separator or group != expected:
            raise DualHeadError(f"{description} has malformed BIOES group {group!r}")
        classes.append(entity_class)
    if len(classes) != len(set(classes)):
        raise DualHeadError(f"{description} repeats an entity class")
    return tuple(classes)


def _label_order_sha256(labels: tuple[str, ...]) -> str:
    payload = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_joint_counts(path: str | Path) -> dict[str, Any]:
    """Load one immutable raw-count artifact and bind its consumed bytes."""
    count_path = Path(path)
    data = count_path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    if document.get("schema") != COUNT_SCHEMA or document.get("schema_version") != COUNT_SCHEMA_VERSION:
        raise DualHeadError(
            f"{count_path}: {document.get('schema')!r}/{document.get('schema_version')!r} is not "
            f"{COUNT_SCHEMA!r}/{COUNT_SCHEMA_VERSION}"
        )
    document["_artifact_sha256"] = hashlib.sha256(data).hexdigest()
    return document


def _validate_joint_counts(
    document: dict[str, Any] | None,
    old_labels: tuple[str, ...],
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    support: dict[str, tuple[str, ...]],
    support_origins: dict[str, dict[str, tuple[str, ...]]],
    *,
    map_sha256: str,
    ontology_sha256: str,
    tagset_sha256: str,
    count_path: str,
) -> dict[str, dict[str, int]]:
    if document is None:
        return {source: {target: 0 for target in support[source]} for source in old_classes}
    where = count_path or "joint-count document"
    if document.get("schema") != COUNT_SCHEMA or document.get("schema_version") != COUNT_SCHEMA_VERSION:
        raise DualHeadError(
            f"{where}: {document.get('schema')!r}/{document.get('schema_version')!r} is not "
            f"{COUNT_SCHEMA!r}/{COUNT_SCHEMA_VERSION}"
        )
    if document.get("audit", {}).get("status") != "passed":
        raise DualHeadError(f"{where}: count audit did not pass")
    if document.get("totals", {}).get("off_support_spans") != 0:
        raise DualHeadError(f"{where}: count artifact contains off-support observations")
    if document.get("counting_contract", {}).get("raw_counts") is not True:
        raise DualHeadError(f"{where}: projection requires raw integer counts")
    hard_map = document.get("hard_map", {})
    if hard_map.get("sha256") != map_sha256:
        raise DualHeadError(
            f"{where}: hard-map sha256 {hard_map.get('sha256')!r} does not match {map_sha256!r}"
        )
    expected_support = {source: list(support[source]) for source in old_classes}
    if hard_map.get("accepted_support") != expected_support:
        raise DualHeadError(f"{where}: accepted support does not match the consumed hard map and tagset")
    expected_origins = {
        source: {target: list(support_origins[source][target]) for target in support[source]}
        for source in old_classes
    }
    if hard_map.get("accepted_support_origins") != expected_origins:
        raise DualHeadError(f"{where}: accepted-support origins do not match the consumed hard map")
    head = document.get("v1_head", {})
    if head.get("labels") != list(old_labels):
        raise DualHeadError(f"{where}: v1 head labels or order do not match the source affine")
    if head.get("class_order") != list(old_classes):
        raise DualHeadError(f"{where}: v1 class order does not match the source affine")
    if head.get("label_order_sha256") != _label_order_sha256(old_labels):
        raise DualHeadError(f"{where}: v1 label-order hash is invalid")
    if document.get("v1_tagset", {}).get("sha256") != tagset_sha256:
        raise DualHeadError(f"{where}: v1 tagset hash does not match the consumed tagset")
    ontology = document.get("v2_ontology", {})
    if ontology.get("sha256") != ontology_sha256:
        raise DualHeadError(f"{where}: v2 ontology hash does not match the correctness map")
    if ontology.get("primary_types") != list(new_classes):
        raise DualHeadError(f"{where}: v2 primary-type order does not match the target affine")
    raw_counts = document.get("counts")
    if not isinstance(raw_counts, dict) or list(raw_counts) != list(old_classes):
        raise DualHeadError(f"{where}: count rows do not match the v1 class order")
    counts: dict[str, dict[str, int]] = {}
    for source in old_classes:
        row = raw_counts.get(source)
        if not isinstance(row, dict) or list(row) != list(support[source]):
            raise DualHeadError(f"{where}: count cells for {source!r} do not match hard support")
        if any(type(value) is not int or value < 0 for value in row.values()):
            raise DualHeadError(f"{where}: count cells for {source!r} must be nonnegative integers")
        counts[source] = dict(row)
    if sum(sum(row.values()) for row in counts.values()) != document.get("totals", {}).get("counted_spans"):
        raise DualHeadError(f"{where}: raw count cells do not sum to counted_spans")
    return counts


def _normalized(values: dict[str, float], order: tuple[str, ...]) -> dict[str, float]:
    total = sum(values[target] for target in order)
    if not math.isfinite(total) or total <= 0:
        raise DualHeadError("affine projection produced a row with no finite positive mass")
    return {target: values[target] / total for target in order}


def _direct_distribution(
    counts: dict[str, int],
    support: tuple[str, ...],
    add_k: float,
) -> dict[str, float]:
    total = sum(counts[target] for target in support)
    denominator = total + add_k * len(support)
    return {target: (counts[target] + add_k) / denominator for target in support}


def _p20_forward_distributions(
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    support: dict[str, tuple[str, ...]],
    counts: dict[str, dict[str, int]],
    group_of_old: dict[str, str],
    members_by_group: dict[str, tuple[str, ...]],
    add_k: float,
) -> dict[str, dict[str, float]]:
    target_order = (OUTSIDE_LABEL, *new_classes)
    bridge: dict[str, dict[str, float]] = {}
    for group, members in members_by_group.items():
        structural = tuple(
            target for target in target_order if any(target in support[source] for source in members)
        )
        pooled = {target: sum(counts[source].get(target, 0) for source in members) for target in structural}
        bridge[group] = _direct_distribution(pooled, structural, add_k)
    projected = {}
    for source in old_classes:
        fine_support = support[source]
        row = {target: bridge[group_of_old[source]].get(target, 0.0) for target in fine_support}
        projected[source] = _normalized(row, fine_support)
    return projected


def _p20_reverse_distributions(
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    reverse_support: dict[str, tuple[str, ...]],
    counts: dict[str, dict[str, int]],
    group_of_old: dict[str, str],
    members_by_group: dict[str, tuple[str, ...]],
    add_k: float,
) -> dict[str, dict[str, float]]:
    old_marginals = {source: sum(counts[source].values()) for source in old_classes}
    group_to_old = {
        group: _direct_distribution(
            {source: old_marginals[source] for source in members},
            members,
            add_k,
        )
        for group, members in members_by_group.items()
    }
    projected = {}
    group_order = tuple(members_by_group)
    for source in new_classes:
        fine_support = reverse_support[source]
        supported_groups = tuple(
            group for group in group_order if any(old in fine_support for old in members_by_group[group])
        )
        group_counts = {
            group: sum(counts[old].get(source, 0) for old in members_by_group[group])
            for group in supported_groups
        }
        new_to_group = _direct_distribution(group_counts, supported_groups, add_k)
        row = {
            old: new_to_group[group_of_old[old]] * group_to_old[group_of_old[old]][old]
            for old in fine_support
        }
        projected[source] = _normalized(row, fine_support)
    return projected


def build_affine_projection(
    document: dict[str, Any],
    old_labels: Iterable[str],
    new_labels: Iterable[str],
    *,
    tagset: Tagset,
    direction: str = V1_TO_V2,
    counts_document: dict[str, Any] | None = None,
    add_k: float = 1.0,
    alpha: float = 0.0,
    p20_cut: str = DEFAULT_P20_CUT,
    map_path: str = "",
    map_sha256: str = "",
    count_path: str = "",
    count_sha256: str | None = None,
    tagset_path: str = "",
    tagset_sha256: str = "",
) -> AffineProjection:
    """Build a directional direct/P20 projection over declared hard support."""
    if direction not in (V1_TO_V2, V2_TO_V1):
        raise DualHeadError(f"unknown affine projection direction {direction!r}")
    if not math.isfinite(add_k) or add_k <= 0:
        raise DualHeadError(f"add-k mass must be finite and positive, got {add_k!r}")
    if not math.isfinite(alpha) or alpha < 0:
        raise DualHeadError(f"P20 pseudocount mass must be finite and nonnegative, got {alpha!r}")
    old = tuple(old_labels)
    new = tuple(new_labels)
    old_classes = _bioes_classes(old, "v1 affine labels")
    new_classes = _bioes_classes(new, "v2 affine labels")
    if not map_sha256:
        map_sha256 = str(document.get("_artifact_sha256") or semantic_sha256(document))
    if not tagset_sha256:
        tagset_sha256 = semantic_sha256(
            {"nodes": tagset.nodes, "cuts": tagset.cuts, "sources": tagset.sources}
        )
    ontology_sha256 = str(document.get("ontology", {}).get("sha256", ""))
    if not ontology_sha256:
        raise DualHeadError(f"{map_path or 'correctness map'}: missing ontology sha256")
    if document.get("ontology", {}).get("primary_types") not in (None, list(new_classes)):
        raise DualHeadError("correctness-map ontology primary types do not match the v2 affine")

    forward_support, support_origins = head_support(
        document,
        tagset,
        old_classes,
        new_classes,
    )
    reverse_support = {
        source: tuple(old_class for old_class in old_classes if source in forward_support[old_class])
        for source in new_classes
    }
    starved = [source for source, targets in reverse_support.items() if not targets]
    if starved:
        raise DualHeadError(
            "hard support leaves v2 classes with no reverse source: " + ", ".join(starved[:6])
        )
    counts = _validate_joint_counts(
        counts_document,
        old,
        old_classes,
        new_classes,
        forward_support,
        support_origins,
        map_sha256=map_sha256,
        ontology_sha256=ontology_sha256,
        tagset_sha256=tagset_sha256,
        count_path=count_path,
    )

    try:
        tagset.cut_targets(p20_cut)
        group_of_old = {source: tagset.project_canonical_cut(source, p20_cut) for source in old_classes}
    except ValueError as error:
        raise DualHeadError(str(error)) from None
    groups = tuple(sorted(set(group_of_old.values())))
    members_by_group = {
        group: tuple(source for source in old_classes if group_of_old[source] == group) for group in groups
    }
    p20_cut_sha256 = semantic_sha256(tagset.cuts[p20_cut])
    forward_p20 = _p20_forward_distributions(
        old_classes,
        new_classes,
        forward_support,
        counts,
        group_of_old,
        members_by_group,
        add_k,
    )
    reverse_p20 = _p20_reverse_distributions(
        old_classes,
        new_classes,
        reverse_support,
        counts,
        group_of_old,
        members_by_group,
        add_k,
    )

    if direction == V1_TO_V2:
        source_labels = old
        target_labels = new
        class_support = forward_support
        class_counts = counts
        class_p20 = forward_p20
    else:
        source_labels = new
        target_labels = old
        class_support = reverse_support
        class_counts = {
            source: {target: counts[target].get(source, 0) for target in reverse_support[source]}
            for source in new_classes
        }
        class_p20 = reverse_p20

    target_index = {label: index for index, label in enumerate(target_labels)}
    direct = torch.zeros((len(source_labels), len(target_labels)), dtype=torch.float64)
    p20 = torch.zeros_like(direct)
    coefficients = torch.zeros_like(direct)
    rows = []
    for source_id, source_label in enumerate(source_labels):
        parts = split_bioes(source_label)
        if parts is None:
            edges = ((OUTSIDE_LABEL, 0, 1.0, 1.0, 1.0),)
            observations = 0
            shrinkage = alpha / (add_k + alpha)
            p20_group = None
        else:
            prefix, source_class = parts
            support = class_support[source_class]
            raw = class_counts[source_class]
            direct_row = _direct_distribution(raw, support, add_k)
            observations = sum(raw.values())
            denominator = observations + add_k * len(support) + alpha
            shrinkage = alpha / denominator
            edges = tuple(
                (
                    OUTSIDE_LABEL if target == OUTSIDE_LABEL else f"{prefix}-{target}",
                    raw[target],
                    direct_row[target],
                    class_p20[source_class][target],
                    (raw[target] + add_k + alpha * class_p20[source_class][target]) / denominator,
                )
                for target in support
            )
            p20_group = group_of_old[source_class] if direction == V1_TO_V2 else None
        receipt_edges = []
        for target_label, raw_count, direct_value, p20_value, coefficient in edges:
            target_id = target_index.get(target_label)
            if target_id is None:
                raise DualHeadError(f"projection target {target_label!r} is absent from the target affine")
            direct[source_id, target_id] = direct_value
            p20[source_id, target_id] = p20_value
            coefficients[source_id, target_id] = coefficient
            receipt_edges.append(
                {
                    "target_id": target_id,
                    "target_label": target_label,
                    "raw_count": raw_count,
                    "direct": direct_value,
                    "p20": p20_value,
                    "coefficient": coefficient,
                }
            )
        rows.append(
            {
                "source_id": source_id,
                "source_label": source_label,
                "p20_group": p20_group,
                "observations": observations,
                "p20_shrinkage": shrinkage,
                "direct_sum": float(direct[source_id].sum()),
                "p20_sum": float(p20[source_id].sum()),
                "coefficient_sum": float(coefficients[source_id].sum()),
                "edges": receipt_edges,
            }
        )

    direct_error = float(torch.max(torch.abs(direct.sum(dim=1) - 1.0)))
    p20_error = float(torch.max(torch.abs(p20.sum(dim=1) - 1.0)))
    coefficient_error = float(torch.max(torch.abs(coefficients.sum(dim=1) - 1.0)))
    if max(direct_error, p20_error, coefficient_error) > 1e-12:
        raise DualHeadError(
            "source-normalized affine projection failed its row-sum invariant: "
            f"direct={direct_error:.3g} p20={p20_error:.3g} blended={coefficient_error:.3g}"
        )
    receipt = {
        "schema": AFFINE_PROJECTION_SCHEMA,
        "schema_version": AFFINE_PROJECTION_SCHEMA_VERSION,
        "direction": direction,
        "map": {"path": map_path, "sha256": map_sha256},
        "counts": {
            "path": count_path or None,
            "sha256": count_sha256,
            "omitted_counts_are_zero": counts_document is None,
        },
        "tagset": {"path": tagset_path, "sha256": tagset_sha256},
        "p20": {
            "cut": p20_cut,
            "cut_sha256": p20_cut_sha256,
            "groups": {group: list(members) for group, members in members_by_group.items()},
        },
        "source_labels": {
            "count": len(source_labels),
            "sha256": _label_order_sha256(source_labels),
            "labels": list(source_labels),
        },
        "target_labels": {
            "count": len(target_labels),
            "sha256": _label_order_sha256(target_labels),
            "labels": list(target_labels),
        },
        "add_k": add_k,
        "alpha": alpha,
        "maximum_source_sum_error": {
            "direct": direct_error,
            "p20": p20_error,
            "blended": coefficient_error,
        },
        "forward_hard_support": {source: list(forward_support[source]) for source in old_classes},
        "forward_hard_support_origins": {
            source: {target: list(support_origins[source][target]) for target in forward_support[source]}
            for source in old_classes
        },
        "rows": rows,
    }
    return AffineProjection(
        direction=direction,
        source_labels=source_labels,
        target_labels=target_labels,
        direct=direct,
        p20=p20,
        coefficients=coefficients,
        receipt=receipt,
    )


def load_affine_projection(
    document: dict[str, Any],
    old_labels: Iterable[str],
    new_labels: Iterable[str],
    *,
    map_path: str | Path,
    count_path: str | Path | None = None,
    tagset_path: str | Path = TAGSET_PATH,
    direction: str = V1_TO_V2,
    add_k: float = 1.0,
    alpha: float = 0.0,
    p20_cut: str = DEFAULT_P20_CUT,
) -> AffineProjection:
    """Load and bind all artifacts needed for one affine transcription."""
    tagset_path = Path(tagset_path)
    tagset = Tagset(str(tagset_path))
    counts_document = load_joint_counts(count_path) if count_path is not None else None
    resolved_count_path = None if count_path is None else Path(count_path)
    return build_affine_projection(
        document,
        old_labels,
        new_labels,
        tagset=tagset,
        direction=direction,
        counts_document=counts_document,
        add_k=add_k,
        alpha=alpha,
        p20_cut=p20_cut,
        map_path=str(map_path),
        map_sha256=str(document.get("_artifact_sha256") or file_sha256(map_path)),
        count_path="" if resolved_count_path is None else str(resolved_count_path),
        count_sha256=(
            None
            if counts_document is None or resolved_count_path is None
            else str(counts_document.get("_artifact_sha256") or file_sha256(resolved_count_path))
        ),
        tagset_path=str(tagset_path),
        tagset_sha256=file_sha256(tagset_path),
    )


def project_affine_parameters(
    projection: AffineProjection,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``target = coefficients.T @ source`` to weight and bias rows."""
    if source_weight.ndim != 2 or source_bias.ndim != 1:
        raise DualHeadError("affine source weight/bias must be a matrix and vector")
    if source_weight.shape[0] != len(projection.source_labels) or source_bias.shape[0] != len(
        projection.source_labels
    ):
        raise DualHeadError("affine source tensors do not match the projection's source labels")
    coefficients = projection.coefficients.to(device=source_weight.device, dtype=torch.float32)
    target_weight = coefficients.T @ source_weight.detach().float()
    target_bias = coefficients.T @ source_bias.detach().float()
    return target_weight.to(source_weight.dtype), target_bias.to(source_bias.dtype)


def initialize_affine_classifier(
    target: nn.Linear,
    source: nn.Linear,
    projection: AffineProjection,
) -> None:
    """Overwrite one target classifier from a source-normalized projection."""
    if source.in_features != target.in_features:
        raise DualHeadError("source and target affine classifiers have different input widths")
    if source.out_features != len(projection.source_labels):
        raise DualHeadError("source affine row count does not match the projection")
    if target.out_features != len(projection.target_labels):
        raise DualHeadError("target affine row count does not match the projection")
    if source.bias is None or target.bias is None:
        raise DualHeadError("cross-tagset affine transcription requires weight and bias")
    target_weight, target_bias = project_affine_parameters(
        projection,
        source.weight,
        source.bias,
    )
    with torch.no_grad():
        target.weight.copy_(target_weight)
        target.bias.copy_(target_bias)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous bytes."""
    value = tensor.detach().cpu().contiguous()
    payload = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def affine_projection_tensor_receipt(
    projection: AffineProjection,
    source: nn.Linear,
    target: nn.Linear,
) -> dict[str, Any]:
    """Bind the exact source and initialized target affine tensors."""
    augmented = torch.cat(
        (source.weight.detach().double(), source.bias.detach().double().unsqueeze(1)),
        dim=1,
    )
    row_sums = projection.coefficients.sum(dim=1).unsqueeze(1)
    reconstruction_error = torch.max(torch.abs(row_sums * augmented - augmented), dim=1).values
    return {
        "source_weight_sha256": tensor_sha256(source.weight),
        "source_bias_sha256": tensor_sha256(source.bias),
        "target_weight_sha256": tensor_sha256(target.weight),
        "target_bias_sha256": tensor_sha256(target.bias),
        "source_weight_shape": list(source.weight.shape),
        "target_weight_shape": list(target.weight.shape),
        "source_dtype": str(source.weight.dtype),
        "target_dtype": str(target.weight.dtype),
        "source_augmented_reconstruction_max_abs": float(reconstruction_error.max()),
        "source_augmented_reconstruction_max_abs_by_row": [float(value) for value in reconstruction_error],
        "projection_sha256": semantic_sha256(projection.receipt),
    }


def write_affine_projection_receipt(
    path: str | Path,
    projection: AffineProjection,
    source: nn.Linear,
    target: nn.Linear,
) -> str:
    """Atomically save coefficient and exact-tensor receipts; return file hash."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = dict(projection.receipt)
    document["tensors"] = affine_projection_tensor_receipt(projection, source, target)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return file_sha256(output)


def load_correctness_map(path: str | Path) -> dict[str, Any]:
    map_path = Path(path)
    data = map_path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    annotation_conventions = document.get("annotation_conventions")
    if document.get("schema") == MAP_EXTENSION_SCHEMA:
        if document.get("schema_version") != 1:
            raise DualHeadError(f"{path}: unsupported correctness-map extension version")

        def load_bound_json(field: str, schema: str) -> tuple[dict[str, Any], str, Path]:
            binding = document.get(field)
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise DualHeadError(f"{path}: {field} must contain exactly path and sha256")
            relative = binding["path"]
            expected_sha256 = binding["sha256"]
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise DualHeadError(f"{path}: {field}.path must be a nonempty relative path")
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or not set(expected_sha256) <= set("0123456789abcdef")
            ):
                raise DualHeadError(f"{path}: {field}.sha256 must be a SHA-256 hex digest")
            bound_path = (map_path.parent / relative).resolve()
            bound_data = bound_path.read_bytes()
            observed_sha256 = hashlib.sha256(bound_data).hexdigest()
            if observed_sha256 != expected_sha256:
                raise DualHeadError(f"{path}: {field} hash mismatch: {observed_sha256} != {expected_sha256}")
            bound_document = json.loads(bound_data.decode("utf-8"))
            if bound_document.get("schema") != schema:
                raise DualHeadError(
                    f"{path}: {field} schema {bound_document.get('schema')!r} is not {schema!r}"
                )
            return bound_document, observed_sha256, bound_path

        base, base_sha256, base_path = load_bound_json("base_map", MAP_SCHEMA)
        extension, extension_sha256, extension_path = load_bound_json(
            "ontology_extension", ONTOLOGY_EXTENSION_SCHEMA
        )
        if extension.get("schema_version") != 1:
            raise DualHeadError(f"{extension_path}: unsupported ontology extension version")
        base_ontology = extension.get("base_ontology")
        if not isinstance(base_ontology, dict) or set(base_ontology) != {"sha256"}:
            raise DualHeadError(f"{extension_path}: base_ontology must contain exactly sha256")
        if base_ontology["sha256"] != base.get("ontology", {}).get("sha256"):
            raise DualHeadError(f"{extension_path}: base ontology does not match {base_path}")
        appended = extension.get("append_primary_types")
        if not isinstance(appended, list) or not appended:
            raise DualHeadError(f"{extension_path}: append_primary_types must be a nonempty list")
        appended_names = []
        for index, entry in enumerate(appended):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
                raise DualHeadError(f"{extension_path}: append_primary_types[{index}] needs a name")
            appended_names.append(entry["name"])
        base_primary_types = list(base.get("ontology", {}).get("primary_types", ()))
        if len(set(appended_names)) != len(appended_names) or set(appended_names) & set(base_primary_types):
            raise DualHeadError(f"{extension_path}: appended primary types must be unique and new")
        unknown_types = extension.get("legacy_outside_unknown_primary_types")
        if (
            not isinstance(unknown_types, list)
            or not unknown_types
            or len(set(unknown_types)) != len(unknown_types)
            or not set(unknown_types) <= set(appended_names)
        ):
            raise DualHeadError(
                f"{extension_path}: legacy_outside_unknown_primary_types must be unique appended types"
            )
        ontology_version = extension.get("ontology_version")
        if not isinstance(ontology_version, str) or not ontology_version:
            raise DualHeadError(f"{extension_path}: ontology_version must be nonempty")
        document = copy.deepcopy(base)
        document["ontology"] = {
            "path": str(extension_path),
            "sha256": extension_sha256,
            "ontology_version": ontology_version,
            "primary_types": [*base_primary_types, *appended_names],
        }
        document["legacy_outside_unknown_primary_types"] = unknown_types
        document["_parent_map_sha256"] = base_sha256
        document["_parent_ontology_sha256"] = base_ontology["sha256"]
        document["_parent_primary_types"] = base_primary_types
    elif document.get("schema") != MAP_SCHEMA:
        raise DualHeadError(
            f"{path}: schema {document.get('schema')!r} is neither {MAP_SCHEMA!r} nor "
            f"{MAP_EXTENSION_SCHEMA!r}"
        )
    for field in ("v1_to_v2", "v2_to_v1", "ontology"):
        if field not in document:
            raise DualHeadError(f"{path}: correctness map has no {field!r}")
    # The objective receipt must bind the file actually consumed. The ontology
    # hash inside the document identifies one input to the map, not the map's
    # accepted sets, fallbacks, or empirical-origin decisions.
    document["_artifact_sha256"] = hashlib.sha256(data).hexdigest()
    if annotation_conventions is not None:
        document["annotation_conventions"] = AnnotationConventions(annotation_conventions).document
    return document


def build_correctness_map(
    document: dict[str, Any],
    old_labels: Iterable[str],
    new_labels: Iterable[str],
    *,
    map_path: str = "",
) -> CorrectnessMap:
    """Expand a name-level map into the two BIOES membership matrices.

    Only the map's canonical entries are used. A corpus label carries a v1
    ontology node, which is what ``old_v1`` is keyed by; the per-source-schema
    entries describe a corpus's own surface labels and are already resolved to
    nodes before a row reaches training.
    """
    old = tuple(old_labels)
    new = tuple(new_labels)
    if OUTSIDE_LABEL not in old or OUTSIDE_LABEL not in new:
        raise DualHeadError("both inventories need an O row")
    new_index = {label: index for index, label in enumerate(new)}
    entries = {
        name: entry for name, entry in document["v1_to_v2"].items() if entry.get("kind") == "canonical"
    }
    if not entries:
        raise DualHeadError(f"{map_path or 'correctness map'}: no canonical v1 entries")

    old_to_new = torch.zeros((len(old), len(new)), dtype=torch.bool)
    new_to_old = torch.zeros((len(new), len(old)), dtype=torch.bool)
    old_outside = old.index(OUTSIDE_LABEL)
    old_to_new[old_outside, new_index[OUTSIDE_LABEL]] = True
    new_to_old[new_index[OUTSIDE_LABEL], old_outside] = True
    unknown_primary_types = tuple(document.get("legacy_outside_unknown_primary_types", ()))
    if len(set(unknown_primary_types)) != len(unknown_primary_types):
        raise DualHeadError(f"{map_path or 'correctness map'}: repeated legacy-O unknown type")
    for primary_type in unknown_primary_types:
        for prefix in BIOES_PREFIXES:
            label = f"{prefix}-{primary_type}"
            new_id = new_index.get(label)
            if new_id is None:
                raise DualHeadError(
                    f"{map_path or 'correctness map'}: legacy-O unknown type {primary_type!r} "
                    "is absent from the new inventory"
                )
            old_to_new[old_outside, new_id] = True
            new_to_old[new_id, old_outside] = True

    unmapped: list[str] = []
    for old_id, label in enumerate(old):
        parts = split_bioes(label)
        if parts is None:
            continue
        prefix, node = parts
        entry = entries.get(node)
        if entry is None:
            unmapped.append(label)
            continue
        for target in entry["accepted"]:
            # A span that the map says may correctly be nothing contributes the
            # O row; every other target keeps this label's boundary letter,
            # because the map retypes spans and never moves their edges.
            name = OUTSIDE_LABEL if target == OUTSIDE_LABEL else f"{prefix}-{target}"
            new_id = new_index.get(name)
            if new_id is None:
                continue
            old_to_new[old_id, new_id] = True
            new_to_old[new_id, old_id] = True
    if unmapped:
        raise DualHeadError(
            f"{map_path or 'correctness map'}: {len(unmapped)} old-head label(s) have no map entry, so"
            f" their cross-head target set is undefined: {', '.join(sorted(unmapped)[:6])}"
        )

    starved_old = [old[i] for i in range(len(old)) if not bool(old_to_new[i].any())]
    starved_new = [new[j] for j in range(len(new)) if not bool(new_to_old[j].any())]
    if starved_old or starved_new:
        raise DualHeadError(
            "the correctness map leaves labels with an empty allowed set: "
            f"old={', '.join(starved_old[:6]) or 'none'}; new={', '.join(starved_new[:6]) or 'none'}"
        )

    empirical = sum(
        1
        for entry in entries.values()
        for origin in entry.get("origin", {}).values()
        if origin == "empirical"
    )
    map_sha256 = document.get("_artifact_sha256")
    if not map_sha256:
        # Programmatic callers and unit tests have no file bytes to bind. Give
        # them a stable semantic-document identity without confusing it with
        # the embedded ontology identity.
        map_sha256 = hashlib.sha256(
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return CorrectnessMap(
        old_labels=old,
        new_labels=new,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        map_sha256=str(map_sha256),
        ontology_sha256=str(document.get("ontology", {}).get("sha256", "")),
        map_path=str(map_path),
        empirical_targets=empirical,
        legacy_outside_unknown_primary_types=unknown_primary_types,
        parent_map_sha256=document.get("_parent_map_sha256"),
        parent_ontology_sha256=document.get("_parent_ontology_sha256"),
        parent_new_labels=tuple(
            new_space_labels(document.get("_parent_primary_types", ()))
            if document.get("_parent_primary_types") is not None
            else ()
        ),
    )


def fallback_source_rows(
    document: dict[str, Any],
    old_labels: Iterable[str],
    new_labels: Iterable[str],
) -> list[list[int]]:
    """Which incumbent rows to average into each new row as a starting point.

    The map names one *fallback* target per old category: the class it becomes
    when nothing argues for a more specific reading. Averaging the incumbent
    rows that fall back to a new class gives that class a head that already
    responds to the same evidence, so training starts near the incumbent's
    projected behaviour instead of at noise. A new class no old category falls
    back to gets no sources and keeps its fresh initialization.
    """
    old = tuple(old_labels)
    new = tuple(new_labels)
    new_index = {label: index for index, label in enumerate(new)}
    entries = {
        name: entry for name, entry in document["v1_to_v2"].items() if entry.get("kind") == "canonical"
    }
    sources: list[list[int]] = [[] for _ in new]
    for old_id, label in enumerate(old):
        parts = split_bioes(label)
        if parts is None:
            sources[new_index[OUTSIDE_LABEL]].append(old_id)
            continue
        prefix, node = parts
        entry = entries.get(node)
        if entry is None:
            raise DualHeadError(f"correctness map has no entry for {node!r}")
        fallback = entry["fallback"]
        name = OUTSIDE_LABEL if fallback == OUTSIDE_LABEL else f"{prefix}-{fallback}"
        new_id = new_index.get(name)
        if new_id is not None:
            sources[new_id].append(old_id)
    return sources


def parse_transition_schedule(spec: str) -> tuple[str, float, float, float]:
    """Parse ``constant:W`` or ``linear:START:END[:SPAN]``.

    Returns (kind, start, end, span). ``SPAN`` is the fraction of the
    optimizer-step horizon the move takes: the default 1 spreads it over the
    whole run, and 0.25 completes it in the first quarter and holds ``END`` for
    the rest. Moving the transition earlier is not the same as shortening the
    run, which is why this is a property of the schedule rather than of
    ``--max-steps``.
    """
    fields = str(spec).split(":")
    kind = fields[0]
    span = "1"
    if kind == "constant" and len(fields) == 2:
        weights = (fields[1], fields[1])
    elif kind == "linear" and len(fields) in (3, 4):
        weights = (fields[1], fields[2])
        if len(fields) == 4:
            span = fields[3]
    else:
        raise DualHeadError(
            f"transition schedule {spec!r} is neither 'constant:W' nor 'linear:START:END[:SPAN]'"
        )
    parsed = []
    for weight in weights:
        try:
            value = float(weight)
        except ValueError:
            raise DualHeadError(f"transition schedule {spec!r} has a non-numeric weight") from None
        if not 0.0 <= value <= 1.0:
            raise DualHeadError(f"transition schedule {spec!r}: {value} is outside the convex range [0, 1]")
        parsed.append(value)
    try:
        span_fraction = float(span)
    except ValueError:
        raise DualHeadError(f"transition schedule {spec!r} has a non-numeric horizon span") from None
    if not 0.0 < span_fraction <= 1.0:
        raise DualHeadError(
            f"transition schedule {spec!r}: a horizon span of {span_fraction} is outside (0, 1]"
        )
    return kind, parsed[0], parsed[1], span_fraction


def transition_weight(schedule: tuple[str, float, float, float], step: int, horizon: int) -> float:
    """The authority the old head still holds at this optimizer step."""
    kind, start, end, span = schedule
    if kind == "constant":
        return start
    if horizon <= 0:
        raise DualHeadError("a linear transition schedule requires a positive optimizer-step horizon")
    return start + (end - start) * min(1.0, max(0.0, step / (horizon * span)))


def transition_completion_step(schedule: tuple[str, float, float, float], horizon: int) -> int | None:
    """First optimizer step at which the old head's authority is permanently zero.

    ``None`` when the schedule never reaches zero, so a caller that retires the
    old head can tell "not yet" from "never". A schedule that starts at zero
    retires it before the first step: seeding has already happened by then, and
    a zero-weight head contributes no gradient at any step after it.
    """
    kind, start, end, span = schedule
    if kind == "constant":
        return 0 if start == 0.0 else None
    if end != 0.0:
        return None
    if start == 0.0:
        return 0
    if horizon <= 0:
        raise DualHeadError("a linear transition schedule requires a positive optimizer-step horizon")
    return math.ceil(horizon * span)


@dataclass(frozen=True)
class HeadLossTerms:
    """Summed loss, weighted normalizer, and raw token coverage for one head."""

    total: torch.Tensor
    normalizer: torch.Tensor
    covered: int


def head_entity_labels(
    own_labels: torch.Tensor,
    cross_labels: torch.Tensor,
    *,
    own_o_label_id: int,
    cross_o_label_id: int,
    cross_membership: torch.Tensor | None = None,
) -> torch.Tensor:
    """Collapse direct or mapped supervision to masked entity-vs-O targets.

    Direct labels take the same precedence as :func:`head_loss_terms`. Tokens
    supervised only through the other ontology keep their class-agnostic
    entity identity without inventing one exact target class.
    """
    if own_labels.shape != cross_labels.shape:
        raise DualHeadError(
            "own- and cross-space labels need the same shape for entity supervision: "
            f"{tuple(own_labels.shape)} vs {tuple(cross_labels.shape)}"
        )
    targets = torch.full_like(own_labels, -100)
    direct = own_labels != -100
    targets[direct] = (own_labels[direct] != own_o_label_id).to(targets.dtype)
    mapped = (cross_labels != -100) & ~direct
    if cross_membership is None:
        targets[mapped] = (cross_labels[mapped] != cross_o_label_id).to(targets.dtype)
    elif bool(mapped.any()):
        allowed = cross_membership.to(own_labels.device)[cross_labels[mapped]]
        allows_outside = allowed[:, own_o_label_id]
        allows_entity = allowed.sum(dim=-1) > allows_outside.to(allowed.dtype)
        unambiguous = allows_outside ^ allows_entity
        mapped_targets = torch.full_like(cross_labels[mapped], -100)
        mapped_targets[unambiguous] = allows_entity[unambiguous].to(mapped_targets.dtype)
        targets[mapped] = mapped_targets
    return targets


def bioes_structure_cost_matrix(labels: list[str], *, boundary_cost: float, type_cost: float) -> torch.Tensor:
    """Cost of predicting label j when label i is gold, by decoded-structure flip.

    A typed BIOES label is a position (B, I, E, S, or O) and a type. Predicting a
    different position changes where the decoded span starts or ends (or
    whether it exists), so it costs ``boundary_cost``; a different type costs
    ``type_cost``; both add. ``O`` is its own position and type. The gold label
    costs zero. Used as a softmax-margin: the loss demands at least that much
    logit margin over each alternative, so the objective spends its effort on
    the alternatives that would flip the BIOES decoding rather than on all
    alternatives equally.
    """
    for cost in (boundary_cost, type_cost):
        if not math.isfinite(cost) or cost < 0:
            raise DualHeadError(f"BIOES structure costs must be finite and nonnegative, got {cost}")

    def parts(label: str) -> tuple[str, str]:
        if label == "O":
            return "O", "O"
        position, separator, entity_type = label.partition("-")
        if separator != "-" or position not in {"B", "I", "E", "S"} or not entity_type:
            raise DualHeadError(f"not a typed BIOES label: {label!r}")
        return position, entity_type

    parsed = [parts(label) for label in labels]
    matrix = torch.zeros((len(labels), len(labels)), dtype=torch.float32)
    for i, (gold_position, gold_type) in enumerate(parsed):
        for j, (position, entity_type) in enumerate(parsed):
            if i == j:
                continue
            matrix[i, j] = boundary_cost * float(position != gold_position) + type_cost * float(
                entity_type != gold_type
            )
    return matrix


def bioes_transition_legality(labels: list[str]) -> torch.Tensor:
    """``legal[i, j]``: may label j follow label i under the route's constrained decoder.

    Mirrors ``pii_bioes.constrained_bioes_decode``: O, E-c, and S-c are closed
    states followed by O, any B, or any S; B-c and I-c continue only to I-c or
    E-c of the same type. The context-aware margin uses this to tell a token
    flip the decoder would accept verbatim from one it would have to repair.
    """
    parsed = []
    for label in labels:
        if label == OUTSIDE_LABEL:
            parsed.append(("O", "O"))
            continue
        split = split_bioes(label)
        assert split is not None
        parsed.append(split)
    legal = torch.zeros((len(labels), len(labels)), dtype=torch.bool)
    for i, (position, entity_type) in enumerate(parsed):
        for j, (next_position, next_type) in enumerate(parsed):
            if position in ("O", "E", "S"):
                legal[i, j] = next_position in ("O", "B", "S")
            else:  # B or I: the span continues in the same type
                legal[i, j] = next_position in ("I", "E") and next_type == entity_type
    return legal


def context_scaled_structure_costs(
    base_rows: torch.Tensor,
    own_labels: torch.Tensor,
    legality: torch.Tensor,
    *,
    outside_id: int,
    illegal_scale: float,
) -> torch.Tensor:
    """Per-token cost rows scaled down where a flip would be illegal next to the gold neighbours.

    ``base_rows`` is ``[tokens, labels]`` (the gold row of the cost matrix per
    supervised token), ``own_labels`` the ``[batch, seq]`` gold ids with -100
    for unsupervised tokens. A flip to label j at token t is accepted verbatim
    by the constrained decoder only when ``legal[y_{t-1}, j]`` and
    ``legal[j, y_{t+1}]``; otherwise its cost is multiplied by ``illegal_scale``.
    Unsupervised or sequence-boundary neighbours count as O.
    """
    if not math.isfinite(illegal_scale) or not 0.0 <= illegal_scale <= 1.0:
        raise DualHeadError(f"illegal-flip scale must lie in [0, 1], got {illegal_scale}")
    padded = own_labels.clone()
    padded[padded == -100] = outside_id
    previous = torch.cat([torch.full_like(padded[:, :1], outside_id), padded[:, :-1]], dim=1)
    following = torch.cat([padded[:, 1:], torch.full_like(padded[:, :1], outside_id)], dim=1)
    supervised = own_labels != -100
    legal = legality.to(own_labels.device)
    accepted = legal[previous[supervised]] & legal.t()[following[supervised]]
    scale = torch.where(accepted, torch.ones_like(base_rows), torch.full_like(base_rows, illegal_scale))
    return base_rows * scale


def coarse_incompatibility_matrix(labels: list[str], cut: str = DEFAULT_P20_CUT) -> torch.Tensor:
    """``incompatible[i, j]``: would label j in place of gold i survive the coarse-compatible decoder?

    True when the flip lands outside the gold's coarse bucket under ``cut``
    (another bucket, or O for an entity gold), i.e. a token-level argmax loss
    the bucket-compatible repair cannot absorb. For gold O only a spurious span
    start (B or S) counts; a lone I or E after O is repaired away. Same-bucket
    flips of any position are compatible here: their extent risk is carried by
    the structure-cost margin, this matrix only forecasts coarse damage.
    """
    tagset = Tagset()
    buckets: list[str | None] = []
    for label in labels:
        split = split_bioes(label)
        if split is None:
            buckets.append(None)
            continue
        category = split[1]
        try:
            buckets.append(str(tagset.project_canonical_cut(category, cut)))
        except Exception:  # a successor-only type has no cut entry: it is its own bucket
            buckets.append(category)
    incompatible = torch.zeros((len(labels), len(labels)), dtype=torch.bool)
    for i, gold_bucket in enumerate(buckets):
        for j, bucket in enumerate(buckets):
            if i == j:
                continue
            if gold_bucket is None:
                incompatible[i, j] = labels[j].startswith(("B-", "S-"))
            else:
                incompatible[i, j] = bucket is None or bucket != gold_bucket
    return incompatible


def continuation_label_mask(labels: list[str]) -> torch.Tensor:
    """Boolean ``[labels]`` mask of I- and E- labels (tokens that continue a span)."""
    return torch.tensor([label.startswith(("I-", "E-")) for label in labels], dtype=torch.bool)


def span_risk_gate(
    logits: torch.Tensor,
    own_labels: torch.Tensor,
    incompatible: torch.Tensor,
    continuation: torch.Tensor,
    *,
    outside_id: int,
    threshold: float,
    scale: float,
    weight: float,
) -> torch.Tensor:
    """Per-token weight multiplier forecasting a coarse-mismatched decode of the gold span.

    For each supervised entity token the risk margin is the gold logit minus the
    best coarse-incompatible alternative (detached, see
    ``coarse_incompatibility_matrix``); a gold span's risk is the minimum over
    its tokens and the token that follows it. The multiplier is
    ``1 + weight * sigmoid((threshold - risk) / scale)`` on the span's tokens and
    1 elsewhere, so spans a small logit shift away from a bad decode get more
    objective mass. Returns a ``[batch, seq]`` tensor; costs one masked max per
    token and two segment minima.
    """
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(scale) or scale <= 0:
        raise DualHeadError("risk gate needs a nonnegative weight and a positive scale")
    with torch.no_grad():
        batch, seq = own_labels.shape
        flat = logits.reshape(-1, logits.shape[-1]).float()
        labels = own_labels.reshape(-1)
        supervised = labels != -100
        safe = labels.masked_fill(~supervised, outside_id)
        gold_logit = flat.gather(1, safe.unsqueeze(1)).squeeze(1)
        blocked = flat.masked_fill(~incompatible.to(flat.device)[safe], float("-inf"))
        best_bad = blocked.max(dim=-1).values
        margin = torch.where(
            torch.isfinite(best_bad), gold_logit - best_bad, torch.full_like(gold_logit, float("inf"))
        )
        entity = supervised & (safe != outside_id)
        continues = entity & continuation.to(labels.device)[safe]
        # a segment starts at every token that does not continue a span (B, S, O, unsupervised);
        # Every row starts a segment, including a clipped leading I/E label.
        starts = ~continues
        starts[::seq] = True
        segment = torch.cumsum(starts.to(torch.int64), dim=0) - 1
        span_min = margin.new_full((int(segment[-1].item()) + 1,), float("inf"))
        span_min = span_min.scatter_reduce(0, segment, margin, reduce="amin")
        following = torch.cat([margin[1:], margin.new_full((1,), float("inf"))])
        following[seq - 1 :: seq] = float("inf")
        span_min = span_min.scatter_reduce(0, segment, following, reduce="amin")
        risk = span_min[segment]
        gate = 1.0 + weight * torch.sigmoid((threshold - risk) / scale)
        gate = torch.where(entity, gate, torch.ones_like(gate))
        return gate.reshape(batch, seq)


COARSE_CUT_SCHEMA = "pii-ontology-v3-coarse-cut"


def _label_parts(label: str) -> tuple[str, str]:
    """(position, type) with O as its own position and type."""
    split = split_bioes(label)
    return (OUTSIDE_LABEL, OUTSIDE_LABEL) if split is None else split


def load_bucket_map(path: str | Path) -> dict[str, str]:
    """Entity type -> coarse bucket from a JSON coarse cut (data/pii-annotations/ontology-v3/coarse-cut-v1.json)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != COARSE_CUT_SCHEMA:
        raise DualHeadError(f"{path}: not a {COARSE_CUT_SCHEMA} document")
    bucket_of: dict[str, str] = {}
    for bucket, types in document["buckets"].items():
        for entity_type in types:
            if entity_type in bucket_of:
                raise DualHeadError(f"{path}: {entity_type} appears in two buckets")
            bucket_of[entity_type] = bucket
    return bucket_of


def bucketed_structure_cost_matrix(
    labels: list[str],
    bucket_of: dict[str, str],
    *,
    boundary_cost: float,
    type_cost: float,
    same_bucket_scale: float,
) -> torch.Tensor:
    """Structure cost that punishes severe flips more than minor ones.

    Like ``bioes_structure_cost_matrix``, but a type flip that stays inside the
    gold's coarse bucket (a locality read as an admin_area) costs
    ``type_cost * same_bucket_scale``, because the coarse-compatible decoder
    keeps the span identification; a cross-bucket flip or O costs the full
    ``type_cost``. Position flips cost ``boundary_cost`` as before.
    """
    for value in (boundary_cost, type_cost, same_bucket_scale):
        if not math.isfinite(value) or value < 0:
            raise DualHeadError("costs and the same-bucket scale must be finite and nonnegative")
    parsed = [_label_parts(label) for label in labels]
    unknown = sorted(
        {
            entity_type
            for _, entity_type in parsed
            if entity_type != OUTSIDE_LABEL and entity_type not in bucket_of
        }
    )
    if unknown:
        raise DualHeadError(f"types without a coarse bucket: {unknown}")
    matrix = torch.zeros((len(labels), len(labels)), dtype=torch.float32)
    for i, (gold_position, gold_type) in enumerate(parsed):
        for j, (position, entity_type) in enumerate(parsed):
            if i == j:
                continue
            cost = boundary_cost * float(position != gold_position)
            if entity_type != gold_type:
                same_bucket = (
                    gold_type != OUTSIDE_LABEL
                    and entity_type != OUTSIDE_LABEL
                    and bucket_of[gold_type] == bucket_of[entity_type]
                )
                cost += type_cost * (same_bucket_scale if same_bucket else 1.0)
            matrix[i, j] = cost
    return matrix


def bucketed_incompatibility_matrix(labels: list[str], bucket_of: dict[str, str]) -> torch.Tensor:
    """Risk-gate incompatibility from a JSON coarse cut (see ``coarse_incompatibility_matrix``)."""
    parsed = [_label_parts(label) for label in labels]
    incompatible = torch.zeros((len(labels), len(labels)), dtype=torch.bool)
    for i, (_, gold_type) in enumerate(parsed):
        for j, (position, entity_type) in enumerate(parsed):
            if i == j:
                continue
            if gold_type == OUTSIDE_LABEL:
                incompatible[i, j] = position in ("B", "S")
            else:
                incompatible[i, j] = entity_type == OUTSIDE_LABEL or bucket_of.get(
                    entity_type
                ) != bucket_of.get(gold_type)
    return incompatible


def mapped_structure_cost_rows(cost_matrix: torch.Tensor, cross_membership: torch.Tensor) -> torch.Tensor:
    """Per cross-label cost row: the least cost over the labels the map allows.

    A mapped target admits a set of own-vocabulary labels; an alternative's
    cost is its distance from the nearest admitted label, so admitted labels
    cost zero and the margin is demanded only against structure-flipping ones.
    """
    if cross_membership.dtype != torch.bool or cross_membership.shape[1] != cost_matrix.shape[0]:
        raise DualHeadError("cross membership must be a boolean [old, new] matrix over the cost labels")
    rows = []
    for allowed in cross_membership:
        if not bool(allowed.any()):
            rows.append(torch.zeros(cost_matrix.shape[1], dtype=cost_matrix.dtype))
            continue
        rows.append(cost_matrix[allowed].min(dim=0).values)
    return torch.stack(rows)


def segmentation_allowed(allowed, prefix_masks, label_names):
    """Expand mapped types across permitted letters, never across entity types."""
    if label_names is None or len(label_names) != allowed.shape[-1]:
        raise DualHeadError("internal segmentation requires the head's BIOES label inventory")
    kinds = [label.split("-", 1)[1] if label != "O" else None for label in label_names]
    same_type = torch.tensor(
        [[left is not None and left == right for right in kinds] for left in kinds],
        device=allowed.device,
        dtype=torch.float32,
    )
    bits = torch.tensor(
        [PREFIX_BITS.get(label.split("-", 1)[0], 0) for label in label_names], device=allowed.device
    )
    expanded = (allowed.float() @ same_type > 0) & ((prefix_masks[:, None] & bits) != 0)
    # An accepted-set map may deliberately include O; relaxation must not
    # remove an existing accepted type choice.
    expanded |= allowed & (bits == 0)
    active = prefix_masks != 0
    if torch.any(active & ~expanded.any(dim=-1)):
        raise DualHeadError("internal segmentation has no allowed entity label")
    return torch.where(active[:, None], expanded, allowed)


def segmentation_margin_rows(allowed, structure_cost_matrix):
    """Minimum cost to any acceptable target, without a tokens×labels² tensor."""
    unique, inverse = torch.unique(allowed, dim=0, return_inverse=True)
    return torch.stack([structure_cost_matrix[row].amin(dim=0) for row in unique])[inverse]


def head_loss_terms(
    logits: torch.Tensor,
    own_labels: torch.Tensor,
    cross_labels: torch.Tensor,
    cross_membership: torch.Tensor,
    *,
    own_o_label_id: int | None = None,
    cross_o_label_id: int | None = None,
    o_token_weight: float = 1.0,
    token_objective_weights: torch.Tensor | None = None,
    structure_cost_matrix: torch.Tensor | None = None,
    mapped_structure_cost_rows: torch.Tensor | None = None,
    transition_legality: torch.Tensor | None = None,
    illegal_flip_scale: float = 1.0,
    risk_gate: torch.Tensor | None = None,
    segmentation_prefix_masks: torch.Tensor | None = None,
    label_names: tuple[str, ...] | list[str] | None = None,
    cross_outside_membership: torch.Tensor | None = None,
) -> HeadLossTerms:
    """One head's total loss: own-vocabulary CE plus mapped marginal CE.

    ``own_labels`` holds this head's row ids where the row is annotated in this
    head's vocabulary and -100 elsewhere; ``cross_labels`` holds the *other*
    head's row ids for the tokens this head must learn through the map.
    Trusted O targets and explicitly weighted primary targets may carry
    different relative weights. The returned normalizer is the exact applied
    weight mass, while ``covered`` counts supervised tokens with positive
    objective weight.

    With ``structure_cost_matrix`` (own targets) and
    ``mapped_structure_cost_rows`` (mapped targets) the cross-entropy becomes a
    softmax-margin: each alternative's logit is raised by its BIOES structure
    cost before the log-partition, so the gold path must win by that margin.
    """
    if not math.isfinite(o_token_weight) or o_token_weight <= 0:
        raise DualHeadError(f"O-token weight must be finite and positive, got {o_token_weight}")
    if o_token_weight != 1.0 and (own_o_label_id is None or cross_o_label_id is None):
        raise DualHeadError("non-unit O-token weight requires both heads' O-label ids")
    flat = logits.reshape(-1, logits.shape[-1]).float()
    own = own_labels.reshape(-1)
    cross = cross_labels.reshape(-1)
    outside_rows = None
    if cross_outside_membership is not None:
        if (
            own_labels.ndim != 2
            or cross_outside_membership.shape != (own_labels.shape[0], logits.shape[-1])
            or own_o_label_id is None
            or cross_o_label_id is None
        ):
            raise DualHeadError("corpus outside membership requires [batch, head labels] and O-label ids")
        outside_rows = cross_outside_membership.to(device=flat.device, dtype=torch.bool)
        if not outside_rows[:, own_o_label_id].all():
            raise DualHeadError("corpus outside membership must include O")
    prefix_masks = None
    if segmentation_prefix_masks is not None:
        if segmentation_prefix_masks.shape != own_labels.shape:
            raise DualHeadError("segmentation masks must match label shape")
        prefix_masks = segmentation_prefix_masks.reshape(-1).to(flat.device)
        if torch.any((prefix_masks < 0) | (prefix_masks > 15)):
            raise DualHeadError("invalid BIOES segmentation mask")
        if (structure_cost_matrix is not None and illegal_flip_scale != 1.0) or risk_gate is not None:
            raise DualHeadError(
                "internal segmentation does not support exact-path context margins or risk gates"
            )
        if not bool(prefix_masks.any()):
            prefix_masks = None
    if token_objective_weights is None:
        objective_weights = torch.ones_like(own, dtype=flat.dtype)
    else:
        if token_objective_weights.shape != own_labels.shape:
            raise DualHeadError(
                "token objective weights must match label shape: "
                f"{tuple(token_objective_weights.shape)} vs {tuple(own_labels.shape)}"
            )
        objective_weights = token_objective_weights.reshape(-1).to(
            device=flat.device,
            dtype=flat.dtype,
        )
        if not torch.all(torch.isfinite(objective_weights)) or torch.any(objective_weights < 0):
            raise DualHeadError("token objective weights must be finite and nonnegative")
        unsupervised = (own == -100) & (cross == -100)
        if torch.any(unsupervised & (objective_weights != 0)):
            raise DualHeadError("unsupervised tokens must have zero objective weight")
    if risk_gate is not None:
        if risk_gate.shape != own_labels.shape:
            raise DualHeadError("risk gate must match label shape")
        objective_weights = objective_weights * risk_gate.reshape(-1).to(flat.device, flat.dtype)
    total = flat.new_zeros(())
    normalizer = flat.new_zeros(())
    covered = 0

    direct = (own != -100) & (objective_weights > 0)
    if bool(direct.any()):
        selected = flat[direct]
        direct_targets = own[direct]
        allowed_direct = None
        if prefix_masks is not None:
            allowed_direct = torch.nn.functional.one_hot(direct_targets, flat.shape[-1]).bool()
            allowed_direct = segmentation_allowed(allowed_direct, prefix_masks[direct], label_names)
        if structure_cost_matrix is not None:
            cost_rows = structure_cost_matrix.to(selected.device, selected.dtype)[direct_targets]
            if allowed_direct is not None:
                cost_rows = segmentation_margin_rows(
                    allowed_direct, structure_cost_matrix.to(selected.device, selected.dtype)
                )
            if transition_legality is not None and illegal_flip_scale != 1.0:
                cost_rows = context_scaled_structure_costs(
                    cost_rows,
                    own_labels.masked_fill(~direct.reshape(own_labels.shape), -100),
                    transition_legality,
                    outside_id=own_o_label_id if own_o_label_id is not None else 0,
                    illegal_scale=illegal_flip_scale,
                )
            augmented = selected + cost_rows
        else:
            augmented = selected
        denominator = torch.logsumexp(augmented, dim=-1)
        chosen = selected.gather(1, direct_targets.unsqueeze(1)).squeeze(1)
        if allowed_direct is not None:
            chosen = torch.logsumexp(selected.masked_fill(~allowed_direct, float("-inf")), dim=-1)
        weights = objective_weights[direct]
        if o_token_weight != 1.0:
            weights = torch.where(
                direct_targets == own_o_label_id,
                weights * o_token_weight,
                weights,
            )
        total = total + torch.sum(weights * (denominator - chosen))
        normalizer = normalizer + weights.sum()
        covered += int(direct.sum())

    mapped = (cross != -100) & (own == -100) & (objective_weights > 0)
    if bool(mapped.any()):
        selected = flat[mapped]
        mapped_targets = cross[mapped]
        allowed = cross_membership.to(selected.device)[mapped_targets]
        if outside_rows is not None:
            row_ids = torch.arange(own_labels.shape[0], device=flat.device).repeat_interleave(
                own_labels.shape[1]
            )
            allowed = torch.where(
                (mapped_targets == cross_o_label_id)[:, None], outside_rows[row_ids[mapped]], allowed
            )
        if prefix_masks is not None:
            allowed = segmentation_allowed(allowed, prefix_masks[mapped], label_names)
        if mapped_structure_cost_rows is not None:
            cost_rows = mapped_structure_cost_rows.to(selected.device, selected.dtype)[mapped_targets]
            if prefix_masks is not None or outside_rows is not None:
                if structure_cost_matrix is None:
                    raise DualHeadError("segmentation margins require the head's structure cost matrix")
                cost_rows = segmentation_margin_rows(
                    allowed, structure_cost_matrix.to(selected.device, selected.dtype)
                )
            augmented = selected + cost_rows
        else:
            augmented = selected
        denominator = torch.logsumexp(augmented, dim=-1)
        numerator = torch.logsumexp(selected.masked_fill(~allowed, float("-inf")), dim=-1)
        weights = objective_weights[mapped]
        if o_token_weight != 1.0:
            weights = torch.where(
                mapped_targets == cross_o_label_id,
                weights * o_token_weight,
                weights,
            )
        total = total + torch.sum(weights * (denominator - numerator))
        normalizer = normalizer + weights.sum()
        covered += int(mapped.sum())
    return HeadLossTerms(total=total, normalizer=normalizer, covered=covered)


def head_loss(
    logits: torch.Tensor,
    own_labels: torch.Tensor,
    cross_labels: torch.Tensor,
    cross_membership: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Backward-compatible unweighted summed loss and raw token coverage."""
    terms = head_loss_terms(logits, own_labels, cross_labels, cross_membership)
    return terms.total, terms.covered


def dual_head_loss(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    old_labels: torch.Tensor,
    new_labels: torch.Tensor,
    correctness: CorrectnessMap,
    old_weight: float,
    o_token_weight: float = 1.0,
    token_objective_weights: torch.Tensor | None = None,
    structure_cost_matrix: torch.Tensor | None = None,
    mapped_structure_cost_rows: torch.Tensor | None = None,
    transition_legality: torch.Tensor | None = None,
    illegal_flip_scale: float = 1.0,
    risk_gate: torch.Tensor | None = None,
    segmentation_prefix_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Blend the two heads' objectives by the transition weight.

    Each head is normalized by the number of tokens it actually supervises, so
    the blend weight means what it says: at ``old_weight == 0`` the old head
    contributes no gradient at all, and at 1 the new head does not. The BIOES
    structure-cost margin (see ``bioes_structure_cost_matrix``) applies to the
    new head only: its own targets through ``structure_cost_matrix`` and its
    mapped old-space targets through ``mapped_structure_cost_rows``.
    """
    if not 0.0 <= old_weight <= 1.0:
        raise DualHeadError(f"transition weight {old_weight} is not a convex blend weight")
    old_terms = head_loss_terms(
        old_logits,
        old_labels,
        new_labels,
        correctness.new_to_old,
        segmentation_prefix_masks=segmentation_prefix_masks,
        label_names=correctness.old_labels,
        own_o_label_id=correctness.old_outside_id,
        cross_o_label_id=correctness.new_outside_id,
        o_token_weight=o_token_weight,
        token_objective_weights=token_objective_weights,
    )
    new_terms = head_loss_terms(
        new_logits,
        new_labels,
        old_labels,
        correctness.old_to_new,
        segmentation_prefix_masks=segmentation_prefix_masks,
        label_names=correctness.new_labels,
        own_o_label_id=correctness.new_outside_id,
        cross_o_label_id=correctness.old_outside_id,
        o_token_weight=o_token_weight,
        token_objective_weights=token_objective_weights,
        structure_cost_matrix=structure_cost_matrix,
        mapped_structure_cost_rows=mapped_structure_cost_rows,
        transition_legality=transition_legality,
        illegal_flip_scale=illegal_flip_scale,
        risk_gate=risk_gate,
    )
    old_mean = old_terms.total / old_terms.normalizer if old_terms.covered else old_logits.sum() * 0.0
    new_mean = new_terms.total / new_terms.normalizer if new_terms.covered else new_logits.sum() * 0.0
    loss = old_weight * old_mean + (1.0 - old_weight) * new_mean
    telemetry = {
        "dual_old_weight": float(old_weight),
        "dual_old_loss": float(old_mean.detach()),
        "dual_new_loss": float(new_mean.detach()),
        "dual_old_tokens": float(old_terms.covered),
        "dual_new_tokens": float(new_terms.covered),
        "dual_old_loss_normalizer": float(old_terms.normalizer.detach()),
        "dual_new_loss_normalizer": float(new_terms.normalizer.detach()),
        "dual_o_token_weight": float(o_token_weight),
    }
    return loss, telemetry


def _self_test() -> None:
    """Exercise matrix construction and the two limits of the blend."""
    document = {
        "schema": MAP_SCHEMA,
        "ontology": {"sha256": "test"},
        "v1_to_v2": {
            "city": {
                "kind": "canonical",
                "fallback": "locality",
                "accepted": ["locality"],
                "origin": {"locality": "definitional"},
            },
            "contact": {
                "kind": "canonical",
                "fallback": "O",
                "accepted": ["O", "email", "phone_number"],
                "origin": {"email": "definitional", "phone_number": "empirical", "O": "definitional"},
            },
        },
        "v2_to_v1": {},
    }
    old_labels = new_space_labels(["city", "contact"])
    new_labels = new_space_labels(["email", "locality", "phone_number"])
    correctness = build_correctness_map(document, old_labels, new_labels, map_path="<self-test>")
    assert correctness.empirical_targets == 1
    s_city, s_contact = (old_labels.index(f"S-{node}") for node in ("city", "contact"))
    email, locality, phone = (new_labels.index(f"S-{t}") for t in ("email", "locality", "phone_number"))
    assert correctness.old_to_new[s_city].tolist() == [i == locality for i in range(len(new_labels))]
    assert correctness.old_to_new[s_contact, email] and correctness.old_to_new[s_contact, phone]
    assert correctness.old_to_new[s_contact, 0], "an O-acceptable v1 node must reach the new head's O row"
    assert correctness.new_to_old[locality, s_city] and not correctness.new_to_old[email, s_city]
    assert not correctness.old_to_new[old_labels.index("B-city"), locality], (
        "a mapped target must keep its own boundary letter"
    )

    torch.manual_seed(0)
    old_logits = torch.randn(1, 4, len(old_labels))
    new_logits = torch.randn(1, 4, len(new_labels))
    # token 0: old-space city; token 1: new-space locality; token 2: O; token 3: masked
    old_labels_t = torch.tensor([[s_city, -100, 0, -100]])
    new_labels_t = torch.tensor([[-100, locality, 0, -100]])

    _, telemetry = dual_head_loss(old_logits, new_logits, old_labels_t, new_labels_t, correctness, 1.0)
    assert telemetry["dual_old_tokens"] == 3 and telemetry["dual_new_tokens"] == 3

    old_logits.requires_grad_(True)
    new_logits.requires_grad_(True)
    loss, _ = dual_head_loss(old_logits, new_logits, old_labels_t, new_labels_t, correctness, 0.0)
    loss.backward()
    assert old_logits.grad is not None and torch.all(old_logits.grad == 0), "weight 0 must free the old head"
    assert new_logits.grad is not None and torch.any(new_logits.grad != 0), "the new head must still learn"

    # A fade compressed into part of the horizon reaches zero there and stays.
    quarter = parse_transition_schedule("linear:1.0:0.0:0.25")
    assert transition_weight(quarter, 0, 1000) == 1.0
    assert transition_weight(quarter, 125, 1000) == 0.5
    assert transition_weight(quarter, 250, 1000) == 0.0
    assert transition_weight(quarter, 900, 1000) == 0.0
    assert transition_completion_step(quarter, 1000) == 250
    whole = parse_transition_schedule("linear:1.0:0.0")
    assert transition_weight(whole, 250, 1000) == 0.75, "an unqualified fade still spans the whole horizon"
    assert transition_completion_step(whole, 1000) == 1000
    assert transition_completion_step(parse_transition_schedule("constant:0.0"), 1000) == 0
    assert transition_completion_step(parse_transition_schedule("constant:1.0"), 1000) is None
    assert transition_completion_step(parse_transition_schedule("linear:1.0:0.5"), 1000) is None, (
        "a fade that stops short of zero never retires the old head"
    )
    for bad in ("linear:1.0:0.0:0", "linear:1.0:0.0:1.5", "linear:1.0:0.0:x", "linear:1.0"):
        try:
            parse_transition_schedule(bad)
        except DualHeadError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not parse as a transition schedule")

    sources = fallback_source_rows(document, old_labels, new_labels)
    assert old_labels.index("S-city") in sources[locality], "city falls back to locality"
    assert old_labels.index("S-contact") in sources[0], "contact falls back to O"
    assert sources[0][0] == old_labels.index(OUTSIDE_LABEL), "O seeds the new O row first"

    # A mapped token with exactly one allowed row is plain cross-entropy on it.
    single = torch.tensor([[locality]])
    logits = torch.randn(1, 1, len(new_labels))
    mapped, _ = head_loss(logits, torch.tensor([[-100]]), torch.tensor([[s_city]]), correctness.old_to_new)
    direct, _ = head_loss(logits, single, torch.tensor([[-100]]), correctness.old_to_new)
    assert torch.allclose(mapped, direct, atol=1e-6)

    print("PII_DUAL_HEAD self-test ok", flush=True)


if __name__ == "__main__":
    _self_test()
