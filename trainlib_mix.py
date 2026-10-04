"""Training-mixture weighting shared across tasks: duplicate-input mass sharing and group caps.

Rows are dicts carrying a sampling weight; what counts as "the same input" and
what a cap groups by (language, speaker, source, corpus) are supplied by the
caller, so a PII mixture and a speech mixture can use the same rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from typing import Any


def share_duplicate_mass(
    rows: list[dict[str, Any]],
    weights: list[float] | None,
    *,
    key: Callable[[dict[str, Any]], Hashable],
    identity: str,
    schema: str,
) -> tuple[list[float] | None, dict[str, Any]]:
    """Average pre-share weights per identical input, then split equally among its rows.

    Rows with the same ``key`` are one input seen through several label
    variants; together they get the weight of one average variant, each an
    equal part, and the result is renormalized to a distribution. Returns the
    input weights unchanged (possibly None) when no input repeats. This is
    exposure accounting, not near-duplicate removal.
    """
    if not rows:
        raise ValueError("weight sharing requires nonempty rows")
    original = [1.0] * len(rows) if weights is None else list(weights)
    if len(original) != len(rows) or any(not math.isfinite(weight) or weight < 0 for weight in original):
        raise ValueError("sampling weights must align with rows and be finite nonnegative")
    if math.fsum(original) <= 0:
        raise ValueError("sampling weights have no positive mass")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[key(row)].append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    receipt = {
        "schema": schema,
        "policy": "mean_group_weight_shared_equally",
        "identity": identity,
        "rows": len(rows),
        "distinct_inputs": len(groups),
        "multiply_annotated_inputs": len(duplicate_groups),
        "rows_in_multiply_annotated_inputs": sum(map(len, duplicate_groups)),
        "largest_multiplicity": max(map(len, groups.values())),
        "group_membership_sha256": hashlib.sha256(
            json.dumps(list(groups.values()), separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if not duplicate_groups:
        return weights, receipt
    shared = list(original)
    for indices in duplicate_groups:
        per_variant = math.fsum(original[index] for index in indices) / len(indices) ** 2
        for index in indices:
            shared[index] = per_variant
    total = math.fsum(shared)
    receipt["pre_normalization_mass_ratio"] = total / math.fsum(original)
    return [weight / total for weight in shared], receipt


def apply_caps(
    rows: list[dict[str, Any]],
    caps: dict[str, Any],
    *,
    group: Callable[[dict[str, Any]], Hashable],
    movable: Callable[[dict[str, Any]], bool],
    weight: str = "sampling_weight",
) -> dict[str, Any]:
    """Bound each group's share of total mass by moving mass between movable rows.

    ``caps`` is ``{"default": share, "groups": {group: share}}``. A group over its
    cap has its movable rows scaled down; the freed mass goes to movable rows of
    groups below their caps, in proportion, so the total movable mass and every
    fixed row are unchanged. A group whose fixed rows alone exceed its cap is
    reported in ``unattainable_from_fixed`` and takes no freed mass.
    """

    def masses() -> Counter:
        totals = Counter()
        for row in rows:
            totals[(group(row), movable(row))] += row[weight]
        return totals

    total = math.fsum(row[weight] for row in rows)
    movable_total = math.fsum(row[weight] for row in rows if movable(row))
    before = Counter()
    for (name, _movable), mass in masses().items():
        before[name] += mass / total
    fixed_groups, unattainable = set(), {}
    for _ in range(100):
        totals = masses()
        over = {}
        for name in {name for name, _movable in totals}:
            cap = caps["groups"].get(name, caps["default"]) * total
            fixed, moving = totals[(name, False)], totals[(name, True)]
            if fixed + moving > cap * (1 + 1e-9):
                if fixed >= cap:
                    unattainable[name] = fixed / total
                    continue
                over[name] = (cap - fixed) / moving
        if not over:
            break
        freed = math.fsum(totals[(name, True)] * (1 - factor) for name, factor in over.items())
        fixed_groups.update(over)
        # Groups at their cap, or over it from fixed rows alone, take no freed mass.
        frozen = fixed_groups | set(unattainable)
        receivers = math.fsum(
            mass for (name, is_movable), mass in totals.items() if is_movable and name not in frozen
        )
        if receivers <= 0:
            raise ValueError("caps leave no uncapped movable rows to take the freed mass")
        for row in rows:
            if not movable(row):
                continue
            name = group(row)
            if name in over:
                row[weight] *= over[name]
            elif name not in frozen:
                row[weight] *= 1 + freed / receivers
        if abs(math.fsum(row[weight] for row in rows if movable(row)) - movable_total) > 1e-9:
            raise RuntimeError("capping changed the total movable mass")
    after = Counter()
    for (name, _movable), mass in masses().items():
        after[name] += mass / total
    return {
        "default": caps["default"],
        "groups": caps["groups"],
        "capped": {name: {"before": before[name], "after": after[name]} for name in sorted(fixed_groups)},
        "unattainable_from_fixed": unattainable,
    }
