"""Share one encoder-input exposure budget among retained label variants."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any


def share_annotation_sampling_mass(
    rows: list[dict[str, Any]], weights: list[float] | None
) -> tuple[list[float] | None, dict[str, Any]]:
    """Average pre-share weights per identical input, then split equally.

    Apply after compiling pool masses, before mixing supervision with MLM and
    applying language multipliers. Pool normalization must not run afterward.
    Language and exact text define the encoder input; annotation labels, source
    filenames and annotation versions do not make another independent input.
    This is exposure accounting, not a replacement for partial-overlap dedup.

    With unequal pre-share weights, the arithmetic mean gives the group the
    weight of one average variant. Each variant receives an equal part of that
    group budget. Global normalization retains a probability distribution.
    """
    if not rows:
        raise ValueError("annotation weight sharing requires nonempty rows")
    original = [1.0] * len(rows) if weights is None else list(weights)
    if len(original) != len(rows) or any(not math.isfinite(weight) or weight < 0 for weight in original):
        raise ValueError("annotation sampling weights must align and be finite nonnegative")
    if math.fsum(original) <= 0:
        raise ValueError("annotation sampling weights have no positive mass")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        language, text = row["lang"], row["text"]
        if not isinstance(language, str) or not isinstance(text, str):
            raise ValueError("annotation weight sharing requires string language and text")
        groups[language, text].append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    receipt = {
        "schema": "pii-annotation-sampling-share/v1",
        "policy": "mean_group_weight_shared_equally",
        "identity": "exact_language_and_encoder_input_text",
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
