"""Share one encoder-input exposure budget among retained label variants."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainlib_mix import share_duplicate_mass  # noqa: E402


def share_annotation_sampling_mass(
    rows: list[dict[str, Any]], weights: list[float] | None
) -> tuple[list[float] | None, dict[str, Any]]:
    """Average pre-share weights per identical input, then split equally.

    Apply after compiling pool masses, before mixing supervision with MLM and
    applying language multipliers. Pool normalization must not run afterward.
    Language and exact text define the encoder input; annotation labels, source
    filenames and annotation versions do not make another independent input.
    This is exposure accounting, not a replacement for partial-overlap dedup.
    """
    for row in rows:
        if not isinstance(row["lang"], str) or not isinstance(row["text"], str):
            raise ValueError("annotation weight sharing requires string language and text")
    return share_duplicate_mass(
        rows,
        weights,
        key=lambda row: (row["lang"], row["text"]),
        identity="exact_language_and_encoder_input_text",
        schema="pii-annotation-sampling-share/v1",
    )
