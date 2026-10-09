#!/usr/bin/env python
"""Exact-offset P9 preparation for GLiNER2 calibration.

GLiNER2's public training format names entity surface strings, then finds every
matching surface in the document.  That cannot distinguish one labelled
occurrence from an identical unlabelled occurrence.  This module keeps the
installed encoder and entity path but replaces only that lossy preparation:
gold character spans are aligned directly to the processor's word offsets.

It also supplies deterministic, gold-independent overlapping text windows and
a CJK-aware word splitter.  The model's existing span representation shares
weights across widths, so raising ``max_width`` changes its enumerated span
cap without adding learned parameters.
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pii_annotation_cache import normalize_span, validate_spans  # noqa: E402
from pii_gliner2_projection import project_canonical_spans  # noqa: E402
from pii_projector import Tagset  # noqa: E402

EXACT_PREPARATION_VERSION = "gliner2-exact-offset-p9-v4"
OVERLAP_MIN_COVERAGE_NUMERATOR = 4
OVERLAP_MIN_COVERAGE_DENOMINATOR = 5
_TAGSET = Tagset()


def _is_cjk(character: str) -> bool:
    value = ord(character)
    return any(
        lower <= value <= upper
        for lower, upper in (
            (0x1100, 0x11FF),
            (0x2E80, 0x2FFF),
            (0x3040, 0x30FF),
            (0x3130, 0x318F),
            (0x31F0, 0x31FF),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
        )
    )


class CjkAwareTokenSplitter:
    """Preserve GLiNER2's lexical tokens while exposing CJK characters."""

    __slots__ = ()

    _PATTERN = re.compile(
        r"""(?:https?://[a-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+|www\.[a-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)
        |@[a-z0-9_]+
        |[^\W\u1100-\u11ff\u2e80-\u2fff\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\uf900-\ufaff\U00020000-\U0002fa1f]+
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text: str, lower: bool = True) -> Iterable[tuple[str, int, int]]:
        for match in self._PATTERN.finditer(text):
            token = match.group()
            if not any(_is_cjk(character) for character in token):
                if all(character == "_" or character.isalnum() for character in token):
                    segment_start = 0
                    segment_group = _word_group(token[0])
                    for index, character in enumerate(token[1:], 1):
                        group = _word_group(character)
                        if group == segment_group:
                            continue
                        piece = token[segment_start:index]
                        yield (
                            piece.lower() if lower else piece,
                            match.start() + segment_start,
                            match.start() + index,
                        )
                        segment_start = index
                        segment_group = group
                    piece = token[segment_start:]
                    yield (
                        piece.lower() if lower else piece,
                        match.start() + segment_start,
                        match.end(),
                    )
                else:
                    yield (token.lower() if lower else token), match.start(), match.end()
                continue
            buffer_start = 0
            for index, character in enumerate(token):
                if not _is_cjk(character):
                    continue
                if buffer_start < index:
                    piece = token[buffer_start:index]
                    yield (
                        piece.lower() if lower else piece,
                        match.start() + buffer_start,
                        match.start() + index,
                    )
                yield (
                    character.lower() if lower else character,
                    match.start() + index,
                    match.start() + index + 1,
                )
                buffer_start = index + 1
            if buffer_start < len(token):
                piece = token[buffer_start:]
                yield (
                    piece.lower() if lower else piece,
                    match.start() + buffer_start,
                    match.end(),
                )


def _word_group(character: str) -> str:
    if character.isdecimal():
        return "decimal"
    if character == "_":
        return "underscore"
    name = unicodedata.name(character, "")
    for script in (
        "ARABIC",
        "CYRILLIC",
        "DEVANAGARI",
        "GREEK",
        "HEBREW",
        "LATIN",
        "THAI",
    ):
        if script in name:
            return script
    return unicodedata.category(character)


@dataclass(frozen=True)
class TextWindow:
    start: int
    end: int
    text: str


class RepresentationError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def text_windows(text: str, *, max_chars: int, overlap_chars: int) -> list[TextWindow]:
    """Return fixed overlapping slices whose boundaries never consult gold."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    if not text:
        return [TextWindow(0, 0, "")]
    stride = max_chars - overlap_chars
    starts = list(range(0, len(text), stride))
    if starts and starts[-1] + max_chars < len(text):
        starts.append(len(text) - max_chars)
    windows = []
    seen = set()
    for start in starts:
        end = min(start + max_chars, len(text))
        if (start, end) in seen:
            continue
        seen.add((start, end))
        windows.append(TextWindow(start, end, text[start:end]))
        if end == len(text):
            break
    return windows


def canonical_p9_spans(row: dict[str, Any]) -> list[tuple[int, int, str]]:
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError("row text must be a string")
    spans = sorted(
        (tuple(normalize_span(_TAGSET, "canonical", span)) for span in row.get("spans", [])),
        key=lambda span: (span[0], span[1], span[2]),
    )
    validate_spans("row", 1, text, spans, allow_overlaps=True)
    inventory = sorted(_TAGSET.cut_targets("redaction_9_v1"))
    projected, unmapped, _ = project_canonical_spans(
        _TAGSET,
        spans,
        inventory,
        output_schema="redaction_9_v1",
        alias_policy="all",
    )
    if unmapped:
        raise AssertionError(f"P9 projection left canonical spans unmapped: {unmapped!r}")
    return projected


def window_projected_row(
    row: dict[str, Any],
    *,
    max_chars: int,
    overlap_chars: int,
    drop_straddling_windows: bool,
    span_projector: Callable[[dict[str, Any]], list[tuple[int, int, str]]] = canonical_p9_spans,
) -> list[dict[str, Any]]:
    return _project_text_windows(
        row,
        text_windows(row["text"], max_chars=max_chars, overlap_chars=overlap_chars),
        drop_straddling_windows=drop_straddling_windows,
        span_projector=span_projector,
    )


def _project_text_windows(
    row: dict[str, Any],
    windows: Sequence[TextWindow],
    *,
    drop_straddling_windows: bool,
    span_projector: Callable[[dict[str, Any]], list[tuple[int, int, str]]] = canonical_p9_spans,
) -> list[dict[str, Any]]:
    # The windowing, straddle handling and provenance are label-set agnostic;
    # only the projection into an output inventory is not. Injecting it lets a
    # different ontology reuse this without touching the frozen P9 default.
    projected = span_projector(row)
    result = []
    source_id = str(row.get("id") or hashlib.sha256(row["text"].encode()).hexdigest())
    for window_index, window in enumerate(windows):
        local_spans = []
        straddles = []
        for start, end, label in projected:
            if window.start <= start and end <= window.end:
                local_spans.append((start - window.start, end - window.start, label))
            elif start < window.end and window.start < end:
                straddles.append((start, end, label))
        if drop_straddling_windows and straddles:
            continue
        result.append(
            {
                "input": window.text,
                "provenance": {
                    "preparation": EXACT_PREPARATION_VERSION,
                    "source_id": source_id,
                    "lang": row.get("lang"),
                    "membership_split": row.get("membership_split"),
                    "window_index": window_index,
                    "window_start": window.start,
                    "window_end": window.end,
                    "projected_spans": [list(span) for span in local_spans],
                    "straddling_projected_spans": [list(span) for span in straddles],
                },
            }
        )
    return result


def configure_processor(processor: Any) -> None:
    processor.word_splitter = CjkAwareTokenSplitter()


def configure_span_width(model: Any, max_width: int) -> dict[str, int]:
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    layer = model.span_rep.span_rep_layer
    original = int(model.max_width)
    model.max_width = max_width
    model.config.max_width = max_width
    layer.max_width = max_width
    return {"original": original, "configured": max_width}


def _word_span_for_chars(
    starts: Sequence[int], ends: Sequence[int], start: int, end: int
) -> tuple[int, int, int, int]:
    start_indices = [index for index, value in enumerate(starts) if value == start]
    end_indices = [index for index, value in enumerate(ends) if value == end]
    if start_indices and end_indices:
        first = start_indices[0]
        last = end_indices[-1]
        if first <= last:
            return first, last, start, end

    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(zip(starts, ends))
        if token_start < end and start < token_end
    ]
    if not overlapping:
        raise RepresentationError("unaligned_span", f"character span [{start}, {end})")
    first = overlapping[0]
    last = overlapping[-1]
    aligned_start = starts[first]
    aligned_end = ends[last]
    intersection = min(end, aligned_end) - max(start, aligned_start)
    gold_length = end - start
    aligned_length = aligned_end - aligned_start
    if (
        intersection <= 0
        or intersection * OVERLAP_MIN_COVERAGE_DENOMINATOR < gold_length * OVERLAP_MIN_COVERAGE_NUMERATOR
        or intersection * OVERLAP_MIN_COVERAGE_DENOMINATOR < aligned_length * OVERLAP_MIN_COVERAGE_NUMERATOR
    ):
        raise RepresentationError(
            "unaligned_span",
            f"character span [{start}, {end}) minimally maps to "
            f"[{aligned_start}, {aligned_end}) below symmetric 80% coverage",
        )
    return first, last, aligned_start, aligned_end


def transform_exact_entity_row(
    processor: Any,
    row: dict[str, Any],
    labels: Sequence[str],
    *,
    max_input_tokens: int,
    max_span_width: int,
    alignment_receipt: list[dict[str, Any]] | None = None,
    unrepresentable_receipt: list[dict[str, Any]] | None = None,
) -> Any:
    """Create one GLiNER2 record with exact word-span supervision."""
    configure_processor(processor)
    processor.change_mode(False)
    text = row["input"]
    processed_text = text if text.endswith((".", "!", "?")) else text + "."
    entities = {label: [] for label in labels}
    transformed = processor.transform_and_format(processed_text, {"entities": entities})
    if transformed.task_types != ["entities"]:
        raise RepresentationError("task_shape", f"unexpected tasks {transformed.task_types!r}")
    if len(transformed.input_ids) > max_input_tokens:
        raise RepresentationError(
            "input_too_long",
            f"{len(transformed.input_ids)} encoded tokens exceeds {max_input_tokens}",
        )
    positions: dict[str, list[tuple[int, int]]] = {label: [] for label in labels}
    for start, end, label in row["provenance"]["projected_spans"]:
        if label not in positions:
            raise RepresentationError("unknown_label", str(label))
        try:
            first, last, aligned_start, aligned_end = _word_span_for_chars(
                transformed.start_token_idx,
                transformed.end_token_idx,
                int(start),
                int(end),
            )
            width = last - first + 1
            if width > max_span_width:
                raise RepresentationError(
                    "span_too_wide",
                    f"span [{start}, {end}) has word width {width}>{max_span_width}",
                )
        except RepresentationError as error:
            if unrepresentable_receipt is None or error.reason not in {
                "span_too_wide",
                "unaligned_span",
            }:
                raise
            unrepresentable_receipt.append(
                {
                    "gold": [int(start), int(end)],
                    "label": str(label),
                    "reason": error.reason,
                    "detail": error.detail,
                }
            )
            continue
        if alignment_receipt is not None and (aligned_start, aligned_end) != (start, end):
            alignment_receipt.append(
                {
                    "gold": [int(start), int(end)],
                    "represented": [int(aligned_start), int(aligned_end)],
                    "label": str(label),
                    "match_rule": "symmetric-80-percent-character-coverage",
                }
            )
        positions[label].append((first, last))
    fields = [positions[label] if positions[label] else (-1, -1) for label in labels]
    transformed.structure_labels = [[1, [fields]]]
    return transformed


def token_budgeted_projected_rows(
    processor: Any,
    row: dict[str, Any],
    labels: Sequence[str],
    *,
    max_chars: int,
    overlap_chars: int,
    max_input_tokens: int,
    max_span_width: int,
) -> tuple[list[dict[str, Any]], int]:
    """Refine only over-budget fixed windows, without consulting gold."""
    pending = text_windows(row["text"], max_chars=max_chars, overlap_chars=overlap_chars)
    accepted: dict[tuple[int, int], TextWindow] = {}
    visited: set[tuple[int, int]] = set()
    split_count = 0
    while pending:
        window = pending.pop(0)
        key = (window.start, window.end)
        if key in visited:
            continue
        visited.add(key)
        probe = {"input": window.text, "provenance": {"projected_spans": []}}
        try:
            transform_exact_entity_row(
                processor,
                probe,
                labels,
                max_input_tokens=max_input_tokens,
                max_span_width=max_span_width,
            )
        except RepresentationError as error:
            if error.reason != "input_too_long":
                raise
            length = window.end - window.start
            if length <= overlap_chars + 1:
                raise RepresentationError(
                    "token_budget_unsatisfied",
                    f"{length}-character window cannot fit {max_input_tokens} encoded tokens "
                    f"while preserving {overlap_chars}-character overlap",
                ) from error
            child_length = (length + overlap_chars + 1) // 2
            pending.extend(
                (
                    TextWindow(
                        window.start,
                        window.start + child_length,
                        row["text"][window.start : window.start + child_length],
                    ),
                    TextWindow(
                        window.end - child_length,
                        window.end,
                        row["text"][window.end - child_length : window.end],
                    ),
                )
            )
            split_count += 1
            continue
        accepted[key] = window
    windows = [accepted[key] for key in sorted(accepted)]
    return (
        _project_text_windows(row, windows, drop_straddling_windows=False),
        split_count,
    )


def collate_exact_entity_rows(
    processor: Any,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    max_input_tokens: int,
    max_span_width: int,
) -> Any:
    transformed = [
        transform_exact_entity_row(
            processor,
            row,
            labels,
            max_input_tokens=max_input_tokens,
            max_span_width=max_span_width,
        )
        for row in rows
    ]
    return processor._pad_batch(transformed)


def _loss_exclusion_intervals(row: dict[str, Any]) -> list[tuple[int, int]]:
    provenance = row["provenance"]
    window_start = int(provenance.get("window_start", 0))
    window_end = int(provenance.get("window_end", window_start + len(row["input"])))
    intervals = []
    for start, end, _label in provenance.get("straddling_projected_spans", []):
        local_start = max(int(start), window_start) - window_start
        local_end = min(int(end), window_end) - window_start
        if local_start < local_end:
            intervals.append((local_start, local_end))
    intervals.extend(
        (int(item["gold"][0]), int(item["gold"][1]))
        for item in provenance.get("unrepresentable_complete_spans", [])
    )
    return sorted(set(intervals))


def word_span_loss_exclusion_mask(
    transformed: Any,
    row: dict[str, Any],
    *,
    max_span_width: int,
) -> Any:
    """Mark enumerated word spans that overlap intentionally unsupervised text."""
    import torch

    intervals = _loss_exclusion_intervals(row)
    mask = torch.zeros(
        (len(transformed.start_token_idx), max_span_width),
        dtype=torch.bool,
    )
    if not intervals:
        return mask
    for start in range(len(transformed.start_token_idx)):
        for width in range(max_span_width):
            end = start + width
            if end >= len(transformed.end_token_idx):
                break
            char_start = int(transformed.start_token_idx[start])
            char_end = int(transformed.end_token_idx[end])
            mask[start, width] = any(
                char_start < ignored_end and ignored_start < char_end
                for ignored_start, ignored_end in intervals
            )
    return mask


def collate_exact_entity_rows_with_masks(
    processor: Any,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    max_input_tokens: int,
    max_span_width: int,
) -> tuple[Any, list[Any]]:
    transformed = []
    for row in rows:
        generated_receipt: list[dict[str, Any]] = []
        record = transform_exact_entity_row(
            processor,
            row,
            labels,
            max_input_tokens=max_input_tokens,
            max_span_width=max_span_width,
            unrepresentable_receipt=generated_receipt,
        )
        expected_receipt = row["provenance"].get("unrepresentable_complete_spans", [])

        def receipt_key(item: dict[str, Any]) -> tuple[int, int, str, str]:
            return (
                int(item["gold"][0]),
                int(item["gold"][1]),
                str(item["label"]),
                str(item["reason"]),
            )

        if sorted(map(receipt_key, generated_receipt)) != sorted(map(receipt_key, expected_receipt)):
            raise RepresentationError(
                "unreachable_receipt_mismatch",
                f"generated={generated_receipt!r}, declared={expected_receipt!r}",
            )
        transformed.append(record)
    masks = [
        word_span_loss_exclusion_mask(record, row, max_span_width=max_span_width)
        for record, row in zip(transformed, rows)
    ]
    return processor._pad_batch(transformed), masks


def representation_ledger(
    processor: Any,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    max_chars: int,
    overlap_chars: int,
    max_input_tokens: int,
    max_span_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize exact training windows and a per-language rejection ledger."""
    accepted = []
    failures = []
    unrepresentable_occurrences = []
    adaptive_token_splits = 0
    source_coverage: dict[str, Counter[tuple[int, int, str]]] = {}
    source_gold: dict[str, Counter[tuple[int, int, str]]] = {}
    source_languages = {}
    for row in rows:
        source_id = str(row["id"])
        source_languages[source_id] = str(row["lang"])
        source_gold[source_id] = Counter(canonical_p9_spans(row))
        source_coverage[source_id] = Counter()
        windows, source_token_splits = token_budgeted_projected_rows(
            processor,
            row,
            labels,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            max_input_tokens=max_input_tokens,
            max_span_width=max_span_width,
        )
        adaptive_token_splits += source_token_splits
        for window in windows:
            alignment_receipt: list[dict[str, Any]] = []
            unrepresentable_receipt: list[dict[str, Any]] = []
            try:
                transform_exact_entity_row(
                    processor,
                    window,
                    labels,
                    max_input_tokens=max_input_tokens,
                    max_span_width=max_span_width,
                    alignment_receipt=alignment_receipt,
                    unrepresentable_receipt=unrepresentable_receipt,
                )
            except RepresentationError as error:
                failures.append(
                    {
                        "source_id": source_id,
                        "lang": row["lang"],
                        "window_index": window["provenance"]["window_index"],
                        "reason": error.reason,
                        "detail": error.detail,
                    }
                )
                continue
            window["provenance"]["alignment_adjustments"] = alignment_receipt
            window["provenance"]["unrepresentable_complete_spans"] = unrepresentable_receipt
            accepted.append(window)
            offset = int(window["provenance"]["window_start"])
            unrepresentable_keys = {
                (int(item["gold"][0]), int(item["gold"][1]), str(item["label"]))
                for item in unrepresentable_receipt
            }
            unrepresentable_occurrences.extend(
                {
                    **item,
                    "source_id": source_id,
                    "lang": row["lang"],
                    "window_index": window["provenance"]["window_index"],
                    "source_gold": [
                        int(item["gold"][0]) + offset,
                        int(item["gold"][1]) + offset,
                        str(item["label"]),
                    ],
                }
                for item in unrepresentable_receipt
            )
            source_coverage[source_id].update(
                (int(start) + offset, int(end) + offset, str(label))
                for start, end, label in window["provenance"]["projected_spans"]
                if (int(start), int(end), str(label)) not in unrepresentable_keys
            )
    uncovered = []
    for source_id, gold in source_gold.items():
        covered = source_coverage[source_id]
        for span in gold:
            if covered[span] == 0:
                uncovered.append(
                    {
                        "source_id": source_id,
                        "lang": source_languages[source_id],
                        "span": list(span),
                    }
                )
    counts = Counter(failure["reason"] for failure in failures)
    language_counts: dict[str, Counter[str]] = {}
    for row in rows:
        language_counts.setdefault(str(row["lang"]), Counter())["source_documents"] += 1
    for window in accepted:
        language_counts.setdefault(str(window["provenance"]["lang"]), Counter())["accepted_windows"] += 1
    for failure in failures:
        language_counts.setdefault(str(failure["lang"]), Counter())["rejected_windows"] += 1
    for item in uncovered:
        language_counts.setdefault(str(item["lang"]), Counter())["uncovered_spans"] += 1
    ledger = {
        "version": EXACT_PREPARATION_VERSION,
        "settings": {
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "max_input_tokens": max_input_tokens,
            "max_span_width": max_span_width,
            "word_splitter": "cjk-aware-character-boundaries",
            "gold_independent_windows": True,
            "adaptive_token_budget": True,
            "drop_training_windows_that_cut_gold": False,
            "straddling_gold_policy": "retain-window-and-mask-boundary-fragments-in-loss",
            "boundary_match": "exact-else-symmetric-80-percent-character-coverage",
        },
        "counts": {
            "source_documents": len(rows),
            "accepted_windows": len(accepted),
            "rejected_windows": len(failures),
            "adaptive_token_splits": adaptive_token_splits,
            "unrepresentable_occurrences": len(unrepresentable_occurrences),
            "uncovered_spans": len(uncovered),
        },
        "failure_reasons": dict(sorted(counts.items())),
        "languages": {key: dict(value) for key, value in sorted(language_counts.items())},
        "failures": failures,
        "unrepresentable_occurrences": unrepresentable_occurrences,
        "uncovered": uncovered,
    }
    return accepted, ledger
