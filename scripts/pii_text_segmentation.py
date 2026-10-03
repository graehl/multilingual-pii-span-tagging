#!/usr/bin/env python
"""Exact-offset multilingual sentence views for labeled PII records.

Sentence splitting is a view transform: callers must assign a source record to
train/validation/test before invoking this module.  Every candidate boundary
strictly inside a labeled span is removed, so no derived view contains a
clipped entity.  A document with N segments contributes N source-weighted
volume units.  ``long_view_alpha`` divides that mass between the full document
and its segment views: at alpha 0.5 the full row weighs N/2 and each segment
weighs 1/2.  The segment-matched training default is alpha 0; positive alpha is
a length-robustness contrast.  Language and other adjustments apply later.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from importlib.metadata import version as distribution_version
from typing import Protocol

SAT_MODEL = "sat-3l-sm"
SAT_MODEL_REVISION = "137da054051ad9f1eac42025f758db4ac9f22535"


class CharacterSpanSplitter(Protocol):
    """Batch sentence splitter whose pieces reconstruct each input exactly."""

    segmenter_model: str
    segmenter_model_revision: str
    segmenter_name: str
    segmenter_version: str

    def split(self, texts: Sequence[str]) -> Iterable[Sequence[str]]: ...


class SaTCharacterSpanSplitter:
    """Lazy wrapper around the multilingual SaT sentence segmenter."""

    def __init__(
        self,
        model: str = SAT_MODEL,
        *,
        model_revision: str = SAT_MODEL_REVISION,
        batch_size: int = 32,
        split_on_input_newlines: bool = False,
        device: str = "cpu",
    ) -> None:
        try:
            from wtpsplit import SaT
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "SaT sentence views require wtpsplit; install the pinned pixi-gemma4 dependency"
            ) from error
        self.model_name = model
        self.batch_size = batch_size
        self.segmenter_name = "wtpsplit.SaT"
        self.segmenter_version = distribution_version("wtpsplit")
        self.segmenter_model = f"segment-any-text/{model}" if "/" not in model else model
        self.segmenter_model_revision = model_revision
        self.segmenter_split_on_input_newlines = split_on_input_newlines
        self.segmenter_device = device
        self.model = SaT(model, from_pretrained_kwargs={"revision": model_revision})
        self.model.model.model.to(device)

    def split(self, texts: Sequence[str]) -> Iterable[Sequence[str]]:
        rows = self.model.split(
            list(texts),
            batch_size=self.batch_size,
            split_on_input_newlines=False,
            strip_whitespace=False,
        )
        if not self.segmenter_split_on_input_newlines:
            return rows
        return (
            [fragment for piece in pieces for fragment in split_at_line_boundaries(piece)] for pieces in rows
        )


def split_at_line_boundaries(text: str) -> list[str]:
    """Split at input line boundaries while retaining every delimiter exactly."""
    raw_pieces = text.splitlines(keepends=True)
    pieces: list[str] = []
    leading = ""
    for piece in raw_pieces:
        if piece.strip():
            pieces.append(leading + piece)
            leading = ""
        elif pieces:
            pieces[-1] += piece
        else:
            leading += piece
    if leading:
        pieces.append(leading)
    if not pieces:
        pieces = [text]
    if "".join(pieces) != text:
        raise AssertionError("line-boundary split did not reconstruct its input")
    return pieces


def segmenter_provenance(splitter: CharacterSpanSplitter) -> dict[str, str | bool]:
    provenance: dict[str, str | bool] = {
        "segmenter_model": splitter.segmenter_model,
        "segmenter_model_revision": splitter.segmenter_model_revision,
        "segmenter_name": splitter.segmenter_name,
        "segmenter_version": splitter.segmenter_version,
    }
    split_on_newlines = getattr(splitter, "segmenter_split_on_input_newlines", None)
    if isinstance(split_on_newlines, bool):
        provenance["segmenter_split_on_input_newlines"] = split_on_newlines
    device = getattr(splitter, "segmenter_device", None)
    if isinstance(device, str):
        provenance["segmenter_device"] = device
    return provenance


def _validate_spans(text: str, spans: Sequence[Sequence[object]]) -> None:
    prior_end = 0
    for span in spans:
        if len(span) < 2:
            raise ValueError(f"span needs start and end offsets: {span!r}")
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"span [{start}, {end}) is outside text length {len(text)}")
        if start < prior_end:
            raise ValueError(f"spans overlap or are not sorted at [{start}, {end})")
        prior_end = end


def _merge_protected_boundaries(
    text: str,
    pieces: Sequence[str],
    protected_spans: Sequence[Sequence[object]],
) -> list[tuple[int, int]]:
    if "".join(pieces) != text:
        raise ValueError("sentence splitter pieces do not reconstruct the input exactly")
    _validate_spans(text, protected_spans)
    candidate_boundaries = []
    offset = 0
    for piece in pieces[:-1]:
        offset += len(piece)
        candidate_boundaries.append(offset)
    boundaries = [
        boundary
        for boundary in candidate_boundaries
        if not any(int(start) < boundary < int(end) for start, end, *_ in protected_spans)
    ]
    offsets = [0, *boundaries, len(text)]
    return [(start, end) for start, end in zip(offsets, offsets[1:]) if start < end]


def sentence_spans_batch(
    texts: Sequence[str],
    protected_spans: Sequence[Sequence[Sequence[object]]],
    splitter: CharacterSpanSplitter,
) -> Iterator[list[tuple[int, int]]]:
    """Yield exact sentence spans, suppressing boundaries inside labels."""
    if len(texts) != len(protected_spans):
        raise ValueError("texts and protected-span rows must align one-to-one")
    split_rows = iter(splitter.split(texts))
    observed = 0
    for text, spans, pieces in zip(texts, protected_spans, split_rows, strict=True):
        observed += 1
        yield _merge_protected_boundaries(text, pieces, spans)
    if observed != len(texts):
        raise AssertionError(f"sentence splitter returned {observed} rows for {len(texts)} inputs")


def sentence_spans(
    text: str,
    protected_spans: Sequence[Sequence[object]],
    splitter: CharacterSpanSplitter,
) -> list[tuple[int, int]]:
    return next(sentence_spans_batch([text], [protected_spans], splitter))


def source_content_id(row: dict) -> str:
    """Stable pre-view identity used to detect split leakage."""
    canonical = json.dumps(
        {
            "text": row["text"],
            "spans": row.get("spans", []),
            "lang": row.get("lang"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _rebased_spans(row: dict, start: int, end: int) -> list[list[object]]:
    rebased = []
    for span_start, span_end, label, *rest in row.get("spans", []):
        if span_end <= start or end <= span_start:
            continue
        if span_start < start or end < span_end:
            raise AssertionError(f"view [{start}, {end}) cuts labeled span [{span_start}, {span_end})")
        rebased.append([span_start - start, span_end - start, label, *rest])
    return rebased


def rebase_spans(row: dict, start: int, end: int) -> list[list[object]]:
    """Return labels wholly contained in one exact source interval."""
    return _rebased_spans(row, start, end)


def expand_training_views(
    row: dict,
    splitter: CharacterSpanSplitter,
    *,
    long_view_alpha: float = 0.0,
) -> list[dict]:
    """Expand one row while assigning segment-proportional total mass."""
    if not 0 <= long_view_alpha <= 1:
        raise ValueError("long_view_alpha must be in [0, 1]")
    content_id = row.get("source_content_id") or source_content_id(row)
    group_id = row.get("view_group_id") or row.get("source_window_id") or content_id
    base_weight = float(row.get("sampling_weight", 1.0))
    spans = sentence_spans(row["text"], row.get("spans", []), splitter)
    source_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    if row.get("source_text_sha256", source_hash) != source_hash:
        raise ValueError("source text hash does not match the training-view source text")
    intake_row_id = row.get("intake_row_id") or row.get("id") or content_id
    provenance = segmenter_provenance(splitter)
    segment_rows = []
    for segment_index, (start, end) in enumerate(spans):
        segment_spans = _rebased_spans(row, start, end)
        if not row["text"][start:end].strip():
            continue
        if row.get("supervision") == "annotated_spans_only" and not segment_spans:
            continue
        segment_rows.append(
            {
                **row,
                "text": row["text"][start:end],
                "spans": segment_spans,
                "intake_row_id": intake_row_id,
                "sentence_ordinal": segment_index + 1,
                "source_end": end,
                "source_sentence_count": len(spans),
                "source_start": start,
                "source_text_sha256": source_hash,
                "text_view": "segment",
                "view_index": segment_index,
                "view_offset": start,
                "view_group_id": group_id,
                "source_content_id": content_id,
                "view_base_sampling_weight": base_weight,
                **provenance,
            }
        )

    if (
        len(segment_rows) == 1
        and segment_rows[0]["text"] == row["text"]
        and segment_rows[0]["spans"] == row.get("spans", [])
    ):
        return [
            {
                **segment_rows[0],
                "text_view": "paragraph+segment",
                "text_view_memberships": ["paragraph", "segment"],
                "sampling_weight": base_weight,
            }
        ]

    segment_count = len(segment_rows)
    views = []
    if long_view_alpha:
        views.append(
            {
                **row,
                "text_view": "paragraph",
                "text_view_memberships": ["paragraph"],
                "view_index": 0,
                "view_offset": 0,
                "view_group_id": group_id,
                "source_content_id": content_id,
                "view_base_sampling_weight": base_weight,
                "sampling_weight": base_weight * segment_count * long_view_alpha,
            }
        )
    if long_view_alpha < 1:
        views.extend(
            {
                **segment,
                "text_view_memberships": ["segment"],
                "sampling_weight": base_weight * (1.0 - long_view_alpha),
            }
            for segment in segment_rows
        )
    return views


def placeholder_partition(
    text: str,
    spans: Sequence[tuple[int, int]],
    placeholder_pattern: re.Pattern[str],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Order-insensitive multiset of placeholder multisets induced by segments."""
    partition = []
    for start, end in spans:
        counter = Counter(
            match.group(1) if match.lastindex else match.group(0)
            for match in placeholder_pattern.finditer(text, start, end)
        )
        if counter:
            partition.append(tuple(sorted(counter.items())))
    return tuple(sorted(partition))


def _placeholder_occurrence_keys(text: str, placeholder_pattern: re.Pattern[str]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    occurrences = []
    for match in placeholder_pattern.finditer(text):
        identity = match.group(1) if match.lastindex else match.group(0)
        counts[identity] += 1
        occurrences.append((identity, counts[identity]))
    return occurrences


def placeholder_alignment_metrics(
    source_text: str,
    target_text: str,
    placeholder_pattern: re.Pattern[str],
) -> dict[str, float | int | bool]:
    """Score how far a surviving placeholder order moves from the diagonal.

    Repeated placeholder identities are distinguished by occurrence number.
    Cardinality mismatch is reported but not coerced into an order score; the
    transport survival gate owns that hard failure.
    """
    source = _placeholder_occurrence_keys(source_text, placeholder_pattern)
    target = _placeholder_occurrence_keys(target_text, placeholder_pattern)
    same_occurrences = Counter(source) == Counter(target)
    result: dict[str, float | int | bool] = {
        "placeholder_occurrences": len(source),
        "same_occurrences": same_occurrences,
    }
    if not same_occurrences:
        return result
    if len(source) < 2:
        return {
            **result,
            "mean_normalized_displacement": 0.0,
            "max_normalized_displacement": 0.0,
            "inversion_rate": 0.0,
        }
    target_rank = {occurrence: rank for rank, occurrence in enumerate(target)}
    reordered_ranks = [target_rank[occurrence] for occurrence in source]
    denominator = len(source) - 1
    displacements = [
        abs(source_rank - target_position) / denominator
        for source_rank, target_position in enumerate(reordered_ranks)
    ]
    inversions = sum(
        left > right
        for left_index, left in enumerate(reordered_ranks)
        for right in reordered_ranks[left_index + 1 :]
    )
    pairs = len(source) * (len(source) - 1) // 2
    return {
        **result,
        "mean_normalized_displacement": sum(displacements) / len(displacements),
        "max_normalized_displacement": max(displacements),
        "inversion_rate": inversions / pairs,
    }
