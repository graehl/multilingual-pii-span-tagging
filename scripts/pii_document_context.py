"""Tokenize neighboring text while preserving target-only character offsets."""

from __future__ import annotations

import hashlib
from typing import Any


def adjacent_document_context(
    source: dict[str, Any], sentence_spans: list[tuple[int, int]]
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """Recover complete adjacent sentences from a verified source excerpt."""
    text = source["annotation_context"]
    origin = source["annotation_context_start"]
    start = source["source_document_start"] - origin
    end = start + len(source["text"])
    if start < 0 or text[start:end] != source["text"] or origin + end != source["source_document_end"]:
        raise ValueError("source target does not match its character interval")
    spans = []
    for a, b in sentence_spans:
        piece = text[a:b]
        if piece.strip():
            spans.append((a + len(piece) - len(piece.lstrip()), b - len(piece) + len(piece.rstrip())))
    paragraph = source.get("annotation_context_kind") == "blank_line_paragraph"
    complete = origin == 0 and hashlib.sha256(text.encode()).hexdigest() == source["source_document_sha256"]
    selected: dict[str, str] = {"before": "", "after": ""}
    intervals = {}
    for side in selected:
        eligible = [
            (i, a, b) for i, (a, b) in enumerate(spans) if (b <= start if side == "before" else a >= end)
        ]
        if not eligible:
            continue
        i, a, b = eligible[-1] if side == "before" else eligible[0]
        gap = text[b:start] if side == "before" else text[end:a]
        verified = (
            (i > 0 or origin == 0 or paragraph)
            if side == "before"
            else (i < len(spans) - 1 or complete or paragraph)
        )
        if not gap.strip() and verified:
            selected[side] = text[a:b]
            intervals[side] = [origin + a, origin + b]
    return selected, intervals


def resolve_context_side(requested: str | None, saved: str = "both", *, exact_resume: bool = False) -> str:
    """Inherit a recorded neighbor policy; exact resumes cannot change it."""
    if saved not in {"both", "previous"} or requested not in {None, "both", "previous"}:
        raise ValueError("context side must be both or previous")
    if exact_resume and requested is not None and requested != saved:
        raise ValueError("context side cannot change during exact resume; start a new initialization stage")
    return requested if requested is not None else saved


def resolve_context_configurations(requested, saved=None, *, exact_resume=False):
    """Validate equally sampled immediate-neighbor offsets and preserve resume policy."""
    value = saved if requested is None and exact_resume else requested
    if value is not None:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("context configurations must be a nonempty list of offset lists")
        for offsets in value:
            if (
                not isinstance(offsets, (list, tuple))
                or any(type(offset) is not int for offset in offsets)
                or 0 not in offsets
                or any(offset not in (-1, 0, 1) for offset in offsets)
                or list(offsets) != sorted(set(offsets))
            ):
                raise ValueError(
                    "context offsets must be sorted, unique, include 0, and belong to [-1, 0, 1]"
                )
        value = [list(offsets) for offsets in value]
        if len({tuple(offsets) for offsets in value}) != len(value):
            raise ValueError("context configurations must be distinct; each is sampled equally")
    if exact_resume and value != saved:
        raise ValueError(
            "context configurations cannot change during exact resume; start a new initialization stage"
        )
    return value


def resolve_context_weights(requested, configurations, saved=None, *, exact_resume=False):
    """Normalized sampling weights for the context configurations; None samples them equally."""
    value = saved if requested is None and exact_resume else requested
    if value is not None:
        if configurations is None:
            raise ValueError("--context-configuration-weights requires --context-configurations")
        if (
            not isinstance(value, (list, tuple))
            or len(value) != len(configurations)
            or any(isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0 for w in value)
        ):
            raise ValueError("context configuration weights must be one positive number per configuration")
        total = float(sum(value))
        value = [float(w) / total for w in value]
    if exact_resume and value != saved:
        raise ValueError(
            "context configuration weights cannot change during exact resume; start a new initialization stage"
        )
    return value


def check_context_mixture(context_field, side: str, configurations) -> None:
    """A context mixture needs a context field and may not select a hidden neighbor."""
    if configurations is None:
        return
    if not context_field:
        raise ValueError("--context-configurations requires --context-field")
    if side == "previous" and any(1 in offsets for offsets in configurations):
        raise ValueError("--context-side previous forbids the next neighbor (1) in --context-configurations")


# Optional flag on a structured context: the target is known to be its document's
# first sentence. Absent means "no previous sentence given", which also covers a
# previous sentence that exists but is unknown or withheld.
DOCUMENT_START = "document_start"


def document_context_parts(context) -> tuple[str, str, bool]:
    """Validate a structured context and return (before, after, document_start)."""
    if (
        not isinstance(context, dict)
        or not {"before", "after"} <= set(context) <= {"before", "after", DOCUMENT_START}
        or not isinstance(context["before"], str)
        or not isinstance(context["after"], str)
    ):
        raise ValueError(
            "document context requires string before and after fields and an optional document_start flag"
        )
    start = context.get(DOCUMENT_START, False)
    if not isinstance(start, bool):
        raise ValueError("document_start must be a boolean")
    if start and context["before"].strip():
        raise ValueError("a document's first sentence has no previous sentence")
    return context["before"], context["after"], start


def select_document_context(context: dict[str, str] | str, offsets: list[int]):
    """Omit unavailable or unselected neighbors without adding placeholder tokens.

    A known document start is previous-side information: it is kept exactly
    when the previous neighbor is selected.
    """
    if isinstance(context, str):
        return context if -1 in offsets else ""
    before, after, start = document_context_parts(context)
    selected = {"before": before if -1 in offsets else "", "after": after if 1 in offsets else ""}
    if start and -1 in offsets:
        selected[DOCUMENT_START] = True
    return selected


def encode_document_context(
    tokenizer: Any,
    text: str,
    context: dict[str, str] | str,
    max_length: int,
    separator: str = "\n\n",
    *,
    side: str = "both",
    document_start_marker: bool = False,
) -> tuple[dict[str, list[int]], list[tuple[int, int]]] | None:
    """Keep the target and nearest context tokens in one pretrained input template.

    Return None when there is no context or the target alone exhausts capacity;
    the caller then uses its ordinary target-windowing path. The two neighbors
    split spare capacity equally, transferring unused capacity to the other side.
    Previous-only selection happens here, after any target splitting has inserted
    within-row neighbors; omitted successors cannot affect target tokenization.

    With ``document_start_marker``, a context flagged as a document start puts
    two separator tokens after the start token, ``<s></s></s> target``: the
    pretrained pair template with an empty first segment, distinct from the bare
    ``<s> target`` that means no previous sentence is given. Without it the flag
    is ignored, so models trained before the marker see unchanged inputs.
    """
    side = resolve_context_side(side)
    if isinstance(context, str):
        if not context.strip() or len(tokenizer(text, truncation=False)["input_ids"]) >= max_length:
            return None
        prefix = context.strip() + separator
        previous_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            encoded = tokenizer(
                prefix + text, truncation=True, max_length=max_length, return_offsets_mapping=True
            )
        finally:
            tokenizer.truncation_side = previous_side
        shift = len(prefix)
        offsets = [
            (0, 0) if b <= a or a < shift else (a - shift, b - shift)
            for a, b in encoded.pop("offset_mapping")
        ]
        if not any(b > a for a, b in offsets):
            raise ValueError("context truncation left no target tokens")
        return encoded, offsets
    before, after, document_start = document_context_parts(context)
    before = before.strip()
    after = after.strip() if side == "both" else ""
    document_start = document_start and document_start_marker
    if not before and not after and not document_start:
        return None
    prefix = before + "\n\n" if before else ""
    suffix = "\n\n" + after if after else ""
    combined = prefix + text + suffix
    start, end = len(prefix), len(prefix) + len(text)
    encoded = tokenizer(
        combined,
        truncation=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        verbose=False,
    )
    raw_offsets = encoded.pop("offset_mapping")
    special = encoded.pop("special_tokens_mask")
    encoded = {key: list(values) for key, values in encoded.items()}
    if document_start:
        if not special or not special[0] or encoded["input_ids"][0] != tokenizer.bos_token_id:
            raise ValueError("document-start marker needs an input that opens with the start token")
        unexpected = set(encoded) - {"input_ids", "attention_mask"}
        if unexpected:
            raise ValueError(f"document-start marker cannot extend {sorted(unexpected)}")
        encoded["input_ids"][1:1] = [tokenizer.sep_token_id] * 2
        if "attention_mask" in encoded:
            encoded["attention_mask"][1:1] = [1, 1]
        raw_offsets[1:1] = [(0, 0), (0, 0)]
        special[1:1] = [1, 1]
    target, left, right, specials = [], [], [], []
    offsets = []
    for i, ((a, b), is_special) in enumerate(zip(raw_offsets, special, strict=True)):
        offsets.append((0, 0))
        if is_special:
            specials.append(i)
        elif a < end and b > start:
            if (a < start and combined[a:start].strip()) or (b > end and combined[end:b].strip()):
                raise ValueError("context token crosses a non-whitespace target boundary")
            target.append(i)
            offsets[i] = (max(0, a - start), min(len(text), b - start))
        elif b <= start:
            left.append(i)
        else:
            right.append(i)
    if not target:
        raise ValueError("document context has no target tokens")
    spare = max_length - len(target) - len(specials)
    if spare <= 0:
        return None
    left_count = min(len(left), (spare + 1) // 2)
    right_count = min(len(right), spare - left_count)
    left_count = min(len(left), spare - right_count)
    keep = sorted(specials + target + (left[-left_count:] if left_count else []) + right[:right_count])
    return {key: [values[i] for i in keep] for key, values in encoded.items()}, [offsets[i] for i in keep]
