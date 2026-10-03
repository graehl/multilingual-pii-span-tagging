#!/usr/bin/env python3
"""Train and apply a proposal-preserving character endpoint ranker."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

from scripts.pii_eval import predict_hf_tokcls
from scripts.pii_ontology_v2 import load_ontology

SCHEMA = "pii-character-boundary-refiner-v1"
ENDPOINTS = ("start", "end")
DEFAULT_RADIUS = 1
DEFAULT_CONTEXT = 3
DEFAULT_HASH_BUCKETS = 1 << 18
# Template 1 is the deployed feature set. Template 2 adds endpoint-conjoined
# adjacent-character features so start and end can weigh the same character
# differently. Template 3 adds endpoint-conjoined skip-gram pairs (one position
# skipped) over characters and categories. Template 4 adds language-conjoined
# boundary and affix bigrams (particles, case suffixes, proclitics) and SCRIPT
# v1 script-group features. Deployed consumers implement only 1.
FEATURE_TEMPLATES = (1, 2, 3, 4)
DEFAULT_GROUPS_PER_LANGUAGE = 5000
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_SEED = 155


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl_new(path: Path, rows: Iterable[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def span_values(span: Sequence | dict, *, prediction: bool = False) -> tuple[int, int, str]:
    if isinstance(span, dict):
        label_key = "label" if prediction else "type"
        return int(span["start"]), int(span["end"]), str(span[label_key])
    start, end, label = span
    return int(start), int(end), str(label)


def _mix_feature(namespace: int, values: Sequence[int], buckets: int) -> int:
    value = (0xCBF29CE484222325 ^ namespace) & 0xFFFFFFFFFFFFFFFF
    for item in values:
        value ^= int(item) & 0xFFFFFFFFFFFFFFFF
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value % buckets


def _character_value(text: str, index: int) -> int:
    if index < 0:
        return 0x110000
    if index >= len(text):
        return 0x110001
    return ord(text[index])


def _category_value(character: int) -> int:
    if character > 0x10FFFF:
        return character - 0x110000
    category = unicodedata.category(chr(character))
    return (ord(category[0]) << 8) | ord(category[1])


@functools.cache
def _string_value(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _script_group_value(character: int) -> int:
    """SCRIPT v1 (script, supercategory) group; text edges and unassigned scalars get fixed values."""
    if character > 0x10FFFF:
        return character - 0x110000
    from scripts.pii_character_projection import load_script_v1_character_groups

    group = load_script_v1_character_groups().get(chr(character))
    return 2 if group is None else _string_value("\x1f".join(group))


def _collapsed_window(text: str, position: int, context: int) -> dict[int, int]:
    """Characters at offsets -context..context-1, whitespace runs collapsed to one space."""
    window: dict[int, int] = {}
    for direction, offsets in ((-1, range(-1, -context - 1, -1)), (1, range(0, context))):
        index = position - 1 if direction < 0 else position
        for relative in offsets:
            if not 0 <= index < len(text):
                window[relative] = _character_value(text, index)
                index += direction
                continue
            if text[index].isspace():
                window[relative] = ord(" ")
                while 0 <= index < len(text) and text[index].isspace():
                    index += direction
            else:
                window[relative] = ord(text[index])
                index += direction
    # Feature order follows offset order, as in the uncollapsed window.
    return dict(sorted(window.items()))


def boundary_feature_ids(
    text: str,
    position: int,
    endpoint: str,
    label_id: int,
    *,
    context: int = DEFAULT_CONTEXT,
    buckets: int = DEFAULT_HASH_BUCKETS,
    template: int = 1,
    language: str = "",
    collapse_whitespace: bool = False,
) -> tuple[int, ...]:
    """Stable hashed local-character features for one half-open endpoint.

    With `collapse_whitespace`, each whitespace run on either side of the
    boundary counts as one plain space, so the context window reaches past
    repeated spaces instead of being consumed by them.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    if template not in FEATURE_TEMPLATES:
        raise ValueError(f"unknown boundary feature template {template!r}")
    if template >= 4 and not language:
        raise ValueError("feature template 4 needs the row language")
    if not 0 <= position <= len(text):
        raise ValueError(f"endpoint {position} is outside text of length {len(text)}")
    endpoint_id = ENDPOINTS.index(endpoint)
    features = [
        _mix_feature(0, (), buckets),
        _mix_feature(1, (endpoint_id,), buckets),
        _mix_feature(2, (label_id,), buckets),
        _mix_feature(3, (endpoint_id, label_id), buckets),
    ]
    characters = (
        _collapsed_window(text, position, context)
        if collapse_whitespace
        else {relative: _character_value(text, position + relative) for relative in range(-context, context)}
    )
    for relative, character in characters.items():
        features.append(_mix_feature(10 + relative + context, (character,), buckets))
        features.append(_mix_feature(20 + relative + context, (_category_value(character),), buckets))
    left = characters[-1]
    right = characters[0]
    features.extend(
        (
            _mix_feature(40, (left, right), buckets),
            _mix_feature(41, (_category_value(left), _category_value(right)), buckets),
            _mix_feature(42, (label_id, left, right), buckets),
            _mix_feature(43, (endpoint_id, left, right), buckets),
        )
    )
    if context >= 2:
        features.extend(
            (
                _mix_feature(44, (characters[-2], left, right), buckets),
                _mix_feature(45, (left, right, characters[1]), buckets),
            )
        )
    if template >= 2:
        features.extend(
            (
                _mix_feature(50, (endpoint_id, _category_value(left)), buckets),
                _mix_feature(51, (endpoint_id, _category_value(right)), buckets),
                _mix_feature(52, (endpoint_id, left), buckets),
                _mix_feature(53, (endpoint_id, right), buckets),
                _mix_feature(
                    54, (endpoint_id, label_id, _category_value(left), _category_value(right)), buckets
                ),
            )
        )
    if template >= 3:
        if context < 3:
            raise ValueError("feature templates 3 and 4 need context of at least 3")
        # (a, b) pairs skip exactly one position between them.
        skips = ((-2, 0), (-1, 1), (-3, -1), (0, 2))
        for index, (a, b) in enumerate(skips):
            features.append(_mix_feature(60 + index, (endpoint_id, characters[a], characters[b]), buckets))
            features.append(
                _mix_feature(
                    64 + index,
                    (endpoint_id, _category_value(characters[a]), _category_value(characters[b])),
                    buckets,
                )
            )
    if template == 4:
        lang = _string_value(language)
        left_script = _script_group_value(left)
        right_script = _script_group_value(right)
        features.extend(
            (
                _mix_feature(80, (lang, endpoint_id, left, right), buckets),
                # Two characters before and after the boundary: span-final
                # particles/suffixes and span-initial proclitics per language.
                _mix_feature(81, (lang, endpoint_id, characters[-2], left), buckets),
                _mix_feature(82, (lang, endpoint_id, right, characters[1]), buckets),
                _mix_feature(83, (lang, endpoint_id, _category_value(left), _category_value(right)), buckets),
                _mix_feature(84, (endpoint_id, left_script, right_script), buckets),
                _mix_feature(85, (endpoint_id, label_id, int(left_script != right_script)), buckets),
            )
        )
    return tuple(features)


_OPENING = "Ps"
_CLOSING = "Pe"


def bracket_imbalance(text: str) -> int:
    """Unmatched opening plus closing Unicode brackets (categories Ps/Pe) in ``text``.

    Brackets are matched by nesting depth, not by glyph pair. Quotation marks
    are ignored because their opening and closing roles differ by language.
    """
    depth = unmatched_closing = 0
    for character in text:
        category = unicodedata.category(character)
        if category == _OPENING:
            depth += 1
        elif category == _CLOSING:
            if depth:
                depth -= 1
            else:
                unmatched_closing += 1
    return depth + unmatched_closing


def strip_whitespace_edges(rows: Sequence[dict], predictions: Sequence[dict]) -> list[dict]:
    """Trim leading and trailing whitespace from every proposal before refinement.

    A proposal that is entirely whitespace is kept unchanged, so span count and
    types are preserved like the refiner itself.
    """
    output = []
    for row, prediction in zip(rows, predictions, strict=True):
        if row["id"] != prediction["id"]:
            raise ValueError("prediction rows do not align with texts")
        text = row["text"]
        spans = []
        for span in prediction["preds"]:
            start, end, _label = span_values(span, prediction=True)
            new_start, new_end = start, end
            while new_start < new_end and text[new_start].isspace():
                new_start += 1
            while new_end > new_start and text[new_end - 1].isspace():
                new_end -= 1
            spans.append({**span, "start": new_start, "end": new_end} if new_start < new_end else dict(span))
        output.append({**prediction, "preds": spans})
    return output


def _removes_word_character(text: str, start: int, end: int, new_start: int, new_end: int) -> bool:
    """Whether moving [start, end) to [new_start, new_end) drops a letter, mark or digit."""
    removed = text[start:new_start] + text[new_end:end]
    return any(unicodedata.category(character)[0] in "LMN" for character in removed)


def span_pair_feature_ids(
    text: str,
    start: int,
    end: int,
    label_id: int,
    *,
    buckets: int = DEFAULT_HASH_BUCKETS,
) -> tuple[int, ...]:
    """Hashed features that see a candidate span's start and end together.

    They let one-sided delimiters be scored as such: a span whose first
    character is an opening bracket without a matching closing one, or whose
    outside neighbours form a bracket pair the span does not include.
    """
    if not 0 <= start < end <= len(text):
        raise ValueError(f"span {start}:{end} is outside text of length {len(text)}")
    first = ord(text[start])
    last = ord(text[end - 1])
    before = _character_value(text, start - 1)
    after = _character_value(text, end)
    imbalance = min(bracket_imbalance(text[start:end]), 2)
    return (
        _mix_feature(70, (label_id, first, last), buckets),
        _mix_feature(71, (_category_value(first), _category_value(last)), buckets),
        _mix_feature(72, (before, after), buckets),
        _mix_feature(73, (_category_value(before), _category_value(after)), buckets),
        _mix_feature(74, (label_id, imbalance), buckets),
        _mix_feature(75, (first, last), buckets),
        _mix_feature(
            76,
            (_category_value(before), _category_value(first), _category_value(last), _category_value(after)),
            buckets,
        ),
    )


@dataclass(frozen=True)
class SpanTrainingGroup:
    row_index: int
    language: str
    label: str
    candidates: tuple[tuple[int, int], ...]
    target_index: int
    # The tagger proposal a proposal-centered group surrounds; None for gold-centered groups.
    source: tuple[int, int] | None = None


def span_training_groups(rows: Sequence[dict], radius: int = DEFAULT_RADIUS) -> list[SpanTrainingGroup]:
    """One group per gold span: every (start, end) within ±radius, other same-type gold excluded."""
    if radius <= 0:
        raise ValueError("character endpoint radius must be positive")
    groups = []
    for row_index, row in enumerate(rows):
        text = str(row["text"])
        spans = [span_values(span) for span in row["spans"]]
        taken = set(spans)
        for start, end, label in spans:
            if not 0 <= start < end <= len(text):
                raise ValueError(f"{row.get('id', row_index)!r}: gold span is outside its text")
            candidates = tuple(
                (a, b)
                for a in range(max(0, start - radius), min(len(text), start + radius) + 1)
                for b in range(max(0, end - radius), min(len(text), end + radius) + 1)
                if a < b and ((a, b) == (start, end) or (a, b, label) not in taken)
            )
            if len(candidates) < 2:
                continue
            groups.append(
                SpanTrainingGroup(
                    row_index=row_index,
                    language=str(row["lang"]),
                    label=label,
                    candidates=candidates,
                    target_index=candidates.index((start, end)),
                )
            )
    return groups


def proposal_span_groups(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    radius: int = DEFAULT_RADIUS,
) -> list[SpanTrainingGroup]:
    """One group per tagger proposal that a ±radius move can make exactly right.

    Candidates surround the proposal, as at decode time, rather than the gold
    span; the target is the nearest same-type gold span reachable within ±radius
    at both endpoints, which is the proposal itself when it is already exact.
    This teaches when to stay as well as where to move. Proposals with no
    reachable same-type gold (type errors, spurious or far-off spans) are not
    boundary work and are skipped.
    """
    if radius <= 0:
        raise ValueError("character endpoint radius must be positive")
    groups = []
    for row_index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        if row["id"] != prediction["id"]:
            raise ValueError("prediction rows do not align with texts")
        text = str(row["text"])
        gold = [span_values(span) for span in row["spans"]]
        taken = set(gold)
        for span in prediction["preds"]:
            start, end, label = span_values(span, prediction=True)
            reachable = [
                (x, y)
                for x, y, t in gold
                if t == label and abs(x - start) <= radius and abs(y - end) <= radius
            ]
            if not reachable:
                continue
            target = (
                (start, end)
                if (start, end) in reachable
                else min(reachable, key=lambda g: (abs(g[0] - start) + abs(g[1] - end), g))
            )
            candidates = tuple(
                (a, b)
                for a in range(max(0, start - radius), min(len(text), start + radius) + 1)
                for b in range(max(0, end - radius), min(len(text), end + radius) + 1)
                if a < b and ((a, b) == target or (a, b, label) not in taken)
            )
            groups.append(
                SpanTrainingGroup(
                    row_index=row_index,
                    language=str(row["lang"]),
                    label=label,
                    candidates=candidates,
                    target_index=candidates.index(target),
                    source=(start, end),
                )
            )
    return groups


def joint_candidate_feature_ids(
    text: str,
    start: int,
    end: int,
    label_id: int,
    *,
    context: int,
    buckets: int,
    template: int,
    language: str = "",
    collapse_whitespace: bool = False,
) -> tuple[int, ...]:
    """All features scoring one (start, end) candidate: both endpoints plus the pair."""
    common = dict(
        context=context,
        buckets=buckets,
        template=template,
        language=language,
        collapse_whitespace=collapse_whitespace,
    )
    return (
        boundary_feature_ids(text, start, "start", label_id, **common)
        + boundary_feature_ids(text, end, "end", label_id, **common)
        + span_pair_feature_ids(text, start, end, label_id, buckets=buckets)
    )


def span_group_batch(
    groups: Sequence[SpanTrainingGroup],
    rows: Sequence[dict],
    label_ids: dict[str, int],
    *,
    context: int,
    buckets: int,
    device: torch.device,
    template: int = 1,
    collapse_whitespace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch span groups so the ranker's summed score is the joint candidate score."""
    width = max(len(group.candidates) for group in groups)
    feature_rows, masks, targets = [], [], []
    for group in groups:
        if group.label not in label_ids:
            raise ValueError(f"training label {group.label!r} is outside the v2 primary inventory")
        text = rows[group.row_index]["text"]
        features = [
            joint_candidate_feature_ids(
                text,
                a,
                b,
                label_ids[group.label],
                context=context,
                buckets=buckets,
                template=template,
                language=group.language,
                collapse_whitespace=collapse_whitespace,
            )
            for a, b in group.candidates
        ]
        mask = [True] * len(features)
        while len(features) < width:
            features.append(features[group.target_index])
            mask.append(False)
        feature_rows.append(features)
        masks.append(mask)
        targets.append(group.target_index)
    return (
        torch.tensor(feature_rows, dtype=torch.int64, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def score_pair_requests(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    model: HashedBoundaryRanker,
    label_ids: dict[str, int],
    supported_languages: set[str],
    *,
    radius: int,
    buckets: int,
    device: torch.device,
    batch_size: int = 8192,
) -> dict[tuple[int, int, int, str], float]:
    """Joint-feature scores for every (start, end) candidate the decoder can reach."""
    requests = set()
    for row_index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        if row["lang"] not in supported_languages:
            continue
        length = len(row["text"])
        for span in prediction["preds"]:
            start, end, label = span_values(span, prediction=True)
            for a in range(max(0, start - radius), min(length, start + radius) + 1):
                for b in range(max(0, end - radius), min(length, end + radius) + 1):
                    if a < b:
                        requests.add((row_index, a, b, label))
    ordered = sorted(requests)
    scores: dict[tuple[int, int, int, str], float] = {}
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(ordered), batch_size):
            batch = ordered[begin : begin + batch_size]
            feature_ids = torch.tensor(
                [
                    span_pair_feature_ids(rows[row_index]["text"], a, b, label_ids[label], buckets=buckets)
                    for row_index, a, b, label in batch
                ],
                dtype=torch.int64,
                device=device,
            )
            scores.update(zip(batch, model(feature_ids).float().cpu().tolist(), strict=True))
    return scores


class HashedBoundaryRanker(nn.Module):
    """A linear local-context ranker whose zero state is the identity decoder."""

    def __init__(self, buckets: int = DEFAULT_HASH_BUCKETS) -> None:
        super().__init__()
        if buckets <= 0:
            raise ValueError("hash bucket count must be positive")
        self.weights = nn.Embedding(buckets, 1)
        nn.init.zeros_(self.weights.weight)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        return self.weights(feature_ids.long()).squeeze(-1).sum(dim=-1)


class _OnnxBoundaryRanker(nn.Module):
    def __init__(self, ranker: HashedBoundaryRanker) -> None:
        super().__init__()
        self.ranker = ranker

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        return self.ranker(feature_ids).unsqueeze(-1)


@dataclass(frozen=True)
class TrainingGroup:
    row_index: int
    language: str
    position: int
    endpoint: str
    label: str
    candidates: tuple[int, ...]
    target_index: int


def boundary_training_groups(rows: Sequence[dict], radius: int = DEFAULT_RADIUS) -> list[TrainingGroup]:
    """Make gold endpoint groups without treating another gold endpoint as negative."""
    if radius <= 0:
        raise ValueError("character endpoint radius must be positive")
    groups = []
    for row_index, row in enumerate(rows):
        text = str(row["text"])
        spans = [span_values(span) for span in row["spans"]]
        positives = {
            endpoint: {start if endpoint == "start" else end for start, end, _label in spans}
            for endpoint in ENDPOINTS
        }
        for start, end, label in spans:
            if not 0 <= start < end <= len(text):
                raise ValueError(f"{row.get('id', row_index)!r}: gold span is outside its text")
            for endpoint, position in (("start", start), ("end", end)):
                candidates = tuple(
                    candidate
                    for candidate in range(max(0, position - radius), min(len(text), position + radius) + 1)
                    if candidate == position or candidate not in positives[endpoint]
                )
                if len(candidates) < 2:
                    continue
                groups.append(
                    TrainingGroup(
                        row_index=row_index,
                        language=str(row["lang"]),
                        position=position,
                        endpoint=endpoint,
                        label=label,
                        candidates=candidates,
                        target_index=candidates.index(position),
                    )
                )
    return groups


def sample_groups_by_language(
    groups: Sequence[TrainingGroup],
    limit: int,
    seed: int,
) -> tuple[list[TrainingGroup], dict[str, int]]:
    if limit <= 0:
        raise ValueError("per-language endpoint-group limit must be positive")
    by_language: dict[str, list[TrainingGroup]] = defaultdict(list)
    for group in groups:
        by_language[group.language].append(group)
    selected = []
    for language, candidates in sorted(by_language.items()):
        if len(candidates) > limit:
            language_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(language))
            candidates = random.Random(language_seed).sample(candidates, limit)
        selected.extend(candidates)
    return selected, dict(sorted(Counter(group.language for group in selected).items()))


def group_batch(
    groups: Sequence[TrainingGroup],
    rows: Sequence[dict],
    label_ids: dict[str, int],
    *,
    context: int,
    buckets: int,
    device: torch.device,
    template: int = 1,
    collapse_whitespace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_rows = []
    masks = []
    targets = []
    width = max(len(group.candidates) for group in groups)
    for group in groups:
        if group.label not in label_ids:
            raise ValueError(f"training label {group.label!r} is outside the v2 primary inventory")
        features = [
            boundary_feature_ids(
                rows[group.row_index]["text"],
                candidate,
                group.endpoint,
                label_ids[group.label],
                context=context,
                buckets=buckets,
                template=template,
                language=group.language,
                collapse_whitespace=collapse_whitespace,
            )
            for candidate in group.candidates
        ]
        mask = [True] * len(features)
        while len(features) < width:
            features.append(features[group.target_index])
            mask.append(False)
        feature_rows.append(features)
        masks.append(mask)
        targets.append(group.target_index)
    return (
        torch.tensor(feature_rows, dtype=torch.int64, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def score_endpoint_requests(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    model: HashedBoundaryRanker,
    label_ids: dict[str, int],
    supported_languages: set[str],
    *,
    radius: int,
    context: int,
    buckets: int,
    device: torch.device,
    batch_size: int = 8192,
    template: int = 1,
    collapse_whitespace: bool = False,
) -> dict[tuple[int, int, str, str], float]:
    requests = set()
    for row_index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        if row["id"] != prediction["id"]:
            raise ValueError("prediction rows do not align with texts")
        if row["lang"] not in supported_languages:
            continue
        text_length = len(row["text"])
        for span in prediction["preds"]:
            start, end, label = span_values(span, prediction=True)
            if label not in label_ids:
                raise ValueError(f"prediction label {label!r} is outside the v2 primary inventory")
            for endpoint, position in (("start", start), ("end", end)):
                for candidate in range(
                    max(0, position - radius),
                    min(text_length, position + radius) + 1,
                ):
                    requests.add((row_index, candidate, endpoint, label))
    ordered = sorted(requests)
    scores: dict[tuple[int, int, str, str], float] = {}
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(ordered), batch_size):
            batch = ordered[begin : begin + batch_size]
            feature_ids = torch.tensor(
                [
                    boundary_feature_ids(
                        rows[row_index]["text"],
                        position,
                        endpoint,
                        label_ids[label],
                        context=context,
                        buckets=buckets,
                        template=template,
                        language=str(rows[row_index]["lang"]),
                        collapse_whitespace=collapse_whitespace,
                    )
                    for row_index, position, endpoint, label in batch
                ],
                dtype=torch.int64,
                device=device,
            )
            values = model(feature_ids).float().cpu().tolist()
            scores.update(zip(batch, values, strict=True))
    return scores


def refine_row_spans(
    text: str,
    predictions: Sequence[dict],
    score: Callable[[int, str, str, int], float],
    *,
    radius: int = DEFAULT_RADIUS,
    movement_penalty: float = 0.0,
    paired_movement_penalty: float | None = None,
    bracket_penalty: float = 0.0,
    pair_score: Callable[[int, int, str], float] | None = None,
    protect_word_characters: bool = False,
    endpoint_radii: tuple[int, int] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Jointly shift endpoints without joining source-disjoint overlap components.

    With `protect_word_characters`, a candidate may never remove a letter, mark
    or digit from its proposal: only punctuation, symbols and spaces are trimmed.
    `endpoint_radii=(start_radius, end_radius)` searches ±start_radius at the
    start and ±end_radius at the end, each at most `radius` (the scored range);
    the default is ±radius at both.

    `score(candidate, endpoint, label, source)` scores one candidate character for the
    proposal endpoint originally at `source`. Each character of endpoint movement
    costs `movement_penalty`. When `paired_movement_penalty` is set, a start and
    end moving in opposite directions (trimming or widening both sides) cost that
    amount per paired character instead of two independent movements; one-sided
    and same-direction shifts are unchanged. `bracket_penalty` is charged per
    unmatched bracket inside a candidate span. `pair_score(start, end, label)`
    adds a learned score for features that see both endpoints together.
    """
    source = sorted(
        (span_values(span, prediction=True) for span in predictions),
        key=lambda item: (item[0], item[1], item[2]),
    )
    source_overlap = [left[1] > right[0] for left, right in zip(source, source[1:])]
    source_multiplicity = Counter(source)
    telemetry = Counter(
        proposals=len(source),
        changed_spans=0,
        changed_starts=0,
        changed_ends=0,
        overlapping_source_rows=int(any(source_overlap)),
        overlapping_refined_rows=0,
        duplicate_source_spans=len(source) - len(set(source)),
        duplicate_refined_spans=0,
    )
    start_radius, end_radius = (radius, radius) if endpoint_radii is None else endpoint_radii
    if not (0 <= start_radius <= radius and 0 <= end_radius <= radius):
        raise ValueError(f"endpoint radii {endpoint_radii} must lie within 0..{radius}")
    alternatives = []
    for start, end, label in source:
        candidates = []
        for candidate_start in range(max(0, start - start_radius), min(len(text), start + start_radius) + 1):
            for candidate_end in range(max(0, end - end_radius), min(len(text), end + end_radius) + 1):
                if candidate_start >= candidate_end:
                    continue
                if protect_word_characters and _removes_word_character(
                    text, start, end, candidate_start, candidate_end
                ):
                    continue
                start_shift = candidate_start - start
                end_shift = candidate_end - end
                movement = abs(start_shift) + abs(end_shift)
                cost = movement_penalty * movement
                if paired_movement_penalty is not None and start_shift * end_shift < 0:
                    paired = min(abs(start_shift), abs(end_shift))
                    cost += (paired_movement_penalty - 2 * movement_penalty) * paired
                if bracket_penalty:
                    cost += bracket_penalty * bracket_imbalance(text[candidate_start:candidate_end])
                joint = 0.0 if pair_score is None else pair_score(candidate_start, candidate_end, label)
                candidates.append(
                    (
                        candidate_start,
                        candidate_end,
                        label,
                        score(candidate_start, "start", label, start)
                        + score(candidate_end, "end", label, end)
                        + joint
                        - cost,
                        movement,
                    )
                )
        alternatives.append(candidates)
    states: list[tuple[float, int, list[tuple[int, int, str]]]] = [(0.0, 0, [])]
    for index, candidates in enumerate(alternatives):
        next_states = []
        for candidate_start, candidate_end, label, candidate_score, movement in candidates:
            candidate = (candidate_start, candidate_end, label)
            if index == 0 or source_overlap[index - 1]:
                allowed_multiplicity = max(1, source_multiplicity[candidate])
                compatible = [state for state in states if state[2].count(candidate) < allowed_multiplicity]
            else:
                compatible = [state for state in states if state[2][-1][1] <= candidate_start]
            if not compatible:
                continue
            prior_score, prior_movement, prior_path = max(
                compatible,
                key=lambda state: (state[0], -state[1], state[2]),
            )
            next_states.append(
                (
                    prior_score + candidate_score,
                    prior_movement + movement,
                    [*prior_path, (candidate_start, candidate_end, label)],
                )
            )
        if not next_states:
            raise AssertionError("identity candidates must preserve source overlap components")
        states = next_states
    selected = max(states, key=lambda state: (state[0], -state[1], state[2]))[2]
    telemetry["overlapping_refined_rows"] = int(
        any(left[1] > right[0] for left, right in zip(selected, selected[1:]))
    )
    telemetry["duplicate_refined_spans"] = len(selected) - len(set(selected))
    for original, refined in zip(source, selected, strict=True):
        if original != refined:
            telemetry["changed_spans"] += 1
            telemetry["changed_starts"] += original[0] != refined[0]
            telemetry["changed_ends"] += original[1] != refined[1]
    return [{"start": start, "end": end, "label": label} for start, end, label in selected], dict(telemetry)


def refine_rows_with_scores(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    score_of: Callable[[int, int, str, str, int], float],
    *,
    supported_languages: set[str],
    radius: int,
    movement_penalty: float,
    paired_movement_penalty: float | None = None,
    bracket_penalty: float = 0.0,
    pair_score_of: Callable[[int, int, int, str], float] | None = None,
    protect_word_characters: bool = False,
    endpoint_radii: tuple[int, int] | None = None,
) -> tuple[list[dict], dict]:
    """Refine rows in supported languages, identity-route the rest, and aggregate telemetry.

    `score_of(row_index, candidate, endpoint, label, source)` scores one candidate
    character for the proposal endpoint originally at `source`;
    `pair_score_of(row_index, start, end, label)` optionally scores the pair.
    """
    totals = Counter()
    per_language: dict[str, Counter] = defaultdict(Counter)
    output = []
    for row_index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        if row["id"] != prediction["id"]:
            raise ValueError("prediction rows do not align with texts")
        if row["lang"] in supported_languages:
            refined, telemetry = refine_row_spans(
                row["text"],
                prediction["preds"],
                lambda position, endpoint, label, source, row_index=row_index: score_of(
                    row_index, position, endpoint, label, source
                ),
                radius=radius,
                movement_penalty=movement_penalty,
                paired_movement_penalty=paired_movement_penalty,
                bracket_penalty=bracket_penalty,
                protect_word_characters=protect_word_characters,
                endpoint_radii=endpoint_radii,
                pair_score=None
                if pair_score_of is None
                else lambda start, end, label, row_index=row_index: pair_score_of(
                    row_index, start, end, label
                ),
            )
        else:
            refined = [dict(span) for span in prediction["preds"]]
            telemetry = {
                "proposals": len(refined),
                "changed_spans": 0,
                "changed_starts": 0,
                "changed_ends": 0,
                "overlapping_source_rows": 0,
            }
            telemetry["identity_language_rows"] = 1
        totals.update(telemetry)
        per_language[row["lang"]].update(telemetry)
        output.append({**prediction, "preds": refined})
    return output, {
        "aggregate": dict(totals),
        "per_language": {language: dict(counts) for language, counts in sorted(per_language.items())},
    }


def refine_prediction_rows(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    model: HashedBoundaryRanker,
    config: dict,
    *,
    device: torch.device,
    endpoint_scores: dict[tuple[int, int, str, str], float] | None = None,
    pair_scores: dict[tuple[int, int, int, str], float] | None = None,
) -> tuple[list[dict], dict]:
    """Refine proposals with a trained ranker and config.

    With `trim_whitespace` in the config, proposal edges are stripped of
    whitespace first; precomputed scores must then come from stripped proposals.
    """
    label_ids = {label: index for index, label in enumerate(config["labels"])}
    supported = set(config["supported_languages"])
    if config.get("trim_whitespace"):
        predictions = strip_whitespace_edges(rows, predictions)
    scores = endpoint_scores
    if scores is None:
        scores = score_endpoint_requests(
            rows,
            predictions,
            model,
            label_ids,
            supported,
            radius=int(config["radius"]),
            context=int(config["context"]),
            buckets=int(config["hash_buckets"]),
            device=device,
            template=int(config.get("feature_template", 1)),
            collapse_whitespace=bool(config.get("collapse_whitespace", False)),
        )
    if config.get("pair_features") and pair_scores is None:
        pair_scores = score_pair_requests(
            rows,
            predictions,
            model,
            label_ids,
            supported,
            radius=int(config["radius"]),
            buckets=int(config["hash_buckets"]),
            device=device,
        )
    paired = config.get("paired_movement_penalty")
    return refine_rows_with_scores(
        rows,
        predictions,
        lambda row_index, position, endpoint, label, _source: scores[(row_index, position, endpoint, label)],
        supported_languages=supported,
        radius=int(config["radius"]),
        movement_penalty=float(config.get("movement_penalty", 0.0)),
        paired_movement_penalty=None if paired is None else float(paired),
        bracket_penalty=float(config.get("bracket_penalty", 0.0)),
        protect_word_characters=bool(config.get("protect_word_characters", False)),
        endpoint_radii=None if config.get("endpoint_radii") is None else tuple(config["endpoint_radii"]),
        pair_score_of=None
        if not config.get("pair_features")
        else lambda row_index, start, end, label: pair_scores[(row_index, start, end, label)],
    )


def exact_typed_counts_by_document(
    rows: Sequence[dict], predictions: Sequence[dict]
) -> list[tuple[int, int, int]]:
    counts = []
    for row, prediction in zip(rows, predictions, strict=True):
        if row["id"] != prediction["id"]:
            raise ValueError("prediction rows do not align with gold")
        gold = {span_values(span) for span in row["spans"]}
        predicted = {span_values(span, prediction=True) for span in prediction["preds"]}
        counts.append((len(gold & predicted), len(predicted), len(gold)))
    return counts


def score_from_counts(counts: Iterable[tuple[int, int, int]]) -> dict[str, float | int]:
    true_positives = predicted_count = gold_count = 0
    for document_true_positives, document_predicted, document_gold in counts:
        true_positives += document_true_positives
        predicted_count += document_predicted
        gold_count += document_gold
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "predicted": predicted_count,
        "gold": gold_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def exact_typed_score(rows: Sequence[dict], predictions: Sequence[dict]) -> dict[str, float | int]:
    return score_from_counts(exact_typed_counts_by_document(rows, predictions))


def document_bootstrap_f1_standard_error(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    *,
    samples: int,
    seed: int,
) -> float:
    if samples < 2:
        raise ValueError("bootstrap requires at least two samples")
    counts = exact_typed_counts_by_document(rows, predictions)
    if not counts:
        raise ValueError("bootstrap requires at least one document")
    generator = random.Random(seed)
    scores = []
    for _ in range(samples):
        sample = (counts[generator.randrange(len(counts))] for _ in counts)
        scores.append(float(score_from_counts(sample)["f1"]))
    mean = sum(scores) / len(scores)
    return math.sqrt(sum((score - mean) ** 2 for score in scores) / (len(scores) - 1))


def select_earliest_epoch_within_one_standard_error(
    curve: Sequence[dict], standard_error: float
) -> dict[str, float | int | list[int]]:
    if standard_error < 0:
        raise ValueError("standard error must be nonnegative")
    numeric_best = max(curve, key=lambda record: (float(record["selector"]["f1"]), -record["epoch"]))
    best_f1 = float(numeric_best["selector"]["f1"])
    threshold = best_f1 - standard_error
    eligible_epochs = sorted(
        int(record["epoch"]) for record in curve if float(record["selector"]["f1"]) >= threshold
    )
    return {
        "numeric_best_epoch": int(numeric_best["epoch"]),
        "numeric_best_f1": best_f1,
        "standard_error": standard_error,
        "threshold": threshold,
        "eligible_epochs": eligible_epochs,
        "selected_epoch": eligible_epochs[0],
    }


def select_largest_penalty_within_one_standard_error(
    curve: Sequence[dict], standard_error: float
) -> dict[str, float | list[float]]:
    if standard_error < 0:
        raise ValueError("standard error must be nonnegative")
    numeric_best = max(
        curve,
        key=lambda record: (float(record["selector"]["f1"]), float(record["movement_penalty"])),
    )
    best_f1 = float(numeric_best["selector"]["f1"])
    threshold = best_f1 - standard_error
    eligible_penalties = sorted(
        float(record["movement_penalty"]) for record in curve if float(record["selector"]["f1"]) >= threshold
    )
    return {
        "numeric_best_penalty": float(numeric_best["movement_penalty"]),
        "numeric_best_f1": best_f1,
        "standard_error": standard_error,
        "threshold": threshold,
        "eligible_penalties": eligible_penalties,
        "selected_penalty": eligible_penalties[-1],
    }


def candidate_oracle_score(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    radius: int,
) -> dict[str, float | int]:
    true_positives = predicted_count = gold_count = 0
    for row, prediction in zip(rows, predictions, strict=True):
        gold = [span_values(span) for span in row["spans"]]
        predicted = [span_values(span, prediction=True) for span in prediction["preds"]]
        edges = [
            [
                pred_index
                for pred_index, candidate in enumerate(predicted)
                if item[2] == candidate[2]
                and abs(item[0] - candidate[0]) <= radius
                and abs(item[1] - candidate[1]) <= radius
            ]
            for item in gold
        ]
        pred_to_gold: dict[int, int] = {}

        def augment(gold_index: int, seen: set[int]) -> bool:
            for pred_index in edges[gold_index]:
                if pred_index in seen:
                    continue
                seen.add(pred_index)
                if pred_index not in pred_to_gold or augment(pred_to_gold[pred_index], seen):
                    pred_to_gold[pred_index] = gold_index
                    return True
            return False

        true_positives += sum(augment(index, set()) for index in range(len(gold)))
        predicted_count += len(set(predicted))
        gold_count += len(set(gold))
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "predicted": predicted_count,
        "gold": gold_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def validate_partition_receipt(
    receipt_path: Path,
    train_paths: Sequence[Path],
    validation_path: Path,
) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "pii-v2-complete-calibration-corpus":
        raise ValueError("partition receipt has the wrong schema")
    if receipt.get("train_validation_text_overlap") != 0:
        raise ValueError("partition receipt reports train/validation text overlap")
    if receipt.get("validation", {}).get("source_group_overlap") != 0:
        raise ValueError("partition receipt reports train/validation source-group overlap")
    outputs = receipt["outputs"]
    by_name = {path.name: path for path in train_paths}
    expected = {
        outputs["native-v2-complete"]["path"]: outputs["native-v2-complete"]["sha256"],
        outputs["view-native-v2-train"]["path"]: outputs["view-native-v2-train"]["sha256"],
    }
    if set(by_name) != set(expected):
        raise ValueError(f"training files {sorted(by_name)} do not match receipt {sorted(expected)}")
    for name, expected_hash in expected.items():
        if file_sha256(by_name[name]) != expected_hash:
            raise ValueError(f"training file {name} differs from the partition receipt")
    validation = outputs["validation"]
    if validation_path.name != validation["path"] or file_sha256(validation_path) != validation["sha256"]:
        raise ValueError("validation file differs from the partition receipt")
    return receipt


def core_languages(path: Path) -> set[str]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(entry["code"]) for entry in policy["languages"]}


def load_refiner(path: Path, device: torch.device) -> tuple[HashedBoundaryRanker, dict]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError(f"unsupported boundary-refiner schema {config.get('schema')!r}")
    model = HashedBoundaryRanker(int(config["hash_buckets"]))
    model.load_state_dict(load_file(path / "model.safetensors", device=str(device)), strict=True)
    return model.to(device).eval(), config


def boundary_refiner_deployment_config(config: dict) -> dict:
    # Deployed consumers compute template-1 features and independent movement
    # costs; exporting anything else would silently change their decisions.
    if int(config.get("feature_template", 1)) != 1:
        raise ValueError("deployed boundary refiners support only feature template 1")
    if config.get("paired_movement_penalty") is not None:
        raise ValueError("deployed boundary refiners do not implement paired movement penalties")
    unsupported = [
        key
        for key in (
            "bracket_penalty",
            "pair_features",
            "protect_word_characters",
            "trim_whitespace",
            "collapse_whitespace",
        )
        if config.get(key)
    ]
    if unsupported:
        raise ValueError(f"deployed boundary refiners do not implement {unsupported}")
    radius = int(config["radius"])
    start_radius, end_radius = (int(value) for value in (config.get("endpoint_radii") or (radius, radius)))
    return {
        "schema": SCHEMA,
        "schema_version": int(config["schema_version"]),
        "context": int(config["context"]),
        "feature_count": len(
            boundary_feature_ids(
                "",
                0,
                "start",
                0,
                context=int(config["context"]),
                buckets=int(config["hash_buckets"]),
            )
        ),
        "hash_buckets": int(config["hash_buckets"]),
        "start_offsets": list(range(-start_radius, start_radius + 1)),
        "end_offsets": list(range(-end_radius, end_radius + 1)),
        "movement_penalty": float(config.get("movement_penalty", 0.0)),
        "labels": list(config["labels"]),
        "supported_languages": list(config["supported_languages"]),
    }


def export_boundary_refiner(refiner: Path, output: Path) -> dict:
    import numpy as np
    import onnxruntime as ort

    model_path = output / "boundary-refiner.onnx"
    config_path = output / "boundary-refiner.json"
    card_path = output / "boundary-refiner.README.md"
    receipt_path = output / "boundary-refiner-export-receipt.json"
    existing = [path for path in (model_path, config_path, card_path, receipt_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite boundary-refiner artifacts: {existing}")

    model, config = load_refiner(refiner, torch.device("cpu"))
    deployment_config = boundary_refiner_deployment_config(config)
    labels = deployment_config["labels"]
    examples = (
        ("Oslo (Norway).", 4, "end", labels.index("locality")),
        ("CP:24003", 3, "start", labels.index("postal_code")),
        ("Jalapeño €42", 8, "end", labels.index("monetary_amount")),
        ("電話番号", 0, "start", labels.index("phone_number")),
        ("", 0, "end", labels.index("person_name")),
    )
    feature_ids = torch.tensor(
        [
            boundary_feature_ids(
                text,
                position,
                endpoint,
                label_id,
                context=deployment_config["context"],
                buckets=deployment_config["hash_buckets"],
            )
            for text, position, endpoint, label_id in examples
        ],
        dtype=torch.int64,
    )
    wrapped = _OnnxBoundaryRanker(model).eval()
    output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (feature_ids,),
        model_path,
        input_names=["feature_ids"],
        output_names=["scores"],
        dynamic_axes={"feature_ids": {0: "requests"}, "scores": {0: "requests"}},
        opset_version=18,
        dynamo=False,
    )
    config_path.write_text(
        json.dumps(deployment_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with torch.no_grad():
        expected = wrapped(feature_ids).numpy()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(["scores"], {"feature_ids": feature_ids.numpy()})[0]
    max_absolute_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"ONNX boundary-refiner parity failed: max error {max_absolute_error}")

    receipt = {
        "schema": SCHEMA,
        "operation": "export-onnx",
        "source": {
            "refiner": str(refiner.resolve()),
            "model_sha256": file_sha256(refiner / "model.safetensors"),
            "config_sha256": file_sha256(refiner / "config.json"),
        },
        "outputs": {
            model_path.name: {"sha256": file_sha256(model_path)},
            config_path.name: {"sha256": file_sha256(config_path)},
        },
        "validation": {
            "provider": "CPUExecutionProvider",
            "requests": len(examples),
            "max_absolute_error": max_absolute_error,
            "rtol": 1e-6,
            "atol": 1e-6,
        },
    }
    card_path.write_text(
        "# Character-boundary endpoint ranker\n\n"
        "Frozen shared hashed linear ranker for refining existing entity spans. "
        "It does not detect entities or change their types.\n\n"
        f"Input: `feature_ids`, int64 `[requests, {deployment_config['feature_count']}]`. "
        "Output: `scores`, float32 `[requests]`. Dynamic request count; ONNX opset 18.\n\n"
        "Generate features with `boundary_feature_ids` in "
        "`scripts/pii_character_boundary_refiner.py`, preserving Unicode codepoints "
        "and the input text exactly. This component performs no normalization. "
        "Use the companion JSON for hash size, label ordering, character context, "
        "candidate offsets, movement penalty and supported-language gate. "
        "Apply the constrained span decoder after endpoint scoring; the graph "
        "alone does not enforce span validity or non-overlap.\n\n"
        f"Source weights: `{receipt['source']['model_sha256']}`. "
        f"Source configuration: `{receipt['source']['config_sha256']}`.\n\n"
        f"ONNX Runtime {ort.__version__}, CPUExecutionProvider: "
        f"{len(examples)} diverse/empty-text feature cases, maximum absolute "
        f"score difference {max_absolute_error:.9g} versus PyTorch. "
        "This verifies numerical export parity, not language-specific quality. "
        "Quality, training ancestry and selection evidence belong with the "
        "source refiner receipt. Unsupported languages must pass through unchanged.\n",
        encoding="utf-8",
    )
    receipt["outputs"][card_path.name] = {"sha256": file_sha256(card_path)}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def export_onnx(args: argparse.Namespace) -> None:
    receipt = export_boundary_refiner(args.refiner, args.output)
    print(json.dumps(receipt, sort_keys=True))


def validate_fit_output_directory(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(path)
    # run trackers pre-create `<artifact>.meta.md` sidecars for declared outputs
    unexpected = sorted(item.name for item in path.iterdir() if not item.name.endswith(".meta.md"))
    if unexpected:
        raise FileExistsError(f"output directory contains prior artifacts: {unexpected}")


def proposal_predictions(model_path: Path, rows: list[dict], log) -> list[dict]:
    """Primary-span proposals the refiner is selected against.

    A tagger with per-language output biases needs each row's language, which
    only the Ont3 predictor supplies; other taggers keep the original decoder.
    Reference types are not refined and are left out, as in apply-sweep.
    """
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if config.get("pii_language_bias_languages"):
        from scripts.pii_ont3_eval import predict_rows

        predictions, _ = predict_rows(
            model_path,
            rows,
            config.get("pii_predicate_channels", []),
            merge_adjacent_types=frozenset({"person_name", "organization"}),
        )
        references = {"person_reference", "organization_reference"}
        return [
            {
                "id": row["id"],
                "preds": [p for p in prediction["reference_aware_preds"] if p["label"] not in references],
            }
            for row, prediction in zip(rows, predictions, strict=True)
        ]
    decoded = predict_hf_tokcls(
        [row["text"] for row in rows],
        log,
        model_id=str(model_path),
        bioes_project_legal=True,
        bucket_compat="fine",
        bucket_compat_type="preponderance",
        preponderance_weight="nfc-char",
        hf_windowing="token-capacity",
    )
    return [
        {"id": row["id"], "preds": predictions, "latency_s": round(latency, 4)}
        for row, (predictions, latency) in zip(rows, decoded, strict=True)
    ]


def fit(args: argparse.Namespace) -> None:
    validate_fit_output_directory(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("boundary-refiner fitting requires CUDA for selector proposal decoding")
    partition = validate_partition_receipt(args.partition_receipt, args.train, args.validation)
    train_rows = [row for path in args.train for row in read_jsonl(path)]
    validation_rows = read_jsonl(args.validation)
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_validation_rows:
        validation_rows = validation_rows[: args.max_validation_rows]
    for row in train_rows + validation_rows:
        if row.get("supervision") != "complete" or row.get("label_space") != "v2":
            raise ValueError("boundary refinement requires complete native-v2 rows")
    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    if len(train_ids) != len(train_rows) or len(validation_ids) != len(validation_rows):
        raise ValueError("boundary train/validation IDs must be unique")
    if train_ids & validation_ids:
        raise ValueError("boundary train/validation IDs overlap")
    ontology = load_ontology()
    labels = list(ontology.primary_types)
    label_ids = {label: index for index, label in enumerate(labels)}
    all_groups = boundary_training_groups(train_rows, args.radius)
    groups, group_counts = sample_groups_by_language(
        all_groups,
        args.groups_per_language,
        args.seed,
    )
    supported_languages = sorted(group_counts)
    core = core_languages(args.language_round)
    supported_core = sorted(core & set(supported_languages))
    identity_core = sorted(core - set(supported_languages))
    group_total = sum(group_counts.values())
    shares = {language: count / group_total for language, count in group_counts.items()}
    if any(shares[language] < 0.01 for language in supported_languages):
        raise ValueError(f"supported-language endpoint share below 1%: {shares}")

    def log(message: str) -> None:
        line = f"character-boundary-refiner selector: {message}"
        print(line, flush=True)
        headline_file = os.environ.get("AGENTCTL_HEADLINE_FILE")
        if headline_file:
            Path(headline_file).write_text(line + "\n", encoding="utf-8")

    base_predictions = proposal_predictions(args.model, validation_rows, log)
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    model = HashedBoundaryRanker(args.hash_buckets).to(device)
    config = {
        "schema": SCHEMA,
        "schema_version": 1,
        "labels": labels,
        "radius": args.radius,
        "movement_penalty": 0.0,
        "context": args.context,
        "hash_buckets": args.hash_buckets,
        "supported_languages": supported_languages,
        "identity_core_languages": identity_core,
    }
    initial_predictions, initial_telemetry = refine_prediction_rows(
        validation_rows,
        base_predictions,
        model,
        config,
        device=device,
    )
    if initial_predictions != base_predictions:
        raise AssertionError("zero-start boundary ranker must reproduce the no-refiner control")
    base_score = exact_typed_score(validation_rows, base_predictions)
    curve = [
        {
            "epoch": 0,
            "updates": 0,
            "selector": base_score,
            "telemetry": initial_telemetry,
        }
    ]
    candidate_states = {0: copy.deepcopy(model.state_dict())}
    candidate_predictions = {0: initial_predictions}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(groups), generator=generator).tolist()
        loss_sum = 0.0
        seen = 0
        correct = 0
        for begin in range(0, len(order), args.batch_size):
            selected = [groups[index] for index in order[begin : begin + args.batch_size]]
            features, mask, targets = group_batch(
                selected,
                train_rows,
                label_ids,
                context=args.context,
                buckets=args.hash_buckets,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model(features).masked_fill(~mask, float("-inf"))
            loss = F.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()
            updates += 1
            loss_sum += float(loss.detach()) * len(selected)
            seen += len(selected)
            correct += int((scores.argmax(dim=-1) == targets).sum())
        refined, telemetry = refine_prediction_rows(
            validation_rows,
            base_predictions,
            model,
            config,
            device=device,
        )
        selector = exact_typed_score(validation_rows, refined)
        changed = int(telemetry["aggregate"].get("changed_spans", 0))
        record = {
            "epoch": epoch,
            "updates": updates,
            "training_loss": loss_sum / seen,
            "training_group_accuracy": correct / seen,
            "selector": selector,
            "telemetry": telemetry,
        }
        curve.append(record)
        candidate_states[epoch] = copy.deepcopy(model.state_dict())
        candidate_predictions[epoch] = refined
        log(
            f"epoch {epoch}/{args.epochs} train-loss={record['training_loss']:.6f} "
            f"selector-f1={selector['f1']:.6f} changed-spans={changed}"
        )
    numeric_best_epoch = int(
        max(curve, key=lambda record: (float(record["selector"]["f1"]), -record["epoch"]))["epoch"]
    )
    standard_error = document_bootstrap_f1_standard_error(
        validation_rows,
        candidate_predictions[numeric_best_epoch],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    trust_region = select_earliest_epoch_within_one_standard_error(curve, standard_error)
    selected_epoch = int(trust_region["selected_epoch"])
    selected_predictions = candidate_predictions[selected_epoch]
    model.load_state_dict(candidate_states[selected_epoch])
    oracle = candidate_oracle_score(validation_rows, base_predictions, args.radius)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        args.output / "model.safetensors",
    )
    config.update(
        {
            "selected_epoch": selected_epoch,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "groups_per_language_cap": args.groups_per_language,
                "bootstrap_samples": args.bootstrap_samples,
            },
        }
    )
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl_new(args.output / "selector-base.jsonl", base_predictions)
    write_jsonl_new(args.output / "selector-refined.jsonl", selected_predictions)
    receipt = {
        "schema": SCHEMA,
        "status": "mechanism_smoke" if args.max_train_rows or args.max_validation_rows else "development_fit",
        "source": {
            "git_commit": git_commit(),
            "model": str(args.model.resolve()),
            "model_config_sha256": file_sha256(args.model / "config.json"),
            "training": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.train],
            "validation": {
                "path": str(args.validation.resolve()),
                "sha256": file_sha256(args.validation),
            },
            "partition_receipt": {
                "path": str(args.partition_receipt.resolve()),
                "sha256": file_sha256(args.partition_receipt),
                "membership_sha256": partition["validation"]["manifest_sha256"],
            },
        },
        "recipe": config,
        "training_groups": {
            "before_sampling": len(all_groups),
            "selected": len(groups),
            "per_language": group_counts,
            "expected_shares": shares,
            "supported_core_languages": supported_core,
            "identity_core_languages": identity_core,
            "additional_languages": sorted(set(supported_languages) - core),
        },
        "selection": {
            "metric": "exact typed character-span micro F1 on the enlarged complete-native selector",
            "rule": (
                "earliest epoch whose F1 is within one document-bootstrap standard error of the numeric best"
            ),
            "noise_estimate": {
                "unit": "document",
                "samples": args.bootstrap_samples,
                "seed": args.seed,
                **trust_region,
            },
            "curve": curve,
            "selected_epoch": selected_epoch,
            "no_refiner": base_score,
            "radius_one_candidate_oracle": oracle,
            "selected": exact_typed_score(validation_rows, selected_predictions),
        },
        "claim_boundary": (
            "Adaptive external-development mechanism. The character refiner is routed only to "
            "languages with source-faithful complete training spans; remaining core languages "
            "retain the no-refiner decoder. No release claim follows without fresh source-group-"
            "disjoint confirmation."
        ),
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(
        f"selected epoch {selected_epoch} (numeric best {numeric_best_epoch}, "
        f"one-SE floor {trust_region['threshold']:.6f}): {base_score['f1']:.6f} -> "
        f"{receipt['selection']['selected']['f1']:.6f}; oracle={oracle['f1']:.6f}"
    )


def calibrate_movement_penalty(args: argparse.Namespace) -> None:
    validate_fit_output_directory(args.output)
    rows = read_jsonl(args.gold)
    base_predictions = read_jsonl(args.predictions)
    penalties = sorted(set(args.movement_penalty))
    if not penalties or penalties[0] < 0:
        raise ValueError("movement-penalty grid must contain nonnegative values")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA movement-penalty calibration requested without CUDA")
    model, parent_config = load_refiner(args.refiner, device)
    label_ids = {label: index for index, label in enumerate(parent_config["labels"])}
    endpoint_scores = score_endpoint_requests(
        rows,
        base_predictions,
        model,
        label_ids,
        set(parent_config["supported_languages"]),
        radius=int(parent_config["radius"]),
        context=int(parent_config["context"]),
        buckets=int(parent_config["hash_buckets"]),
        device=device,
    )
    curve = []
    candidate_predictions = {}
    candidate_telemetry = {}
    for penalty in penalties:
        candidate_config = {**parent_config, "movement_penalty": penalty}
        refined, telemetry = refine_prediction_rows(
            rows,
            base_predictions,
            model,
            candidate_config,
            device=device,
            endpoint_scores=endpoint_scores,
        )
        selector = exact_typed_score(rows, refined)
        curve.append(
            {
                "movement_penalty": penalty,
                "selector": selector,
                "telemetry": telemetry,
            }
        )
        candidate_predictions[penalty] = refined
        candidate_telemetry[penalty] = telemetry
        print(
            f"character-boundary-refiner calibration: penalty={penalty:g} "
            f"selector-f1={selector['f1']:.6f} "
            f"changed-spans={telemetry['aggregate'].get('changed_spans', 0)}",
            flush=True,
        )
    numeric_best = max(
        curve,
        key=lambda record: (float(record["selector"]["f1"]), float(record["movement_penalty"])),
    )
    numeric_best_penalty = float(numeric_best["movement_penalty"])
    standard_error = document_bootstrap_f1_standard_error(
        rows,
        candidate_predictions[numeric_best_penalty],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    trust_region = select_largest_penalty_within_one_standard_error(curve, standard_error)
    selected_penalty = float(trust_region["selected_penalty"])
    selected_predictions = candidate_predictions[selected_penalty]
    selected_config = copy.deepcopy(parent_config)
    selected_config["movement_penalty"] = selected_penalty
    selected_config["calibration"] = {
        "grid": penalties,
        "rule": (
            "largest movement penalty whose F1 is within one document-bootstrap standard "
            "error of the numeric best"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        **trust_region,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        args.output / "model.safetensors",
    )
    (args.output / "config.json").write_text(
        json.dumps(selected_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl_new(args.output / "selector-base.jsonl", base_predictions)
    write_jsonl_new(args.output / "selector-refined.jsonl", selected_predictions)
    receipt = {
        "schema": SCHEMA,
        "status": "development_calibration",
        "source": {
            "git_commit": git_commit(),
            "parent_refiner": str(args.refiner.resolve()),
            "parent_model_sha256": file_sha256(args.refiner / "model.safetensors"),
            "parent_config_sha256": file_sha256(args.refiner / "config.json"),
            "parent_receipt_sha256": file_sha256(args.refiner / "receipt.json"),
            "gold": {"path": str(args.gold.resolve()), "sha256": file_sha256(args.gold)},
            "predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": file_sha256(args.predictions),
            },
        },
        "recipe": selected_config,
        "selection": {
            "metric": "exact typed character-span micro F1 on the enlarged complete-native selector",
            "rule": selected_config["calibration"]["rule"],
            "noise_estimate": {
                "unit": "document",
                "samples": args.bootstrap_samples,
                "seed": args.seed,
                **trust_region,
            },
            "curve": curve,
            "no_refiner": exact_typed_score(rows, base_predictions),
            "selected": exact_typed_score(rows, selected_predictions),
            "selected_telemetry": candidate_telemetry[selected_penalty],
        },
        "claim_boundary": (
            "Adaptive external-development calibration on the enlarged internal selector. "
            "No external view selected the penalty, and no release claim follows without fresh "
            "source-group-disjoint confirmation."
        ),
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    headline = (
        f"selected movement penalty {selected_penalty:g} (numeric best {numeric_best_penalty:g}, "
        f"one-SE floor {trust_region['threshold']:.6f}): "
        f"{receipt['selection']['no_refiner']['f1']:.6f} -> "
        f"{receipt['selection']['selected']['f1']:.6f}"
    )
    print(f"character-boundary-refiner calibration: {headline}", flush=True)
    headline_file = os.environ.get("AGENTCTL_HEADLINE_FILE")
    if headline_file:
        Path(headline_file).write_text(
            f"character-boundary-refiner calibration: {headline}\n",
            encoding="utf-8",
        )


def apply(args: argparse.Namespace) -> None:
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {args.receipt}")
    rows = read_jsonl(args.gold)
    predictions = read_jsonl(args.predictions)
    device = torch.device(args.device)
    model, config = load_refiner(args.refiner, device)
    refined, telemetry = refine_prediction_rows(rows, predictions, model, config, device=device)
    write_jsonl_new(args.output, refined)
    receipt = {
        "schema": SCHEMA,
        "operation": "apply",
        "source": {
            "git_commit": git_commit(),
            "refiner": str(args.refiner.resolve()),
            "refiner_receipt_sha256": file_sha256(args.refiner / "receipt.json"),
            "gold": {"path": str(args.gold.resolve()), "sha256": file_sha256(args.gold)},
            "predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": file_sha256(args.predictions),
            },
        },
        "output": {"path": str(args.output.resolve()), "sha256": file_sha256(args.output)},
        "recipe": config,
        "telemetry": telemetry,
        "contract": (
            "preserve proposal count and v2 type; move each start/end by at most one Unicode "
            "codepoint; retain nonempty spans; introduce no overlap between source-disjoint "
            "adjacent proposals; introduce no duplicate typed span beyond source multiplicity; "
            "identity-route unsupported languages"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "telemetry": telemetry["aggregate"]}, sort_keys=True))


def apply_sweep(args: argparse.Namespace) -> None:
    """Apply the frozen serving ranker to every saved operating point."""
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {args.receipt}")
    rows = read_jsonl(args.gold)
    sweep = json.loads(args.predictions.read_text())
    if sweep.get("boundary_refinement"):
        raise ValueError("sweep already has boundary refinement")
    if sweep["rows"] != len(rows):
        raise ValueError("sweep and input row counts differ")
    model, config = load_refiner(args.refiner, torch.device(args.device))
    references = {"person_reference", "organization_reference"}
    telemetry = {}
    for threshold, predictions in sweep["points"].items():
        omitted = sum(p["label"] in references for row in predictions for p in row["preds"])
        if omitted and not args.exclude_reference_spans:
            raise ValueError("reference labels require explicit --exclude-reference-spans")
        primary = [
            {**row, "preds": [p for p in row["preds"] if p["label"] not in references]} for row in predictions
        ]
        sweep["points"][threshold], telemetry[threshold] = refine_prediction_rows(
            rows, primary, model, config, device=torch.device(args.device)
        )
        telemetry[threshold]["excluded_reference_predictions"] = omitted
        print(f"[refine] threshold={threshold} {telemetry[threshold]['aggregate']}", flush=True)
    receipt = {
        "schema": SCHEMA,
        "operation": "apply-sweep",
        "source": {
            "git_commit": git_commit(),
            "predictions": {"path": str(args.predictions), "sha256": file_sha256(args.predictions)},
            "inputs": {"path": str(args.gold), "sha256": file_sha256(args.gold)},
            "refiner": str(args.refiner),
            "model_sha256": file_sha256(args.refiner / "model.safetensors"),
            "config_sha256": file_sha256(args.refiner / "config.json"),
        },
        "recipe": config,
        "exclude_reference_spans": args.exclude_reference_spans,
        "telemetry": telemetry,
    }
    sweep["boundary_refinement"] = {key: value for key, value in receipt.items() if key != "telemetry"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sweep) + "\n")
    receipt["output"] = {"path": str(args.output), "sha256": file_sha256(args.output)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, action="append", required=True)
    fit_parser.add_argument("--validation", type=Path, required=True)
    fit_parser.add_argument("--partition-receipt", type=Path, required=True)
    fit_parser.add_argument("--model", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--language-round", type=Path, default=Path("scripts/pii_language_round.yaml"))
    fit_parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, choices=(DEFAULT_RADIUS,))
    fit_parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    fit_parser.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)
    fit_parser.add_argument("--groups-per-language", type=int, default=DEFAULT_GROUPS_PER_LANGUAGE)
    fit_parser.add_argument("--epochs", type=int, default=8)
    fit_parser.add_argument("--batch-size", type=int, default=2048)
    fit_parser.add_argument("--learning-rate", type=float, default=0.03)
    fit_parser.add_argument("--weight-decay", type=float, default=1e-4)
    fit_parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    fit_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fit_parser.add_argument("--max-train-rows", type=int, default=0)
    fit_parser.add_argument("--max-validation-rows", type=int, default=0)
    fit_parser.set_defaults(func=fit)
    calibrate_parser = subparsers.add_parser("calibrate-movement")
    calibrate_parser.add_argument("--refiner", type=Path, required=True)
    calibrate_parser.add_argument("--gold", type=Path, required=True)
    calibrate_parser.add_argument("--predictions", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--movement-penalty", type=float, action="append", required=True)
    calibrate_parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    calibrate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    calibrate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    calibrate_parser.set_defaults(func=calibrate_movement_penalty)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--refiner", type=Path, required=True)
    apply_parser.add_argument("--gold", type=Path, required=True)
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--receipt", type=Path, required=True)
    apply_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    apply_parser.set_defaults(func=apply)
    sweep_parser = subparsers.add_parser("apply-sweep")
    sweep_parser.add_argument("--refiner", type=Path, required=True)
    sweep_parser.add_argument(
        "--gold", type=Path, required=True, help="text/language rows; annotations are not read"
    )
    sweep_parser.add_argument("--predictions", type=Path, required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument("--receipt", type=Path, required=True)
    sweep_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    sweep_parser.add_argument("--exclude-reference-spans", action="store_true")
    sweep_parser.set_defaults(func=apply_sweep)
    export_parser = subparsers.add_parser("export-onnx")
    export_parser.add_argument("--refiner", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.set_defaults(func=export_onnx)
    args = parser.parse_args()
    if getattr(args, "epochs", 1) <= 0 or getattr(args, "batch_size", 1) <= 0:
        parser.error("epochs and batch size must be positive")
    if getattr(args, "context", 1) <= 0:
        parser.error("character context must be positive")
    if getattr(args, "bootstrap_samples", 2) < 2:
        parser.error("bootstrap samples must be at least two")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
