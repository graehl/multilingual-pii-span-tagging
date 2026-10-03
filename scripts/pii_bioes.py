"""BIOES sequence-legality diagnostics shared by PII token classifiers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def split_bioes_label(label: str) -> tuple[str, str | None]:
    """Return the BIOES boundary prefix and optional entity type."""
    if label == "O":
        return "O", None
    try:
        prefix, entity_type = label.split("-", 1)
    except ValueError as error:
        raise ValueError(f"invalid BIOES label without an entity type: {label!r}") from error
    if prefix not in {"B", "I", "E", "S"} or not entity_type:
        raise ValueError(f"invalid BIOES label: {label!r}")
    return prefix, entity_type


def count_bioes_violations(labels: Sequence[str]) -> tuple[int, int]:
    """Count invalid start, adjacent-transition, and end constraints.

    The denominator is ``len(labels) + 1`` for a non-empty sequence: one
    start constraint, one constraint between each adjacent pair, and one end
    constraint. Type agreement is required inside a multi-token entity.
    """
    if not labels:
        return 0, 0
    parts = [split_bioes_label(label) for label in labels]
    invalid = int(parts[0][0] not in {"O", "B", "S"})
    for (previous_prefix, previous_type), (current_prefix, current_type) in zip(parts, parts[1:]):
        if previous_prefix in {"O", "E", "S"}:
            valid = current_prefix in {"O", "B", "S"}
        else:
            valid = current_prefix in {"I", "E"} and current_type == previous_type
        invalid += int(not valid)
    invalid += int(parts[-1][0] not in {"O", "E", "S"})
    return invalid, len(labels) + 1


def _typed_label_ids(id2label: dict[int, str]) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Resolve O and complete B/I/E/S quartets without assuming label order."""
    outside = [label_id for label_id, label in id2label.items() if label == "O"]
    if len(outside) != 1:
        raise ValueError(f"expected exactly one O label, found {len(outside)}")

    by_type: dict[str, dict[str, int]] = {}
    for label_id, label in id2label.items():
        prefix, entity_type = split_bioes_label(label)
        if prefix == "O":
            continue
        assert entity_type is not None
        if prefix in by_type.setdefault(entity_type, {}):
            raise ValueError(f"duplicate {prefix} label for entity type {entity_type!r}")
        by_type[entity_type][prefix] = label_id

    quartets = []
    for entity_type in sorted(by_type):
        labels = by_type[entity_type]
        missing = {"B", "I", "E", "S"} - labels.keys()
        if missing:
            raise ValueError(f"incomplete BIOES labels for {entity_type!r}: missing {sorted(missing)}")
        quartets.append((labels["B"], labels["I"], labels["E"], labels["S"]))
    return outside[0], quartets


def decode_bioes_spans(label_ids, id2label):
    """Decode one compact token-label sequence into exact typed spans."""
    spans = []
    current = None
    for token_index, label_id in enumerate(label_ids):
        label = id2label[int(label_id)]
        if label == "O":
            if current:
                spans.append(tuple(current))
                current = None
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix in ("B", "S") or (current and current[2] != entity_type):
            if current:
                spans.append(tuple(current))
            current = [token_index, token_index + 1, entity_type]
        elif current:
            current[1] = token_index + 1
        else:
            current = [token_index, token_index + 1, entity_type]
        if prefix in ("E", "S"):
            spans.append(tuple(current))
            current = None
    if current:
        spans.append(tuple(current))
    return set(spans)


def constrained_bioes_decode(
    scores: np.ndarray,
    id2label: dict[int, str],
    *,
    same_type_split_cost: float = 0.0,
    same_type_split_boundaries: Sequence[bool] | None = None,
) -> np.ndarray:
    """Return the highest-scoring legal typed-BIOES path.

    ``scores`` has shape ``[tokens, labels]`` and contains independent token
    emission scores. By default the decoder adds no transition score: it only
    excludes illegal BIOES starts, ends, boundary transitions, and
    cross-entity-type continuations. A positive ``same_type_split_cost`` may
    penalize ``S/E-c -> S/B-c`` at selected token boundaries without making
    the corresponding merge mandatory. The recurrence remains linear in the
    number of labels rather than quadratic in the full typed-label vocabulary.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [tokens, labels], got {scores.shape}")
    token_count, label_count = scores.shape
    if token_count == 0:
        return np.empty(0, dtype=np.int64)
    if set(id2label) != set(range(label_count)):
        raise ValueError("id2label keys must exactly cover the score columns")
    if not np.isfinite(same_type_split_cost) or same_type_split_cost < 0:
        raise ValueError("same_type_split_cost must be finite and nonnegative")
    split_boundaries = (
        np.zeros(max(0, token_count - 1), dtype=bool)
        if same_type_split_boundaries is None
        else np.asarray(same_type_split_boundaries, dtype=bool)
    )
    expected_boundary_shape = (max(0, token_count - 1),)
    if split_boundaries.shape != expected_boundary_shape:
        raise ValueError(
            "same_type_split_boundaries must have shape "
            f"{expected_boundary_shape}, got {split_boundaries.shape}"
        )
    if same_type_split_cost and same_type_split_boundaries is None:
        raise ValueError("positive same_type_split_cost requires same_type_split_boundaries")

    outside, quartets = _typed_label_ids(id2label)
    begin = np.asarray([labels[0] for labels in quartets], dtype=np.int64)
    inside = np.asarray([labels[1] for labels in quartets], dtype=np.int64)
    end = np.asarray([labels[2] for labels in quartets], dtype=np.int64)
    single = np.asarray([labels[3] for labels in quartets], dtype=np.int64)
    closed = np.concatenate(([outside], end, single))

    negative_infinity = np.array(-np.inf, dtype=scores.dtype)
    previous = np.full(label_count, negative_infinity, dtype=scores.dtype)
    initial = np.concatenate(([outside], begin, single))
    previous[initial] = scores[0, initial]
    backpointers = np.full((token_count, label_count), -1, dtype=np.int32)

    for token_index in range(1, token_count):
        current = np.full(label_count, negative_infinity, dtype=scores.dtype)
        best_closed = int(closed[np.argmax(previous[closed])])
        closed_targets = np.concatenate(([outside], begin, single))
        if same_type_split_cost == 0 or not split_boundaries[token_index - 1]:
            current[closed_targets] = previous[best_closed] + scores[token_index, closed_targets]
            backpointers[token_index, closed_targets] = best_closed
        else:
            current[outside] = previous[best_closed] + scores[token_index, outside]
            backpointers[token_index, outside] = best_closed

            closed_type_scores = np.maximum(previous[end], previous[single])
            closed_type_labels = np.where(previous[end] >= previous[single], end, single)
            best_type = int(np.argmax(closed_type_scores))
            second_type_scores = closed_type_scores.copy()
            second_type_scores[best_type] = negative_infinity
            second_type = int(np.argmax(second_type_scores)) if len(quartets) > 1 else -1
            for type_index, (begin_label, single_label) in enumerate(zip(begin, single, strict=True)):
                other_label = outside
                other_score = previous[outside]
                other_type = best_type if best_type != type_index else second_type
                if other_type >= 0 and closed_type_scores[other_type] > other_score:
                    other_label = int(closed_type_labels[other_type])
                    other_score = closed_type_scores[other_type]

                same_label = int(closed_type_labels[type_index])
                same_score = closed_type_scores[type_index] - same_type_split_cost
                predecessor = same_label if same_score > other_score else other_label
                predecessor_score = same_score if same_score > other_score else other_score
                current[[begin_label, single_label]] = (
                    predecessor_score + scores[token_index, [begin_label, single_label]]
                )
                backpointers[token_index, [begin_label, single_label]] = predecessor

        choose_begin = previous[begin] >= previous[inside]
        best_open = np.where(choose_begin, begin, inside)
        open_score = previous[best_open]
        current[inside] = open_score + scores[token_index, inside]
        current[end] = open_score + scores[token_index, end]
        backpointers[token_index, inside] = best_open
        backpointers[token_index, end] = best_open
        previous = current

    final_label = int(closed[np.argmax(previous[closed])])
    path = np.empty(token_count, dtype=np.int64)
    path[-1] = final_label
    for token_index in range(token_count - 1, 0, -1):
        path[token_index - 1] = backpointers[token_index, path[token_index]]
    return path


def constrained_boundary_bioes_decode(
    scores: np.ndarray,
    id2label: dict[int, str],
    entity_start_allowed: Sequence[bool],
    entity_end_allowed: Sequence[bool],
    entity_single_allowed: Sequence[bool] | None = None,
    *,
    same_type_split_cost: float = 0.0,
    same_type_split_boundaries: Sequence[bool] | None = None,
) -> np.ndarray:
    """Return the best legal path whose entity endpoints obey two masks.

    ``entity_start_allowed[t]`` controls B/S labels at token ``t`` and
    ``entity_end_allowed[t]`` controls E labels. S labels require both masks,
    or the separately supplied ``entity_single_allowed`` mask when a token can
    cover the same candidate start and end. O and interior labels keep their
    original emissions. This preserves the ordinary typed-BIOES score while
    restricting which character boundaries a token may cover when opening or
    closing a span.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [tokens, labels], got {scores.shape}")
    token_count, label_count = scores.shape
    if set(id2label) != set(range(label_count)):
        raise ValueError("id2label keys must exactly cover the score columns")
    start_allowed = np.asarray(entity_start_allowed, dtype=bool)
    end_allowed = np.asarray(entity_end_allowed, dtype=bool)
    expected_shape = (token_count,)
    if start_allowed.shape != expected_shape:
        raise ValueError(f"entity_start_allowed must have shape {expected_shape}, got {start_allowed.shape}")
    if end_allowed.shape != expected_shape:
        raise ValueError(f"entity_end_allowed must have shape {expected_shape}, got {end_allowed.shape}")
    single_allowed = (
        start_allowed & end_allowed
        if entity_single_allowed is None
        else np.asarray(entity_single_allowed, dtype=bool)
    )
    if single_allowed.shape != expected_shape:
        raise ValueError(
            f"entity_single_allowed must have shape {expected_shape}, got {single_allowed.shape}"
        )
    if token_count == 0:
        return np.empty(0, dtype=np.int64)

    _outside, quartets = _typed_label_ids(id2label)
    begin = np.asarray([labels[0] for labels in quartets], dtype=np.int64)
    end = np.asarray([labels[2] for labels in quartets], dtype=np.int64)
    single = np.asarray([labels[3] for labels in quartets], dtype=np.int64)
    active_scores = scores.copy()
    active_scores[np.ix_(~start_allowed, begin)] = -np.inf
    active_scores[np.ix_(~end_allowed, end)] = -np.inf
    active_scores[np.ix_(~single_allowed, single)] = -np.inf
    return constrained_bioes_decode(
        active_scores,
        id2label,
        same_type_split_cost=same_type_split_cost,
        same_type_split_boundaries=same_type_split_boundaries,
    )


def constrained_bucket_bioes_decode(
    scores: np.ndarray,
    id2label: dict[int, str],
    bucket_of: dict[str, str],
    top_k_non_o: int | None = None,
    beam_width: int | None = None,
    bucket_reduction: str = "max",
    bucket_temperature: float = 1.0,
) -> np.ndarray:
    """Return the exact best legal path at a type-equivalence cut.

    Each state is O or one BIOES prefix paired with a compatibility bucket.
    A state's token score aggregates the model-native fine-label logits in
    that bucket and prefix using ``bucket_reduction``. The selected fine type
    may change within a span while the bucket and BIOES geometry stay
    consistent. The returned IDs identify the highest-logit original member
    of every selected state.

    ``top_k_non_o`` masks the full score vector down to each token's highest
    k non-O labels before the same exact recurrence. It exists to measure
    sparse-cache sufficiency; the unmasked path is the reference decoder.

    ``beam_width`` is an experimental state beam. The recurrence first merges
    dominated paths ending in the same BIOES state, then retains only the best
    width states globally. A width at least as large as the lattice is exact.

    ``bucket_reduction`` controls how native fine-label evidence becomes a
    compatibility-state emission. ``max`` preserves the best native logit.
    ``logsumexp`` and ``logmeanexp`` softly combine all members at
    ``bucket_temperature``; the latter removes the explicit bucket-size bonus.
    The returned fine label remains the highest-logit member of the selected
    state, independently of the state's aggregate evidence.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [tokens, labels], got {scores.shape}")
    token_count, label_count = scores.shape
    if token_count == 0:
        return np.empty(0, dtype=np.int64)
    if set(id2label) != set(range(label_count)):
        raise ValueError("id2label keys must exactly cover the score columns")
    if top_k_non_o is not None and top_k_non_o <= 0:
        raise ValueError("top_k_non_o must be positive when supplied")
    if beam_width is not None and beam_width <= 0:
        raise ValueError("beam_width must be positive when supplied")
    if bucket_reduction not in {"max", "logsumexp", "logmeanexp"}:
        raise ValueError(f"unknown bucket_reduction {bucket_reduction!r}")
    if not np.isfinite(bucket_temperature) or bucket_temperature <= 0:
        raise ValueError("bucket_temperature must be positive and finite")

    outside = [label_id for label_id, label in id2label.items() if label == "O"]
    if len(outside) != 1:
        raise ValueError(f"expected exactly one O label, found {len(outside)}")
    outside = outside[0]

    members: dict[str, dict[str, list[int]]] = {}
    for label_id, label in id2label.items():
        prefix, entity_type = split_bioes_label(label)
        if prefix == "O":
            continue
        assert entity_type is not None
        if entity_type not in bucket_of:
            raise ValueError(f"no compatibility bucket for entity type {entity_type!r}")
        members.setdefault(bucket_of[entity_type], {}).setdefault(prefix, []).append(label_id)
    buckets = sorted(members)
    for bucket in buckets:
        missing = {"B", "I", "E", "S"} - members[bucket].keys()
        if missing:
            raise ValueError(f"compatibility bucket {bucket!r} lacks prefixes {sorted(missing)}")

    active_scores = scores
    if top_k_non_o is not None and top_k_non_o < label_count - 1:
        active_scores = np.full_like(scores, -np.inf)
        active_scores[:, outside] = scores[:, outside]
        non_o = np.asarray([label_id for label_id in range(label_count) if label_id != outside])
        keep_count = min(top_k_non_o, len(non_o))
        keep_positions = np.argpartition(scores[:, non_o], -keep_count, axis=1)[:, -keep_count:]
        keep_ids = non_o[keep_positions]
        rows = np.arange(token_count)[:, None]
        active_scores[rows, keep_ids] = scores[rows, keep_ids]

    prefixes = ("B", "I", "E", "S")
    state_count = 1 + 4 * len(buckets)
    emissions = np.full((token_count, state_count), -np.inf, dtype=scores.dtype)
    emission_labels = np.full((token_count, state_count), -1, dtype=np.int64)
    emissions[:, 0] = active_scores[:, outside]
    emission_labels[:, 0] = outside
    for bucket_index, bucket in enumerate(buckets):
        for prefix_index, prefix in enumerate(prefixes):
            state = 1 + 4 * bucket_index + prefix_index
            label_ids = np.asarray(members[bucket][prefix], dtype=np.int64)
            choices = np.argmax(active_scores[:, label_ids], axis=1)
            member_scores = active_scores[:, label_ids]
            if bucket_reduction == "max":
                emissions[:, state] = member_scores[np.arange(token_count), choices]
            else:
                emissions[:, state] = _temperature_logsumexp(
                    member_scores,
                    bucket_temperature,
                    normalize=bucket_reduction == "logmeanexp",
                )
            emission_labels[:, state] = label_ids[choices]

    begin = np.asarray([1 + 4 * index for index in range(len(buckets))], dtype=np.int64)
    inside = begin + 1
    end = begin + 2
    single = begin + 3
    closed = np.concatenate(([0], end, single))
    closed_targets = np.concatenate(([0], begin, single))

    previous = np.full(state_count, -np.inf, dtype=scores.dtype)
    initial = closed_targets
    previous[initial] = emissions[0, initial]
    if beam_width is not None:
        if token_count == 1:
            previous[np.setdiff1d(np.arange(state_count), closed)] = -np.inf
        _prune_state_beam(previous, beam_width)
    backpointers = np.full((token_count, state_count), -1, dtype=np.int32)
    for token_index in range(1, token_count):
        current = np.full(state_count, -np.inf, dtype=scores.dtype)
        best_closed = int(closed[np.argmax(previous[closed])])
        current[closed_targets] = previous[best_closed] + emissions[token_index, closed_targets]
        backpointers[token_index, closed_targets] = best_closed

        choose_begin = previous[begin] >= previous[inside]
        best_open = np.where(choose_begin, begin, inside)
        open_score = previous[best_open]
        current[inside] = open_score + emissions[token_index, inside]
        current[end] = open_score + emissions[token_index, end]
        backpointers[token_index, inside] = best_open
        backpointers[token_index, end] = best_open
        if beam_width is not None:
            if token_index == token_count - 1:
                current[np.setdiff1d(np.arange(state_count), closed)] = -np.inf
            _prune_state_beam(current, beam_width)
        previous = current

    final_state = int(closed[np.argmax(previous[closed])])
    state_path = np.empty(token_count, dtype=np.int64)
    state_path[-1] = final_state
    for token_index in range(token_count - 1, 0, -1):
        state_path[token_index - 1] = backpointers[token_index, state_path[token_index]]
    return emission_labels[np.arange(token_count), state_path]


def _temperature_logsumexp(values: np.ndarray, temperature: float, normalize: bool) -> np.ndarray:
    """Stable row-wise temperature log-sum-exp, preserving all-masked rows."""
    maxima = np.max(values, axis=1)
    result = np.full(len(values), -np.inf, dtype=values.dtype)
    finite = np.isfinite(maxima)
    if np.any(finite):
        shifted = (values[finite] - maxima[finite, None]) / temperature
        result[finite] = maxima[finite] + temperature * np.log(np.exp(shifted).sum(axis=1))
        if normalize:
            result[finite] -= temperature * np.log(values.shape[1])
    return result


def interpolate_bioes_sibling_scores(
    scores: np.ndarray,
    id2label: dict[int, str],
    bucket_of: dict[str, str],
    *,
    alpha: float,
    temperature: float = 1.0,
) -> np.ndarray:
    """Interpolate finite fine logits with normalized coarse sibling evidence.

    Siblings share BIOES prefix and compatibility class. O is unchanged.
    Alpha is in [0, 1]; at 1 all sibling fine scores tie.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or set(id2label) != set(range(scores.shape[1])):
        raise ValueError("scores must be [tokens, labels] with id2label covering every column")
    if not np.isfinite(scores).all():
        raise ValueError("sibling interpolation requires finite full logits")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("backoff alpha must be finite and in [0, 1]")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("backoff temperature must be positive and finite")
    _typed_label_ids(id2label)
    groups: dict[tuple[str, str], list[int]] = {}
    for label_id, label in id2label.items():
        prefix, fine_type = split_bioes_label(label)
        if fine_type is None:
            continue
        if fine_type not in bucket_of:
            raise ValueError(f"no compatibility bucket for entity type {fine_type!r}")
        groups.setdefault((prefix, bucket_of[fine_type]), []).append(label_id)
    result = scores.copy()
    for ids in groups.values():
        aggregate = _temperature_logsumexp(scores[:, ids], temperature, normalize=True)
        result[:, ids] = (1 - alpha) * scores[:, ids] + alpha * aggregate[:, None]
    return result


def constrained_backoff_bioes_decode(
    scores: np.ndarray,
    id2label: dict[int, str],
    bucket_of: dict[str, str],
    *,
    alpha: float,
    temperature: float = 1.0,
) -> np.ndarray:
    """Decode exact fine-compatible BIOES after coarse sibling interpolation.

    For a fixed span and alpha < 1, sibling fine labels retain their summed
    original-logit ranking. Alpha = 1 removes this fine-type preference.
    """
    interpolated = interpolate_bioes_sibling_scores(
        scores, id2label, bucket_of, alpha=alpha, temperature=temperature
    )
    return constrained_bioes_decode(interpolated, id2label)


def _prune_state_beam(state_scores: np.ndarray, beam_width: int) -> None:
    """In-place global pruning after best-path-per-state merging."""
    finite = np.flatnonzero(np.isfinite(state_scores))
    if len(finite) <= beam_width:
        return
    keep_positions = np.argpartition(state_scores[finite], -beam_width)[-beam_width:]
    keep = finite[keep_positions]
    dropped = np.ones(len(state_scores), dtype=bool)
    dropped[keep] = False
    state_scores[dropped] = -np.inf
