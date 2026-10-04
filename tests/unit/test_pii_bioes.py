import itertools

import numpy as np
import torch

from scripts.pii_bioes import (
    constrained_bioes_decode,
    constrained_boundary_bioes_decode,
    constrained_bucket_bioes_decode,
    count_bioes_violations,
    split_bioes_label,
)
from scripts.pii_crf_model import FactorizedBIOESCRF

ID2LABEL = {
    0: "O",
    1: "B-name",
    2: "I-name",
    3: "E-name",
    4: "S-name",
}


def test_constrained_decode_repairs_single_token_orphan_inside():
    scores = np.asarray([[0.0, 1.0, 5.0, 2.0, 4.0]])
    decoded = constrained_bioes_decode(scores, ID2LABEL)
    assert decoded.tolist() == [4]


def test_constrained_decode_matches_brute_force_optimum():
    rng = np.random.default_rng(7)
    for token_count in range(1, 5):
        scores = rng.normal(size=(token_count, len(ID2LABEL)))
        decoded = constrained_bioes_decode(scores, ID2LABEL)
        decoded_score = sum(scores[token, label] for token, label in enumerate(decoded))

        legal = []
        for candidate in itertools.product(ID2LABEL, repeat=token_count):
            labels = [ID2LABEL[label_id] for label_id in candidate]
            if count_bioes_violations(labels)[0] == 0:
                legal.append(sum(scores[token, label] for token, label in enumerate(candidate)))
        assert np.isclose(decoded_score, max(legal))
        assert count_bioes_violations([ID2LABEL[label_id] for label_id in decoded])[0] == 0


def test_constrained_decode_does_not_require_grouped_label_ids():
    id2label = {
        0: "I-name",
        1: "S-name",
        2: "O",
        3: "E-name",
        4: "B-name",
    }
    scores = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
        ]
    )
    decoded = constrained_bioes_decode(scores, id2label)
    assert decoded.tolist() == [4, 3]


def test_constrained_decode_matches_zero_transition_crf():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    scores = np.random.default_rng(11).normal(size=(7, len(labels))).astype(np.float32)
    expected = FactorizedBIOESCRF(labels).decode(
        torch.from_numpy(scores).unsqueeze(0),
        torch.ones(1, len(scores), dtype=torch.bool),
    )[0]
    assert constrained_bioes_decode(scores, id2label).tolist() == expected


def test_constrained_decode_zero_split_cost_is_exact_identity():
    scores = np.random.default_rng(13).normal(size=(8, len(ID2LABEL))).astype(np.float32)

    baseline = constrained_bioes_decode(scores, ID2LABEL)
    explicit_zero = constrained_bioes_decode(
        scores,
        ID2LABEL,
        same_type_split_cost=0.0,
        same_type_split_boundaries=[True] * (len(scores) - 1),
    )

    assert explicit_zero.tolist() == baseline.tolist()


def test_constrained_decode_softly_penalizes_adjacent_same_type_spans():
    scores = np.full((2, len(ID2LABEL)), -20.0)
    scores[:, 0] = 0.0
    scores[0, 1] = 4.7
    scores[0, 4] = 5.0
    scores[1, 3] = 4.7
    scores[1, 4] = 5.0

    baseline = constrained_bioes_decode(scores, ID2LABEL)
    joined = constrained_bioes_decode(
        scores,
        ID2LABEL,
        same_type_split_cost=0.7,
        same_type_split_boundaries=[True],
    )
    blocked = constrained_bioes_decode(
        scores,
        ID2LABEL,
        same_type_split_cost=0.7,
        same_type_split_boundaries=[False],
    )

    assert baseline.tolist() == [4, 4]
    assert joined.tolist() == [1, 3]
    assert blocked.tolist() == baseline.tolist()


def test_constrained_decode_split_cost_matches_brute_force_optimum():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    split_cost = 0.9
    rng = np.random.default_rng(17)
    for token_count in range(2, 5):
        scores = rng.normal(size=(token_count, len(labels)))
        boundaries = [index % 2 == 0 for index in range(token_count - 1)]
        decoded = constrained_bioes_decode(
            scores,
            id2label,
            same_type_split_cost=split_cost,
            same_type_split_boundaries=boundaries,
        )

        legal_scores = []
        for candidate in itertools.product(id2label, repeat=token_count):
            candidate_labels = [id2label[label_id] for label_id in candidate]
            if count_bioes_violations(candidate_labels)[0]:
                continue
            score = sum(scores[token, label] for token, label in enumerate(candidate))
            for boundary, left, right in zip(
                boundaries,
                candidate_labels[:-1],
                candidate_labels[1:],
                strict=True,
            ):
                left_prefix, left_type = split_bioes_label(left)
                right_prefix, right_type = split_bioes_label(right)
                if (
                    boundary
                    and left_prefix in {"E", "S"}
                    and right_prefix in {"B", "S"}
                    and left_type == right_type
                ):
                    score -= split_cost
            legal_scores.append(score)

        decoded_score = sum(scores[token, label] for token, label in enumerate(decoded))
        decoded_labels = [id2label[label_id] for label_id in decoded]
        for boundary, left, right in zip(
            boundaries,
            decoded_labels[:-1],
            decoded_labels[1:],
            strict=True,
        ):
            left_prefix, left_type = split_bioes_label(left)
            right_prefix, right_type = split_bioes_label(right)
            if (
                boundary
                and left_prefix in {"E", "S"}
                and right_prefix in {"B", "S"}
                and left_type == right_type
            ):
                decoded_score -= split_cost
        assert np.isclose(decoded_score, max(legal_scores))


def test_constrained_decode_rejects_invalid_split_cost_inputs():
    scores = np.zeros((2, len(ID2LABEL)))

    with np.testing.assert_raises_regex(ValueError, "finite and nonnegative"):
        constrained_bioes_decode(scores, ID2LABEL, same_type_split_cost=-1.0)
    with np.testing.assert_raises_regex(ValueError, "requires same_type_split_boundaries"):
        constrained_bioes_decode(scores, ID2LABEL, same_type_split_cost=1.0)
    with np.testing.assert_raises_regex(ValueError, "must have shape"):
        constrained_bioes_decode(
            scores,
            ID2LABEL,
            same_type_split_cost=1.0,
            same_type_split_boundaries=[True, True],
        )


def test_boundary_constrained_decode_replaces_an_inside_word_singleton():
    scores = np.full((2, len(ID2LABEL)), -10.0)
    scores[:, 0] = 0.0
    scores[0, 4] = 10.0  # S-name ends inside the word and is forbidden.
    scores[0, 1] = 4.0
    scores[1, 3] = 4.0

    decoded = constrained_boundary_bioes_decode(
        scores,
        ID2LABEL,
        entity_start_allowed=[True, False],
        entity_end_allowed=[False, True],
    )

    assert decoded.tolist() == [1, 3]


def test_boundary_constrained_decode_validates_token_aligned_masks():
    scores = np.zeros((2, len(ID2LABEL)))

    with np.testing.assert_raises_regex(ValueError, "entity_start_allowed"):
        constrained_boundary_bioes_decode(scores, ID2LABEL, [True], [True, True])
    with np.testing.assert_raises_regex(ValueError, "entity_end_allowed"):
        constrained_boundary_bioes_decode(scores, ID2LABEL, [True, True], [True])


def test_bucket_decode_matches_typed_decode_at_fine_compatibility():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    scores = np.random.default_rng(19).normal(size=(9, len(labels))).astype(np.float32)

    expected = constrained_bioes_decode(scores, id2label)
    actual = constrained_bucket_bioes_decode(
        scores,
        id2label,
        {"name": "name", "city": "city"},
    )

    assert actual.tolist() == expected.tolist()


def test_bucket_decode_matches_brute_force_optimum():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    bucket_of = {"name": "entity", "city": "entity"}
    scores = np.random.default_rng(23).normal(size=(4, len(labels)))

    decoded = constrained_bucket_bioes_decode(scores, id2label, bucket_of)
    decoded_score = sum(scores[token, label] for token, label in enumerate(decoded))
    legal_scores = []
    for candidate in itertools.product(id2label, repeat=len(scores)):
        bucket_labels = []
        for label_id in candidate:
            label = id2label[label_id]
            if label == "O":
                bucket_labels.append(label)
                continue
            prefix, fine_type = label.split("-", 1)
            bucket_labels.append(f"{prefix}-{bucket_of[fine_type]}")
        if count_bioes_violations(bucket_labels)[0] == 0:
            legal_scores.append(sum(scores[token, label] for token, label in enumerate(candidate)))

    assert np.isclose(decoded_score, max(legal_scores))


def test_entity_bucket_decode_changes_prefix_and_type_and_exposes_top_k_gap():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    scores = np.full((3, len(labels)), -10.0, dtype=np.float32)
    scores[:, 0] = 0.0
    scores[0, 1] = 5.0  # B-name
    scores[1, 5] = 5.0  # greedy B-city, illegal after B at the coarse level
    scores[1, 6] = 4.0  # second-best I-city repairs the prefix and changes type
    scores[2, 3] = 5.0  # E-name changes type again

    entity = {"name": "entity", "city": "entity"}
    exact = constrained_bucket_bioes_decode(scores, id2label, entity)
    top1 = constrained_bucket_bioes_decode(scores, id2label, entity, top_k_non_o=1)
    top2 = constrained_bucket_bioes_decode(scores, id2label, entity, top_k_non_o=2)

    assert exact.tolist() == [1, 6, 3]
    assert top1.tolist() != exact.tolist()
    assert top2.tolist() == exact.tolist()


def test_bucket_state_beam_matches_exact_when_it_keeps_the_full_lattice():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    bucket_of = {"name": "name", "city": "place"}
    scores = np.random.default_rng(29).normal(size=(12, len(labels))).astype(np.float32)

    exact = constrained_bucket_bioes_decode(scores, id2label, bucket_of)
    wide_beam = constrained_bucket_bioes_decode(
        scores,
        id2label,
        bucket_of,
        beam_width=1 + 4 * len(set(bucket_of.values())),
    )

    assert wide_beam.tolist() == exact.tolist()


def test_bucket_state_beam_always_returns_a_closed_legal_path():
    scores = np.full((3, len(ID2LABEL)), -10.0, dtype=np.float32)
    scores[:, 1] = 10.0  # Repeated B is attractive but cannot terminate a legal path.
    scores[:, 0] = 0.0

    decoded = constrained_bucket_bioes_decode(
        scores,
        ID2LABEL,
        {"name": "entity"},
        beam_width=1,
    )

    assert count_bioes_violations([ID2LABEL[label_id] for label_id in decoded])[0] == 0


def test_bucket_logsumexp_uses_distributed_fine_label_evidence():
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    id2label = dict(enumerate(labels))
    scores = np.full((1, len(labels)), -10.0, dtype=np.float64)
    scores[0, 0] = 1.3
    scores[0, 4] = 1.0
    scores[0, 8] = 1.0
    entity = {"name": "entity", "city": "entity"}

    maximum = constrained_bucket_bioes_decode(scores, id2label, entity)
    soft_mass = constrained_bucket_bioes_decode(
        scores,
        id2label,
        entity,
        bucket_reduction="logsumexp",
        bucket_temperature=1.0,
    )
    soft_mean = constrained_bucket_bioes_decode(
        scores,
        id2label,
        entity,
        bucket_reduction="logmeanexp",
        bucket_temperature=1.0,
    )

    assert maximum.tolist() == [0]
    assert soft_mass.tolist() == [4]
    assert soft_mean.tolist() == [0]


def test_bucket_temperature_must_be_positive_and_finite():
    scores = np.zeros((1, len(ID2LABEL)), dtype=np.float32)
    for temperature in (0.0, -1.0, np.inf, np.nan):
        with np.testing.assert_raises(ValueError):
            constrained_bucket_bioes_decode(
                scores,
                ID2LABEL,
                {"name": "entity"},
                bucket_reduction="logsumexp",
                bucket_temperature=temperature,
            )
