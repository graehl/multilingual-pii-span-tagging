import torch

from scripts.pii_gliner2_calibration_smoke import (
    c_weighted_struct_loss,
    deterministic_loss_mask,
    emitted_movement,
    maximum_one_to_one_matches,
    normalized_c_weighted_bce,
    select_disjoint_rows,
    symmetric_overlap,
)


def test_c_one_matches_unweighted_kept_sum():
    scores = torch.tensor([0.0, 1.0, -1.0])
    labels = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.tensor([True, True, False])
    observed = normalized_c_weighted_bce(scores, labels, mask, positive_weight=1.0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(scores[:2], labels[:2], reduction="sum")
    assert torch.allclose(observed, expected)


def test_c_changes_gradient_ratio_without_changing_total_weight_scale():
    def gradients(c):
        scores = torch.zeros(2, requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        loss = normalized_c_weighted_bce(scores, labels, torch.ones(2, dtype=torch.bool), positive_weight=c)
        loss.backward()
        return loss.detach(), scores.grad.detach()

    base_loss, base_gradient = gradients(1.0)
    tilted_loss, tilted_gradient = gradients(3.0)
    assert torch.allclose(base_loss, tilted_loss)
    assert torch.allclose(tilted_gradient[0].abs() / tilted_gradient[1].abs(), torch.tensor(3.0))
    assert torch.allclose(base_gradient[0].abs() / base_gradient[1].abs(), torch.tensor(1.0))


def test_deterministic_mask_always_keeps_positives_and_validity():
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    valid = torch.tensor([True, True, False, True, True])
    first = deterministic_loss_mask(labels, valid, negative_keep=0.5, seed=17)
    second = deterministic_loss_mask(labels, valid, negative_keep=0.5, seed=17)
    assert torch.equal(first, second)
    assert first[0] and first[3]
    assert not first[2]


def test_span_exclusion_removes_gradient_from_ignored_geometry():
    class LossOwner:
        _pii_negative_keep = 1.0
        _pii_mask_seed = 17
        _pii_positive_weight = 3.0
        _pii_active_span_exclusion = torch.tensor([[False, False], [True, False]])

        @staticmethod
        def count_embed(_embeddings, _count):
            return torch.tensor([[[1.0], [1.0]]])

    span_rep = torch.tensor([[[0.0], [0.0]], [[0.0], [0.0]]], requires_grad=True)
    schema_emb = torch.tensor([[0.0], [0.0], [0.0]])
    structure = [1, [[(0, 0), (-1, -1)]]]
    loss = c_weighted_struct_loss(
        LossOwner(),
        span_rep,
        schema_emb,
        structure,
        torch.zeros(1, 4, dtype=torch.bool),
    )
    loss.backward()
    gradients = span_rep.grad.squeeze(-1)
    assert gradients[1, 0] == 0
    assert gradients[0, 0] != 0


def test_disjoint_selection_is_stable():
    rows = [
        {
            "id": str(index),
            "lang": "en",
            "provenance": {
                "source_path": "source.jsonl",
                "source_line": index + 1,
                "source_id": str(index),
                "lang": "en",
                "projected_spans": [[0, 1, "name"]],
            },
        }
        for index in range(10)
    ]
    first = select_disjoint_rows(rows, train_docs=3, score_docs=4, seed=7)
    second = select_disjoint_rows(rows, train_docs=3, score_docs=4, seed=7)
    assert first == second
    assert {row["id"] for row in first[0]}.isdisjoint(row["id"] for row in first[1])


def test_maximum_overlap_matching_is_one_to_one():
    gold = [
        {"start": 0, "end": 5, "label": "name"},
        {"start": 5, "end": 10, "label": "name"},
    ]
    predictions = [{"start": 0, "end": 10, "label": "name"}]
    assert maximum_one_to_one_matches(gold, predictions, symmetric_overlap) == 0
    predictions = [
        {"start": 0, "end": 5, "label": "name"},
        {"start": 5, "end": 10, "label": "name"},
    ]
    assert maximum_one_to_one_matches(gold, predictions, symmetric_overlap) == 2


def test_emitted_movement_counts_added_removed_and_relabelled():
    before = [
        [
            {"start": 0, "end": 3, "label": "name", "confidence": 0.8},
            {"start": 5, "end": 8, "label": "name", "confidence": 0.8},
        ]
    ]
    after = [
        [
            {"start": 0, "end": 3, "label": "organization", "confidence": 0.8},
            {"start": 9, "end": 12, "label": "name", "confidence": 0.8},
        ]
    ]
    assert emitted_movement(before, after, 0.5) == {
        "added": 2,
        "removed": 2,
        "relabelled": 1,
        "changed_docs": 1,
    }
