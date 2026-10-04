import torch
from transformers import AutoModel, BertConfig

from scripts.pii_crf_model import (
    CRF_ARCHITECTURE,
    FactorizedBIOESCRF,
    MMBertCrfForTokenClassification,
    compact_valid_tokens,
)

LABELS = ["O", "B-a", "I-a", "E-a", "S-a", "B-b", "I-b", "E-b", "S-b"]


def test_crf_viterbi_rejects_invalid_start_and_cross_type_continuation() -> None:
    crf = FactorizedBIOESCRF(LABELS)
    emissions = torch.zeros(1, 2, len(LABELS))
    emissions[0, 0, LABELS.index("I-a")] = 10
    emissions[0, 0, LABELS.index("B-a")] = 9
    emissions[0, 1, LABELS.index("E-b")] = 10
    emissions[0, 1, LABELS.index("E-a")] = 9

    path = crf.decode(emissions, torch.ones(1, 2, dtype=torch.bool))[0]

    assert [LABELS[label_id] for label_id in path] == ["B-a", "E-a"]


def test_crf_negative_log_likelihood_is_finite_and_differentiable() -> None:
    crf = FactorizedBIOESCRF(LABELS)
    emissions = torch.randn(2, 4, len(LABELS), requires_grad=True)
    tags = torch.tensor(
        [
            [LABELS.index("B-a"), LABELS.index("E-a"), 0, 0],
            [LABELS.index("S-b"), 0, 0, 0],
        ]
    )
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    loss = crf.neg_log_likelihood(emissions, tags, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all()


def test_compact_valid_tokens_preserves_order_without_rowwise_copies() -> None:
    emissions = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    tags = torch.arange(10).reshape(2, 5)
    mask = torch.tensor(
        [
            [False, True, True, False, False],
            [True, False, True, True, False],
        ]
    )

    compact_emissions, compact_tags, compact_mask = compact_valid_tokens(emissions, tags, mask)

    assert compact_tags.tolist() == [[1, 2, 0], [5, 7, 8]]
    assert compact_emissions[0, :2].tolist() == emissions[0, [1, 2]].tolist()
    assert compact_mask.tolist() == [[True, True, False], [True, True, True]]


def test_crf_model_round_trip_preserves_logits_and_special_token_mask(tmp_path) -> None:
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    config.architectures = [MMBertCrfForTokenClassification.__name__]
    config.pii_decoder = CRF_ARCHITECTURE
    config.pii_classifier_dropout = 0.0
    model = MMBertCrfForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, LABELS.index("B-a"), LABELS.index("E-a"), -100]]),
    }

    with torch.no_grad():
        before = model(**inputs)
    model.save_pretrained(tmp_path)
    restored = MMBertCrfForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        after = restored(**inputs)

    assert torch.isfinite(before.loss)
    assert torch.allclose(before.logits, after.logits)
