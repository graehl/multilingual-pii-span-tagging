"""Acceptable GLiNER labels are alternatives, not simultaneous positives."""

import torch
import torch.nn.functional as F

from scripts.pii_gliner2_mapped_loss import acceptable_span_loss


def test_either_acceptable_label_satisfies_gold_without_penalizing_the_other():
    groups = [{"start": 0, "end": 0, "labels": [0, 1]}]
    keep = torch.ones(1, dtype=torch.bool)
    losses = []
    for logits in ([8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]):
        scores = torch.tensor(logits).reshape(1, 2, 1, 1).requires_grad_()
        loss = acceptable_span_loss(scores, groups, [True, True], keep, masking_rate=0)
        loss.backward()
        assert torch.isfinite(scores.grad).all()
        losses.append(loss.item())
    assert abs(losses[0] - losses[1]) < 1e-7
    assert losses[0] < 0.001
    assert losses[2] > 6


def test_singleton_recovers_binary_loss_and_unknown_types_have_no_negative_gradient():
    scores = torch.tensor([0.7, -0.8, 1.3, 0.2]).reshape(1, 2, 2, 1).requires_grad_()
    groups = [{"start": 0, "end": 0, "labels": [0]}]
    valid = torch.ones(2, dtype=torch.bool)
    target = torch.zeros_like(scores)
    target[0, 0, 0, 0] = 1
    expected = F.binary_cross_entropy_with_logits(scores, target, reduction="sum")
    actual = acceptable_span_loss(scores, groups, [True, True], valid)
    torch.testing.assert_close(actual, expected)
    masked = acceptable_span_loss(scores, groups, [True, False], valid)
    masked.backward()
    assert scores.grad[0, 1].abs().sum() == 0
    assert scores.grad[0, 0, 0, 0] < 0


def test_processor_keeps_alternatives_through_real_training_dataset():
    from gliner2.processor import SamplingConfig
    from gliner2.training.trainer import ExtractorDataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from scripts.pii_gliner2_mapped_training import MappedSchemaTransformer, training_records

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "Paris": 2, ".": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    processor = MappedSchemaTransformer(
        tokenizer=tokenizer, sampling_config=SamplingConfig(synthetic_entity_label_prob=0)
    )
    records = training_records(
        [
            {
                "text": "Paris.",
                "entities": {"location": ["Paris"]},
                "acceptable_targets": {
                    "groups": [{"start": 0, "end": 0, "labels": ["location", "locality"]}],
                    "unknown_types": ["locality"],
                    "complete": True,
                },
            }
        ],
        {"location": "location", "locality": "city"},
    )
    dataset = ExtractorDataset(records, shuffle=False)
    batch = processor.collate_fn_train([dataset[0]])
    metadata = batch.structure_labels[0][0][2]
    assert metadata["groups"] == [{"start": 0, "end": 0, "labels": [0, 1]}]
    assert metadata["known_negative"] == [True, False]
