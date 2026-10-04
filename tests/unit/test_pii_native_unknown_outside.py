"""Native incomplete-type supervision retains positives and other negatives."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from scripts.pii_encoder_train import (
    OTokenLossWeightMixin,
    PartialLabelDataCollator,
    SpanDataset,
    token_classification_loss,
)


def tokenizer():
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "Alice": 2, "met": 3, "her": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


class LogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 3, 3))
        self.config = SimpleNamespace(id2label={0: "O", 1: "S-person_name", 2: "S-person_reference"})


class StockTrainer:
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        assert "native_outside_allowed" not in inputs
        outputs = SimpleNamespace(logits=model.logits)
        # An unrelated auxiliary must survive replacement of primary CE.
        loss = token_classification_loss(outputs.logits, inputs["labels"]) + 2.0
        return (loss, outputs) if return_outputs else loss


class NativeTrainer(OTokenLossWeightMixin, StockTrainer):
    native_objective = True


def test_native_unknown_outside_through_dataset_collator_and_loss():
    tok = tokenizer()
    plain = {"text": "Alice met her", "spans": [[0, 5, "person_name"]], "label_space": "v2"}
    data = SpanDataset(
        [plain, {**plain, "unknown_primary_types": ["person_reference"]}],
        tok,
        {"O": 0, "S-person_name": 1, "S-person_reference": 2},
        max_len=32,
        native_label_space="v2",
    )
    batch = PartialLabelDataCollator(tok)([data[0], data[1]])
    assert batch["native_outside_allowed"].tolist() == [[1, 0, 0], [1, 0, 1]]
    assert batch["labels"].tolist() == [[1, 0, 0], [1, 0, 0]]
    model = LogitModel()
    trainer = NativeTrainer()
    loss = trainer.compute_loss(model, dict(batch))
    expected = (4 * torch.log(torch.tensor(3.0)) + 2 * torch.log(torch.tensor(1.5))) / 6 + 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    # A known positive is unchanged. An unknown reference is no longer pushed
    # down as a negative; unrelated person-name false positives still are.
    torch.testing.assert_close(model.logits.grad[0, 0], model.logits.grad[1, 0])
    assert model.logits.grad[0, 2, 2] > 0
    assert model.logits.grad[1, 2, 2] < 0
    assert model.logits.grad[1, 2, 1] > 0

    baseline = dict(batch)
    baseline["native_outside_allowed"] = torch.tensor([[1, 0, 0], [1, 0, 0]])
    baseline_loss = trainer.compute_loss(model, baseline)
    torch.testing.assert_close(baseline_loss, torch.log(torch.tensor(3.0)) + 2)
    trainer.o_token_loss_weight = 0.5
    weighted = trainer.compute_loss(model, dict(batch))
    expected_weighted = (3 * torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(1.5))) / 4 + 2
    torch.testing.assert_close(weighted, expected_weighted)


def test_native_unknown_type_rejects_misspelling_and_partial_rows():
    row = {"text": "Alice met her", "spans": [[0, 5, "person_name"]], "label_space": "v2"}
    for changed, message in (
        ({"unknown_primary_types": ["typo"]}, "absent from the native head"),
        ({"unknown_primary_types": ["person_name"], "supervision": "annotated_spans_only"}, "complete rows"),
    ):
        data = SpanDataset(
            [{**row, **changed}],
            tokenizer(),
            {"O": 0, "S-person_name": 1},
            max_len=32,
            native_label_space="v2",
        )
        with pytest.raises(ValueError, match=message):
            data[0]
