"""Mapped corpus coverage preserves negatives only for covered entity types."""

from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from scripts.pii_encoder_train import DualHeadLossMixin, PartialLabelDataCollator, SpanDataset


def test_corpus_outside_supervision_reaches_mapped_loss_and_update(tmp_path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "Alice": 2, "waits": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    old = {"O": 0, "S-source_person": 1}
    new = {"O": 0, "S-person_name": 1, "S-credential": 2}
    row = {
        "text": "Alice waits",
        "spans": [[0, 5, "source_person"]],
        "label_space": "v1",
        "unknown_primary_types": ["credential"],
    }
    data = SpanDataset([row], tok, old, 32, secondary_label2id=new, mapped_outside_by_row=True)
    batch = PartialLabelDataCollator(tok)([data[0]])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(1, 2, 3))

        def forward(self, **inputs):
            assert "mapped_outside_allowed" not in inputs
            return SimpleNamespace(logits=self.scores)

    class Trainer(DualHeadLossMixin):
        dual_head_retired = True
        o_token_loss_weight = 0.75

        def bioes_loss_options(self, *args, **kwargs):
            return {}

    trainer = Trainer()
    trainer.correctness_map = SimpleNamespace(
        old_to_new=torch.tensor([[True, False, False], [False, True, False]]),
        old_outside_id=0,
        new_outside_id=0,
        new_labels=tuple(new),
    )
    model = Model()
    loss = trainer.compute_loss(model, dict(batch))
    expected = (torch.log(torch.tensor(3.0)) + 0.75 * torch.log(torch.tensor(1.5))) / 1.75
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.scores.grad[0, 1, 1] > 0  # known corpus negatives suppress false names
    assert model.scores.grad[0, 1, 2] < 0  # unannotated credentials are not false negatives
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    assert model.scores[0, 1, 1] < 0 < model.scores[0, 1, 2]
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = Model()
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    torch.testing.assert_close(
        trainer.compute_loss(restored, dict(batch)), trainer.compute_loss(model, dict(batch))
    )

    # Removing the coverage-aware mask returns ordinary mapped O supervision.
    baseline = dict(batch)
    baseline.pop("mapped_outside_allowed")
    zero = Model()
    baseline_loss = trainer.compute_loss(zero, baseline)
    torch.testing.assert_close(baseline_loss, torch.log(torch.tensor(3.0)))
    baseline_loss.backward()
    assert zero.scores.grad[0, 1, 2] > 0
