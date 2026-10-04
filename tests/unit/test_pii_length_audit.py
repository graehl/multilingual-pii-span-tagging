from types import SimpleNamespace

import pytest
import torch

from scripts.pii_encoder_train import LengthAuditCollator, WeightedSamplingTrainerMixin


def _base(features):
    assert all("pii_binned_length" not in f for f in features), "audit field must not reach the base collator"
    width = max(len(f["input_ids"]) for f in features)
    return {
        "input_ids": torch.tensor([f["input_ids"] + [1] * (width - len(f["input_ids"])) for f in features])
    }


def test_collator_flags_items_whose_encoded_length_left_the_batched_length():
    # Batched as length 10; one item encodes to 10 (0%), one to 12 (+20%, beyond 10%).
    features = [
        {"input_ids": list(range(10)), "pii_binned_length": 10},
        {"input_ids": list(range(12)), "pii_binned_length": 10},
    ]
    batch = LengthAuditCollator(_base)(features)
    items, violations, deviation = batch["pii_length_audit"].tolist()
    assert (items, violations) == (2, 1)
    assert deviation == pytest.approx(0.2)
    assert "pii_binned_length" in features[0], "the caller's features are not mutated"


def test_items_without_a_batched_length_are_not_audited():
    batch = LengthAuditCollator(_base)([{"input_ids": [0, 1, 2]}])
    assert batch["pii_length_audit"].tolist() == [0.0, 0.0, 0.0]


class _Base:
    def training_step(self, model, inputs, num_items_in_batch=None):
        assert "pii_length_audit" not in inputs
        return torch.tensor(0.0)

    def train(self, *args, **kwargs):
        return "done"


class _Trainer(WeightedSamplingTrainerMixin, _Base):
    state = SimpleNamespace(global_step=7)


def test_trainer_strips_audit_warns_per_violating_batch_and_summarizes(capsys):
    trainer = _Trainer()
    trainer.training_step(None, {"pii_length_audit": torch.tensor([8.0, 0.0, 0.05])})
    trainer.training_step(None, {"pii_length_audit": torch.tensor([8.0, 2.0, 0.30])})
    assert trainer.train() == "done"
    lines = capsys.readouterr().out.splitlines()
    warnings = [line for line in lines if "TRAIN-LENGTH-AUDIT: warning" in line]
    assert len(warnings) == 1 and "step=7 2/8" in warnings[0] and "max 30.0%" in warnings[0]
    summary = next(line for line in lines if "TRAIN-LENGTH-AUDIT: summary" in line)
    assert "summary warning items=16 batches=2" in summary and "items=2 (12.50%)" in summary
    assert "batches=1 max_deviation=30.0%" in summary
