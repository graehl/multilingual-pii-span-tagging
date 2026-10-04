import subprocess
from pathlib import Path

import pytest

from checkpoint_mirror import CheckpointMirror


def test_mirror_publishes_complete_run_and_rotates_obsolete_generations(tmp_path):
    source = tmp_path / "worker"
    source.mkdir()
    checkpoint = source / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (source / "best").symlink_to(checkpoint.name)
    destination = tmp_path / "home" / "run"
    mirror = CheckpointMirror(str(destination), interval_seconds=0)

    mirror.save(source)
    previous = destination.resolve()
    assert (destination / "best" / "optimizer.pt").read_bytes() == b"optimizer"
    (checkpoint / "model.safetensors").write_bytes(b"new weights")
    mirror.save(source)
    assert destination.resolve() != previous
    assert not previous.exists()
    assert (destination / "best" / "model.safetensors").read_bytes() == b"new weights"
    mirror.save(source)
    generations = list((destination.parent / ".run.mirror").glob("snapshot-*"))
    assert len(generations) == 1
    assert all(Path(p).is_dir() for p in generations)


def test_failed_transfer_keeps_last_published_copy(tmp_path, monkeypatch):
    source = tmp_path / "worker"
    source.mkdir()
    (source / "weights").write_bytes(b"old")
    destination = tmp_path / "home"
    mirror = CheckpointMirror(str(destination), interval_seconds=0)
    mirror.save(source)
    old = destination.resolve()
    (source / "weights").write_bytes(b"new")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(23, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        mirror.save(source)
    assert destination.resolve() == old
    assert (destination / "weights").read_bytes() == b"old"


def test_real_trainer_mirrors_resumable_checkpoint_and_final_output(tmp_path):
    from types import SimpleNamespace

    import torch
    from transformers import Trainer, TrainingArguments

    from trainlib import CheckpointMirrorCallback, is_valid_trainer_checkpoint

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, input_ids, labels):
            logits = self.linear(input_ids)
            return {"loss": ((logits - labels) ** 2).mean(), "logits": logits}

    output = tmp_path / "worker"
    destination = tmp_path / "home"
    callback = CheckpointMirrorCallback(
        SimpleNamespace(
            checkpoint_mirror=str(destination),
            checkpoint_mirror_interval=0,
            checkpoint_mirror_timeout=30,
            checkpoint_mirror_ssh="ssh",
        )
    )
    trainer = Trainer(
        model=Model(),
        args=TrainingArguments(
            output_dir=str(output),
            max_steps=2,
            save_steps=1,
            save_total_limit=1,
            report_to=[],
            use_cpu=True,
            disable_tqdm=True,
        ),
        train_dataset=[{"input_ids": torch.tensor([1.0]), "labels": torch.tensor([2.0])}] * 8,
        callbacks=[callback],
    )
    trainer.train()
    assert is_valid_trainer_checkpoint(destination / "checkpoint-2")
    assert not (destination / "checkpoint-1").exists()
    (output / "final.txt").write_text("finished")
    callback.finish(output)
    assert (destination / "final.txt").read_text() == "finished"
