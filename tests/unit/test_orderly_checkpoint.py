import json
import signal
from types import SimpleNamespace

import pytest

import trainlib
from trainlib import OrderlyCheckpointCallback


def _control():
    return SimpleNamespace(should_save=False, should_training_stop=False)


def _write_resume_checkpoint(root, step):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    return checkpoint


def test_orderly_stop_requests_save_and_verifies_resume_checkpoint(tmp_path) -> None:
    callback = OrderlyCheckpointCallback()
    callback.request_stop("test termination", signum=signal.SIGTERM)
    control = _control()

    callback.on_step_end(None, SimpleNamespace(global_step=17), control)

    assert control.should_save is True
    assert control.should_training_stop is True
    checkpoint = _write_resume_checkpoint(tmp_path, 17)
    callback.on_save(None, SimpleNamespace(global_step=17), control)
    assert callback.require_resumable_checkpoint(tmp_path) == checkpoint
    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
        callback.raise_if_terminated()


def test_orderly_stop_rejects_incomplete_checkpoint(tmp_path) -> None:
    callback = OrderlyCheckpointCallback()
    callback.request_stop("test termination")
    control = _control()
    callback.on_step_end(None, SimpleNamespace(global_step=9), control)
    callback.on_save(None, SimpleNamespace(global_step=9), control)

    with pytest.raises(RuntimeError, match="incomplete resume checkpoint"):
        callback.require_resumable_checkpoint(tmp_path)


def test_checkpoint_only_request_is_consumed_after_save() -> None:
    callback = OrderlyCheckpointCallback()
    callback.request_checkpoint("test snapshot")
    first = _control()

    callback.on_step_end(None, SimpleNamespace(global_step=4), first)

    assert first.should_save is True
    assert first.should_training_stop is False
    callback.on_save(None, SimpleNamespace(global_step=4), first)
    second = _control()
    callback.on_step_end(None, SimpleNamespace(global_step=5), second)
    assert second.should_save is False
    assert second.should_training_stop is False


def test_second_termination_signal_forces_default_exit(monkeypatch) -> None:
    callback = OrderlyCheckpointCallback()
    signal_calls = []
    kill_calls = []
    monkeypatch.setattr(trainlib.signal, "signal", lambda *args: signal_calls.append(args))
    monkeypatch.setattr(trainlib.os, "getpid", lambda: 1234)
    monkeypatch.setattr(trainlib.os, "kill", lambda *args: kill_calls.append(args))
    monkeypatch.setattr(trainlib.os, "write", lambda *_args: 0)

    callback._handle_termination_signal(signal.SIGTERM, None)
    callback._handle_termination_signal(signal.SIGTERM, None)

    assert callback.termination_signal == signal.SIGTERM
    assert signal_calls == [(signal.SIGTERM, signal.SIG_DFL)]
    assert kill_calls == [(1234, signal.SIGTERM)]


def test_external_stop_predicate_is_checked_at_optimizer_boundary() -> None:
    callback = OrderlyCheckpointCallback(stop_requested=lambda: True, stop_reason="kill marker")
    control = _control()

    callback.on_step_end(None, SimpleNamespace(global_step=3), control)

    assert control.should_save is True
    assert control.should_training_stop is True
