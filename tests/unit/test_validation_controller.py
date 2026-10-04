"""Unit tests for trainlib.ValidationController in isolation (no Trainer).

The characterization test proves behavior-preservation through the delegating
RegionScaleTrainer; these prove the controller is genuinely host-agnostic — driven by a
plain recording-hooks object — which is the point of the extraction (the alignment
finetuner reuses it). Also exercises the simple/debuggable mode: no LR controller
attached (the _try_patience_lr_anneal hook always declines), so observe() reduces to
plain patience early-stopping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainlib import ValidationController  # noqa: E402


class RecordingHooks:
    """Minimal non-Trainer host implementing the ValidationController hook protocol."""

    def __init__(self, *, exact_candidates=(), exact_losses=()):
        self.events: list = []
        self._cands = iter(exact_candidates)
        self._exacts = iter(exact_losses)
        self.try_anneal_calls = 0

    # LR hooks — inert here (no PatienceLrController attached): the simple/debuggable mode.
    def _update_val_cycle_progress(self, running_loss):
        pass

    def _maybe_adjust_patience_lr_rebound(self, running_loss):
        pass

    def _try_patience_lr_anneal(self, running_loss):
        self.try_anneal_calls += 1
        return False

    def _reset_val_cadence(self, step):
        pass

    # effect hooks — record only
    def _resume_recent_checkpoint_capture(self, *, reason):
        self.events.append(("resume", reason))

    def _save_val_best(self):
        self.events.append(("save_val_best",))

    def _write_val_decode_snapshot(self, event, *, running_loss=None):
        self.events.append(("snapshot", event))

    def _exact_eval_candidate(self):
        return next(self._cands)

    def _compute_exact_val_loss(self, state):
        return next(self._exacts)

    def _save_exact_best_state(self, *, loss, source, source_steps, state):
        self.events.append(("save_exact_best", source))

    def _pause_recent_checkpoint_capture(self, *, rewind_to_step, reason):
        self.events.append(("pause", rewind_to_step))


def test_simple_mode_improve_then_patience_stop():
    vc = ValidationController(val_patience=2, val_min_delta=0.001)
    hooks = RecordingHooks()

    assert vc.observe(1.0, step=10, cycles_complete=1, hooks=hooks) is False
    assert (vc.val_best_loss, vc.patience_count) == (1.0, 0)
    assert vc.observe(0.9, step=20, cycles_complete=2, hooks=hooks) is False
    assert (vc.val_best_loss, vc.patience_count) == (0.9, 0)

    # Plateau: 0.9 is not below 0.9 - min_delta, so patience climbs to the cap and stops.
    assert vc.observe(0.9, step=30, cycles_complete=3, hooks=hooks) is False
    assert vc.patience_count == 1
    assert vc.observe(0.9, step=40, cycles_complete=4, hooks=hooks) is True
    assert vc.patience_count == 2
    assert ("snapshot", "stop") in hooks.events
    # On exhaustion the controller consulted the (absent) LR controller, which declined.
    assert hooks.try_anneal_calls == 1


def test_exact_first_baseline_resets_patience():
    vc = ValidationController(val_patience=3, val_min_delta=0.001, exact_early_stopping=0.05)
    hooks = RecordingHooks(exact_candidates=[("cur", "STATE", [20])], exact_losses=[0.8])

    assert vc.observe(1.0, step=10, cycles_complete=1, hooks=hooks) is False  # NEW BEST
    # 1.10 >= val_best(1.0) + 0.05 triggers an exact recheck; first check sets the baseline.
    assert vc.observe(1.10, step=20, cycles_complete=2, hooks=hooks) is False
    assert vc.exact_checks == 1
    assert vc.exact_best_loss == 0.8
    assert vc.patience_count == 0  # establishing the baseline resets patience
    assert ("save_exact_best", "cur") in hooks.events


def test_exact_disabled_by_default():
    vc = ValidationController(val_patience=3, val_min_delta=0.001)
    assert vc.exact_enabled() is False


if __name__ == "__main__":
    test_simple_mode_improve_then_patience_stop()
    test_exact_first_baseline_resets_patience()
    test_exact_disabled_by_default()
    print("ValidationController standalone tests OK")
