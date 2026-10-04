"""Unit tests for trainlib.PatienceLrController in isolation (no Trainer).

The characterization test proves behavior-preservation through the delegating
RegionScaleTrainer; these prove the controller is host-agnostic — driven by a plain
fake host exposing only the cumulative scale, the val-cycle counter, and the optimizer
mutation hook.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainlib import PatienceLrController  # noqa: E402

DEFAULTS = dict(
    patience_lr_min_factor=0.0,
    patience_lr_floor=0.0,
    patience_lr_max_increase_factor=1.0,
    patience_lr_rebound_ratio=1.25,
    patience_lr_rebound_ema=2.0,
    patience_lr_rebound_window=3,
    patience_lr_rebound_up_power=0.5,
    patience_lr_rebound_down_power=0.5,
    patience_lr_rebound_down_overshoot=1.0,
    val_min_delta=0.001,
)


class FakeHost:
    """Minimal non-Trainer host exposing the PatienceLrController hook surface."""

    def __init__(self, scale=1.0):
        self._patience_lr_scale = float(scale)
        self._val_cycles_complete = 0
        self.applied: list = []

    def _apply_patience_lr_scale_change(self, next_scale, running_loss, *, reason, allow_increase):
        self.applied.append((round(float(next_scale), 6), allow_increase))
        self._patience_lr_scale = float(next_scale)
        return True


def test_anneal_halves_scale_then_caps_at_stages():
    lr = PatienceLrController(patience_lr_factor=0.5, patience_lr_anneal_stages=1, **DEFAULTS)
    host = FakeHost()

    assert lr._try_patience_lr_anneal(1.0, hooks=host) is True
    assert host.applied == [(0.5, False)]  # halved LR, a decrease
    assert host._patience_lr_scale == 0.5
    # Stage cap reached (events=1 >= anneal_stages=1): the next exhaustion declines.
    assert lr._try_patience_lr_anneal(1.0, hooks=host) is False
    assert host.applied == [(0.5, False)]  # no further LR change


def test_strong_progress_drives_rebound_up():
    lr = PatienceLrController(patience_lr_factor=0.5, patience_lr_anneal_stages=1, **DEFAULTS)
    host = FakeHost()

    lr._update_val_cycle_progress(1.0)  # seed the progress tracker
    host._val_cycles_complete = 1
    assert lr._try_patience_lr_anneal(1.0, hooks=host) is True  # scale -> 0.5, rebound watch starts
    host.applied.clear()

    host._val_cycles_complete = 2
    lr._update_val_cycle_progress(0.5)  # a sharp improvement
    lr._maybe_adjust_patience_lr_rebound(0.5, hooks=host)
    # Rebound up: geometric move 0.5 -> 0.5*sqrt(2) toward the pre-anneal scale, allow_increase.
    assert host.applied == [(0.707107, True)]
    assert round(host._patience_lr_scale, 6) == 0.707107  # 0.5 * sqrt(2)


def test_no_anneal_when_factor_unity():
    lr = PatienceLrController(patience_lr_factor=1.0, patience_lr_anneal_stages=1, **DEFAULTS)
    host = FakeHost()
    assert lr._try_patience_lr_anneal(1.0, hooks=host) is False
    assert host.applied == []


if __name__ == "__main__":
    test_anneal_halves_scale_then_caps_at_stages()
    test_strong_progress_drives_rebound_up()
    test_no_anneal_when_factor_unity()
    print("PatienceLrController standalone tests OK")
