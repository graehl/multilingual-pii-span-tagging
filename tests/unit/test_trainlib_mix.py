import math

import pytest

from trainlib_mix import apply_caps, share_duplicate_mass


def test_duplicate_inputs_share_one_average_weight():
    rows = [{"id": "a", "audio": "x"}, {"id": "b", "audio": "x"}, {"id": "c", "audio": "y"}]
    weights, receipt = share_duplicate_mass(
        rows, [1.0, 3.0, 2.0], key=lambda row: row["audio"], identity="audio", schema="test/v1"
    )
    # Group x averages to 2.0, split as 1.0 each; y keeps 2.0; then normalized.
    assert weights == pytest.approx([0.25, 0.25, 0.5])
    assert receipt["multiply_annotated_inputs"] == 1 and receipt["schema"] == "test/v1"


def test_no_duplicates_returns_weights_unchanged():
    rows = [{"k": 1}, {"k": 2}]
    assert share_duplicate_mass(rows, None, key=lambda row: row["k"], identity="k", schema="s")[0] is None


def test_speaker_caps_move_mass_between_movable_rows_only():
    rows = [
        {"speaker": "s1", "pool": "human", "sampling_weight": 0.4},
        {"speaker": "s1", "pool": "teacher", "sampling_weight": 0.3},
        {"speaker": "s2", "pool": "teacher", "sampling_weight": 0.3},
    ]
    report = apply_caps(
        rows,
        {"default": 0.5, "groups": {}},
        group=lambda row: row["speaker"],
        movable=lambda row: row["pool"] == "teacher",
    )
    assert rows[0]["sampling_weight"] == 0.4  # fixed row untouched
    assert rows[1]["sampling_weight"] == pytest.approx(0.1)
    assert rows[2]["sampling_weight"] == pytest.approx(0.5)
    assert math.fsum(row["sampling_weight"] for row in rows) == pytest.approx(1.0)
    assert report["capped"]["s1"]["after"] == pytest.approx(0.5)
