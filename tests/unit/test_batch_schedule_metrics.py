import random

import pytest

from trainlib import (
    WeightedLengthBatchSampler,
    batch_schedule_metrics,
    schedule_score,
    sweep_batch_schedules,
)


def synthetic(n=6000, seed=11):
    """Sentence-like lengths with a sparse long tail, two branches, and a few heavy rows."""
    rng = random.Random(seed)
    lengths = [rng.randint(6, 60) if rng.random() < 0.96 else rng.randint(150, 512) for _ in range(n)]
    keys = ["gold" if rng.random() < 0.4 else "base" for _ in range(n)]
    weights = [(0.3 / keys.count("gold")) if k == "gold" else (0.7 / keys.count("base")) for k in keys]
    for heavy in rng.sample(range(n), 20):
        weights[heavy] *= 40
    return lengths, weights, keys


def test_metrics_on_a_hand_built_schedule():
    lengths = [10, 10, 20, 20]
    # Two steps of one batch each; row 0 appears twice with the same mate the second time.
    metrics = batch_schedule_metrics([[0, 1], [0, 1]], lengths=lengths, gradient_accumulation_steps=1)
    assert metrics["padding_waste"] == 0.0
    assert metrics["duplicate_row_batches"] == 0.0
    assert metrics["repeated_mates"] == 1.0
    padded = batch_schedule_metrics([[0, 2]], lengths=lengths, gradient_accumulation_steps=1)
    assert padded["padding_waste"] == pytest.approx(1 - 30 / 40)
    twice = batch_schedule_metrics([[0, 0]], lengths=lengths, gradient_accumulation_steps=1)
    assert twice["duplicate_row_batches"] == 1.0


def test_band_random_beats_sorted_window_on_randomness_at_small_padding_cost():
    lengths, weights, keys = synthetic()

    def make(configuration):
        return WeightedLengthBatchSampler(
            lengths=lengths,
            weights=weights,
            batch_size=8,
            gradient_accumulation_steps=8,
            seed=173,
            epoch_examples=3200,
            length_window_steps=50,
            **configuration,
        )

    results = sweep_batch_schedules(
        make,
        [{"batch_formation": "sorted-window"}, {"batch_formation": "band-random", "padding_budget": 0.10}],
        epochs=4,
        lengths=lengths,
        gradient_accumulation_steps=8,
        weights=weights,
        keys=keys,
    )
    (_, sorted_metrics, _), (_, band_metrics, _) = results
    assert band_metrics["duplicate_row_batches"] < sorted_metrics["duplicate_row_batches"]
    assert band_metrics["repeated_mates"] < sorted_metrics["repeated_mates"]
    assert band_metrics["padding_waste"] <= 0.10
    # Both arrange exact weighted draws; after the first epoch the draws differ only
    # because batching consumes the shared random stream differently.
    assert band_metrics["key_share_error"] < 0.01
    assert sorted_metrics["key_share_error"] < 0.01


def test_scores_rank_configurations_and_reject_unknown_metrics():
    assert schedule_score(
        {"padding_waste": 0.1, "repeated_mates": 0.2}, {"padding_waste": 1, "repeated_mates": 2}
    ) == (pytest.approx(0.5))
    with pytest.raises(ValueError, match="unknown schedule metrics"):
        schedule_score({"padding_waste": 0.1}, {"speed": 1})
