import math
import random
from collections import Counter

import pytest

from trainlib import WeightedLengthBatchSampler, exposure_gap_diagnostics


def _setup(n=2000, seed=4):
    rng = random.Random(seed)
    lengths = [rng.randint(6, 80) for _ in range(n)]
    weights = [rng.random() ** 3 + 1e-3 for _ in range(n)]  # skewed: many rows expected < 1 per epoch
    return lengths, weights


def _run(policy, epochs, lengths, weights, epoch_examples=640):
    sampler = WeightedLengthBatchSampler(
        lengths=lengths,
        weights=weights,
        batch_size=8,
        gradient_accumulation_steps=8,
        seed=173,
        epoch_examples=epoch_examples,
        length_window_steps=10,
        draw_policy=policy,
    )
    return [batch for _ in range(epochs) for batch in sampler]


def test_carried_draws_every_row_the_floor_or_ceiling_of_its_cumulative_expectation():
    lengths, weights = _setup()
    total = sum(weights)
    for epochs in (1, 3, 7):
        batches = _run("carried", epochs, lengths, weights)
        counts = Counter(i for b in batches for i in b)
        draws = epochs * 640
        for row, weight in enumerate(weights):
            expected = draws * weight / total
            assert math.floor(expected) - 1e-9 <= counts[row] <= math.ceil(expected) + 1e-9, (row, expected)


def test_systematic_is_exact_per_epoch_but_not_across_epochs():
    lengths, weights = _setup()
    total = sum(weights)
    batches = _run("systematic", 7, lengths, weights)
    counts = Counter(i for b in batches for i in b)
    outside = sum(
        not (math.floor(7 * 640 * w / total) <= counts[r] <= math.ceil(7 * 640 * w / total))
        for r, w in enumerate(weights)
    )
    assert outside > 0


def test_diagnostics_separate_independent_from_carried_exposure():
    lengths, weights = _setup()
    independent = exposure_gap_diagnostics(
        _run("independent", 20, lengths, weights), gradient_accumulation_steps=8, weights=weights
    )
    carried = exposure_gap_diagnostics(
        _run("carried", 20, lengths, weights), gradient_accumulation_steps=8, weights=weights
    )
    # Independent draws sit near the binomial reference; the carried wheel is far more even.
    assert independent["count_dispersion"] == pytest.approx(1.0, abs=0.25)
    assert carried["count_dispersion"] < 0.2
    assert carried["gap_cv_ratio"] < independent["gap_cv_ratio"]
