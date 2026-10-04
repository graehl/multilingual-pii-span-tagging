import random
from collections import Counter

from trainlib import WeightedLengthBatchSampler, _band_random_physical


def _lengths(n=4000, seed=3):
    rng = random.Random(seed)
    # Dense body plus a sparse long tail, like sentence lengths.
    return [rng.randint(8, 60) if rng.random() < 0.97 else rng.randint(200, 512) for _ in range(n)]


def test_batching_only_partitions_the_draws():
    lengths = _lengths()
    draws = list(range(len(lengths))) * 2
    batches, stats = _band_random_physical(
        list(draws),
        lengths=lengths,
        batch_size=8,
        rng=random.Random(1),
        gradient_accumulation_steps=8,
        logical_steps=1000,
    )
    assert Counter(i for b in batches for i in b) == Counter(draws)
    assert stats.items == len(draws)
    assert stats.band_random_items + stats.sparse_sorted_items + stats.remainder_sorted_items == len(draws)


def test_occupants_share_a_band_except_where_the_tail_forces_a_stretch():
    lengths = _lengths()
    batches, stats = _band_random_physical(
        list(range(len(lengths))),
        lengths=lengths,
        batch_size=8,
        rng=random.Random(1),
        gradient_accumulation_steps=8,
        logical_steps=63,
    )
    spreads = sorted(max(lengths[i] for i in b) / min(lengths[i] for i in b) for b in batches)
    # A +-5% band bounds the spread by 1.05/0.95 when no stretch was needed.
    unstretched = (
        len(batches)
        - stats.stretched_batches
        - (stats.sparse_sorted_items + stats.remainder_sorted_items) // 8
    )
    assert sum(spread <= 1.05 / 0.95 + 1e-9 for spread in spreads) >= unstretched


def test_co_occupants_are_random_not_fixed_neighbours():
    lengths = [20] * 800  # one dense band: sorted order would fix every batch
    first, _ = _band_random_physical(
        list(range(800)),
        lengths=lengths,
        batch_size=8,
        rng=random.Random(1),
        gradient_accumulation_steps=1,
        logical_steps=1,
    )
    second, _ = _band_random_physical(
        list(range(800)),
        lengths=lengths,
        batch_size=8,
        rng=random.Random(2),
        gradient_accumulation_steps=1,
        logical_steps=1,
    )
    assert {frozenset(b) for b in first} != {frozenset(b) for b in second}


def test_sampler_formations_draw_the_same_items():
    lengths = _lengths()
    weights = [1.0 + (i % 5) for i in range(len(lengths))]
    drawn = {}
    for formation in ("sorted-window", "band-random"):
        sampler = WeightedLengthBatchSampler(
            lengths=lengths,
            weights=weights,
            batch_size=8,
            gradient_accumulation_steps=8,
            seed=173,
            epoch_examples=3200,
            length_window_steps=50,
            batch_formation=formation,
        )
        drawn[formation] = Counter(i for b in sampler for i in b)
    assert drawn["sorted-window"] == drawn["band-random"]
