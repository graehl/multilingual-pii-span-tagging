from collections import Counter

import pytest

from trainlib import (
    RecursiveWeightedBatchSampler,
    WeightedLengthBatchSampler,
    example_weights_from_pools,
)


def _flatten(batches):
    return [index for batch in batches for index in batch]


def test_pool_weights_compile_to_total_mass_and_within_pool_factors() -> None:
    labels = ["primary", "primary", "replay", "replay"]

    equal = example_weights_from_pools(labels, {"primary": 0.8, "replay": 0.2})
    factored = example_weights_from_pools(
        labels,
        {"primary": 0.8, "replay": 0.2},
        example_factors=[1.0, 3.0, 1.0, 1.0],
    )

    assert equal == pytest.approx([0.4, 0.4, 0.1, 0.1])
    assert factored == pytest.approx([0.2, 0.6, 0.1, 0.1])
    with pytest.raises(ValueError, match="missing weights"):
        example_weights_from_pools(labels, {"primary": 1.0})
    with pytest.raises(ValueError, match="no examples"):
        example_weights_from_pools(labels, {"primary": 0.8, "replay": 0.1, "unused": 0.1})
    zero = example_weights_from_pools(labels, {"primary": 1.0, "replay": 0.0})
    assert zero == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_uniform_weight_sampler_is_a_deterministic_shuffled_full_pass() -> None:
    kwargs = {
        "lengths": [(index * 17) % 100 + 1 for index in range(101)],
        "weights": [1.0] * 101,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "seed": 31,
        "length_window_steps": 4,
    }
    first = WeightedLengthBatchSampler(**kwargs)
    replay = WeightedLengthBatchSampler(**kwargs)

    first_epoch = list(first)
    second_epoch = list(first)

    assert len(first_epoch) == len(first) == 13
    assert sorted(_flatten(first_epoch)) == list(range(101))
    assert first_epoch == list(replay)
    assert first_epoch != second_epoch
    assert sorted(_flatten(second_epoch)) == list(range(101))


def test_random_formation_ignores_lengths_and_preserves_draws():
    options = dict(weights=[1.0] * 64, batch_size=8, gradient_accumulation_steps=1, seed=19)
    forward = WeightedLengthBatchSampler(lengths=list(range(1, 65)), batch_formation="random", **options)
    backward = WeightedLengthBatchSampler(lengths=list(range(64, 0, -1)), batch_formation="random", **options)
    grouped = WeightedLengthBatchSampler(lengths=list(range(1, 65)), **options)
    random_batches = list(forward)
    assert random_batches == list(backward)
    assert sorted(_flatten(random_batches)) == sorted(_flatten(list(grouped))) == list(range(64))
    assert max(random_batches[0]) - min(random_batches[0]) > 20


def test_weighted_sampler_hits_pool_mass_and_rotates_full_support() -> None:
    pool_labels = ["large"] * 800 + ["small"] * 200
    weights = example_weights_from_pools(pool_labels, {"large": 0.5, "small": 0.5})
    lengths = [(index * 29) % 256 + 4 for index in range(1000)]
    sampler = WeightedLengthBatchSampler(
        lengths=lengths,
        weights=weights,
        batch_size=16,
        gradient_accumulation_steps=4,
        epoch_examples=1000,
        seed=7,
        length_window_steps=8,
    )

    usage = Counter()
    pool_usage = Counter()
    actual_tokens = padded_tokens = 0
    for _ in range(20):
        for batch in sampler:
            batch_lengths = [lengths[index] for index in batch]
            actual_tokens += sum(batch_lengths)
            padded_tokens += max(batch_lengths) * len(batch_lengths)
            for index in batch:
                usage[index] += 1
                pool_usage[pool_labels[index]] += 1

    assert pool_usage["large"] / sum(pool_usage.values()) == pytest.approx(0.5, abs=0.002)
    assert min(usage.values()) > 0
    assert len(usage) == len(weights)
    assert padded_tokens / actual_tokens < 1.08


def test_recursive_sampler_emits_homogeneous_full_batches_and_hits_probabilities() -> None:
    pool_labels = ["primary"] * 600 + ["pretraining"] * 400
    weights = example_weights_from_pools(
        pool_labels,
        {"primary": 0.7, "pretraining": 0.3},
    )
    lengths = [(index * 29) % 256 + 4 for index in range(1000)]
    sampler = RecursiveWeightedBatchSampler(
        lengths=lengths,
        weights=weights,
        pool_keys=pool_labels,
        batch_size=8,
        gradient_accumulation_steps=4,
        epoch_examples=32000,
        seed=7,
        length_bucket_width=32,
        default_variant_probability=0.1,
        variant_probability_by_pool={"pretraining": 0.8},
    )

    pool_batches = Counter()
    variant_batches = Counter()
    actual_tokens = padded_tokens = 0
    for batch in sampler:
        assert len(batch) == 8
        assert len({variant for _index, variant in batch}) == 1
        assert len({pool_labels[index] for index, _variant in batch}) == 1
        assert len({index for index, _variant in batch}) == len(batch)
        pool = pool_labels[batch[0][0]]
        variant = batch[0][1]
        pool_batches[pool] += 1
        variant_batches[pool] += int(variant)
        batch_lengths = [lengths[index] for index, _variant in batch]
        actual_tokens += sum(batch_lengths)
        padded_tokens += max(batch_lengths) * len(batch_lengths)

    assert pool_batches["pretraining"] / sum(pool_batches.values()) == pytest.approx(0.3, abs=0.02)
    assert variant_batches["primary"] / pool_batches["primary"] == pytest.approx(0.1, abs=0.02)
    assert variant_batches["pretraining"] / pool_batches["pretraining"] == pytest.approx(
        0.8,
        abs=0.03,
    )
    assert padded_tokens / actual_tokens < 1.15
    assert sampler.last_epoch_statistics["physical_batches"] == 4000
    exposure = sampler.last_epoch_statistics["packing_exposure"]
    assert exposure["reference"] == "independent_weighted_rows_within_selected_pool"
    assert exposure["maximum_absolute_bucket_share_error"] < 0.03


def test_recursive_sampler_can_disable_length_binning() -> None:
    sampler = RecursiveWeightedBatchSampler(
        lengths=[4, 40, 80, 120] * 4,
        weights=[1.0] * 16,
        pool_keys=["pool"] * 16,
        batch_size=4,
        gradient_accumulation_steps=1,
        epoch_examples=160,
        seed=19,
        length_bucket_width=0,
    )

    list(sampler)

    exposure = sampler.last_epoch_statistics["packing_exposure"]
    assert exposure["target_bucket_shares_by_pool"] == {"pool": {"0": 1.0}}
    assert exposure["realized_bucket_shares_by_pool"] == {"pool": {"0": 1.0}}
    assert exposure["maximum_absolute_bucket_share_error"] == 0.0


def test_recursive_sampler_rounds_to_complete_logical_steps_and_replays_with_replacement() -> None:
    sampler = RecursiveWeightedBatchSampler(
        lengths=[4, 5, 6, 7],
        weights=[1.0] * 4,
        pool_keys=["pool"] * 4,
        batch_size=4,
        gradient_accumulation_steps=3,
        epoch_examples=13,
        seed=11,
        length_bucket_width=16,
    )

    batches = list(sampler)

    assert len(batches) == 6
    assert sum(len(batch) for batch in batches) == 24
    assert all(sorted(index for index, _variant in batch) == [0, 1, 2, 3] for batch in batches)
