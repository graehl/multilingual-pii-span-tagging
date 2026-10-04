"""Standalone metrics for the N-stream LogicalMixBatchSampler.

Per topics/logical-batch-stream-mixing.md the sampler is judged on a
non-randomness-vs-speed Pareto, so this suite measures the metrics directly and
ratchets them against the best values achieved so far. Tightening a bound is a
deliberate edit; a regression should fail.

Metrics:
  - stream proportions amortize to the target weights (priority 1);
  - the anchor (largest, without-replacement) covers each item exactly N times
    over N epochs; replay streams cover ~N on average;
  - per-logical-batch stream diversity (all streams present) is near-total when
    weights allow;
  - length jaggedness and direct padded-token compute inflation stay low (speed);
  - per-logical-batch length buckets resemble the epoch distribution, so an
    optimizer step is not systematically all-long or all-short;
  - co-residency is not static across epochs (randomness).
"""

import random
from pathlib import Path

from trainlib import LogicalMixBatchSampler, MixStream, batch_schedule_metrics

ROOT = Path(__file__).resolve().parents[2]
GAS = 8
BATCH = 4
LOGICAL = BATCH * GAS
N_EPOCHS = 20

# Ratchet bounds — at least as good as best achieved. Tighten deliberately.
# 2026-07-07: persistent replay decks/deficit -> proportions near-exact, replay
# coverage cv ~0.06 (was 0.26).
BEST = {
    "prop_abs_err": 0.005,  # |empirical - target| per stream
    "diverse_frac_min": 0.99,  # fraction of logical batches containing every stream
    "pad_waste_max": 0.13,  # mean physical-batch padding waste
    "coverage_cv_max": 0.08,  # per-replay-stream coverage coeff-of-variation (uniformity)
    "static_co_resident_items_max": 0,  # items whose exact logical-batch partners never change
    "co_resident_partner_frac_max": 0.25,  # most-common exact partner-set fraction for any item
    "loose_compute_inflation_max": 1.04,  # padded-token compute multiplier at chosen strictness
    "loose_length_tv_max": 0.065,  # logical-step length-bucket TV at chosen strictness
    "chosen_diverse_frac_min": 0.99,  # stream_strict_every=8 keeps near-strict stream representation
}


def _streams():
    rng = random.Random(0)
    lengths = [0] * 960
    for i in range(0, 600):
        lengths[i] = rng.randint(8, 24)  # A: anchor, short
    for i in range(600, 900):
        lengths[i] = rng.randint(30, 60)  # B: medium
    for i in range(900, 960):
        lengths[i] = rng.randint(80, 160)  # C: long, low count
    streams = [
        MixStream(0, 600, 0.70, "A"),
        MixStream(600, 300, 0.15, "B"),
        MixStream(900, 60, 0.15, "C"),
    ]
    return lengths, streams


def _stream_of(idx: int) -> str:
    return "A" if idx < 600 else ("B" if idx < 900 else "C")


def _run(lengths, streams, epochs, *, stream_strict_every=1):
    s = LogicalMixBatchSampler(
        lengths=lengths,
        batch_size=BATCH,
        gradient_accumulation_steps=GAS,
        seed=123,
        streams=streams,
        stream_strict_every=stream_strict_every,
    )
    epoch_phys = [list(iter(s)) for _ in range(epochs)]
    return epoch_phys


def _metrics(lengths, epoch_phys):
    """The shared schedule metrics (trainlib.batch_schedule_metrics) over all epochs."""
    assert all(len(phys) % GAS == 0 for phys in epoch_phys), "epochs must hold whole optimizer steps"
    return batch_schedule_metrics(
        [batch for phys in epoch_phys for batch in phys],
        lengths=lengths,
        gradient_accumulation_steps=GAS,
        keys=[_stream_of(index) for index in range(len(lengths))],
    )


def _pad_waste(lengths, epoch_phys):
    return _metrics(lengths, epoch_phys)["padding_waste"]


def _compute_inflation(lengths, epoch_phys):
    return _metrics(lengths, epoch_phys)["compute_inflation"]


def _diverse_frac(lengths, epoch_phys):
    return 1 - _metrics(lengths, epoch_phys)["steps_missing_a_key"]


def _length_representation_tv(lengths, epoch_phys):
    return _metrics(lengths, epoch_phys)["step_length_tv"]


def test_two_stream_back_compat_still_constructs():
    # streams=None path must still accept the legacy kwargs unchanged
    s = LogicalMixBatchSampler(
        lengths=[1] * 1100,
        primary_count=100,
        generic_count=1000,
        generic_fraction=0.2,
        batch_size=4,
        gradient_accumulation_steps=8,
        seed=0,
    )
    assert "generic_mix_frac=0.2" in s.summary()


def test_nstream_length_matches_streamed_batches():
    lengths, streams = _streams()
    s = LogicalMixBatchSampler(
        lengths=lengths,
        batch_size=BATCH,
        gradient_accumulation_steps=GAS,
        seed=123,
        streams=streams,
        stream_strict_every=8,
    )

    first_len = len(s)
    assert len(list(iter(s))) == first_len
    second_len = len(s)
    assert len(list(iter(s))) == second_len


def test_nstream_length_does_not_materialize_or_mutate_epoch():
    lengths, streams = _streams()
    untouched = LogicalMixBatchSampler(
        lengths=lengths,
        batch_size=BATCH,
        gradient_accumulation_steps=GAS,
        seed=123,
        streams=streams,
        stream_strict_every=8,
    )
    measured = LogicalMixBatchSampler(
        lengths=lengths,
        batch_size=BATCH,
        gradient_accumulation_steps=GAS,
        seed=123,
        streams=streams,
        stream_strict_every=8,
    )

    assert len(measured) == len(untouched)
    assert list(iter(measured)) == list(iter(untouched))


def test_stream_proportions_and_coverage_and_diversity():
    lengths, streams = _streams()
    targets = {"A": 0.70, "B": 0.15, "C": 0.15}
    epoch_phys = _run(lengths, streams, N_EPOCHS)

    from collections import Counter

    usage = Counter()
    stream_use = Counter()
    diverse = 0
    n_logical = 0
    pad_waste = []
    for phys in epoch_phys:
        for lo in range(0, len(phys), GAS):
            window = phys[lo : lo + GAS]
            n_logical += 1
            present = set()
            for b in window:
                for idx in b:
                    usage[idx] += 1
                    stream_use[_stream_of(idx)] += 1
                    present.add(_stream_of(idx))
                blens = [lengths[i] for i in b]
                pad_waste.append(1 - sum(blens) / (max(blens) * len(blens)))
            if present == {"A", "B", "C"}:
                diverse += 1

    tot = sum(stream_use.values())
    for name, target in targets.items():
        assert abs(stream_use[name] / tot - target) <= BEST["prop_abs_err"], (name, stream_use[name] / tot)

    # anchor A is without replacement -> exactly N uses per item
    assert all(usage[i] == N_EPOCHS for i in range(600)), "anchor coverage must be exactly N"
    # no starvation anywhere
    assert min(usage[i] for i in range(960)) >= 1

    # replay-stream coverage uniformity: persistent decks keep each item's use near
    # the stream mean (low coeff of variation), not a fresh binomial subset per epoch
    import statistics as stats

    for name, rg in (("B", range(600, 900)), ("C", range(900, 960))):
        u = [usage[i] for i in rg]
        cv = stats.pstdev(u) / (sum(u) / len(u))
        assert cv <= BEST["coverage_cv_max"], (name, cv)

    assert diverse / n_logical >= BEST["diverse_frac_min"], diverse / n_logical
    assert sum(pad_waste) / len(pad_waste) <= BEST["pad_waste_max"], sum(pad_waste) / len(pad_waste)


def test_strictness_knob_trades_diversity_for_padding():
    lengths, streams = _streams()
    strict = _run(lengths, streams, N_EPOCHS)
    loose = _run(lengths, streams, N_EPOCHS, stream_strict_every=8)

    strict_pad = _pad_waste(lengths, strict)
    loose_pad = _pad_waste(lengths, loose)
    loose_compute = _compute_inflation(lengths, loose)
    strict_length_tv = _length_representation_tv(lengths, strict)
    loose_length_tv = _length_representation_tv(lengths, loose)
    strict_diverse = _diverse_frac(lengths, strict)
    loose_diverse = _diverse_frac(lengths, loose)

    assert loose_compute <= BEST["loose_compute_inflation_max"], loose_compute
    assert loose_pad < strict_pad * 0.2, (strict_pad, loose_pad)
    assert loose_length_tv <= BEST["loose_length_tv_max"], loose_length_tv
    assert loose_length_tv < strict_length_tv, (strict_length_tv, loose_length_tv)
    assert strict_diverse >= BEST["diverse_frac_min"], strict_diverse
    assert loose_diverse >= BEST["chosen_diverse_frac_min"], loose_diverse


def test_low_weight_stream_amortizes_not_starves():
    # a stream far below one physical batch per step must still be seen, amortized
    lengths = [1] * 1000
    streams = [MixStream(0, 900, 0.97, "big"), MixStream(900, 100, 0.03, "tiny")]
    epoch_phys = _run(lengths, streams, N_EPOCHS)
    from collections import Counter

    usage = Counter()
    for phys in epoch_phys:
        for b in phys:
            for idx in b:
                usage[idx] += 1
    assert all(usage[i] >= 1 for i in range(900, 1000)), "tiny stream starved"
    tiny_total = sum(usage[i] for i in range(900, 1000))
    big_total = sum(usage[i] for i in range(0, 900))
    frac = tiny_total / (tiny_total + big_total)
    assert 0.015 <= frac <= 0.05, frac  # ~3% amortized


def test_co_residency_is_not_static_across_epochs():
    lengths, streams = _streams()
    epoch_phys = _run(lengths, streams, N_EPOCHS)

    def partners_of(anchor_idx, phys):
        for b in phys:
            if anchor_idx in b:
                return frozenset(b) - {anchor_idx}
        return frozenset()

    # a fixed anchor item should not have identical batch-mates every epoch
    partner_sets = {partners_of(0, phys) for phys in epoch_phys}
    assert len(partner_sets) > 1, "anchor item 0 had identical co-residents every epoch (not random)"
    # and successive epochs are not byte-identical schedules
    assert epoch_phys[0] != epoch_phys[1]

    from collections import Counter, defaultdict

    partners_by_item = defaultdict(list)
    for phys in epoch_phys:
        for lo in range(0, len(phys), GAS):
            logical_indices = [idx for batch in phys[lo : lo + GAS] for idx in batch]
            for idx in logical_indices:
                partners_by_item[idx].append(frozenset(logical_indices) - {idx})
    static_items = 0
    max_partner_frac = 0.0
    for partner_history in partners_by_item.values():
        if len(partner_history) <= 1:
            continue
        counts = Counter(partner_history)
        max_partner_frac = max(max_partner_frac, max(counts.values()) / len(partner_history))
        if len(counts) == 1 and len(partner_history) > 1:
            static_items += 1
    assert static_items <= BEST["static_co_resident_items_max"], static_items
    assert max_partner_frac <= BEST["co_resident_partner_frac_max"], max_partner_frac
