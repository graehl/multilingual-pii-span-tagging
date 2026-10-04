"""Ordered bounded input preparation, errors, and consumed-state replay."""

import random
import time

import pytest

from trainlib_input import BufferedIterator


def test_training_buffer_accounts_for_lattice_metadata():
    from types import SimpleNamespace

    import torch

    from trainlib import buffer_input_batches

    dataset = SimpleNamespace(
        input_state=lambda: None, commit_input_state=lambda _: None, restore_input_state=lambda _: None
    )
    batch = {
        "audio": torch.zeros(4),
        "reference_counts": [2],
        "reference_keys": ["identity"],
        "reference_metadata": [
            {
                "text": "عام 2008",
                "paths": ["ألفين وثمانية", "ألفان وثمانية"],
                "audio_sha256": None,
                "speed": 1.0,
            }
        ],
    }
    config = {"max_batches": 2, "max_bytes": 8192, "max_batch_bytes": 4096}
    with buffer_input_batches(iter([batch]), dataset, config) as stream:
        assert next(stream) is batch
        assert list(stream) == []
    assert not stream.worker.is_alive()
    batch["reference_metadata"][0]["text"] = "x" * 8192
    with buffer_input_batches(iter([batch]), dataset, config) as stream:
        with pytest.raises(ValueError, match="max_batch_bytes"):
            next(stream)
    with buffer_input_batches(iter([object()]), dataset, config) as stream:
        with pytest.raises(TypeError, match="Unsupported buffered batch payload"):
            next(stream)


def test_prefetch_preserves_order_and_consumed_random_state():
    rng = random.Random(7)
    committed = [rng.getstate()]
    expected = random.Random(7)

    def values():
        for _ in range(20):
            yield rng.random()

    stream = BufferedIterator(
        iter(values()),
        max_batches=4,
        max_bytes=32,
        max_batch_bytes=8,
        size=lambda _: 8,
        snapshot=rng.getstate,
        commit=lambda state: committed.__setitem__(0, state),
        restore=rng.setstate,
    )
    try:
        actual = [next(stream) for _ in range(3)]
        time.sleep(0.02)
        assert actual == [expected.random() for _ in range(3)]
        assert committed[0] == expected.getstate()
        assert stream.stats()["peak_reserved_bytes"] <= 32
    finally:
        stream.close()
    assert not stream.worker.is_alive()
    assert rng.getstate() == expected.getstate()


def test_prefetch_propagates_failure_and_rejects_oversized_batch():
    def failed():
        yield b"a"
        raise ValueError("failed transform")

    with BufferedIterator(failed(), max_batches=2, max_bytes=8, max_batch_bytes=4, size=len) as stream:
        assert next(stream) == b"a"
        with pytest.raises(ValueError, match="failed transform"):
            next(stream)
    with BufferedIterator(
        iter([b"too large"]), max_batches=2, max_bytes=8, max_batch_bytes=4, size=len
    ) as stream:
        with pytest.raises(ValueError, match="max_batch_bytes"):
            next(stream)


def test_latency_summaries_are_cumulative_bounded_and_finish_once():
    summaries = []
    with BufferedIterator(
        iter(range(9)),
        max_batches=3,
        max_bytes=24,
        max_batch_bytes=8,
        size=lambda _: 8,
        window=4,
        stats_sink=summaries.append,
    ) as stream:
        assert list(stream) == list(range(9))
    stream.close()
    assert [s["event"] for s in summaries] == ["periodic", "periodic", "closed"]
    last = summaries[-1]
    assert last["histograms"]["producer"]["count"] == 9
    assert last["histograms"]["consumer_wait"]["count"] == 8
    assert last["histograms"]["startup_wait"]["count"] == 1
    assert sum(last["depth_counts"].values()) == 9
    for histogram in last["histograms"].values():
        assert len(histogram["counts"]) == len(histogram["upper_bounds_us"])
        assert sum(histogram["counts"]) == histogram["count"]


def test_tail_latency_increases_depth_without_exceeding_capacity():
    def slow_source():
        for index in range(12):
            time.sleep(0.005)
            yield index

    with BufferedIterator(
        slow_source(), max_batches=5, max_bytes=40, max_batch_bytes=8, size=lambda _: 8, window=4
    ) as stream:
        assert list(stream) == list(range(12))
    summary = stream.stats()
    assert max(map(int, summary["depth_counts"])) > 2
    assert summary["peak_reserved_bytes"] <= 40
    assert summary["starved_batches"] > 0
    assert summary["producer_slower_than_consumer"]
