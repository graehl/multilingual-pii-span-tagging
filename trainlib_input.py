"""Ordered CPU preparation with bounded payload reservations and latency lookahead."""

import bisect
import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator


class LatencyHistogram:
    """Constant-space cumulative microsecond buckets, directly unnested in DuckDB."""

    bounds_us = [2**power for power in range(30)]

    def __init__(self):
        self.counts = [0] * (len(self.bounds_us) + 1)
        self.seconds = 0.0
        self.maximum = 0.0

    def add(self, seconds: float) -> None:
        self.counts[bisect.bisect_left(self.bounds_us, seconds * 1e6)] += 1
        self.seconds += seconds
        self.maximum = max(self.maximum, seconds)

    def summary(self) -> dict:
        return {
            "upper_bounds_us": [*self.bounds_us, None],
            "counts": list(self.counts),
            "count": sum(self.counts),
            "sum_seconds": self.seconds,
            "max_seconds": self.maximum,
        }


def validate_input_pipeline(config: dict) -> dict:
    allowed = {"max_batches", "max_bytes", "max_batch_bytes", "quantile", "window"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown input_pipeline fields or non-object")
    result = {"quantile": 0.99, "window": 128, **config}
    for key in ("max_batches", "max_bytes", "max_batch_bytes", "window"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(f"input_pipeline {key} must be a positive integer")
    if result["max_batch_bytes"] > result["max_bytes"]:
        raise ValueError("input_pipeline max_batch_bytes exceeds max_bytes")
    if not isinstance(result["quantile"], (float, int)) or not 0 < result["quantile"] <= 1:
        raise ValueError("input_pipeline quantile must be in (0,1]")
    return result


class BufferedIterator:
    """One producer; scheduling order is independent of preparation latency.

    The task owns finite I/O deadlines and must close the iterator on early stop.
    max_batch_bytes reserves space before preparation, including the in-flight
    payload. Temporary transform workspace and consumer-owned batches are extra.
    Private producer RNG snapshots commit only as batches reach the consumer.
    """

    def __init__(
        self,
        source: Iterator,
        *,
        max_batches: int,
        max_bytes: int,
        max_batch_bytes: int,
        size: Callable,
        quantile: float = 0.99,
        window: int = 128,
        snapshot: Callable | None = None,
        commit: Callable | None = None,
        restore: Callable | None = None,
        stats_sink: Callable | None = None,
    ):
        if min(max_batches, max_bytes, max_batch_bytes, window) < 1 or max_batch_bytes > max_bytes:
            raise ValueError("Positive buffer bounds and max_batch_bytes <= max_bytes required")
        if not 0 < quantile <= 1:
            raise ValueError("Buffer quantile must be in (0,1]")
        if any(hook is not None for hook in (snapshot, commit, restore)) and not all(
            hook is not None for hook in (snapshot, commit, restore)
        ):
            raise ValueError("Producer state requires snapshot, commit and restore together")
        self.source, self.size = source, size
        self.snapshot, self.commit, self.restore = snapshot, commit, restore
        self.consumed_state = snapshot() if snapshot else None
        self.max_batches, self.max_bytes, self.max_batch_bytes = max_batches, max_bytes, max_batch_bytes
        self.quantile, self.window = quantile, window
        self.stats_sink = stats_sink
        self.histograms = {
            key: LatencyHistogram()
            for key in ("producer", "consumer_wait", "consumer_interval", "startup_wait")
        }
        self.depth_counts = {}
        self.starved_batches = 0
        self.closed = False
        self.depth = min(2, max_batches, max_bytes // max_batch_bytes)
        self.latencies, self.consumptions = deque(maxlen=window), deque(maxlen=window)
        self.ready = deque()
        self.condition = threading.Condition()
        self.stopped = self.finished = False
        self.error = None
        self.error_delivered = False
        self.reserved = self.peak_reserved = self.produced = self.consumed = 0
        self.wait_seconds = self.producer_seconds = 0.0
        self.last_return = None
        self.worker = threading.Thread(target=self._produce, name="trainlib-input")
        self.worker.start()

    def _produce(self) -> None:
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: (
                            self.stopped
                            or (
                                len(self.ready) < self.depth
                                and self.reserved + self.max_batch_bytes <= self.max_bytes
                            )
                        )
                    )
                    if self.stopped:
                        return
                    self.reserved += self.max_batch_bytes
                    self.peak_reserved = max(self.peak_reserved, self.reserved)
                started = time.perf_counter()
                try:
                    value = next(self.source)
                except StopIteration:
                    with self.condition:
                        self.reserved -= self.max_batch_bytes
                    return
                elapsed = time.perf_counter() - started
                amount = self.size(value)
                if amount > self.max_batch_bytes:
                    raise ValueError(
                        f"Prepared batch {amount} exceeds max_batch_bytes={self.max_batch_bytes}"
                    )
                state = self.snapshot() if self.snapshot else None
                with self.condition:
                    self.reserved += amount - self.max_batch_bytes
                    self.ready.append((value, amount, state))
                    self.latencies.append(elapsed)
                    self.histograms["producer"].add(elapsed)
                    self.producer_seconds += elapsed
                    self.produced += 1
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        before = time.perf_counter()
        with self.condition:
            if self.stopped:
                raise StopIteration
            if self.last_return is not None:
                self.consumptions.append(before - self.last_return)
                self.histograms["consumer_interval"].add(before - self.last_return)
            if len(self.consumptions) >= min(4, self.window) and self.latencies:
                tail = sorted(self.latencies)[math.ceil(self.quantile * len(self.latencies)) - 1]
                typical = statistics.median(self.consumptions)
                self.depth = min(
                    self.max_batches,
                    self.max_bytes // self.max_batch_bytes,
                    max(1, math.ceil(tail / max(typical, 1e-9)) + 1),
                )
            self.condition.notify_all()
            starved = not self.ready
            self.condition.wait_for(lambda: self.ready or self.finished or self.stopped)
            if not self.ready:
                if self.error:
                    self.error_delivered = True
                    raise self.error
                raise StopIteration
            value, amount, state = self.ready.popleft()
            self.reserved -= amount
            self.consumed_state = state
            if self.commit:
                self.commit(state)
            self.consumed += 1
            now = time.perf_counter()
            if self.last_return is not None:
                self.wait_seconds += now - before
                self.histograms["consumer_wait"].add(now - before)
                self.starved_batches += int(starved)
            else:
                self.histograms["startup_wait"].add(now - before)
            self.depth_counts[str(self.depth)] = self.depth_counts.get(str(self.depth), 0) + 1
            self.last_return = now
            self.condition.notify_all()
        if self.stats_sink and self.consumed % self.window == 0:
            self.stats_sink({"event": "periodic", **self.stats()})
        return value

    def stats(self) -> dict:
        with self.condition:
            tail = (
                sorted(self.latencies)[math.ceil(self.quantile * len(self.latencies)) - 1]
                if self.latencies
                else 0
            )
            typical = statistics.median(self.consumptions) if self.consumptions else 0
            return {
                "produced": self.produced,
                "producer_error": str(self.error) if self.error else None,
                "consumed": self.consumed,
                "depth": self.depth,
                "histograms": {key: value.summary() for key, value in self.histograms.items()},
                "depth_counts": dict(self.depth_counts),
                "starved_batches": self.starved_batches,
                "discarded_prepared_batches": self.produced - self.consumed if self.closed else 0,
                "max_batches": self.max_batches,
                "max_bytes": self.max_bytes,
                "max_batch_bytes": self.max_batch_bytes,
                "peak_reserved_bytes": self.peak_reserved,
                "quantile": self.quantile,
                "window": self.window,
                "tail_producer_seconds": tail,
                "typical_consumer_seconds": typical,
                "wait_seconds": self.wait_seconds,
                "producer_batches_per_second": self.produced / self.producer_seconds
                if self.producer_seconds
                else 0,
                "producer_slower_than_consumer": bool(
                    typical and self.produced and self.producer_seconds / self.produced > typical
                ),
            }

    def close(self) -> None:
        if self.closed:
            return
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.worker.join()
        if self.restore:
            self.restore(self.consumed_state)
        self.ready.clear()
        close = getattr(self.source, "close", None)
        if close:
            close()
        self.closed = True
        if self.stats_sink:
            self.stats_sink({"event": "closed", **self.stats()})
        if self.error and not self.error_delivered:
            self.error_delivered = True
            raise self.error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
