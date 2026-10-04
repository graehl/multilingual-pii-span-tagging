"""Routine run metrics: published when tracked, inert and harmless when not."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import agents_run_metrics as metrics  # noqa: E402


def test_platform_facts_name_the_stack():
    facts = metrics.platform_facts()
    # A throughput number is meaningless without the stack it was measured on.
    assert facts["python"]
    assert "kernel" in facts


def test_publish_is_a_no_op_outside_a_tracked_run(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTCTL_RUN_DIR", raising=False)
    assert metrics.publish({"peak_vram_reserved_gib": 1.0}) is None


def test_publish_merges_rather_than_overwrites(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTCTL_RUN_DIR", str(tmp_path))
    metrics.publish({"a": 1}, prefix="train_")
    metrics.publish({"b": 2}, prefix="decode_")
    written = json.loads((tmp_path / "propagate.json").read_text(encoding="utf-8"))
    # Several phases of one run contribute without coordinating.
    assert written == {"train_a": 1, "decode_b": 2}


def test_run_metrics_reports_a_rate_and_wall_time(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTCTL_RUN_DIR", str(tmp_path))
    metrics.RunMetrics("train").start().finish(steps=10, samples=640, batch_shape="8x8")
    written = json.loads((tmp_path / "propagate.json").read_text(encoding="utf-8"))
    assert written["train_steps"] == 10
    assert written["train_samples_per_second"] > 0
    assert written["train_wall_seconds"] >= 0
    assert written["train_batch_shape"] == "8x8"
    assert written["platform_python"]


def test_none_valued_facts_are_dropped(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTCTL_RUN_DIR", str(tmp_path))
    metrics.publish({"present": 1, "absent": None})
    written = json.loads((tmp_path / "propagate.json").read_text(encoding="utf-8"))
    assert "absent" not in written


def test_decode_bracket_publishes(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTCTL_RUN_DIR", str(tmp_path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from decodelib import DecodeMetrics

    with DecodeMetrics() as run:
        run.finish(rows=4, tokens=128, batch_size=2)
    written = json.loads((tmp_path / "propagate.json").read_text(encoding="utf-8"))
    assert written["decode_tokens"] == 128
    assert written["decode_tokens_per_second"] > 0
