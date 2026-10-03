"""Load the shared GPU-run metrics helpers from ~/agents, or degrade to no-ops.

Deliberately not pinned by hash, unlike `agents_run_quality`. That module scores
artifacts, so silent drift would change results and an exact revision is the point. These
helpers only observe: they never touch what a run computes, and pinning them would mean
every metrics improvement needed a coordinated pin bump in every consumer. A missing or
older shared checkout therefore costs metrics, never correctness.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

CHECKOUT_ENV = "AGENTS_RUN_QUALITY_CHECKOUT"
_MODULE_NAME = "_aip_draft_agents_run_metrics"


def _candidates() -> list[Path]:
    configured = os.environ.get(CHECKOUT_ENV)
    roots = [Path(configured).expanduser()] if configured else []
    return roots + [Path.home() / "agents"]


def _load() -> Any | None:
    if os.environ.get("AGENTS_RUN_METRICS_ENABLED") == "0":
        return None
    if module := sys.modules.get(_MODULE_NAME):
        return module
    for root in _candidates():
        source = root / "run_quality" / "run_metrics.py"
        if not source.is_file():
            continue
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    return None


_shared = _load()


class _InertMetrics:
    """Stands in when the shared checkout is absent, so callers need no conditionals."""

    def __init__(self, phase: str = ""):
        self.phase = phase

    def start(self) -> _InertMetrics:
        return self

    def finish(self, **_facts: Any) -> None:
        return None


RunMetrics = getattr(_shared, "RunMetrics", _InertMetrics)
platform_facts = getattr(_shared, "platform_facts", dict)
peak_memory = getattr(_shared, "peak_memory", dict)
reset_peak_memory = getattr(_shared, "reset_peak_memory", lambda: None)
publish = getattr(_shared, "publish", lambda *_args, **_kwargs: None)
available = _shared is not None

__all__ = [
    "RunMetrics",
    "available",
    "peak_memory",
    "platform_facts",
    "publish",
    "reset_peak_memory",
]
