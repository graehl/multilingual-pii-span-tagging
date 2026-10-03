#!/usr/bin/env python
"""Resolve how much each intake method's O labels should be trusted.

A span-annotated row asserts that everything outside its spans is not an entity.
That claim is only as good as the labeller's recall, so a high-precision
low-recall source contributes false negatives at every O position. This maps a
row's source to a relative weight for its O tokens, which the training loss
applies per row — the weight has to vary within a physical batch, because one
batch mixes intake methods.

All values live in ``scripts/pii_o_weight.yaml``. This module holds none: a
source matching no rule gets the file's ``default``, and the resolved weight is
reported so a corpus can record what it trained under.

Distinct from ``supervision: annotated_spans_only``, which masks non-span tokens
outright. Weighting is the continuous version for sources better than "unknown"
and worse than gold; masked rows never reach this table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

SCHEMA = "pii-o-weight-v1"
DEFAULT_CONFIG = Path(__file__).with_name("pii_o_weight.yaml")


class OWeightTable:
    """Source-pattern to O-label trust, loaded from one config."""

    def __init__(self, config: dict):
        if config.get("schema") != SCHEMA:
            raise ValueError(f"expected schema {SCHEMA}, got {config.get('schema')!r}")
        self.scaling_factor = float(config.get("scaling_factor", 1.0))
        if not 0.0 <= self.scaling_factor:
            raise ValueError("scaling_factor must be nonnegative")
        self.default = float(config.get("default", 1.0))
        raw = config.get("sources") or {}
        # longest pattern first, so a specific source beats a generic one
        # regardless of how the file happens to be ordered
        self.patterns = sorted(
            ((str(name), float(weight)) for name, weight in raw.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for name, weight in self.patterns:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"source {name!r}: O weight must be in [0, 1], got {weight}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "OWeightTable":
        return cls(yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8")))

    def match(self, source: str | None) -> tuple[str, float]:
        """Return the matched pattern name and its unscaled weight."""
        text = (source or "").casefold()
        for name, weight in self.patterns:
            if name.casefold() in text:
                return name, weight
        return "default", self.default

    def weight_for(self, source: str | None) -> float:
        """The O weight this row's tokens should carry, scaling applied."""
        _name, weight = self.match(source)
        return max(0.0, weight * self.scaling_factor)

    def explain(self, source: str | None) -> dict:
        name, weight = self.match(source)
        return {
            "source": source,
            "matched": name,
            "base_weight": weight,
            "scaling_factor": self.scaling_factor,
            "effective_weight": max(0.0, weight * self.scaling_factor),
        }


def _self_test() -> int:
    table = OWeightTable.load()
    cases = [
        # real src values from the union corpus
        ("de-transport-semantic-policy-v2-admitted", 1.0),
        ("ar-transport-semantic-policy-v2-a", 1.0),
        ("gold", 1.0),
        # a source nobody listed falls back rather than erroring
        ("nemotron-full", table.default),
        ("some-unlisted-source", table.default),
        (None, table.default),
    ]
    bad = 0
    for source, want in cases:
        got = table.weight_for(source)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'ok ' if ok else 'FAIL'} {str(source):42s} -> {got:.2f} ({table.match(source)[0]})")
    # A deliberately precision-shaped source trusts O less than gold. Teacher
    # pools marked annotated_spans_only bypass this table altogether.
    consensus = table.weight_for("intersection_consensus")
    trusted = table.weight_for("gold")
    ordered = consensus < trusted
    print(f"{'ok ' if ordered else 'FAIL'} ordering: intersection {consensus} < gold {trusted}")
    bad += not ordered
    # scaling applies to everything, including the default
    scaled = OWeightTable({**yaml.safe_load(DEFAULT_CONFIG.read_text()), "scaling_factor": 0.5})
    halved = abs(scaled.weight_for("gold") - 0.5 * table.weight_for("gold")) < 1e-9
    print(f"{'ok ' if halved else 'FAIL'} scaling_factor 0.5 halves gold -> {scaled.weight_for('gold')}")
    bad += not halved
    print("PASS" if not bad else f"{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_self_test())
