#!/usr/bin/env python3
"""Annotation-convention classes for source-conditioning tokens.

A conditioning token is meant to absorb the labelling idiosyncrasy of the process that
produced a row, so the shared weights are not forced to average incompatible conventions.
The raw ``src`` field is too fine for that: the pool carries 44 values, the largest is 44%
of rows, and sixteen are under 500 rows each, so most per-source tokens would be
undertrained noise. Classes group by pipeline, which is the axis the token is for.

``unknown`` is index 0 and is never assigned by a rule. It is what production feeds and
what training-time dropout substitutes, so that condition is trained rather than
extrapolated.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SPEC = Path(__file__).with_name("pii_source_classes_v1.json")


class SourceClasses:
    """An ordered class vocabulary plus the rules that map a raw ``src`` onto it."""

    def __init__(self, spec: dict):
        if spec.get("schema") != "pii-source-classes/v1":
            raise ValueError(f"unsupported source-class schema {spec.get('schema')!r}")
        self.unknown = spec["unknown_class"]
        self.fallback = spec["fallback_class"]
        self.rules = []
        for rule in spec["rules"]:
            kinds = [key for key in ("equals", "prefix", "contains") if key in rule]
            if len(kinds) != 1:
                raise ValueError(f"rule for {rule.get('class')!r} needs exactly one match key")
            self.rules.append((rule["class"], kinds[0], rule[kinds[0]]))
        names = [self.unknown] + [name for name, _, _ in self.rules] + [self.fallback]
        seen = set()
        self.names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                self.names.append(name)
        self.index = {name: position for position, name in enumerate(self.names)}
        if self.index[self.unknown] != 0:
            raise ValueError("the unknown class must be index 0")

    @classmethod
    def load(cls, path: str | Path | None = None) -> SourceClasses:
        return cls(json.loads(Path(path or DEFAULT_SPEC).read_text(encoding="utf-8")))

    def class_of(self, src: str | None) -> str:
        """First matching rule wins, so rule order in the spec is the precedence."""
        if not src:
            return self.fallback
        for name, kind, value in self.rules:
            if (
                (kind == "equals" and src == value)
                or (kind == "prefix" and src.startswith(value))
                or (kind == "contains" and value in src)
            ):
                return name
        return self.fallback

    def id_of(self, src: str | None) -> int:
        return self.index[self.class_of(src)]

    @property
    def unknown_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.names)


if __name__ == "__main__":
    import collections
    import glob
    import sys

    classes = SourceClasses.load()
    print(f"{len(classes)} classes: {', '.join(classes.names)}")
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    if not pattern:
        raise SystemExit(0)
    rows = collections.Counter()
    members = collections.defaultdict(set)
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            src = json.loads(line).get("src")
            name = classes.class_of(src)
            rows[name] += 1
            members[name].add(src)
    total = sum(rows.values())
    print(f"\n{total} rows\n")
    for name in classes.names:
        count = rows.get(name, 0)
        print(
            f"{count:8d}  {100 * count / total if total else 0:5.2f}%  {name:14s}  {len(members[name])} sources"
        )
