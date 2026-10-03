#!/usr/bin/env python
"""Stable, named seed forks shared by PII data and training stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEME = "pii-seed-tree-v1"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("pii_reproducibility.yaml")


@dataclass(frozen=True)
class SeedTree:
    root_seed: int
    scheme: str = SCHEME

    def fork(self, *path: object) -> int:
        if not path or any(not str(part) for part in path):
            raise ValueError("seed fork path must contain non-empty components")
        payload = "\0".join((self.scheme, str(self.root_seed), *(str(part) for part in path)))
        # NumPy/Transformers seeds must fit uint32. Four digest bytes retain
        # stable, independent branches without relying on process hash state.
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")

    def describe(self, **branches: tuple[object, ...]) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "root_seed": self.root_seed,
            "forks": {name: self.fork(*path) for name, path in sorted(branches.items())},
        }


def load_seed_tree(path: Path = DEFAULT_CONFIG_PATH, *, root_seed: int | None = None) -> SeedTree:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{path}: reproducibility config schema_version must be 1")
    scheme = config.get("scheme")
    if scheme != SCHEME:
        raise ValueError(f"{path}: unsupported seed-tree scheme {scheme!r}")
    configured_root = config.get("root_seed") if root_seed is None else root_seed
    if not isinstance(configured_root, int) or configured_root < 0:
        raise ValueError(f"{path}: root_seed must be a nonnegative integer")
    return SeedTree(configured_root, scheme)
