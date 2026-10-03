"""Lightweight run configuration inheritance shared by drivers and trainers."""

import copy
import json
from pathlib import Path


def merge_run_config(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge_run_config(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def read_run_config_basis(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in chain:
        raise ValueError(f"Run config basis cycle: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be an object: {path}")
    bases = data.pop("basis", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or any(not isinstance(base, str) for base in bases):
        raise ValueError("Run config basis must be a filename or list of filenames")
    result = {}
    for base in bases:
        result = merge_run_config(result, read_run_config_basis(path.parent / base, (*chain, path)))
    return merge_run_config(result, data)
