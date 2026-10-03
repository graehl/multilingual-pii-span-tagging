"""Reject quarantined annotation products at the training-input boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "research/pii/frontier/evidence/local-llm-label-quarantine-v1.json"
)


def validate_training_inputs(paths: Iterable[Path], policy_path: Path = POLICY_PATH) -> None:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy["schema"] != "pii-local-llm-label-quarantine/v1" or policy["status"] != "active":
        raise ValueError(f"unsupported training quarantine policy: {policy_path}")
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        digest = hashlib.sha256()
        first_blocked = None
        with path.open("rb") as source:
            for line_number, line in enumerate(source, 1):
                digest.update(line)
                row = json.loads(line)
                reason = quarantine_reason(row, policy)
                if reason and first_blocked is None:
                    first_blocked = f"{path}:{line_number}: {reason}"
        file_reason = policy["blocked_file_sha256"].get(digest.hexdigest())
        if file_reason or first_blocked:
            detail = first_blocked or f"{path}: {file_reason}"
            raise ValueError(
                f"quarantined training supervision: {detail}; "
                f"repair or replace labels under {policy_path} before admission"
            )


def quarantine_reason(row: dict[str, Any], policy: dict[str, Any]) -> str | None:
    for field in policy["source_fields"]:
        value = row.get(field)
        if isinstance(value, str) and value.startswith(tuple(policy["blocked_source_prefixes"])):
            return f"{field}={value}"
    return None
