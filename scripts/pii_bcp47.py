#!/usr/bin/env python3
"""Small shared helpers for the project's BCP 47 language-tag subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BCP47_COMMON_RE = re.compile(
    r"^(?P<language>[A-Za-z]{2,8})"
    r"(?:-(?P<script>[A-Za-z]{4}))?"
    r"(?:-(?P<region>[A-Za-z]{2}|[0-9]{3}))?$"
)


def canonical_bcp47_tag(value: str) -> str:
    """Canonicalize the language[-Script][-REGION] subset used by PII fixtures."""
    match = BCP47_COMMON_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported BCP 47 language tag: {value!r}")
    parts = [match.group("language").lower()]
    if script := match.group("script"):
        parts.append(script.title())
    if region := match.group("region"):
        parts.append(region.upper() if region.isalpha() else region)
    return "-".join(parts)


def bcp47_has_region(value: str) -> bool:
    match = BCP47_COMMON_RE.fullmatch(canonical_bcp47_tag(value))
    assert match is not None
    return match.group("region") is not None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(text)


def _load_tagged_jsonl(path: str | Path) -> tuple[list[str], list[str], Counter[str]]:
    tags = []
    identities = []
    source_fields: Counter[str] = Counter()
    seen = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            identity = row.get("id")
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{path}:{line_number}: missing nonempty id")
            if identity in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {identity!r}")
            seen.add(identity)
            raw_tag = row.get("bcp47")
            source_field = "bcp47"
            if raw_tag is None:
                raw_tag = row.get("lang")
                source_field = "lang"
            if not isinstance(raw_tag, str):
                raise ValueError(f"{path}:{line_number}: missing bcp47 and lang")
            tags.append(canonical_bcp47_tag(raw_tag))
            identities.append(identity)
            source_fields[source_field] += 1
    return tags, identities, source_fields


def materialize_jsonl_sidecar(
    input_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Write one canonical BCP 47 tag per JSONL row plus a provenance receipt."""
    tags, identities, source_fields = _load_tagged_jsonl(input_path)
    identity_digest = hashlib.sha256()
    for identity in identities:
        identity_digest.update(identity.encode("utf-8"))
        identity_digest.update(b"\n")
    _write_new_text(output_path, "".join(f"{tag}\n" for tag in tags))
    receipt = {
        "schema": "pii-bcp47-sidecar",
        "version": 1,
        "status": "complete",
        "input": {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "rows": len(tags),
            "ordered_id_sha256_with_lf": identity_digest.hexdigest(),
        },
        "output": {
            "path": str(output_path),
            "sha256": _file_sha256(output_path),
            "rows": len(tags),
        },
        "source_fields": dict(sorted(source_fields.items())),
        "tag_counts": dict(sorted(Counter(tags).items())),
    }
    _write_new_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    receipt = materialize_jsonl_sidecar(args.input, args.output, args.receipt)
    print(
        f"BCP47-SIDECAR: rows={receipt['output']['rows']} tags={receipt['tag_counts']} output={args.output}"
    )


if __name__ == "__main__":
    main()
