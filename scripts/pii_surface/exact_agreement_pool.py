"""Build a candidate surface pool from a base pool and exact model/teacher agreements."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from pii_surface_audit import iter_jsonl, normalize_value
    from pii_surface_pool import tags_for_span

    from pii_surface.agreement_mining import REPORT_SCHEMA, SCHEMA
except ModuleNotFoundError:  # Imported as scripts.pii_surface.exact_agreement_pool in tests.
    from scripts.pii_surface.agreement_mining import REPORT_SCHEMA, SCHEMA
    from scripts.pii_surface_audit import iter_jsonl, normalize_value
    from scripts.pii_surface_pool import tags_for_span

EXACT_TIER = "exact_same_label"
EXACT_STATUS = "agreement_candidate_quality_unreviewed"
OUTPUT_STATUS = "candidate_pool_not_admitted"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def split_for_source(source_text_sha256: str, holdout_modulus: int, holdout_fold: int) -> str:
    return "audit" if int(source_text_sha256[:16], 16) % holdout_modulus == holdout_fold else "train"


def validated_agreement_report(
    report_path: Path,
    agreement_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{report_path}: expected schema {REPORT_SCHEMA!r}")
    if report.get("status") != "candidate_mining_complete_quality_unreviewed":
        raise ValueError(f"{report_path}: agreement mining is not complete")
    exact_output = (report.get("outputs") or {}).get("exact_agreement")
    if not isinstance(exact_output, dict):
        raise ValueError(f"{report_path}: missing exact_agreement output receipt")
    expected_hash = exact_output.get("sha256")
    observed_hash = file_sha256(agreement_path)
    if expected_hash != observed_hash:
        raise ValueError(f"{agreement_path}: sha256 differs from exact_agreement receipt in {report_path}")
    return report


def validate_base_row(row: dict[str, Any], where: str) -> None:
    required_strings = ("pool_version", "entry_id", "split", "lang", "tag", "value")
    if any(not isinstance(row.get(field), str) or not row[field] for field in required_strings):
        raise ValueError(f"{where}: malformed base-pool string field")
    if row["split"] not in {"train", "audit"}:
        raise ValueError(f"{where}: unsupported split {row['split']!r}")
    count = row.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"{where}: count must be a positive integer")
    if row.get("sampling_count_receipt") is not None:
        raise ValueError(f"{where}: merge the observed pool before authentic-frequency reweighting")
    source_counts = row.get("source_counts")
    if not isinstance(source_counts, dict) or not source_counts:
        raise ValueError(f"{where}: source_counts must be a nonempty object")
    if any(
        not isinstance(source, str)
        or not source
        or not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or source_count <= 0
        for source, source_count in source_counts.items()
    ):
        raise ValueError(f"{where}: source_counts has an invalid entry")
    if sum(source_counts.values()) != count:
        raise ValueError(f"{where}: source_counts must sum to count")
    hashes = row.get("source_record_hashes")
    if not isinstance(hashes, list) or not hashes or any(not valid_sha256(item) for item in hashes):
        raise ValueError(f"{where}: source_record_hashes must contain source SHA-256 values")


def validate_exact_candidate(row: dict[str, Any], where: str) -> None:
    if row.get("schema") != SCHEMA:
        raise ValueError(f"{where}: expected schema {SCHEMA!r}")
    if row.get("tier") != EXACT_TIER or row.get("status") != EXACT_STATUS:
        raise ValueError(f"{where}: only unreviewed exact same-label agreements are accepted")
    for field in ("candidate_id", "language", "tag", "surface", "context"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"{where}: {field} must be a nonempty string")
    if not valid_sha256(row.get("source_text_sha256")):
        raise ValueError(f"{where}: source_text_sha256 must be 64 lowercase hex characters")
    start, end, context_start = row.get("start"), row.get("end"), row.get("context_start")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (start, end, context_start)):
        raise ValueError(f"{where}: start/end/context_start must be integers")
    relative_start = start - context_start
    relative_end = end - context_start
    context = row["context"]
    if not 0 <= relative_start < relative_end <= len(context):
        raise ValueError(f"{where}: candidate bounds fall outside context")
    if context[relative_start:relative_end] != row["surface"]:
        raise ValueError(f"{where}: surface does not match the declared context bounds")
    if not tags_for_span(row["language"], row["tag"].lower(), row["surface"]):
        raise ValueError(f"{where}: unsupported surface tag {row['tag']!r}")


def add_aggregate(
    aggregates: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    split: str,
    language: str,
    tag: str,
    value: str,
    count: int,
    source_counts: dict[str, int],
    source_record_hashes: Iterable[str],
) -> None:
    key = split, language, tag, value
    aggregate = aggregates.setdefault(
        key,
        {"count": 0, "source_counts": Counter(), "source_record_hashes": set()},
    )
    aggregate["count"] += count
    aggregate["source_counts"].update(source_counts)
    aggregate["source_record_hashes"].update(source_record_hashes)


def build_exact_agreement_pool(
    *,
    base_pool_path: Path,
    base_report_path: Path,
    agreement_path: Path,
    agreement_report_path: Path,
    output_pool_version: str,
    agreement_source_name: str = "exact-v11-gemma31b",
    holdout_modulus: int = 10,
    holdout_fold: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not output_pool_version:
        raise ValueError("output_pool_version must be nonempty")
    if not agreement_source_name:
        raise ValueError("agreement_source_name must be nonempty")
    if holdout_modulus < 2:
        raise ValueError("holdout_modulus must be at least 2")
    if not 0 <= holdout_fold < holdout_modulus:
        raise ValueError("holdout_fold must be in [0, holdout_modulus)")

    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if base_report.get("pool_version") is None:
        raise ValueError(f"{base_report_path}: missing pool_version")
    validated_agreement_report(agreement_report_path, agreement_path)

    aggregates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    base_versions = set()
    base_rows = 0
    for line_number, row in enumerate(iter_jsonl(base_pool_path), 1):
        validate_base_row(row, f"{base_pool_path}:{line_number}")
        base_versions.add(row["pool_version"])
        base_rows += 1
        add_aggregate(
            aggregates,
            split=row["split"],
            language=row["lang"],
            tag=row["tag"],
            value=row["value"],
            count=row["count"],
            source_counts=row["source_counts"],
            source_record_hashes=row["source_record_hashes"],
        )
    if len(base_versions) != 1:
        raise ValueError(f"{base_pool_path}: expected exactly one pool version")
    base_version = base_versions.pop()
    if base_report["pool_version"] != base_version:
        raise ValueError(f"{base_report_path}: pool_version differs from {base_pool_path}")

    candidate_ids = set()
    agreement_rows = 0
    agreement_documents = set()
    agreement_cells = Counter()
    agreement_splits = Counter()
    for line_number, row in enumerate(iter_jsonl(agreement_path), 1):
        validate_exact_candidate(row, f"{agreement_path}:{line_number}")
        candidate_id = row["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"{agreement_path}:{line_number}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        agreement_rows += 1
        source_hash = row["source_text_sha256"]
        agreement_documents.add(source_hash)
        split = split_for_source(source_hash, holdout_modulus, holdout_fold)
        agreement_splits[split] += 1
        for tag, value in tags_for_span(row["language"], row["tag"].lower(), row["surface"]):
            agreement_cells[(split, row["language"], tag)] += 1
            add_aggregate(
                aggregates,
                split=split,
                language=row["language"],
                tag=tag,
                value=value,
                count=1,
                source_counts={agreement_source_name: 1},
                source_record_hashes=(source_hash,),
            )

    entries = []
    for (split, language, tag, value), aggregate in sorted(aggregates.items()):
        entries.append(
            {
                "pool_version": output_pool_version,
                "pool_status": OUTPUT_STATUS,
                "entry_id": stable_hash(output_pool_version, split, language, tag, value)[:20],
                "split": split,
                "lang": language,
                "tag": tag,
                "value": value,
                "normalized": normalize_value(value),
                "count": aggregate["count"],
                "source_counts": dict(sorted(aggregate["source_counts"].items())),
                "source_record_hashes": sorted(aggregate["source_record_hashes"]),
            }
        )

    groups = Counter((row["split"], row["lang"], row["tag"]) for row in entries)
    observed = Counter()
    for row in entries:
        observed[(row["split"], row["lang"], row["tag"])] += row["count"]
    report = {
        "schema_version": 1,
        "status": OUTPUT_STATUS,
        "pool_version": output_pool_version,
        "contract": {
            "admission": "candidate only; holistic language/tag review is required before training admission",
            "agreement_tier": "identical half-open span and canonical fine label from v11 and Gemma-4-31B",
            "partition": "source_text_sha256 is assigned to train/audit before surface aggregation",
            "frequency": "each accepted source occurrence contributes one count; base observed counts are retained",
        },
        "configuration": {
            "agreement_source_name": agreement_source_name,
            "holdout_modulus": holdout_modulus,
            "holdout_fold": holdout_fold,
        },
        "inputs": {
            "base_pool": {
                "path": str(base_pool_path),
                "sha256": file_sha256(base_pool_path),
                "pool_version": base_version,
                "rows": base_rows,
                "report": {
                    "path": str(base_report_path),
                    "sha256": file_sha256(base_report_path),
                },
            },
            "exact_agreement": {
                "path": str(agreement_path),
                "sha256": file_sha256(agreement_path),
                "rows": agreement_rows,
                "source_documents": len(agreement_documents),
                "report": {
                    "path": str(agreement_report_path),
                    "sha256": file_sha256(agreement_report_path),
                },
            },
        },
        "coverage": {
            "train_languages": sorted({row["lang"] for row in entries if row["split"] == "train"}),
            "groups": [
                {
                    "split": split,
                    "lang": language,
                    "tag": tag,
                    "distinct_values": groups[(split, language, tag)],
                    "observed_count": observed[(split, language, tag)],
                    "agreement_occurrences": agreement_cells[(split, language, tag)],
                }
                for split, language, tag in sorted(groups)
            ],
        },
        "agreement_rows_by_split": dict(sorted(agreement_splits.items())),
        "entries": len(entries),
    }
    return entries, report
