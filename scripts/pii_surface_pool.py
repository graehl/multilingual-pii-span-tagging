#!/usr/bin/env python
"""Build and sample provenance-preserving natural PII surface pools.

The pool stores observed frequency rather than deduplicating to a uniform
lexicon: common values are legitimate distribution mass. Source records are
partitioned before aggregation so an audit fold remains available even when a
surface occurs independently in both folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pii_surface_audit import (
        iter_jsonl,
        normalize_value,
        parse_span,
        resolve_input,
        row_language,
        script_profile,
    )
except ModuleNotFoundError:  # Imported as scripts.pii_surface_pool in tests.
    from scripts.pii_surface_audit import (
        iter_jsonl,
        normalize_value,
        parse_span,
        resolve_input,
        row_language,
        script_profile,
    )

POOL_VERSION = "natural-gold-surface-v3"
SURFACE_NORMALIZATIONS = ("legacy-nfkc-whitespace", "preserve")
CARRIER_ROLES = ("train-and-audit", "train-only")
NAME_COMPONENT_MODES = ("conservative-derived", "direct-only")
POOL_PROVENANCE = re.compile(r"natural-pool:([^:|]+):([^:|]+)")
WHITESPACE_COMPONENT_LANGUAGES = frozenset(
    {"cs", "de", "en", "es", "fr", "id", "it", "nl", "pl", "pt", "sv", "tr", "uk"}
)
VIETNAMESE_COMMON_FAMILY_NAMES = frozenset(
    {"Nguyễn", "Trần", "Lê", "Phạm", "Vũ", "Đặng", "Bùi", "Dương", "Mai", "Hoàng"}
)
CHINESE_COMPOUND_FAMILY_NAMES = (
    "欧阳",
    "司马",
    "上官",
    "诸葛",
    "东方",
    "皇甫",
    "尉迟",
    "公孙",
    "慕容",
    "长孙",
    "宇文",
    "司徒",
    "司空",
)
DIRECT_LABEL_TAGS = {
    "address": ("ADDRESS",),
    "city": ("CITY",),
    "clinician_name": ("CLINICIAN_NAME",),
    "country": ("COUNTRY",),
    "county": ("COUNTY",),
    "family_name": ("FAMILY_NAME",),
    "financial_org": ("FINANCIAL_ORG",),
    "given_name": ("GIVEN_NAME",),
    "healthcare_org": ("HEALTHCARE_ORG",),
    "job_area": ("JOB_AREA",),
    "job_title": ("JOB_TITLE",),
    "location": ("LOCATION",),
    "middle_name": ("MIDDLE_NAME",),
    "name_prefix": ("NAME_PREFIX",),
    "occupation": ("OCCUPATION",),
    "org_department": ("ORG_DEPARTMENT",),
    "organization": ("ORGANIZATION",),
    "patient_name": ("PATIENT_NAME",),
    "person_name": ("PERSON_NAME",),
    "postal_code": ("POSTAL_CODE",),
    "region": ("REGION",),
    "relative_name": ("RELATIVE_NAME",),
    "secondary_address": ("SECONDARY_ADDRESS",),
    "state": ("STATE",),
    "street_address": ("STREET_ADDRESS",),
}
SUPPORTED_POOL_TAGS = frozenset(tag for tags in DIRECT_LABEL_TAGS.values() for tag in tags)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(dataset: str, row_id: str, row: dict[str, Any]) -> str:
    """Stable source identity that does not depend on a local checkout path."""
    payload = {
        "dataset": dataset,
        "row_id": row_id,
        "lang": row.get("lang"),
        "text": row.get("text"),
        "spans": row.get("spans"),
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def source_manifest(paths: list[Path]) -> tuple[Path, dict[str, Any]] | None:
    """Find the one onboarding manifest governing every resolved input shard."""
    manifests = set()
    for path in paths:
        found = next(
            (parent / "manifest.json" for parent in path.parents if (parent / "manifest.json").is_file()),
            None,
        )
        if found is None:
            return None
        manifests.add(found.resolve())
    if len(manifests) != 1:
        return None
    path = manifests.pop()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("upstream"), dict):
        return None
    return path, raw


def load_source_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("upstream"), dict):
        raise ValueError(f"{path}: source manifest requires an upstream object")
    return path.resolve(), raw


def named_paths(specifications: list[str]) -> dict[str, Path]:
    resolved = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"expected NAME=PATH, got {specification!r}")
        name, raw_path = specification.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"expected nonempty NAME=PATH, got {specification!r}")
        if name in resolved:
            raise ValueError(f"duplicate named path: {name}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"named path is not a file: {path}")
        resolved[name] = path
    return resolved


@dataclass(frozen=True)
class SurfaceAdjudication:
    dataset: str
    row_id: str
    start: int
    end: int
    label: str
    value: str
    reason: str

    def span_key(self) -> tuple[str, str, int, int, str, str]:
        return (self.dataset, self.row_id, self.start, self.end, self.label, self.value)


def load_surface_adjudications(path: Path) -> tuple[set[SurfaceAdjudication], dict[str, Any]]:
    adjudications = set()
    adjudicated_spans = set()
    for line_number, row in enumerate(iter_jsonl(path), 1):
        required = {"dataset", "id", "start", "end", "label", "value", "reason", "action"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}")
        if row["action"] != "exclude":
            raise ValueError(f"{path}:{line_number}: action must be 'exclude'")
        start, end = row["start"], row["end"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"{path}:{line_number}: invalid half-open span")
        strings = {key: row[key] for key in ("dataset", "id", "label", "value", "reason")}
        if any(not isinstance(value, str) or not value for value in strings.values()):
            raise ValueError(f"{path}:{line_number}: string fields must be nonempty")
        adjudication = SurfaceAdjudication(
            dataset=row["dataset"],
            row_id=row["id"],
            start=start,
            end=end,
            label=row["label"].casefold(),
            value=row["value"],
            reason=row["reason"],
        )
        if adjudication.span_key() in adjudicated_spans:
            raise ValueError(f"{path}:{line_number}: duplicate adjudicated span")
        adjudications.add(adjudication)
        adjudicated_spans.add(adjudication.span_key())
    return adjudications, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "entries": len(adjudications),
    }


def valid_name_token(token: str) -> bool:
    if len(token) < 2 or len(token) > 40:
        return False
    saw_letter = False
    for character in token:
        category = unicodedata.category(character)
        if category.startswith("L"):
            saw_letter = True
        elif category.startswith("M") or character in "-'’ʼ":
            continue
        else:
            return False
    return saw_letter


def name_components(language: str, value: str) -> tuple[str, str] | None:
    """Return only conservative given/family splits; retain all full names."""
    value = " ".join(value.split())
    if language == "zh" and script_profile(value) == "Han" and 2 <= len(value) <= 4:
        family_length = 2 if value.startswith(CHINESE_COMPOUND_FAMILY_NAMES) and len(value) > 2 else 1
        return value[family_length:], value[:family_length]
    tokens = value.split()
    if not 2 <= len(tokens) <= 4 or not all(valid_name_token(token) for token in tokens):
        return None
    if language == "vi":
        if tokens[0] not in VIETNAMESE_COMMON_FAMILY_NAMES:
            return None
        return tokens[-1], tokens[0]
    if language not in WHITESPACE_COMPONENT_LANGUAGES:
        return None
    return tokens[0], tokens[-1]


def tags_for_span(
    language: str,
    label: str,
    value: str,
    *,
    derive_name_components: bool = True,
) -> list[tuple[str, str]]:
    label = label.casefold()
    tagged = [(tag, value) for tag in DIRECT_LABEL_TAGS.get(label, ())]
    if (
        derive_name_components
        and label == "person_name"
        and (components := name_components(language, value)) is not None
    ):
        given, family = components
        tagged.extend((("GIVEN_NAME", given), ("FAMILY_NAME", family)))
    return tagged


@dataclass
class Aggregate:
    count: int = 0
    source_counts: Counter[str] | None = None
    record_hashes: set[str] | None = None

    def __post_init__(self) -> None:
        self.source_counts = Counter() if self.source_counts is None else self.source_counts
        self.record_hashes = set() if self.record_hashes is None else self.record_hashes

    def add(self, dataset: str, source_record_hash: str) -> None:
        self.count += 1
        assert self.source_counts is not None
        assert self.record_hashes is not None
        self.source_counts[dataset] += 1
        self.record_hashes.add(source_record_hash)


def build_pool(
    input_specs: list[tuple[str, list[Path]]],
    *,
    pool_version: str = POOL_VERSION,
    source_manifests_by_dataset: dict[str, Path] | None = None,
    surface_normalization: str = "legacy-nfkc-whitespace",
    languages: set[str] | None = None,
    tags: set[str] | None = None,
    carrier_role: str = "train-and-audit",
    name_component_mode: str = "conservative-derived",
    holdout_modulus: int = 10,
    holdout_fold: int = 0,
    require_source_manifests: bool = False,
    surface_adjudications: set[SurfaceAdjudication] | None = None,
    surface_adjudication_receipt: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not pool_version:
        raise ValueError("pool_version must be nonempty")
    if surface_normalization not in SURFACE_NORMALIZATIONS:
        raise ValueError(f"unsupported surface normalization: {surface_normalization}")
    if carrier_role not in CARRIER_ROLES:
        raise ValueError(f"unsupported carrier role: {carrier_role}")
    if name_component_mode not in NAME_COMPONENT_MODES:
        raise ValueError(f"unsupported name-component mode: {name_component_mode}")
    if holdout_modulus < 2:
        raise ValueError("holdout_modulus must be at least 2")
    if not 0 <= holdout_fold < holdout_modulus:
        raise ValueError("holdout_fold must be in [0, holdout_modulus)")
    if tags is not None:
        tags = {tag.upper() for tag in tags}
        if unknown_tags := tags - SUPPORTED_POOL_TAGS:
            raise ValueError("unsupported pool tags: " + ", ".join(sorted(unknown_tags)))
    source_manifests_by_dataset = source_manifests_by_dataset or {}
    surface_adjudications = surface_adjudications or set()
    adjudication_by_span = {adjudication.span_key(): adjudication for adjudication in surface_adjudications}
    if len(adjudication_by_span) != len(surface_adjudications):
        raise ValueError("surface adjudications contain duplicate adjudicated spans")
    input_datasets = {dataset for dataset, _ in input_specs}
    unknown_manifests = set(source_manifests_by_dataset) - input_datasets
    if unknown_manifests:
        raise ValueError(
            "source manifests name unknown input datasets: " + ", ".join(sorted(unknown_manifests))
        )
    unknown_adjudication_datasets = {
        adjudication.dataset for adjudication in surface_adjudications
    } - input_datasets
    if unknown_adjudication_datasets:
        raise ValueError(
            "surface adjudications name unknown input datasets: "
            + ", ".join(sorted(unknown_adjudication_datasets))
        )

    aggregates: dict[tuple[str, str, str, str], Aggregate] = defaultdict(Aggregate)
    rows_by_split: Counter[str] = Counter()
    spans_by_split: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    input_files: dict[str, list[str]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    missing_source_manifests = []
    matched_adjudications: set[SurfaceAdjudication] = set()
    adjudication_reasons: Counter[str] = Counter()

    for dataset, paths in input_specs:
        input_files[dataset] = [str(path) for path in paths]
        manifest = (
            load_source_manifest(source_manifests_by_dataset[dataset])
            if dataset in source_manifests_by_dataset
            else source_manifest(paths)
        )
        if manifest is None:
            missing_source_manifests.append(dataset)
        else:
            manifest_path, manifest_row = manifest
            source_manifests[dataset] = {
                "dataset": manifest_row.get("dataset"),
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "upstream": manifest_row["upstream"],
            }
        for path in paths:
            for line_number, row in enumerate(iter_jsonl(path), 1):
                language = row_language(row)
                text = row.get("text")
                if language is None or not isinstance(text, str):
                    rejected["rows_missing_language_or_text"] += 1
                    continue
                if languages is not None and language not in languages:
                    continue
                row_id = str(row.get("id", line_number))
                hashed_record = record_hash(dataset, row_id, row)
                source_text_sha256 = stable_hash(text)
                split = (
                    "audit" if int(source_text_sha256[:16], 16) % holdout_modulus == holdout_fold else "train"
                )
                rows_by_split[split] += 1
                spans = row.get("spans")
                if not isinstance(spans, list):
                    rejected["rows_with_non_list_spans"] += 1
                    continue
                for raw_span in spans:
                    span = parse_span(raw_span)
                    if span is None or span.start < 0 or span.end <= span.start or span.end > len(text):
                        rejected["invalid_spans"] += 1
                        continue
                    value = text[span.start : span.end]
                    if not value.strip():
                        rejected["empty_values"] += 1
                        continue
                    adjudication = adjudication_by_span.get(
                        (dataset, row_id, span.start, span.end, span.label.casefold(), value)
                    )
                    if adjudication is not None:
                        matched_adjudications.add(adjudication)
                        adjudication_reasons[adjudication.reason] += 1
                        rejected["adjudicated_surface_spans"] += 1
                        continue
                    emitted = tags_for_span(
                        language,
                        span.label,
                        value,
                        derive_name_components=name_component_mode == "conservative-derived",
                    )
                    if not emitted:
                        rejected["unsupported_labels"] += 1
                        continue
                    emitted = [item for item in emitted if tags is None or item[0] in tags]
                    if not emitted:
                        rejected["spans_outside_tag_scope"] += 1
                        continue
                    spans_by_split[split] += 1
                    for tag, surface in emitted:
                        stored_surface = (
                            " ".join(unicodedata.normalize("NFKC", surface).split())
                            if surface_normalization == "legacy-nfkc-whitespace"
                            else surface
                        )
                        if not normalize_value(stored_surface):
                            rejected["empty_normalized_values"] += 1
                            continue
                        aggregates[(split, language, tag, stored_surface)].add(dataset, hashed_record)

    if require_source_manifests and missing_source_manifests:
        raise ValueError(
            "source manifests are required but missing or ambiguous for: "
            + ", ".join(sorted(missing_source_manifests))
        )
    unmatched_adjudications = surface_adjudications - matched_adjudications
    if unmatched_adjudications:
        first = sorted(
            unmatched_adjudications,
            key=lambda item: (item.dataset, item.row_id, item.start, item.end, item.label),
        )[0]
        raise ValueError(
            "surface adjudications did not match configured pool input; "
            f"unmatched={len(unmatched_adjudications)} first="
            f"{first.dataset}:{first.row_id}:{first.start}:{first.end}:{first.label}"
        )

    entries = []
    for (split, language, tag, surface), aggregate in sorted(aggregates.items()):
        assert aggregate.source_counts is not None
        assert aggregate.record_hashes is not None
        entry_id = stable_hash(f"{pool_version}\0{split}\0{language}\0{tag}\0{surface}")[:20]
        entries.append(
            {
                "pool_version": pool_version,
                "entry_id": entry_id,
                "split": split,
                "lang": language,
                "tag": tag,
                "value": surface,
                "normalized": normalize_value(surface),
                "count": aggregate.count,
                "source_counts": dict(sorted(aggregate.source_counts.items())),
                "source_record_hashes": sorted(aggregate.record_hashes),
            }
        )

    group_counts: Counter[tuple[str, str, str]] = Counter()
    group_weight: Counter[tuple[str, str, str]] = Counter()
    for entry in entries:
        key = (entry["split"], entry["lang"], entry["tag"])
        group_counts[key] += 1
        group_weight[key] += entry["count"]
    groups = [
        {
            "split": split,
            "lang": language,
            "tag": tag,
            "distinct_values": group_counts[(split, language, tag)],
            "observed_count": group_weight[(split, language, tag)],
        }
        for split, language, tag in sorted(group_counts)
    ]
    normalized_by_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for entry in entries:
        normalized_by_group[(entry["split"], entry["lang"], entry["tag"])].add(entry["normalized"])
    language_tags = sorted({(entry["lang"], entry["tag"]) for entry in entries})
    train_audit_overlap = []
    for language, tag in language_tags:
        train_values = normalized_by_group[("train", language, tag)]
        audit_values = normalized_by_group[("audit", language, tag)]
        intersection = train_values & audit_values
        train_audit_overlap.append(
            {
                "lang": language,
                "tag": tag,
                "train_distinct_normalized": len(train_values),
                "audit_distinct_normalized": len(audit_values),
                "intersection_distinct_normalized": len(intersection),
                "audit_seen_in_train_fraction": (
                    len(intersection) / len(audit_values) if audit_values else None
                ),
            }
        )
    train_languages = sorted({entry["lang"] for entry in entries if entry["split"] == "train"})
    name_component_semantics = (
        "Full person names and directly labeled given/family components are retained. Generic person-name "
        "spans never produce derived components."
        if name_component_mode == "direct-only"
        else (
            "Full person names are always retained. Given/family splits are restricted to conservative "
            "given-first whitespace-language cases, compact Han Chinese names, and Vietnamese family-first "
            "names whose first token is in the frozen family-name allowlist; ambiguous cases back off."
        )
    )
    report = {
        "schema_version": 1,
        "pool_version": pool_version,
        "semantics": {
            "frequency": "Observed counts are retained as sampling weights; exact repeats are not filtered.",
            "partition": (
                "Source texts are assigned to train or audit before surface aggregation; repeated text stays "
                "in one fold across row and dataset identities."
            ),
            "record_identity": (
                "Source-record hashes bind dataset alias, row ID, language, text, and spans for provenance; "
                "local paths and record identity do not affect partitioning."
            ),
            "name_components": name_component_semantics,
        },
        "configuration": {
            "languages": sorted(languages) if languages is not None else None,
            "tags": sorted(tags) if tags is not None else None,
            "carrier_role": carrier_role,
            "name_component_mode": name_component_mode,
            "holdout_modulus": holdout_modulus,
            "holdout_fold": holdout_fold,
            "surface_normalization": surface_normalization,
            "vietnamese_family_name_allowlist": sorted(VIETNAMESE_COMMON_FAMILY_NAMES),
            "surface_adjudication": surface_adjudication_receipt,
        },
        "coverage": {
            "train_languages": train_languages,
            "missing_requested_languages": (
                sorted(languages - set(train_languages)) if languages is not None else []
            ),
            "train_tags_by_language": {
                language: sorted(
                    {
                        entry["tag"]
                        for entry in entries
                        if entry["split"] == "train" and entry["lang"] == language
                    }
                )
                for language in train_languages
            },
        },
        "input_files": input_files,
        "source_manifests": source_manifests,
        "missing_source_manifests": sorted(missing_source_manifests),
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "source_spans_by_split": dict(sorted(spans_by_split.items())),
        "rejected": dict(sorted(rejected.items())),
        "surface_adjudication": {
            "configured": len(surface_adjudications),
            "matched": len(matched_adjudications),
            "excluded_by_reason": dict(sorted(adjudication_reasons.items())),
        },
        "groups": groups,
        "train_audit_value_overlap": train_audit_overlap,
        "entries": len(entries),
    }
    return entries, report


@dataclass(frozen=True)
class PoolValue:
    value: str
    count: int
    source_observed_count: int
    provenance: str
    entry_id: str
    normalized: str
    source_counts: dict[str, int]
    source_record_hashes: tuple[str, ...]
    sampling_count_receipt: dict[str, Any] | None


class SurfacePool:
    def __init__(self, groups: dict[tuple[str, str], tuple[PoolValue, ...]], version: str):
        self.groups = groups
        self.version = version
        self._metadata_by_provenance = {
            item.provenance: {
                "pool_version": self.version,
                "entry_id": item.entry_id,
                "value_normalized": item.normalized,
                "observed_count": item.source_observed_count,
                "sampling_count": item.count,
                "sampling_count_receipt": item.sampling_count_receipt,
                "source_counts": item.source_counts,
                "source_record_hashes": list(item.source_record_hashes),
            }
            for values in groups.values()
            for item in values
        }

    @classmethod
    def load(cls, path: Path, *, split: str = "train") -> "SurfacePool":
        groups: dict[tuple[str, str], list[PoolValue]] = defaultdict(list)
        versions: set[str] = set()
        for row in iter_jsonl(path):
            if row.get("split") != split:
                continue
            language, tag, value = row.get("lang"), row.get("tag"), row.get("value")
            count, entry_id, version = row.get("count"), row.get("entry_id"), row.get("pool_version")
            if (
                not isinstance(language, str)
                or not language
                or not isinstance(tag, str)
                or not tag
                or not isinstance(value, str)
                or not value
                or not isinstance(entry_id, str)
                or not entry_id
                or not isinstance(version, str)
                or not version
            ):
                raise ValueError(f"{path}: malformed surface-pool string field")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(f"{path}: surface-pool count must be positive")
            source_observed_count = row.get("source_observed_count", count)
            if (
                not isinstance(source_observed_count, int)
                or isinstance(source_observed_count, bool)
                or source_observed_count <= 0
            ):
                raise ValueError(f"{path}: source_observed_count must be a positive integer")
            sampling_count_receipt = row.get("sampling_count_receipt")
            if sampling_count_receipt is not None and not isinstance(sampling_count_receipt, dict):
                raise ValueError(f"{path}: sampling_count_receipt must be an object")
            if sampling_count_receipt is not None and sampling_count_receipt.get("sampling_count") != count:
                raise ValueError(f"{path}: sampling_count_receipt does not match count")
            versions.add(version)
            groups[(language, tag)].append(
                PoolValue(
                    value=value,
                    count=count,
                    source_observed_count=source_observed_count,
                    provenance=f"natural-pool:{version}:{entry_id}",
                    entry_id=entry_id,
                    normalized=str(row.get("normalized", "")),
                    source_counts=dict(row.get("source_counts") or {}),
                    source_record_hashes=tuple(row.get("source_record_hashes") or ()),
                    sampling_count_receipt=(
                        dict(sampling_count_receipt) if sampling_count_receipt is not None else None
                    ),
                )
            )
        if len(versions) != 1:
            raise ValueError(f"{path}: expected exactly one pool version, found {sorted(versions)}")
        return cls({key: tuple(values) for key, values in groups.items()}, versions.pop())

    def draw(
        self,
        language: str,
        tag: str,
        rng: random.Random,
        *,
        count_temperature: float = 1.0,
    ) -> tuple[str, str] | None:
        if not 0 <= count_temperature <= 1:
            raise ValueError("count_temperature must be in [0, 1]")
        values = self.groups.get((language, tag))
        if not values:
            return None
        selected = rng.choices(
            values,
            weights=[item.count**count_temperature for item in values],
            k=1,
        )[0]
        return selected.value, selected.provenance

    def provenance_metadata(self, provenance: str) -> list[dict[str, Any]]:
        """Resolve every pool entry named by a simple or composed generator string."""
        tokens = {match.group(0) for match in POOL_PROVENANCE.finditer(provenance)}
        unknown = tokens - self._metadata_by_provenance.keys()
        if unknown:
            raise ValueError(f"unknown surface-pool provenance: {', '.join(sorted(unknown))}")
        return sorted(
            (self._metadata_by_provenance[token] for token in tokens),
            key=lambda item: item["entry_id"],
        )

    def distinct_count(self, language: str, tag: str) -> int:
        return len(self.groups.get((language, tag), ()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH_OR_GLOB")
    parser.add_argument(
        "--source-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="bind an input alias to an explicit provenance manifest",
    )
    parser.add_argument("--language", action="append", help="Keep only this language; repeatable.")
    parser.add_argument("--tag", action="append", help="Keep only this emitted tag; repeatable.")
    parser.add_argument("--pool-version", default=POOL_VERSION)
    parser.add_argument("--carrier-role", choices=CARRIER_ROLES, default="train-and-audit")
    parser.add_argument(
        "--name-component-mode",
        choices=NAME_COMPONENT_MODES,
        default="conservative-derived",
        help="derive conservative components from generic person names, or retain direct labels only",
    )
    parser.add_argument(
        "--surface-normalization",
        choices=SURFACE_NORMALIZATIONS,
        default="legacy-nfkc-whitespace",
    )
    parser.add_argument("--holdout-modulus", type=int, default=10)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument(
        "--require-source-manifests",
        action="store_true",
        help="fail unless every input resolves to exactly one onboarding manifest with upstream metadata",
    )
    parser.add_argument(
        "--surface-adjudication",
        type=Path,
        help="exclude exact provenance-bound source spans listed in a JSONL adjudication",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = [resolve_input(specification) for specification in args.input]
        source_manifests_by_dataset = named_paths(args.source_manifest)
        surface_adjudications, adjudication_receipt = (
            load_surface_adjudications(args.surface_adjudication)
            if args.surface_adjudication is not None
            else (set(), None)
        )
        entries, report = build_pool(
            inputs,
            pool_version=args.pool_version,
            source_manifests_by_dataset=source_manifests_by_dataset,
            surface_normalization=args.surface_normalization,
            languages=set(args.language) if args.language else None,
            tags=set(args.tag) if args.tag else None,
            carrier_role=args.carrier_role,
            name_component_mode=args.name_component_mode,
            holdout_modulus=args.holdout_modulus,
            holdout_fold=args.holdout_fold,
            require_source_manifests=args.require_source_manifests,
            surface_adjudications=surface_adjudications,
            surface_adjudication_receipt=adjudication_receipt,
        )
    except ValueError as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for entry in entries:
            output.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"entries": len(entries), "rows_by_split": report["rows_by_split"]}, sort_keys=True))
    print(args.out)
    print(args.report)


if __name__ == "__main__":
    main()
