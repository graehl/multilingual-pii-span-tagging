#!/usr/bin/env python
"""Audit multilingual PII surface and masked-carrier diversity with provenance."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, TextIO

JSONL_SUFFIXES = (".jsonl", ".jsonl.gz")
SCRIPT_MARKERS = (
    ("LATIN", "Latin"),
    ("ARABIC", "Arabic"),
    ("CJK UNIFIED", "Han"),
    ("CJK COMPATIBILITY", "Han"),
    ("IDEOGRAPH", "Han"),
    ("HIRAGANA", "Hiragana"),
    ("KATAKANA", "Katakana"),
    ("HANGUL", "Hangul"),
    ("CYRILLIC", "Cyrillic"),
    ("DEVANAGARI", "Devanagari"),
    ("GREEK", "Greek"),
    ("HEBREW", "Hebrew"),
    ("THAI", "Thai"),
    ("BENGALI", "Bengali"),
    ("GURMUKHI", "Gurmukhi"),
    ("GUJARATI", "Gujarati"),
    ("TAMIL", "Tamil"),
    ("TELUGU", "Telugu"),
    ("KANNADA", "Kannada"),
    ("MALAYALAM", "Malayalam"),
)
PRESERVED_PUNCTUATION = frozenset("-_/.:@+()[]#")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


@dataclass
class ValueStats:
    count: int = 0
    normalized_values: Counter[str] = field(default_factory=Counter)
    display_values: dict[str, str] = field(default_factory=dict)
    scripts: Counter[str] = field(default_factory=Counter)
    shapes: Counter[str] = field(default_factory=Counter)

    def add(self, value: str) -> None:
        normalized = normalize_value(value)
        self.count += 1
        self.normalized_values[normalized] += 1
        self.display_values.setdefault(normalized, value)
        self.scripts[script_profile(value)] += 1
        self.shapes[shape_signature(value)] += 1

    def render(self, top_k: int) -> dict[str, Any]:
        distinct = len(self.normalized_values)
        singletons = sum(count == 1 for count in self.normalized_values.values())
        top_values = [
            {
                "count": count,
                "normalized": normalized,
                "surface": self.display_values[normalized],
            }
            for normalized, count in self.normalized_values.most_common(top_k)
        ]
        return {
            "count": self.count,
            "distinct_normalized": distinct,
            "duplicate_occurrences": self.count - distinct,
            "duplicate_occurrence_fraction": ratio(self.count - distinct, self.count),
            "singleton_normalized": singletons,
            "max_frequency": max(self.normalized_values.values(), default=0),
            "scripts": dict(sorted(self.scripts.items())),
            "shapes_top": dict(self.shapes.most_common(top_k)),
            "values_top": top_values,
        }


@dataclass
class CarrierStats:
    count: int = 0
    carriers: Counter[str] = field(default_factory=Counter)

    def add(self, carrier: str) -> None:
        self.count += 1
        self.carriers[carrier] += 1

    def render(self, top_k: int) -> dict[str, Any]:
        distinct = len(self.carriers)
        return {
            "count": self.count,
            "distinct_masked": distinct,
            "duplicate_occurrences": self.count - distinct,
            "duplicate_occurrence_fraction": ratio(self.count - distinct, self.count),
            "max_frequency": max(self.carriers.values(), default=0),
            "masked_top": [
                {"count": count, "text": carrier} for carrier, count in self.carriers.most_common(top_k)
            ],
        }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_value(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def character_script(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    for marker, script in SCRIPT_MARKERS:
        if marker in name:
            return script
    return "OtherLetter"


def script_profile(value: str) -> str:
    scripts = sorted({script for character in value if (script := character_script(character))})
    if not scripts:
        return "NoLetters"
    return "+".join(scripts)


def shape_signature(value: str) -> str:
    categories: list[str] = []
    for character in unicodedata.normalize("NFKC", value):
        category = unicodedata.category(character)
        if category.startswith("L"):
            token = "L"
        elif category == "Nd":
            token = "D"
        elif character.isspace():
            token = "_"
        elif category.startswith("M"):
            token = "M"
        elif character in PRESERVED_PUNCTUATION:
            token = character
        elif category.startswith("P"):
            token = "P"
        elif category.startswith("S"):
            token = "S"
        else:
            token = "X"
        categories.append(token)

    runs: list[str] = []
    for token in categories:
        if runs and runs[-1].split("{", 1)[0] == token:
            base, _, count_text = runs[-1].partition("{")
            count = 1 if not count_text else int(count_text.rstrip("}"))
            runs[-1] = f"{base}{{{count + 1}}}"
        else:
            runs.append(token)
    return "".join(runs)


def parse_span(raw: Any) -> Span | None:
    if isinstance(raw, dict):
        start, end = raw.get("start"), raw.get("end")
        label = raw.get("label", raw.get("type"))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
        start, end, label = raw[:3]
    else:
        return None
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
        return None
    return Span(start=start, end=end, label=label)


def row_language(row: dict[str, Any]) -> str | None:
    language = row.get("lang")
    if isinstance(language, str) and language:
        return language
    for container_name in ("meta", "metadata"):
        container = row.get(container_name)
        if isinstance(container, dict):
            language = container.get("lang")
            if isinstance(language, str) and language:
                return language
    return None


def provenance_queues(row: dict[str, Any]) -> dict[tuple[int, int], deque[dict[str, Any]]]:
    queues: dict[tuple[int, int], deque[dict[str, Any]]] = defaultdict(deque)
    for raw in row.get("span_provenance", []):
        if not isinstance(raw, dict):
            continue
        start, end = raw.get("start"), raw.get("end")
        if isinstance(start, int) and isinstance(end, int):
            queues[(start, end)].append(raw)
    return queues


def row_source(row: dict[str, Any], dataset: str) -> str:
    for key in ("src", "source", "mix_source"):
        value = row.get(key)
        if value:
            return str(value)
    for container_name in ("meta", "metadata"):
        metadata = row.get(container_name)
        if isinstance(metadata, dict):
            for key in ("src", "source", "mix_source", "source_dataset", "corpus"):
                value = metadata.get(key)
                if value:
                    return str(value)
    return dataset


def origin_family(origin_kind: str, origin: str) -> str:
    """Collapse instance-level provenance into a stable generator family."""
    if origin_kind == "row_source":
        return f"row-source:{origin}"
    if "natural-pool:" in origin:
        return "natural-pool"
    if origin.startswith("name:native:faker:"):
        return "name-native-faker"
    if origin.startswith("name:latin"):
        return "name-latin"
    if origin.startswith("faker:"):
        return "faker"
    if origin.startswith("name:"):
        return "name-other"
    return origin.split(":", 1)[0]


def mask_carrier(text: str, spans: list[Span]) -> tuple[str, int]:
    pieces: list[str] = []
    cursor = 0
    overlaps = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            overlaps += 1
            continue
        pieces.append(text[cursor : span.start])
        pieces.append(f"<{span.label}>")
        cursor = span.end
    pieces.append(text[cursor:])
    return normalize_value("".join(pieces)), overlaps


def open_jsonl(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open_jsonl(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def resolve_input(specification: str) -> tuple[str, list[Path]]:
    if "=" not in specification:
        raise ValueError(f"input must be NAME=PATH_OR_GLOB, got {specification!r}")
    name, pattern = specification.split("=", 1)
    if not name or not pattern:
        raise ValueError(f"input must be NAME=PATH_OR_GLOB, got {specification!r}")
    root = Path(pattern)
    if root.is_dir():
        paths = sorted(
            path for path in root.rglob("*") if path.is_file() and path.name.endswith(JSONL_SUFFIXES)
        )
    elif glob.has_magic(pattern):
        paths = sorted(Path(match) for match in glob.glob(pattern, recursive=True) if Path(match).is_file())
    elif root.is_file():
        paths = [root]
    else:
        paths = []
    paths = [path for path in paths if path.name.endswith(JSONL_SUFFIXES)]
    if not paths:
        raise ValueError(f"input {name!r} resolved to no JSONL files: {pattern}")
    return name, paths


def render_keyed_counter(counter: Counter[tuple[str, ...]], names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [dict(zip(names, key, strict=True), count=count) for key, count in sorted(counter.items())]


def audit(
    input_specs: list[tuple[str, list[Path]]],
    *,
    languages: set[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    row_counts: Counter[tuple[str, str, str]] = Counter()
    span_counts: Counter[tuple[str, str, str, str]] = Counter()
    value_stats: dict[tuple[str, str, str, str, str], ValueStats] = defaultdict(ValueStats)
    aggregate_value_stats: dict[tuple[str, str, str, str], ValueStats] = defaultdict(ValueStats)
    carrier_stats: dict[tuple[str, str, str], CarrierStats] = defaultdict(CarrierStats)
    carrier_multisets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    diagnostics: Counter[str] = Counter()
    input_files: dict[str, list[str]] = {}

    for dataset, paths in input_specs:
        input_files[dataset] = [str(path) for path in paths]
        for path in paths:
            for row in iter_jsonl(path):
                text = row.get("text")
                language = row_language(row)
                if not isinstance(text, str) or language is None:
                    diagnostics["rows_missing_text_or_language"] += 1
                    continue
                if languages is not None and language not in languages:
                    continue
                source = row_source(row, dataset)
                row_counts[(dataset, language, source)] += 1
                queues = provenance_queues(row)
                raw_spans = row.get("spans", [])
                if not isinstance(raw_spans, list):
                    diagnostics["rows_with_non_list_spans"] += 1
                    continue

                valid_spans: list[Span] = []
                for raw_span in raw_spans:
                    span = parse_span(raw_span)
                    if span is None:
                        diagnostics["malformed_spans"] += 1
                        continue
                    if span.start < 0 or span.end <= span.start or span.end > len(text):
                        diagnostics["invalid_span_offsets"] += 1
                        continue
                    valid_spans.append(span)
                    provenance = (
                        queues[(span.start, span.end)].popleft() if queues[(span.start, span.end)] else None
                    )
                    if row.get("span_provenance") and provenance is None:
                        diagnostics["spans_missing_provenance_match"] += 1
                    generator = provenance.get("generator") if provenance else None
                    origin_kind = "generator" if generator else "row_source"
                    origin = str(generator or source)
                    surface = text[span.start : span.end]
                    if not surface:
                        diagnostics["empty_surfaces"] += 1
                    span_counts[(dataset, language, span.label, origin)] += 1
                    value_stats[(dataset, language, span.label, origin_kind, origin)].add(surface)
                    family = origin_family(origin_kind, origin)
                    aggregate_value_stats[(dataset, language, span.label, family)].add(surface)

                diagnostics["unused_provenance_records"] += sum(len(queue) for queue in queues.values())
                carrier, overlaps = mask_carrier(text, valid_spans)
                diagnostics["overlapping_spans_skipped_in_carrier_mask"] += overlaps
                carrier_stats[(dataset, language, source)].add(carrier)
                carrier_multisets[(dataset, language)][carrier] += 1

    groups = []
    for (dataset, language, label, origin_kind, origin), stats in sorted(value_stats.items()):
        groups.append(
            {
                "dataset": dataset,
                "language": language,
                "label": label,
                "origin_kind": origin_kind,
                "origin": origin,
                **stats.render(top_k),
            }
        )
    carriers = []
    for (dataset, language, source), stats in sorted(carrier_stats.items()):
        carriers.append(
            {
                "dataset": dataset,
                "language": language,
                "row_source": source,
                **stats.render(top_k),
            }
        )
    aggregate_groups = []
    for (dataset, language, label, family), stats in sorted(aggregate_value_stats.items()):
        aggregate_groups.append(
            {
                "dataset": dataset,
                "language": language,
                "label": label,
                "origin_family": family,
                **stats.render(top_k),
            }
        )

    carrier_pairs = []
    languages_seen = sorted({language for _, language in carrier_multisets})
    for language in languages_seen:
        datasets = sorted(dataset for dataset, candidate in carrier_multisets if candidate == language)
        for left_name, right_name in combinations(datasets, 2):
            left = carrier_multisets[(left_name, language)]
            right = carrier_multisets[(right_name, language)]
            intersection = left & right
            union = left | right
            distinct_intersection = len(set(left) & set(right))
            distinct_union = len(set(left) | set(right))
            carrier_pairs.append(
                {
                    "language": language,
                    "left_dataset": left_name,
                    "right_dataset": right_name,
                    "left_count": sum(left.values()),
                    "right_count": sum(right.values()),
                    "left_distinct": len(left),
                    "right_distinct": len(right),
                    "distinct_intersection": distinct_intersection,
                    "distinct_union": distinct_union,
                    "distinct_jaccard": ratio(distinct_intersection, distinct_union),
                    "multiset_intersection": sum(intersection.values()),
                    "multiset_union": sum(union.values()),
                    "exact_multiset_equal": left == right,
                }
            )

    return {
        "schema_version": 1,
        "semantics": {
            "duplicate_values": (
                "Reported as distribution evidence only; exact entity-value recurrence is never rejected "
                "or treated as train/evaluation overlap by this audit."
            ),
            "masked_carriers": (
                "Entity spans are replaced by their fine labels before normalization, so carrier repetition "
                "does not depend on entity-value equality."
            ),
            "carrier_pairs": (
                "Pairwise carrier comparisons use those masked, normalized carrier multisets. Exact equality "
                "therefore means the compared datasets differ only inside labeled entity spans."
            ),
            "origin_families": (
                "Compact surface groups collapse instance-level natural-pool IDs and locale-specific generator "
                "names; surface_groups retains the full provenance detail."
            ),
            "normalization": "Unicode NFKC, case-folding, whitespace collapse.",
            "script_profile": "Set of Unicode letter scripts in each surface; digits and punctuation do not make a value mixed-script.",
        },
        "configuration": {
            "languages": sorted(languages) if languages is not None else None,
            "top_k": top_k,
        },
        "input_files": input_files,
        "totals": {
            "rows": sum(row_counts.values()),
            "spans": sum(span_counts.values()),
            "datasets": len(input_files),
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "row_counts": render_keyed_counter(row_counts, ("dataset", "language", "row_source")),
        "span_counts": render_keyed_counter(span_counts, ("dataset", "language", "label", "origin")),
        "surface_groups": groups,
        "surface_aggregate_groups": aggregate_groups,
        "carrier_groups": carriers,
        "carrier_pair_groups": carrier_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH_OR_GLOB",
        help="Repeatable named JSONL/JSONL.GZ file, directory, or quoted glob.",
    )
    parser.add_argument("--language", action="append", help="Keep only this language; repeatable.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    try:
        input_specs = [resolve_input(specification) for specification in args.input]
    except ValueError as error:
        parser.error(str(error))
    result = audit(input_specs, languages=set(args.language) if args.language else None, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["totals"], sort_keys=True))
    print(json.dumps(result["diagnostics"], sort_keys=True))
    print(args.out)


if __name__ == "__main__":
    main()
