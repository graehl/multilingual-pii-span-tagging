#!/usr/bin/env python3
# acli: 1 complete
"""Plan and retrieve targeted non-proof ont3 annotation candidates.

``plan`` audits a reviewed factorized surface corpus, identifies low-support
primary tags, positive predicates, and categorical subclasses, and derives a
bounded set of literal, context-anchor, and structural retrieval queries.
``retrieve`` applies that frozen plan to pinned FineWeb/FineWeb2 streams and
emits exact-offset sentence candidates.  Retrieval is positive candidate
selection only: no selected sentence or unmatched character is supervision
until a teacher pass and independent review admit it.
``gold-supplement`` recovers diverse exact primary positives already present in
trusted ont2 gold, masks all factor channels, and excludes the fixed realizer
audit by source hash.
``primary-review-packet`` exposes adjacent document context but no retrieval
condition or earlier proposal to an independent ordinary-span annotator.
``primary-intersection-review`` binds exact primary agreements to literal
context without admitting them, and ``primary-intersection-materialize``
converts a complete root adjudication into weighted positive-only training
rows while keeping every nonaccepted character unknown.
``successor-packet`` joins mechanically healthy primary proposals back to their
retrieved context without upgrading either the retrieval hint or proposal to
reviewed supervision. ``successor-review-remainder`` excludes already paid
independent rows and reference-banned rows without exposing the reference
successor proposals to the next annotator. ``successor-intersection-review``
binds exact successor agreements to their literal carriers and context without
admitting them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import acli
except ImportError:
    sys.path.insert(0, str(Path.home() / "agents"))
    import acli

from scripts.pii_fineweb_sources import (
    FINEWEB2_CONFIGS,
    FINEWEB2_DATASET,
    FINEWEB2_REVISION,
    FINEWEB_EN_DATASET,
    FINEWEB_EN_REVISION,
)
from scripts.pii_text_segmentation import (
    SaTCharacterSpanSplitter,
    segmenter_provenance,
    sentence_spans,
)

PLAN_SCHEMA = "pii-ont3-targeted-retrieval-plan/v1"
CANDIDATE_SCHEMA = "pii-ont3-targeted-retrieval-candidate/v1"
RECEIPT_SCHEMA = "pii-ont3-targeted-retrieval-receipt/v1"
PRIMARY_SUPPLEMENT_SCHEMA = "pii-ont3-primary-surface-supplement/v1"
PRIMARY_SUPPLEMENT_RECEIPT_SCHEMA = "pii-ont3-primary-surface-supplement-receipt/v1"
PRIMARY_INTERSECTION_REVIEW_SCHEMA = "pii-ont3-primary-exact-intersection-review/v1"
PRIMARY_INTERSECTION_ADJUDICATION_SCHEMA = (
    "pii.ontology-v3.targeted-surface-primary-exact-intersection-root-review.v1"
)
PRIMARY_INTERSECTION_MATERIALIZATION_SCHEMA = (
    "pii-ont3-primary-exact-intersection-reviewed-materialization/v1"
)
PRIMARY_INTERSECTION_MATERIALIZATION_RECEIPT_SCHEMA = (
    "pii-ont3-primary-exact-intersection-reviewed-materialization-receipt/v1"
)
SUCCESSOR_INTERSECTION_REVIEW_SCHEMA = "pii-ont3-successor-exact-intersection-review/v1"
SUCCESSOR_INTERSECTION_ADJUDICATION_SCHEMA = (
    "pii.ontology-v3.targeted-surface-successor-exact-intersection-root-review.v1"
)
SUCCESSOR_INTERSECTION_MATERIALIZATION_RECEIPT_SCHEMA = (
    "pii-ont3-successor-exact-intersection-reviewed-materialization-receipt/v1"
)
ANNOTATED_SPANS_ONLY = "annotated_spans_only"
WORD_RE = re.compile(r"[^\W\d_][\w'’\-]{3,}", re.UNICODE)
STRUCTURAL_PATTERNS = {
    "primary:email": re.compile(
        r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@(?:[\w-]+\.)+[\w-]{2,}(?![\w-])",
        re.UNICODE,
    ),
    "primary:url": re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE),
    "primary:monetary_amount": re.compile(
        r"(?:[$€£¥₽₹₩₺₴₫₱₪₦₲]|\b(?:USD|EUR|GBP|JPY|CNY|RMB|RUB|INR|KRW|TRY|UAH|VND|PHP|ILS|NGN|BTC|ETH)\b)"
        r"\s?\d[\d.,\u00a0 ]*",
        re.IGNORECASE,
    ),
}
SURFACE_ANCHOR_STOPWORDS = {
    "de": {"der", "die", "das", "den", "dem", "des", "eine", "einer", "einem", "einen", "ihre", "seine"},
    "en": {"the", "this", "that", "their", "there", "these", "those", "whose", "your"},
    "es": {
        "del",
        "desde",
        "ella",
        "ellos",
        "esta",
        "este",
        "estos",
        "nuestra",
        "nuestro",
        "para",
        "responsable",
    },
    "fr": {"cette", "dans", "elle", "leur", "leurs", "notre", "votre", "vous"},
    "ru": {"его", "ему", "её", "который", "наш", "этот"},
}
CURATED_ID_QUERIES = {
    "primary:government_id": {
        "de": ("Ausweisnummer", "Passnummer", "Steuernummer", "Zulassungsnummer"),
        "en": ("government ID", "license number", "passport number", "registration number"),
        "es": ("cédula", "licencia profesional", "número de pasaporte", "número de registro"),
        "fr": ("numéro de licence", "numéro de passeport", "numéro d’inscription", "numéro d'identification"),
        "ja": ("免許番号", "登録番号", "旅券番号", "身分証明書番号"),
        "ko": ("면허번호", "등록번호", "여권번호", "주민등록번호"),
        "ru": ("номер лицензии", "номер паспорта", "регистрационный номер", "СНИЛС"),
        "zh": ("许可证号", "注册号", "护照号", "身份证号"),
    },
    "subclass:government_id_kind=professional_license": {
        "de": ("Arztnummer", "LANR", "Zulassungsnummer"),
        "en": ("medical license", "professional license", "provider number"),
        "es": ("colegiatura", "licencia profesional", "matrícula profesional"),
        "fr": ("numéro d’inscription", "numéro RPPS", "licence professionnelle"),
        "ja": ("医籍登録番号", "免許番号", "登録番号"),
        "ko": ("의사면허번호", "면허번호", "등록번호"),
        "ru": ("врачебная лицензия", "номер лицензии", "сертификат специалиста"),
        "zh": ("医师执业证书", "执业证号", "注册号"),
    },
}


class RetrievalError(RuntimeError):
    """Malformed input or an incomplete retrieval output."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(value: Mapping[str, Any], length: int = 24) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RetrievalError(f"{path}:{line_number}: {error}") from None
            if not isinstance(row, dict):
                raise RetrievalError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    write_new_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def row_conditions(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    target = row.get("target")
    primary = row.get("primary_tag")
    if not isinstance(target, str) or not target or not isinstance(primary, str) or not primary:
        raise RetrievalError("surface-corpus row requires nonempty target and primary_tag")
    observed: list[tuple[str, str]] = [(f"primary:{primary}", target)]
    for factor in row.get("factors", []):
        if not isinstance(factor, str):
            raise RetrievalError("surface-corpus factor must be a string")
        if factor.startswith("predicate:") and factor.endswith("=true"):
            observed.append((factor, target))
        elif factor.startswith("subclass:") and not factor.startswith("subclass:name_component="):
            observed.append((factor, target))
    for region in row.get("regions", []):
        if not isinstance(region, Mapping) or region.get("kind") != "subclass":
            continue
        family = region.get("family")
        value = region.get("value")
        start = region.get("start")
        end = region.get("end")
        if not isinstance(family, str) or not isinstance(value, str):
            raise RetrievalError("subclass region requires string family and value")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(target):
            raise RetrievalError("subclass region has invalid target-relative offsets")
        observed.append((f"subclass:{family}={value}", target[start:end]))
    return list(dict.fromkeys(observed))


def surface_anchor_words(language: str, surface: str) -> list[str]:
    stopwords = SURFACE_ANCHOR_STOPWORDS.get(language, set())
    return [
        word
        for word in WORD_RE.findall(surface)
        if len(word) >= 4 and word.casefold() not in stopwords and not word.isdigit()
    ]


def plan_retrieval(
    corpus_path: Path,
    *,
    split: str,
    minimum_examples: int,
    minimum_distinct: int,
    queries_per_cell: int,
) -> dict[str, Any]:
    if minimum_examples <= 0 or minimum_distinct <= 0 or queries_per_cell <= 0:
        raise RetrievalError("support floors and queries-per-cell must be positive")
    rows = read_jsonl(corpus_path)
    examples: Counter[tuple[str, str]] = Counter()
    surfaces: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    surface_anchors: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    source_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    selected_rows = 0
    observed_languages: set[str] = set()
    for row in rows:
        if row.get("split") != split:
            continue
        language = row.get("language")
        if not isinstance(language, str) or not language:
            raise RetrievalError("surface-corpus row requires nonempty language")
        observed_languages.add(language)
        selected_rows += 1
        group = str(row.get("source_group_id", ""))
        for condition, surface in row_conditions(row):
            key = language, condition
            examples[key] += 1
            surfaces[key][surface] += 1
            source_groups[key].add(group)
            for word in surface_anchor_words(language, surface):
                surface_anchors[key][word] += 1

    cells = []
    queries = []
    planned_keys = set(examples)
    planned_keys.update(
        (language, condition) for language in observed_languages for condition in STRUCTURAL_PATTERNS
    )
    for language, condition in sorted(planned_keys):
        example_count = examples[language, condition]
        distinct_count = len(surfaces[language, condition])
        if example_count >= minimum_examples and distinct_count >= minimum_distinct:
            continue
        key = language, condition
        cell_id = stable_id({"language": language, "condition": condition})
        structural_queries: list[dict[str, Any]] = []
        curated_queries: list[dict[str, Any]] = []
        anchor_queries: list[dict[str, Any]] = []
        exact_queries: list[dict[str, Any]] = []
        if condition in STRUCTURAL_PATTERNS:
            structural_queries.append(
                {
                    "kind": "structural_regex",
                    "value": STRUCTURAL_PATTERNS[condition].pattern,
                    "flags": STRUCTURAL_PATTERNS[condition].flags,
                }
            )
        for value in CURATED_ID_QUERIES.get(condition, {}).get(language, ()):
            curated_queries.append(
                {"kind": "curated_surface_context", "value": value, "flags": re.IGNORECASE}
            )
        for surface, count in sorted(
            surfaces[key].items(), key=lambda item: (-item[1], -len(item[0]), item[0])
        ):
            if len(surface.strip()) < 4 or "<mask>" in surface or "@example." in surface.lower():
                continue
            exact_queries.append({"kind": "exact_surface", "value": surface, "observed_count": count})
        if condition.startswith("predicate:") or condition in {
            "primary:organization_reference",
            "primary:person_reference",
            "primary:government_id",
            "subclass:government_id_kind=professional_license",
        }:
            for word, count in sorted(
                surface_anchors[key].items(), key=lambda item: (-item[1], -len(item[0]), item[0])
            ):
                anchor_queries.append(
                    {
                        "kind": "surface_anchor",
                        "value": word,
                        "flags": re.IGNORECASE,
                        "observed_count": count,
                    }
                )
        if structural_queries:
            cell_queries = structural_queries[:1]
        else:
            cell_queries = curated_queries[:4]
            remaining = queries_per_cell - len(cell_queries)
            anchor_budget = remaining // 2 if exact_queries else remaining
            cell_queries.extend(anchor_queries[:anchor_budget])
            remaining = queries_per_cell - len(cell_queries)
            cell_queries.extend(exact_queries[:remaining])
            remaining = queries_per_cell - len(cell_queries)
            cell_queries.extend(anchor_queries[anchor_budget : anchor_budget + remaining])
        deduplicated = []
        seen = set()
        for query in cell_queries:
            identity = query["value"]
            if identity in seen:
                continue
            seen.add(identity)
            query = {
                **query,
                "cell_id": cell_id,
                "condition": condition,
                "language": language,
            }
            query["query_id"] = stable_id(query)
            deduplicated.append(query)
            if len(deduplicated) == queries_per_cell:
                break
        if not deduplicated:
            continue
        queries.extend(deduplicated)
        cells.append(
            {
                "cell_id": cell_id,
                "language": language,
                "condition": condition,
                "examples": example_count,
                "distinct_surfaces": distinct_count,
                "source_groups": len(source_groups[key]),
                "example_deficit": max(0, minimum_examples - example_count),
                "distinct_deficit": max(0, minimum_distinct - distinct_count),
                "queries": len(deduplicated),
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "proof_data_used": False,
        "role": "nonproof_targeted_annotation_candidate_plan",
        "source_corpus": {
            "path": str(corpus_path),
            "sha256": sha256_file(corpus_path),
            "split": split,
            "rows": selected_rows,
        },
        "support_floor": {
            "minimum_examples": minimum_examples,
            "minimum_distinct_surfaces": minimum_distinct,
        },
        "query_policy": {
            "queries_per_cell": queries_per_cell,
            "surface_matching": "case-sensitive exact substring; structural patterns preserve exact offsets",
            "surface_anchors": ">=4-codepoint word-like substrings inside known positive surfaces, with a small language stoplist",
            "identifier_queries": "curated language-specific carrier terms plus substrings of known positive identifiers",
            "supervision": "retrieval candidates only; neither matches nor unmatched text are labels",
        },
        "cells": cells,
        "queries": queries,
        "counts": {
            "cells": len(cells),
            "queries": len(queries),
            "languages": len({cell["language"] for cell in cells}),
        },
    }


def fineweb_stream(
    language: str, *, seed: int, shuffle_buffer: int, upstream_split: str | None = None
) -> Iterable[Mapping[str, Any]]:
    if upstream_split not in {None, "train", "test"}:
        raise RetrievalError(f"unsupported upstream split: {upstream_split!r}")
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RetrievalError("retrieve requires datasets in the pinned pixi-gemma4 environment") from error
    if language == "en":
        dataset = load_dataset(
            FINEWEB_EN_DATASET,
            "sample-10BT",
            split=upstream_split or "train",
            revision=FINEWEB_EN_REVISION,
            streaming=True,
        )
    else:
        config = FINEWEB2_CONFIGS.get(language)
        if config is None:
            raise RetrievalError(f"no pinned FineWeb2 config for language {language!r}")
        dataset = load_dataset(
            FINEWEB2_DATASET,
            config,
            split=upstream_split or "test",
            revision=FINEWEB2_REVISION,
            streaming=True,
        )
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def compile_query(query: Mapping[str, Any]) -> re.Pattern[str]:
    if query["kind"] == "structural_regex":
        return re.compile(str(query["value"]), int(query.get("flags", 0)))
    return re.compile(re.escape(str(query["value"])))


def paragraph_bounds(text: str, start: int, end: int, max_chars: int) -> tuple[int, int, str]:
    left_break = text.rfind("\n\n", 0, start)
    right_break = text.find("\n\n", end)
    left = 0 if left_break < 0 else left_break + 2
    right = len(text) if right_break < 0 else right_break
    if right - left <= max_chars:
        return left, right, "blank_line_paragraph"
    half = max_chars // 2
    left = max(left, start - half)
    right = min(right, max(end + half, left + max_chars))
    if right - left > max_chars:
        right = left + max_chars
    return left, right, "bounded_paragraph_window"


def document_identity(row: Mapping[str, Any], language: str) -> str:
    return str(row.get("id") or stable_id({"language": language, "url": row.get("url"), "text": row["text"]}))


def source_descriptor(
    row: Mapping[str, Any], language: str, *, upstream_split: str | None = None
) -> dict[str, Any]:
    fields = (
        "id",
        "url",
        "dump",
        "date",
        "file_path",
        "language",
        "language_score",
        "language_script",
        "top_langs",
        "minhash_cluster_size",
    )
    return {
        "dataset": FINEWEB_EN_DATASET if language == "en" else FINEWEB2_DATASET,
        "dataset_revision": FINEWEB_EN_REVISION if language == "en" else FINEWEB2_REVISION,
        "dataset_config": "sample-10BT" if language == "en" else FINEWEB2_CONFIGS[language],
        "upstream_split": upstream_split or ("train" if language == "en" else "test"),
        "declared_dataset_license": "",
        **{field: row.get(field) for field in fields if row.get(field) is not None},
    }


def exclusion_hashes(paths: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            for field in ("text_sha256", "source_text_sha256", "source_document_sha256"):
                value = row.get(field)
                if isinstance(value, str) and value:
                    excluded.add(value.removeprefix("sha256:"))
            text = row.get("text")
            if isinstance(text, str):
                excluded.add(sha256_text(text))
    return excluded


def retrieve_candidates(
    plan: Mapping[str, Any],
    *,
    languages: Sequence[str],
    quota_per_cell: int,
    max_documents_per_language: int,
    shuffle_buffer: int,
    seed: int,
    max_context_chars: int,
    adjacent_context_chars: int,
    splitter: Any,
    excluded_hashes: set[str],
    conditions: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise RetrievalError(f"expected {PLAN_SCHEMA}, got {plan.get('schema')!r}")
    if adjacent_context_chars < 0:
        raise RetrievalError("adjacent-context-chars must be nonnegative")
    if any(
        value <= 0
        for value in (quota_per_cell, max_documents_per_language, shuffle_buffer, max_context_chars)
    ):
        raise RetrievalError("retrieval limits must be positive")
    selected_languages = list(dict.fromkeys(languages))
    selected_conditions = set(conditions)
    available_conditions = {query["condition"] for query in plan["queries"]}
    unknown_conditions = sorted(selected_conditions - available_conditions)
    if unknown_conditions:
        raise RetrievalError("plan has no queries for conditions: " + ", ".join(unknown_conditions))
    query_rows = [
        query
        for query in plan["queries"]
        if query["language"] in selected_languages
        and (not selected_conditions or query["condition"] in selected_conditions)
    ]
    queries_by_language: dict[str, list[tuple[dict[str, Any], re.Pattern[str]]]] = defaultdict(list)
    for query in query_rows:
        queries_by_language[query["language"]].append((query, compile_query(query)))
    missing = sorted(set(selected_languages) - set(queries_by_language))
    if missing:
        raise RetrievalError("plan has no queries for languages: " + ", ".join(missing))

    outputs: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    documents_scanned: Counter[str] = Counter()
    documents_matched: Counter[str] = Counter()
    duplicates = Counter()
    output_hashes: set[str] = set()
    provenance = segmenter_provenance(splitter)
    for language in selected_languages:
        active = queries_by_language[language]
        stream_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{language}".encode()).digest()[:8], "big")
        for source_row in fineweb_stream(language, seed=stream_seed, shuffle_buffer=shuffle_buffer):
            documents_scanned[language] += 1
            if documents_scanned[language] > max_documents_per_language:
                break
            text = source_row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            document_hash = sha256_text(text)
            if document_hash in excluded_hashes:
                duplicates["excluded_document"] += 1
                continue
            matches = []
            for query, pattern in active:
                if counts[language, query["condition"]] >= quota_per_cell:
                    continue
                match = pattern.search(text)
                if match:
                    matches.append((query, match.start(), match.end()))
            if not matches:
                if all(counts[language, query["condition"]] >= quota_per_cell for query, _ in active):
                    break
                continue
            documents_matched[language] += 1
            by_window: dict[tuple[int, int, str], list[tuple[dict[str, Any], int, int]]] = defaultdict(list)
            for query, start, end in matches:
                bounds = paragraph_bounds(text, start, end, max_context_chars)
                by_window[bounds].append((query, start, end))
            for (window_start, window_end, window_kind), window_matches in sorted(by_window.items()):
                window = text[window_start:window_end]
                spans = sentence_spans(window, [], splitter)
                for local_start, local_end in spans:
                    document_start = window_start + local_start
                    document_end = window_start + local_end
                    contained = [
                        (query, start, end)
                        for query, start, end in window_matches
                        if document_start <= start
                        and end <= document_end
                        and counts[language, query["condition"]] < quota_per_cell
                    ]
                    if not contained:
                        continue
                    sentence = text[document_start:document_end]
                    sentence_hash = sha256_text(sentence)
                    if sentence_hash in excluded_hashes or sentence_hash in output_hashes:
                        duplicates["excluded_or_duplicate_sentence"] += 1
                        continue
                    output_hashes.add(sentence_hash)
                    source_id = document_identity(source_row, language)
                    query_matches = []
                    newly_filled_conditions: set[str] = set()
                    for query, start, end in contained:
                        newly_filled_conditions.add(query["condition"])
                        query_matches.append(
                            {
                                "query_id": query["query_id"],
                                "cell_id": query["cell_id"],
                                "condition": query["condition"],
                                "kind": query["kind"],
                                "value": query["value"],
                                "document_start": start,
                                "document_end": end,
                                "sentence_start": start - document_start,
                                "sentence_end": end - document_start,
                                "matched_text": text[start:end],
                            }
                        )
                    for condition in newly_filled_conditions:
                        counts[language, condition] += 1
                    intended_split = "development" if int(document_hash[:8], 16) % 5 == 0 else "train"
                    outputs.append(
                        {
                            "schema": CANDIDATE_SCHEMA,
                            "id": stable_id(
                                {
                                    "dataset": source_descriptor(source_row, language)["dataset"],
                                    "document_id": source_id,
                                    "start": document_start,
                                    "end": document_end,
                                }
                            ),
                            "language": language,
                            "bcp47": language,
                            "text": sentence,
                            "text_sha256": sentence_hash,
                            "source_document_id": source_id,
                            "source_document_sha256": document_hash,
                            "source_document_start": document_start,
                            "source_document_end": document_end,
                            "source_context_before": text[
                                max(0, document_start - adjacent_context_chars) : document_start
                            ],
                            "source_context_after": text[
                                document_end : min(len(text), document_end + adjacent_context_chars)
                            ],
                            "source_window_kind": window_kind,
                            "intended_split": intended_split,
                            "retrieval_matches": query_matches,
                            "source": source_descriptor(source_row, language),
                            "segmentation": provenance,
                            "supervision_scope": "unlabeled_retrieval_candidate_only",
                            "proof_data_used": False,
                        }
                    )
            if all(counts[language, query["condition"]] >= quota_per_cell for query, _ in active):
                break
    requested_cells = sorted({(query["language"], query["condition"]) for query in query_rows})
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "proof_data_used": False,
        "role": "nonproof_targeted_annotation_candidates",
        "plan_schema": plan["schema"],
        "configuration": {
            "languages": selected_languages,
            "quota_per_cell": quota_per_cell,
            "max_documents_per_language": max_documents_per_language,
            "shuffle_buffer": shuffle_buffer,
            "seed": seed,
            "max_context_chars": max_context_chars,
            "adjacent_context_chars": adjacent_context_chars,
            "conditions": sorted(selected_conditions) if selected_conditions else "all_plan_conditions",
            "upstream_dataset_license_field": "empty at both pinned revisions; source URLs and dataset pins retained",
        },
        "segmentation": provenance,
        "counts": {
            "candidate_sentences": len(outputs),
            "source_documents": len({row["source_document_sha256"] for row in outputs}),
            "by_language": dict(sorted(Counter(row["language"] for row in outputs).items())),
            "by_split": dict(sorted(Counter(row["intended_split"] for row in outputs).items())),
            "documents_scanned_by_language": dict(sorted(documents_scanned.items())),
            "documents_matched_by_language": dict(sorted(documents_matched.items())),
            "duplicates": dict(sorted(duplicates.items())),
            "cell_yield": [
                {
                    "language": language,
                    "condition": condition,
                    "requested": quota_per_cell,
                    "retrieved": counts[language, condition],
                }
                for language, condition in requested_cells
            ],
        },
        "admission": "none; teacher annotation and independent review are required",
    }
    return outputs, receipt


def cmd_plan(args: Any) -> dict[str, Any]:
    plan = plan_retrieval(
        args.corpus,
        split=args.split,
        minimum_examples=args.minimum_examples,
        minimum_distinct=args.minimum_distinct,
        queries_per_cell=args.queries_per_cell,
    )
    write_new_json(args.output, plan)
    return {
        "kind": "planned",
        "cells": len(plan["cells"]),
        "queries": len(plan["queries"]),
        "output": str(args.output),
    }


def cmd_retrieve(args: Any) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    languages = args.language or sorted({query["language"] for query in plan.get("queries", [])})
    splitter = SaTCharacterSpanSplitter(
        args.segmenter_model,
        model_revision=args.segmenter_revision,
        batch_size=args.segmenter_batch_size,
        device=args.segmenter_device,
    )
    rows, receipt = retrieve_candidates(
        plan,
        languages=languages,
        quota_per_cell=args.quota_per_cell,
        max_documents_per_language=args.max_documents_per_language,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_context_chars=args.max_context_chars,
        adjacent_context_chars=args.adjacent_context_chars,
        splitter=splitter,
        excluded_hashes=exclusion_hashes(args.exclude_jsonl),
        conditions=args.condition,
    )
    write_new_text(args.output, jsonl_text(rows))
    receipt["plan"] = {"path": str(args.plan), "sha256": sha256_file(args.plan)}
    receipt["output"] = {"path": str(args.output), "sha256": sha256_file(args.output), "rows": len(rows)}
    write_new_json(args.receipt, receipt)
    return {"kind": "retrieved", "rows": len(rows), "output": str(args.output), "receipt": str(args.receipt)}


def prepare_annotation_packets(
    candidate_path: Path,
    output_directory: Path,
    receipt_path: Path,
    *,
    conditions: Sequence[str],
) -> dict[str, Any]:
    selected_conditions = set(conditions)
    if not selected_conditions:
        raise RetrievalError("annotation packet requires at least one --condition")
    candidates = read_jsonl(candidate_path)
    available_conditions = {
        match.get("condition")
        for row in candidates
        for match in row.get("retrieval_matches", [])
        if isinstance(match, Mapping)
    }
    missing = sorted(selected_conditions - available_conditions)
    if missing:
        raise RetrievalError("candidate input has no rows for conditions: " + ", ".join(missing))
    selected: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    for row in candidates:
        if row.get("schema") != CANDIDATE_SCHEMA or row.get("proof_data_used") is not False:
            raise RetrievalError("annotation packet accepts only explicit non-proof retrieval candidates")
        matched_conditions = {
            match.get("condition") for match in row.get("retrieval_matches", []) if isinstance(match, Mapping)
        }
        admitted_conditions = sorted(matched_conditions & selected_conditions)
        if not admitted_conditions:
            continue
        text = row.get("text")
        language = row.get("language")
        if not isinstance(text, str) or not text or sha256_text(text) != row.get("text_sha256"):
            raise RetrievalError("candidate text is missing or does not match text_sha256")
        if not isinstance(language, str) or not language:
            raise RetrievalError("candidate language is missing")
        document_id = str(row["source_document_id"])
        condition_counts.update(admitted_conditions)
        selected.append(
            {
                "id": f"ont3-targeted:{row['id']}",
                "text": text,
                "text_sha256": row["text_sha256"],
                "lang": language,
                "source_group_id": document_id,
                "annotation_document_id": document_id,
                "annotation_domain": "fineweb_targeted_surface",
                "intended_split": row["intended_split"],
                "source_document_sha256": row["source_document_sha256"],
                "source_document_start": row["source_document_start"],
                "source_document_end": row["source_document_end"],
                "source_context_before": row["source_context_before"],
                "source_context_after": row["source_context_after"],
                "retrieval_conditions": admitted_conditions,
                "retrieval_matches": row["retrieval_matches"],
                "retrieval_source": row["source"],
                "retrieval_candidate_id": row["id"],
                "proof_data_used": False,
                "supervision_scope": "unlabeled_teacher_annotation_input",
            }
        )
    if not selected:
        raise RetrievalError("condition filter selected no annotation candidates")
    selected.sort(
        key=lambda row: (
            row["lang"],
            row["source_group_id"],
            row["source_document_start"],
            row["id"],
        )
    )
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_language[row["lang"]].append(row)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for language, rows in sorted(by_language.items()):
        path = output_directory / f"{language}.jsonl"
        write_new_text(path, jsonl_text(rows))
        outputs.append(
            {"language": language, "path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        )
    receipt = {
        "schema": "pii-ont3-targeted-annotation-packet/v1",
        "proof_data_used": False,
        "role": "unlabeled_nonproof_teacher_annotation_input",
        "candidate_input": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "rows": len(candidates),
        },
        "condition_filter": sorted(selected_conditions),
        "counts": {
            "rows": len(selected),
            "source_documents": len({row["source_document_sha256"] for row in selected}),
            "by_language": dict(sorted(Counter(row["lang"] for row in selected).items())),
            "by_split": dict(sorted(Counter(row["intended_split"] for row in selected).items())),
            "by_retrieval_condition": dict(sorted(condition_counts.items())),
        },
        "outputs": outputs,
        "admission": "none; primary and successor annotation plus independent review are required",
    }
    write_new_json(receipt_path, receipt)
    return receipt


def normalized_surface_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def excluded_surface_source_hashes(paths: Sequence[Path], *, split: str) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("split") != split:
                continue
            source_hash = row.get("source_text_sha256")
            if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
                raise RetrievalError(f"{path}: {split!r} row lacks a source_text_sha256")
            hashes.add(source_hash)
    return hashes


def _diverse_primary_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    maximum: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: row["rank"])
    first_by_surface: dict[str, dict[str, Any]] = {}
    for row in ordered:
        first_by_surface.setdefault(row["surface_identity"], row)
    selected = sorted(first_by_surface.values(), key=lambda row: row["rank"])[:maximum]
    selected_ids = {row["identity"] for row in selected}
    if len(selected) < maximum:
        selected.extend(row for row in ordered if row["identity"] not in selected_ids)
    return selected[:maximum]


def _gold_primary_candidates(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    cells: set[tuple[str, str]],
    excluded_hashes: set[str],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]], Counter[str]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_source_spans: set[tuple[str, int, int, str]] = set()
    source_shards: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    selected_languages = {language for language, _tag in cells}
    base = manifest_path.parent
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping) or not shard.get("gold_path"):
            continue
        shard_language = shard.get("lang")
        if shard_language not in selected_languages:
            continue
        path = base / str(shard["gold_path"])
        actual_sha256 = sha256_file(path)
        if actual_sha256 != shard.get("gold_sha256"):
            raise RetrievalError(f"gold shard hash mismatch: {path}")
        source_shards.append(
            {
                "dataset": shard.get("dataset"),
                "split": shard.get("split"),
                "language": shard_language,
                "path": str(path),
                "sha256": actual_sha256,
            }
        )
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise RetrievalError(f"{path}:{line_number}: blank gold row")
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise RetrievalError(f"{path}:{line_number}: gold row must be an object")
                text = row.get("text")
                row_id = row.get("id")
                language = row.get("lang")
                spans = row.get("spans")
                if (
                    not isinstance(text, str)
                    or not text
                    or not isinstance(row_id, str)
                    or not row_id
                    or language != shard_language
                    or not isinstance(spans, list)
                ):
                    raise RetrievalError(f"{path}:{line_number}: malformed gold row")
                source_hash = sha256_text(text)
                if source_hash in excluded_hashes:
                    rejected["fixed_audit_source_hash"] += 1
                    continue
                for span_index, span in enumerate(spans):
                    if not isinstance(span, Mapping):
                        raise RetrievalError(f"{path}:{line_number}: span {span_index} is not an object")
                    tag = span.get("type")
                    cell = (language, tag)
                    if cell not in cells:
                        continue
                    start = span.get("start")
                    end = span.get("end")
                    if (
                        not isinstance(start, int)
                        or isinstance(start, bool)
                        or not isinstance(end, int)
                        or isinstance(end, bool)
                        or not 0 <= start < end <= len(text)
                    ):
                        raise RetrievalError(f"{path}:{line_number}: invalid target span {span_index}")
                    surface_identity = normalized_surface_identity(text[start:end])
                    if not surface_identity:
                        rejected["empty_normalized_surface"] += 1
                        continue
                    source_span_key = (source_hash, start, end, str(tag))
                    if source_span_key in seen_source_spans:
                        rejected["duplicate_source_span"] += 1
                        continue
                    seen_source_spans.add(source_span_key)
                    identity = stable_id(
                        {"source_sha256": source_hash, "start": start, "end": end, "tag": tag},
                        length=32,
                    )
                    candidates[cell].append(
                        {
                            "identity": identity,
                            "rank": sha256_text(f"{language}\0{tag}\0{surface_identity}\0{identity}"),
                            "surface_identity": surface_identity,
                            "source_hash": source_hash,
                            "source_row_id": row_id,
                            "source_line": line_number,
                            "source_path": str(path),
                            "source_path_sha256": actual_sha256,
                            "source_dataset": row.get("source_dataset", shard.get("dataset")),
                            "source_schema": row.get("source_schema"),
                            "source_split": row.get("source_split", shard.get("split")),
                            "text": text,
                            "language": language,
                            "span": {"start": start, "end": end, "type": tag},
                        }
                    )
    return candidates, source_shards, rejected


def build_gold_primary_supplement(
    manifest_path: Path,
    plan_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    tags: Sequence[str],
    languages: Sequence[str],
    exclude_corpora: Sequence[Path],
    exclude_split: str,
    max_per_cell: int,
) -> dict[str, Any]:
    if max_per_cell <= 0:
        raise RetrievalError("--max-per-cell must be positive")
    if not exclude_corpora:
        raise RetrievalError("gold supplement requires at least one fixed-audit --exclude-corpus")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "pii-ontology-v2-training-gold":
        raise RetrievalError("gold supplement requires a pii-ontology-v2-training-gold manifest")
    if plan.get("schema") != PLAN_SCHEMA:
        raise RetrievalError("gold supplement requires the frozen targeted-retrieval plan")
    plan_cells = {
        (cell.get("language"), str(cell.get("condition", "")).removeprefix("primary:"))
        for cell in plan.get("cells", [])
        if isinstance(cell, Mapping) and str(cell.get("condition", "")).startswith("primary:")
    }
    selected_languages = set(languages) or {language for language, _tag in plan_cells}
    if not selected_languages or not all(
        isinstance(language, str) and language for language in selected_languages
    ):
        raise RetrievalError("gold supplement resolved no valid languages")
    selected_tags = set(tags)
    cells = (
        {(language, tag) for language in selected_languages for tag in selected_tags}
        if selected_tags
        else {(language, tag) for language, tag in plan_cells if language in selected_languages}
    )
    if not cells:
        raise RetrievalError("gold supplement resolved no primary tag/language cells")
    excluded_hashes = excluded_surface_source_hashes(exclude_corpora, split=exclude_split)
    manifest_sha256 = sha256_file(manifest_path)
    candidates, source_shards, rejected = _gold_primary_candidates(
        manifest,
        manifest_path,
        cells=cells,
        excluded_hashes=excluded_hashes,
    )
    selected: list[dict[str, Any]] = []
    cell_counts = []
    for language, tag in sorted(cells):
        available = candidates.get((language, tag), [])
        retained = _diverse_primary_candidates(available, maximum=max_per_cell)
        selected.extend(retained)
        cell_counts.append(
            {
                "language": language,
                "tag": tag,
                "available_spans": len(available),
                "available_distinct_surfaces": len({row["surface_identity"] for row in available}),
                "retained_spans": len(retained),
                "retained_distinct_surfaces": len({row["surface_identity"] for row in retained}),
            }
        )
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in selected:
        key = (row["source_path"], row["source_line"], row["source_row_id"])
        destination = grouped.setdefault(
            key,
            {
                "schema": PRIMARY_SUPPLEMENT_SCHEMA,
                "id": "ont2-gold-surface:"
                + stable_id(
                    {
                        "manifest": manifest_sha256,
                        "source_path": row["source_path"],
                        "source_line": row["source_line"],
                        "source_row_id": row["source_row_id"],
                    }
                ),
                "source_group_id": "ont2-gold-row:"
                + stable_id(
                    {
                        "source_path": row["source_path"],
                        "source_line": row["source_line"],
                        "source_row_id": row["source_row_id"],
                    }
                ),
                "text": row["text"],
                "lang": row["language"],
                "spans": [],
                "source_gold": {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "shard_path": row["source_path"],
                    "shard_sha256": row["source_path_sha256"],
                    "line": row["source_line"],
                    "row_id": row["source_row_id"],
                    "dataset": row["source_dataset"],
                    "schema": row["source_schema"],
                    "split": row["source_split"],
                },
                "proof_data_used": False,
                "supervision_scope": (
                    "selected exact primary-span positives only; "
                    "all predicate and subclass channels unknown/masked"
                ),
            },
        )
        destination["spans"].append(row["span"])
    outputs = []
    for row in grouped.values():
        row["spans"] = sorted(
            {json.dumps(span, sort_keys=True): span for span in row["spans"]}.values(),
            key=lambda span: (span["start"], span["end"], span["type"]),
        )
        outputs.append(row)
    outputs.sort(key=lambda row: row["id"])
    if not outputs:
        raise RetrievalError("gold supplement retained no primary spans")
    write_new_text(output_path, jsonl_text(outputs))
    receipt = {
        "schema": PRIMARY_SUPPLEMENT_RECEIPT_SCHEMA,
        "proof_data_used": False,
        "role": "factor_masked_nonproof_primary_surface_training_supplement",
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "configuration": {
            "languages": sorted(selected_languages),
            "tags": sorted(selected_tags) if selected_tags else "low_support_primary_cells_from_plan",
            "max_per_language_tag_cell": max_per_cell,
            "exclude_split": exclude_split,
            "exclude_corpora": [{"path": str(path), "sha256": sha256_file(path)} for path in exclude_corpora],
            "source_grouping": "gold row identity; upstream document identity is unavailable",
        },
        "source_shards": source_shards,
        "counts": {
            "rows": len(outputs),
            "spans": sum(len(row["spans"]) for row in outputs),
            "source_groups": len({row["source_group_id"] for row in outputs}),
            "excluded_source_hashes": len(excluded_hashes),
            "by_cell": cell_counts,
            "rejected": dict(sorted(rejected.items())),
        },
        "checks": {
            "source_shard_hashes_verified": True,
            "fixed_audit_source_hashes_excluded": True,
            "only_selected_exact_positive_spans_retained": True,
            "outside_text_is_not_dense_O_supervision": True,
            "predicate_and_subclass_channels_are_unknown": True,
            "surface_diversity_precedes_repeat_sampling_within_each_cell": True,
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "admission": (
            "trustworthy existing ont2 primary positives only; use supplement train rows "
            "against the separately frozen ont3 audit"
        ),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_gold_supplement(args: Any) -> dict[str, Any]:
    receipt = build_gold_primary_supplement(
        args.manifest,
        args.plan,
        args.output,
        args.receipt,
        tags=args.tag or (),
        languages=args.language or (),
        exclude_corpora=args.exclude_corpus or (),
        exclude_split=args.exclude_split,
        max_per_cell=args.max_per_cell,
    )
    return {
        "kind": "gold_primary_supplement",
        "rows": receipt["counts"]["rows"],
        "spans": receipt["counts"]["spans"],
        "output": str(args.output),
        "receipt": str(args.receipt),
    }


def _offset_guide(text: str) -> list[list[int | str]]:
    return [
        [match.start(), match.end(), match.group()] for match in re.finditer(r"\w+(?:[’']\w+)*|[^\w\s]", text)
    ]


def _primary_review_guidance(row: Mapping[str, Any]) -> str:
    guidance = {
        "bcp47": row["lang"],
        "completion_semantics": (
            "Return every reliable positive ordinary ontology-v2 span. This is a proposal-only "
            "view: omissions and every untagged character remain unknown until independent review."
        ),
        "document_context_after": row.get("source_context_after", ""),
        "document_context_before": row.get("source_context_before", ""),
        "offset_guide": _offset_guide(str(row["text"])),
        "review_mode": (
            "proposal-blind ordinary-span annotation with document context; no prior annotation "
            "or retrieval condition is visible"
        ),
        "row_id": row["id"],
    }
    return json.dumps(guidance, ensure_ascii=False, sort_keys=True)


def prepare_primary_review_packets(
    source_directory: Path,
    exclude_directory: Path,
    output_directory: Path,
    receipt_path: Path,
    *,
    additional_exclude_directories: Sequence[Path] = (),
) -> dict[str, Any]:
    source_paths = sorted(source_directory.glob("*.jsonl"))
    if not source_paths:
        raise RetrievalError("primary review packet found no normalized language inputs")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    inputs = []
    total_source_rows = 0
    total_excluded_rows = 0
    total_review_rows = 0
    seen_ids: set[str] = set()
    for source_path in source_paths:
        language = source_path.stem
        source_rows = read_jsonl(source_path)
        source_ids = {row.get("id") for row in source_rows}
        excluded_id_set: set[str] = set()
        previous_views = []
        for directory in (exclude_directory, *additional_exclude_directories):
            exclude_path = directory / f"{language}.jsonl"
            if not exclude_path.is_file():
                continue
            excluded_rows = read_jsonl(exclude_path)
            excluded_ids = [row.get("id") for row in excluded_rows]
            if any(not isinstance(row_id, str) or not row_id for row_id in excluded_ids):
                raise RetrievalError(f"{exclude_path}: excluded row has no id")
            if len(excluded_ids) != len(set(excluded_ids)):
                raise RetrievalError(f"{exclude_path}: duplicate excluded row id")
            view_id_set = set(excluded_ids)
            if not view_id_set <= source_ids:
                raise RetrievalError(f"{language}: excluded ids are absent from normalized source")
            excluded_id_set.update(view_id_set)
            previous_views.append(
                {
                    "path": str(exclude_path),
                    "rows": len(excluded_rows),
                    "sha256": sha256_file(exclude_path),
                }
            )
        rows = []
        for source_row in source_rows:
            row_id = source_row.get("id")
            text = source_row.get("text")
            if (
                not isinstance(row_id, str)
                or not row_id
                or row_id in seen_ids
                or not isinstance(text, str)
                or not text
                or source_row.get("lang") != language
                or source_row.get("proof_data_used") is not False
            ):
                raise RetrievalError(f"{source_path}: malformed or duplicate source row {row_id!r}")
            seen_ids.add(row_id)
            if row_id in excluded_id_set:
                continue
            row = {
                key: value
                for key, value in source_row.items()
                if key not in {"retrieval_conditions", "retrieval_matches"}
            }
            row["primary_review_guidance"] = _primary_review_guidance(source_row)
            row["supervision_scope"] = "unadmitted_independent_primary_positive_proposal"
            rows.append(row)
        output_path = output_directory / f"{language}.jsonl"
        write_new_text(output_path, jsonl_text(rows))
        outputs.append(
            {
                "language": language,
                "path": str(output_path),
                "rows": len(rows),
                "sha256": sha256_file(output_path),
            }
        )
        inputs.append(
            {
                "language": language,
                "normalized_source": {
                    "path": str(source_path),
                    "rows": len(source_rows),
                    "sha256": sha256_file(source_path),
                },
                "previous_view": previous_views[0] if previous_views else None,
                "additional_previous_views": previous_views[1:],
            }
        )
        total_source_rows += len(source_rows)
        total_excluded_rows += len(excluded_id_set)
        total_review_rows += len(rows)
    receipt = {
        "schema": "pii-ont3-targeted-primary-review-packet/v1",
        "proof_data_used": False,
        "role": "context_bearing_proposal_blind_independent_primary_input",
        "inputs": inputs,
        "counts": {
            "source_rows": total_source_rows,
            "excluded_previously_processed_rows": total_excluded_rows,
            "review_rows": total_review_rows,
        },
        "outputs": outputs,
        "checks": {
            "excluded_ids_are_source_subset": True,
            "prior_primary_proposals_withheld": True,
            "retrieval_conditions_withheld": True,
            "document_context_retained_in_guidance": True,
            "proof_rows_opened": 0,
        },
        "admission": (
            "none; exact cross-view positives require explicit root review and weights, while "
            "omissions, disagreements, and untagged text remain unknown"
        ),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_primary_review_packet(args: Any) -> dict[str, Any]:
    receipt = prepare_primary_review_packets(
        args.source_directory,
        args.exclude_directory,
        args.output_directory,
        args.receipt,
        additional_exclude_directories=args.additional_exclude_directory or (),
    )
    return {
        "kind": "primary_review_packet",
        "rows": receipt["counts"]["review_rows"],
        "excluded": receipt["counts"]["excluded_previously_processed_rows"],
        "output_directory": str(args.output_directory),
        "receipt": str(args.receipt),
    }


def _primary_prediction_spans(
    text: str,
    row: Mapping[str, Any],
    *,
    path: Path,
) -> list[dict[str, Any]]:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise RetrievalError(f"{path}: primary view row has no id")
    parse_stats = row.get("parse_stats")
    if parse_stats is not None and (
        not isinstance(parse_stats, Mapping)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value != 0
            for value in parse_stats.values()
        )
    ):
        raise RetrievalError(f"{path}: selected prediction has nonzero parse failures for {row_id}")
    spans = []
    for index, prediction in enumerate(row.get("preds", [])):
        if not isinstance(prediction, Mapping):
            raise RetrievalError(f"{path}: {row_id} prediction {index} is not an object")
        start = prediction.get("start")
        end = prediction.get("end")
        label = prediction.get("label")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(label, str)
            or not label
            or not 0 <= start < end <= len(text)
        ):
            raise RetrievalError(f"{path}: invalid prediction for {row_id}")
        spans.append({"start": start, "end": end, "type": label, "surface": text[start:end]})
    ordered = sorted(spans, key=lambda span: (span["start"], span["end"], span["type"]))
    if ordered != spans:
        raise RetrievalError(f"{path}: predictions are out of order for {row_id}")
    if len({(span["start"], span["end"], span["type"]) for span in spans}) != len(spans):
        raise RetrievalError(f"{path}: duplicate prediction for {row_id}")
    if any(left["end"] > right["start"] for left, right in zip(spans, spans[1:])):
        raise RetrievalError(f"{path}: predictions overlap for {row_id}")
    return spans


def prepare_primary_intersection_review(
    source_directory: Path,
    reference_directory: Path,
    independent_directories: Sequence[Path],
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if not independent_directories:
        raise RetrievalError("primary intersection review requires an independent directory")
    source_paths = sorted(source_directory.glob("*.jsonl"))
    if not source_paths:
        raise RetrievalError("primary intersection review found no normalized language inputs")
    review_rows = []
    inputs = []
    exact_by_language: Counter[str] = Counter()
    exact_by_type: Counter[str] = Counter()
    independent_rows_by_origin: Counter[str] = Counter()
    source_row_count = 0
    quarantined_rows = 0
    for source_path in source_paths:
        language = source_path.stem
        source_rows = read_jsonl(source_path)
        source_ids = [row.get("id") for row in source_rows]
        if any(not isinstance(row_id, str) or not row_id for row_id in source_ids):
            raise RetrievalError(f"{source_path}: source row has no id")
        if len(source_ids) != len(set(source_ids)):
            raise RetrievalError(f"{source_path}: duplicate source row id")
        reference_path = reference_directory / f"{language}.jsonl"
        reference_rows = read_jsonl(reference_path)
        reference_by_id = {row.get("id"): row for row in reference_rows}
        if len(reference_by_id) != len(reference_rows) or set(source_ids) != set(reference_by_id):
            raise RetrievalError(f"{language}: reference view ids do not match normalized source")
        independent_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
        independent_inputs = []
        for directory in independent_directories:
            independent_path = directory / f"{language}.jsonl"
            if not independent_path.is_file():
                continue
            rows = read_jsonl(independent_path)
            for row in rows:
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id:
                    raise RetrievalError(f"{independent_path}: independent row has no id")
                if row_id in independent_by_id:
                    raise RetrievalError(f"{language}: duplicate independent row id {row_id}")
                independent_by_id[row_id] = (row, independent_path)
                independent_rows_by_origin[str(independent_path.parent)] += 1
            independent_inputs.append(
                {
                    "path": str(independent_path),
                    "rows": len(rows),
                    "sha256": sha256_file(independent_path),
                }
            )
        if set(source_ids) != set(independent_by_id):
            missing = sorted(set(source_ids) - set(independent_by_id))
            extra = sorted(set(independent_by_id) - set(source_ids))
            raise RetrievalError(
                f"{language}: independent union does not match normalized source; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        for source_row in source_rows:
            row_id = str(source_row["id"])
            text = source_row.get("text")
            if (
                not isinstance(text, str)
                or not text
                or source_row.get("lang") != language
                or source_row.get("proof_data_used") is not False
            ):
                raise RetrievalError(f"{source_path}: malformed source row {row_id}")
            reference_row = reference_by_id[row_id]
            independent_row, independent_path = independent_by_id[row_id]
            reference_banned = reference_row.get("annotation_banned") is True
            independent_banned = independent_row.get("annotation_banned") is True
            reference_spans = (
                []
                if reference_banned
                else _primary_prediction_spans(text, reference_row, path=reference_path)
            )
            independent_spans = (
                []
                if independent_banned
                else _primary_prediction_spans(text, independent_row, path=independent_path)
            )
            reference_keys = {(span["start"], span["end"], span["type"]): span for span in reference_spans}
            independent_keys = {
                (span["start"], span["end"], span["type"]): span for span in independent_spans
            }
            exact_keys = sorted(set(reference_keys) & set(independent_keys))
            if reference_banned or independent_banned:
                quarantined_rows += 1
                exact_keys = []
            if exact_keys:
                intersections = [reference_keys[key] for key in exact_keys]
                exact_by_language[language] += len(intersections)
                exact_by_type.update(span["type"] for span in intersections)
                review_rows.append(
                    {
                        "schema": "pii-ont3-primary-exact-intersection-review/v1",
                        "id": row_id,
                        "language": language,
                        "bcp47": source_row.get("bcp47", language),
                        "text": text,
                        "source_context_before": source_row.get("source_context_before", ""),
                        "source_context_after": source_row.get("source_context_after", ""),
                        "independent_origin": str(independent_path.parent),
                        "exact_intersections": intersections,
                        "reference_only": [
                            reference_keys[key] for key in sorted(set(reference_keys) - set(independent_keys))
                        ],
                        "independent_only": [
                            independent_keys[key]
                            for key in sorted(set(independent_keys) - set(reference_keys))
                        ],
                        "proof_data_used": False,
                        "review_status": "unreviewed",
                        "supervision_scope": "unadmitted_exact_positive_candidates",
                    }
                )
        source_row_count += len(source_rows)
        inputs.append(
            {
                "language": language,
                "source": {
                    "path": str(source_path),
                    "rows": len(source_rows),
                    "sha256": sha256_file(source_path),
                },
                "reference_view": {
                    "path": str(reference_path),
                    "rows": len(reference_rows),
                    "sha256": sha256_file(reference_path),
                },
                "independent_views": independent_inputs,
            }
        )
    write_new_text(output_path, jsonl_text(review_rows))
    receipt = {
        "schema": "pii-ont3-primary-exact-intersection-review-receipt/v1",
        "proof_data_used": False,
        "role": "unadmitted_literal_review_packet",
        "inputs": inputs,
        "counts": {
            "source_rows": source_row_count,
            "candidate_rows": len(review_rows),
            "exact_intersection_spans": sum(exact_by_type.values()),
            "exact_intersection_spans_by_language": dict(sorted(exact_by_language.items())),
            "exact_intersection_spans_by_type": dict(sorted(exact_by_type.items())),
            "independent_rows_by_origin": dict(sorted(independent_rows_by_origin.items())),
            "quarantined_rows": quarantined_rows,
        },
        "checks": {
            "source_and_reference_id_sets_equal": True,
            "source_and_independent_union_id_sets_equal": True,
            "independent_row_ids_unique_across_origins": True,
            "exact_boundaries_types_and_surfaces_equal": True,
            "proof_rows_opened": 0,
        },
        "output": {
            "path": str(output_path),
            "rows": len(review_rows),
            "sha256": sha256_file(output_path),
        },
        "admission": "none; every exact candidate requires literal root adjudication",
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_primary_intersection_review(args: Any) -> dict[str, Any]:
    receipt = prepare_primary_intersection_review(
        args.source_directory,
        args.reference_directory,
        args.independent_directory or (),
        args.output,
        args.receipt,
    )
    return {
        "kind": "primary_intersection_review",
        "rows": receipt["counts"]["candidate_rows"],
        "spans": receipt["counts"]["exact_intersection_spans"],
        "output": str(args.output),
        "receipt": str(args.receipt),
    }


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RetrievalError(f"{path}: invalid {name} JSON: {error}") from None
    if not isinstance(value, dict):
        raise RetrievalError(f"{path}: {name} must be an object")
    return value


def materialize_primary_intersection_review(
    source_directory: Path,
    review_packet_path: Path,
    adjudication_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Materialize only root-accepted exact intersections as partial positives."""
    if not run_id:
        raise RetrievalError("primary intersection materialization requires AGENTCTL_RUN_ID")
    from scripts.pii_ontology_v2 import load_ontology

    adjudication = _load_json_object(adjudication_path, name="root adjudication")
    if adjudication.get("schema") != PRIMARY_INTERSECTION_ADJUDICATION_SCHEMA:
        raise RetrievalError(f"{adjudication_path}: unexpected root-adjudication schema")
    if adjudication.get("status") != "root_review_complete_materialization_authorized":
        raise RetrievalError(f"{adjudication_path}: root review has not authorized materialization")
    if adjudication.get("proof_data_used") is not False:
        raise RetrievalError(f"{adjudication_path}: proof_data_used must be false")
    packet_declaration = adjudication.get("packet")
    review_contract = adjudication.get("review_contract")
    admission = adjudication.get("admission")
    if not all(isinstance(value, Mapping) for value in (packet_declaration, review_contract, admission)):
        raise RetrievalError(f"{adjudication_path}: missing packet, review_contract, or admission")
    packet_hash = sha256_file(review_packet_path)
    if packet_hash != packet_declaration.get("sha256"):
        raise RetrievalError(
            f"{review_packet_path}: SHA-256 {packet_hash} differs from root-adjudicated packet"
        )
    if review_contract.get("default_exact_intersection_decision") != "accept":
        raise RetrievalError(f"{adjudication_path}: materializer supports accept-except review only")
    objective_weight = review_contract.get("objective_weight")
    if (
        isinstance(objective_weight, bool)
        or not isinstance(objective_weight, (int, float))
        or not 0 < objective_weight <= 1
    ):
        raise RetrievalError(f"{adjudication_path}: objective_weight must be in (0, 1]")
    if (
        admission.get("supervision") != ANNOTATED_SPANS_ONLY
        or admission.get("label_space") != "v2"
        or admission.get("outside_span_objective_weight") != 0.0
        or admission.get("predicate_and_subclass_objective_weight") != 0.0
    ):
        raise RetrievalError(f"{adjudication_path}: unsupported admission contract")

    rejected: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for index, decision in enumerate(adjudication.get("rejected_exact_intersections", [])):
        if not isinstance(decision, Mapping):
            raise RetrievalError(f"{adjudication_path}: rejection {index} is not an object")
        key = (
            decision.get("id"),
            decision.get("start"),
            decision.get("end"),
            decision.get("type"),
        )
        if (
            not isinstance(key[0], str)
            or not key[0]
            or not isinstance(key[1], int)
            or isinstance(key[1], bool)
            or not isinstance(key[2], int)
            or isinstance(key[2], bool)
            or not isinstance(key[3], str)
            or not key[3]
        ):
            raise RetrievalError(f"{adjudication_path}: malformed rejection {index}")
        if key in rejected:
            raise RetrievalError(f"{adjudication_path}: duplicate rejection {key}")
        rejected[key] = dict(decision)

    source_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
    source_inputs = []
    for source_path in sorted(source_directory.glob("*.jsonl")):
        language = source_path.stem
        rows = read_jsonl(source_path)
        for row in rows:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise RetrievalError(f"{source_path}: source row has no id")
            if row_id in source_by_id:
                raise RetrievalError(f"{source_directory}: duplicate source row id {row_id}")
            if row.get("lang") != language or row.get("proof_data_used") is not False:
                raise RetrievalError(f"{source_path}: malformed language/proof fields for {row_id}")
            source_by_id[row_id] = (row, source_path)
        source_inputs.append(
            {
                "language": language,
                "path": str(source_path),
                "rows": len(rows),
                "sha256": sha256_file(source_path),
            }
        )
    if not source_by_id:
        raise RetrievalError("primary intersection materialization found no normalized source rows")

    allowed_labels = set(load_ontology().primary_types)
    review_rows = read_jsonl(review_packet_path)
    accepted_by_language: Counter[str] = Counter()
    accepted_by_type: Counter[str] = Counter()
    rejected_by_language: Counter[str] = Counter()
    rejected_by_type: Counter[str] = Counter()
    observed_rejections: set[tuple[str, int, int, str]] = set()
    observed_review_ids: set[str] = set()
    output_rows = []
    exact_span_count = 0
    for line_number, review_row in enumerate(review_rows, 1):
        row_id = review_row.get("id")
        language = review_row.get("language")
        text = review_row.get("text")
        if review_row.get("schema") != PRIMARY_INTERSECTION_REVIEW_SCHEMA:
            raise RetrievalError(f"{review_packet_path}:{line_number}: unexpected schema")
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in observed_review_ids
            or not isinstance(language, str)
            or not language
            or not isinstance(text, str)
            or not text
            or review_row.get("proof_data_used") is not False
        ):
            raise RetrievalError(f"{review_packet_path}:{line_number}: malformed review row")
        observed_review_ids.add(row_id)
        if row_id not in source_by_id:
            raise RetrievalError(f"{review_packet_path}:{line_number}: unknown source id {row_id}")
        source_row, source_path = source_by_id[row_id]
        if source_path.stem != language or source_row.get("text") != text:
            raise RetrievalError(f"{review_packet_path}:{line_number}: source text/language mismatch")
        exact_spans = review_row.get("exact_intersections")
        if not isinstance(exact_spans, list) or not exact_spans:
            raise RetrievalError(f"{review_packet_path}:{line_number}: no exact intersections")
        accepted_spans = []
        previous_end = 0
        seen_span_keys: set[tuple[str, int, int, str]] = set()
        for span_index, span in enumerate(exact_spans):
            if not isinstance(span, Mapping):
                raise RetrievalError(
                    f"{review_packet_path}:{line_number}: span {span_index} is not an object"
                )
            start = span.get("start")
            end = span.get("end")
            label = span.get("type")
            surface = span.get("surface")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not isinstance(label, str)
                or label not in allowed_labels
                or not 0 <= start < end <= len(text)
                or surface != text[start:end]
                or start < previous_end
            ):
                raise RetrievalError(f"{review_packet_path}:{line_number}: invalid exact span {span_index}")
            key = (row_id, start, end, label)
            if key in seen_span_keys:
                raise RetrievalError(f"{review_packet_path}:{line_number}: duplicate exact span {key}")
            seen_span_keys.add(key)
            previous_end = end
            exact_span_count += 1
            if key in rejected:
                decision = rejected[key]
                if decision.get("language") != language or decision.get("surface") != surface:
                    raise RetrievalError(f"{adjudication_path}: rejection metadata differs for {key}")
                observed_rejections.add(key)
                rejected_by_language[language] += 1
                rejected_by_type[label] += 1
                continue
            accepted_spans.append([start, end, label])
            accepted_by_language[language] += 1
            accepted_by_type[label] += 1
        if not accepted_spans:
            continue
        source_group_id = source_row.get("source_group_id")
        annotation_domain = source_row.get("annotation_domain")
        intended_split = source_row.get("intended_split")
        source_document_sha256 = source_row.get("source_document_sha256")
        source_document_start = source_row.get("source_document_start")
        source_document_end = source_row.get("source_document_end")
        bcp47 = review_row.get("bcp47")
        if (
            not isinstance(source_group_id, str)
            or not source_group_id
            or not isinstance(annotation_domain, str)
            or not annotation_domain
            or not isinstance(intended_split, str)
            or not intended_split
            or not isinstance(source_document_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_document_sha256)
            or not isinstance(source_document_start, int)
            or isinstance(source_document_start, bool)
            or not isinstance(source_document_end, int)
            or isinstance(source_document_end, bool)
            or source_document_start < 0
            or source_document_end <= source_document_start
            or source_row.get("text_sha256") != sha256_text(text)
            or not isinstance(bcp47, str)
            or not bcp47
        ):
            raise RetrievalError(f"{source_path}: {row_id} lacks complete normalized provenance")
        output_rows.append(
            {
                "schema": PRIMARY_INTERSECTION_MATERIALIZATION_SCHEMA,
                "id": f"{row_id}:primary-exact-intersection-reviewed-v1",
                "source_row_id": row_id,
                "source_group_id": source_group_id,
                "text": text,
                "text_sha256": sha256_text(text),
                "lang": language,
                "bcp47": bcp47,
                "label_space": "v2",
                "spans": accepted_spans,
                "primary_span_objective_weights": [float(objective_weight)] * len(accepted_spans),
                "supervision": ANNOTATED_SPANS_ONLY,
                "src": "ont3-targeted-primary-exact-intersection-reviewed-v1",
                "mix_source": "ont3-targeted-primary-exact-intersection-reviewed-v1",
                "surface_origin": "real_original",
                "intended_split": intended_split,
                "annotation_domain": annotation_domain,
                "source_document_sha256": source_document_sha256,
                "source_document_start": source_document_start,
                "source_document_end": source_document_end,
                "proof_data_used": False,
                "supervision_scope": "root_reviewed_exact_primary_positive_only",
                "annotation_evidence": {
                    "agentctl_run_id": run_id,
                    "review_packet_sha256": packet_hash,
                    "root_adjudication_sha256": sha256_file(adjudication_path),
                    "independent_origin": review_row.get("independent_origin"),
                    "all_nonaccepted_primary_cells": "unknown_weight_zero",
                    "predicate_and_subclass_cells": "unknown_weight_zero",
                },
            }
        )

    if observed_rejections != set(rejected):
        missing = sorted(set(rejected) - observed_rejections)
        raise RetrievalError(f"{adjudication_path}: rejections absent from review packet: {missing}")
    declared_summary = adjudication.get("summary")
    if not isinstance(declared_summary, Mapping):
        raise RetrievalError(f"{adjudication_path}: missing summary")
    actual_summary = {
        "accepted_rows": len(output_rows),
        "accepted_exact_intersections": sum(accepted_by_type.values()),
        "rejected_exact_intersections": len(observed_rejections),
        "accepted_by_language": dict(sorted(accepted_by_language.items())),
        "accepted_by_type": dict(sorted(accepted_by_type.items())),
    }
    for key, value in actual_summary.items():
        if declared_summary.get(key) != value:
            raise RetrievalError(f"{adjudication_path}: declared {key} differs from materialized value")
    if (
        packet_declaration.get("candidate_rows") != len(review_rows)
        or packet_declaration.get("exact_intersection_spans") != exact_span_count
    ):
        raise RetrievalError(f"{adjudication_path}: packet counts differ from bound review packet")
    if admission.get("authorized_rows") != len(output_rows) or admission.get("authorized_spans") != sum(
        accepted_by_type.values()
    ):
        raise RetrievalError(f"{adjudication_path}: admission counts differ from materialization")

    write_new_text(output_path, jsonl_text(output_rows))
    receipt = {
        "schema": PRIMARY_INTERSECTION_MATERIALIZATION_RECEIPT_SCHEMA,
        "proof_data_used": False,
        "agentctl_run_id": run_id,
        "inputs": {
            "normalized_sources": source_inputs,
            "review_packet": {
                "path": str(review_packet_path),
                "sha256": packet_hash,
                "rows": len(review_rows),
            },
            "root_adjudication": {
                "path": str(adjudication_path),
                "sha256": sha256_file(adjudication_path),
            },
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(output_rows),
            "spans": sum(accepted_by_type.values()),
            "spans_by_language": dict(sorted(accepted_by_language.items())),
            "spans_by_type": dict(sorted(accepted_by_type.items())),
            "surface_origin": "real_original",
        },
        "rejected": {
            "spans": len(observed_rejections),
            "spans_by_language": dict(sorted(rejected_by_language.items())),
            "spans_by_type": dict(sorted(rejected_by_type.items())),
        },
        "contract": {
            "supervision": ANNOTATED_SPANS_ONLY,
            "label_space": "v2",
            "primary_span_objective_weight": float(objective_weight),
            "outside_span_objective_weight": 0.0,
            "predicate_and_subclass_objective_weight": 0.0,
            "view_only_spans_admitted": 0,
            "dense_negative_supervision": False,
        },
        "checks": {
            "review_packet_hash_matches_adjudication": True,
            "all_root_rejections_resolved_exactly_once": True,
            "materialized_text_matches_normalized_source": True,
            "all_spans_in_bounds_ordered_nonoverlapping_and_ontology_v2": True,
            "declared_and_materialized_counts_equal": True,
            "proof_rows_opened": 0,
        },
        "admission": (
            "eligible as provenance-tracked weighted exact primary positives; all omitted, "
            "disagreed, predicate, subclass, and background cells remain unknown"
        ),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_primary_intersection_materialize(args: Any) -> dict[str, Any]:
    run_id = os.environ.get("AGENTCTL_RUN_ID", "")
    receipt = materialize_primary_intersection_review(
        args.source_directory,
        args.review_packet,
        args.adjudication,
        args.output,
        args.receipt,
        run_id=run_id,
    )
    return {
        "kind": "primary_intersection_materialization",
        "rows": receipt["output"]["rows"],
        "spans": receipt["output"]["spans"],
        "output": str(args.output),
        "receipt": str(args.receipt),
    }


def _successor_guidance(row: Mapping[str, Any], base_spans: Sequence[Mapping[str, Any]]) -> str:
    text = str(row["text"])
    guidance = {
        "base_annotation_status": (
            "mechanically healthy but fallible Luna ontology-v2 proposal; use its exact spans as "
            "candidate carriers, while independent review of both primary and successor channels "
            "remains required before training admission"
        ),
        "base_spans": list(base_spans),
        "bcp47": row["lang"],
        "completion_semantics": {
            "bernoulli_inventory": (
                "Return every reliable true role on every compatible carrier. Omission remains "
                "unknown until independent review declares the row complete."
            ),
            "categorical_omissions": (
                "A missing categorical family or name component remains unknown and masked. "
                "Use Q only for a reliably known none-of-the-listed outcome."
            ),
            "primary_reference_inventory": (
                "Sweep the complete input for person_reference and organization_reference. "
                "Omissions are not negatives until independent review accepts completeness."
            ),
        },
        "document_context_after": row.get("source_context_after", ""),
        "document_context_before": row.get("source_context_before", ""),
        "offset_guide": _offset_guide(text),
        "review_mode": (
            "complete successor proposal with no prior successor output visible; the controller "
            "will independently adjudicate both this answer and the supplied primary proposal"
        ),
        "row_id": row["id"],
    }
    return json.dumps(guidance, ensure_ascii=False, sort_keys=True)


def prepare_successor_packets(
    source_directory: Path,
    primary_directory: Path,
    output_directory: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    source_paths = sorted(source_directory.glob("*.jsonl"))
    if not source_paths:
        raise RetrievalError("successor packet found no normalized language inputs")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    input_receipts = []
    total_rows = 0
    total_base_spans = 0
    banned = []
    seen_ids: set[str] = set()
    for source_path in source_paths:
        language = source_path.stem
        primary_path = primary_directory / f"{language}.jsonl"
        source_rows = read_jsonl(source_path)
        primary_rows = read_jsonl(primary_path)
        primary_sha256 = sha256_file(primary_path)
        primary_by_id = {row.get("id"): row for row in primary_rows}
        if len(primary_by_id) != len(primary_rows):
            raise RetrievalError(f"{primary_path}: duplicate or missing proposal id")
        if {row.get("id") for row in source_rows} != set(primary_by_id):
            raise RetrievalError(f"{language}: primary proposal ids do not match normalized source ids")
        rows = []
        for source_row in source_rows:
            row_id = source_row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise RetrievalError(f"{source_path}: invalid or duplicate source id {row_id!r}")
            seen_ids.add(row_id)
            proposal = primary_by_id[row_id]
            if proposal.get("annotation_banned") is True:
                banned.append({"id": row_id, "language": language, "reason": "primary_annotation_banned"})
                continue
            text = source_row.get("text")
            if not isinstance(text, str) or not text or source_row.get("lang") != language:
                raise RetrievalError(f"{source_path}: malformed source row {row_id!r}")
            base_spans = []
            for index, prediction in enumerate(proposal.get("preds", [])):
                if not isinstance(prediction, Mapping):
                    raise RetrievalError(f"{primary_path}: {row_id} prediction {index} is not an object")
                start = prediction.get("start")
                end = prediction.get("end")
                label = prediction.get("label")
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or not isinstance(label, str)
                    or not label
                    or not 0 <= start < end <= len(text)
                ):
                    raise RetrievalError(f"{primary_path}: invalid prediction for {row_id}")
                base_spans.append({"start": start, "end": end, "surface": text[start:end], "type": label})
            base_spans.sort(key=lambda span: (span["start"], span["end"], span["type"]))
            row = dict(source_row)
            row["annotation_partition"] = source_row.get("intended_split")
            row["base_spans"] = base_spans
            row["bcp47"] = language
            row["primary_proposal_artifact"] = {
                "path": str(primary_path),
                "sha256": primary_sha256,
                "row_id": row_id,
            }
            row["successor_annotation_guidance"] = _successor_guidance(row, base_spans)
            row["supervision_scope"] = "unadmitted_primary_and_successor_teacher_proposals"
            rows.append(row)
            total_base_spans += len(base_spans)
        output_path = output_directory / f"{language}.jsonl"
        write_new_text(output_path, jsonl_text(rows))
        total_rows += len(rows)
        outputs.append(
            {
                "language": language,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "rows": len(rows),
            }
        )
        input_receipts.append(
            {
                "language": language,
                "normalized_source": {
                    "path": str(source_path),
                    "sha256": sha256_file(source_path),
                    "rows": len(source_rows),
                },
                "primary_proposals": {
                    "path": str(primary_path),
                    "sha256": primary_sha256,
                    "rows": len(primary_rows),
                },
            }
        )
    receipt = {
        "schema": "pii-ont3-targeted-successor-packet/v1",
        "proof_data_used": False,
        "role": "unadmitted_context_bearing_successor_teacher_input",
        "inputs": input_receipts,
        "counts": {
            "source_rows": len(seen_ids),
            "successor_rows": total_rows,
            "base_proposal_spans": total_base_spans,
            "banned_primary_rows": len(banned),
        },
        "banned": banned,
        "outputs": outputs,
        "checks": {
            "source_and_primary_id_sets_equal": True,
            "base_span_offsets_reconstructed_from_normalized_text": True,
            "retrieval_conditions_withheld_from_teacher_guidance": True,
            "primary_proposals_not_upgraded_to_reviewed_gold": True,
            "proof_rows_opened": 0,
        },
        "admission": "none; independent review of both primary and successor channels is required",
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_successor_packet(args: Any) -> dict[str, Any]:
    receipt = prepare_successor_packets(
        args.source_directory,
        args.primary_directory,
        args.output_directory,
        args.receipt,
    )
    return {
        "kind": "successor_packet",
        "rows": receipt["counts"]["successor_rows"],
        "base_spans": receipt["counts"]["base_proposal_spans"],
        "banned": receipt["counts"]["banned_primary_rows"],
        "output_directory": str(args.output_directory),
        "receipt": str(args.receipt),
    }


def prepare_successor_review_remainder(
    source_directory: Path,
    reference_directory: Path,
    completed_directory: Path,
    output_directory: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    source_paths = sorted(source_directory.glob("*.jsonl"))
    if not source_paths:
        raise RetrievalError("successor review remainder found no source language inputs")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    inputs = []
    reference_banned = []
    total_source_rows = 0
    total_completed_rows = 0
    total_review_rows = 0
    seen_ids: set[str] = set()
    for source_path in source_paths:
        language = source_path.stem
        reference_path = reference_directory / f"{language}.jsonl"
        source_rows = read_jsonl(source_path)
        reference_rows = read_jsonl(reference_path)
        completed_path = completed_directory / f"{language}.jsonl"
        completed_rows = read_jsonl(completed_path) if completed_path.exists() else []

        def keyed(rows: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Mapping[str, Any]]:
            result: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id or row_id in result:
                    raise RetrievalError(f"{path}: duplicate or missing row id {row_id!r}")
                result[row_id] = row
            return result

        source_by_id = keyed(source_rows, source_path)
        reference_by_id = keyed(reference_rows, reference_path)
        completed_by_id = keyed(completed_rows, completed_path)
        if set(source_by_id) != set(reference_by_id):
            raise RetrievalError(f"{language}: successor source/reference id sets differ")
        if not set(completed_by_id) <= set(source_by_id):
            raise RetrievalError(f"{language}: completed ids are not a source subset")

        review_rows = []
        completed_ids = set(completed_by_id)
        for row in source_rows:
            row_id = row["id"]
            if row_id in seen_ids:
                raise RetrievalError(f"duplicate cross-language source id {row_id!r}")
            seen_ids.add(row_id)
            if row.get("proof_data_used") is not False:
                raise RetrievalError(f"{source_path}: {row_id} is not explicitly non-proof")
            guidance_text = row.get("successor_annotation_guidance")
            if not isinstance(guidance_text, str):
                raise RetrievalError(f"{source_path}: {row_id} lacks successor guidance")
            try:
                guidance = json.loads(guidance_text)
            except json.JSONDecodeError as error:
                raise RetrievalError(f"{source_path}: {row_id} guidance: {error}") from None
            if not isinstance(guidance, Mapping) or "retrieval_conditions" in guidance:
                raise RetrievalError(f"{source_path}: {row_id} guidance exposes retrieval state")

            reference_row = reference_by_id[row_id]
            is_banned = reference_row.get("annotation_banned") is True
            if is_banned:
                if row_id in completed_ids:
                    raise RetrievalError(f"{language}: completed row {row_id} is reference-banned")
                reference_banned.append({"id": row_id, "language": language})
                continue
            parse_stats = reference_row.get("parse_stats", {})
            if not isinstance(parse_stats, Mapping) or any(parse_stats.values()):
                raise RetrievalError(f"{reference_path}: usable row {row_id} has parse defects")
            if row_id in completed_ids:
                completed_row = completed_by_id[row_id]
                completed_stats = completed_row.get("parse_stats", {})
                if (
                    completed_row.get("annotation_banned") is True
                    or not isinstance(completed_stats, Mapping)
                    or any(completed_stats.values())
                ):
                    raise RetrievalError(f"{completed_path}: completed row {row_id} is unhealthy")
                continue
            review_rows.append(row)

        output_path = output_directory / f"{language}.jsonl"
        write_new_text(output_path, jsonl_text(review_rows))
        total_source_rows += len(source_rows)
        total_completed_rows += len(completed_rows)
        total_review_rows += len(review_rows)
        outputs.append(
            {
                "language": language,
                "path": str(output_path),
                "rows": len(review_rows),
                "sha256": sha256_file(output_path),
            }
        )
        inputs.append(
            {
                "language": language,
                "source": {
                    "path": str(source_path),
                    "rows": len(source_rows),
                    "sha256": sha256_file(source_path),
                },
                "reference": {
                    "path": str(reference_path),
                    "rows": len(reference_rows),
                    "sha256": sha256_file(reference_path),
                },
                "completed": {
                    "path": str(completed_path),
                    "rows": len(completed_rows),
                    "sha256": sha256_file(completed_path) if completed_path.exists() else None,
                },
            }
        )
    receipt = {
        "schema": "pii-ont3-targeted-successor-independent-remainder-receipt/v1",
        "proof_data_used": False,
        "role": "unadmitted_proposal_blind_independent_successor_input",
        "inputs": inputs,
        "outputs": outputs,
        "reference_banned": reference_banned,
        "counts": {
            "source_rows": total_source_rows,
            "completed_paid_rows": total_completed_rows,
            "reference_banned_rows": len(reference_banned),
            "review_rows": total_review_rows,
        },
        "checks": {
            "source_and_reference_id_sets_equal": True,
            "completed_ids_are_healthy_source_subsets": True,
            "completed_reference_banned_and_review_partition_source": True,
            "retrieval_conditions_withheld_from_guidance": True,
            "reference_successor_outputs_not_rendered": True,
            "proof_rows_opened": 0,
        },
        "admission": (
            "none; exact cross-view successor assertions require literal root review, while "
            "nonmatches, omissions, and all background cells remain unknown"
        ),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_successor_review_remainder(args: Any) -> dict[str, Any]:
    receipt = prepare_successor_review_remainder(
        args.source_directory,
        args.reference_directory,
        args.completed_directory,
        args.output_directory,
        args.receipt,
    )
    return {
        "kind": "successor_review_remainder",
        "rows": receipt["counts"]["review_rows"],
        "completed": receipt["counts"]["completed_paid_rows"],
        "banned": receipt["counts"]["reference_banned_rows"],
        "output_directory": str(args.output_directory),
        "receipt": str(args.receipt),
    }


def _successor_annotation_contract(
    subclass_spec_path: Path,
) -> tuple[set[str], dict[str, tuple[set[str], set[str]]]]:
    spec = _load_json_object(subclass_spec_path, name="subclass specification")
    primary_types = spec.get("primary_output_types")
    bernoulli_channels = spec.get("bernoulli_channels")
    family_blocks = [*spec.get("families", []), *spec.get("sidecar_families", [])]
    if (
        not isinstance(primary_types, list)
        or not all(isinstance(label, str) and label for label in primary_types)
        or not isinstance(bernoulli_channels, list)
        or not isinstance(family_blocks, list)
    ):
        raise RetrievalError(f"{subclass_spec_path}: malformed successor output inventory")
    prediction_labels = set(primary_types)
    for index, channel in enumerate(bernoulli_channels):
        if not isinstance(channel, Mapping):
            raise RetrievalError(f"{subclass_spec_path}: Bernoulli channel {index} is not an object")
        name = channel.get("name")
        if not isinstance(name, str) or not name or name in prediction_labels:
            raise RetrievalError(f"{subclass_spec_path}: invalid or duplicate channel {name!r}")
        prediction_labels.add(name)
    families: dict[str, tuple[set[str], set[str]]] = {}
    for index, family in enumerate(family_blocks):
        if not isinstance(family, Mapping):
            raise RetrievalError(f"{subclass_spec_path}: family {index} is not an object")
        name = family.get("name")
        applicable_types = family.get("applicable_types")
        outcomes = family.get("outcomes")
        if (
            not isinstance(name, str)
            or not name
            or name in families
            or not isinstance(applicable_types, list)
            or not all(isinstance(label, str) and label for label in applicable_types)
            or not isinstance(outcomes, list)
            or not all(isinstance(value, str) and value for value in outcomes)
        ):
            raise RetrievalError(f"{subclass_spec_path}: malformed or duplicate family {name!r}")
        families[name] = (set(applicable_types), set(outcomes))
    return prediction_labels, families


def _healthy_successor_row(row: Mapping[str, Any], path: Path, row_id: str) -> bool:
    parse_stats = row.get("parse_stats")
    if not isinstance(parse_stats, Mapping):
        raise RetrievalError(f"{path}: successor row {row_id} lacks parse statistics")
    has_defect = any(value != 0 for value in parse_stats.values())
    is_banned = row.get("annotation_banned") is True
    if has_defect and not is_banned:
        raise RetrievalError(f"{path}: defective successor row {row_id} is not quarantined")
    return not is_banned and not has_defect


def _successor_predictions(
    text: str,
    row: Mapping[str, Any],
    path: Path,
    row_id: str,
    allowed_labels: set[str],
) -> list[dict[str, Any]]:
    raw_predictions = row.get("preds")
    if not isinstance(raw_predictions, list):
        raise RetrievalError(f"{path}: successor row {row_id} lacks predictions")
    predictions = []
    seen: set[tuple[int, int, str]] = set()
    for index, prediction in enumerate(raw_predictions):
        if not isinstance(prediction, Mapping):
            raise RetrievalError(f"{path}: {row_id} prediction {index} is not an object")
        start = prediction.get("start")
        end = prediction.get("end")
        label = prediction.get("label")
        key = (start, end, label)
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(label, str)
            or label not in allowed_labels
            or not 0 <= start < end <= len(text)
            or key in seen
        ):
            raise RetrievalError(f"{path}: malformed or duplicate prediction {index} for {row_id}")
        seen.add(key)
        predictions.append({"start": start, "end": end, "label": label, "surface": text[start:end]})
    return sorted(predictions, key=lambda item: (item["start"], item["end"], item["label"]))


def _successor_subclasses(
    text: str,
    row: Mapping[str, Any],
    path: Path,
    row_id: str,
    families: Mapping[str, tuple[set[str], set[str]]],
) -> list[dict[str, Any]]:
    raw_subclasses = row.get("subclass_spans")
    if not isinstance(raw_subclasses, list):
        raise RetrievalError(f"{path}: successor row {row_id} lacks subclass spans")
    subclasses = []
    seen: set[tuple[int, int, str, int, int, str, str]] = set()
    for index, component in enumerate(raw_subclasses):
        if not isinstance(component, Mapping):
            raise RetrievalError(f"{path}: {row_id} subclass {index} is not an object")
        carrier_start = component.get("carrier_start")
        carrier_end = component.get("carrier_end")
        primary_type = component.get("type")
        start = component.get("start")
        end = component.get("end")
        family = component.get("family")
        value = component.get("value")
        key = (carrier_start, carrier_end, primary_type, start, end, family, value)
        family_contract = families.get(family) if isinstance(family, str) else None
        if (
            not isinstance(carrier_start, int)
            or isinstance(carrier_start, bool)
            or not isinstance(carrier_end, int)
            or isinstance(carrier_end, bool)
            or not isinstance(primary_type, str)
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(family, str)
            or not isinstance(value, str)
            or family_contract is None
            or primary_type not in family_contract[0]
            or value not in family_contract[1]
            or not 0 <= carrier_start <= start < end <= carrier_end <= len(text)
            or key in seen
        ):
            raise RetrievalError(f"{path}: malformed or duplicate subclass {index} for {row_id}")
        seen.add(key)
        subclasses.append(
            {
                "carrier_start": carrier_start,
                "carrier_end": carrier_end,
                "type": primary_type,
                "start": start,
                "end": end,
                "family": family,
                "value": value,
                "carrier_surface": text[carrier_start:carrier_end],
                "surface": text[start:end],
            }
        )
    return sorted(
        subclasses,
        key=lambda item: (
            item["carrier_start"],
            item["carrier_end"],
            item["type"],
            item["family"],
            item["start"],
            item["end"],
            item["value"],
        ),
    )


def prepare_successor_intersection_review(
    source_directory: Path,
    reference_directory: Path,
    independent_directories: Sequence[Path],
    subclass_spec_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Bind exact healthy successor assertions to source text for root review."""
    if not independent_directories:
        raise RetrievalError("successor intersection review requires an independent directory")
    allowed_labels, families = _successor_annotation_contract(subclass_spec_path)
    source_paths = sorted(source_directory.glob("*.jsonl"))
    if not source_paths:
        raise RetrievalError("successor intersection review found no source language inputs")
    review_rows = []
    inputs = []
    exact_predictions_by_language: Counter[str] = Counter()
    exact_predictions_by_label: Counter[str] = Counter()
    exact_subclasses_by_language: Counter[str] = Counter()
    exact_subclasses_by_value: Counter[str] = Counter()
    independent_rows_by_origin: Counter[str] = Counter()
    source_row_count = 0
    reference_banned_rows = 0
    independent_banned_rows = 0
    seen_source_ids: set[str] = set()
    for source_path in source_paths:
        language = source_path.stem
        source_rows = read_jsonl(source_path)
        source_by_id: dict[str, dict[str, Any]] = {}
        for row in source_rows:
            row_id = row.get("id")
            if (
                not isinstance(row_id, str)
                or not row_id
                or row_id in source_by_id
                or row_id in seen_source_ids
            ):
                raise RetrievalError(f"{source_path}: duplicate or missing source row id {row_id!r}")
            if row.get("lang") != language or row.get("proof_data_used") is not False:
                raise RetrievalError(f"{source_path}: malformed language/proof fields for {row_id}")
            source_by_id[row_id] = row
            seen_source_ids.add(row_id)
        reference_path = reference_directory / f"{language}.jsonl"
        reference_rows = read_jsonl(reference_path)
        reference_by_id = {row.get("id"): row for row in reference_rows}
        if (
            len(reference_by_id) != len(reference_rows)
            or None in reference_by_id
            or set(reference_by_id) != set(source_by_id)
        ):
            raise RetrievalError(f"{language}: reference view ids do not match successor source")
        independent_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
        independent_inputs = []
        for directory in independent_directories:
            independent_path = directory / f"{language}.jsonl"
            if not independent_path.is_file():
                continue
            rows = read_jsonl(independent_path)
            for row in rows:
                row_id = row.get("id")
                if (
                    not isinstance(row_id, str)
                    or not row_id
                    or row_id in independent_by_id
                    or row_id not in source_by_id
                ):
                    raise RetrievalError(
                        f"{independent_path}: duplicate, missing, or unknown row id {row_id!r}"
                    )
                independent_by_id[row_id] = (row, independent_path)
                independent_rows_by_origin[str(independent_path.parent)] += 1
            independent_inputs.append(
                {
                    "path": str(independent_path),
                    "rows": len(rows),
                    "sha256": sha256_file(independent_path),
                }
            )
        reference_banned_ids = {
            row_id
            for row_id, row in reference_by_id.items()
            if not _healthy_successor_row(row, reference_path, str(row_id))
        }
        if set(independent_by_id) != set(source_by_id) - reference_banned_ids:
            missing = sorted((set(source_by_id) - reference_banned_ids) - set(independent_by_id))
            extra = sorted(set(independent_by_id) - (set(source_by_id) - reference_banned_ids))
            raise RetrievalError(
                f"{language}: independent union does not match non-banned reference rows; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        for row_id, source_row in source_by_id.items():
            text = source_row.get("text")
            if not isinstance(text, str) or not text:
                raise RetrievalError(f"{source_path}: malformed source text for {row_id}")
            reference_row = reference_by_id[row_id]
            if row_id in reference_banned_ids:
                reference_banned_rows += 1
                continue
            independent_row, independent_path = independent_by_id[row_id]
            if not _healthy_successor_row(independent_row, independent_path, row_id):
                independent_banned_rows += 1
                continue
            reference_predictions = _successor_predictions(
                text, reference_row, reference_path, row_id, allowed_labels
            )
            independent_predictions = _successor_predictions(
                text, independent_row, independent_path, row_id, allowed_labels
            )
            reference_subclasses = _successor_subclasses(
                text, reference_row, reference_path, row_id, families
            )
            independent_subclasses = _successor_subclasses(
                text, independent_row, independent_path, row_id, families
            )
            prediction_key = lambda item: (item["start"], item["end"], item["label"])
            subclass_key = lambda item: (
                item["carrier_start"],
                item["carrier_end"],
                item["type"],
                item["start"],
                item["end"],
                item["family"],
                item["value"],
            )
            reference_prediction_by_key = {prediction_key(item): item for item in reference_predictions}
            independent_prediction_by_key = {prediction_key(item): item for item in independent_predictions}
            reference_subclass_by_key = {subclass_key(item): item for item in reference_subclasses}
            independent_subclass_by_key = {subclass_key(item): item for item in independent_subclasses}
            exact_prediction_keys = sorted(
                set(reference_prediction_by_key) & set(independent_prediction_by_key)
            )
            exact_subclass_keys = sorted(set(reference_subclass_by_key) & set(independent_subclass_by_key))
            if not exact_prediction_keys and not exact_subclass_keys:
                continue
            exact_predictions = [reference_prediction_by_key[key] for key in exact_prediction_keys]
            exact_subclasses = [reference_subclass_by_key[key] for key in exact_subclass_keys]
            exact_predictions_by_language[language] += len(exact_predictions)
            exact_predictions_by_label.update(item["label"] for item in exact_predictions)
            exact_subclasses_by_language[language] += len(exact_subclasses)
            exact_subclasses_by_value.update(f"{item['family']}={item['value']}" for item in exact_subclasses)
            review_rows.append(
                {
                    "schema": SUCCESSOR_INTERSECTION_REVIEW_SCHEMA,
                    "id": row_id,
                    "language": language,
                    "bcp47": source_row.get("bcp47", language),
                    "text": text,
                    "source_context_before": source_row.get("source_context_before", ""),
                    "source_context_after": source_row.get("source_context_after", ""),
                    "base_spans": source_row.get("base_spans", []),
                    "independent_origin": str(independent_path.parent),
                    "exact_predictions": exact_predictions,
                    "exact_subclasses": exact_subclasses,
                    "reference_only_predictions": [
                        reference_prediction_by_key[key]
                        for key in sorted(
                            set(reference_prediction_by_key) - set(independent_prediction_by_key)
                        )
                    ],
                    "independent_only_predictions": [
                        independent_prediction_by_key[key]
                        for key in sorted(
                            set(independent_prediction_by_key) - set(reference_prediction_by_key)
                        )
                    ],
                    "reference_only_subclasses": [
                        reference_subclass_by_key[key]
                        for key in sorted(set(reference_subclass_by_key) - set(independent_subclass_by_key))
                    ],
                    "independent_only_subclasses": [
                        independent_subclass_by_key[key]
                        for key in sorted(set(independent_subclass_by_key) - set(reference_subclass_by_key))
                    ],
                    "proof_data_used": False,
                    "review_status": "unreviewed",
                    "supervision_scope": "unadmitted_exact_positive_candidates",
                }
            )
        source_row_count += len(source_rows)
        inputs.append(
            {
                "language": language,
                "source": {
                    "path": str(source_path),
                    "rows": len(source_rows),
                    "sha256": sha256_file(source_path),
                },
                "reference": {
                    "path": str(reference_path),
                    "rows": len(reference_rows),
                    "sha256": sha256_file(reference_path),
                },
                "independent": independent_inputs,
            }
        )
    write_new_text(output_path, jsonl_text(review_rows))
    receipt = {
        "schema": "pii-ont3-successor-exact-intersection-review-receipt/v1",
        "proof_data_used": False,
        "role": "unadmitted_literal_successor_review_packet",
        "inputs": {
            "languages": inputs,
            "subclass_spec": {
                "path": str(subclass_spec_path),
                "sha256": sha256_file(subclass_spec_path),
            },
        },
        "counts": {
            "source_rows": source_row_count,
            "candidate_rows": len(review_rows),
            "reference_banned_rows": reference_banned_rows,
            "independent_banned_rows": independent_banned_rows,
            "exact_predictions": sum(exact_predictions_by_label.values()),
            "exact_predictions_by_language": dict(sorted(exact_predictions_by_language.items())),
            "exact_predictions_by_label": dict(sorted(exact_predictions_by_label.items())),
            "exact_subclasses": sum(exact_subclasses_by_value.values()),
            "exact_subclasses_by_language": dict(sorted(exact_subclasses_by_language.items())),
            "exact_subclasses_by_value": dict(sorted(exact_subclasses_by_value.items())),
            "independent_rows_by_origin": dict(sorted(independent_rows_by_origin.items())),
        },
        "checks": {
            "source_and_reference_id_sets_equal": True,
            "independent_union_equals_reference_healthy_rows": True,
            "independent_row_ids_unique_across_origins": True,
            "exact_boundaries_labels_families_values_and_surfaces_equal": True,
            "terminal_quarantines_excluded": True,
            "proof_rows_opened": 0,
        },
        "output": {
            "path": str(output_path),
            "rows": len(review_rows),
            "sha256": sha256_file(output_path),
        },
        "admission": (
            "none; every exact prediction and subclass assertion requires literal root review, "
            "and omissions, disagreements, quarantines, and background remain unknown"
        ),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_successor_intersection_review(args: Any) -> dict[str, Any]:
    receipt = prepare_successor_intersection_review(
        args.source_directory,
        args.reference_directory,
        args.independent_directory or (),
        args.subclass_spec,
        args.output,
        args.receipt,
    )
    return {
        "kind": "successor_intersection_review",
        "rows": receipt["counts"]["candidate_rows"],
        "predictions": receipt["counts"]["exact_predictions"],
        "subclasses": receipt["counts"]["exact_subclasses"],
        "output": str(args.output),
        "receipt": str(args.receipt),
    }


def _maximal_nonoverlapping_references(
    references: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    excluded = []
    for reference in sorted(
        references,
        key=lambda item: (
            -(item["end"] - item["start"]),
            item["start"],
            item["end"],
            item["label"],
        ),
    ):
        candidate = dict(reference)
        if any(
            candidate["start"] < incumbent["end"] and incumbent["start"] < candidate["end"]
            for incumbent in selected
        ):
            excluded.append(candidate)
        else:
            selected.append(candidate)
    order = lambda item: (item["start"], item["end"], item["label"])
    return sorted(selected, key=order), sorted(excluded, key=order)


def materialize_successor_intersection_review(
    source_directory: Path,
    review_packet_path: Path,
    adjudication_path: Path,
    subclass_spec_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Materialize root-accepted exact successor positives with factor-local weights."""
    if not run_id:
        raise RetrievalError("successor intersection materialization requires AGENTCTL_RUN_ID")
    from scripts.pii_subclass import load_subclass_spec, validate_sequence_grammar

    spec = load_subclass_spec(subclass_spec_path)
    family_by_name = spec.family_by_name
    bernoulli_by_name = spec.bernoulli_by_name
    reference_types = set(spec.primary_output_types)
    adjudication = _load_json_object(adjudication_path, name="root adjudication")
    if adjudication.get("schema") != SUCCESSOR_INTERSECTION_ADJUDICATION_SCHEMA:
        raise RetrievalError(f"{adjudication_path}: unexpected successor adjudication schema")
    if adjudication.get("status") != "root_review_complete_materialization_authorized":
        raise RetrievalError(f"{adjudication_path}: root review has not authorized materialization")
    if adjudication.get("proof_data_used") is not False:
        raise RetrievalError(f"{adjudication_path}: proof_data_used must be false")
    packet_declaration = adjudication.get("packet")
    review_contract = adjudication.get("review_contract")
    review_summary = adjudication.get("review_summary")
    authorization = adjudication.get("materialization_authorization")
    if not all(
        isinstance(value, Mapping)
        for value in (packet_declaration, review_contract, review_summary, authorization)
    ):
        raise RetrievalError(f"{adjudication_path}: incomplete root-review contract")
    packet_hash = sha256_file(review_packet_path)
    if packet_hash != packet_declaration.get("sha256"):
        raise RetrievalError(f"{review_packet_path}: SHA-256 {packet_hash} differs from root-reviewed packet")
    if review_contract.get("default_exact_assertion_decision") != "accept":
        raise RetrievalError(f"{adjudication_path}: only accept-except review is supported")
    objective_weight = review_contract.get("objective_weight")
    carrier_weight = review_contract.get("activating_carrier_primary_objective_weight")
    if (
        isinstance(objective_weight, bool)
        or not isinstance(objective_weight, (int, float))
        or not 0 < objective_weight <= 1
        or carrier_weight != 0.0
    ):
        raise RetrievalError(f"{adjudication_path}: unsupported objective or carrier weight")
    expected_authorization = {
        "supervision": ANNOTATED_SPANS_ONLY,
        "label_space": "v2",
        "accepted_reference_primary_weight": objective_weight,
        "accepted_bernoulli_positive_weight": objective_weight,
        "accepted_trainable_subclass_weight": objective_weight,
        "activating_carrier_primary_weight": 0.0,
        "outside_span_objective_weight": 0.0,
        "known_negative_objective_weight": 0.0,
        "unasserted_factor_objective_weight": 0.0,
        "reference_form_learning_weight": 0.0,
    }
    for key, expected in expected_authorization.items():
        if authorization.get(key) != expected:
            raise RetrievalError(f"{adjudication_path}: unsupported authorization field {key}")
    ordinary_policy = authorization.get("ordinary_ont2_supervision")
    if not isinstance(ordinary_policy, str) or "do not promote candidate base spans" not in ordinary_policy:
        raise RetrievalError(f"{adjudication_path}: ordinary ont2 non-promotion is not explicit")

    prediction_rejections: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for index, item in enumerate(adjudication.get("rejected_exact_predictions", [])):
        if not isinstance(item, Mapping):
            raise RetrievalError(f"{adjudication_path}: prediction rejection {index} is not an object")
        key = (item.get("id"), item.get("start"), item.get("end"), item.get("label"))
        if (
            not isinstance(key[0], str)
            or not key[0]
            or not isinstance(key[1], int)
            or isinstance(key[1], bool)
            or not isinstance(key[2], int)
            or isinstance(key[2], bool)
            or not isinstance(key[3], str)
            or not key[3]
            or key in prediction_rejections
        ):
            raise RetrievalError(f"{adjudication_path}: malformed prediction rejection {index}")
        prediction_rejections[key] = item
    subclass_rejections: dict[tuple[str, int, int, str, int, int, str, str], Mapping[str, Any]] = {}
    for index, item in enumerate(adjudication.get("rejected_exact_subclasses", [])):
        if not isinstance(item, Mapping):
            raise RetrievalError(f"{adjudication_path}: subclass rejection {index} is not an object")
        key = (
            item.get("id"),
            item.get("carrier_start"),
            item.get("carrier_end"),
            item.get("type"),
            item.get("start"),
            item.get("end"),
            item.get("family"),
            item.get("value"),
        )
        if (
            not isinstance(key[0], str)
            or not key[0]
            or any(not isinstance(value, int) or isinstance(value, bool) for value in (*key[1:3], *key[4:6]))
            or not isinstance(key[3], str)
            or not isinstance(key[6], str)
            or not isinstance(key[7], str)
            or key in subclass_rejections
        ):
            raise RetrievalError(f"{adjudication_path}: malformed subclass rejection {index}")
        subclass_rejections[key] = item

    source_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
    source_inputs = []
    for source_path in sorted(source_directory.glob("*.jsonl")):
        language = source_path.stem
        rows = read_jsonl(source_path)
        for row in rows:
            row_id = row.get("id")
            text = row.get("text")
            if (
                not isinstance(row_id, str)
                or not row_id
                or row_id in source_by_id
                or row.get("lang") != language
                or row.get("proof_data_used") is not False
                or not isinstance(text, str)
                or not text
            ):
                raise RetrievalError(f"{source_path}: malformed or duplicate source row {row_id!r}")
            source_by_id[row_id] = (row, source_path)
        source_inputs.append(
            {
                "language": language,
                "path": str(source_path),
                "rows": len(rows),
                "sha256": sha256_file(source_path),
            }
        )
    if not source_by_id:
        raise RetrievalError("successor intersection materialization found no source rows")

    review_rows = read_jsonl(review_packet_path)
    if len(review_rows) != packet_declaration.get("rows"):
        raise RetrievalError(f"{review_packet_path}: row count differs from root review")
    observed_review_ids: set[str] = set()
    observed_prediction_rejections: set[tuple[str, int, int, str]] = set()
    observed_subclass_rejections: set[tuple[str, int, int, str, int, int, str, str]] = set()
    accepted_prediction_counts: Counter[str] = Counter()
    accepted_subclass_counts: Counter[str] = Counter()
    materialized_prediction_counts: Counter[str] = Counter()
    materialized_subclass_counts: Counter[str] = Counter()
    output_rows = []
    projection_exclusions = []
    assertion_count = 0
    accepted_assertion_count = 0

    for line_number, review_row in enumerate(review_rows, 1):
        row_id = review_row.get("id")
        language = review_row.get("language")
        text = review_row.get("text")
        if (
            review_row.get("schema") != SUCCESSOR_INTERSECTION_REVIEW_SCHEMA
            or not isinstance(row_id, str)
            or not row_id
            or row_id in observed_review_ids
            or not isinstance(language, str)
            or not language
            or not isinstance(text, str)
            or not text
            or review_row.get("proof_data_used") is not False
            or row_id not in source_by_id
        ):
            raise RetrievalError(f"{review_packet_path}:{line_number}: malformed review row")
        observed_review_ids.add(row_id)
        source_row, source_path = source_by_id[row_id]
        if source_path.stem != language or source_row.get("text") != text:
            raise RetrievalError(f"{review_packet_path}:{line_number}: source text/language mismatch")
        base_spans = source_row.get("base_spans")
        if not isinstance(base_spans, list):
            raise RetrievalError(f"{source_path}: {row_id} lacks candidate base spans")
        base_by_identity: dict[tuple[int, int, str], dict[str, Any]] = {}
        for index, span in enumerate(base_spans):
            if not isinstance(span, Mapping):
                raise RetrievalError(f"{source_path}: {row_id} base span {index} is not an object")
            identity = (span.get("start"), span.get("end"), span.get("type"))
            if (
                not isinstance(identity[0], int)
                or isinstance(identity[0], bool)
                or not isinstance(identity[1], int)
                or isinstance(identity[1], bool)
                or not isinstance(identity[2], str)
                or not 0 <= identity[0] < identity[1] <= len(text)
                or span.get("surface") != text[identity[0] : identity[1]]
                or identity in base_by_identity
            ):
                raise RetrievalError(f"{source_path}: malformed base span {index} for {row_id}")
            base_by_identity[identity] = dict(span)

        exact_predictions = review_row.get("exact_predictions")
        exact_subclasses = review_row.get("exact_subclasses")
        if not isinstance(exact_predictions, list) or not isinstance(exact_subclasses, list):
            raise RetrievalError(f"{review_packet_path}:{line_number}: exact assertions are not lists")
        assertion_count += len(exact_predictions) + len(exact_subclasses)
        accepted_predictions = []
        for item in exact_predictions:
            if not isinstance(item, Mapping):
                raise RetrievalError(f"{review_packet_path}:{line_number}: prediction is not an object")
            key = (row_id, item.get("start"), item.get("end"), item.get("label"))
            rejection = prediction_rejections.get(key)
            if rejection is not None:
                if rejection.get("surface") != item.get("surface"):
                    raise RetrievalError(f"{adjudication_path}: rejection surface differs for {key}")
                observed_prediction_rejections.add(key)
                continue
            accepted_predictions.append(dict(item))
            accepted_prediction_counts[str(item["label"])] += 1
        accepted_subclasses = []
        for item in exact_subclasses:
            if not isinstance(item, Mapping):
                raise RetrievalError(f"{review_packet_path}:{line_number}: subclass is not an object")
            key = (
                row_id,
                item.get("carrier_start"),
                item.get("carrier_end"),
                item.get("type"),
                item.get("start"),
                item.get("end"),
                item.get("family"),
                item.get("value"),
            )
            rejection = subclass_rejections.get(key)
            if rejection is not None:
                if rejection.get("surface") != item.get("surface"):
                    raise RetrievalError(f"{adjudication_path}: rejection surface differs for {key}")
                observed_subclass_rejections.add(key)
                continue
            accepted_subclasses.append(dict(item))
            accepted_subclass_counts[f"{item['family']}={item['value']}"] += 1
        accepted_assertion_count += len(accepted_predictions) + len(accepted_subclasses)

        references = [item for item in accepted_predictions if item["label"] in reference_types]
        roles = [item for item in accepted_predictions if item["label"] not in reference_types]
        if any(item["label"] not in bernoulli_by_name for item in roles):
            raise RetrievalError(f"{review_packet_path}:{line_number}: unknown Bernoulli assertion")
        projected_references, excluded_references = _maximal_nonoverlapping_references(references)
        projected_reference_by_identity = {
            (item["start"], item["end"], item["label"]): item for item in projected_references
        }
        excluded_reference_by_identity = {
            (item["start"], item["end"], item["label"]): item for item in excluded_references
        }
        for item in excluded_references:
            projection_exclusions.append(
                {
                    "id": row_id,
                    "kind": "reference",
                    "assertion": item,
                    "reason": "overlaps a longer accepted reference in the one-layer BIOES projection",
                }
            )

        groups: dict[tuple[int, int, str], dict[str, Any]] = {}

        def group_for(identity: tuple[int, int, str], *, reference: bool) -> dict[str, Any]:
            group = groups.setdefault(
                identity,
                {
                    "is_reference": reference,
                    "roles": [],
                    "subclasses": [],
                    "sidecars": [],
                },
            )
            if group["is_reference"] != reference:
                raise RetrievalError(f"{row_id}: carrier identity has conflicting reference status")
            return group

        for item in projected_references:
            identity = (item["start"], item["end"], item["label"])
            group_for(identity, reference=True)
            materialized_prediction_counts[item["label"]] += 1

        for role in roles:
            channel = bernoulli_by_name[role["label"]]
            matching_projected = [
                identity
                for identity in projected_reference_by_identity
                if identity[0] == role["start"]
                and identity[1] == role["end"]
                and identity[2] in channel.applicable_types
            ]
            matching_excluded = [
                identity
                for identity in excluded_reference_by_identity
                if identity[0] == role["start"]
                and identity[1] == role["end"]
                and identity[2] in channel.applicable_types
            ]
            matching_base = [
                identity
                for identity in base_by_identity
                if identity[0] == role["start"]
                and identity[1] == role["end"]
                and identity[2] in channel.applicable_types
            ]
            if matching_projected:
                candidates = matching_projected
            elif matching_excluded:
                projection_exclusions.append(
                    {
                        "id": row_id,
                        "kind": "bernoulli",
                        "assertion": role,
                        "reason": "activating reference was excluded by the one-layer BIOES projection",
                    }
                )
                continue
            else:
                candidates = matching_base
            if len(candidates) != 1:
                raise RetrievalError(
                    f"{row_id}: {role['label']} has {len(candidates)} exact compatible carriers"
                )
            identity = candidates[0]
            group_for(identity, reference=identity in projected_reference_by_identity)["roles"].append(role)
            materialized_prediction_counts[role["label"]] += 1

        for component in accepted_subclasses:
            identity = (
                component["carrier_start"],
                component["carrier_end"],
                component["type"],
            )
            family = family_by_name.get(component["family"])
            if family is None or identity[2] not in family.applicable_types:
                raise RetrievalError(f"{row_id}: accepted subclass has no compatible family")
            if identity[2] in reference_types:
                if identity in excluded_reference_by_identity:
                    projection_exclusions.append(
                        {
                            "id": row_id,
                            "kind": "subclass",
                            "assertion": component,
                            "reason": "activating reference was excluded by the one-layer BIOES projection",
                        }
                    )
                    continue
                if identity not in projected_reference_by_identity:
                    raise RetrievalError(f"{row_id}: reference subclass lacks an accepted carrier")
                is_reference = True
            else:
                if identity not in base_by_identity:
                    raise RetrievalError(f"{row_id}: subclass lacks its exact candidate base carrier")
                is_reference = False
            target = "sidecars" if component["family"] in spec.sidecar_family_names else "subclasses"
            group_for(identity, reference=is_reference)[target].append(component)
            materialized_subclass_counts[f"{component['family']}={component['value']}"] += 1

        for identity, group in sorted(groups.items()):
            family_components: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
            for component in group["subclasses"]:
                family_components[component["family"]].append(
                    (component["start"], component["end"], component["value"])
                )
            for family_name, components in family_components.items():
                try:
                    validate_sequence_grammar(family_by_name[family_name], components)
                except ValueError as error:
                    raise RetrievalError(f"{row_id}: {error}") from error
            predicate_spans = []
            if group["roles"]:
                attrs = {
                    role["label"]: [[identity[0], identity[1]]]
                    for role in sorted(group["roles"], key=lambda item: item["label"])
                }
                predicate_spans.append(
                    {
                        "start": identity[0],
                        "end": identity[1],
                        "type": identity[2],
                        "attrs": attrs,
                        "objective_weights": {
                            "O": 0.0,
                            "other": 0.0,
                            **{label: float(objective_weight) for label in attrs},
                        },
                    }
                )
            trainable_subclasses = [
                {
                    **{
                        key: component[key]
                        for key in (
                            "carrier_start",
                            "carrier_end",
                            "type",
                            "start",
                            "end",
                            "family",
                            "value",
                        )
                    },
                    "objective_weight": float(objective_weight),
                    "learning_weight": 1.0,
                }
                for component in sorted(
                    group["subclasses"],
                    key=lambda item: (item["family"], item["start"], item["end"], item["value"]),
                )
            ]
            sidecars = [
                {
                    **{
                        key: component[key]
                        for key in (
                            "carrier_start",
                            "carrier_end",
                            "type",
                            "start",
                            "end",
                            "family",
                            "value",
                        )
                    },
                    "evidence_weight": float(objective_weight),
                    "objective_weight": 0.0,
                    "learning_weight": 0.0,
                }
                for component in sorted(
                    group["sidecars"],
                    key=lambda item: (item["family"], item["start"], item["end"], item["value"]),
                )
            ]
            output_row = {
                "id": (f"{row_id}:successor-exact-positive-v1:{identity[0]}-{identity[1]}-{identity[2]}"),
                "label_space": "v2",
                "lang": language,
                "predicate_seed": {
                    "method": "root-reviewed-luna-terra-successor-exact-intersection",
                    "source_row_id": row_id,
                },
                "primary_span_objective_weights": [float(objective_weight) if group["is_reference"] else 0.0],
                "reviewed_materialization": {
                    "ordinary_carrier_primary_supervision": (
                        "exact_reference_positive" if group["is_reference"] else "masked"
                    ),
                    "outside_and_unasserted_factors": "unknown",
                    "root_review": str(adjudication_path),
                },
                "seed_id": stable_id(
                    {
                        "source_row_id": row_id,
                        "carrier": identity,
                        "roles": predicate_spans,
                        "subclasses": trainable_subclasses,
                        "sidecars": sidecars,
                    }
                ),
                "source_group_id": source_row.get("source_group_id", row_id),
                "source_row_id": row_id,
                "spans": [list(identity)],
                "src": "ont3-targeted-surface-successor-exact-positive-v1",
                "supervision": ANNOTATED_SPANS_ONLY,
                "surface_origin": "fineweb_retrieved_literal_nonproof",
                "text": text,
            }
            if predicate_spans:
                output_row["predicate_spans"] = predicate_spans
            if trainable_subclasses:
                output_row["subclass_spans"] = trainable_subclasses
            if sidecars:
                output_row["reference_form_spans"] = sidecars
            output_rows.append(output_row)

    if observed_prediction_rejections != set(prediction_rejections):
        missing = sorted(set(prediction_rejections) - observed_prediction_rejections)
        raise RetrievalError(f"{adjudication_path}: unobserved prediction rejections {missing[:5]}")
    if observed_subclass_rejections != set(subclass_rejections):
        missing = sorted(set(subclass_rejections) - observed_subclass_rejections)
        raise RetrievalError(f"{adjudication_path}: unobserved subclass rejections {missing[:5]}")
    declared_counts = {
        "assertions_reviewed": assertion_count,
        "accepted_assertions": accepted_assertion_count,
        "rejected_assertions": len(prediction_rejections) + len(subclass_rejections),
        "predictions_reviewed": sum(accepted_prediction_counts.values()) + len(prediction_rejections),
        "accepted_predictions": sum(accepted_prediction_counts.values()),
        "rejected_predictions": len(prediction_rejections),
        "subclasses_reviewed": sum(accepted_subclass_counts.values()) + len(subclass_rejections),
        "accepted_subclasses": sum(accepted_subclass_counts.values()),
        "rejected_subclasses": len(subclass_rejections),
    }
    if any(review_summary.get(key) != value for key, value in declared_counts.items()):
        raise RetrievalError(
            f"{adjudication_path}: declared review counts differ from packet {declared_counts}"
        )
    if packet_declaration.get("exact_predictions") != declared_counts["predictions_reviewed"]:
        raise RetrievalError(f"{adjudication_path}: packet prediction count differs")
    if packet_declaration.get("exact_subclasses") != declared_counts["subclasses_reviewed"]:
        raise RetrievalError(f"{adjudication_path}: packet subclass count differs")
    if not output_rows:
        raise RetrievalError("successor intersection materialization produced no positive rows")

    write_new_text(output_path, jsonl_text(output_rows))
    projection_by_kind = Counter(item["kind"] for item in projection_exclusions)
    materialized_assertions = sum(materialized_prediction_counts.values()) + sum(
        materialized_subclass_counts.values()
    )
    if materialized_assertions + len(projection_exclusions) != accepted_assertion_count:
        raise RetrievalError("successor materialization did not account for every accepted assertion")
    receipt = {
        "schema": SUCCESSOR_INTERSECTION_MATERIALIZATION_RECEIPT_SCHEMA,
        "agentctl_run_id": run_id,
        "proof_data_used": False,
        "inputs": {
            "source": source_inputs,
            "review_packet": {
                "path": str(review_packet_path),
                "rows": len(review_rows),
                "sha256": packet_hash,
            },
            "root_adjudication": {
                "path": str(adjudication_path),
                "sha256": sha256_file(adjudication_path),
            },
            "subclass_spec": {
                "path": str(subclass_spec_path),
                "sha256": sha256_file(subclass_spec_path),
            },
        },
        "review": {
            **declared_counts,
            "accepted_predictions_by_label": dict(sorted(accepted_prediction_counts.items())),
            "accepted_subclasses_by_value": dict(sorted(accepted_subclass_counts.items())),
        },
        "materialized": {
            "output_rows": len(output_rows),
            "assertions": materialized_assertions,
            "predictions_by_label": dict(sorted(materialized_prediction_counts.items())),
            "subclasses_by_value": dict(sorted(materialized_subclass_counts.items())),
            "projection_exclusions": len(projection_exclusions),
            "projection_exclusions_by_kind": dict(sorted(projection_by_kind.items())),
            "projection_exclusion_details": projection_exclusions,
        },
        "objective_weights": {
            "accepted_reference_primary": float(objective_weight),
            "accepted_bernoulli_positive": float(objective_weight),
            "accepted_trainable_subclass": float(objective_weight),
            "activating_ordinary_carrier_primary": 0.0,
            "outside_span": 0.0,
            "known_negative": 0.0,
            "unasserted_factor": 0.0,
            "reference_form_learning": 0.0,
        },
        "checks": {
            "every_root_rejection_observed_exactly_once": True,
            "every_root_accepted_assertion_materialized_or_projection_ledgered": True,
            "one_carrier_per_output_row_prevents_primary_overlap": True,
            "candidate_ordinary_primary_loss_masked": True,
            "omissions_disagreements_and_background_unknown": True,
            "proof_rows_opened": 0,
        },
        "output": {
            "path": str(output_path),
            "rows": len(output_rows),
            "sha256": sha256_file(output_path),
        },
        "policy": (
            "weight only root-accepted exact reference, Bernoulli, and trainable subclass "
            "positives at 0.9; retain reference-form evidence at zero learning weight; mask "
            "candidate ordinary carriers and all unknown cells; keep nested accepted references "
            "in the projection ledger rather than forcing overlap into one-layer BIOES"
        ),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def cmd_successor_intersection_materialize(args: Any) -> dict[str, Any]:
    receipt = materialize_successor_intersection_review(
        args.source_directory,
        args.review_packet,
        args.adjudication,
        args.subclass_spec,
        args.output,
        args.receipt,
        run_id=os.environ.get("AGENTCTL_RUN_ID", ""),
    )
    return {
        "kind": "successor_intersection_materialization",
        "rows": receipt["materialized"]["output_rows"],
        "assertions": receipt["materialized"]["assertions"],
        "projection_exclusions": receipt["materialized"]["projection_exclusions"],
        "output": str(args.output),
        "receipt": str(args.receipt),
    }


def cmd_packet(args: Any) -> dict[str, Any]:
    receipt = prepare_annotation_packets(
        args.input,
        args.output_directory,
        args.receipt,
        conditions=args.condition,
    )
    return {
        "kind": "annotation_packet",
        "rows": receipt["counts"]["rows"],
        "languages": len(receipt["counts"]["by_language"]),
        "output_directory": str(args.output_directory),
        "receipt": str(args.receipt),
    }


class TestSplitter:
    segmenter_model = "test"
    segmenter_model_revision = "test"
    segmenter_name = "test"
    segmenter_version = "test"

    def split(self, texts: Sequence[str]) -> Iterable[Sequence[str]]:
        for text in texts:
            pieces = re.split(r"(?<=[.!?])\s+", text)
            rebuilt = []
            cursor = 0
            for piece in pieces:
                end = cursor + len(piece)
                while end < len(text) and text[end].isspace():
                    end += 1
                rebuilt.append(text[cursor:end])
                cursor = end
            yield rebuilt


def cmd_self_test(_args: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        corpus = directory / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "split": "train",
                    "language": "en",
                    "primary_tag": "email",
                    "target": "name@example.org",
                    "left": "Please email ",
                    "right": " for an appointment.",
                    "factors": [],
                    "regions": [],
                    "source_group_id": "g1",
                    "example_id": "e1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        plan = plan_retrieval(
            corpus, split="train", minimum_examples=5, minimum_distinct=2, queries_per_cell=5
        )
        assert plan["cells"][0]["condition"] == "primary:email"
        assert plan["queries"][0]["kind"] == "structural_regex"
        pattern = compile_query(plan["queries"][0])
        match = pattern.search("Write to user@host.test. Thanks.")
        assert match and match.group() == "user@host.test"
        bounds = paragraph_bounds("one\n\ntwo user@host.test.\n\nthree", 9, 23, 100)
        assert bounds == (5, 24, "blank_line_paragraph")
        assert sentence_spans("First. Second.", [], TestSplitter()) == [(0, 7), (7, 14)]
        candidate_path = directory / "candidates.jsonl"
        candidate_text = "The witness reported it."
        candidate_path.write_text(
            json.dumps(
                {
                    "schema": CANDIDATE_SCHEMA,
                    "id": "c1",
                    "language": "en",
                    "text": candidate_text,
                    "text_sha256": sha256_text(candidate_text),
                    "source_document_id": "d1",
                    "source_document_sha256": "1" * 64,
                    "source_document_start": 10,
                    "source_document_end": 34,
                    "source_context_before": "Before. ",
                    "source_context_after": " After.",
                    "intended_split": "train",
                    "retrieval_matches": [{"condition": "predicate:witness_or_bystander=true"}],
                    "source": {"dataset": "test"},
                    "proof_data_used": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        packet_receipt = prepare_annotation_packets(
            candidate_path,
            directory / "packets",
            directory / "packet-receipt.json",
            conditions=["predicate:witness_or_bystander=true"],
        )
        assert packet_receipt["counts"]["rows"] == 1
        assert (directory / "packets" / "en.jsonl").exists()

        gold_path = directory / "gold" / "en.jsonl"
        gold_path.parent.mkdir()
        retained_text = "Email alice@example.org."
        excluded_text = "Email holdout@example.org."
        remainder_text = "Email carol@example.net."
        gold_rows = [
            {
                "id": "gold-1",
                "lang": "en",
                "source_dataset": "test",
                "source_schema": "test",
                "source_split": "train",
                "text": retained_text,
                "spans": [{"start": 6, "end": 23, "type": "email"}],
            },
            {
                "id": "gold-2",
                "lang": "en",
                "source_dataset": "test",
                "source_schema": "test",
                "source_split": "train",
                "text": excluded_text,
                "spans": [{"start": 6, "end": 25, "type": "email"}],
            },
        ]
        gold_path.write_text(jsonl_text(gold_rows), encoding="utf-8")
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "pii-ontology-v2-training-gold",
                    "shards": [
                        {
                            "dataset": "test",
                            "split": "train",
                            "lang": "en",
                            "gold_path": "gold/en.jsonl",
                            "gold_sha256": sha256_file(gold_path),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        plan_path = directory / "plan.json"
        plan_path.write_text(
            json.dumps({"schema": PLAN_SCHEMA, "cells": [{"language": "en", "condition": "primary:email"}]}),
            encoding="utf-8",
        )
        exclusion_path = directory / "fixed-corpus.jsonl"
        exclusion_path.write_text(
            json.dumps({"split": "audit", "source_text_sha256": sha256_text(excluded_text)}) + "\n",
            encoding="utf-8",
        )
        supplement_path = directory / "supplement.jsonl"
        supplement_receipt = build_gold_primary_supplement(
            manifest_path,
            plan_path,
            supplement_path,
            directory / "supplement-receipt.json",
            tags=(),
            languages=(),
            exclude_corpora=(exclusion_path,),
            exclude_split="audit",
            max_per_cell=5,
        )
        supplement_rows = read_jsonl(supplement_path)
        assert supplement_receipt["counts"]["rows"] == 1
        assert supplement_rows[0]["text"] == retained_text
        assert supplement_rows[0]["spans"] == [{"start": 6, "end": 23, "type": "email"}]
        assert "predicate_spans" not in supplement_rows[0]

        normalized_directory = directory / "normalized"
        primary_directory = directory / "primary"
        normalized_directory.mkdir()
        primary_directory.mkdir()
        normalized_rows = [
            {
                "id": "target-1",
                "text": retained_text,
                "lang": "en",
                "source_group_id": "doc-1",
                "annotation_domain": "self-test",
                "source_document_sha256": sha256_text("doc-1"),
                "source_document_start": 0,
                "source_document_end": len(retained_text),
                "text_sha256": sha256_text(retained_text),
                "source_context_before": "Before. ",
                "source_context_after": " After.",
                "intended_split": "train",
                "retrieval_conditions": ["primary:email"],
                "proof_data_used": False,
            },
            {
                "id": "target-2",
                "text": excluded_text,
                "lang": "en",
                "source_group_id": "doc-2",
                "annotation_domain": "self-test",
                "source_document_sha256": sha256_text("doc-2"),
                "source_document_start": 0,
                "source_document_end": len(excluded_text),
                "text_sha256": sha256_text(excluded_text),
                "source_context_before": "",
                "source_context_after": "",
                "intended_split": "development",
                "retrieval_conditions": ["primary:email"],
                "proof_data_used": False,
            },
            {
                "id": "target-3",
                "text": remainder_text,
                "lang": "en",
                "source_group_id": "doc-3",
                "annotation_domain": "self-test",
                "source_document_sha256": sha256_text("doc-3"),
                "source_document_start": 0,
                "source_document_end": len(remainder_text),
                "text_sha256": sha256_text(remainder_text),
                "source_context_before": "Before third. ",
                "source_context_after": " After third.",
                "intended_split": "train",
                "retrieval_conditions": ["primary:email"],
                "proof_data_used": False,
            },
        ]
        (normalized_directory / "en.jsonl").write_text(jsonl_text(normalized_rows), encoding="utf-8")
        previous_review_directory = directory / "previous-primary-review"
        previous_review_directory.mkdir()
        (previous_review_directory / "en.jsonl").write_text(
            jsonl_text([{"id": "target-1", "preds": []}]),
            encoding="utf-8",
        )
        additional_review_directory = directory / "additional-primary-review"
        additional_review_directory.mkdir()
        (additional_review_directory / "en.jsonl").write_text(
            jsonl_text([{"id": "target-2", "preds": []}]),
            encoding="utf-8",
        )
        primary_review_receipt = prepare_primary_review_packets(
            normalized_directory,
            previous_review_directory,
            directory / "primary-review-packet",
            directory / "primary-review-packet-receipt.json",
            additional_exclude_directories=(additional_review_directory,),
        )
        primary_review_rows = read_jsonl(directory / "primary-review-packet" / "en.jsonl")
        primary_review_guidance = json.loads(primary_review_rows[0]["primary_review_guidance"])
        assert primary_review_receipt["counts"] == {
            "source_rows": 3,
            "excluded_previously_processed_rows": 2,
            "review_rows": 1,
        }
        assert primary_review_rows[0]["id"] == "target-3"
        assert "retrieval_conditions" not in primary_review_rows[0]
        assert "base_spans" not in primary_review_guidance
        assert "retrieval_conditions" not in primary_review_guidance
        assert primary_review_guidance["document_context_before"] == "Before third. "
        reference_review_directory = directory / "reference-primary-review"
        independent_review_a = directory / "independent-primary-review-a"
        independent_review_b = directory / "independent-primary-review-b"
        for review_directory in (
            reference_review_directory,
            independent_review_a,
            independent_review_b,
        ):
            review_directory.mkdir()
        (reference_review_directory / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-1",
                        "preds": [{"start": 6, "end": 23, "label": "email"}],
                    },
                    {
                        "id": "target-2",
                        "preds": [{"start": 6, "end": 25, "label": "email"}],
                        "parse_stats": {"bad_json": 1},
                        "annotation_banned": True,
                    },
                    {
                        "id": "target-3",
                        "preds": [{"start": 6, "end": 23, "label": "email"}],
                    },
                ]
            ),
            encoding="utf-8",
        )
        (independent_review_a / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-1",
                        "preds": [{"start": 6, "end": 23, "label": "email"}],
                    },
                    {"id": "target-2", "preds": []},
                ]
            ),
            encoding="utf-8",
        )
        (independent_review_b / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-3",
                        "preds": [{"start": 6, "end": 23, "label": "email"}],
                    }
                ]
            ),
            encoding="utf-8",
        )
        intersection_receipt = prepare_primary_intersection_review(
            normalized_directory,
            reference_review_directory,
            (independent_review_a, independent_review_b),
            directory / "primary-intersection-review.jsonl",
            directory / "primary-intersection-review-receipt.json",
        )
        intersection_rows = read_jsonl(directory / "primary-intersection-review.jsonl")
        assert intersection_receipt["counts"]["source_rows"] == 3
        assert intersection_receipt["counts"]["candidate_rows"] == 2
        assert intersection_receipt["counts"]["exact_intersection_spans_by_type"] == {"email": 2}
        assert intersection_receipt["counts"]["quarantined_rows"] == 1
        assert [row["id"] for row in intersection_rows] == ["target-1", "target-3"]
        assert all(row["review_status"] == "unreviewed" for row in intersection_rows)
        intersection_path = directory / "primary-intersection-review.jsonl"
        adjudication_path = directory / "primary-intersection-adjudication.json"
        adjudication_path.write_text(
            json.dumps(
                {
                    "schema": PRIMARY_INTERSECTION_ADJUDICATION_SCHEMA,
                    "status": "root_review_complete_materialization_authorized",
                    "proof_data_used": False,
                    "packet": {
                        "sha256": sha256_file(intersection_path),
                        "candidate_rows": 2,
                        "exact_intersection_spans": 2,
                    },
                    "review_contract": {
                        "default_exact_intersection_decision": "accept",
                        "objective_weight": 0.9,
                    },
                    "summary": {
                        "accepted_rows": 1,
                        "accepted_exact_intersections": 1,
                        "rejected_exact_intersections": 1,
                        "accepted_by_language": {"en": 1},
                        "accepted_by_type": {"email": 1},
                    },
                    "rejected_exact_intersections": [
                        {
                            "language": "en",
                            "id": "target-3",
                            "start": 6,
                            "end": 23,
                            "type": "email",
                            "surface": "carol@example.net",
                            "reason": "self-test rejection",
                        }
                    ],
                    "admission": {
                        "authorized_rows": 1,
                        "authorized_spans": 1,
                        "supervision": ANNOTATED_SPANS_ONLY,
                        "label_space": "v2",
                        "outside_span_objective_weight": 0.0,
                        "predicate_and_subclass_objective_weight": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        materialization_receipt = materialize_primary_intersection_review(
            normalized_directory,
            intersection_path,
            adjudication_path,
            directory / "primary-intersection-reviewed.jsonl",
            directory / "primary-intersection-reviewed-receipt.json",
            run_id="self-test-run",
        )
        materialized_rows = read_jsonl(directory / "primary-intersection-reviewed.jsonl")
        assert materialization_receipt["output"]["rows"] == 1
        assert materialization_receipt["output"]["spans"] == 1
        assert materialization_receipt["rejected"]["spans"] == 1
        assert materialized_rows[0]["source_row_id"] == "target-1"
        assert materialized_rows[0]["spans"] == [[6, 23, "email"]]
        assert materialized_rows[0]["primary_span_objective_weights"] == [0.9]
        assert materialized_rows[0]["supervision"] == ANNOTATED_SPANS_ONLY
        assert materialized_rows[0]["surface_origin"] == "real_original"
        (primary_directory / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-1",
                        "preds": [{"start": 6, "end": 23, "label": "email"}],
                        "annotation_banned": False,
                    },
                    {"id": "target-2", "preds": [], "annotation_banned": True},
                    {"id": "target-3", "preds": [], "annotation_banned": True},
                ]
            ),
            encoding="utf-8",
        )
        successor_receipt = prepare_successor_packets(
            normalized_directory,
            primary_directory,
            directory / "successor",
            directory / "successor-receipt.json",
        )
        successor_rows = read_jsonl(directory / "successor" / "en.jsonl")
        successor_guidance = json.loads(successor_rows[0]["successor_annotation_guidance"])
        assert successor_receipt["counts"]["successor_rows"] == 1
        assert successor_receipt["counts"]["banned_primary_rows"] == 2
        assert successor_rows[0]["base_spans"][0]["surface"] == "alice@example.org"
        assert "retrieval_conditions" not in successor_guidance
        successor_review_source = directory / "successor-review-source"
        successor_review_reference = directory / "successor-review-reference"
        successor_review_completed = directory / "successor-review-completed"
        for review_directory in (
            successor_review_source,
            successor_review_reference,
            successor_review_completed,
        ):
            review_directory.mkdir()
        successor_review_rows = []
        for row in normalized_rows:
            review_row = dict(row)
            review_row["base_spans"] = []
            review_row["successor_annotation_guidance"] = json.dumps({"row_id": row["id"]}, sort_keys=True)
            successor_review_rows.append(review_row)
        (successor_review_source / "en.jsonl").write_text(jsonl_text(successor_review_rows), encoding="utf-8")
        (successor_review_reference / "en.jsonl").write_text(
            jsonl_text(
                [
                    {"id": "target-1", "parse_stats": {}, "annotation_banned": False},
                    {"id": "target-2", "parse_stats": {}, "annotation_banned": True},
                    {"id": "target-3", "parse_stats": {}, "annotation_banned": False},
                ]
            ),
            encoding="utf-8",
        )
        (successor_review_completed / "en.jsonl").write_text(
            jsonl_text([{"id": "target-1", "parse_stats": {}, "annotation_banned": False}]),
            encoding="utf-8",
        )
        successor_remainder_receipt = prepare_successor_review_remainder(
            successor_review_source,
            successor_review_reference,
            successor_review_completed,
            directory / "successor-review-remainder",
            directory / "successor-review-remainder-receipt.json",
        )
        successor_remainder_rows = read_jsonl(directory / "successor-review-remainder" / "en.jsonl")
        assert successor_remainder_receipt["counts"] == {
            "source_rows": 3,
            "completed_paid_rows": 1,
            "reference_banned_rows": 1,
            "review_rows": 1,
        }
        assert [row["id"] for row in successor_remainder_rows] == ["target-3"]
        successor_intersection_reference = directory / "successor-intersection-reference"
        successor_intersection_a = directory / "successor-intersection-a"
        successor_intersection_b = directory / "successor-intersection-b"
        for review_directory in (
            successor_intersection_reference,
            successor_intersection_a,
            successor_intersection_b,
        ):
            review_directory.mkdir()
        reference_carrier = {"start": 6, "end": 23, "label": "person_reference"}
        reference_form = {
            "carrier_start": 6,
            "carrier_end": 23,
            "type": "person_reference",
            "start": 6,
            "end": 23,
            "family": "reference_form",
            "value": "symbolic_pseudonym",
        }
        (successor_intersection_reference / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-1",
                        "preds": [reference_carrier, {"start": 6, "end": 23, "label": "patient"}],
                        "subclass_spans": [reference_form],
                        "parse_stats": {},
                        "annotation_banned": False,
                    },
                    {
                        "id": "target-2",
                        "preds": [],
                        "subclass_spans": [],
                        "parse_stats": {"bad_json": 1},
                        "annotation_banned": True,
                    },
                    {
                        "id": "target-3",
                        "preds": [],
                        "subclass_spans": [],
                        "parse_stats": {},
                        "annotation_banned": False,
                    },
                ]
            ),
            encoding="utf-8",
        )
        (successor_intersection_a / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-1",
                        "preds": [reference_carrier],
                        "subclass_spans": [reference_form],
                        "parse_stats": {},
                        "annotation_banned": False,
                    }
                ]
            ),
            encoding="utf-8",
        )
        (successor_intersection_b / "en.jsonl").write_text(
            jsonl_text(
                [
                    {
                        "id": "target-3",
                        "preds": [],
                        "subclass_spans": [],
                        "parse_stats": {"out_of_order": 1},
                        "annotation_banned": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        test_subclass_spec = PROJECT_ROOT / "scripts/pii_subclass_families_v4.json"
        successor_intersection_receipt = prepare_successor_intersection_review(
            successor_review_source,
            successor_intersection_reference,
            (successor_intersection_a, successor_intersection_b),
            test_subclass_spec,
            directory / "successor-intersection-review.jsonl",
            directory / "successor-intersection-review-receipt.json",
        )
        successor_intersection_rows = read_jsonl(directory / "successor-intersection-review.jsonl")
        assert successor_intersection_receipt["counts"]["candidate_rows"] == 1
        assert successor_intersection_receipt["counts"]["exact_predictions"] == 1
        assert successor_intersection_receipt["counts"]["exact_subclasses"] == 1
        assert successor_intersection_receipt["counts"]["reference_banned_rows"] == 1
        assert successor_intersection_receipt["counts"]["independent_banned_rows"] == 1
        assert successor_intersection_rows[0]["exact_predictions"][0]["surface"] == "alice@example.org"
        assert successor_intersection_rows[0]["reference_only_predictions"][0]["label"] == "patient"
        successor_intersection_packet = directory / "successor-intersection-review.jsonl"
        successor_intersection_adjudication = directory / "successor-intersection-adjudication.json"
        successor_intersection_adjudication.write_text(
            json.dumps(
                {
                    "schema": SUCCESSOR_INTERSECTION_ADJUDICATION_SCHEMA,
                    "status": "root_review_complete_materialization_authorized",
                    "proof_data_used": False,
                    "packet": {
                        "path": str(successor_intersection_packet),
                        "sha256": sha256_file(successor_intersection_packet),
                        "rows": 1,
                        "exact_predictions": 1,
                        "exact_subclasses": 1,
                    },
                    "review_contract": {
                        "default_exact_assertion_decision": "accept",
                        "objective_weight": 0.9,
                        "activating_carrier_primary_objective_weight": 0.0,
                    },
                    "review_summary": {
                        "assertions_reviewed": 2,
                        "accepted_assertions": 2,
                        "rejected_assertions": 0,
                        "predictions_reviewed": 1,
                        "accepted_predictions": 1,
                        "rejected_predictions": 0,
                        "subclasses_reviewed": 1,
                        "accepted_subclasses": 1,
                        "rejected_subclasses": 0,
                    },
                    "rejected_exact_predictions": [],
                    "rejected_exact_subclasses": [],
                    "materialization_authorization": {
                        "supervision": ANNOTATED_SPANS_ONLY,
                        "label_space": "v2",
                        "accepted_reference_primary_weight": 0.9,
                        "accepted_bernoulli_positive_weight": 0.9,
                        "accepted_trainable_subclass_weight": 0.9,
                        "activating_carrier_primary_weight": 0.0,
                        "outside_span_objective_weight": 0.0,
                        "known_negative_objective_weight": 0.0,
                        "unasserted_factor_objective_weight": 0.0,
                        "reference_form_learning_weight": 0.0,
                        "ordinary_ont2_supervision": (
                            "Retained externally; do not promote candidate base spans here."
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        successor_materialization_receipt = materialize_successor_intersection_review(
            successor_review_source,
            successor_intersection_packet,
            successor_intersection_adjudication,
            test_subclass_spec,
            directory / "successor-intersection-materialized.jsonl",
            directory / "successor-intersection-materialization-receipt.json",
            run_id="self-test-run",
        )
        successor_materialized_rows = read_jsonl(directory / "successor-intersection-materialized.jsonl")
        assert successor_materialization_receipt["materialized"]["output_rows"] == 1
        assert successor_materialization_receipt["materialized"]["assertions"] == 2
        assert successor_materialization_receipt["materialized"]["projection_exclusions"] == 0
        assert successor_materialized_rows[0]["spans"] == [[6, 23, "person_reference"]]
        assert successor_materialized_rows[0]["primary_span_objective_weights"] == [0.9]
        assert successor_materialized_rows[0]["reference_form_spans"][0]["learning_weight"] == 0.0
    return {"kind": "self_test", "status": "pass"}


def build_parser() -> Any:
    parser = acli.argument_parser(
        description=__doc__,
        capabilities=("complete",),
        exit_codes={0: "success", 2: "invalid input or incomplete output"},
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan", help="Audit support and freeze retrieval queries (seconds; blocking summary)."
    )
    plan.add_argument("--corpus", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--split", default="train")
    plan.add_argument("--minimum-examples", type=int, default=100)
    plan.add_argument("--minimum-distinct", type=int, default=25)
    plan.add_argument("--queries-per-cell", type=int, default=12)
    acli.add_standard_args(plan)
    plan.set_defaults(func=cmd_plan)

    retrieve = subparsers.add_parser(
        "retrieve", help="Stream pinned FineWeb sources and emit candidates (minutes; blocking summary)."
    )
    retrieve.add_argument("--plan", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--receipt", type=Path, required=True)
    retrieve.add_argument("--language", action="append", default=[])
    retrieve.add_argument(
        "--condition",
        action="append",
        default=[],
        help="retrieve only this exact plan condition (repeatable; default: all)",
    )
    retrieve.add_argument("--exclude-jsonl", action="append", type=Path, default=[])
    retrieve.add_argument("--quota-per-cell", type=int, default=8)
    retrieve.add_argument("--max-documents-per-language", type=int, default=5000)
    retrieve.add_argument("--shuffle-buffer", type=int, default=1000)
    retrieve.add_argument("--seed", type=int, default=20260830)
    retrieve.add_argument("--max-context-chars", type=int, default=12000)
    retrieve.add_argument(
        "--adjacent-context-chars",
        type=int,
        default=1000,
        help="retain exact source context on each side for later reference review",
    )
    retrieve.add_argument("--segmenter-model", default="sat-3l-sm")
    retrieve.add_argument("--segmenter-revision", default="137da054051ad9f1eac42025f758db4ac9f22535")
    retrieve.add_argument("--segmenter-batch-size", type=int, default=32)
    retrieve.add_argument("--segmenter-device", default="cpu")
    acli.add_standard_args(retrieve)
    retrieve.set_defaults(func=cmd_retrieve)

    gold_supplement = subparsers.add_parser(
        "gold-supplement",
        help=(
            "Select diverse factor-masked primary surfaces from trusted ont2 gold "
            "(seconds; blocking summary)."
        ),
    )
    gold_supplement.add_argument("--manifest", type=Path, required=True)
    gold_supplement.add_argument("--plan", type=Path, required=True)
    gold_supplement.add_argument("--output", type=Path, required=True)
    gold_supplement.add_argument("--receipt", type=Path, required=True)
    gold_supplement.add_argument(
        "--language",
        action="append",
        default=None,
        help="retain this BCP-47 language bucket (repeatable; default: languages in the plan)",
    )
    gold_supplement.add_argument(
        "--tag",
        action="append",
        default=None,
        help="retain this primary tag in every selected language (repeatable; default: plan cells)",
    )
    gold_supplement.add_argument(
        "--exclude-corpus",
        action="append",
        type=Path,
        default=None,
        required=True,
        help="exclude source hashes in --exclude-split from this factorized corpus (repeatable)",
    )
    gold_supplement.add_argument("--exclude-split", default="audit")
    gold_supplement.add_argument("--max-per-cell", type=int, default=250)
    acli.add_standard_args(gold_supplement)
    gold_supplement.set_defaults(func=cmd_gold_supplement)

    primary_review_packet = subparsers.add_parser(
        "primary-review-packet",
        help=(
            "Prepare context-bearing proposal-blind primary inputs, excluding completed rows "
            "without resending them (seconds; blocking summary)."
        ),
    )
    primary_review_packet.add_argument("--source-directory", type=Path, required=True)
    primary_review_packet.add_argument(
        "--exclude-directory",
        type=Path,
        required=True,
        help="language JSONL outputs already processed; a missing language file excludes no rows",
    )
    primary_review_packet.add_argument(
        "--additional-exclude-directory",
        action="append",
        type=Path,
        default=None,
        help="additional language JSONL outputs whose row IDs are also excluded (repeatable)",
    )
    primary_review_packet.add_argument("--output-directory", type=Path, required=True)
    primary_review_packet.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(primary_review_packet)
    primary_review_packet.set_defaults(func=cmd_primary_review_packet)

    primary_intersection_review = subparsers.add_parser(
        "primary-intersection-review",
        help=(
            "Bind exact reference/independent primary agreements to literal context for root "
            "review without admitting them (seconds; blocking summary)."
        ),
    )
    primary_intersection_review.add_argument("--source-directory", type=Path, required=True)
    primary_intersection_review.add_argument("--reference-directory", type=Path, required=True)
    primary_intersection_review.add_argument(
        "--independent-directory",
        action="append",
        type=Path,
        default=None,
        required=True,
        help="disjoint language JSONL view directory (repeatable; union must cover the source)",
    )
    primary_intersection_review.add_argument("--output", type=Path, required=True)
    primary_intersection_review.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(primary_intersection_review)
    primary_intersection_review.set_defaults(func=cmd_primary_intersection_review)

    primary_intersection_materialize = subparsers.add_parser(
        "primary-intersection-materialize",
        help=(
            "Materialize root-reviewed exact agreements as weighted positive-only v2 rows "
            "without inferring background or factor negatives (seconds; blocking summary)."
        ),
    )
    primary_intersection_materialize.add_argument("--source-directory", type=Path, required=True)
    primary_intersection_materialize.add_argument("--review-packet", type=Path, required=True)
    primary_intersection_materialize.add_argument("--adjudication", type=Path, required=True)
    primary_intersection_materialize.add_argument("--output", type=Path, required=True)
    primary_intersection_materialize.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(primary_intersection_materialize)
    primary_intersection_materialize.set_defaults(func=cmd_primary_intersection_materialize)

    successor_packet = subparsers.add_parser(
        "successor-packet",
        help=(
            "Join usable primary proposals to normalized source context for ont3 review "
            "(seconds; blocking summary)."
        ),
    )
    successor_packet.add_argument("--source-directory", type=Path, required=True)
    successor_packet.add_argument("--primary-directory", type=Path, required=True)
    successor_packet.add_argument("--output-directory", type=Path, required=True)
    successor_packet.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(successor_packet)
    successor_packet.set_defaults(func=cmd_successor_packet)

    successor_review_remainder = subparsers.add_parser(
        "successor-review-remainder",
        help=(
            "Exclude reference-banned and already paid independent successor rows without "
            "rendering the reference successor output (seconds; blocking summary)."
        ),
    )
    successor_review_remainder.add_argument("--source-directory", type=Path, required=True)
    successor_review_remainder.add_argument("--reference-directory", type=Path, required=True)
    successor_review_remainder.add_argument("--completed-directory", type=Path, required=True)
    successor_review_remainder.add_argument("--output-directory", type=Path, required=True)
    successor_review_remainder.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(successor_review_remainder)
    successor_review_remainder.set_defaults(func=cmd_successor_review_remainder)

    successor_intersection_review = subparsers.add_parser(
        "successor-intersection-review",
        help=(
            "Bind exact healthy reference/independent successor assertions to literal context "
            "for root review without admitting them (seconds; blocking summary)."
        ),
    )
    successor_intersection_review.add_argument("--source-directory", type=Path, required=True)
    successor_intersection_review.add_argument("--reference-directory", type=Path, required=True)
    successor_intersection_review.add_argument(
        "--independent-directory",
        action="append",
        type=Path,
        default=None,
        required=True,
        help=(
            "disjoint independent language JSONL directory (repeatable; union must cover all "
            "non-banned reference rows)"
        ),
    )
    successor_intersection_review.add_argument("--subclass-spec", type=Path, required=True)
    successor_intersection_review.add_argument("--output", type=Path, required=True)
    successor_intersection_review.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(successor_intersection_review)
    successor_intersection_review.set_defaults(func=cmd_successor_intersection_review)

    successor_intersection_materialize = subparsers.add_parser(
        "successor-intersection-materialize",
        help=(
            "Materialize root-reviewed exact successor positives with local factor weights and "
            "unknown background (seconds; blocking summary)."
        ),
    )
    successor_intersection_materialize.add_argument("--source-directory", type=Path, required=True)
    successor_intersection_materialize.add_argument("--review-packet", type=Path, required=True)
    successor_intersection_materialize.add_argument("--adjudication", type=Path, required=True)
    successor_intersection_materialize.add_argument("--subclass-spec", type=Path, required=True)
    successor_intersection_materialize.add_argument("--output", type=Path, required=True)
    successor_intersection_materialize.add_argument("--receipt", type=Path, required=True)
    acli.add_standard_args(successor_intersection_materialize)
    successor_intersection_materialize.set_defaults(func=cmd_successor_intersection_materialize)

    packet = subparsers.add_parser(
        "packet",
        help="Filter candidates into language-specific teacher packets (seconds; blocking summary).",
    )
    packet.add_argument("--input", type=Path, required=True)
    packet.add_argument("--output-directory", type=Path, required=True)
    packet.add_argument("--receipt", type=Path, required=True)
    packet.add_argument("--condition", action="append", required=True)
    acli.add_standard_args(packet)
    packet.set_defaults(func=cmd_packet)

    self_test = subparsers.add_parser("self-test", help="Run dependency-light invariant checks (seconds).")
    acli.add_standard_args(self_test)
    self_test.set_defaults(func=cmd_self_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        acli.maybe_complete(parser)
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (FileExistsError, OSError, RetrievalError, ValueError) as error:
        acli.die(str(error), 2)
    acli.emit(result, acli.resolve_format(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
