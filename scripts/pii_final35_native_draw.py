#!/usr/bin/env python3
# acli: 1 complete
"""Draw native running-text candidates with preliminary source exclusions.

The ontology-v3 teacher-versus-encoder programme needs, in all 35 target
languages, three mutually disjoint evaluation roles (teacher-selection
development, rotating non-proof audit, sealed final audit) plus training
material, all built from genuine native running text rather than translated
or generated surfaces. This tool streams the pinned FineWeb / FineWeb2
snapshots, assigns each normalized source document to one role by a frozen hash rule,
excludes declared prior source identities and text hashes, sentence-segments the
document with the pinned SaT model, and samples sentences per language,
role, and stratum:

- ``identifier``: the sentence contains a universal structured PII cue
  (email, URL, phone-like digit run, date-like number, handle);
- ``reference``: the sentence contains several reference cues from the
  per-language lexicon (pronouns, role nouns, kinship nouns, institutional
  nouns, honorifics); and
- ``ordinary``: any sentence that passes the length and text-quality filters.

Every output row keeps the source document identity and hash, the exact
character interval, the enclosing paragraph as annotation context, the
segmenter provenance, the stratum, the role, and the hash bucket. Nothing here
is supervision. The selected lexical/semantic partial-overlap detector must
clear candidates before annotation or split admission; these preliminary
exclusions alone do not establish novelty or freedom from contamination.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import acli
except ImportError:
    sys.path.insert(0, str(Path.home() / "agents"))
    import acli

from scripts.pii_ont3_surface_retrieval import (
    document_identity,
    fineweb_stream,
    paragraph_bounds,
    sha256_text,
    source_descriptor,
    stable_id,
)
from scripts.pii_text_segmentation import (
    SAT_MODEL,
    SAT_MODEL_REVISION,
    SaTCharacterSpanSplitter,
    segmenter_provenance,
    sentence_spans,
)

DRAW_SCHEMA = "pii-final35-native-draw-candidate/v1"
RECEIPT_SCHEMA = "pii-final35-native-draw-receipt/v1"
ROLE_BUCKETS = {
    "development": (0, 20),
    "rotation-1": (20, 45),
    "rotation-2": (45, 70),
    "final": (70, 100),
}
TRAINING_ROLE = "training"
# Training material comes only from the rotation buckets: the development and
# final buckets never train. Declared prior sources are excluded by source ID
# and text hashes (--exclude-jsonl). Partial-overlap screening is a separate
# required gate before annotation or admission, including within the draw.
# A run draws either evaluation roles or the training role, never both.
TRAINING_SOURCE_ROLES = ("rotation-1", "rotation-2")
STRATA = ("identifier", "reference", "ordinary")
NO_WORD_BOUNDARY_LANGUAGES = frozenset({"zh", "ja", "th"})
MAX_WHITESPACE_WORD_CHARS = 60
LEXICON_FIELDS = (
    "personal_pronouns",
    "person_role_nouns",
    "kinship_nouns",
    "organization_nouns",
    "honorifics",
)
IDENTIFIER_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"(?:https?://|www\.)\S+"),
    "handle": re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}"),
    "phone": re.compile(r"(?:\+|\(?\d)[\d\s().-]{7,}\d"),
    "long_digits": re.compile(r"\d{5,}"),
    "date": re.compile(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b"),
}


class DrawError(ValueError):
    pass


def role_for_document(document_hash: str) -> tuple[str, int]:
    bucket = int(document_hash[:8], 16) % 100
    for role, (low, high) in ROLE_BUCKETS.items():
        if low <= bucket < high:
            return role, bucket
    raise DrawError(f"bucket {bucket} has no role")


def draw_role_for_document(document_hash: str, quotas: dict[tuple[str, str], int]) -> tuple[str, str, int]:
    """Return (draw role, bucket role, bucket) for one document under the run's quotas.

    A training run (only ``training/*`` quotas) maps documents from the rotation
    buckets to the training role; documents from other buckets keep their bucket
    role, which has no quota in such a run and is therefore skipped.
    """
    bucket_role, bucket = role_for_document(document_hash)
    training_run = any(role == TRAINING_ROLE for role, _ in quotas)
    if training_run and bucket_role in TRAINING_SOURCE_ROLES:
        return TRAINING_ROLE, bucket_role, bucket
    return bucket_role, bucket_role, bucket


def load_lexicon(directory: Path, language: str) -> dict[str, list[str]]:
    path = directory / f"{language}.json"
    if not path.exists():
        raise DrawError(f"{language}: no reference-cue lexicon at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    return {field: [str(item) for item in lexicon[field]] for field in LEXICON_FIELDS}


def compile_cues(language: str, lexicon: dict[str, list[str]]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for field, items in lexicon.items():
        for item in items:
            escaped = re.escape(item.casefold())
            if language in NO_WORD_BOUNDARY_LANGUAGES:
                pattern = re.compile(escaped)
            else:
                pattern = re.compile(rf"(?<!\w){escaped}(?!\w)")
            compiled.append((field, pattern))
    return compiled


def letter_ratio(text: str) -> float:
    letters = sum(1 for char in text if unicodedata.category(char).startswith("L"))
    return letters / max(1, len(text))


def passes_quality(
    text: str, minimum_chars: int, maximum_chars: int, *, language: str | None = None
) -> str | None:
    stripped = text.strip()
    if len(stripped) < minimum_chars:
        return "too_short"
    if len(stripped) > maximum_chars:
        return "too_long"
    if letter_ratio(stripped) < 0.5:
        return "low_letter_ratio"
    if "\n" in stripped:
        return "internal_line_break"
    # Whitespace does not delimit words in these languages.
    if language not in NO_WORD_BOUNDARY_LANGUAGES:
        longest = max((len(token) for token in stripped.split()), default=0)
        if longest > MAX_WHITESPACE_WORD_CHARS:
            return "overlong_token"
    # Uncased scripts and embedded acronyms are not uppercase banners.
    if (
        len(stripped) > 40
        and stripped.isupper()
        and sum(char.isupper() for char in stripped) / len(stripped) > 0.6
    ):
        return "all_caps"
    return None


def identifier_hits(text: str) -> list[str]:
    return [name for name, pattern in IDENTIFIER_PATTERNS.items() if pattern.search(text)]


def reference_hits(text: str, cues: list[tuple[str, re.Pattern[str]]]) -> dict[str, int]:
    folded = text.casefold()
    hits: Counter[str] = Counter()
    for field, pattern in cues:
        hits[field] += len(pattern.findall(folded))
    return {field: count for field, count in hits.items() if count}


def classify(
    text: str, cues: list[tuple[str, re.Pattern[str]]], reference_minimum: int
) -> tuple[str, dict[str, Any]]:
    identifiers = identifier_hits(text)
    references = reference_hits(text, cues)
    reference_total = sum(references.values())
    distinct_fields = len(references)
    evidence = {"identifier": identifiers, "reference": references}
    if identifiers:
        return "identifier", evidence
    if reference_total >= reference_minimum and distinct_fields >= 2:
        return "reference", evidence
    return "ordinary", evidence


def draw_language(
    language: str,
    *,
    quotas: dict[tuple[str, str], int],
    cues: list[tuple[str, re.Pattern[str]]],
    splitter: Any,
    seed: int,
    shuffle_buffer: int,
    max_documents: int,
    max_sentences_per_document: int,
    minimum_chars: int,
    maximum_chars: int,
    context_chars: int,
    reference_minimum: int,
    excluded_hashes: set[str],
    excluded_source_ids: set[str] | None = None,
    source_documents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(
        int.from_bytes(hashlib.sha256(f"{seed}\0{language}\0draw".encode()).digest()[:8], "big")
    )
    stream_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{language}".encode()).digest()[:8], "big")
    counts: Counter[tuple[str, str]] = Counter()
    rejected: Counter[str] = Counter()
    documents_scanned = 0
    documents_used = 0
    outputs: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    provenance = segmenter_provenance(splitter)
    needed = {key: quota for key, quota in quotas.items() if quota > 0}

    def satisfied() -> bool:
        return all(counts[key] >= quota for key, quota in needed.items())

    for source_row in fineweb_stream(language, seed=stream_seed, shuffle_buffer=shuffle_buffer):
        if satisfied() or documents_scanned >= max_documents:
            break
        documents_scanned += 1
        text = source_row.get("text")
        if not isinstance(text, str):
            rejected["document_too_short"] += 1
            continue
        source_id = document_identity(source_row, language)
        if excluded_source_ids is not None and source_id in excluded_source_ids:
            rejected["excluded_source_id"] += 1
            continue
        raw_document_hash = sha256_text(text)
        # Normalize before segmenting, not after. The grouped annotation format
        # requires NFKC input and its runner refuses a row that is not already
        # normalized, and normalizing later would move every offset this draw
        # computes. Doing it here keeps segment offsets, the document hash and
        # the emitted text consistent with one another.
        text = unicodedata.normalize("NFKC", text)
        if len(text.strip()) < minimum_chars:
            rejected["document_too_short"] += 1
            continue
        document_hash = sha256_text(text)
        if document_hash in excluded_hashes or raw_document_hash in excluded_hashes:
            rejected["excluded_document"] += 1
            continue
        role, bucket_role, bucket = draw_role_for_document(document_hash, quotas)
        if all(counts[(role, stratum)] >= quotas.get((role, stratum), 0) for stratum in STRATA):
            rejected["role_quota_full"] += 1
            continue
        spans = sentence_spans(text, [], splitter)
        candidates: dict[str, list[tuple[int, int, int, dict[str, Any]]]] = defaultdict(list)
        for ordinal, (start, end) in enumerate(spans, 1):
            sentence = text[start:end]
            reason = passes_quality(sentence, minimum_chars, maximum_chars, language=language)
            if reason is not None:
                rejected[reason] += 1
                continue
            stratum, evidence = classify(sentence, cues, reference_minimum)
            candidates[stratum].append((ordinal, start, end, evidence))
        taken = 0
        descriptor = source_descriptor(source_row, language)
        for stratum in STRATA:
            key = (role, stratum)
            while (
                candidates[stratum]
                and counts[key] < quotas.get(key, 0)
                and taken < max_sentences_per_document
            ):
                index = rng.randrange(len(candidates[stratum]))
                ordinal, start, end, evidence = candidates[stratum].pop(index)
                sentence = text[start:end]
                stripped_start = start + (len(sentence) - len(sentence.lstrip()))
                stripped_end = end - (len(sentence) - len(sentence.rstrip()))
                sentence = text[stripped_start:stripped_end]
                sentence_hash = sha256_text(sentence)
                if sentence_hash in seen_sentences or sentence_hash in excluded_hashes:
                    rejected["duplicate_sentence"] += 1
                    continue
                seen_sentences.add(sentence_hash)
                para_start, para_end, window_kind = paragraph_bounds(
                    text, stripped_start, stripped_end, context_chars
                )
                outputs.append(
                    {
                        "schema": DRAW_SCHEMA,
                        "id": stable_id(
                            {
                                "dataset": descriptor["dataset"],
                                "document_id": source_id,
                                "start": stripped_start,
                                "end": stripped_end,
                            }
                        ),
                        "lang": language,
                        "bcp47": language,
                        "text": sentence,
                        "text_sha256": sentence_hash,
                        "draw_role": role,
                        "bucket_role": bucket_role,
                        "draw_stratum": stratum,
                        "partition_bucket": bucket,
                        "cue_evidence": evidence,
                        "source_document_id": source_id,
                        "source_document_sha256": document_hash,
                        "sentence_ordinal": ordinal,
                        "source_document_start": stripped_start,
                        "source_document_end": stripped_end,
                        "annotation_context": text[para_start:para_end],
                        "annotation_context_start": para_start,
                        "annotation_context_kind": window_kind,
                        "source": descriptor,
                        "segmentation": provenance,
                        "supervision_scope": "unlabeled_native_draw_candidate_only",
                    }
                )
                counts[key] += 1
                taken += 1
        if taken:
            documents_used += 1
            if source_documents is not None:
                source_documents[(language, source_id, document_hash)] = {
                    "schema": "pii-final35-native-draw-document/v1",
                    "id": source_id,
                    "lang": language,
                    "text": text,
                    "text_sha256": document_hash,
                    "source": descriptor,
                    "draw_role": role,
                    "bucket_role": bucket_role,
                    "partition_bucket": bucket,
                    "supervision_scope": "unlabeled_native_source_document_only",
                }
    summary = {
        "language": language,
        "documents_scanned": documents_scanned,
        "documents_used": documents_used,
        "rows": len(outputs),
        "counts": {f"{role}/{stratum}": count for (role, stratum), count in sorted(counts.items())},
        "quota_shortfall": {
            f"{role}/{stratum}": quota - counts[(role, stratum)]
            for (role, stratum), quota in sorted(needed.items())
            if counts[(role, stratum)] < quota
        },
        "rejected": dict(sorted(rejected.items())),
        "satisfied": satisfied(),
    }
    return outputs, summary


def parse_quotas(values: list[str]) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    for value in values:
        try:
            key, count = value.split("=")
            role, stratum = key.split("/")
        except ValueError as error:
            raise DrawError(f"quota must look like role/stratum=N: {value!r}") from error
        if (role not in ROLE_BUCKETS and role != TRAINING_ROLE) or stratum not in STRATA:
            raise DrawError(f"unknown role/stratum in quota {value!r}")
        quotas[(role, stratum)] = int(count)
    if not quotas:
        raise DrawError("at least one --quota is required")
    roles = {role for role, _ in quotas}
    if TRAINING_ROLE in roles and roles != {TRAINING_ROLE}:
        raise DrawError(
            "a training draw takes only training/* quotas; evaluation roles are drawn in a separate run "
            "so no document is claimed by two roles"
        )
    return quotas


def draw(args: Any) -> dict[str, Any]:
    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    quotas = parse_quotas(args.quota)
    source_documents = {} if args.source_documents else None
    if args.source_documents and (
        args.source_documents.exists()
        or args.source_documents.resolve() == args.receipt.resolve()
        or args.source_documents.resolve()
        in {(args.output_dir / f"{lang}.jsonl").resolve() for lang in languages}
    ):
        raise DrawError("--source-documents must name a new, distinct output")
    excluded: set[str] = set()
    excluded_source_ids: set[str] = set()
    for path in args.exclude_jsonl:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    for field in ("text_sha256", "source_document_sha256"):
                        if row.get(field):
                            excluded.add(row[field])
                    if isinstance(row.get("text"), str):
                        excluded.add(sha256_text(row["text"]))
                        excluded.add(sha256_text(unicodedata.normalize("NFKC", row["text"])))
                    for source_id in (row.get("source_document_id"), row.get("source", {}).get("id")):
                        if source_id:
                            excluded_source_ids.add(str(source_id))
    splitter = SaTCharacterSpanSplitter(
        SAT_MODEL,
        model_revision=SAT_MODEL_REVISION,
        batch_size=args.segmenter_batch_size,
        device=args.segmenter_device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for language in languages:
        target = args.output_dir / f"{language}.jsonl"
        if target.exists():
            raise DrawError(f"{target} already exists; draws are immutable once written")
        cues = compile_cues(language, load_lexicon(args.cue_lexicon_dir, language))
        rows, summary = draw_language(
            language,
            quotas=quotas,
            cues=cues,
            splitter=splitter,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
            max_documents=args.max_documents_per_language,
            max_sentences_per_document=args.max_sentences_per_document,
            minimum_chars=args.minimum_chars,
            maximum_chars=args.maximum_chars,
            context_chars=args.context_chars,
            reference_minimum=args.reference_minimum,
            excluded_hashes=excluded,
            excluded_source_ids=excluded_source_ids,
            source_documents=source_documents,
        )
        with target.open("x", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["path"] = str(target)
        summary["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        summaries.append(summary)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "proof_data_used": False,
        "role_buckets": {role: list(bounds) for role, bounds in ROLE_BUCKETS.items()},
        "role_rule": "sha256(document text) first eight hex digits modulo 100; a document belongs to exactly one role",
        "strata": list(STRATA),
        "quotas": {f"{role}/{stratum}": count for (role, stratum), count in sorted(quotas.items())},
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "max_documents_per_language": args.max_documents_per_language,
        "max_sentences_per_document": args.max_sentences_per_document,
        "sentence_chars": [args.minimum_chars, args.maximum_chars],
        "whitespace_word_length_filter": {
            "maximum_chars": MAX_WHITESPACE_WORD_CHARS,
            "not_applicable_languages": sorted(NO_WORD_BOUNDARY_LANGUAGES),
        },
        "context_chars": args.context_chars,
        "reference_minimum": args.reference_minimum,
        "identifier_patterns": {name: pattern.pattern for name, pattern in IDENTIFIER_PATTERNS.items()},
        "cue_lexicon_dir": str(args.cue_lexicon_dir),
        "segmenter": segmenter_provenance(splitter),
        "excluded_hashes": len(excluded),
        "excluded_source_ids": len(excluded_source_ids),
        "exclusion_policy": "source-id-and-raw-or-nfkc-text-v1",
        "deduplication_status": "Preliminary source/hash exclusions only; lexical/semantic partial-overlap clearance is required before annotation or split admission.",
        "languages": summaries,
        "supervision": "unlabeled candidates only; annotation plus independent review are required before any row is gold",
    }
    if source_documents is not None:
        args.source_documents.parent.mkdir(parents=True, exist_ok=True)
        with args.source_documents.open("x", encoding="utf-8") as sink:
            for document in source_documents.values():
                sink.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
        receipt["source_documents"] = {
            "path": str(args.source_documents),
            "sha256": hashlib.sha256(args.source_documents.read_bytes()).hexdigest(),
            "rows": len(source_documents),
            "unit": "full_source_document_before_segmentation",
        }
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return receipt


def build_parser() -> Any:
    parser = acli.argument_parser(
        description=__doc__,
        capabilities=("complete",),
        exit_codes={0: "success", 2: "invalid input or existing output"},
    )
    parser.add_argument("--languages", required=True, help="comma-separated BCP-47 codes in draw order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--source-documents",
        type=Path,
        help="Save complete selected source documents for document-level language identification and exact sentence joins",
    )
    parser.add_argument("--cue-lexicon-dir", type=Path, required=True)
    parser.add_argument("--quota", action="append", default=[], help="role/stratum=N (repeatable)")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--max-documents-per-language", type=int, default=20000)
    parser.add_argument("--max-sentences-per-document", type=int, default=2)
    parser.add_argument("--minimum-chars", type=int, default=40)
    parser.add_argument("--maximum-chars", type=int, default=600)
    parser.add_argument("--context-chars", type=int, default=1500)
    parser.add_argument("--reference-minimum", type=int, default=3)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--segmenter-device", default="cpu")
    parser.add_argument("--segmenter-batch-size", type=int, default=32)
    acli.add_standard_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        acli.maybe_complete(parser)
    args = parser.parse_args(argv)
    try:
        result = draw(args)
    except (FileExistsError, OSError, DrawError, ValueError, KeyError) as error:
        acli.die(str(error), 2)
    acli.emit(result, acli.resolve_format(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
