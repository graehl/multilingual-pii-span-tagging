#!/usr/bin/env python3
# acli: 1 complete
"""Select crawl paragraphs that carry rare identifier types, by weighted needles.

Streams the pinned FineWeb (en) / FineWeb2 snapshots in dataset order (or local
JSONL documents), scans a bounded, resumable document range per language, and
scores each paragraph by weighted needle hits. A needle is a surface pattern
with a weight, optionally a checksum or shape validator and optionally required
native cue words near the match. A paragraph whose capped hit weight reaches the
threshold is selected. With --sentences, selected paragraphs are
sentence-segmented with the pinned SaT model into annotation-packet rows, each
carrying its previous sentence as optional context.

The document range is the before-filtering budget: --start skips documents,
--scan bounds how many are read, and --cursor records the next unread index per
language so a later run continues where this one stopped (for smokes and
deliberately small gathers). Documents are normalized to NFKC before any offset
is computed, and only documents whose hash falls in the training rotation
buckets are selectable, as in pii_final35_native_draw. Output rows are
unlabeled candidates; annotation and the partial-overlap gate come after.

Every sentence of a selected paragraph is emitted, with or without a hit: the
paragraph is in-domain for the needle. Annotate a paragraph as one session
(pii_api_label.py --session-field paragraph_id); the default --max-chars matches
that runner's 2,400-character session cap so a selected paragraph fits whole.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import acli
except ImportError:
    sys.path.insert(0, str(Path.home() / "agents"))
    import acli

from scripts.pii_final35_native_draw import (  # noqa: E402
    ROLE_BUCKETS,
    TRAINING_SOURCE_ROLES,
    role_for_document,
)
from scripts.pii_needle_validators import VALIDATORS  # noqa: E402
from scripts.pii_ont3_surface_retrieval import (  # noqa: E402
    document_identity,
    sha256_text,
    source_descriptor,
    stable_id,
)

NEEDLE_SCHEMA = "pii-needle-set/v1"
PARAGRAPH_SCHEMA = "pii-needle-paragraph/v1"
SENTENCE_SCHEMA = "pii-needle-sentence/v1"
CURSOR_SCHEMA = "pii-needle-cursor/v2"
USED_REGION_SCHEMA = "pii-used-region/v1"
RECEIPT_SCHEMA = "pii-needle-select-receipt/v1"
NO_WORD_BOUNDARY_LANGUAGES = frozenset({"zh", "ja", "th"})


class SelectError(ValueError):
    pass


def load_needles(path: Path) -> dict[str, Any]:
    """Load and compile a needle set; reject unknown validators and bad weights."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != NEEDLE_SCHEMA or not config.get("needles"):
        raise SelectError(f"{path}: expected a nonempty {NEEDLE_SCHEMA} needle set")
    compiled = []
    names = set()
    for needle in config["needles"]:
        name = needle["name"]
        if name in names:
            raise SelectError(f"duplicate needle name {name!r}")
        names.add(name)
        weight = float(needle["weight"])
        if weight <= 0:
            raise SelectError(f"needle {name!r} needs a positive weight")
        validator = needle.get("validator")
        if validator is not None and validator not in VALIDATORS:
            raise SelectError(f"needle {name!r} names unknown validator {validator!r}")
        cues = {
            lang: [str(word).casefold() for word in words] for lang, words in needle.get("cues", {}).items()
        }
        compiled.append(
            {
                "name": name,
                "type": needle["type"],
                "pattern": re.compile(needle["pattern"], re.IGNORECASE if needle.get("ignore_case") else 0),
                "weight": weight,
                "validator": VALIDATORS[validator] if validator else None,
                "languages": set(needle["languages"]) if needle.get("languages") else None,
                "cues": cues,
                "cue_window": int(needle.get("cue_window", 40)),
            }
        )
    return {"config": config, "needles": compiled, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cue_near(text: str, start: int, end: int, cues: list[str], window: int) -> bool:
    around = text[max(0, start - window) : end + window].casefold()
    return any(cue in around for cue in cues)


def needle_hits(text: str, language: str, needles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return validated, cue-satisfied hits of every needle that applies to the language."""
    hits = []
    for needle in needles:
        if needle["languages"] is not None and language not in needle["languages"]:
            continue
        cues = needle["cues"].get(language, []) + needle["cues"].get("*", [])
        if needle["cues"] and not cues:
            continue
        for match in needle["pattern"].finditer(text):
            value = match.group(0)
            if needle["validator"] is not None and not needle["validator"](value):
                continue
            if cues and not cue_near(text, match.start(), match.end(), cues, needle["cue_window"]):
                continue
            hits.append(
                {
                    "needle": needle["name"],
                    "type": needle["type"],
                    "start": match.start(),
                    "end": match.end(),
                    "surface": value,
                    "weight": needle["weight"],
                }
            )
    return hits


def score(hits: list[dict[str, Any]], max_hits_per_needle: int) -> float:
    """Sum hit weights, counting at most max_hits_per_needle hits of any one needle."""
    used: Counter[str] = Counter()
    total = 0.0
    for hit in sorted(hits, key=lambda item: -item["weight"]):
        if used[hit["needle"]] < max_hits_per_needle:
            used[hit["needle"]] += 1
            total += hit["weight"]
    return total


def paragraphs(text: str, mode: str) -> list[tuple[int, int]]:
    """Paragraph intervals: nonblank lines, or blank-line separated blocks."""
    separator = re.compile(r"\n" if mode == "line" else r"\n\s*\n")
    spans, cursor = [], 0
    for match in separator.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    trimmed = []
    for start, end in spans:
        piece = text[start:end]
        left = start + len(piece) - len(piece.lstrip())
        right = end - (len(piece) - len(piece.rstrip()))
        if right > left:
            trimmed.append((left, right))
    return trimmed


def stream_identity(language: str, split: str, local: Path | None) -> dict[str, Any]:
    """The pinned document source a locator is relative to."""
    if local is not None:
        return {
            "kind": "local_jsonl",
            "path": str(local.resolve()),
            "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        }
    from scripts.pii_fineweb_sources import (
        FINEWEB2_CONFIGS,
        FINEWEB2_DATASET,
        FINEWEB2_REVISION,
        FINEWEB_EN_DATASET,
        FINEWEB_EN_REVISION,
    )

    if language == "en":
        if split != "train":
            raise SelectError("FineWeb sample-10BT has only a train split")
        return {
            "kind": "hf_parquet",
            "dataset": FINEWEB_EN_DATASET,
            "config": "sample-10BT",
            "revision": FINEWEB_EN_REVISION,
            "split": split,
            "prefix": "sample/10BT/",
        }
    config = FINEWEB2_CONFIGS.get(language)
    if config is None:
        raise SelectError(f"no pinned FineWeb2 config for language {language!r}")
    return {
        "kind": "hf_parquet",
        "dataset": FINEWEB2_DATASET,
        "config": config,
        "revision": FINEWEB2_REVISION,
        "split": split,
        "prefix": f"data/{config}/{split}/",
    }


def read_documents(
    identity: dict[str, Any], start: dict[str, Any] | None
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None, Mapping[str, Any]]]:
    """Yield (locator, next unread locator, document) from `start` in stable file/row order.

    A hub locator is a parquet file of the pinned revision and a row within it;
    a local locator is a line number. Seeking reads only from the row group
    that contains the start row.
    """
    if identity["kind"] == "local_jsonl":
        first = int(start["line"]) if start else 0
        with Path(identity["path"]).open(encoding="utf-8") as source:
            lines = source.readlines()
        for number in range(first, len(lines)):
            if lines[number].strip():
                following = {"line": number + 1} if number + 1 < len(lines) else None
                yield {"line": number}, following, json.loads(lines[number])
        return
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    files = sorted(
        name
        for name in HfApi().list_repo_files(
            identity["dataset"], repo_type="dataset", revision=identity["revision"]
        )
        if name.startswith(identity["prefix"]) and name.endswith(".parquet")
    )
    if not files:
        raise SelectError(f"no parquet files under {identity['prefix']} at {identity['revision']}")
    file_index = files.index(start["file"]) if start else 0
    first_row = int(start["row"]) if start else 0
    filesystem = HfFileSystem()
    for index in range(file_index, len(files)):
        name = files[index]
        parquet = pq.ParquetFile(
            filesystem.open(f"datasets/{identity['dataset']}@{identity['revision']}/{name}")
        )
        total = parquet.metadata.num_rows
        offset = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if offset + size <= first_row:
                offset += size
                continue
            for position, document in enumerate(parquet.read_row_group(group).to_pylist()):
                row = offset + position
                if row < first_row:
                    continue
                if row + 1 < total:
                    following = {"file": name, "row": row + 1}
                else:
                    following = {"file": files[index + 1], "row": 0} if index + 1 < len(files) else None
                yield {"file": name, "row": row}, following, document
            offset += size
        first_row = 0


def load_document(source_locator: Mapping[str, Any]) -> str:
    """Re-read the located document as NFKC text, verified against its recorded hash.

    Offsets recorded with the locator index this text, so any wider window
    around a selected paragraph or sentence can be sliced from it later.
    """
    identity = {
        key: source_locator[key]
        for key in ("kind", "dataset", "revision", "config", "split", "path", "sha256")
        if key in source_locator
    }
    if source_locator["kind"] == "local_jsonl":
        start = {"line": source_locator["line"]}
    else:
        identity["prefix"] = str(Path(source_locator["file"]).parent) + "/"
        start = {"file": source_locator["file"], "row": source_locator["row"]}
    _, _, row = next(read_documents(identity, start))
    text = unicodedata.normalize("NFKC", row["text"])
    if sha256_text(text) != source_locator["document_sha256_nfkc"]:
        raise SelectError("located document does not match its recorded hash")
    return text


class UsedRegions:
    """Document spans and texts already taken by any run or imported pipeline.

    A span is (source document id, start, end) in NFKC document characters; a
    text hash covers the same paragraph or sentence found in another document.
    """

    def __init__(self) -> None:
        self.spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.hashes: set[str] = set()

    def add_row(self, row: Mapping[str, Any]) -> None:
        document = row.get("source_document_id") or (row.get("source") or {}).get("id")
        start, end = row.get("source_document_start"), row.get("source_document_end")
        if document and isinstance(start, int) and isinstance(end, int):
            self.spans[str(document)].append((start, end))
        if document:
            # Previous sentences shown as context outside the selected span are used too.
            self.spans[str(document)].extend((a, b) for a, b in row.get("context_spans", []))
        if row.get("text_sha256"):
            self.hashes.add(row["text_sha256"])
        if isinstance(row.get("text"), str):
            self.hashes.add(sha256_text(unicodedata.normalize("NFKC", row["text"])))
        self.hashes.update(row.get("sentence_sha256", []))
        self.hashes.update(row.get("context_sentence_sha256", []))

    def add_paths(self, paths: Iterable[Path]) -> int:
        count = 0
        for path in paths:
            with Path(path).open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        self.add_row(json.loads(line))
                        count += 1
        return count

    def overlaps(self, document: str, start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in self.spans.get(document, ()))


def select_language(
    language: str,
    documents: Iterable[tuple[dict[str, Any], dict[str, Any] | None, Mapping[str, Any]]],
    *,
    identity: dict[str, Any],
    scan: int,
    needles: list[dict[str, Any]],
    threshold: float,
    max_hits_per_needle: int,
    paragraph_mode: str,
    min_chars: int,
    max_chars: int,
    max_paragraphs_per_document: int,
    roles: Iterable[str] = TRAINING_SOURCE_ROLES,
    context_chars: int = 600,
    used: UsedRegions | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used = used if used is not None else UsedRegions()
    roles = frozenset(roles)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    hit_counts: Counter[str] = Counter()
    first = following = None
    exhausted = True
    for locator, following, row in documents:
        if counts["documents_scanned"] >= scan:
            following = locator
            exhausted = False
            break
        first = first or locator
        counts["documents_scanned"] += 1
        raw = row.get("text")
        if not isinstance(raw, str) or not raw.strip():
            counts["empty_document"] += 1
            continue
        text = unicodedata.normalize("NFKC", raw)
        document_hash = sha256_text(text)
        bucket_role, bucket = role_for_document(document_hash)
        if bucket_role not in roles:
            counts["unselected_bucket_role"] += 1
            continue
        taken = 0
        source_id = document_identity(row, language)
        descriptor = None
        for ordinal, (p_start, p_end) in enumerate(paragraphs(text, paragraph_mode), 1):
            counts["paragraphs_scanned"] += 1
            if not min_chars <= p_end - p_start <= max_chars:
                continue
            paragraph = text[p_start:p_end]
            hits = needle_hits(paragraph, language, needles)
            weight = score(hits, max_hits_per_needle)
            if hits and weight < threshold:
                counts["paragraphs_below_threshold_with_hits"] += 1
            if weight < threshold:
                continue
            paragraph_hash = sha256_text(paragraph)
            if used.overlaps(source_id, p_start, p_end) or paragraph_hash in used.hashes:
                counts["previously_used_paragraph"] += 1
                continue
            if taken >= max_paragraphs_per_document:
                counts["document_paragraph_cap"] += 1
                continue
            taken += 1
            used.spans[source_id].append((p_start, p_end))
            used.hashes.add(paragraph_hash)
            hit_counts.update(hit["needle"] for hit in hits)
            descriptor = descriptor or source_descriptor(row, language, upstream_split=identity.get("split"))
            locator_record = {
                **{key: value for key, value in identity.items() if key != "prefix"},
                **locator,
                "document_id": source_id,
                "url": row.get("url"),
                "dump": row.get("dump"),
                "warc_file": row.get("file_path"),
                "document_sha256_raw": sha256_text(raw),
                "document_sha256_nfkc": document_hash,
                "offsets": "characters of the NFKC-normalized document text",
            }
            selected.append(
                {
                    "schema": PARAGRAPH_SCHEMA,
                    "id": stable_id(
                        {
                            "dataset": descriptor.get("dataset"),
                            "document_id": source_id,
                            "start": p_start,
                            "end": p_end,
                        }
                    ),
                    "lang": language,
                    "text": paragraph,
                    "text_sha256": paragraph_hash,
                    "source_locator": locator_record,
                    "needle_weight": weight,
                    "needle_hits": hits,
                    "paragraph_ordinal": ordinal,
                    "source_document_start": p_start,
                    "source_document_end": p_end,
                    # Seek offset of the preceding context: the segmenter finds the previous
                    # sentence in this window, across paragraph boundaries.
                    "preceding_context": text[max(0, p_start - context_chars) : p_start],
                    "preceding_context_start": max(0, p_start - context_chars),
                    "first_hit_document_offset": p_start + min(hit["start"] for hit in hits),
                    "source_document_id": source_id,
                    "source_document_sha256": document_hash,
                    "draw_role": "training" if bucket_role in TRAINING_SOURCE_ROLES else bucket_role,
                    "bucket_role": bucket_role,
                    "partition_bucket": bucket,
                    "draw_stratum": "needle",
                    "source": descriptor,
                    "supervision_scope": "unlabeled_needle_candidate_only",
                }
            )
        if taken:
            counts["documents_selected"] += 1
    summary = {
        "language": language,
        "range": {"first": first, "next": None if exhausted and following is None else following},
        "counts": dict(counts),
        "paragraphs_selected": len(selected),
        "needle_hits_in_selected": dict(hit_counts.most_common()),
    }
    return selected, summary


def sentence_rows(
    paragraph_rows: list[dict[str, Any]],
    splitter: Any,
    provenance: dict[str, Any],
    used_hashes: set[str] | frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Segment selected paragraphs into packet rows with previous-sentence context.

    A sentence whose text was already used elsewhere stays in its paragraph for
    context but is flagged previously_used so admission can skip it.
    """
    from scripts.pii_text_segmentation import sentence_spans_batch

    texts, slots = [], []
    for row in paragraph_rows:
        has_previous = bool(row["preceding_context"].strip())
        if has_previous:
            texts.append(row["preceding_context"])
        texts.append(row["text"])
        slots.append(has_previous)
    segmented = iter(sentence_spans_batch(texts, [[] for _ in texts], splitter))
    rows = []
    for paragraph, has_previous in zip(paragraph_rows, slots, strict=True):
        previous = previous_span = None
        base = paragraph["source_document_start"]
        if has_previous:
            previous_spans = next(segmented)
            if previous_spans:
                last_start, last_end = previous_spans[-1]
                piece = paragraph["preceding_context"][last_start:last_end]
                previous = piece.strip() or None
                if previous is not None:
                    offset = paragraph["preceding_context_start"] + last_start
                    left = offset + len(piece) - len(piece.lstrip())
                    previous_span = [left, left + len(previous)]
        spans = next(segmented)
        for ordinal, (start, end) in enumerate(spans, 1):
            piece = paragraph["text"][start:end]
            left = start + len(piece) - len(piece.lstrip())
            right = end - (len(piece) - len(piece.rstrip()))
            if right <= left:
                continue
            sentence = paragraph["text"][left:right]
            hits = [hit for hit in paragraph["needle_hits"] if left <= hit["start"] and hit["end"] <= right]
            rows.append(
                {
                    "schema": SENTENCE_SCHEMA,
                    "id": stable_id({"paragraph": paragraph["id"], "start": left, "end": right}),
                    "lang": paragraph["lang"],
                    "bcp47": paragraph["lang"],
                    "text": sentence,
                    "text_sha256": sha256_text(sentence),
                    "previously_used": sha256_text(sentence) in used_hashes,
                    "previous_sentence": previous,
                    # NFKC document span of previous_sentence (None when absent).
                    "previous_sentence_document_span": previous_span,
                    "needle_hits": [
                        {**hit, "start": hit["start"] - left, "end": hit["end"] - left} for hit in hits
                    ],
                    "paragraph_id": paragraph["id"],
                    "sentence_ordinal_in_paragraph": ordinal,
                    "source_document_start": base + left,
                    "source_document_end": base + right,
                    "annotation_context": paragraph["text"],
                    "annotation_context_start": base,
                    "annotation_context_kind": "needle_paragraph",
                    "segmentation": provenance,
                    **{
                        key: paragraph[key]
                        for key in (
                            "source_locator",
                            "source_document_id",
                            "source_document_sha256",
                            "draw_role",
                            "bucket_role",
                            "partition_bucket",
                            "draw_stratum",
                            "source",
                            "supervision_scope",
                        )
                    },
                }
            )
            previous, previous_span = sentence, [base + left, base + right]
    return rows


def read_cursor(
    path: Path | None, languages: list[str], identities: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any] | None]:
    """Next unread locator per language; None starts at the first document."""
    if path is None or not path.exists():
        return {language: None for language in languages}
    cursor = json.loads(path.read_text(encoding="utf-8"))
    if cursor.get("schema") != CURSOR_SCHEMA:
        raise SelectError(f"{path}: not a {CURSOR_SCHEMA} cursor")
    positions: dict[str, dict[str, Any] | None] = {}
    for language in languages:
        entry = cursor["languages"].get(language)
        if entry is None:
            positions[language] = None
            continue
        if entry["stream"] != identities[language]:
            raise SelectError(f"{path}: cursor for {language} was recorded on a different stream")
        if entry["next"] is None:
            raise SelectError(f"{path}: {language} stream is exhausted")
        positions[language] = entry["next"]
    return positions


def locator_label(locator: dict[str, Any] | None) -> str:
    if locator is None:
        return "start"
    if "line" in locator:
        return f"line{locator['line']}"
    return f"{Path(locator['file']).stem}-{locator['row']}"


def write_cursor(path: Path, updates: dict[str, dict[str, Any]]) -> None:
    cursor = {"schema": CURSOR_SCHEMA, "languages": {}}
    if path.exists():
        cursor = json.loads(path.read_text(encoding="utf-8"))
    cursor["languages"].update(updates)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(cursor, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: Any) -> dict[str, Any]:
    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    local = {}
    for item in args.local_jsonl:
        language, _, path = item.partition("=")
        local[language] = Path(path)
    needles = load_needles(args.needles)
    if args.types:
        wanted = {name.strip() for name in args.types.split(",") if name.strip()}
        unknown = wanted - {needle["type"] for needle in needles["needles"]}
        if unknown:
            raise SelectError(f"no needle targets type(s): {', '.join(sorted(unknown))}")
        needles["needles"] = [needle for needle in needles["needles"] if needle["type"] in wanted]
    identities = {
        language: stream_identity(language, args.split, local.get(language)) for language in languages
    }
    positions = read_cursor(args.cursor, languages, identities)
    if args.start_file is not None:
        positions = {language: {"file": args.start_file, "row": args.start_row} for language in languages}
    elif args.start_line is not None:
        positions = {language: {"line": args.start_line} for language in languages}
    if args.all_roles and args.roles:
        raise SelectError("--all-roles and --roles are exclusive")
    roles = tuple(ROLE_BUCKETS) if args.all_roles else TRAINING_SOURCE_ROLES
    if args.roles:
        roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
        if unknown := set(roles) - set(ROLE_BUCKETS):
            raise SelectError(f"unknown bucket role(s): {', '.join(sorted(unknown))}")
    used = UsedRegions()
    ledger_rows = used.add_paths([args.used_ledger] if args.used_ledger and args.used_ledger.exists() else [])
    imported_rows = used.add_paths(args.exclude_jsonl)
    prior_hashes = frozenset(used.hashes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splitter = provenance = None
    if args.sentences:
        from scripts.pii_text_segmentation import (
            SAT_MODEL,
            SAT_MODEL_REVISION,
            SaTCharacterSpanSplitter,
            segmenter_provenance,
        )

        splitter = SaTCharacterSpanSplitter(
            SAT_MODEL, model_revision=SAT_MODEL_REVISION, batch_size=32, device=args.segmenter_device
        )
        provenance = segmenter_provenance(splitter)
    summaries, cursor_updates = [], {}
    ledger_entries = []
    for language in languages:
        stem = f"{language}.{locator_label(positions[language])}.n{args.scan}"
        paragraph_path = args.output_dir / f"{stem}.paragraphs.jsonl"
        if paragraph_path.exists():
            raise SelectError(f"{paragraph_path} already exists; selections are immutable")
        rows, summary = select_language(
            language,
            read_documents(identities[language], positions[language]),
            identity=identities[language],
            used=used,
            scan=args.scan,
            needles=needles["needles"],
            threshold=args.threshold,
            max_hits_per_needle=args.max_hits_per_needle,
            paragraph_mode=args.paragraph_mode,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            max_paragraphs_per_document=args.max_paragraphs_per_document,
            roles=roles,
            context_chars=args.context_chars,
        )
        with paragraph_path.open("x", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["paragraphs"] = {
            "path": str(paragraph_path),
            "sha256": hashlib.sha256(paragraph_path.read_bytes()).hexdigest(),
        }
        sentence_hashes: dict[str, list[str]] = defaultdict(list)
        context: dict[str, tuple[list[int], str]] = {}
        if args.sentences:
            sentences = sentence_rows(rows, splitter, provenance, prior_hashes)
            starts = {row["id"]: row["source_document_start"] for row in rows}
            for sentence in sentences:
                sentence_hashes[sentence["paragraph_id"]].append(sentence["text_sha256"])
                span = sentence["previous_sentence_document_span"]
                if span is not None and span[0] < starts[sentence["paragraph_id"]]:
                    context[sentence["paragraph_id"]] = (span, sha256_text(sentence["previous_sentence"]))
            sentence_path = args.output_dir / f"{stem}.sentences.jsonl"
            with sentence_path.open("x", encoding="utf-8") as sink:
                for row in sentences:
                    sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            summary["sentences"] = {
                "path": str(sentence_path),
                "sha256": hashlib.sha256(sentence_path.read_bytes()).hexdigest(),
                "rows": len(sentences),
                "rows_with_hits": sum(bool(row["needle_hits"]) for row in sentences),
                "rows_previously_used": sum(row["previously_used"] for row in sentences),
            }
        for row in rows:
            ledger_entries.append(
                {
                    "schema": USED_REGION_SCHEMA,
                    "use": "needle-select",
                    "receipt": str(args.receipt),
                    "lang": language,
                    "source_locator": row["source_locator"],
                    "source_document_id": row["source_document_id"],
                    "source_document_sha256": row["source_document_sha256"],
                    "source_document_start": row["source_document_start"],
                    "source_document_end": row["source_document_end"],
                    "text_sha256": row["text_sha256"],
                    "sentence_sha256": sentence_hashes.get(row["id"], []),
                    # The previous sentence preceding the paragraph, when segmented.
                    "context_spans": [context[row["id"]][0]] if row["id"] in context else [],
                    "context_sentence_sha256": [context[row["id"]][1]] if row["id"] in context else [],
                }
            )
        summary["stream"] = identities[language]
        summaries.append(summary)
        cursor_updates[language] = {"stream": identities[language], "next": summary["range"]["next"]}
    if args.used_ledger is not None:
        with args.used_ledger.open("a", encoding="utf-8") as sink:
            for entry in ledger_entries:
                sink.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    if args.cursor is not None:
        write_cursor(args.cursor, cursor_updates)
    return {
        "schema": RECEIPT_SCHEMA,
        "needles": {"path": str(args.needles), "sha256": needles["sha256"]},
        "types": sorted({needle["type"] for needle in needles["needles"]}),
        "threshold": args.threshold,
        "max_hits_per_needle": args.max_hits_per_needle,
        "paragraph_mode": args.paragraph_mode,
        "paragraph_chars": [args.min_chars, args.max_chars],
        "context_chars": args.context_chars,
        "max_paragraphs_per_document": args.max_paragraphs_per_document,
        "roles": sorted(roles),
        "normalization": "NFKC before offsets",
        "segmentation": provenance,
        "cursor": str(args.cursor) if args.cursor else None,
        "used_regions": {
            "ledger": str(args.used_ledger) if args.used_ledger else None,
            "ledger_rows_loaded": ledger_rows,
            "imported_rows": imported_rows,
            "imports": [str(path) for path in args.exclude_jsonl],
            "appended": len(ledger_entries) if args.used_ledger else 0,
        },
        "languages": summaries,
        "supervision": "unlabeled candidates; annotation and the partial-overlap gate are required before admission",
    }


def build_parser() -> Any:
    parser = acli.argument_parser(
        description=__doc__,
        capabilities=("complete",),
        exit_codes={0: "success", 2: "invalid input or existing output"},
    )
    parser.add_argument("--languages", required=True, help="comma-separated BCP-47 codes")
    parser.add_argument("--needles", type=Path, required=True, help="pii-needle-set/v1 JSON")
    parser.add_argument(
        "--types",
        help="comma-separated primary types; only needles of these types score (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--scan", type=int, required=True, help="documents to read per language before filtering"
    )
    parser.add_argument("--start-file", help="parquet file (repo path) to start in; overrides --cursor")
    parser.add_argument("--start-row", type=int, default=0, help="row within --start-file")
    parser.add_argument(
        "--start-line", type=int, help="first line of a --local-jsonl source; overrides --cursor"
    )
    parser.add_argument("--cursor", type=Path, help="per-language next-locator file, read and updated")
    parser.add_argument(
        "--used-ledger",
        type=Path,
        help="append-only JSONL of used document regions; read before selecting, extended after",
    )
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        type=Path,
        default=[],
        help="rows already used by other pipelines (draws, packets): their document spans and texts are skipped",
    )
    parser.add_argument("--split", default="train", help="upstream split (FineWeb2 has train and test)")
    parser.add_argument(
        "--local-jsonl",
        action="append",
        default=[],
        help="lang=path document JSONL instead of the hub stream",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0, help="minimum capped needle weight to select a paragraph"
    )
    parser.add_argument(
        "--max-hits-per-needle", type=int, default=2, help="hits of one needle counted toward the threshold"
    )
    parser.add_argument("--paragraph-mode", choices=("line", "blank-line"), default="line")
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2400,
        help="longest paragraph selected; keep at or under the annotation session's source-character cap",
    )
    parser.add_argument("--max-paragraphs-per-document", type=int, default=2)
    parser.add_argument(
        "--context-chars",
        type=int,
        default=600,
        help="preceding document text kept and segmented for the first sentence's previous sentence",
    )
    parser.add_argument(
        "--all-roles",
        action="store_true",
        help="also select development/final bucket documents (never for training)",
    )
    parser.add_argument(
        "--roles",
        help="comma-separated bucket roles to select, e.g. development for held-out rows "
        "(default: the training rotations)",
    )
    parser.add_argument(
        "--sentences",
        action="store_true",
        help="also write SaT-segmented sentence rows with previous-sentence context",
    )
    parser.add_argument("--segmenter-device", default="cpu")
    acli.add_standard_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        acli.maybe_complete(parser)
    args = parser.parse_args(argv)
    try:
        if args.scan <= 0 or args.threshold <= 0 or args.max_hits_per_needle <= 0:
            raise SelectError("--scan, --threshold and --max-hits-per-needle must be positive")
        if args.receipt.exists():
            raise SelectError(f"{args.receipt} already exists")
        result = run(args)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except (OSError, SelectError, ValueError, KeyError) as error:
        acli.die(str(error), 2)
    acli.emit(result, acli.resolve_format(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
