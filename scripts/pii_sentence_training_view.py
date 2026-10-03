#!/usr/bin/env python
"""Materialize provenance-complete sentence rows for PII training.

This is a view transform over already partitioned training files. It retains
every source field, identifies the exact intake line, suppresses sentence
boundaries inside gold spans, and rebases labels. A sentence beyond tokenizer
capacity is explicitly divided at lexical boundaries that do not bisect gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from scripts.pii_text_segmentation import (
    SaTCharacterSpanSplitter,
    rebase_spans,
    segmenter_provenance,
    sentence_spans_batch,
    source_content_id,
)

CAPACITY_EXCEPTION = "span-safe-lexical-token-capacity-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_provenance(tokenizer, source: str) -> dict:
    identity = {
        "add_special_tokens": True,
        "class": type(tokenizer).__name__,
        "model_max_length": tokenizer.model_max_length,
        "name_or_path": tokenizer.name_or_path,
        "source": source,
    }
    revision = (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if revision:
        identity["revision"] = str(revision)
        return identity
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        identity["backend_sha256"] = hashlib.sha256(backend.to_str().encode()).hexdigest()
        return identity
    raise ValueError("tokenizer has neither an immutable revision nor a fingerprintable backend")


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])


def capacity_split_intervals(
    text: str,
    spans: list[list],
    tokenizer,
    max_tokens: int,
    *,
    fallback_boundaries: tuple[int, ...] = (),
) -> list[tuple[int, int, int]]:
    """Split one over-capacity sentence at lexical, span-safe boundaries.

    ``fallback_boundaries`` are source-token boundaries for scripts whose
    reconstructed text has no lexical delimiters. They are considered only
    when the current interval has no usable whitespace or punctuation edge.
    """
    pending = [(0, len(text))]
    accepted = []
    while pending:
        start, end = pending.pop()
        count = token_count(tokenizer, text[start:end])
        if count <= max_tokens:
            accepted.append((start, end, count))
            continue
        midpoint = (start + end) / 2
        candidates = []
        for boundary in range(start + 1, end):
            if not (
                text[boundary - 1].isspace() or text[boundary].isspace() or text[boundary - 1] in ";,:.!?)]}"
            ):
                continue
            if not text[start:boundary].strip() or not text[boundary:end].strip():
                continue
            if any(span_start < boundary < span_end for span_start, span_end, *_ in spans):
                continue
            candidates.append(boundary)
        if candidates and fallback_boundaries:
            minimum_side = min(16, max(2, (end - start) // 10))
            candidates = [
                boundary for boundary in candidates if min(boundary - start, end - boundary) >= minimum_side
            ]
        if not candidates:
            candidates = [
                boundary
                for boundary in fallback_boundaries
                if start < boundary < end
                and text[start:boundary].strip()
                and text[boundary:end].strip()
                and not any(span_start < boundary < span_end for span_start, span_end, *_ in spans)
            ]
        if not candidates:
            raise ValueError(
                "over-capacity sentence has no lexical boundary and no declared source-token "
                "boundary outside its gold spans: "
                f"{count} tokens against capacity {max_tokens}"
            )
        shortlist = sorted(candidates, key=lambda boundary: (abs(boundary - midpoint), boundary))[:32]
        scored = []
        for boundary in shortlist:
            left_count = token_count(tokenizer, text[start:boundary])
            right_count = token_count(tokenizer, text[boundary:end])
            scored.append((max(left_count, right_count), abs(boundary - midpoint), boundary))
        _largest_child, _distance, boundary = min(scored)
        pending.extend(((boundary, end), (start, boundary)))
    accepted.sort()
    if "".join(text[start:end] for start, end, _count in accepted) != text:
        raise AssertionError("capacity pieces do not reconstruct their sentence")
    return accepted


def write_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def parse_source_file(source_dir: Path, value: str) -> tuple[str, Path]:
    if "=" in value:
        output_name, source_value = value.split("=", 1)
    else:
        output_name = value
        source_value = value
    if not output_name or Path(output_name).name != output_name:
        raise ValueError(f"output name must be one filename: {output_name!r}")
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = source_dir / source_path
    if not source_path.is_file():
        raise ValueError(f"training source is not a file: {source_path}")
    return output_name, source_path


def validate_source_row(row: dict, *, locator: str) -> None:
    text = row.get("text")
    spans = row.get("spans")
    language = row.get("lang")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{locator}: text must be a nonempty string")
    if not isinstance(spans, list):
        raise ValueError(f"{locator}: spans must be a list")
    if not isinstance(language, str) or not language:
        raise ValueError(f"{locator}: lang must be a nonempty string")
    if any(
        field in row
        for field in ("sampling_intake_factor", "sentence_ordinal", "segmenter_name", "text_view")
    ):
        raise ValueError(f"{locator}: source is already a derived text view")


def sentence_rows_for_batch(
    rows: list[dict],
    line_numbers: list[int],
    *,
    source_name: str,
    source_sha256: str,
    splitter,
    tokenizer,
    max_tokens: int,
) -> tuple[list[dict], dict]:
    texts = [row["text"] for row in rows]
    offsets_by_row = list(sentence_spans_batch(texts, [row["spans"] for row in rows], splitter))
    provenance = segmenter_provenance(splitter)
    output_rows = []
    source_token_lengths = [
        len(input_ids)
        for input_ids in tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )["input_ids"]
    ]
    for row, line_number, offsets in zip(rows, line_numbers, offsets_by_row, strict=True):
        locator = f"{source_name}:{line_number}"
        if not offsets:
            raise ValueError(f"{locator}: sentence splitter returned no intervals")
        source_hash = hashlib.sha256(row["text"].encode()).hexdigest()
        declared_hash = row.get("source_text_sha256")
        if declared_hash is not None and declared_hash != source_hash:
            raise ValueError(f"{locator}: declared source_text_sha256 does not match text")
        intake_row_id = locator
        for ordinal, (start, end) in enumerate(offsets, 1):
            text = row["text"][start:end]
            if not text.strip():
                raise ValueError(f"{locator}: sentence {ordinal} contains only whitespace")
            output_rows.append(
                {
                    **row,
                    "intake_row_id": intake_row_id,
                    "intake_line_number": line_number,
                    "intake_source_file": source_name,
                    "intake_source_file_sha256": source_sha256,
                    "sentence_ordinal": ordinal,
                    "source_content_id": source_content_id(row),
                    "source_end": end,
                    "source_sentence_count": len(offsets),
                    "source_start": start,
                    "source_text_sha256": source_hash,
                    "spans": rebase_spans(row, start, end),
                    "text": text,
                    "text_view": "sentence",
                    "view_group_id": intake_row_id,
                    "view_offset": start,
                    **provenance,
                }
            )

    encoded = tokenizer(
        [row["text"] for row in output_rows],
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )["input_ids"]
    token_lengths = [len(input_ids) for input_ids in encoded]
    expanded_rows = []
    overcapacity_sentences = 0
    overcapacity_pieces = 0
    for row, count in zip(output_rows, token_lengths, strict=True):
        if count <= max_tokens:
            pieces = [(0, len(row["text"]), count)]
        else:
            pieces = capacity_split_intervals(row["text"], row["spans"], tokenizer, max_tokens)
            overcapacity_sentences += 1
            overcapacity_pieces += len(pieces)
        for piece_ordinal, (start, end, piece_count) in enumerate(pieces, 1):
            piece = {
                **row,
                "sentence_piece_count": len(pieces),
                "sentence_piece_ordinal": piece_ordinal,
                "source_end": row["source_start"] + end,
                "source_start": row["source_start"] + start,
                "spans": rebase_spans(row, start, end),
                "text": row["text"][start:end],
                "view_offset": row["source_start"] + start,
                "view_token_count": piece_count,
            }
            if len(pieces) > 1:
                piece["overcapacity_exception"] = CAPACITY_EXCEPTION
                piece["text_view"] = "sentence-overcapacity-piece"
            expanded_rows.append(piece)
    output_rows = expanded_rows

    rows_by_intake = {}
    for row in output_rows:
        rows_by_intake.setdefault(row["intake_row_id"], []).append(row)
    for source_row, line_number, source_token_count in zip(
        rows, line_numbers, source_token_lengths, strict=True
    ):
        intake_row_id = f"{source_name}:{line_number}"
        derived = rows_by_intake[intake_row_id]
        for view_index, row in enumerate(derived):
            row["intake_token_count"] = source_token_count
            row["view_index"] = view_index
        if "".join(row["text"] for row in derived) != source_row["text"]:
            raise AssertionError("derived sentence texts do not reconstruct their intake row")
        source_spans = [list(span) for span in source_row["spans"]]
        restored_spans = sorted(
            [
                [span[0] + row["source_start"], span[1] + row["source_start"], *span[2:]]
                for row in derived
                for span in row["spans"]
            ],
            key=lambda span: (span[0], span[1], span[2:]),
        )
        if restored_spans != source_spans:
            raise AssertionError("derived sentence spans do not reconstruct their intake row")

    return output_rows, {
        "characters": sum(len(row["text"]) for row in output_rows),
        "max_sentence_characters": max(map(len, (row["text"] for row in output_rows))),
        "max_original_sentence_tokens": max(token_lengths),
        "max_sentence_tokens": max(row["view_token_count"] for row in output_rows),
        "overcapacity_pieces": overcapacity_pieces,
        "overcapacity_sentences": overcapacity_sentences,
        "output_rows": len(output_rows),
        "spans": sum(len(row["spans"]) for row in output_rows),
    }


def materialize_file(
    source_path: Path,
    destination_path: Path,
    *,
    source_name: str,
    splitter,
    tokenizer,
    max_tokens: int,
    batch_size: int,
    include_languages: frozenset[str],
    include_sources: frozenset[str],
) -> dict:
    source_sha256 = sha256_file(source_path)
    partial_path = destination_path.with_suffix(destination_path.suffix + ".partial")
    input_rows = 0
    output_rows = 0
    input_characters = 0
    output_characters = 0
    input_spans = 0
    output_spans = 0
    source_rows = 0
    excluded_rows = 0
    max_original_sentence_tokens = 0
    max_sentence_characters = 0
    max_sentence_tokens = 0
    overcapacity_pieces = 0
    overcapacity_sentences = 0
    languages = Counter()
    batch_rows: list[dict] = []
    batch_lines: list[int] = []

    def flush(destination) -> None:
        nonlocal output_rows, output_characters, output_spans
        nonlocal max_original_sentence_tokens, max_sentence_characters, max_sentence_tokens
        nonlocal overcapacity_pieces, overcapacity_sentences
        if not batch_rows:
            return
        derived, stats = sentence_rows_for_batch(
            batch_rows,
            batch_lines,
            source_name=source_name,
            source_sha256=source_sha256,
            splitter=splitter,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        for row in derived:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_rows += stats["output_rows"]
        output_characters += stats["characters"]
        output_spans += stats["spans"]
        max_original_sentence_tokens = max(
            max_original_sentence_tokens, stats["max_original_sentence_tokens"]
        )
        max_sentence_characters = max(max_sentence_characters, stats["max_sentence_characters"])
        max_sentence_tokens = max(max_sentence_tokens, stats["max_sentence_tokens"])
        overcapacity_pieces += stats["overcapacity_pieces"]
        overcapacity_sentences += stats["overcapacity_sentences"]
        batch_rows.clear()
        batch_lines.clear()

    with (
        source_path.open(encoding="utf-8") as source,
        partial_path.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{source_name}:{line_number}: blank JSONL line")
            row = json.loads(line)
            validate_source_row(row, locator=f"{source_name}:{line_number}")
            source_rows += 1
            if include_languages and row["lang"] not in include_languages:
                excluded_rows += 1
                continue
            if include_sources and row.get("src") not in include_sources:
                excluded_rows += 1
                continue
            input_rows += 1
            input_characters += len(row["text"])
            input_spans += len(row["spans"])
            languages[row["lang"]] += 1
            batch_rows.append(row)
            batch_lines.append(line_number)
            if len(batch_rows) == batch_size:
                flush(destination)
        flush(destination)
    if not input_rows:
        raise ValueError(f"empty JSONL input: {source_path}")
    if input_characters != output_characters or input_spans != output_spans:
        raise AssertionError("file-level character or span conservation failed")
    partial_path.replace(destination_path)
    return {
        "excluded_rows": excluded_rows,
        "input_characters": input_characters,
        "input_path": str(source_path.resolve()),
        "input_rows": input_rows,
        "input_sha256": source_sha256,
        "input_spans": input_spans,
        "language_intake_rows": dict(sorted(languages.items())),
        "max_original_sentence_tokens": max_original_sentence_tokens,
        "max_sentence_characters": max_sentence_characters,
        "max_sentence_tokens": max_sentence_tokens,
        "overcapacity_pieces": overcapacity_pieces,
        "overcapacity_sentences": overcapacity_sentences,
        "output_characters": output_characters,
        "output_path": str(destination_path.resolve()),
        "output_rows": output_rows,
        "output_sha256": sha256_file(destination_path),
        "output_spans": output_spans,
        "selection": {
            "include_languages": sorted(include_languages),
            "include_sources": sorted(include_sources),
            "source_field": "src",
        },
        "source_rows": source_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="source filename, or output-name=source-path; repeat for every training pool",
    )
    parser.add_argument(
        "--copy",
        action="append",
        default=[],
        help="non-training file copied byte-for-byte from source-dir",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--include-language",
        action="append",
        default=[],
        help="retain only this exact lang value; repeatable and applied to every --input",
    )
    parser.add_argument(
        "--include-source",
        action="append",
        default=[],
        help="retain only this exact src value; repeatable and applied to every --input",
    )
    parser.add_argument(
        "--segmenter-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device for the frozen SaT sentence segmenter",
    )
    args = parser.parse_args()
    if not os.environ.get("AGENTCTL_RUN_ID"):
        parser.error("sentence training-view production requires a tracked agentctl run")
    if args.max_tokens <= 2:
        parser.error("--max-tokens must leave room for content and special tokens")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.destination.exists():
        parser.error(f"--destination already exists: {args.destination}")
    include_languages = frozenset(args.include_language)
    include_sources = frozenset(args.include_source)
    if len(include_languages) != len(args.include_language):
        parser.error("--include-language values must be unique")
    if len(include_sources) != len(args.include_source):
        parser.error("--include-source values must be unique")

    try:
        inputs = [parse_source_file(args.source_dir, value) for value in args.input]
        copies = [parse_source_file(args.source_dir, value) for value in args.copy]
    except ValueError as error:
        parser.error(str(error))
    names = [name for name, _ in [*inputs, *copies]]
    if len(names) != len(set(names)):
        parser.error("destination filenames must be unique")

    args.destination.mkdir(parents=True)
    splitter = SaTCharacterSpanSplitter(
        batch_size=args.batch_size,
        split_on_input_newlines=True,
        device=args.segmenter_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    files = {}
    for output_name, source_path in inputs:
        files[output_name] = materialize_file(
            source_path,
            args.destination / output_name,
            source_name=output_name,
            splitter=splitter,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            include_languages=include_languages,
            include_sources=include_sources,
        )
        print(
            f"SENTENCE-VIEW: {output_name} "
            f"{files[output_name]['input_rows']} intake rows -> "
            f"{files[output_name]['output_rows']} sentence rows; "
            f"max={files[output_name]['max_sentence_tokens']} tokens; "
            f"capacity-exceptions={files[output_name]['overcapacity_sentences']}",
            flush=True,
        )
    copied = {}
    for output_name, source_path in copies:
        destination_path = args.destination / output_name
        shutil.copyfile(source_path, destination_path)
        copied[output_name] = {
            "input_path": str(source_path.resolve()),
            "output_path": str(destination_path.resolve()),
            "sha256": sha256_file(destination_path),
        }

    receipt = {
        "batch_size": args.batch_size,
        "copied_files": copied,
        "files": files,
        "language_provenance": "supplied lang field on each already-partitioned intake row",
        "input_selection": {
            "include_languages": sorted(include_languages),
            "include_sources": sorted(include_sources),
            "semantics": "logical AND across nonempty filters; exact equality within each field",
            "source_field": "src",
        },
        "max_tokens": args.max_tokens,
        "operation": "provenance-complete sentence training view",
        "overcapacity_exception": {
            "policy": CAPACITY_EXCEPTION,
            "semantics": "lexical boundary nearest balanced token mass, outside every gold span",
        },
        "run_id": os.environ["AGENTCTL_RUN_ID"],
        "sampling": {
            "intake_row_reweighting": "none",
            "row_unit": "sentence, except explicit over-capacity sentence pieces",
        },
        "schema": "pii-sentence-training-view-v3",
        "segmenter": segmenter_provenance(splitter),
        "tokenizer": tokenizer_provenance(tokenizer, args.tokenizer),
        "validation": [
            "each input text is exactly reconstructed by its sentence rows",
            "each input span is preserved exactly once and rebased without clipping",
            "every sentence identifies its intake file, line, source interval, and source hash",
            "sentence rows carry no factor that equalizes original intake-row mass",
            "no emitted sentence row exceeds the declared tokenizer capacity",
        ],
    }
    write_json(args.destination / "sentence-view-receipt.json", receipt)
    print(
        f"SENTENCE-VIEW: complete destination={args.destination} "
        f"receipt={args.destination / 'sentence-view-receipt.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
