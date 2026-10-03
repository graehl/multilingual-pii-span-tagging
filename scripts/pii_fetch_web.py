#!/usr/bin/env python3
"""Fetch the public web text of O4's teacher-annotated training rows.

About a quarter of O4's training draws are FineWeb and FineWeb-2 sentences
labeled by an LLM teacher. The labels are not released, but the text is
public: `records/o4-training-membership.csv` gives each row's dataset,
config, revision, split, record id and character offsets inside the record,
plus the SHA-256 of the exact training text. This command streams each needed
config, cuts the rows out of their records and keeps only rows whose hash
matches, so the output is exactly O4's text. Annotate it with your own
teacher (`annotate --web-receipt`) and add it with `mixture --annotated`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acli

MEMBERSHIP = ROOT / "research/pii/frontier/software/records/o4-training-membership.csv"
# Seeds and shuffle buffer of the paper's FineWeb draws (text-free draw receipts).
DRAW_STREAMS = (
    (20260903, 10000),  # native draw v1
    (20260904, 10000),  # training draw v1 rest3
    (20260909, 10000),  # training draw v2
    (20260927, 10000),  # general top-up
    (20260830, 1000),  # surface retrieval default
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wanted_rows(membership: Path, languages: set[str] | None):
    groups = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    with membership.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["locator_status"] != "upstream_pointer":
                continue
            if languages and row["lang"] not in languages:
                continue
            if not row["upstream_document_start"]:
                skipped["no_document_offsets"] += 1
                continue
            key = (row["dataset"], row["dataset_config"], row["dataset_revision"], row["upstream_split"])
            groups[key][row["upstream_record_id"]].append(row)
    return groups, skipped


def row_text(document: str, row: dict) -> tuple[str | None, bool]:
    """The row's exact text inside its record, and whether its offsets needed repair.

    Some intake batches recorded document offsets shifted by a character or
    two; the hash decides, so a nearby same-length slice is accepted only
    when it reproduces the training text exactly.
    """
    start, end = int(row["upstream_document_start"]), int(row["upstream_document_end"])
    target = row["training_text_sha256"]
    # Some intake stored NFKC-normalized text (full-width punctuation to ASCII)
    # with offsets on the normalized document; try that form second.
    for form, text_source in (("raw", document), ("nfkc", unicodedata.normalize("NFKC", document))):
        for shift in (0, *(sign * step for step in range(1, 33) for sign in (-1, 1))):
            if start + shift < 0 or end + shift > len(text_source):
                continue
            text = text_source[start + shift : end + shift]
            if row["window_start"]:
                text = text[int(row["window_start"]) : int(row["window_end"])]
            if sha256_text(text) == target:
                return text, shift != 0 or form != "raw"
    return None, False


def draw_seed(seed: int, language: str) -> int:
    """The per-language stream seed the original FineWeb draws used."""
    return int.from_bytes(hashlib.sha256(f"{seed}\0{language}".encode()).digest()[:8], "big")


def lookup(dataset: str, config: str, revision: str, split: str, wanted: set[str], threads: int):
    """Records with the wanted ids, found by reading only the id column of every data file.

    Slower than a lucky stream scan but exhaustive: rows the recorded draws
    cannot replay (their draw is not recorded) are still found exactly.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from datasets import load_dataset_builder
    from huggingface_hub import HfFileSystem

    files = load_dataset_builder(dataset, config, revision=revision).config.data_files[split]

    def search(url: str) -> list[dict]:
        found = []
        with HfFileSystem().open(url.removeprefix("hf://")) as handle:
            parquet = pq.ParquetFile(handle)
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(group, columns=["id"]).column("id").to_pylist()
                if wanted.intersection(ids):
                    table = parquet.read_row_group(group, columns=["id", "text"]).to_pylist()
                    found.extend(record for record in table if record["id"] in wanted)
        return found

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for records in pool.map(search, files):
            yield from records


def fetch(args) -> dict:
    from datasets import load_dataset

    groups, skipped = wanted_rows(args.membership, set(args.language) if args.language else None)
    args.out.mkdir(parents=True, exist_ok=False)
    rows_path = args.out / "rows.jsonl"
    report = {}
    matched = []
    with rows_path.open("x", encoding="utf-8") as sink:
        for (dataset, config, revision, split), targets in sorted(groups.items()):
            started = time.time()
            remaining = dict(targets)
            language = next(iter(targets.values()))[0]["lang"]
            stats = Counter()

            def scan(stream, label: str, limit: int = args.max_scan) -> None:
                scanned = 0
                for record in stream:
                    scanned += 1
                    rows = remaining.pop(record["id"], None)
                    if rows:
                        stats[f"documents_found_{label}"] += 1
                        for row in rows:
                            text, repaired = row_text(record["text"], row)
                            if text is None:
                                stats["rows_hash_mismatch"] += 1
                                continue
                            stats["rows_matched"] += 1
                            stats["rows_offset_repaired"] += repaired
                            output = {
                                "id": row["training_id"],
                                "pool_row_1based": int(row["pool_row_1based"]),
                                "text": text,
                                "lang": row["lang"],
                                "source": {
                                    "dataset": dataset,
                                    "config": config,
                                    "split": split,
                                    "id": record["id"],
                                },
                            }
                            sink.write(json.dumps(output, ensure_ascii=False) + "\n")
                            matched.append((output["id"], output["lang"], row["training_text_sha256"]))
                    if not remaining or scanned >= limit:
                        break
                stats[f"records_scanned_{label}"] += scanned

            base = load_dataset(dataset, config, revision=revision, split=split, streaming=True)
            scan(base, "plain")
            # Rows drawn through a seeded shuffle sit wherever that shuffle put
            # them; replay each recorded draw's stream for what is still missing.
            for seed, buffer_size in DRAW_STREAMS:
                if not remaining:
                    break
                shuffled = base.shuffle(seed=draw_seed(seed, language), buffer_size=buffer_size)
                scan(shuffled, f"draw{seed}")
            if remaining and args.lookup_threads:
                found = lookup(dataset, config, revision, split, set(remaining), args.lookup_threads)
                scan(found, "lookup", limit=len(remaining))
            report[f"{config}/{split}"] = {
                "documents_wanted": len(targets),
                "documents_missing": len(remaining),
                "rows_wanted": sum(map(len, targets.values())),
                **dict(stats),
                "seconds": round(time.time() - started, 1),
            }
    receipt = {
        "schema": "pii-o4-web-intake-receipt/v1",
        "purpose": "Exact public text of O4 teacher-annotated training rows, hash-verified against the membership record",
        "membership_sha256": hashlib.sha256(args.membership.read_bytes()).hexdigest(),
        "rows": sha256_text("".join(f"{i}\t{l}\t{h}\n" for i, l, h in sorted(matched))),
        "row_hashes": sorted([i, l, h] for i, l, h in matched),
        "output": {"name": rows_path.name, "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest()},
        "skipped": dict(skipped),
        "by_config": report,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    totals = Counter()
    for value in report.values():
        totals.update(
            {
                key: value.get(key, 0)
                for key in ("rows_wanted", "rows_matched", "rows_hash_mismatch", "rows_offset_repaired")
            }
        )
    return {
        "ok": True,
        "out": str(args.out.resolve()),
        "configs": len(report),
        **totals,
        "skipped": dict(skipped),
    }


def main() -> None:
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument(
        "--out", type=Path, required=True, help="New directory for rows.jsonl and receipt.json"
    )
    parser.add_argument("--membership", type=Path, default=MEMBERSHIP)
    parser.add_argument("--language", action="append", help="Only rows of this language code; repeatable")
    parser.add_argument(
        "--max-scan", type=int, default=500000, help="Stop streaming a config after this many records"
    )
    parser.add_argument(
        "--lookup-threads",
        type=int,
        default=16,
        help="Parallel data files read for the exhaustive id lookup of rows the scans missed; 0 skips it",
    )
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    import os

    # Streaming readers can abort at interpreter teardown, so both outcomes
    # flush their report and leave immediately.
    try:
        result = fetch(args)
    except (OSError, ValueError, KeyError) as error:
        try:
            acli.die(str(error), acli.ExitCode.SOFTWARE)
        except SystemExit as exit_status:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(int(exit_status.code or 1))
    acli.emit(result, fmt=acli.resolve_format(args))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
