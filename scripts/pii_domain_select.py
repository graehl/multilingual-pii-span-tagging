#!/usr/bin/env python3
# acli: 1 complete
"""Select crawl paragraphs that resemble example text from a target domain.

Streams the pinned FineWeb (en) / FineWeb-2 snapshots from a resumable
per-language cursor, keeps training-bucket documents only (as
pii_final35_native_draw does), splits them into line paragraphs and skips
paragraphs a used-region ledger already records. Each paragraph is embedded
with multilingual-E5-base and scored by its similarity to the domain examples
minus its similarity to the other candidates (a hub correction: generic text
such as site chrome is near everything, so raw similarity ranks it first).
The best --rows paragraphs per language are kept; the scanned range advances
the cursor and the kept paragraphs extend the ledger, so the next run reads
deeper rather than rereading.

Domain examples come from one or more files given as PATH[:WEIGHT]: JSONL rows
with "text" (and optional "lang"), or plain text with one example per line.
The files are concatenated; a weight counts each of that file's examples that
many times, as if its lines were repeated (fractions allowed). A paragraph's
domain score is the mean cosine of its K nearest examples, where an example
fills at most WEIGHT of the K slots. Examples in the paragraph's own language
are used when there are any; otherwise all examples are.

Only example text is read, never labels. Output rows are unlabeled
candidates; screening and annotation come after.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import acli
except ImportError:
    sys.path.insert(0, str(Path.home() / "agents"))
    import acli

from scripts.pii_final35_native_draw import TRAINING_SOURCE_ROLES, role_for_document  # noqa: E402
from scripts.pii_needle_select import (  # noqa: E402
    USED_REGION_SCHEMA,
    UsedRegions,
    paragraphs,
    read_cursor,
    read_documents,
    stream_identity,
    write_cursor,
)
from scripts.pii_ont3_surface_retrieval import (  # noqa: E402
    document_identity,
    sha256_text,
    source_descriptor,
    stable_id,
)

MODEL = "intfloat/multilingual-e5-base"
RECEIPT_SCHEMA = "pii-domain-select-receipt/v1"
NEIGHBORS = 5
# A candidate in one of these languages must have at least half its letters in
# the script: FineWeb-2 labels documents, not lines, and embedded English lines
# otherwise win cross-language similarity.
SCRIPTS = {
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "ar": [(0x0600, 0x06FF)],
    "fa": [(0x0600, 0x06FF)],
    "ur": [(0x0600, 0x06FF)],
    "he": [(0x0590, 0x05FF)],
    "el": [(0x0370, 0x03FF)],
    "ru": [(0x0400, 0x04FF)],
    "uk": [(0x0400, 0x04FF)],
    "th": [(0x0E00, 0x0E7F)],
    "hi": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
}


class SelectError(ValueError):
    pass


def parse_weighted(spec: str) -> tuple[Path, float]:
    """PATH or PATH:WEIGHT; a trailing number after the last colon is the weight."""
    path, separator, weight = spec.rpartition(":")
    if separator:
        try:
            value = float(weight)
        except ValueError:
            return Path(spec), 1.0
        if value <= 0:
            raise SelectError(f"weight must be positive: {spec}")
        return Path(path), value
    return Path(spec), 1.0


def load_examples(specs: list[str], min_chars: int) -> list[dict[str, Any]]:
    examples = []
    for spec in specs:
        path, weight = parse_weighted(spec)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        jsonl = all(line.lstrip().startswith("{") for line in lines)
        for line in lines:
            row = json.loads(line) if jsonl else {"text": line}
            text = row.get("text")
            if not isinstance(text, str):
                raise SelectError(f"{path}: example row lacks text")
            # Short fragments ("N .", "------") are near everything; they cannot mark a domain.
            if len(text.strip()) >= min_chars:
                examples.append({"text": text, "lang": row.get("lang"), "weight": weight, "file": str(path)})
    if not examples:
        raise SelectError(f"no domain example of at least {min_chars} characters")
    return examples


def in_script(text: str, language: str) -> bool:
    ranges = SCRIPTS.get(language)
    if ranges is None:
        return True
    letters = [char for char in text if char.isalpha()]
    inside = sum(any(low <= ord(char) <= high for low, high in ranges) for char in letters)
    return bool(letters) and inside * 2 >= len(letters)


def weighted_neighbor_mean(similarity, weights, k: int):
    """Mean similarity of the k nearest examples, each filling at most its weight of the k slots.

    similarity is (candidates, examples); weights is (examples,). Integer weights
    give exactly the score of repeating each example that many times.
    """
    import torch

    depth = min(similarity.shape[1], k * max(1, int(torch.ceil(k / weights.min()).item())))
    values, index = similarity.topk(depth, dim=1)
    mass = weights[index]
    before = mass.cumsum(dim=1) - mass
    taken = torch.clamp(torch.minimum(mass, k - before), min=0)
    return (taken * values).sum(dim=1) / taken.sum(dim=1)


class Embedder:
    def __init__(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(MODEL, dtype=dtype).to(device).eval()

    def __call__(self, texts: list[str]):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), 128):
                batch = self.tokenizer(
                    ["passage: " + text for text in texts[start : start + 128]],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = ((hidden * mask).sum(1) / mask.sum(1)).float()
                out.append(torch.nn.functional.normalize(pooled, dim=1))
        return torch.cat(out)


def candidates(language, identity, start, used, args) -> tuple[list[dict[str, Any]], dict | None, int]:
    found, scanned, following = [], 0, start
    for locator, following, row in read_documents(identity, start):
        scanned += 1
        raw = row.get("text")
        if isinstance(raw, str):
            text = unicodedata.normalize("NFKC", raw)
            document_hash = sha256_text(text)
            bucket_role, bucket = role_for_document(document_hash)
            if bucket_role in TRAINING_SOURCE_ROLES:
                source_id = document_identity(row, language)
                descriptor = source_descriptor(row, language, upstream_split=identity.get("split"))
                taken = 0
                for begin, end in paragraphs(text, "line"):
                    if taken >= args.per_document or not args.min_chars <= end - begin <= args.max_chars:
                        continue
                    piece = text[begin:end]
                    if not in_script(piece, language):
                        continue
                    if used.overlaps(source_id, begin, end) or sha256_text(piece) in used.hashes:
                        continue
                    taken += 1
                    found.append(
                        {
                            "id": stable_id(
                                {
                                    "dataset": descriptor.get("dataset"),
                                    "document_id": source_id,
                                    "start": begin,
                                    "end": end,
                                }
                            ),
                            "lang": language,
                            "text": piece,
                            "text_sha256": sha256_text(piece),
                            "source_locator": {
                                **{key: value for key, value in identity.items() if key != "prefix"},
                                **locator,
                                "document_id": source_id,
                                "document_sha256_nfkc": document_hash,
                                "offsets": "characters of the NFKC-normalized document text",
                            },
                            "source_document_id": source_id,
                            "source_document_sha256": document_hash,
                            "source_document_start": begin,
                            "source_document_end": end,
                            "bucket_role": bucket_role,
                            "partition_bucket": bucket,
                            "source": descriptor,
                            "supervision_scope": "unlabeled_domain_candidate_only",
                        }
                    )
        if scanned >= args.scan:
            break
    return found, following, scanned


def run(args: Any) -> dict[str, Any]:
    import torch

    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    examples = load_examples(args.domain, args.min_example_chars)
    embed = Embedder(args.device)
    example_vectors = embed([example["text"] for example in examples])
    example_weights = torch.tensor([example["weight"] for example in examples], device=example_vectors.device)
    identities = {language: stream_identity(language, "train", None) for language in languages}
    positions = read_cursor(args.cursor, languages, identities)
    used = UsedRegions()
    if args.used_ledger.exists():
        used.add_paths([args.used_ledger])
    used.add_paths(args.exclude_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, cursor_updates, ledger = [], {}, []
    for language in languages:
        target = args.output_dir / f"{language}.jsonl"
        if target.exists():
            raise SelectError(f"{target} already exists; selections are immutable")
        found, following, scanned = candidates(
            language, identities[language], positions[language], used, args
        )
        rows = []
        if found:
            vectors = embed([row["text"] for row in found])
            own = [i for i, example in enumerate(examples) if example["lang"] == language]
            chosen = torch.tensor(own or list(range(len(examples))), device=vectors.device)
            domain = weighted_neighbor_mean(
                vectors @ example_vectors[chosen].T, example_weights[chosen], NEIGHBORS
            )
            background = vectors @ vectors.T
            background.fill_diagonal_(-2.0)
            depth = min(NEIGHBORS, len(found) - 1)
            hub = background.topk(depth, dim=1).values.mean(dim=1) if depth else torch.zeros_like(domain)
            scores = domain - hub
            for rank, i in enumerate(scores.argsort(descending=True)[: args.rows].tolist()):
                rows.append(
                    {
                        **found[i],
                        "draw_stratum": "domain",
                        "domain_score": round(float(scores[i]), 6),
                        "domain_similarity": round(float(domain[i]), 6),
                        "domain_examples": "same-language" if own else "all-languages",
                        "rank": rank,
                    }
                )
        with target.open("x", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            ledger.append(
                {
                    "schema": USED_REGION_SCHEMA,
                    "use": "domain-select",
                    "lang": language,
                    "source_locator": row["source_locator"],
                    "source_document_id": row["source_document_id"],
                    "source_document_sha256": row["source_document_sha256"],
                    "source_document_start": row["source_document_start"],
                    "source_document_end": row["source_document_end"],
                    "text_sha256": row["text_sha256"],
                }
            )
        cursor_updates[language] = {"stream": identities[language], "next": following}
        summaries.append(
            {
                "language": language,
                "documents_scanned": scanned,
                "candidates": len(found),
                "rows": len(rows),
                "range": {"first": positions[language], "next": following},
                "path": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    with args.used_ledger.open("a", encoding="utf-8") as sink:
        for entry in ledger:
            sink.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    write_cursor(args.cursor, cursor_updates)
    return {
        "schema": RECEIPT_SCHEMA,
        "model": MODEL,
        "neighbors": NEIGHBORS,
        "score": "weighted nearest-example cosine minus mean cosine to the 5 nearest other candidates",
        "domain_files": [
            {
                "path": str(path),
                "weight": weight,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "examples": sum(example["file"] == str(path) for example in examples),
            }
            for path, weight in map(parse_weighted, args.domain)
        ],
        "roles": sorted(TRAINING_SOURCE_ROLES),
        "paragraph_chars": [args.min_chars, args.max_chars],
        "per_document": args.per_document,
        "scan": args.scan,
        "cursor": str(args.cursor),
        "used_ledger": str(args.used_ledger),
        "languages": summaries,
        "supervision": "unlabeled candidates; screening and annotation are required before training use",
    }


def build_parser() -> Any:
    parser = acli.argument_parser(
        description=__doc__,
        capabilities=("complete",),
        exit_codes={0: "success", 2: "invalid input or existing output"},
    )
    parser.add_argument("--languages", required=True, help="comma-separated language codes")
    parser.add_argument(
        "--domain", action="append", required=True, help="PATH[:WEIGHT] of domain examples; repeatable"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--cursor", type=Path, required=True, help="per-language next-document file, read and updated"
    )
    parser.add_argument(
        "--used-ledger", type=Path, required=True, help="append-only JSONL of used paragraphs"
    )
    parser.add_argument(
        "--exclude-jsonl", action="append", type=Path, default=[], help="rows never to select"
    )
    parser.add_argument("--scan", type=int, default=3000, help="documents to read per language")
    parser.add_argument("--rows", type=int, default=200, help="paragraphs kept per language")
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--min-example-chars", type=int, default=40)
    parser.add_argument("--per-document", type=int, default=3, help="paragraph candidates per document")
    parser.add_argument("--device", default="cuda")
    acli.add_standard_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        acli.maybe_complete(parser)
    args = parser.parse_args(argv)
    try:
        if args.scan <= 0 or args.rows <= 0:
            raise SelectError("--scan and --rows must be positive")
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
