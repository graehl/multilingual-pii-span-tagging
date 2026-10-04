#!/usr/bin/env python
"""Build and query lexical and semantic train/evaluation overlap indexes.

The two retrieval views intentionally remain independent:

* lexical candidates come from a persistent SQLite FTS5 trigram index and
  are reranked by character 3--6-gram F1 (``chrf3_6_f1``);
* semantic candidates come from a persistent FAISS sentence-vector index.

The fusion commands merge candidates from independently built corpus indexes
before ``join`` records both global nearest-three lists and their candidate-ID
intersection.  ``join`` does not choose contamination thresholds.  A later
filter can therefore vary lexical and semantic cutoffs without rebuilding
either index or recomputing embeddings.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overlaplib import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_HASH_BUCKETS,
    SCHEMA_VERSION,
    _load_transformer_embedder,
    audit_faiss_recall,
    build_faiss_database,
    build_hashed_lexical_database,
    build_lexical_database,
    embed_jsonl_corpus,
    fuse_lexical_neighbors,
    fuse_semantic_neighbors,
    join_neighbors,
    query_exact_semantic_database,
    query_faiss_database,
    query_hashed_lexical_database,
    query_lexical_database,
    write_manifest,
)


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"invalid source name: {name!r}")
    return name, Path(path)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    commands = argument_parser.add_subparsers(dest="command", required=True)

    lexical_build = commands.add_parser("build-lexical", help="build persistent FTS5 corpus index")
    lexical_build.add_argument("--train", action="append", type=parse_named_path, required=True)
    lexical_build.add_argument("--output", type=Path, required=True)
    lexical_build.add_argument("--manifest", type=Path)

    lexical_query = commands.add_parser("query-lexical", help="write exact-reranked lexical top-k")
    lexical_query.add_argument("--database", type=Path, required=True)
    lexical_query.add_argument("--eval", action="append", type=parse_named_path, required=True)
    lexical_query.add_argument("--output", type=Path, required=True)
    lexical_query.add_argument("--manifest", type=Path)
    lexical_query.add_argument("--top-k", type=int, default=3)
    lexical_query.add_argument("--candidate-limit", type=int, default=128)
    lexical_query.add_argument("--query-term-limit", type=int, default=96)
    lexical_query.add_argument("--workers", type=int, default=1)

    hashed_build = commands.add_parser(
        "build-hashed-lexical", help="build direct-bucket memory-mapped character-ngram postings"
    )
    hashed_build.add_argument("--train", action="append", type=parse_named_path, required=True)
    hashed_build.add_argument("--output", type=Path, required=True)
    hashed_build.add_argument("--manifest", type=Path)
    hashed_build.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)

    hashed_query = commands.add_parser(
        "query-hashed-lexical", help="write chrF top-k from hashed mmap candidates"
    )
    hashed_query.add_argument("--database", type=Path, required=True)
    hashed_query.add_argument("--eval", action="append", type=parse_named_path, required=True)
    hashed_query.add_argument("--output", type=Path, required=True)
    hashed_query.add_argument("--manifest", type=Path)
    hashed_query.add_argument("--top-k", type=int, default=3)
    hashed_query.add_argument("--candidate-limit", type=int, default=128)
    hashed_query.add_argument("--bucket-limit", type=int, default=96)
    hashed_query.add_argument("--workers", type=int, default=1)

    embed = commands.add_parser("embed", help="materialize E5 query or passage vectors")
    embed.add_argument("--input", action="append", type=parse_named_path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--metadata", type=Path, required=True)
    embed.add_argument("--manifest", type=Path)
    embed.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    embed.add_argument("--role", choices=("query", "passage"), required=True)
    embed.add_argument("--batch-size", type=int, default=128)
    embed.add_argument("--max-length", type=int, default=512)
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")

    embed_pair = commands.add_parser(
        "embed-pair", help="load one model and materialize passage plus query vector databases"
    )
    embed_pair.add_argument("--train", action="append", type=parse_named_path, required=True)
    embed_pair.add_argument("--eval", action="append", type=parse_named_path, required=True)
    embed_pair.add_argument("--train-output", type=Path, required=True)
    embed_pair.add_argument("--train-metadata", type=Path, required=True)
    embed_pair.add_argument("--eval-output", type=Path, required=True)
    embed_pair.add_argument("--eval-metadata", type=Path, required=True)
    embed_pair.add_argument("--manifest", type=Path)
    embed_pair.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    embed_pair.add_argument("--batch-size", type=int, default=128)
    embed_pair.add_argument("--max-length", type=int, default=512)
    embed_pair.add_argument("--device", default="cuda")
    embed_pair.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")

    faiss_build = commands.add_parser("build-faiss", help="build persistent FAISS HNSW index")
    faiss_build.add_argument("--embeddings", type=Path, required=True)
    faiss_build.add_argument("--output", type=Path, required=True)
    faiss_build.add_argument("--manifest", type=Path)
    faiss_build.add_argument("--hnsw-m", type=int, default=32)
    faiss_build.add_argument("--ef-construction", type=int, default=200)

    faiss_query = commands.add_parser("query-faiss", help="write semantic top-k from FAISS")
    faiss_query.add_argument("--index", type=Path, required=True)
    faiss_query.add_argument("--train-metadata", type=Path, required=True)
    faiss_query.add_argument("--eval-embeddings", type=Path, required=True)
    faiss_query.add_argument("--eval-metadata", type=Path, required=True)
    faiss_query.add_argument("--output", type=Path, required=True)
    faiss_query.add_argument("--manifest", type=Path)
    faiss_query.add_argument("--top-k", type=int, default=3)
    faiss_query.add_argument("--ef-search", type=int, default=256)

    exact_semantic = commands.add_parser(
        "query-exact-semantic", help="write exact blocked-cosine semantic top-k"
    )
    exact_semantic.add_argument("--train-embeddings", type=Path, required=True)
    exact_semantic.add_argument("--train-metadata", type=Path, required=True)
    exact_semantic.add_argument("--eval-embeddings", type=Path, required=True)
    exact_semantic.add_argument("--eval-metadata", type=Path, required=True)
    exact_semantic.add_argument("--output", type=Path, required=True)
    exact_semantic.add_argument("--manifest", type=Path)
    exact_semantic.add_argument("--top-k", type=int, default=3)
    exact_semantic.add_argument("--block-size", type=int, default=256)
    exact_semantic.add_argument("--device", default="cuda")

    faiss_audit = commands.add_parser(
        "audit-faiss", help="measure approximate top-k recall against exact flat search"
    )
    faiss_audit.add_argument("--index", type=Path, required=True)
    faiss_audit.add_argument("--train-embeddings", type=Path, required=True)
    faiss_audit.add_argument("--eval-embeddings", type=Path, required=True)
    faiss_audit.add_argument("--manifest", type=Path)
    faiss_audit.add_argument("--top-k", type=int, default=3)
    faiss_audit.add_argument("--ef-search", type=int, default=256)
    faiss_audit.add_argument("--sample-size", type=int, default=128)

    join = commands.add_parser("join", help="persist both top-k lists and shared candidates")
    join.add_argument("--lexical", type=Path, required=True)
    join.add_argument("--semantic", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--manifest", type=Path)

    fuse = commands.add_parser(
        "fuse-lexical", help="union independent lexical top-k candidates and exact-rerank"
    )
    fuse.add_argument("--input", action="append", type=Path, required=True)
    fuse.add_argument("--output", type=Path, required=True)
    fuse.add_argument("--manifest", type=Path)

    semantic_fuse = commands.add_parser(
        "fuse-semantic", help="union independent exact semantic top-k results and rerank"
    )
    semantic_fuse.add_argument("--input", action="append", type=Path, required=True)
    semantic_fuse.add_argument("--output", type=Path, required=True)
    semantic_fuse.add_argument("--manifest", type=Path)
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build-lexical":
        payload = build_lexical_database(args.output, args.train)
    elif args.command == "query-lexical":
        payload = query_lexical_database(
            args.database,
            args.eval,
            args.output,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            query_term_limit=args.query_term_limit,
            workers=args.workers,
        )
    elif args.command == "build-hashed-lexical":
        payload = build_hashed_lexical_database(
            args.output,
            args.train,
            bucket_count=args.hash_buckets,
        )
    elif args.command == "query-hashed-lexical":
        payload = query_hashed_lexical_database(
            args.database,
            args.eval,
            args.output,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            bucket_limit=args.bucket_limit,
            workers=args.workers,
        )
    elif args.command == "embed":
        payload = embed_jsonl_corpus(
            args.input,
            args.output,
            args.metadata,
            model_name=args.model,
            role=args.role,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype_name=args.dtype,
        )
    elif args.command == "embed-pair":
        loaded_embedder = _load_transformer_embedder(args.model, args.device, args.dtype)
        train_payload = embed_jsonl_corpus(
            args.train,
            args.train_output,
            args.train_metadata,
            model_name=args.model,
            role="passage",
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype_name=args.dtype,
            loaded_embedder=loaded_embedder,
        )
        eval_payload = embed_jsonl_corpus(
            args.eval,
            args.eval_output,
            args.eval_metadata,
            model_name=args.model,
            role="query",
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype_name=args.dtype,
            loaded_embedder=loaded_embedder,
        )
        payload = {
            "kind": "pii-overlap-sentence-embedding-pair",
            "schema_version": SCHEMA_VERSION,
            "train": train_payload,
            "eval": eval_payload,
        }
    elif args.command == "build-faiss":
        payload = build_faiss_database(
            args.embeddings,
            args.output,
            hnsw_m=args.hnsw_m,
            ef_construction=args.ef_construction,
        )
    elif args.command == "query-faiss":
        payload = query_faiss_database(
            args.index,
            args.train_metadata,
            args.eval_embeddings,
            args.eval_metadata,
            args.output,
            top_k=args.top_k,
            ef_search=args.ef_search,
        )
    elif args.command == "query-exact-semantic":
        payload = query_exact_semantic_database(
            args.train_embeddings,
            args.train_metadata,
            args.eval_embeddings,
            args.eval_metadata,
            args.output,
            top_k=args.top_k,
            block_size=args.block_size,
            device=args.device,
        )
    elif args.command == "audit-faiss":
        payload = audit_faiss_recall(
            args.index,
            args.train_embeddings,
            args.eval_embeddings,
            top_k=args.top_k,
            ef_search=args.ef_search,
            sample_size=args.sample_size,
        )
    elif args.command == "join":
        payload = join_neighbors(args.lexical, args.semantic, args.output)
    elif args.command == "fuse-lexical":
        payload = fuse_lexical_neighbors(args.input, args.output)
    elif args.command == "fuse-semantic":
        payload = fuse_semantic_neighbors(args.input, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    write_manifest(payload, args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
