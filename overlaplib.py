"""Near-duplicate retrieval between query rows and indexed rows: lexical and semantic views.

Shared by any task that must keep evaluation text out of training or find
repeated material: the PII program's overlap screens use it through
scripts/pii_overlap_neighbors.py (CLI) and scripts/pii_eval_overlap.py.

The two retrieval views stay independent:

* lexical candidates come from a persistent SQLite FTS5 trigram index (or a
  memory-mapped hashed n-gram index) and are reranked by character 3--6-gram
  F1 (``chrf3_6_f1``);
* semantic candidates come from sentence vectors (multilingual E5 by default),
  searched exactly in blocks or through a FAISS HNSW index.

``join_neighbors`` records both nearest-k lists and their intersection without
choosing thresholds; ``passing_candidates`` applies a two-view cut. Rows are
JSONL with a string ``text``; index-side ids are ``source_name:line``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import mmap
import os
import shutil
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
LEXICAL_ORDERS = (3, 4, 5, 6)
HASHED_CANDIDATE_ORDERS = (4, 5, 6)
DEFAULT_HASH_BUCKETS = 1 << 24
SCHEMA_VERSION = 1


def normalize_text(text: str) -> str:
    """Normalize without transliterating or erasing script distinctions."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_language(row: dict[str, Any], dataset_name: str, identifier: str) -> str:
    row_language = row.get("lang")
    if isinstance(row_language, str) and 2 <= len(row_language) <= 3 and row_language.isalpha():
        return row_language.casefold()
    id_prefix = identifier.partition(":")[0]
    if 2 <= len(id_prefix) <= 3 and id_prefix.isalpha():
        return id_prefix.casefold()
    parts = dataset_name.split("-")
    if len(parts) >= 2 and parts[0] == "openner" and parts[1].isalpha():
        return parts[1].casefold()
    return parts[0]


def char_ngrams(text: str, order: int) -> Counter[str]:
    if order <= 0:
        raise ValueError("ngram order must be positive")
    if len(text) < order:
        return Counter()
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def chrf3_6(left: str, right: str) -> dict[str, float]:
    """Symmetric character 3--6-gram F1 and directional containment.

    ``left_containment`` is the fraction of the evaluation-side (left)
    n-grams found in the training-side text.  It catches a copied segment
    embedded in a longer training window, while ``chrf3_6_f1`` remains the
    symmetric score used for nearest-neighbor ranking.
    """

    per_order: list[tuple[float, float, float]] = []
    for order in LEXICAL_ORDERS:
        left_counts = char_ngrams(left, order)
        right_counts = char_ngrams(right, order)
        left_total = sum(left_counts.values())
        right_total = sum(right_counts.values())
        if not left_total or not right_total:
            continue
        common = sum(min(count, right_counts.get(gram, 0)) for gram, count in left_counts.items())
        left_recall = common / left_total
        right_recall = common / right_total
        f1 = (
            2 * left_recall * right_recall / (left_recall + right_recall)
            if left_recall + right_recall
            else 0.0
        )
        per_order.append((f1, left_recall, right_recall))
    if not per_order:
        exact = float(left == right and bool(left))
        return {
            "chrf3_6_f1": exact,
            "eval_containment": exact,
            "train_containment": exact,
        }
    count = len(per_order)
    return {
        "chrf3_6_f1": sum(item[0] for item in per_order) / count,
        "eval_containment": sum(item[1] for item in per_order) / count,
        "train_containment": sum(item[2] for item in per_order) / count,
    }


def jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: missing string text")
            yield line_number, row


@contextmanager
def atomic_output(path: Path, mode: str = "w", **kwargs: Any) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open(mode, **kwargs) as output:
        yield output
    temporary.replace(path)


def _create_lexical_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE manifest (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE documents (
            rowid INTEGER PRIMARY KEY,
            train_id TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            lang TEXT,
            text_sha256 TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            raw_text TEXT NOT NULL
        );
        CREATE INDEX documents_sha256 ON documents(text_sha256);
        CREATE VIRTUAL TABLE lexical_index USING fts5(
            normalized_text,
            content='documents',
            content_rowid='rowid',
            tokenize='trigram'
        );
        """
    )


def build_lexical_database(output: Path, sources: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing database: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale partial database exists: {temporary}")
    connection = sqlite3.connect(temporary)
    counts: Counter[str] = Counter()
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        _create_lexical_schema(connection)
        insert = """
            INSERT INTO documents(
                train_id, source_name, source_line, lang, text_sha256,
                normalized_text, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        for source_name, path in sources:
            for line_number, row in jsonl_rows(path):
                normalized = normalize_text(row["text"])
                train_id = f"{source_name}:{line_number}"
                connection.execute(
                    insert,
                    (
                        train_id,
                        source_name,
                        line_number,
                        row.get("lang"),
                        sha256_text(normalized),
                        normalized,
                        row["text"],
                    ),
                )
                counts[source_name] += 1
                if sum(counts.values()) % 10_000 == 0:
                    print(f"lexical-build: {sum(counts.values())} documents", flush=True)
        connection.execute("INSERT INTO lexical_index(lexical_index) VALUES ('rebuild')")
        connection.execute("CREATE VIRTUAL TABLE lexical_vocab USING fts5vocab(lexical_index, 'row')")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pii-overlap-lexical-index",
            "normalization": "Unicode NFKC + casefold + whitespace collapse",
            "candidate_index": "SQLite FTS5 trigram",
            "reported_score": "mean symmetric character-ngram F1 over orders 3,4,5,6",
            "sources": [
                {
                    "name": name,
                    "path": str(path.resolve()),
                    "documents": counts[name],
                    "sha256": _sha256_file(path),
                }
                for name, path in sources
            ],
            "documents": sum(counts.values()),
        }
        connection.execute(
            "INSERT INTO manifest(key, value_json) VALUES (?, ?)",
            ("build", json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        temporary.replace(output)
        return manifest
    except BaseException:
        connection.close()
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _exact_hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def _hashing_vectorizer(bucket_count: int) -> Any:
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer

    if bucket_count <= 0 or bucket_count & (bucket_count - 1):
        raise ValueError("hash bucket count must be a positive power of two")
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(min(HASHED_CANDIDATE_ORDERS), max(HASHED_CANDIDATE_ORDERS)),
        n_features=bucket_count,
        alternate_sign=False,
        binary=True,
        lowercase=False,
        norm=None,
        dtype=np.uint8,
    )


def build_hashed_lexical_database(
    output: Path,
    sources: Sequence[tuple[str, Path]],
    *,
    bucket_count: int,
) -> dict[str, Any]:
    """Build direct-bucket mmap postings plus compact text/metadata stores."""

    import numpy as np

    if output.exists():
        raise FileExistsError(f"refusing to replace existing database: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale partial database exists: {temporary}")
    temporary.mkdir()
    text_offsets = [0]
    metadata_offsets = [0]
    exact_hashes: list[int] = []
    counts: Counter[str] = Counter()
    texts_path = temporary / "normalized-texts.utf8"
    metadata_path = temporary / "metadata.jsonl"
    try:
        with texts_path.open("wb") as texts, metadata_path.open("wb") as metadata:
            for source_name, path in sources:
                for line_number, row in jsonl_rows(path):
                    normalized = normalize_text(row["text"])
                    encoded_text = normalized.encode("utf-8")
                    texts.write(encoded_text)
                    text_offsets.append(text_offsets[-1] + len(encoded_text))
                    exact_hashes.append(_exact_hash64(normalized))
                    encoded_metadata = (
                        json.dumps(
                            {
                                "train_id": f"{source_name}:{line_number}",
                                "source_name": source_name,
                                "source_line": line_number,
                                "lang": row.get("lang"),
                                "text_sha256": sha256_text(normalized),
                                "text": row["text"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                    metadata.write(encoded_metadata)
                    metadata_offsets.append(metadata_offsets[-1] + len(encoded_metadata))
                    counts[source_name] += 1
                    if sum(counts.values()) % 25_000 == 0:
                        print(f"hashed-lexical/materialize: {sum(counts.values())}", flush=True)
        np.save(temporary / "text-offsets.u64.npy", np.asarray(text_offsets, dtype=np.uint64))
        np.save(temporary / "metadata-offsets.u64.npy", np.asarray(metadata_offsets, dtype=np.uint64))

        def normalized_documents() -> Iterator[str]:
            for _source_name, path in sources:
                for _line_number, row in jsonl_rows(path):
                    yield normalize_text(row["text"])

        vectorizer = _hashing_vectorizer(bucket_count)
        print("hashed-lexical/vectorize: begin", flush=True)
        document_features = vectorizer.transform(normalized_documents())
        if document_features.shape[0] != sum(counts.values()):
            raise AssertionError("vectorizer row count differs from materialized document count")
        print(
            f"hashed-lexical/vectorize: {document_features.shape[0]} docs, "
            f"{document_features.nnz} document-bucket pairs",
            flush=True,
        )
        postings_matrix = document_features.tocsc()
        np.save(temporary / "bucket-indptr.npy", postings_matrix.indptr)
        np.save(temporary / "postings.u32.npy", postings_matrix.indices.astype(np.uint32))
        exact_values = np.asarray(exact_hashes, dtype=np.uint64)
        exact_order = np.argsort(exact_values, kind="stable")
        np.save(temporary / "exact-hashes.sorted.u64.npy", exact_values[exact_order])
        np.save(temporary / "exact-docs.sorted.u32.npy", exact_order.astype(np.uint32))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pii-overlap-hashed-mmap-lexical-index",
            "normalization": "Unicode NFKC + casefold + whitespace collapse",
            "candidate_ngrams": list(HASHED_CANDIDATE_ORDERS),
            "hash": "scikit-learn HashingVectorizer MurmurHash3 direct bucket",
            "hash_buckets": bucket_count,
            "posting_type": "uint32 document rows",
            "reported_score": "mean symmetric character-ngram F1 over orders 3,4,5,6",
            "documents": sum(counts.values()),
            "document_bucket_pairs": int(postings_matrix.nnz),
            "sources": [
                {
                    "name": name,
                    "path": str(path.resolve()),
                    "documents": counts[name],
                    "sha256": _sha256_file(path),
                }
                for name, path in sources
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return manifest
    except BaseException:
        raise


class HashedLexicalDatabase:
    def __init__(self, path: Path):
        import numpy as np

        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.vectorizer = _hashing_vectorizer(self.manifest["hash_buckets"])
        self.bucket_indptr = np.load(path / "bucket-indptr.npy", mmap_mode="r")
        self.postings = np.load(path / "postings.u32.npy", mmap_mode="r")
        self.text_offsets = np.load(path / "text-offsets.u64.npy", mmap_mode="r")
        self.metadata_offsets = np.load(path / "metadata-offsets.u64.npy", mmap_mode="r")
        self.exact_hashes = np.load(path / "exact-hashes.sorted.u64.npy", mmap_mode="r")
        self.exact_docs = np.load(path / "exact-docs.sorted.u32.npy", mmap_mode="r")
        self._texts_file = (path / "normalized-texts.utf8").open("rb")
        self._metadata_file = (path / "metadata.jsonl").open("rb")
        self.texts = mmap.mmap(self._texts_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.metadata = mmap.mmap(self._metadata_file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self.texts.close()
        self.metadata.close()
        self._texts_file.close()
        self._metadata_file.close()

    def normalized_text(self, document_row: int) -> str:
        start = int(self.text_offsets[document_row])
        end = int(self.text_offsets[document_row + 1])
        return self.texts[start:end].decode("utf-8")

    def metadata_row(self, document_row: int) -> dict[str, Any]:
        start = int(self.metadata_offsets[document_row])
        end = int(self.metadata_offsets[document_row + 1])
        return json.loads(self.metadata[start:end])

    def exact_document_rows(self, normalized: str) -> list[int]:
        import numpy as np

        value = np.uint64(_exact_hash64(normalized))
        left = int(np.searchsorted(self.exact_hashes, value, side="left"))
        right = int(np.searchsorted(self.exact_hashes, value, side="right"))
        return [int(row) for row in self.exact_docs[left:right]]

    def candidates(self, normalized: str, *, bucket_limit: int, candidate_limit: int) -> list[int]:
        import numpy as np

        features = self.vectorizer.transform([normalized]).indices
        document_count = self.manifest["documents"]
        bucket_rows = []
        for bucket in features:
            start = int(self.bucket_indptr[bucket])
            end = int(self.bucket_indptr[bucket + 1])
            if end > start:
                bucket_rows.append((end - start, int(bucket), start, end))
        bucket_rows.sort()
        bucket_rows = bucket_rows[:bucket_limit]
        document_parts = []
        weight_parts = []
        for frequency, _bucket, start, end in bucket_rows:
            documents = np.asarray(self.postings[start:end], dtype=np.int64)
            document_parts.append(documents)
            weight_parts.append(
                np.full(
                    len(documents),
                    math.log((document_count + 1) / (frequency + 1)),
                    dtype=np.float32,
                )
            )
        if not document_parts:
            return self.exact_document_rows(normalized)
        all_documents = np.concatenate(document_parts)
        all_weights = np.concatenate(weight_parts)
        unique_documents, inverse = np.unique(all_documents, return_inverse=True)
        scores = np.bincount(inverse, weights=all_weights)
        take = min(candidate_limit, len(unique_documents))
        if take < len(unique_documents):
            selected = np.argpartition(scores, -take)[-take:]
        else:
            selected = np.arange(len(unique_documents))
        ranked = selected[np.argsort(-scores[selected], kind="stable")]
        candidates = [int(unique_documents[index]) for index in ranked]
        for exact in self.exact_document_rows(normalized):
            if exact not in candidates:
                candidates.insert(0, exact)
        return candidates


def hashed_lexical_neighbors(
    database: HashedLexicalDatabase,
    text: str,
    *,
    top_k: int,
    candidate_limit: int,
    bucket_limit: int,
) -> list[dict[str, Any]]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    exact_hash = sha256_text(normalized)
    scored = []
    for document_row in database.candidates(
        normalized, bucket_limit=bucket_limit, candidate_limit=candidate_limit
    ):
        train_text = database.normalized_text(document_row)
        metadata = database.metadata_row(document_row)
        scored.append(
            {
                "train_row": document_row + 1,
                **metadata,
                "exact_normalized_match": metadata["text_sha256"] == exact_hash,
                **chrf3_6(normalized, train_text),
            }
        )
    scored.sort(
        key=lambda item: (
            -item["chrf3_6_f1"],
            -item["eval_containment"],
            item["train_id"],
        )
    )
    result = scored[:top_k]
    for rank, item in enumerate(result, 1):
        item["rank"] = rank
    return result


def _query_hashed_lexical_dataset_star(arguments: tuple[Any, ...]) -> dict[str, Any]:
    return _query_hashed_lexical_dataset(*arguments)


def _query_hashed_lexical_dataset(
    database_path: Path,
    evaluation: tuple[str, Path],
    output: Path,
    top_k: int,
    candidate_limit: int,
    bucket_limit: int,
) -> dict[str, Any]:
    dataset_name, path = evaluation
    database = HashedLexicalDatabase(database_path)
    count = 0
    with atomic_output(output, encoding="utf-8") as destination:
        for line_number, row in jsonl_rows(path):
            eval_id = row.get("id") or f"{dataset_name}:{line_number}"
            neighbors = hashed_lexical_neighbors(
                database,
                row["text"],
                top_k=top_k,
                candidate_limit=candidate_limit,
                bucket_limit=bucket_limit,
            )
            destination.write(
                json.dumps(
                    {
                        "eval_id": eval_id,
                        "eval_dataset": dataset_name,
                        "eval_source_line": line_number,
                        "lang": infer_language(row, dataset_name, eval_id),
                        "text_sha256": sha256_text(normalize_text(row["text"])),
                        "text": row["text"],
                        "lexical_top3": neighbors,
                        "retrieval_top_k": top_k,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
            if count % 500 == 0:
                print(f"hashed-query/{dataset_name}: {count}", flush=True)
    database.close()
    return {"dataset": dataset_name, "evaluations": count}


def query_hashed_lexical_database(
    database: Path,
    evaluations: Sequence[tuple[str, Path]],
    output: Path,
    *,
    top_k: int,
    candidate_limit: int,
    bucket_limit: int,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    temporary_paths = [
        output.with_name(f".{output.name}.{dataset_name}.worker-{os.getpid()}")
        for dataset_name, _ in evaluations
    ]
    arguments = [
        (database, evaluation, temporary, top_k, candidate_limit, bucket_limit)
        for evaluation, temporary in zip(evaluations, temporary_paths, strict=True)
    ]
    try:
        if workers == 1:
            results = [_query_hashed_lexical_dataset(*argument) for argument in arguments]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
                results = list(executor.map(_query_hashed_lexical_dataset_star, arguments))
        with atomic_output(output, encoding="utf-8") as destination:
            for temporary in temporary_paths:
                with temporary.open(encoding="utf-8") as source:
                    shutil.copyfileobj(source, destination)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    counts = {result["dataset"]: result["evaluations"] for result in results}
    return {"evaluations": sum(counts.values()), "by_dataset": counts, "output": str(output)}


def _document_frequencies(connection: sqlite3.Connection, grams: Sequence[str]) -> dict[str, int]:
    if not grams:
        return {}
    result: dict[str, int] = {}
    for start in range(0, len(grams), 500):
        batch = grams[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT term, doc FROM lexical_vocab WHERE term IN ({placeholders})", batch
        )
        result.update((term, int(count)) for term, count in rows)
    return result


def _fts_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def lexical_query_terms(
    connection: sqlite3.Connection,
    normalized: str,
    *,
    limit: int,
) -> list[str]:
    """Choose rare 6-character phrases, then rare trigrams, for FTS recall."""

    unique_trigrams = list(dict.fromkeys(char_ngrams(normalized, 3)))
    if not unique_trigrams:
        return []
    document_count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
    frequencies = _document_frequencies(connection, unique_trigrams)

    def idf(gram: str) -> float:
        return math.log((document_count + 1) / (frequencies.get(gram, 0) + 1))

    phrases = list(dict.fromkeys(normalized[index : index + 6] for index in range(len(normalized) - 5)))
    phrases.sort(
        key=lambda phrase: (
            -sum(idf(phrase[index : index + 3]) for index in range(4)),
            phrase,
        )
    )
    trigrams = sorted(unique_trigrams, key=lambda gram: (-idf(gram), gram))
    # Longer exact phrases give useful BM25 ranking; individual rare trigrams
    # provide a recall backstop for edits within those phrases.
    long_budget = max(1, limit * 3 // 4) if phrases else 0
    chosen = phrases[:long_budget]
    chosen.extend(trigrams[: limit - len(chosen)])
    return chosen


def lexical_neighbors(
    connection: sqlite3.Connection,
    text: str,
    *,
    top_k: int = 3,
    candidate_limit: int = 128,
    query_term_limit: int = 96,
) -> list[dict[str, Any]]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    terms = lexical_query_terms(connection, normalized, limit=query_term_limit)
    if not terms:
        return []
    match_query = " OR ".join(_fts_quote(term) for term in terms)
    candidates = connection.execute(
        """
        SELECT d.rowid, d.train_id, d.source_name, d.source_line, d.lang,
               d.text_sha256, d.normalized_text, d.raw_text,
               bm25(lexical_index) AS candidate_score
        FROM lexical_index
        JOIN documents AS d ON d.rowid = lexical_index.rowid
        WHERE lexical_index MATCH ?
        ORDER BY candidate_score
        LIMIT ?
        """,
        (match_query, candidate_limit),
    ).fetchall()
    exact_hash = sha256_text(normalized)
    exacts = connection.execute(
        """
        SELECT rowid, train_id, source_name, source_line, lang,
               text_sha256, normalized_text, raw_text, -1e300
        FROM documents WHERE text_sha256 = ?
        """,
        (exact_hash,),
    ).fetchall()
    by_rowid = {int(row[0]): row for row in candidates}
    by_rowid.update((int(row[0]), row) for row in exacts)
    scored = []
    for row in by_rowid.values():
        scores = chrf3_6(normalized, row[6])
        scored.append(
            {
                "train_row": int(row[0]),
                "train_id": row[1],
                "source_name": row[2],
                "source_line": int(row[3]),
                "lang": row[4],
                "text_sha256": row[5],
                "exact_normalized_match": row[5] == exact_hash,
                **scores,
                "text": row[7],
            }
        )
    scored.sort(
        key=lambda item: (
            -item["chrf3_6_f1"],
            -item["eval_containment"],
            item["train_id"],
        )
    )
    result = scored[:top_k]
    for rank, item in enumerate(result, 1):
        item["rank"] = rank
    return result


def query_lexical_database(
    database: Path,
    evaluations: Sequence[tuple[str, Path]],
    output: Path,
    *,
    top_k: int,
    candidate_limit: int,
    query_term_limit: int,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    temporary_paths = [
        output.with_name(f".{output.name}.{dataset_name}.worker-{os.getpid()}")
        for dataset_name, _ in evaluations
    ]
    arguments = [
        (
            database,
            evaluation,
            temporary,
            top_k,
            candidate_limit,
            query_term_limit,
        )
        for evaluation, temporary in zip(evaluations, temporary_paths, strict=True)
    ]
    try:
        if workers == 1:
            results = [_query_lexical_dataset(*argument) for argument in arguments]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
                results = list(executor.map(_query_lexical_dataset_star, arguments))
        with atomic_output(output, encoding="utf-8") as destination:
            for temporary in temporary_paths:
                with temporary.open(encoding="utf-8") as source:
                    shutil.copyfileobj(source, destination)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    counts = {result["dataset"]: result["evaluations"] for result in results}
    return {"evaluations": sum(counts.values()), "by_dataset": counts, "output": str(output)}


def _open_lexical_database(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    # Let SQLite map the immutable posting-list database instead of copying
    # hot pages through a small connection-local cache. SQLite silently caps
    # this at the platform/build maximum when the index is smaller.
    connection.execute("PRAGMA mmap_size=8589934592")
    return connection


def _query_lexical_dataset_star(arguments: tuple[Any, ...]) -> dict[str, Any]:
    return _query_lexical_dataset(*arguments)


def _query_lexical_dataset(
    database: Path,
    evaluation: tuple[str, Path],
    output: Path,
    top_k: int,
    candidate_limit: int,
    query_term_limit: int,
) -> dict[str, Any]:
    dataset_name, path = evaluation
    connection = _open_lexical_database(database)
    count = 0
    with atomic_output(output, encoding="utf-8") as destination:
        for line_number, row in jsonl_rows(path):
            eval_id = row.get("id") or f"{dataset_name}:{line_number}"
            neighbors = lexical_neighbors(
                connection,
                row["text"],
                top_k=top_k,
                candidate_limit=candidate_limit,
                query_term_limit=query_term_limit,
            )
            record = {
                "eval_id": eval_id,
                "eval_dataset": dataset_name,
                "eval_source_line": line_number,
                "lang": infer_language(row, dataset_name, eval_id),
                "text_sha256": sha256_text(normalize_text(row["text"])),
                "text": row["text"],
                "lexical_top3": neighbors,
                "retrieval_top_k": top_k,
            }
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            if count % 500 == 0:
                print(f"lexical-query/{dataset_name}: {count} evaluations", flush=True)
    connection.close()
    return {"dataset": dataset_name, "evaluations": count}


def _load_transformer_embedder(model_name: str, device: str, dtype_name: str) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(device)
    return tokenizer, model, functional


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def embed_jsonl_corpus(
    inputs: Sequence[tuple[str, Path]],
    output: Path,
    metadata_output: Path,
    *,
    model_name: str,
    role: str,
    batch_size: int,
    max_length: int,
    device: str,
    dtype_name: str,
    loaded_embedder: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if output.exists() or metadata_output.exists():
        raise FileExistsError("refusing to replace existing embedding or metadata artifact")
    tokenizer, model, functional = loaded_embedder or _load_transformer_embedder(
        model_name, device, dtype_name
    )
    rows: list[tuple[str, str | None, str, str, int]] = []
    for source_name, path in inputs:
        for line_number, row in jsonl_rows(path):
            # Passage IDs must exactly match the lexical database's stable
            # source+line identity even when a source row carries its own ID.
            # Query/evaluation IDs remain the dataset-provided IDs used by
            # prediction and gold artifacts.
            identifier = (
                f"{source_name}:{line_number}"
                if role == "passage"
                else row.get("id") or f"{source_name}:{line_number}"
            )
            rows.append((identifier, row.get("lang"), row["text"], source_name, line_number))
    if not rows:
        raise ValueError("cannot embed an empty corpus")
    dimension = int(model.config.hidden_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    vectors = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), dimension),
    )
    prefix = "query: " if role == "query" else "passage: "
    for start in range(0, len(rows), batch_size):
        batch = [prefix + normalize_text(item[2]) for item in rows[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = functional.normalize(pooled.to(torch.float32), p=2, dim=1)
        vectors[start : start + len(batch)] = pooled.cpu().numpy().astype(np.float16)
        if (start + len(batch)) % 1_000 < batch_size:
            print(f"semantic-embed/{role}: {start + len(batch)}/{len(rows)}", flush=True)
    vectors.flush()
    del vectors
    temporary.replace(output)
    with atomic_output(metadata_output, encoding="utf-8") as destination:
        for vector_row, (identifier, lang, text, source_name, line_number) in enumerate(rows):
            destination.write(
                json.dumps(
                    {
                        "vector_row": vector_row,
                        "id": identifier,
                        "source_name": source_name,
                        "source_line": line_number,
                        "lang": lang,
                        "text_sha256": sha256_text(normalize_text(text)),
                        "text": text,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    revision = getattr(model.config, "_commit_hash", None)
    return {
        "kind": "pii-overlap-sentence-embeddings",
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "model_revision": revision,
        "pooling": "attention-mask mean of final hidden states, L2 normalized",
        "role_prefix": prefix.strip(),
        "max_length": max_length,
        "dtype": "float16",
        "shape": [len(rows), dimension],
        "embeddings": str(output),
        "metadata": str(metadata_output),
    }


def build_faiss_database(
    embeddings: Path,
    output: Path,
    *,
    hnsw_m: int,
    ef_construction: int,
) -> dict[str, Any]:
    import faiss
    import numpy as np

    if output.exists():
        raise FileExistsError(f"refusing to replace existing vector index: {output}")
    vectors = np.load(embeddings, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError(f"expected two-dimensional embeddings: {vectors.shape}")
    index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    for start in range(0, len(vectors), 10_000):
        index.add(np.asarray(vectors[start : start + 10_000], dtype=np.float32))
        print(f"faiss-build: {min(start + 10_000, len(vectors))}/{len(vectors)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    faiss.write_index(index, str(temporary))
    temporary.replace(output)
    return {
        "kind": "pii-overlap-faiss-hnsw",
        "schema_version": SCHEMA_VERSION,
        "metric": "inner product over L2-normalized vectors (cosine)",
        "vectors": int(index.ntotal),
        "dimension": int(index.d),
        "hnsw_m": hnsw_m,
        "ef_construction": ef_construction,
        "embeddings": str(embeddings),
        "index": str(output),
    }


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in jsonl_rows(path)]


def query_faiss_database(
    index_path: Path,
    train_metadata_path: Path,
    eval_embeddings_path: Path,
    eval_metadata_path: Path,
    output: Path,
    *,
    top_k: int,
    ef_search: int,
) -> dict[str, Any]:
    import faiss
    import numpy as np

    index = faiss.read_index(str(index_path))
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError(f"expected FAISS HNSW index, got {type(index).__name__}")
    index.hnsw.efSearch = ef_search
    train_metadata = _read_metadata(train_metadata_path)
    eval_metadata = _read_metadata(eval_metadata_path)
    eval_vectors = np.load(eval_embeddings_path, mmap_mode="r")
    if len(train_metadata) != index.ntotal:
        raise ValueError("training metadata and vector-index sizes differ")
    if len(eval_metadata) != len(eval_vectors):
        raise ValueError("evaluation metadata and embedding sizes differ")
    with atomic_output(output, encoding="utf-8") as destination:
        for start in range(0, len(eval_vectors), 1_000):
            distances, indices = index.search(
                np.asarray(eval_vectors[start : start + 1_000], dtype=np.float32), top_k
            )
            for local_index, (scores, neighbors) in enumerate(zip(distances, indices, strict=True)):
                eval_row = eval_metadata[start + local_index]
                semantic = []
                for rank, (score, vector_row) in enumerate(zip(scores, neighbors, strict=True), 1):
                    if vector_row < 0:
                        continue
                    train_row = train_metadata[int(vector_row)]
                    semantic.append(
                        {
                            "rank": rank,
                            "train_row": int(vector_row) + 1,
                            "train_id": train_row["id"],
                            "source_name": train_row["source_name"],
                            "source_line": train_row["source_line"],
                            "lang": train_row.get("lang"),
                            "cosine": float(score),
                            "text_sha256": train_row["text_sha256"],
                            "text": train_row["text"],
                        }
                    )
                destination.write(
                    json.dumps(
                        {
                            "eval_id": eval_row["id"],
                            "eval_dataset": eval_row["source_name"],
                            "eval_source_line": eval_row["source_line"],
                            "lang": infer_language(eval_row, eval_row["source_name"], eval_row["id"]),
                            "text_sha256": eval_row["text_sha256"],
                            "text": eval_row["text"],
                            "semantic_top3": semantic,
                            "retrieval_top_k": top_k,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(f"faiss-query: {min(start + 1_000, len(eval_vectors))}/{len(eval_vectors)}", flush=True)
    return {
        "evaluations": len(eval_vectors),
        "top_k": top_k,
        "ef_search": ef_search,
        "output": str(output),
    }


def query_exact_semantic_database(
    train_embeddings_path: Path,
    train_metadata_path: Path,
    eval_embeddings_path: Path,
    eval_metadata_path: Path,
    output: Path,
    *,
    top_k: int,
    block_size: int,
    device: str,
) -> dict[str, Any]:
    """Exact blocked cosine top-k over the frozen float16 vector database."""

    import numpy as np
    import torch
    import torch.nn.functional as functional

    train_metadata = _read_metadata(train_metadata_path)
    eval_metadata = _read_metadata(eval_metadata_path)
    train_memmap = np.load(train_embeddings_path, mmap_mode="r")
    eval_memmap = np.load(eval_embeddings_path, mmap_mode="r")
    if len(train_metadata) != len(train_memmap):
        raise ValueError("training metadata and embedding sizes differ")
    if len(eval_metadata) != len(eval_memmap):
        raise ValueError("evaluation metadata and embedding sizes differ")
    # Re-normalize after float16 storage quantization so the frozen database's
    # exact inner product is also exact cosine for those stored vectors.
    train_vectors = torch.from_numpy(np.array(train_memmap, dtype=np.float32, copy=True)).to(device)
    train_vectors = functional.normalize(train_vectors, p=2, dim=1)
    with atomic_output(output, encoding="utf-8") as destination:
        for start in range(0, len(eval_memmap), block_size):
            queries = torch.from_numpy(
                np.array(eval_memmap[start : start + block_size], dtype=np.float32, copy=True)
            ).to(device)
            queries = functional.normalize(queries, p=2, dim=1)
            with torch.inference_mode():
                scores, indices = torch.topk(queries @ train_vectors.T, k=top_k, dim=1)
            for local_index, (row_scores, row_indices) in enumerate(
                zip(scores.cpu().tolist(), indices.cpu().tolist(), strict=True)
            ):
                eval_row = eval_metadata[start + local_index]
                semantic = []
                for rank, (score, vector_row) in enumerate(zip(row_scores, row_indices, strict=True), 1):
                    train_row = train_metadata[vector_row]
                    semantic.append(
                        {
                            "rank": rank,
                            "train_row": vector_row + 1,
                            "train_id": train_row["id"],
                            "source_name": train_row["source_name"],
                            "source_line": train_row["source_line"],
                            "lang": train_row.get("lang"),
                            "cosine": score,
                            "text_sha256": train_row["text_sha256"],
                            "text": train_row["text"],
                        }
                    )
                destination.write(
                    json.dumps(
                        {
                            "eval_id": eval_row["id"],
                            "eval_dataset": eval_row["source_name"],
                            "eval_source_line": eval_row["source_line"],
                            "lang": infer_language(eval_row, eval_row["source_name"], eval_row["id"]),
                            "text_sha256": eval_row["text_sha256"],
                            "text": eval_row["text"],
                            "semantic_top3": semantic,
                            "retrieval_top_k": top_k,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                f"exact-semantic-query: {min(start + block_size, len(eval_memmap))}/{len(eval_memmap)}",
                flush=True,
            )
    return {
        "kind": "pii-overlap-exact-semantic-top-k",
        "metric": "exact blocked cosine over L2-renormalized frozen float16 vectors",
        "evaluations": len(eval_memmap),
        "training_vectors": len(train_memmap),
        "top_k": top_k,
        "block_size": block_size,
        "device": device,
        "output": str(output),
    }


def audit_faiss_recall(
    index_path: Path,
    train_embeddings_path: Path,
    eval_embeddings_path: Path,
    *,
    top_k: int,
    ef_search: int,
    sample_size: int,
) -> dict[str, Any]:
    """Compare deterministic HNSW results with exact flat inner-product search."""

    import faiss
    import numpy as np

    approximate = faiss.read_index(str(index_path))
    if not isinstance(approximate, faiss.IndexHNSW):
        raise TypeError(f"expected FAISS HNSW index, got {type(approximate).__name__}")
    approximate.hnsw.efSearch = ef_search
    train_vectors = np.load(train_embeddings_path, mmap_mode="r")
    eval_vectors = np.load(eval_embeddings_path, mmap_mode="r")
    if approximate.ntotal != len(train_vectors):
        raise ValueError("training embedding and approximate-index sizes differ")
    sample_size = min(sample_size, len(eval_vectors))
    sample_rows = np.linspace(0, len(eval_vectors) - 1, sample_size, dtype=np.int64)
    queries = np.asarray(eval_vectors[sample_rows], dtype=np.float32)
    _, approximate_neighbors = approximate.search(queries, top_k)
    exact = faiss.IndexFlatIP(train_vectors.shape[1])
    for start in range(0, len(train_vectors), 10_000):
        exact.add(np.asarray(train_vectors[start : start + 10_000], dtype=np.float32))
    _, exact_neighbors = exact.search(queries, top_k)
    intersections = [
        len(set(approximate_row) & set(exact_row))
        for approximate_row, exact_row in zip(approximate_neighbors, exact_neighbors, strict=True)
    ]
    exact_top1 = sum(
        approximate_row[0] == exact_row[0]
        for approximate_row, exact_row in zip(approximate_neighbors, exact_neighbors, strict=True)
    )
    complete_top_k = sum(overlap == top_k for overlap in intersections)
    return {
        "kind": "pii-overlap-faiss-recall-audit",
        "sample_selection": "evenly spaced evaluation vector rows",
        "sample_size": sample_size,
        "top_k": top_k,
        "ef_search": ef_search,
        "top1_accuracy": exact_top1 / sample_size,
        "top_k_member_recall": sum(intersections) / (sample_size * top_k),
        "complete_top_k_fraction": complete_top_k / sample_size,
        "sample_rows": sample_rows.tolist(),
    }


def _keyed_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line_number, row in jsonl_rows(path):
        identifier = row.get("eval_id")
        if not identifier:
            raise ValueError(f"{path}:{line_number}: missing eval_id")
        if identifier in result:
            raise ValueError(f"{path}:{line_number}: duplicate eval_id {identifier!r}")
        result[identifier] = row
    return result


def fuse_lexical_neighbors(paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Union independent lexical retrievers, then exact-rerank their candidates."""

    if len(paths) < 2:
        raise ValueError("lexical fusion requires at least two input artifacts")
    inputs = [(str(path), _keyed_jsonl(path)) for path in paths]
    reference_ids = inputs[0][1].keys()
    for name, rows in inputs[1:]:
        if rows.keys() != reference_ids:
            raise ValueError(f"evaluation IDs differ in lexical input {name}")
    pairwise_top1 = 0
    pairwise_members = 0
    with atomic_output(output, encoding="utf-8") as destination:
        for eval_id in reference_ids:
            rows = [(name, keyed[eval_id]) for name, keyed in inputs]
            if rows[0][1]["lexical_top3"] and rows[1][1]["lexical_top3"]:
                pairwise_top1 += (
                    rows[0][1]["lexical_top3"][0]["train_id"] == rows[1][1]["lexical_top3"][0]["train_id"]
                )
            first_ids = {item["train_id"] for item in rows[0][1]["lexical_top3"]}
            second_ids = {item["train_id"] for item in rows[1][1]["lexical_top3"]}
            pairwise_members += len(first_ids & second_ids)
            candidates: dict[str, dict[str, Any]] = {}
            retrieved_by: dict[str, list[str]] = {}
            for name, row in rows:
                for candidate in row["lexical_top3"]:
                    train_id = candidate["train_id"]
                    candidates.setdefault(train_id, candidate)
                    retrieved_by.setdefault(train_id, []).append(name)
            eval_normalized = normalize_text(rows[0][1]["text"])
            scored = []
            for train_id, candidate in candidates.items():
                item = dict(candidate)
                item.update(chrf3_6(eval_normalized, normalize_text(candidate["text"])))
                item["retrieved_by"] = retrieved_by[train_id]
                scored.append(item)
            scored.sort(
                key=lambda item: (
                    -item["chrf3_6_f1"],
                    -item["eval_containment"],
                    item["train_id"],
                )
            )
            fused = scored[:3]
            for rank, item in enumerate(fused, 1):
                item["rank"] = rank
            reference = rows[0][1]
            destination.write(
                json.dumps(
                    {
                        "eval_id": eval_id,
                        "eval_dataset": reference["eval_dataset"],
                        "eval_source_line": reference["eval_source_line"],
                        "lang": infer_language(reference, reference["eval_dataset"], eval_id),
                        "text_sha256": reference["text_sha256"],
                        "text": reference["text"],
                        "lexical_top3": fused,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    count = len(reference_ids)
    return {
        "kind": "pii-overlap-fused-lexical-top3",
        "inputs": [name for name, _rows in inputs],
        "evaluations": count,
        "fusion": "candidate union followed by exact chrF 3-6 reranking",
        "first_two_top1_agreement": pairwise_top1 / count if count else 0.0,
        "first_two_top3_member_agreement": pairwise_members / (count * 3) if count else 0.0,
        "output": str(output),
    }


def fuse_semantic_neighbors(paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Merge exact semantic results from independent corpus indexes."""

    if len(paths) < 2:
        raise ValueError("semantic fusion requires at least two input artifacts")
    inputs = [(str(path), _keyed_jsonl(path)) for path in paths]
    reference_ids = inputs[0][1].keys()
    for name, rows in inputs[1:]:
        if rows.keys() != reference_ids:
            raise ValueError(f"evaluation IDs differ in semantic input {name}")
    duplicate_candidates = 0
    with atomic_output(output, encoding="utf-8") as destination:
        for eval_id in reference_ids:
            rows = [(name, keyed[eval_id]) for name, keyed in inputs]
            candidates: dict[str, dict[str, Any]] = {}
            retrieved_by: dict[str, list[str]] = {}
            for name, row in rows:
                for candidate in row["semantic_top3"]:
                    train_id = candidate["train_id"]
                    existing = candidates.get(train_id)
                    if existing is not None:
                        duplicate_candidates += 1
                        identity_fields = ("source_name", "source_line", "text_sha256", "text")
                        conflicting = [
                            field for field in identity_fields if existing.get(field) != candidate.get(field)
                        ]
                        if conflicting:
                            fields = ", ".join(conflicting)
                            raise ValueError(
                                f"training ID {train_id!r} aliases different rows in semantic inputs; "
                                f"conflicting fields: {fields}"
                            )
                        if candidate["cosine"] > existing["cosine"]:
                            candidates[train_id] = dict(candidate)
                    else:
                        candidates[train_id] = dict(candidate)
                    retrieved_by.setdefault(train_id, []).append(name)
            scored = []
            for train_id, candidate in candidates.items():
                item = dict(candidate)
                item["retrieved_by"] = retrieved_by[train_id]
                scored.append(item)
            scored.sort(key=lambda item: (-item["cosine"], item["train_id"]))
            fused = scored[:3]
            for rank, item in enumerate(fused, 1):
                item["rank"] = rank
            reference = rows[0][1]
            destination.write(
                json.dumps(
                    {
                        "eval_id": eval_id,
                        "eval_dataset": reference["eval_dataset"],
                        "eval_source_line": reference["eval_source_line"],
                        "lang": reference.get("lang"),
                        "text_sha256": reference["text_sha256"],
                        "text": reference["text"],
                        "semantic_top3": fused,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "kind": "pii-overlap-fused-semantic-top3",
        "inputs": [name for name, _rows in inputs],
        "evaluations": len(reference_ids),
        "fusion": "candidate union followed by exact cosine ranking",
        "duplicate_candidate_occurrences": duplicate_candidates,
        "output": str(output),
    }


def join_neighbors(lexical_path: Path, semantic_path: Path, output: Path) -> dict[str, Any]:
    lexical = _keyed_jsonl(lexical_path)
    semantic = _keyed_jsonl(semantic_path)
    if lexical.keys() != semantic.keys():
        missing_lexical = sorted(semantic.keys() - lexical.keys())[:5]
        missing_semantic = sorted(lexical.keys() - semantic.keys())[:5]
        raise ValueError(
            f"evaluation IDs differ; missing lexical={missing_lexical}, missing semantic={missing_semantic}"
        )
    shared_count = 0
    with atomic_output(output, encoding="utf-8") as destination:
        for eval_id, lexical_row in lexical.items():
            semantic_row = semantic[eval_id]
            lexical_by_id = {item["train_id"]: item for item in lexical_row["lexical_top3"]}
            semantic_by_id = {item["train_id"]: item for item in semantic_row["semantic_top3"]}
            shared = []
            for train_id in lexical_by_id.keys() & semantic_by_id.keys():
                lexical_item = lexical_by_id[train_id]
                semantic_item = semantic_by_id[train_id]
                shared.append(
                    {
                        "train_id": train_id,
                        "lexical_rank": lexical_item["rank"],
                        "semantic_rank": semantic_item["rank"],
                        "chrf3_6_f1": lexical_item["chrf3_6_f1"],
                        "eval_containment": lexical_item["eval_containment"],
                        "semantic_cosine": semantic_item["cosine"],
                        "source_name": lexical_item["source_name"],
                        "source_line": lexical_item["source_line"],
                        "text": lexical_item["text"],
                    }
                )
            shared.sort(key=lambda item: (item["lexical_rank"], item["semantic_rank"]))
            shared_count += bool(shared)
            destination.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "eval_id": eval_id,
                        "eval_dataset": lexical_row["eval_dataset"],
                        "eval_source_line": lexical_row["eval_source_line"],
                        "lang": lexical_row.get("lang"),
                        "text_sha256": lexical_row["text_sha256"],
                        "text": lexical_row["text"],
                        "lexical_top3": lexical_row["lexical_top3"],
                        "semantic_top3": semantic_row["semantic_top3"],
                        "shared_top3": shared,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "evaluations": len(lexical),
        "with_shared_top3_candidate": shared_count,
        "output": str(output),
        "thresholds_applied": False,
    }


def write_manifest(payload: dict[str, Any], path: Path | None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if path is not None:
        with atomic_output(path, encoding="utf-8") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
            destination.write("\n")


DETECTOR_VERSION = "overlap-detector/2"
"""Detector behavior identity recorded in screening receipts.

Bump it whenever a change here or in the decision rule could change which rows
a screen flags; receipts record it instead of file hashes.
"""


def passing_candidates(
    row: dict[str, Any], lexical_threshold: float, semantic_threshold: float
) -> list[dict[str, Any]]:
    """Shared nearest neighbors that pass both cuts: the paper's overlap rule."""
    return [
        candidate
        for candidate in row["shared_top3"]
        if candidate["chrf3_6_f1"] >= lexical_threshold and candidate["semantic_cosine"] >= semantic_threshold
    ]


PAPER_LEXICAL_THRESHOLD = 0.30
PAPER_SEMANTIC_THRESHOLD = 0.875


def text_id(text: str) -> str:
    """Short stable key for a distinct text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def overlapping(
    texts: dict[str, tuple[str, str]],
    index: list[tuple[str, Path]],
    out: Path,
    *,
    device: str,
    model_name: str = DEFAULT_EMBED_MODEL,
    lexical_threshold: float = PAPER_LEXICAL_THRESHOLD,
    semantic_threshold: float = PAPER_SEMANTIC_THRESHOLD,
    workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """Map each text key that overlaps an indexed row to its strongest such neighbor.

    ``texts`` maps a key to (text, language); ``index`` names JSONL files of
    rows with ``id``, ``text`` and ``lang`` (for example evaluation sets). A text
    overlaps when one indexed row is among its nearest three in both views and
    passes both cuts (by default the PII paper's: chrF F1 >= 0.30 and cosine >=
    0.875); semantic similarity alone never counts. Candidates are sharded
    across ``workers`` lexical query processes. Working files go under ``out``.
    """
    out.mkdir(parents=True, exist_ok=False)
    keys = list(texts)
    shard_count = max(1, min(workers, len(keys)))
    batch = []
    for shard in range(shard_count):
        path = out / f"candidates-{shard:02d}.jsonl"
        with path.open("x", encoding="utf-8") as sink:
            for key in keys[shard::shard_count]:
                text, language = texts[key]
                sink.write(json.dumps({"id": key, "text": text, "lang": language}, ensure_ascii=False) + "\n")
        batch.append((f"candidates-{shard:02d}", path))
    dtype = "float16" if device.startswith("cuda") else "float32"
    embedder = _load_transformer_embedder(model_name, device, dtype)

    def embed(inputs: list[tuple[str, Path]], stem: Path, role: str) -> None:
        embed_jsonl_corpus(
            inputs,
            stem.with_suffix(".npy"),
            stem.with_suffix(".metadata.jsonl"),
            model_name=model_name,
            role=role,
            batch_size=256,
            max_length=512,
            device=device,
            dtype_name=dtype,
            loaded_embedder=embedder,
        )

    build_lexical_database(out / "index.sqlite", index)
    embed(index, out / "index", "passage")
    embed(batch, out / "queries", "query")
    query_lexical_database(
        out / "index.sqlite",
        batch,
        out / "lexical.jsonl",
        top_k=3,
        candidate_limit=128,
        query_term_limit=96,
        workers=shard_count,
    )
    query_exact_semantic_database(
        out / "index.npy",
        out / "index.metadata.jsonl",
        out / "queries.npy",
        out / "queries.metadata.jsonl",
        out / "semantic.jsonl",
        top_k=3,
        block_size=256,
        device=device,
    )
    join_neighbors(out / "lexical.jsonl", out / "semantic.jsonl", out / "joined.jsonl")
    found = {}
    with (out / "joined.jsonl").open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            matches = passing_candidates(record, lexical_threshold, semantic_threshold)
            if matches:
                best = max(matches, key=lambda match: match["chrf3_6_f1"])
                found[record["eval_id"]] = {
                    "index_row": best["train_id"],
                    "lexical_f1": best["chrf3_6_f1"],
                    "cosine": best["semantic_cosine"],
                }
    return found
