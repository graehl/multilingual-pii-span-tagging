#!/usr/bin/env python3
"""Unsupervised domain model over sentence embeddings, at segment and document scope.

Fits clusters on the pool's own text and emits, per row, two posteriors over those
clusters: one for the sentence being tagged, and one for the document it was drawn from.
Both are computable at inference from text alone, which is what separates this from the
annotation-source conditioning that production can never supply.

The document posterior is not a pooled segment posterior by default. Pooling cannot create
information the parts lack, so it would differ from the segment view only by averaging.
Embedding the drawn segments joined into one text gives the embedder cross-sentence
context and therefore a genuinely different classifier; `--doc-mode pooled` selects the
cheaper averaging variant for comparison.

Documents here are the 2-4 segments actually drawn from each crawl document, so the
document view is a mild smoothing of the segment view rather than a true document-scale
signal. That attenuation is expected: this exists to decide whether fuller document draws
are worth acquiring, so a null result is weak evidence while a positive one is strong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "pii-domain-model/v1"
DOC_SEPARATOR = "\n"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_posteriors(path: str | Path) -> dict[str, dict[str, Any]]:
    """The `assign` sidecar, keyed by text hash.

    Joining on content rather than on a row identifier is what lets one sidecar serve the
    trainer and the evaluator, which do not share an id space.
    """
    table: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                table[record["text_sha256"]] = record
    return table


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def documents(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group rows by their source document, preserving the order they were drawn in."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        identity = row.get(key)
        if identity:
            grouped.setdefault(identity, []).append(row)
    return grouped


def embedder_max_length(model_name: str) -> int:
    """The model's own position limit, not the shared loader's 8192 default.

    Feeding a 512-position encoder a longer sequence indexes past its position table and
    fails as an opaque device-side assert rather than a clear error, which is exactly what
    joined documents do.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    limit = getattr(config, "max_position_embeddings", None)
    if not isinstance(limit, int) or limit <= 0:
        return 512
    # Room for whatever special tokens the tokenizer adds.
    return max(64, min(8192, limit - 2))


def embed_texts(texts: list[str], model_name: str, batch: int = 64):
    import torch

    from embed_utils import load_embedder

    embed = load_embedder(model_name, log=False, max_length=embedder_max_length(model_name))
    chunks = []
    for start in range(0, len(texts), batch):
        with torch.no_grad():
            chunks.append(embed(texts[start : start + batch]).float().cpu())
    return torch.cat(chunks) if chunks else torch.zeros(0)


def fit(args: argparse.Namespace) -> int:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    rows = read_rows(args.rows)
    texts = [row["text"] for row in rows if row.get("text")]
    if args.limit:
        texts = texts[: args.limit]
    print(f"embedding {len(texts)} segments with {args.embedder}", flush=True)
    vectors = embed_texts(texts, args.embedder, args.batch).numpy()
    print(f"clustering into {args.domains} domains", flush=True)
    kmeans = MiniBatchKMeans(n_clusters=args.domains, random_state=args.seed, n_init=10, batch_size=1024)
    kmeans.fit(vectors)
    counts = np.bincount(kmeans.labels_, minlength=args.domains).tolist()
    spec = {
        "schema": SCHEMA,
        "embedder": args.embedder,
        "domains": args.domains,
        "temperature": args.temperature,
        "seed": args.seed,
        "fitted_on_segments": len(texts),
        "assignment_counts": counts,
        "centroids": kmeans.cluster_centers_.tolist(),
    }
    Path(args.output).write_text(json.dumps(spec) + "\n", encoding="utf-8")
    share = ", ".join(f"{c / len(texts):.1%}" for c in sorted(counts, reverse=True)[:8])
    print(f"wrote {args.output}; largest domain shares: {share}", flush=True)
    return 0


def posteriors(vectors, centroids, temperature: float):
    """Softmax over negative squared distance: a soft assignment, not a hard label."""
    import torch

    distance = torch.cdist(vectors, centroids).pow(2)
    return torch.softmax(-distance / max(temperature, 1e-6), dim=-1)


def assign(args: argparse.Namespace) -> int:
    import torch

    spec = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"unsupported domain model schema {spec.get('schema')!r}")
    centroids = torch.tensor(spec["centroids"], dtype=torch.float32)
    rows = read_rows(args.rows)
    segments = [row.get("text") or "" for row in rows]
    segment_vectors = embed_texts(segments, spec["embedder"], args.batch)
    segment_posterior = posteriors(segment_vectors, centroids, spec["temperature"])

    grouped = documents(rows, args.document_key)
    doc_posterior: dict[str, list[float]] = {}
    if grouped:
        names = list(grouped)
        if args.doc_mode == "concat":
            # The embedder sees the drawn segments as one text, so cross-sentence context
            # reaches the representation; pooling could never produce that.
            joined = [DOC_SEPARATOR.join(r.get("text") or "" for r in grouped[name]) for name in names]
            doc_vectors = embed_texts(joined, spec["embedder"], args.batch)
        else:
            index = {id(row): position for position, row in enumerate(rows)}
            doc_vectors = torch.stack(
                [
                    torch.stack([segment_vectors[index[id(r)]] for r in grouped[name]]).mean(dim=0)
                    for name in names
                ]
            )
        doc_values = posteriors(doc_vectors, centroids, spec["temperature"])
        doc_posterior = {name: doc_values[position].tolist() for position, name in enumerate(names)}

    written = 0
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows):
            text = row.get("text")
            if not text:
                continue
            record = {
                "text_sha256": text_key(text),
                "segment": [round(value, 6) for value in segment_posterior[position].tolist()],
            }
            identity = row.get(args.document_key)
            if identity and identity in doc_posterior:
                record["document"] = [round(value, 6) for value in doc_posterior[identity]]
                record["document_segments"] = len(grouped[identity])
            handle.write(json.dumps(record) + "\n")
            written += 1
    covered = sum(1 for row in rows if row.get(args.document_key) in doc_posterior)
    print(
        f"wrote {written} assignments to {args.output}; "
        f"{covered} carry a document posterior over {len(grouped)} documents "
        f"({args.doc_mode})",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fitter = sub.add_parser("fit", help="cluster segment embeddings into domains")
    fitter.add_argument("--rows", nargs="+", type=Path, required=True)
    fitter.add_argument("--output", type=Path, required=True)
    fitter.add_argument("--embedder", default="intfloat/multilingual-e5-base")
    fitter.add_argument("--domains", type=int, default=16)
    fitter.add_argument("--temperature", type=float, default=0.05)
    fitter.add_argument("--seed", type=int, default=20260910)
    fitter.add_argument("--batch", type=int, default=64)
    fitter.add_argument("--limit", type=int, default=0, help="cap segments used for fitting")
    fitter.set_defaults(func=fit)

    assigner = sub.add_parser("assign", help="emit segment and document posteriors per row")
    assigner.add_argument("--model", type=Path, required=True)
    assigner.add_argument("--rows", nargs="+", type=Path, required=True)
    assigner.add_argument("--output", type=Path, required=True)
    assigner.add_argument("--document-key", default="source_document_sha256")
    assigner.add_argument("--doc-mode", choices=("concat", "pooled"), default="concat")
    assigner.add_argument("--batch", type=int, default=64)
    assigner.set_defaults(func=assign)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
