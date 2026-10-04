#!/usr/bin/env python3
"""Screen a reader's own candidate text before annotation and write a dedup receipt.

`pii-reproduce.py annotate --dedup-receipt RECEIPT` opens an annotator only
after `scripts.pii_dedup_gate.require_annotation_dedup` recomputes, from bound
nearest-neighbor evidence, which candidate rows are new. This tool produces
that evidence and receipt with the paper's own detector, unchanged:

* lexical: SQLite FTS5 trigram candidates reranked by character 3-6-gram F1
  (`pii_overlap_neighbors` build-lexical / query-lexical);
* semantic: multilingual-E5 vectors and exact blocked cosine top-k
  (`pii_overlap_neighbors` embed / query-exact-semantic);
* the nearest three in each view against the comparison roster, joined; the
  nearest four within the candidate batch with self removed
  (`pii_dedup_gate.prepare_within_neighbors`);
* the frozen cut (lexical F1 0.30, cosine 0.875, same candidate in both lists),
  applied by `pii_dedup_gate.materialize_annotation_intake`, which is also what
  writes `retained.jsonl` and `receipt.json`.

A candidate is rejected when it exactly or partially overlaps any comparison
row, or when it belongs to a within-batch near-duplicate group touching such a
row; of each new group only the first input row is kept.

Comparison sources are declared with `--compare ROLE[,ROLE...]=NAME=PATH`.
The gate requires the roster to cover every role in
`pii_dedup_gate.REQUIRED_ROLES`. It accepts several roles on one source, but a
role declared on a file that is not data of that kind would be a false claim.
So for each required role no `--compare` names, this tool binds one explicit
empty source, `declared-empty`, carrying those roles: the receipt then states
that the reader had no rows in that role, rather than borrowing another
file's identity. `screen.declared_empty_roles` in the receipt lists them.
Comparison files and the candidate file are copied byte-for-byte into
`--out` so the receipt stays verifiable after the originals change; the
receipt records the original paths and hashes.

Limits: the within-batch view needs at least four candidate rows (the gate
requires recorded top-four retrieval). A role label does not prove the
reader's roster is complete; screening only compares against what is named.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acli
import overlaplib as neighbors
from scripts import pii_dedup_gate as gate

MODEL = "intfloat/multilingual-e5-base"
# Revision the paper's receipts recorded; a different hub revision is reported, not refused.
PAPER_MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
POLICY = {
    "detector": "dual-nearest-three",
    "lexical_threshold": 0.3,
    "semantic_threshold": 0.875,
    "semantic_model": MODEL,
}
EMPTY_SOURCE = "declared-empty"
BATCH_SOURCE = "batch"
WITHIN_TOP_K = 4


def parse_compare(value: str) -> tuple[list[str], str, Path]:
    roles_text, sep1, rest = value.partition("=")
    name, sep2, path = rest.partition("=")
    if not (sep1 and sep2 and roles_text and name and path):
        raise argparse.ArgumentTypeError(f"expected ROLE[,ROLE...]=NAME=PATH: {value!r}")
    roles = sorted(set(roles_text.split(",")))
    unknown = [role for role in roles if role not in gate.REQUIRED_ROLES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{name}: unknown role(s) {unknown}; roles: {sorted(gate.REQUIRED_ROLES)}"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name == EMPTY_SOURCE:
        raise argparse.ArgumentTypeError(f"invalid or reserved source name {name!r}")
    return roles, name, Path(path)


def check_candidates(path: Path) -> int:
    """The gate needs unique string ids and string text on every candidate row."""
    ids = set()
    for line_number, row in neighbors.jsonl_rows(path):
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{path}:{line_number}: candidate lacks a string id")
        if identifier in ids:
            raise ValueError(f"{path}:{line_number}: duplicate candidate id {identifier!r}")
        ids.add(identifier)
    if len(ids) < WITHIN_TOP_K:
        raise ValueError(
            f"{path}: {len(ids)} candidate rows; the gate's within-batch check needs at least {WITHIN_TOP_K}"
        )
    return len(ids)


def check_comparison(path: Path) -> int:
    """Rows the gate can read: string text, and `source` an object when present."""
    count = 0
    for line_number, row in neighbors.jsonl_rows(path):
        if "source" in row and not isinstance(row["source"], dict):
            raise ValueError(
                f"{path}:{line_number}: `source` must be an object (the gate reads source.id); "
                "rename that field in a copy of the file"
            )
        count += 1
    return count


def copy_bound(source: Path, target: Path) -> dict[str, str]:
    shutil.copyfile(source, target)
    original = gate.file_identity(source)
    copied = gate.file_identity(target)
    if original["sha256"] != copied["sha256"]:
        raise ValueError(f"{source} changed while being copied")
    return copied


def build_roster(
    compares: list[tuple[list[str], str, Path]], directory: Path
) -> tuple[Path, list, list[str]]:
    directory.mkdir()
    sources, covered = [], set()
    for roles, name, path in compares:
        rows = check_comparison(path)
        record = copy_bound(path, directory / f"{name}.jsonl")
        sources.append(
            {"name": name, "roles": roles, **record, "rows": rows, "copied_from": gate.file_identity(path)}
        )
        covered.update(roles)
    missing = sorted(gate.REQUIRED_ROLES - covered)
    if missing:
        empty = directory / f"{EMPTY_SOURCE}.jsonl"
        empty.touch(exist_ok=False)
        sources.append(
            {
                "name": EMPTY_SOURCE,
                "roles": missing,
                **gate.file_identity(empty),
                "rows": 0,
                "meaning": "The reader declared no data in these roles; nothing was compared for them.",
            }
        )
    if not sum(source["rows"] for source in sources):
        raise ValueError("comparison sources hold no rows; name at least one nonempty --compare file")
    roster_path = directory / "roster.json"
    roster_path.write_text(json.dumps({"schema": gate.ROSTER_SCHEMA, "sources": sources}, indent=1) + "\n")
    gate.verify_roster(gate.file_identity(roster_path))
    return roster_path, sources, missing


def manifest(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def detect(
    candidates: Path, sources: list[dict[str, Any]], out: Path, device: str
) -> tuple[list[Path], dict[str, Any]]:
    """Run both retrieval views against the roster and within the batch; return manifests."""
    named = [(source["name"], Path(source["path"])) for source in sources]
    prior_count = sum(source["rows"] for source in sources)
    batch = [(BATCH_SOURCE, candidates)]
    index = out / "index"
    index.mkdir()
    manifests = []

    def record(payload: dict[str, Any], path: Path) -> None:
        manifests.append(manifest(payload, path))

    dtype = "float16" if device.startswith("cuda") else "float32"
    embedder = neighbors._load_transformer_embedder(MODEL, device, dtype)

    def embed(inputs, stem: Path, role: str) -> dict[str, Any]:
        payload = neighbors.embed_jsonl_corpus(
            inputs,
            stem.with_suffix(".npy"),
            stem.with_suffix(".metadata.jsonl"),
            model_name=MODEL,
            role=role,
            batch_size=128,
            max_length=512,
            device=device,
            dtype_name=dtype,
            loaded_embedder=embedder,
        )
        record(payload, stem.with_suffix(".manifest.json"))
        return payload

    def lexical(database: Path, top_k: int, stem: Path) -> None:
        record(
            neighbors.query_lexical_database(
                database,
                batch,
                stem.with_suffix(".jsonl"),
                top_k=top_k,
                candidate_limit=128,
                query_term_limit=96,
                workers=1,
            ),
            stem.with_suffix(".manifest.json"),
        )

    def semantic(passages: Path, top_k: int, stem: Path) -> None:
        record(
            neighbors.query_exact_semantic_database(
                passages.with_suffix(".npy"),
                passages.with_suffix(".metadata.jsonl"),
                out / "queries.npy",
                out / "queries.metadata.jsonl",
                stem.with_suffix(".jsonl"),
                top_k=top_k,
                block_size=256,
                device=device,
            ),
            stem.with_suffix(".manifest.json"),
        )

    # Prior comparison: nearest three against every roster row.
    record(neighbors.build_lexical_database(index / "lexical.sqlite", named), index / "lexical.manifest.json")
    passages = embed(named, index / "passages", "passage")
    queries = embed(batch, out / "queries", "query")
    lexical(index / "lexical.sqlite", 3, out / "prior-lexical")
    semantic(index / "passages", min(3, prior_count), out / "prior-semantic")
    record(
        neighbors.join_neighbors(
            out / "prior-lexical.jsonl", out / "prior-semantic.jsonl", out / "prior-joined.jsonl"
        ),
        out / "prior-joined.manifest.json",
    )
    # Within the batch: nearest four, then self removed before the nearest-three join.
    record(
        neighbors.build_lexical_database(out / "within.sqlite", batch),
        out / "within-lexical-build.manifest.json",
    )
    lexical(out / "within.sqlite", WITHIN_TOP_K, out / "within-lexical4")
    embed(batch, out / "within-passages", "passage")
    semantic(out / "within-passages", WITHIN_TOP_K, out / "within-semantic4")
    record(
        gate.prepare_within_neighbors(
            candidates,
            BATCH_SOURCE,
            out / "within-lexical4.jsonl",
            out / "within-semantic4.jsonl",
            out / "within",
        ),
        out / "within.manifest.json",
    )
    return manifests, {
        "model": MODEL,
        "model_revision": passages["model_revision"],
        "paper_model_revision": PAPER_MODEL_REVISION,
        "revision_matches_paper": passages["model_revision"] == PAPER_MODEL_REVISION
        and queries["model_revision"] == PAPER_MODEL_REVISION,
        "device": device,
        "compute_dtype": dtype,
    }


def screen(args) -> dict[str, Any]:
    started = time.monotonic()
    compares = args.compare
    names = [name for _, name, _ in compares]
    if len(set(names)) != len(names):
        raise ValueError(f"--compare source names must be unique: {names}")
    rows = check_candidates(args.input)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"--out must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    candidates = out / "input.jsonl"
    candidate_copy = copy_bound(args.input, candidates)
    roster_path, sources, declared_empty = build_roster(compares, out / "roster")
    # Library progress lines go to stderr so stdout stays one result object.
    with contextlib.redirect_stdout(sys.stderr):
        manifests, embedding = detect(candidates, sources, out, args.device)
    bundle = {
        "input": candidate_copy,
        "roster": gate.file_identity(roster_path),
        "prior_neighbors": gate.file_identity(out / "prior-joined.jsonl"),
        "within_neighbors": gate.file_identity(out / "within" / "joined-top3.jsonl"),
        "evidence": [gate.file_identity(path) for path in manifests],
        "policy": POLICY,
        **gate.detector_identity(),
        "screen": {
            "tool": "scripts/pii_software_screen.py",
            "tool_sha256": gate.file_identity(Path(__file__))["sha256"],
            "input_copied_from": gate.file_identity(args.input),
            "comparison": [
                {key: source[key] for key in ("name", "roles", "rows") if key in source} for source in sources
            ],
            "declared_empty_roles": declared_empty,
            "embedding": embedding,
            "scope": "Novelty against the named comparison sources and within this batch only; "
            "a role label does not prove the reader's roster is complete.",
        },
    }
    evidence_path = out / "evidence.json"
    evidence_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=1) + "\n")
    receipt = gate.materialize_annotation_intake(evidence_path, out / "retained.jsonl", out / "receipt.json")
    # Replay the annotator's own check before reporting success.
    retained_rows = gate.read_rows(out / "retained.jsonl")
    if retained_rows:
        gate.require_annotation_dedup(out / "receipt.json", out / "retained.jsonl", retained_rows)
    return {
        "ok": True,
        "out": str(out),
        "input_rows": rows,
        "retained": len(receipt["retained_ids"]),
        "rejected": len(receipt["rejected_ids"]),
        "rejected_ids_first_50": receipt["rejected_ids"][:50],
        "retained_path": str(out / "retained.jsonl"),
        "receipt": str(out / "receipt.json"),
        "declared_empty_roles": declared_empty,
        "model_revision_matches_paper": embedding["revision_matches_paper"],
        "seconds": round(time.monotonic() - started, 1),
        "next": f"pii-reproduce.py annotate --input {out / 'retained.jsonl'} --dedup-receipt {out / 'receipt.json'}"
        if retained_rows
        else "No candidate row is new; nothing to annotate.",
    }


def build_parser():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument(
        "--input", type=Path, required=True, help="Candidate JSONL rows {id, text, lang, ...}"
    )
    parser.add_argument(
        "--compare",
        action="append",
        type=parse_compare,
        default=[],
        metavar="ROLE[,ROLE...]=NAME=PATH",
        help="Existing data to screen against (repeatable). Roles: " + ", ".join(sorted(gate.REQUIRED_ROLES)),
    )
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    acli.add_standard_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    if not args.compare:
        acli.die("name at least one --compare source", acli.ExitCode.USAGE)
    try:
        result = screen(args)
    except (OSError, ValueError) as error:
        acli.die(str(error), acli.ExitCode.DATA)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
