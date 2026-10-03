"""Verify an annotation intake against recorded partial-overlap decisions.

This module does not discover corpora or substitute a duplicate detector.
The roster is the declared comparison scope; its completeness requires the
program's inventory review. Every referenced byte and retained row is checked
again before an annotation provider can be opened.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.pii_overlap_filter import passing_candidates
from scripts.pii_overlap_neighbors import join_neighbors, normalize_text, sha256_text

SCHEMA = "pii-annotation-dedup-receipt/v1"
ROSTER_SCHEMA = "pii-dedup-comparison-roster/v1"
REQUIRED_ROLES = frozenset(
    {"prior_draws", "annotation_attempts", "training", "development", "validation", "evaluation", "reserved"}
)


def file_identity(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def checked_file(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_absolute() or file_identity(path)["sha256"] != record["sha256"]:
        raise ValueError(f"stale deduplication evidence: {path}")
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed_rows(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        identifier = row.get(field)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f"missing or duplicate {field} in deduplication evidence")
        result[identifier] = row
    return result


def prepare_within_neighbors(
    input_path: Path, source_name: str, lexical_path: Path, semantic_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Map within-batch retrieval to input IDs and remove self before the top-three join."""
    inputs = read_rows(input_path)
    incoming = keyed_rows(inputs, "id")
    # Source+line identities refer to physical JSONL lines, including blanks.
    by_line = {
        number: json.loads(line)
        for number, line in enumerate(input_path.read_text().splitlines(), 1)
        if line.strip()
    }
    if not source_name or not inputs:
        raise ValueError("within-batch retrieval needs a named, nonempty input")
    destinations = [
        output_dir / name for name in ("lexical-top3.jsonl", "semantic-top3.jsonl", "joined-top3.jsonl")
    ]
    if any(path.exists() for path in destinations):
        raise ValueError("within-batch outputs must be new")
    views, removed = [], {}
    for path, field in ((lexical_path, "lexical_top3"), (semantic_path, "semantic_top3")):
        indexed = keyed_rows(read_rows(path), "eval_id")
        if indexed.keys() != incoming.keys():
            raise ValueError("within-batch retrieval does not cover input IDs")
        view, removed[field] = [], 0
        for identifier, row in incoming.items():
            original = indexed[identifier]
            if original["text"] != row["text"] or original["text_sha256"] != sha256_text(
                normalize_text(row["text"])
            ):
                raise ValueError("within-batch query differs from input text")
            neighbors = original[field]
            if original.get("retrieval_top_k") != 4 or len(neighbors) > min(4, len(inputs)):
                raise ValueError(
                    "within-batch preparation requires recorded top-four retrieval; regenerate legacy query outputs"
                )
            if field == "semantic_top3" and len(neighbors) != min(4, len(inputs)):
                raise ValueError("within-batch semantic retrieval is incomplete")
            mapped, seen = [], set()
            for rank, item in enumerate(neighbors, 1):
                line = item["source_line"]
                target = by_line.get(line)
                if (
                    target is None
                    or item["source_name"] != source_name
                    or item["train_id"] != f"{source_name}:{line}"
                    or item["text"] != target["text"]
                    or item["rank"] != rank
                    or item["train_id"] in seen
                ):
                    raise ValueError("within-batch neighbor identity/rank differs from source")
                seen.add(item["train_id"])
                if target["id"] == identifier:
                    removed[field] += 1
                    continue
                mapped.append(
                    dict(item, source_train_id=item["train_id"], train_id=target["id"], rank=len(mapped) + 1)
                )
            view.append(dict(original, **{field: mapped[:3]}))
        views.append(view)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, view in zip(destinations[:2], views, strict=True):
        with path.open("x", encoding="utf-8") as output:
            for row in view:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    joined = join_neighbors(destinations[0], destinations[1], destinations[2])
    return {
        "schema": "pii-dedup-within-neighbors/v1",
        "input": file_identity(input_path),
        "source_name": source_name,
        "raw_lexical": file_identity(lexical_path),
        "raw_semantic": file_identity(semantic_path),
        "removed_self": removed,
        "outputs": [file_identity(path) for path in destinations],
        "join": joined,
        "clearance": "Within-batch evidence only; prior-corpus comparison and annotation admission remain required.",
    }


def verify_roster(record: dict[str, Any]) -> dict[str, Any]:
    roster = json.loads(checked_file(record).read_text())
    if roster.get("schema") != ROSTER_SCHEMA:
        raise ValueError("unsupported deduplication roster schema")
    roles = set()
    names = set()
    for source in roster["sources"]:
        if source["name"] in names:
            raise ValueError("duplicate source name in deduplication roster")
        names.add(source["name"])
        checked_file(source)
        roles.update(source["roles"])
    if roles != REQUIRED_ROLES:
        raise ValueError(f"deduplication roster roles differ: {sorted(roles ^ REQUIRED_ROLES)}")
    return roster


def verify_decisions(receipt: dict[str, Any]) -> list[str]:
    """Recompute rejection from frozen nearest-neighbor evidence, never a pass bit."""
    source = checked_file(receipt["input"])
    rows = read_rows(source)
    incoming = keyed_rows(rows, "id")
    roster = verify_roster(receipt["roster"])
    policy = receipt["policy"]
    if policy != {
        "detector": "dual-nearest-three",
        "lexical_threshold": 0.3,
        "semantic_threshold": 0.875,
        "semantic_model": "intfloat/multilingual-e5-base",
    }:
        raise ValueError("deduplication receipt differs from the frozen detector policy")
    # Bind the actual detector implementation and model/retrieval manifests.
    for artifact in receipt["evidence"]:
        checked_file(artifact)
    if not receipt["evidence"]:
        raise ValueError("deduplication receipt has no model/retrieval evidence")
    for relative, recorded in receipt["code"].items():
        actual = Path(__file__).resolve().parents[1] / relative
        if file_identity(actual)["sha256"] != recorded:
            raise ValueError(f"deduplication detector code changed: {relative}")
    if set(receipt["code"]) != {
        "scripts/pii_overlap_neighbors.py",
        "scripts/pii_overlap_filter.py",
        "scripts/pii_dedup_gate.py",
    }:
        raise ValueError("deduplication receipt lacks the complete detector code identity")

    prior = keyed_rows(read_rows(checked_file(receipt["prior_neighbors"])), "eval_id")
    within = keyed_rows(read_rows(checked_file(receipt["within_neighbors"])), "eval_id")
    if incoming.keys() != prior.keys() or incoming.keys() != within.keys():
        raise ValueError("deduplication neighbor evidence does not cover every input row")

    # Source identities and normalized exact matches are additional exclusions.
    prior_sources, prior_texts, prior_candidates = set(), set(), {}
    for entry in roster["sources"]:
        for line_number, line in enumerate(Path(entry["path"]).read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"roster source lacks text: {entry['path']}")
            text_hash = sha256_text(normalize_text(row["text"]))
            prior_texts.add(text_hash)
            prior_candidates[f"{entry['name']}:{line_number}"] = text_hash
            for value in (row.get("source_document_id"), row.get("source", {}).get("id")):
                if value:
                    prior_sources.add(str(value))

    rejected = set()
    edges = {identifier: set() for identifier in incoming}
    seen_text = {}
    for identifier, row in incoming.items():
        normalized = normalize_text(row["text"])
        if sha256_text(normalized) in prior_texts or row.get("source_document_id") in prior_sources:
            rejected.add(identifier)
        if normalized in seen_text:
            other = seen_text[normalized]
            edges[identifier].add(other)
            edges[other].add(identifier)
        seen_text[normalized] = identifier
        for kind, evidence in (("prior", prior[identifier]), ("within", within[identifier])):
            if evidence["text_sha256"] != sha256_text(normalized) or evidence["text"] != row["text"]:
                raise ValueError(f"deduplication neighbor text mismatch: {identifier}")
            if len(evidence["lexical_top3"]) > 3 or len(evidence["semantic_top3"]) > 3:
                raise ValueError("deduplication requires nearest three after excluding self")
            lexical_ids = {item["train_id"] for item in evidence["lexical_top3"]}
            semantic_ids = {item["train_id"] for item in evidence["semantic_top3"]}
            shared_ids = {item["train_id"] for item in evidence["shared_top3"]}
            if (
                len(lexical_ids) != len(evidence["lexical_top3"])
                or len(semantic_ids) != len(evidence["semantic_top3"])
                or len(shared_ids) != len(evidence["shared_top3"])
            ):
                raise ValueError("duplicate neighbor identity in deduplication evidence")
            if shared_ids != lexical_ids & semantic_ids:
                raise ValueError("deduplication shared candidates do not match both neighbor lists")
            if kind == "within":
                if identifier in lexical_ids | semantic_ids:
                    raise ValueError("within-batch deduplication still contains the query's self-match")
                if not (lexical_ids | semantic_ids) <= incoming.keys():
                    raise ValueError("within-batch neighbor IDs are not incoming row IDs")
            elif not (lexical_ids | semantic_ids) <= prior_candidates.keys():
                raise ValueError("prior neighbor IDs are absent from the comparison roster")
            expected_semantic_count = min(3, len(incoming) - 1 if kind == "within" else len(prior_candidates))
            if len(semantic_ids) != expected_semantic_count:
                raise ValueError("semantic nearest-neighbor evidence is incomplete")
            lexical = {item["train_id"]: item for item in evidence["lexical_top3"]}
            semantic = {item["train_id"]: item for item in evidence["semantic_top3"]}
            for values, score_field, low in (
                (lexical.values(), "chrf3_6_f1", 0),
                (semantic.values(), "cosine", -1),
            ):
                for item in values:
                    score = item[score_field]
                    if (
                        not isinstance(score, (int, float))
                        or not math.isfinite(score)
                        or not low <= score <= 1
                    ):
                        raise ValueError("invalid overlap similarity score")
            for item in evidence["lexical_top3"] + evidence["semantic_top3"]:
                other = item["train_id"]
                expected_hash = (
                    sha256_text(normalize_text(incoming[other]["text"]))
                    if kind == "within"
                    else prior_candidates[other]
                )
                if sha256_text(normalize_text(item["text"])) != expected_hash:
                    raise ValueError("neighbor surface differs from its declared source")
            for item in evidence["shared_top3"]:
                other = item["train_id"]
                if (
                    item["chrf3_6_f1"] != lexical[other]["chrf3_6_f1"]
                    or item["semantic_cosine"] != semantic[other]["cosine"]
                ):
                    raise ValueError("joined overlap scores differ from retrieved scores")
            matches = passing_candidates(evidence, policy["lexical_threshold"], policy["semantic_threshold"])
            if kind == "prior" and matches:
                rejected.add(identifier)
            for match in matches if kind == "within" else []:
                other = match["train_id"]
                edges[identifier].add(other)
                edges[other].add(identifier)

    # Keep the first input member of each new component. If any member overlaps
    # established data, the component cannot establish additional data.
    visited, retained = set(), []
    for identifier in incoming:
        if identifier in visited:
            continue
        component, pending = set(), [identifier]
        while pending:
            member = pending.pop()
            if member in component:
                continue
            component.add(member)
            pending.extend(edges[member] - component)
        visited.update(component)
        if not component & rejected:
            retained.append(identifier)
    return retained


def materialize_annotation_intake(
    evidence_path: Path, output_path: Path, receipt_path: Path
) -> dict[str, Any]:
    """Materialize only the recomputed novel components of a reviewed evidence bundle."""
    if output_path.exists() or receipt_path.exists() or output_path.resolve() == receipt_path.resolve():
        raise ValueError("deduplicated output and receipt must be new, distinct paths")
    receipt = json.loads(evidence_path.read_text())
    receipt.update(schema=SCHEMA, purpose="additional_annotation")
    retained = verify_decisions(receipt)
    original = keyed_rows(read_rows(Path(receipt["input"]["path"])), "id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for identifier in retained:
            output.write(json.dumps(original[identifier], ensure_ascii=False) + "\n")
    receipt.update(
        retained=file_identity(output_path),
        retained_ids=retained,
        rejected_ids=[identifier for identifier in original if identifier not in set(retained)],
        evidence_bundle=file_identity(evidence_path),
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def require_annotation_dedup(
    receipt_path: Path | None, input_path: Path, docs: list[dict[str, Any]]
) -> dict[str, str]:
    if receipt_path is None:
        raise ValueError(
            "--dedup-receipt is required before annotation; exact/hash-only checks do not qualify"
        )
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("purpose") != "additional_annotation":
        raise ValueError("unsupported annotation deduplication receipt")
    retained_path = checked_file(receipt["retained"])
    if input_path.resolve() != retained_path.resolve():
        raise ValueError("--gold must name the deduplicated file bound by --dedup-receipt")
    retained = verify_decisions(receipt)
    if retained != receipt["retained_ids"]:
        raise ValueError("deduplication retained membership differs from recomputed decisions")
    materialized = read_rows(retained_path)
    original = keyed_rows(read_rows(Path(receipt["input"]["path"])), "id")
    if materialized != [original[identifier] for identifier in retained]:
        raise ValueError("deduplicated rows differ from the original retained input rows")
    if docs != materialized[: len(docs)] or not docs:
        raise ValueError("annotation rows must be a nonempty prefix of the deduplicated input")
    return file_identity(receipt_path)


def require_evaluation_replay(admission_path: Path, input_path: Path, docs: list[dict]) -> dict:
    """Replay a previously admitted Ont3 evaluation file without changing its role."""
    admission = json.loads(admission_path.read_text())
    if admission.get("schema") != "pii-context-continuation-admission/v2":
        raise ValueError("evaluation replay requires the existing Ont3 context admission")
    source = checked_file(admission["outputs"]["evaluation"])
    if source.resolve() != input_path.resolve():
        raise ValueError("evaluation replay must use the exact admitted evaluation file")
    original = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    keyed_rows(original, "id")
    if not docs or docs != original[: len(docs)]:
        raise ValueError("evaluation replay must preserve admitted rows and order")
    if len(original) != admission["outputs"]["evaluation"]["rows"]:
        raise ValueError("evaluation replay admission row count differs")
    return {
        **file_identity(admission_path),
        "purpose": "existing_evaluation_inference",
        "clearance": "No new data admission, training use, or fresh-test claim.",
    }


def require_training_reannotation(
    receipt_path: Path, input_path: Path, docs: list[dict[str, Any]]
) -> dict[str, str]:
    """Verify deliberate repeated annotation without granting novelty clearance.

    The user authorized paired prompt studies on existing training annotations.
    A receipt binds that study and exact prior annotated text. Its training role
    still requires provenance review; this is not an admission or quality gate.
    """
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "pii-training-reannotation-receipt/v1"
        or receipt.get("purpose") != "training_reannotation"
    ):
        raise ValueError("unsupported training reannotation receipt")
    materialized_path = checked_file(receipt["input"])
    if materialized_path.resolve() != input_path.resolve():
        raise ValueError("--gold must name the input bound by --reannotation-receipt")
    materialized = read_rows(materialized_path)
    keyed_rows(materialized, "id")
    if not docs or docs != materialized[: len(docs)]:
        raise ValueError("reannotation rows must be a nonempty prefix of the bound input")
    experiment = json.loads(checked_file(receipt["experiment"]).read_text())
    if (
        experiment.get("purpose") != "paired_prompt_reannotation"
        or experiment.get("training_weight_policy") != "equal_share_per_source_segment"
    ):
        raise ValueError("reannotation needs a paired study with shared source-segment weight")
    prior = set()
    for source in receipt["prior_annotations"]:
        if source.get("role") != "training":
            raise ValueError("training reannotation cannot select reserved or evaluation annotations")
        for row in read_rows(checked_file(source)):
            if not isinstance(row.get("spans"), list):
                raise ValueError("reannotation source must contain prior span annotations")
            if row.get("split", "train") != "train":
                raise ValueError("reannotation source contains a non-training split")
            prior.add((row["id"], row["lang"], row["text"]))
    for row in materialized:
        if (row["id"], row["lang"], row["text"]) not in prior:
            raise ValueError(f"reannotation input is not previously annotated training text: {row['id']}")
    return file_identity(receipt_path)


def require_o4_web_intake(receipt_path: Path, docs: list[dict[str, Any]]) -> dict[str, str]:
    """Admit only rows that are exactly O4's own hash-verified web training text.

    `pii_fetch_web.py` recovers these rows from public FineWeb and binds each
    (id, language, text hash) in its receipt. They were training text already,
    so re-annotating them grants no novelty; the check proves they are nothing
    else.
    """
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "pii-o4-web-intake-receipt/v1":
        raise ValueError(f"{receipt_path} is not a fetch-web receipt")
    admitted = {tuple(item) for item in receipt["row_hashes"]}
    if not docs:
        raise ValueError("web intake needs at least one row")
    for doc in docs:
        if (doc["id"], doc["lang"], sha256_text(doc["text"])) not in admitted:
            raise ValueError(f"row {doc['id']} is not a verified O4 web row in {receipt_path}")
    return file_identity(receipt_path)
