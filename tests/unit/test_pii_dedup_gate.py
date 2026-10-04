"""Recorded detector evidence must gate the real annotation entrypoint."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from overlaplib import chrf3_6, normalize_text, sha256_text
from scripts import pii_api_label as labeler
from scripts import pii_dedup_gate as gate
from scripts import pii_overlap_filter as overlap_filter


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")
    return gate.file_identity(path)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return gate.file_identity(path)


def neighbors(row, candidates):
    lexical, semantic, shared = [], [], []
    for index, (other, cosine) in enumerate(candidates, 1):
        scores = chrf3_6(normalize_text(row["text"]), normalize_text(other["text"]))
        score = scores["chrf3_6_f1"]
        lexical.append({"train_id": other["id"], "rank": index, "text": other["text"], **scores})
        semantic.append({"train_id": other["id"], "rank": index, "text": other["text"], "cosine": cosine})
        shared.append({"train_id": other["id"], "chrf3_6_f1": score, "semantic_cosine": cosine})
    return {
        "eval_id": row["id"],
        "text": row["text"],
        "text_sha256": sha256_text(normalize_text(row["text"])),
        "lexical_top3": lexical,
        "semantic_top3": semantic,
        "shared_top3": shared,
    }


def make_evidence(tmp_path):
    prior = {"id": "history:1", "text": "The patient's visit was booked for Monday."}
    incoming = [
        {"id": "near", "lang": "en", "text": "The patient's visit was booked for Tuesday."},
        {"id": "new", "lang": "en", "text": "Alex photographed a comet beside the observatory."},
        {"id": "within", "lang": "en", "text": "Alex photographed a comet behind the observatory."},
    ]
    source = write_rows(tmp_path / "prior.jsonl", [prior])
    source.update(name="history", roles=sorted(gate.REQUIRED_ROLES))
    roster = write_json(tmp_path / "roster.json", {"schema": gate.ROSTER_SCHEMA, "sources": [source]})
    prior_neighbors = [neighbors(row, [(prior, 0.97 if row["id"] == "near" else 0.2)]) for row in incoming]
    within_neighbors = [
        neighbors(
            row,
            [
                (other, 0.97 if {row["id"], other["id"]} == {"new", "within"} else 0.2)
                for other in incoming
                if other["id"] != row["id"]
            ],
        )
        for row in incoming
    ]
    payload = {
        "input": write_rows(tmp_path / "incoming.jsonl", incoming),
        "roster": roster,
        "prior_neighbors": write_rows(tmp_path / "prior-neighbors.jsonl", prior_neighbors),
        "within_neighbors": write_rows(tmp_path / "within-neighbors.jsonl", within_neighbors),
        "evidence": [
            write_json(tmp_path / "retrieval.json", {"fixture": "frozen model output; no provider calls"})
        ],
        "policy": {
            "detector": "dual-nearest-three",
            "lexical_threshold": 0.3,
            "semantic_threshold": 0.875,
            "semantic_model": "intfloat/multilingual-e5-base",
        },
        **gate.detector_identity(),
    }
    return payload


def legacy_code(revision: int = 0) -> dict:
    """A pre-overlaplib receipt's code identity: one committed revision of each original file."""
    known = json.loads(gate.LEGACY_CODE_HASHES.read_text())["code_sha256"]
    return {relative: hashes[revision] for relative, hashes in known.items()}


def test_legacy_receipt_with_committed_detector_revisions_verifies():
    gate.verify_detector_identity({"code": legacy_code()})


def test_legacy_receipt_with_uncommitted_detector_code_is_refused():
    code = legacy_code()
    code["scripts/pii_dedup_gate.py"] = hashlib.sha256(b"edited locally").hexdigest()
    with pytest.raises(ValueError, match="not a committed revision"):
        gate.verify_detector_identity({"code": code})


def test_receipt_from_another_detector_version_is_refused():
    with pytest.raises(ValueError, match="is not the current"):
        gate.verify_detector_identity({**gate.detector_identity(), "detector_version": "overlap-detector/1"})


@pytest.fixture
def evidence(tmp_path):
    return make_evidence(tmp_path)


def materialize(tmp_path, evidence):
    bundle, output, receipt = tmp_path / "evidence.json", tmp_path / "clean.jsonl", tmp_path / "receipt.json"
    write_json(bundle, evidence)
    assert (
        overlap_filter.main(
            [
                "admit-annotation",
                "--evidence-receipt",
                str(bundle),
                "--clean-output",
                str(output),
                "--manifest",
                str(receipt),
            ]
        )
        == 0
    )
    return output, receipt


def test_partial_overlap_and_within_batch_dedup_survive_materialization(tmp_path, evidence):
    source_rows = gate.read_rows(Path(evidence["input"]["path"]))
    assert len({sha256_text(row["text"]) for row in source_rows}) == 3
    output, receipt = materialize(tmp_path, evidence)
    rows = gate.read_rows(output)
    assert [row["id"] for row in rows] == ["new"]
    assert gate.require_annotation_dedup(receipt, output, rows) == gate.file_identity(receipt)
    assert json.loads(receipt.read_text())["rejected_ids"] == ["near", "within"]


@pytest.mark.parametrize("surface", ["input", "roster", "prior_neighbors", "within_neighbors"])
def test_stale_evidence_blocks_reuse(tmp_path, evidence, surface):
    output, receipt = materialize(tmp_path, evidence)
    path = Path(evidence[surface]["path"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="stale deduplication"):
        gate.require_annotation_dedup(receipt, output, gate.read_rows(output))


def test_incomplete_roster_blocks_materialization(tmp_path, evidence):
    path = Path(evidence["roster"]["path"])
    roster = json.loads(path.read_text())
    roster["sources"][0]["roles"].remove("annotation_attempts")
    evidence["roster"] = write_json(path, roster)
    with pytest.raises(ValueError, match="roster roles"):
        materialize(tmp_path, evidence)
    assert not (tmp_path / "clean.jsonl").exists()


@pytest.mark.parametrize("failure", ["self", "missing", "joined_score", "empty_semantic"])
def test_malformed_neighbor_evidence_cannot_claim_clearance(tmp_path, evidence, failure):
    path = Path(evidence["within_neighbors"]["path"])
    rows = gate.read_rows(path)
    if failure == "self":
        rows[0]["lexical_top3"][0]["train_id"] = rows[0]["eval_id"]
    elif failure == "missing":
        rows.pop()
    elif failure == "joined_score":
        rows[0]["shared_top3"][0]["semantic_cosine"] = 0.99
    else:
        rows[0]["semantic_top3"] = []
        rows[0]["shared_top3"] = []
    evidence["within_neighbors"] = write_rows(path, rows)
    with pytest.raises(ValueError):
        materialize(tmp_path, evidence)
    assert not (tmp_path / "clean.jsonl").exists()


def test_rejected_row_cannot_be_reinserted_into_retained_file(tmp_path, evidence):
    output, receipt_path = materialize(tmp_path, evidence)
    receipt = json.loads(receipt_path.read_text())
    receipt["retained"] = write_rows(output, gate.read_rows(Path(evidence["input"]["path"])))
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="original retained"):
        gate.require_annotation_dedup(receipt_path, output, gate.read_rows(output))


@pytest.mark.parametrize("stale", [False, True])
def test_annotation_cli_checks_gate_before_any_provider_request(tmp_path, evidence, monkeypatch, stale):
    output, receipt = materialize(tmp_path, evidence)
    examples = tmp_path / "examples.json"
    write_json(examples, {"en": {"text": "Ada", "labels": [{"t": "Ada", "type": "person_name"}]}})
    template = tmp_path / "task.txt"
    template.write_text("{tags}\n{format_rules}\n{example}\n{text}")
    predictions = tmp_path / "predictions.jsonl"
    calls = []

    async def request(*args, **kwargs):
        calls.append(kwargs)
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "[]"}], "usage": {}}, 0.1

    monkeypatch.setattr(labeler, "request_one", request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_api_label",
            "--model",
            "test-model",
            "--gold",
            str(output),
            "--dedup-receipt",
            str(receipt),
            "--out",
            str(predictions),
            "--task-template",
            str(template),
            "--examples",
            str(examples),
            "--tags",
            "person_name",
            "--fmt",
            "json-seq",
        ],
    )
    if stale:
        source = Path(evidence["roster"]["path"])
        source.write_text(source.read_text() + "\n")
        with pytest.raises(ValueError, match="stale deduplication"):
            labeler.main()
        assert not predictions.exists() and not calls
    else:
        labeler.main()
        assert len(calls) == 1
        raw = json.loads(Path(str(predictions) + ".raw.jsonl").read_text())
        assert raw["deduplication_receipt"] == gate.file_identity(receipt)


@pytest.mark.parametrize("changed_input", [False, True])
def test_explicit_reannotation_cli_requires_previously_annotated_training_text(
    tmp_path, monkeypatch, changed_input
):
    original = {
        "id": "prior",
        "lang": "en",
        "text": "Dr. Ada Lovelace arrived.",
        "spans": [[0, 16, "person_name"]],
    }
    prior = write_rows(tmp_path / "prior-train.jsonl", [original])
    row = {key: original[key] for key in ("id", "lang", "text")}
    if changed_input:
        row["text"] = "New unannotated data."
    intake = write_rows(tmp_path / "input.jsonl", [row])
    experiment = write_json(
        tmp_path / "experiment.json",
        {
            "purpose": "paired_prompt_reannotation",
            "training_weight_policy": "equal_share_per_source_segment",
        },
    )
    receipt_path = tmp_path / "reannotation.json"
    write_json(
        receipt_path,
        {
            "schema": "pii-training-reannotation-receipt/v1",
            "purpose": "training_reannotation",
            "input": intake,
            "prior_annotations": [dict(prior, role="training")],
            "experiment": experiment,
        },
    )
    examples = tmp_path / "examples.json"
    write_json(examples, {"en": {"text": "Ada", "labels": [{"t": "Ada", "type": "person_name"}]}})
    template = tmp_path / "task.txt"
    template.write_text("{tags}\n{format_rules}\n{example}\n{text}")
    predictions = tmp_path / "predictions.jsonl"
    calls = []

    async def request(*args, **kwargs):
        calls.append(kwargs)
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "[]"}], "usage": {}}, 0.1

    monkeypatch.setattr(labeler, "request_one", request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_api_label",
            "--model",
            "test-model",
            "--gold",
            str(tmp_path / "input.jsonl"),
            "--reannotation-receipt",
            str(receipt_path),
            "--out",
            str(predictions),
            "--task-template",
            str(template),
            "--examples",
            str(examples),
            "--tags",
            "person_name",
            "--fmt",
            "json-seq",
        ],
    )
    if changed_input:
        with pytest.raises(ValueError, match="previously annotated"):
            labeler.main()
        assert not predictions.exists() and not calls
    else:
        labeler.main()
        assert len(calls) == 1
        raw = json.loads(Path(str(predictions) + ".raw.jsonl").read_text())
        assert raw["reannotation_receipt"] == gate.file_identity(receipt_path)
        assert raw["deduplication_receipt"] is None


@pytest.mark.parametrize("sparse_lexical", [False, True])
def test_within_preparation_removes_only_self_and_preserves_three_neighbors(tmp_path, sparse_lexical):
    rows = [
        {"id": str(i), "lang": "en", "text": "same copied text" if i < 2 else f"different text {i}"}
        for i in range(4)
    ]
    input_path = tmp_path / "input.jsonl"
    write_rows(input_path, rows)
    lexical_rows, semantic_rows = [], []
    for i, row in enumerate(rows):
        candidates = [(dict(other, id=f"intake:{j + 1}"), 0.9) for j, other in enumerate(rows)]
        joined = neighbors(row, candidates)
        for field in ("lexical_top3", "semantic_top3"):
            for j, item in enumerate(joined[field]):
                item.update(source_name="intake", source_line=j + 1)
        joined.update(eval_dataset="intake", eval_source_line=i + 1, lang="en", retrieval_top_k=4)
        if sparse_lexical and i == 3:
            joined["lexical_top3"] = [dict(joined["lexical_top3"][3], rank=1)]
        lexical_rows.append(
            {key: value for key, value in joined.items() if key not in {"semantic_top3", "shared_top3"}}
        )
        semantic_rows.append(
            {key: value for key, value in joined.items() if key not in {"lexical_top3", "shared_top3"}}
        )
    lexical_path, semantic_path = tmp_path / "lexical.jsonl", tmp_path / "semantic.jsonl"
    write_rows(lexical_path, lexical_rows)
    write_rows(semantic_path, semantic_rows)
    output = tmp_path / "prepared"
    assert (
        overlap_filter.main(
            [
                "prepare-within",
                "--input",
                str(input_path),
                "--source-name",
                "intake",
                "--lexical",
                str(lexical_path),
                "--semantic",
                str(semantic_path),
                "--output-dir",
                str(output),
                "--manifest",
                str(tmp_path / "prepared.json"),
            ]
        )
        == 0
    )
    for row in gate.read_rows(output / "joined-top3.jsonl"):
        if sparse_lexical and row["eval_id"] == "3":
            assert not row["lexical_top3"] and not row["shared_top3"]
            assert len(row["semantic_top3"]) == 3
            continue
        assert {item["train_id"] for item in row["shared_top3"]} == {"0", "1", "2", "3"} - {row["eval_id"]}
        assert len(row["lexical_top3"]) == len(row["semantic_top3"]) == 3
    assert gate.read_rows(output / "joined-top3.jsonl")[0]["shared_top3"][0]["train_id"] == "1"
