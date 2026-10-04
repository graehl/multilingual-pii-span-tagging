import hashlib
import json

import pytest

from scripts.pii_training_quarantine import POLICY_PATH, validate_training_inputs


def test_local_labels_block_after_reweighting_and_ontology_transfer(tmp_path):
    path = tmp_path / "renamed-pool.jsonl"
    path.write_text(
        json.dumps(
            {
                "mix_source": "frontier-teacher-gemma4-31b-treatment",
                "label_space": "v2",
                "sampling_weight": 0.0,
                "supervision": "annotated_spans_only",
                "text": "x",
                "spans": [],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="quarantined training supervision"):
        validate_training_inputs([path])


def test_complete_gemma_labels_block_in_additional_pool(tmp_path):
    primary = tmp_path / "train.jsonl"
    primary.write_text('{"src":"tab","text":"x","spans":[]}\n')
    extra = tmp_path / "extra.jsonl"
    extra.write_text('{"src":"final35-training-gemma4-fp8-v6-v1","spans":[]}\n')
    with pytest.raises(ValueError, match="extra.jsonl:1"):
        validate_training_inputs([primary, extra])


def test_translator_reviewer_and_text_source_do_not_identify_labeler(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"src": "ar-transport-semantic-policy-v2-admitted", "reviewer": "gemma4"},
        {"src": "final35-training-draw-v1-terra-single-pass", "text": "Gemma Qwen"},
        {"src": "nemotron-full"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    validate_training_inputs([path])


def test_file_identity_blocks_copy_without_known_source_marker(tmp_path):
    path = tmp_path / "copy.jsonl"
    data = b'{"text":"x","spans":[]}\n'
    path.write_bytes(data)
    policy = json.loads(POLICY_PATH.read_text())
    policy["blocked_file_sha256"] = {hashlib.sha256(data).hexdigest(): "fixture"}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="fixture"):
        validate_training_inputs([path], policy_path)
