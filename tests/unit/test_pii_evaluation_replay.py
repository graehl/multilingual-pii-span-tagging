"""Existing evaluation inference preserves the admitted bytes and role."""

import json

import pytest

from scripts.pii_api_label import build_payload, codex_output_health, require_evaluation_replay
from scripts.pii_dedup_gate import file_identity


def test_qwen_no_effort_disables_template_thinking():
    payload = build_payload("qwen3.8-flash-next", "Label this.", 100, "none", "openai")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in build_payload("another-model", "Label this.", 100, "none", "openai")


def test_occurrence_health_preserves_full_reference_and_rejects_invalid_slot():
    text = "Google's lawyer"
    tags = {"organization", "person_reference"}
    rows = [{"t": "Google", "type": "organization", "n": 1}, {"t": text, "type": "person_reference", "n": 1}]
    assert not codex_output_health(json.dumps(rows), text, tags, "json", allow_overlapping_spans=True)[
        "unhealthy"
    ]
    assert codex_output_health(json.dumps(rows), text, tags, "json")["unhealthy"]
    rows[0]["n"] = "wrong"
    assert codex_output_health(json.dumps(rows), text, tags, "json", allow_overlapping_spans=True)[
        "unhealthy"
    ]


def test_evaluation_replay_rejects_changed_text_and_wrong_input(tmp_path):
    rows = [{"id": "one", "lang": "en", "text": "Call Alex.", "spans": [[5, 9, "person_name"]]}]
    source = tmp_path / "evaluation.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n")
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "schema": "pii-context-continuation-admission/v2",
                "outputs": {"evaluation": {**file_identity(source), "rows": 1}},
            }
        )
    )
    assert require_evaluation_replay(admission, source, rows)["purpose"] == "existing_evaluation_inference"
    with pytest.raises(ValueError, match="exact admitted"):
        require_evaluation_replay(admission, tmp_path / "other.jsonl", rows)
    with pytest.raises(ValueError, match="preserve admitted"):
        require_evaluation_replay(admission, source, [{**rows[0], "text": "Call Pat."}])
    source.write_text(json.dumps({**rows[0], "text": "Call Pat."}) + "\n")
    with pytest.raises(ValueError, match="stale"):
        require_evaluation_replay(admission, source, rows)
