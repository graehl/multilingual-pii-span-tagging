import asyncio
import json
import sys
from pathlib import Path

import pytest

import scripts.pii_api_label as labeler
from scripts.pii_api_label import (
    CODEX_CONTEXT_CONFIG,
    CODEX_DISABLED_FEATURES,
    EFFORT_LEVELS,
    RequestFailure,
    build_codex_command,
    build_codex_resume_command,
    build_payload,
    codex_compaction_event_types,
    codex_document_attempts,
    codex_exit_failure,
    codex_output_health,
    codex_output_unhealthy,
    codex_session_id,
    enforce_codex_input_limit,
    paragraph_aligned_commit_ranges,
    parse_codex_events,
    plan_document_session_parts,
    require_fresh_protocol_resume_paths,
    require_isolated_codex_home,
    require_resume_pair,
    require_transport_failure_resume,
    response_text,
    resumed_codex_sessions,
    write_prompt_contract,
)
from scripts.pii_codex_app_server import CodexAppServerError
from scripts.pii_subclass import load_subclass_spec


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_payload_uses_anthropic_effort_contract():
    payload = build_payload("gpt-5.6-terra", "label this", 1536, "max")

    assert payload == {
        "model": "gpt-5.6-terra",
        "max_tokens": 1536,
        "messages": [{"role": "user", "content": "label this"}],
        "stream": False,
        "output_config": {"effort": "max"},
    }
    assert EFFORT_LEVELS == ("none", "low", "medium", "high", "xhigh", "max")


def test_build_payload_disables_anthropic_gateway_reasoning_when_effort_is_none():
    payload = build_payload("model", "prompt", 10, "none")

    assert payload["output_config"] == {"effort": "none"}
    assert "thinking" not in payload


def test_build_payload_uses_deterministic_openai_chat_contract():
    payload = build_payload("local-model", "label this", 512, "none", "openai")

    assert payload == {
        "model": "local-model",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "label this"}],
        "stream": False,
        "temperature": 0.0,
        "reasoning_effort": "none",
    }


def test_response_text_concatenates_only_text_blocks():
    response = {
        "content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "["},
            {"type": "text", "text": "]"},
        ]
    }

    assert response_text(response) == "[]"


def test_response_text_rejects_missing_text():
    with pytest.raises(RequestFailure, match="non-empty text"):
        response_text({"content": [{"type": "thinking", "thinking": "x"}]})


def test_openai_response_text_and_usage_are_normalized():
    response = {
        "choices": [{"message": {"role": "assistant", "content": "[]"}}],
        "usage": {
            "prompt_tokens": 23,
            "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 10},
        },
    }

    assert response_text(response) == "[]"
    assert labeler.usage_details(response) == {
        "input_tokens": 23,
        "cached_input_tokens": 10,
        "cache_write_input_tokens": 0,
        "output_tokens": 6,
        "reasoning_output_tokens": 0,
    }


def test_messages_cache_tokens_are_in_total_and_survive_retry_aggregation():
    raw = {
        "input_tokens": 3,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 3970,
        "output_tokens": 114,
    }
    expected = {
        "input_tokens": 3983,
        "cached_input_tokens": 10,
        "cache_write_input_tokens": 3970,
        "output_tokens": 114,
        "reasoning_output_tokens": 0,
    }
    assert labeler.usage_details({"usage": raw}) == expected
    assert raw["input_tokens"] == 3
    combined = {key: value * 2 for key, value in expected.items()}
    assert labeler.usage_details({"aggregate_usage": combined, "usage": raw}) == combined
    assert labeler.usage_counts({"usage": raw}) == (3983, 114)


def test_openai_response_text_strips_inline_thinking_wrapper():
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<think>\n\n</think>\n\n{"candidate_decisions": []}',
                }
            }
        ]
    }

    assert response_text(response) == '{"candidate_decisions": []}'


def test_stateless_api_retries_one_unhealthy_format(monkeypatch):
    responses = [
        {
            "choices": [{"message": {"role": "assistant", "content": "not-json"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '[{"i":1,"t":"Alex","type":"person_name"}]',
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    ]

    async def fake_request(*_args, **_kwargs):
        return responses.pop(0), 0.1

    monkeypatch.setattr(labeler, "request_one", fake_request)
    response, latency, session = asyncio.run(
        labeler.request_api_annotation(
            object(),
            asyncio.Semaphore(1),
            url="http://localhost/v1/chat/completions",
            backend="openai",
            model="local-model",
            prompt="prompt",
            max_tokens=512,
            effort="none",
            transport_retries=0,
            format_retries=1,
            index=0,
            doc={"text": "Alex"},
            tagset={"person_name"},
            fmt="json-seq",
        )
    )

    assert latency == pytest.approx(0.2)
    assert session is None
    assert response["annotation_health"]["unhealthy"] is False
    assert response["format_attempts"] == 2
    assert response["format_recovery_responses"][0]["annotation_health"]["unhealthy"] is True
    assert response["aggregate_usage"]["input_tokens"] == 20
    assert response["aggregate_usage"]["output_tokens"] == 6


@pytest.mark.parametrize("first_kind", ["healthy", "invalid", "capped"])
@pytest.mark.parametrize("retry_bad", [False, True])
def test_single_fresh_terra_retry_preserves_both_models(monkeypatch, first_kind, retry_bad):
    payloads = []

    async def fake_request(*_args, **kwargs):
        payloads.append(kwargs["payload"])
        first = len(payloads) == 1
        bad = first_kind == "invalid" if first else retry_bad
        return {
            "model": kwargs["payload"]["model"],
            "content": [
                {"type": "text", "text": "invalid" if bad else '[{"t":"Alex","type":"person_name"}]'}
            ],
            "stop_reason": "max_tokens" if first and first_kind == "capped" else "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }, 0.1

    monkeypatch.setattr(labeler, "request_one", fake_request)
    response, _, _ = asyncio.run(
        labeler.request_api_annotation(
            object(),
            asyncio.Semaphore(1),
            url="http://localhost/v1/messages",
            backend="anthropic",
            model="gpt-5.6-luna",
            prompt="same exact annotation request",
            max_tokens=512,
            effort="low",
            transport_retries=0,
            format_retries=1,
            format_retry_model="gpt-5.6-terra",
            format_retry_effort="low",
            index=0,
            doc={"text": "Alex"},
            tagset={"person_name"},
            fmt="json-seq",
        )
    )
    retry = first_kind != "healthy"
    assert [p["model"] for p in payloads] == (
        ["gpt-5.6-luna", "gpt-5.6-terra"] if retry else ["gpt-5.6-luna"]
    )
    assert response["format_attempts"] == (2 if retry else 1)
    assert response["annotation_health"]["unhealthy"] == (retry and retry_bad)
    assert response["annotation_request"] == {"model": response["model"], "effort": "low"}
    if retry:
        assert payloads[0]["messages"] == payloads[1]["messages"]
        initial = response["format_recovery_responses"][0]
        assert initial["model"] == "gpt-5.6-luna"
        assert initial["annotation_health"]["unhealthy"]
        assert response["aggregate_usage"]["input_tokens"] == 20
        if first_kind == "capped":
            assert "unfinished_response" in initial["annotation_health"]["reasons"]


def test_retry_policy_requires_one_explicit_stateless_attempt():
    for model, effort, retries, backend in [
        (None, "low", 1, "anthropic"),
        ("terra", None, 1, "anthropic"),
        ("terra", "low", 2, "anthropic"),
        ("terra", "low", 1, "codex"),
    ]:
        with pytest.raises(ValueError, match="requires"):
            labeler.format_retry_policy(model, effort, retries, backend)


def test_cli_assembles_each_retry_models_own_include(tmp_path, monkeypatch):
    from test_pii_dedup_gate import make_evidence, materialize

    task = tmp_path / "task.txt"
    task.write_text('{{include "shared.txt"}}{{include-model "*gemma*" "gemma.txt"}}{example}\n{text}')
    (tmp_path / "shared.txt").write_text("Shared rules: {tags}\n{format_rules}\n")
    (tmp_path / "gemma.txt").write_text("Gemma name correction.\n")
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps({"en": {"text": "Ada", "labels": [{"t": "Ada", "type": "person_name"}]}}))
    gold, dedup_receipt = materialize(tmp_path, make_evidence(tmp_path))
    output, contract = tmp_path / "pred.jsonl", tmp_path / "contract.json"
    payloads = []

    async def fake_request(*_args, **kwargs):
        payloads.append(kwargs["payload"])
        return {
            "model": kwargs["payload"]["model"],
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "invalid" if len(payloads) == 1 else '[{"i":1,"t":"Alex","type":"person_name"}]',
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }, 0.1

    monkeypatch.setattr(labeler, "request_one", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_api_label",
            "--backend",
            "anthropic",
            "--model",
            "gemma4-12b",
            "--gold",
            str(gold),
            "--dedup-receipt",
            str(dedup_receipt),
            "--out",
            str(output),
            "--task-template",
            str(task),
            "--examples",
            str(examples),
            "--tags",
            "person_name",
            "--fmt",
            "json-seq",
            "--prompt-contract-out",
            str(contract),
            "--effort",
            "low",
            "--format-retries",
            "1",
            "--format-retry-model",
            "gpt-5.6-luna",
            "--format-retry-effort",
            "low",
        ],
    )
    labeler.main()
    assert [p["model"] for p in payloads] == ["gemma4-12b", "gpt-5.6-luna"]
    assert "Gemma name correction." in json.dumps(payloads[0]["messages"])
    assert "Gemma name correction." not in json.dumps(payloads[1]["messages"])
    assert all("Shared rules:" in json.dumps(p["messages"]) for p in payloads)
    receipt = json.loads(contract.read_text())
    assert receipt["template_assembly"]["includes"][-1]["selected"]
    assert not receipt["format_retry_template_assembly"]["includes"][-1]["selected"]
    raw = json.loads(Path(str(output) + ".raw.jsonl").read_text())
    assert "Gemma name correction." not in raw["format_retry_prompt"]
    assert raw["response"]["annotation_request"]["model"] == "gpt-5.6-luna"
    assert json.loads(output.read_text())["preds"] == [{"start": 0, "end": 4, "label": "person_name"}]
    (tmp_path / "gemma.txt").write_text("Changed model instruction.\n")
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    with pytest.raises(ValueError, match="Prompt includes differ"):
        labeler.main()
    assert len(payloads) == 2


def test_resume_cannot_add_remove_or_change_retry_teacher():
    policy = labeler.format_retry_policy("terra", "low", 1, "anthropic")
    contract = {"model": "luna", "backend": {"effort": "low", "format_retry": policy}}
    labeler.require_retry_policy_resume(contract, "luna", "low", policy)
    for model, effort, proposed in [
        ("terra", "low", policy),
        ("luna", "high", policy),
        ("luna", "low", None),
    ]:
        with pytest.raises(ValueError, match="differs"):
            labeler.require_retry_policy_resume(contract, model, effort, proposed)
    del contract["backend"]["format_retry"]
    with pytest.raises(ValueError, match="differs"):
        labeler.require_retry_policy_resume(contract, "luna", "low", policy)


def test_build_codex_command_freezes_low_effort_read_only_run(tmp_path):
    command = build_codex_command("codex", "gpt-5.6-luna", "low", tmp_path)

    assert command[-1] == "-"
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="low"' in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--strict-config" in command
    assert all(f"{key}={json.dumps(value)}" in command for key, value in CODEX_CONTEXT_CONFIG.items())
    assert command.count("--disable") == len(CODEX_DISABLED_FEATURES)
    assert all(feature in command for feature in CODEX_DISABLED_FEATURES)


def test_build_codex_command_persists_document_session(tmp_path):
    command = build_codex_command(
        "codex",
        "gpt-5.6-luna",
        "low",
        tmp_path,
        persistent=True,
    )

    assert "--ephemeral" not in command
    assert command[command.index("--sandbox") + 1] == "read-only"


def test_build_codex_resume_command_names_exact_session():
    command = build_codex_resume_command("codex", "gpt-5.6-luna", "low", "session-1")

    assert command[:3] == ["codex", "exec", "resume"]
    assert command[-2:] == ["session-1", "-"]
    assert "--ignore-user-config" in command
    assert "project_doc_max_bytes=0" in command
    assert "skills.include_instructions=false" in command


def test_document_retry_tree_uses_one_sibling_then_one_fresh_session():
    assert codex_document_attempts("parent", fork_retries=1, fresh_retries=1) == [
        ("continuation", "parent"),
        ("same_context_fork", "parent"),
        ("fresh_recovery", None),
    ]
    assert codex_document_attempts(None, fork_retries=1, fresh_retries=1) == [
        ("fresh_start", None),
        ("fresh_recovery", None),
    ]


def test_protocol_root_allows_explicit_no_retry(monkeypatch):
    observed = {}

    async def fake_run(args):
        observed["args"] = args

    monkeypatch.setattr(labeler, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_api_label.py",
            "--backend",
            "codex",
            "--model",
            "gpt-5.6-luna",
            "--gold",
            "input.jsonl",
            "--dedup-receipt",
            "dedup.json",
            "--out",
            "output.jsonl",
            "--codex-protocol-root",
            "--session-field",
            "document_id",
            "--prompt-contract-out",
            "prompt.json",
            "--session-event-out",
            "events.jsonl",
            "--session-fork-retries",
            "0",
            "--session-recovery-retries",
            "0",
        ],
    )

    labeler.main()

    assert observed["args"].session_recovery_retries == 0


def test_codex_credit_failure_preserves_stdout_and_stops_transport_retries():
    failure = codex_exit_failure(
        '{"type":"error","message":"You have no credits remaining."}',
        "",
        index=2,
        returncode=1,
    )

    assert failure.retryable is False
    assert "no credits remaining" in str(failure)


def test_output_warning_counts_visible_completion_not_hidden_reasoning():
    response = {
        "content": [{"type": "text", "text": "[]"}],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 1500,
            "reasoning_output_tokens": 1401,
        },
        "codex_events": [],
        "session_attempt": {},
    }

    assert not labeler.record_codex_attempt_health(
        response,
        {"text": "no entities"},
        tagset={"person_name"},
        fmt="json-groups",
        output_token_warning=1400,
        allow_overlapping_spans=False,
    )
    assert response["session_attempt"]["health"]["visible_output_tokens"] == 99
    assert not response["session_attempt"]["health"]["output_warning_reached"]

    response["usage"]["reasoning_output_tokens"] = 50
    response["session_attempt"] = {}
    assert labeler.record_codex_attempt_health(
        response,
        {"text": "no entities"},
        tagset={"person_name"},
        fmt="json-groups",
        output_token_warning=1400,
        allow_overlapping_spans=False,
    )
    assert response["session_attempt"]["health"]["visible_output_tokens"] == 1450
    assert response["session_attempt"]["health"]["reasons"] == ["output_token_warning"]


def test_document_retry_executes_sibling_from_parent_then_fresh(monkeypatch):
    commands = []
    texts = [
        '[{"t":"Alex","types":[["person_name"],["person_name"]]}]',
        '[{"t":"Alex","types":[["person_name"],["person_name"]]}]',
        '[{"t":"Alex","types":[["person_name"]]}]',
    ]

    async def fake_request(_semaphore, *, command, **_kwargs):
        commands.append(command)
        number = len(commands)
        return (
            {
                "content": [{"type": "text", "text": texts[number - 1]}],
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "cache_write_input_tokens": 3,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 4,
                },
                "codex_events": [{"type": "thread.started", "thread_id": f"attempt-{number}"}],
            },
            0.1,
        )

    monkeypatch.setattr(labeler, "request_codex_one", fake_request)
    results, final_session = asyncio.run(
        labeler.request_codex_document(
            [(0, "prompt", {"text": "Alex"})],
            asyncio.Semaphore(1),
            initial_command=["fresh"],
            command_name="codex",
            model="gpt-5.6-luna",
            effort="low",
            timeout=1,
            retries=0,
            fork_retries=1,
            recovery_retries=1,
            tagset={"person_name"},
            initial_session_id="parent",
            env={},
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=20,
            fmt="json-groups",
        )
    )

    assert commands[0][-2:] == ["parent", "-"]
    assert commands[1][-2:] == ["parent", "-"]
    assert commands[2] == ["fresh"]
    response, _latency, metadata = results[0]
    assert [
        item["session_attempt"]["kind"] for item in [*response["session_recovery_responses"], response]
    ] == ["continuation", "same_context_fork", "fresh_recovery"]
    assert metadata["prior_context_thread_id"] == "parent"
    assert metadata["history_reset"] is True
    assert metadata["continuable"] is True
    assert response["aggregate_usage"] == {
        "input_tokens": 300,
        "cached_input_tokens": 60,
        "cache_write_input_tokens": 9,
        "output_tokens": 30,
        "reasoning_output_tokens": 12,
    }
    assert final_session == "attempt-3"


def test_json_seq_session_health_rejects_parse_defects_duplicates_and_overlaps():
    tags = {"locality", "person_name"}

    assert not codex_output_unhealthy(
        '[{"i":1,"t":"Alex","type":"person_name"}]',
        "Alex in Rome",
        tags,
        "json-seq",
    )
    assert codex_output_unhealthy(
        '[{"i":1,"t":"Alex","type":"person_name"},{"i":2,"t":"Alex","type":"person_name"}]',
        "Alex in Rome",
        tags,
        "json-seq",
    )
    assert codex_output_unhealthy(
        '[{"i":1,"t":"Alex in Rome","type":"person_name"},{"i":2,"t":"Rome","type":"locality"}]',
        "Alex in Rome",
        tags,
        "json-seq",
    )


def test_json_seq_session_health_can_allow_intentional_predicate_overlap():
    raw = '[{"i":1,"t":"Alex","type":"person_reference"},{"i":2,"t":"Alex","type":"legal_professional"}]'
    health = codex_output_health(
        raw,
        "Alex",
        {"person_reference", "legal_professional"},
        "json-seq",
        allow_overlapping_spans=True,
    )

    assert health["unhealthy"] is False
    assert health["overlapping_spans"] is True
    assert health["overlapping_spans_allowed"] is True


def test_json_offsets_session_health_accepts_exact_intentional_overlap():
    raw = json.dumps(
        [
            {"start": 0, "end": 11, "t": "Alice Smith", "type": "person_reference"},
            {"start": 0, "end": 11, "t": "Alice Smith", "type": "patient"},
        ]
    )

    health = codex_output_health(
        raw,
        "Alice Smith",
        {"person_reference", "patient"},
        "json-offsets",
        allow_overlapping_spans=True,
    )

    assert health["unhealthy"] is False
    assert health["overlapping_spans"] is True
    assert health["overlapping_spans_allowed"] is True


def test_candidate_subclass_health_rejects_a_missing_controller_decision():
    text = "Alice said she agreed"
    ledger = [
        {
            "candidate_id": "C0000",
            "start": 0,
            "end": 5,
            "surface": "Alice",
            "base_type": "person_name",
            "allowed_primary_types": ["person_name", "person_reference"],
            "bernoulli_channels": [
                "care_provider",
                "patient",
                "family_member",
                "witness_or_bystander",
                "investigator_or_law_enforcement",
                "legal_professional",
            ],
            "sources": ["base_person_carrier"],
        },
        {
            "candidate_id": "C0001",
            "start": 11,
            "end": 14,
            "surface": "she",
            "base_type": None,
            "allowed_primary_types": ["O", "person_reference"],
            "bernoulli_channels": [
                "care_provider",
                "patient",
                "family_member",
                "witness_or_bystander",
                "investigator_or_law_enforcement",
                "legal_professional",
            ],
            "sources": ["person_pronoun"],
        },
    ]
    health = codex_output_health(
        json.dumps(
            {
                "candidate_decisions": [
                    {
                        "candidate_id": "C0000",
                        "primary_type": "person_name",
                        "bernoulli_types": [],
                    }
                ],
                "added_candidates": [],
                "subclass_spans": [],
            }
        ),
        text,
        {"person_reference", "organization_reference"},
        "json-candidate-subclasses",
        base_spans=[{"start": 0, "end": 5, "type": "person_name"}],
        candidate_ledger=ledger,
        subclass_spec=load_subclass_spec(Path("scripts/pii_subclass_families_v3.json")),
    )

    assert health["unhealthy"] is True
    assert health["reasons"] == ["missing_or_duplicate_candidate_decision"]


def test_grouped_session_health_names_exact_occurrence_count_mismatch():
    health = codex_output_health(
        '[{"t":"Alex","types":[["person_name"],["person_name"]]}]',
        "Alex",
        {"person_name"},
        "json-groups",
    )

    assert health["unhealthy"] is True
    assert health["reasons"] == ["occurrence_count_mismatch"]
    assert health["parse_stats"]["occurrence_count_mismatch"] == 1


def test_grouped_lexical_session_health_ignores_pronoun_substrings():
    health = codex_output_health(
        '[{"t":"he","types":[["person_reference"]]}]',
        "the applicant said he agreed",
        {"person_reference"},
        "json-groups-lexical",
    )

    assert health["unhealthy"] is False


def test_codex_compaction_events_are_explicit_health_signals():
    assert codex_compaction_event_types(
        {
            "codex_events": [
                {"type": "thread.started"},
                {"type": "context.compacted"},
                {"type": "context.compacted"},
            ]
        }
    ) == ["context.compacted"]


def test_document_retry_does_not_continue_from_compacted_response(monkeypatch):
    commands = []

    async def fake_request(_semaphore, *, command, **_kwargs):
        commands.append(command)
        number = len(commands)
        events = [{"type": "thread.started", "thread_id": f"attempt-{number}"}]
        if number == 1:
            events.append({"type": "context.compacted"})
        return (
            {
                "content": [{"type": "text", "text": "[]"}],
                "usage": {"input_tokens": 100, "output_tokens": 1},
                "codex_events": events,
            },
            0.1,
        )

    monkeypatch.setattr(labeler, "request_codex_one", fake_request)
    results, final_session = asyncio.run(
        labeler.request_codex_document(
            [(0, "prompt", {"text": "no entities"})],
            asyncio.Semaphore(1),
            initial_command=["fresh"],
            command_name="codex",
            model="gpt-5.6-luna",
            effort="low",
            timeout=1,
            retries=0,
            fork_retries=1,
            recovery_retries=1,
            tagset={"person_name"},
            initial_session_id="parent",
            env={},
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=20,
            fmt="json-groups",
        )
    )

    response, _latency, metadata = results[0]
    previous = response["session_recovery_responses"][0]
    assert previous["session_attempt"]["health"]["reasons"] == ["context_compaction_event"]
    assert metadata["attempt_kinds"] == ["continuation", "same_context_fork"]
    assert metadata["continuable"] is True
    assert final_session == "attempt-2"


def test_protocol_root_replays_the_whole_paragraph_after_one_bad_turn(monkeypatch, tmp_path):
    calls = []
    unhealthy_call = 2

    async def fake_forked_one(
        _server,
        _semaphore,
        *,
        source_thread_id,
        index,
        attempt_kind,
        **_kwargs,
    ):
        number = len(calls) + 1
        calls.append((index, source_thread_id, attempt_kind))
        text = "not-json" if number == unhealthy_call else "[]"
        return (
            {
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 100 + number, "output_tokens": 1},
                "codex_events": [],
                "session_attempt": {
                    "kind": attempt_kind,
                    "parent_thread_id": source_thread_id,
                    "forked_thread_id": f"thread-{number}",
                },
            },
            0.1,
            f"thread-{number}",
        )

    monkeypatch.setattr(labeler, "request_codex_forked_one", fake_forked_one)
    entries = []
    for index in range(3):
        doc = {
            "id": f"doc-{index}",
            "text": "no entities",
            "_codex_session_plan": {
                "value": "doc",
                "part": 0,
                "turn_in_part": index + 1,
                "paragraph_value": "doc",
                "paragraph_in_part": 1,
                "turn_in_paragraph": index + 1,
                "segments_before_turn": index,
                "source_chars_before_turn": index * len("no entities"),
                "source_chars_through_turn": (index + 1) * len("no entities"),
            },
        }
        entries.append((index, f"prompt-{index}", doc))

    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            entries,
            object(),
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=labeler.Path("/tmp"),
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    assert [call[0] for call in calls] == [0, 1, 0, 1, 2]
    assert [call[1] for call in calls] == ["parent", "thread-1", "root", "thread-3", "thread-4"]
    assert [
        response["session_attempt"]["kind"]
        for response in [
            *results[0][0]["session_recovery_responses"],
            results[0][0],
        ]
    ] == ["continuation_fork", "protocol_root_paragraph_replay"]
    assert results[0][2]["paragraph_replayed"] is True
    assert results[0][2]["length_incident_eligible"] is False
    assert results[1][2]["length_incident_eligible"] is True
    assert results[2][2]["recovery_attempts"] == 0
    assert all(not result[2]["annotation_banned"] for result in results.values())
    assert final_session == "thread-5"
    event_rows = labeler.codex_attempt_event_rows(
        index=0,
        doc=entries[0][2],
        response=results[0][0],
        session_metadata=results[0][2],
        task_template_path=tmp_path / "prompt.txt",
        language="en",
        domain_field=None,
    )
    assert [row["input_tokens"] for row in event_rows] == [101, 103]


def test_protocol_root_zero_recovery_budget_bans_without_replay(monkeypatch):
    calls = []

    async def fake_forked_one(
        _server,
        _semaphore,
        *,
        source_thread_id,
        index,
        attempt_kind,
        **_kwargs,
    ):
        number = len(calls) + 1
        calls.append((index, source_thread_id, attempt_kind))
        return (
            {
                "content": [{"type": "text", "text": "not-json" if index == 1 else "[]"}],
                "usage": {"input_tokens": 100 + number, "output_tokens": 1},
                "codex_events": [],
                "session_attempt": {
                    "kind": attempt_kind,
                    "parent_thread_id": source_thread_id,
                    "forked_thread_id": f"thread-{number}",
                },
            },
            0.1,
            f"thread-{number}",
        )

    monkeypatch.setattr(labeler, "request_codex_forked_one", fake_forked_one)
    entries = [
        (
            index,
            f"prompt-{index}",
            {
                "id": f"doc-{index}",
                "text": "no entities",
                "_codex_session_plan": {
                    "value": "doc",
                    "part": 0,
                    "turn_in_part": index + 1,
                    "paragraph_value": "doc",
                    "paragraph_in_part": 1,
                    "turn_in_paragraph": index + 1,
                    "segments_before_turn": index,
                    "source_chars_before_turn": index * len("no entities"),
                    "source_chars_through_turn": (index + 1) * len("no entities"),
                },
            },
        )
        for index in range(3)
    ]

    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            entries,
            object(),
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=labeler.Path("/tmp"),
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            recovery_retries=0,
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    assert calls == [
        (0, "parent", "continuation_fork"),
        (1, "thread-1", "continuation_fork"),
        (2, "root", "protocol_root_fork"),
    ]
    assert results[0][2]["annotation_banned"] is False
    assert results[1][2]["annotation_banned"] is True
    assert results[1][2]["recovery_attempts"] == 0
    assert results[1][2]["paragraph_replayed"] is False
    assert results[2][2]["annotation_banned"] is False
    assert final_session == "thread-3"


def test_protocol_root_replays_a_timed_out_turn_once(tmp_path):
    class FakeServer:
        def __init__(self):
            self.fork_sources = []

        async def fork_thread(self, source_thread_id, **_kwargs):
            self.fork_sources.append(source_thread_id)
            return {"id": f"thread-{len(self.fork_sources)}"}

        async def run_turn(self, _thread_id, _prompt, **_kwargs):
            if len(self.fork_sources) == 1:
                raise asyncio.TimeoutError
            return {
                "content": [{"type": "text", "text": "[]"}],
                "usage": {"input_tokens": 100, "output_tokens": 1},
                "codex_events": [],
            }

    server = FakeServer()
    doc = {
        "id": "doc-0",
        "text": "no entities",
        "_codex_session_plan": {"turn_in_paragraph": 1},
    }
    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            [(0, "prompt", doc)],
            server,
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=tmp_path,
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    response, _latency, metadata = results[0]
    failed_attempt = response["session_recovery_responses"][0]
    assert server.fork_sources == ["parent", "root"]
    assert failed_attempt["attempt_failure"]["kind"] == "turn_timeout"
    assert failed_attempt["session_attempt"]["health"]["reasons"] == ["turn_timeout"]
    assert metadata["attempt_kinds"] == ["continuation_fork", "protocol_root_paragraph_replay"]
    assert metadata["paragraph_replayed"] is True
    assert metadata["annotation_banned"] is False
    assert final_session == "thread-2"


def test_protocol_root_gives_later_replay_failure_its_own_recovery(monkeypatch):
    calls = []

    async def fake_forked_one(
        _server,
        _semaphore,
        *,
        source_thread_id,
        index,
        attempt_kind,
        **_kwargs,
    ):
        number = len(calls) + 1
        calls.append((index, source_thread_id, attempt_kind))
        unhealthy = number in {1, 3}
        return (
            {
                "content": [{"type": "text", "text": "not-json" if unhealthy else "[]"}],
                "usage": {"input_tokens": 100 + number, "output_tokens": 1},
                "codex_events": [],
                "session_attempt": {
                    "kind": attempt_kind,
                    "parent_thread_id": source_thread_id,
                    "forked_thread_id": f"thread-{number}",
                },
            },
            0.1,
            f"thread-{number}",
        )

    monkeypatch.setattr(labeler, "request_codex_forked_one", fake_forked_one)
    entries = [
        (
            index,
            f"prompt-{index}",
            {
                "id": f"doc-{index}",
                "text": "no entities",
                "_codex_session_plan": {"turn_in_paragraph": index + 1},
            },
        )
        for index in range(2)
    ]

    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            entries,
            object(),
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=labeler.Path("/tmp"),
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    assert calls == [
        (0, "parent", "continuation_fork"),
        (0, "root", "protocol_root_paragraph_replay"),
        (1, "thread-2", "paragraph_replay_continuation_fork"),
        (1, "root", "protocol_root_row_recovery"),
    ]
    assert results[1][2]["attempt_kinds"] == [
        "paragraph_replay_continuation_fork",
        "protocol_root_row_recovery",
    ]
    assert results[1][2]["recovery_attempts"] == 1
    assert results[1][2]["annotation_banned"] is False
    assert results[1][2]["length_incident_eligible"] is True
    assert final_session == "thread-4"


def test_protocol_root_bans_a_turn_that_times_out_again_on_replay(tmp_path):
    class FakeServer:
        def __init__(self):
            self.fork_sources = []

        async def fork_thread(self, source_thread_id, **_kwargs):
            self.fork_sources.append(source_thread_id)
            return {"id": f"thread-{len(self.fork_sources)}"}

        async def run_turn(self, _thread_id, _prompt, **_kwargs):
            raise asyncio.TimeoutError

    server = FakeServer()
    doc = {
        "id": "doc-0",
        "text": "no entities",
        "_codex_session_plan": {"turn_in_paragraph": 1},
    }
    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            [(0, "prompt", doc)],
            server,
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=tmp_path,
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    response, _latency, metadata = results[0]
    assert server.fork_sources == ["parent", "root"]
    assert response["attempt_failure"]["kind"] == "turn_timeout"
    assert metadata["health_reasons"] == ["turn_timeout"]
    assert metadata["annotation_banned"] is True
    assert final_session is None


def test_protocol_root_replays_a_thread_store_connection_timeout_once(tmp_path):
    class FakeServer:
        def __init__(self):
            self.fork_sources = []

        async def fork_thread(self, source_thread_id, **_kwargs):
            self.fork_sources.append(source_thread_id)
            if len(self.fork_sources) == 1:
                raise CodexAppServerError(
                    "app-server thread/fork failed: thread-store internal error: "
                    "pool timed out while waiting for an open connection"
                )
            return {"id": "thread-2"}

        async def run_turn(self, _thread_id, _prompt, **_kwargs):
            return {
                "content": [{"type": "text", "text": "[]"}],
                "usage": {"input_tokens": 100, "output_tokens": 1},
                "codex_events": [],
            }

    server = FakeServer()
    doc = {
        "id": "doc-0",
        "text": "no entities",
        "_codex_session_plan": {"turn_in_paragraph": 1},
    }
    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            [(0, "prompt", doc)],
            server,
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=tmp_path,
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    response, _latency, metadata = results[0]
    failed_attempt = response["session_recovery_responses"][0]
    assert server.fork_sources == ["parent", "root"]
    assert failed_attempt["attempt_failure"]["kind"] == "thread_store_connection_timeout"
    assert failed_attempt["session_attempt"]["forked_thread_id"] is None
    assert failed_attempt["session_attempt"]["health"]["reasons"] == ["thread_store_connection_timeout"]
    assert metadata["paragraph_replayed"] is True
    assert metadata["annotation_banned"] is False
    assert final_session == "thread-2"


def test_protocol_root_bans_a_repeated_thread_store_connection_timeout(tmp_path):
    class FakeServer:
        def __init__(self):
            self.fork_sources = []

        async def fork_thread(self, source_thread_id, **_kwargs):
            self.fork_sources.append(source_thread_id)
            raise CodexAppServerError(
                "app-server thread/fork failed: thread-store internal error: "
                "pool timed out while waiting for an open connection"
            )

    server = FakeServer()
    doc = {
        "id": "doc-0",
        "text": "no entities",
        "_codex_session_plan": {"turn_in_paragraph": 1},
    }
    results, final_session = asyncio.run(
        labeler.request_codex_forked_document(
            [(0, "prompt", doc)],
            server,
            asyncio.Semaphore(1),
            protocol_root_thread_id="root",
            command_workdir=tmp_path,
            model="gpt-5.6-luna",
            base_instructions="common protocol",
            effort="low",
            timeout=1,
            tagset={"person_name"},
            initial_session_id="parent",
            bootstrap_input_limit=0,
            context_input_limit=0,
            output_token_warning=0,
            fmt="json-groups",
        )
    )

    response, _latency, metadata = results[0]
    assert server.fork_sources == ["parent", "root"]
    assert response["attempt_failure"]["kind"] == "thread_store_connection_timeout"
    assert metadata["health_reasons"] == ["thread_store_connection_timeout"]
    assert metadata["annotation_banned"] is True
    assert final_session is None


def test_unknown_app_server_fork_failure_remains_fatal(tmp_path):
    class FakeServer:
        async def fork_thread(self, _source_thread_id, **_kwargs):
            raise CodexAppServerError("app-server thread/fork failed: unknown defect")

    with pytest.raises(RequestFailure, match="unknown defect"):
        asyncio.run(
            labeler.request_codex_forked_one(
                FakeServer(),
                asyncio.Semaphore(1),
                source_thread_id="parent",
                prompt="prompt",
                command_workdir=tmp_path,
                model="gpt-5.6-luna",
                base_instructions="common protocol",
                effort="low",
                timeout=1,
                index=0,
                attempt_kind="continuation_fork",
            )
        )


def test_document_session_parts_prefer_paragraph_edge_before_hard_limit():
    docs = [
        {"id": "a-1", "document_id": "a", "text": "first"},
        {"id": "a-2", "document_id": "a", "text": "paragraph end\n\n"},
        {"id": "a-3", "document_id": "a", "text": "third"},
        {"id": "a-4", "document_id": "a", "text": "fourth"},
    ]

    plans = plan_document_session_parts(
        docs,
        "document_id",
        max_segments=3,
        max_source_chars=100,
    )

    assert [plan["part"] for plan in plans] == [0, 0, 1, 1]
    assert plans[2]["split_before"] == "paragraph_boundary"
    assert [plan["turn_in_part"] for plan in plans] == [1, 2, 1, 2]
    assert [plan["paragraph_in_part"] for plan in plans] == [1, 1, 1, 1]


def test_protocol_root_commit_ranges_do_not_split_a_paragraph():
    docs = [
        {"id": "a-1", "document_id": "a", "paragraph_id": "p1", "text": "one"},
        {"id": "a-2", "document_id": "a", "paragraph_id": "p1", "text": "two"},
        {"id": "a-3", "document_id": "a", "paragraph_id": "p1", "text": "three"},
        {"id": "a-4", "document_id": "a", "paragraph_id": "p2", "text": "four"},
    ]
    for doc, plan in zip(
        docs,
        plan_document_session_parts(
            docs,
            "document_id",
            max_segments=4,
            max_source_chars=100,
            paragraph_field="paragraph_id",
        ),
        strict=True,
    ):
        doc["_codex_session_plan"] = plan

    assert paragraph_aligned_commit_ranges(docs, 0, 2) == [(0, 3), (3, 4)]
    with pytest.raises(ValueError, match="ends inside a paragraph"):
        paragraph_aligned_commit_ranges(docs, 1, 2)
    assert paragraph_aligned_commit_ranges(
        docs,
        1,
        2,
        allow_transport_failure_prefix=True,
    ) == [(1, 3), (3, 4)]


def test_transport_failure_resume_requires_exact_recovered_prefix(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(
        raw,
        [
            {
                "id": "a",
                "session": {
                    "continuable": True,
                    "recovered_after_transport_failure": True,
                    "thread_id": "thread-a",
                },
            },
            {
                "id": "b",
                "session": {
                    "continuable": True,
                    "recovered_after_transport_failure": True,
                    "thread_id": "thread-b",
                },
            },
        ],
    )

    require_transport_failure_resume(raw, 2)

    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]["session"].pop("recovered_after_transport_failure")
    write_jsonl(raw, rows)
    with pytest.raises(ValueError, match="verified recovered row"):
        require_transport_failure_resume(raw, 2)


def test_document_session_parts_force_segment_edge_and_reject_oversize_row():
    docs = [
        {"id": "a-1", "document_id": "a", "text": "1234"},
        {"id": "a-2", "document_id": "a", "text": "5678"},
        {"id": "a-3", "document_id": "a", "text": "90"},
    ]

    plans = plan_document_session_parts(
        docs,
        "document_id",
        max_segments=2,
        max_source_chars=8,
    )
    assert [plan["part"] for plan in plans] == [0, 0, 1]
    assert plans[2]["split_before"] == "hard_segment_boundary"

    with pytest.raises(ValueError, match="split that row first"):
        plan_document_session_parts(
            docs,
            "document_id",
            max_segments=2,
            max_source_chars=3,
        )


def test_codex_bootstrap_input_limit_rejects_agent_context_overhead():
    with pytest.raises(RequestFailure, match="agent context suppression may have failed"):
        enforce_codex_input_limit(
            {"usage": {"input_tokens": 27_373, "output_tokens": 899}},
            index=0,
            maximum=20_000,
        )


def test_codex_bootstrap_input_limit_allows_disabled_guard():
    enforce_codex_input_limit(
        {"usage": {"input_tokens": 27_373, "output_tokens": 899}},
        index=0,
        maximum=0,
    )


@pytest.mark.parametrize(
    "context_name",
    ["AGENTS.md", "AGENTS.override.md", "memories", "plugins", "skills"],
)
def test_codex_home_rejects_agent_context(tmp_path, context_name):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    context_path = codex_home / context_name
    if context_path.suffix:
        context_path.write_text("context")
    else:
        context_path.mkdir()

    with pytest.raises(SystemExit, match="not context-isolated"):
        require_isolated_codex_home(codex_home)


def test_parse_codex_events_extracts_final_message_and_usage():
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "session"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "[]"}}),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 20, "output_tokens": 2},
                }
            ),
        ]
    )

    response = parse_codex_events(stdout, "warning", 3)

    assert response_text(response) == "[]"
    assert response["usage"] == {"input_tokens": 20, "output_tokens": 2}
    assert response["codex_stderr"] == "warning"
    assert codex_session_id(response) == "session"


def test_parse_codex_events_rejects_incomplete_turn():
    stdout = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "[]"}})

    with pytest.raises(RequestFailure, match="no completed assistant message"):
        parse_codex_events(stdout, "", 4)


def test_resume_pair_requires_equal_exact_prefixes(tmp_path):
    pred = tmp_path / "pred.jsonl"
    raw = tmp_path / "raw.jsonl"
    docs = [{"id": "a"}, {"id": "b"}]
    write_jsonl(pred, [{"id": "a", "preds": []}])
    write_jsonl(raw, [{"id": "a", "response": {}}])

    assert require_resume_pair(pred, raw, docs) == 1
    policy = labeler.format_retry_policy("terra", "low", 1, "anthropic")
    with pytest.raises(ValueError, match="retry policy"):
        require_resume_pair(pred, raw, docs, retry_policy=policy)
    write_jsonl(raw, [{"id": "a", "response": {"format_retry_policy": policy}}])
    assert require_resume_pair(pred, raw, docs, retry_policy=policy) == 1
    with pytest.raises(ValueError, match="retry policy"):
        require_resume_pair(pred, raw, docs)


def test_resume_pair_rejects_one_missing_file(tmp_path):
    pred = tmp_path / "pred.jsonl"
    write_jsonl(pred, [{"id": "a", "preds": []}])

    with pytest.raises(ValueError, match="both exist"):
        require_resume_pair(pred, tmp_path / "raw.jsonl", [{"id": "a"}])


def test_fresh_protocol_resume_requires_new_contract_and_event_paths(tmp_path):
    contract = tmp_path / "tranche.prompt-contract.json"
    events = tmp_path / "tranche.session-events.jsonl"

    require_fresh_protocol_resume_paths(contract, events)
    contract.write_text("{}\n")

    with pytest.raises(ValueError, match="requires new prompt-contract and session-event paths"):
        require_fresh_protocol_resume_paths(contract, events)


def test_resumed_codex_sessions_uses_latest_segment_state_per_document(tmp_path):
    raw = tmp_path / "raw.jsonl"
    docs = [
        {"id": "a-1", "document_id": "a"},
        {"id": "b-1", "document_id": "b"},
        {"id": "a-2", "document_id": "a"},
    ]
    write_jsonl(
        raw,
        [
            {
                "id": "a-1",
                "session": {
                    "field": "document_id",
                    "value": "a",
                    "thread_id": "a-old",
                    "continuable": True,
                },
            },
            {
                "id": "b-1",
                "session": {
                    "field": "document_id",
                    "value": "b",
                    "thread_id": "b-current",
                    "continuable": True,
                },
            },
            {
                "id": "a-2",
                "session": {
                    "field": "document_id",
                    "value": "a",
                    "thread_id": "a-rejected",
                    "continuable": False,
                },
            },
        ],
    )

    assert resumed_codex_sessions(raw, docs, 3, "document_id") == {
        "a": None,
        "b": "b-current",
    }


@pytest.mark.parametrize("retry_model", [None, "gpt-5.6-terra"])
def test_prompt_contract_records_custom_renderer_inputs(tmp_path, retry_model):
    task_path = tmp_path / "task.txt"
    examples_path = tmp_path / "examples.json"
    contract_path = tmp_path / "contract.json"
    task_path.write_text("{tags}\n{format_rules}\n{example}\n{text}\n")
    examples = {"en": {"text": "Alex", "labels": []}}
    examples_path.write_text(json.dumps(examples))

    write_prompt_contract(
        contract_path,
        task_path,
        examples_path,
        task_path.read_text(),
        examples,
        [{"id": "doc-1", "lang": "en", "text": "Alex"}],
        ["person_name"],
        "json-seq",
        "en",
        "gpt-5.6-luna",
        url="http://localhost/v1/messages",
        effort="low",
        input_render="json-string",
        proposal_render="list",
        format_retries=1 if retry_model else 0,
        format_retry_model=retry_model,
        format_retry_effort="low" if retry_model else None,
    )

    contract = json.loads(contract_path.read_text())
    assert contract["input_render"] == "json-string"
    assert contract["proposal_render"] == "list"
    assert contract["task_template_path"] == str(task_path.resolve())
    assert contract["examples_path"] == str(examples_path.resolve())
    assert contract["backend"]["effort"] == "low"
    assert contract["backend"].get("format_retry") == (
        {"model": retry_model, "effort": "low", "max_attempts": 1, "mode": "fresh_annotation"}
        if retry_model
        else None
    )


def test_prompt_contract_records_and_renders_proposal_field(tmp_path):
    task_path = tmp_path / "task.txt"
    examples_path = tmp_path / "examples.json"
    contract_path = tmp_path / "contract.json"
    task_path.write_text("{tags}\n{format_rules}\n{example}\n{text}\n")
    examples = {"en": {"text": "Alex", "labels": []}}
    examples_path.write_text(json.dumps(examples))
    docs = [
        {
            "id": "doc-1",
            "lang": "en",
            "text": "Alex",
            "current_preds": [{"start": 0, "end": 4, "label": "person_name"}],
        }
    ]

    write_prompt_contract(
        contract_path,
        task_path,
        examples_path,
        task_path.read_text(),
        examples,
        docs,
        ["person_name"],
        "json-seq",
        "en",
        "gpt-5.6-luna",
        url="http://localhost/v1/messages",
        effort="low",
        input_render="json-string",
        proposal_render="list",
        proposal_field="current_preds",
    )

    contract = json.loads(contract_path.read_text())
    assert contract["proposal_field"] == "current_preds"
    assert contract["samples"][0]["proposal_count"] == 1
    assert '"surface": "Alex"' in contract["samples"][0]["prompt"]


def test_prompt_contract_records_and_renders_review_guidance_field(tmp_path):
    task_path = tmp_path / "task.txt"
    examples_path = tmp_path / "examples.json"
    contract_path = tmp_path / "contract.json"
    task_path.write_text("{tags}\n{format_rules}\n{example}\nGuidance: {guidance}\n{text}\n")
    examples = {"en": {"text": "Alex", "labels": []}}
    examples_path.write_text(json.dumps(examples))
    docs = [
        {
            "id": "doc-1",
            "lang": "en",
            "text": "The Court decided.",
            "repair_guidance": "Missing the Court organization.",
        }
    ]

    write_prompt_contract(
        contract_path,
        task_path,
        examples_path,
        task_path.read_text(),
        examples,
        docs,
        ["organization"],
        "json-seq",
        "en",
        "gpt-5.6-luna",
        url="http://localhost/v1/messages",
        effort="low",
        guidance_field="repair_guidance",
    )

    contract = json.loads(contract_path.read_text())
    assert contract["guidance_field"] == "repair_guidance"
    assert 'Guidance: "Missing the Court organization."' in contract["samples"][0]["prompt"]


def test_prompt_contract_records_codex_context_isolation(tmp_path):
    task_path = tmp_path / "task.txt"
    examples_path = tmp_path / "examples.json"
    contract_path = tmp_path / "contract.json"
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    task_path.write_text("{tags}\n{format_rules}\n{example}\n{text}\n")
    examples = {"en": {"text": "Alex", "labels": []}}
    examples_path.write_text(json.dumps(examples))

    write_prompt_contract(
        contract_path,
        task_path,
        examples_path,
        task_path.read_text(),
        examples,
        [{"id": "doc-1", "lang": "en", "text": "Alex"}],
        ["person_name"],
        "json-groups",
        "en",
        "gpt-5.6-luna",
        url="unused",
        effort="low",
        backend="codex",
        codex_workdir=tmp_path,
        codex_home=codex_home,
    )

    backend = json.loads(contract_path.read_text())["backend"]
    assert backend["codex_home"] == str(codex_home.resolve())
    assert backend["strict_config"] is True
    assert backend["context_config"] == CODEX_CONTEXT_CONFIG
    assert backend["disabled_features"] == list(CODEX_DISABLED_FEATURES)


def test_prompt_contract_defines_output_warning_as_visible_tokens(tmp_path):
    task_path = tmp_path / "task.txt"
    examples_path = tmp_path / "examples.json"
    contract_path = tmp_path / "contract.json"
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    task_path.write_text("{tags}\n{format_rules}\n{example}\n{text}\n")
    examples = {"en": {"text": "Alex", "labels": []}}
    examples_path.write_text(json.dumps(examples))
    docs = [{"id": "doc-1", "lang": "en", "text": "Alex", "document_id": "doc"}]
    docs[0]["_codex_session_plan"] = plan_document_session_parts(
        docs,
        "document_id",
        max_segments=1,
        max_source_chars=10,
    )[0]

    write_prompt_contract(
        contract_path,
        task_path,
        examples_path,
        task_path.read_text(),
        examples,
        docs,
        ["person_name"],
        "json-groups",
        "en",
        "gpt-5.6-luna",
        url="unused",
        effort="low",
        backend="codex",
        codex_workdir=tmp_path,
        codex_home=codex_home,
        session_field="document_id",
        session_max_segments=1,
        session_max_source_chars=10,
        session_output_token_warning=1400,
    )

    session = json.loads(contract_path.read_text())["backend"]["session"]
    assert session["output_token_warning"] == 1400
    assert session["output_token_warning_basis"] == (
        "visible completion tokens; hidden reasoning tokens excluded"
    )
