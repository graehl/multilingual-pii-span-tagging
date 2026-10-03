#!/usr/bin/env python3
"""Label PII through a frontier-model API or the Codex CLI.

This is the service-model companion to ``pii_llm_label.py``.  Both paths use
the same frozen prompt renderer, surface-alignment parsers, tag inventory, and
prediction JSONL contract; this path additionally preserves each raw response
and its token accounting.  Output is committed in ordered waves so a failed or
interrupted request leaves an exact resumable prefix.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import unicodedata
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import prompt as promptlib  # noqa: E402

if TYPE_CHECKING:
    from scripts.pii_codex_app_server import CodexAppServer, CodexAppServerError
from scripts.pii_dedup_gate import (  # noqa: E402
    require_annotation_dedup,
    require_o4_web_intake,
    require_training_reannotation,
)
from scripts.pii_dedup_gate import (
    require_evaluation_replay as require_evaluation_replay,
)
from scripts.pii_llm_label import (  # noqa: E402
    CORE_TAGS,
    FORMATS,
    PROMPT_DIR,
    SPAN_POLICIES,
    add_source_paragraph_guidance,
    build_prompt,
    build_prompt_contract,
    load_language_rules,
    load_tag_catalog,
    parse_labels,
    parse_labels_grouped,
    parse_labels_grouped_lexical,
    parse_labels_inline,
    parse_labels_offsets,
    parse_labels_redact,
    parse_labels_seq,
    require_language_specific_examples,
    validated_resume_count,
)
from scripts.pii_prompt_template import load_prompt_template  # noqa: E402
from scripts.pii_subclass import (  # noqa: E402
    SubclassSpec,
    load_subclass_spec,
    parse_candidate_subclass_annotation,
    parse_subclass_annotation,
    render_subclass_catalog,
)


class RequestFailure(RuntimeError):
    """A request failed with enough detail to decide whether to retry."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


CODEX_DISABLED_FEATURES = (
    "apps",
    "memories",
    "multi_agent",
    "plugins",
    "remote_plugin",
    "skill_search",
    "shell_tool",
    "unified_exec",
)

CODEX_CONTEXT_CONFIG = {
    "project_doc_max_bytes": 0,
    "memories.use_memories": False,
    "memories.generate_memories": False,
    "skills.include_instructions": False,
    "skills.bundled.enabled": False,
    "agents.enabled": False,
}

CODEX_SESSION_TURN_MARKER = "<!-- CODEX_SESSION_TURN -->"
EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")


def codex_bulk_context_args() -> list[str]:
    """Disable annotation-irrelevant user, project, skill, and agent context."""
    result = [
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
    ]
    for key, value in CODEX_CONTEXT_CONFIG.items():
        result.extend(("--config", f"{key}={json.dumps(value)}"))
    for feature in CODEX_DISABLED_FEATURES:
        result.extend(("--disable", feature))
    return result


def require_isolated_codex_home(codex_home: Path) -> None:
    """Reject homes that can supply annotation-irrelevant agent context."""
    if not codex_home.is_dir():
        raise SystemExit(f"--codex-home is not a directory: {codex_home}")
    forbidden_context = [
        *codex_home.glob("AGENTS*.md"),
        *(codex_home / name for name in ("memories", "plugins", "skills")),
    ]
    forbidden_context = sorted((path for path in forbidden_context if path.exists()), key=str)
    if forbidden_context:
        raise SystemExit(
            "--codex-home is not context-isolated; remove or choose a home without: "
            + ", ".join(str(path) for path in forbidden_context)
        )


def enforce_codex_input_limit(
    response: dict[str, Any],
    *,
    index: int,
    maximum: int,
) -> None:
    input_tokens, _output_tokens = usage_counts(response)
    if maximum and input_tokens > maximum:
        raise RequestFailure(
            f"row {index}: fresh Codex session used {input_tokens} input tokens, "
            f"exceeding --codex-bootstrap-input-limit {maximum}; bulk agent context "
            "suppression may have failed",
            retryable=False,
        )


def build_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    effort: str,
    backend: str = "anthropic",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if backend == "openai":
        payload["temperature"] = 0.0
        payload["reasoning_effort"] = effort
        # Served Qwen and Gemma chat templates read enable_thinking, not reasoning_effort.
        if model.lower().startswith(("qwen", "google/gemma")) and effort == "none":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif backend == "anthropic":
        payload["output_config"] = {"effort": effort}
    elif backend != "anthropic":
        raise ValueError(f"API payload is unsupported for backend: {backend}")
    return payload


def response_text(response: dict[str, Any]) -> str:
    blocks = response.get("content")
    if isinstance(blocks, list):
        texts = [
            block.get("text") for block in blocks if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "".join(part for part in texts if isinstance(part, str)).strip()
    else:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RequestFailure(
                "response has neither content blocks nor one chat-completion choice",
                retryable=False,
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text = content.strip() if isinstance(content, str) else ""
    if not text:
        raise RequestFailure("response has no non-empty text block", retryable=False)
    return promptlib.strip_leading_think_block(text)


def response_text_or_empty(response: dict[str, Any]) -> str:
    """Like ``response_text`` but an empty completion is an empty string.

    A provider turn that returns only thinking blocks or no text is a
    row-level format failure: the parser reports it as malformed output, the
    format-retry policy runs, and the row is banned if every attempt is empty.
    It must not abort a whole tracked pass.
    """
    try:
        return response_text(response)
    except RequestFailure as error:
        if "no non-empty text block" not in str(error):
            raise
        response["empty_completion"] = True
        return ""


async def request_one(
    session: Any,
    semaphore: asyncio.Semaphore,
    *,
    url: str,
    payload: dict[str, Any],
    retries: int,
    index: int,
) -> tuple[dict[str, Any], float]:
    import aiohttp

    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            async with semaphore:
                async with session.post(url, json=payload) as reply:
                    body = await reply.text()
                    if reply.status != 200:
                        raise RequestFailure(
                            f"row {index}: HTTP {reply.status}: {body[:1000]}",
                            retryable=reply.status == 429 or reply.status >= 500,
                        )
                    response = json.loads(body)
            # An empty completion is a row-level format failure handled by the
            # caller's format-retry policy, not a transport failure.
            response_text_or_empty(response)
            return response, time.perf_counter() - started
        except RequestFailure as error:
            if not error.retryable or attempt == retries:
                raise
            last_error: Exception = error
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries:
                raise RequestFailure(f"row {index}: {error}", retryable=True) from error
            last_error = error
        delay = min(30.0, 2.0**attempt)
        print(f"RETRY: row {index} attempt {attempt + 1}/{retries + 1}: {last_error}; sleep {delay:g}s")
        await asyncio.sleep(delay)
    raise AssertionError("retry loop must return or raise")


def build_codex_command(
    command: str,
    model: str,
    effort: str,
    workdir: Path,
    *,
    persistent: bool = False,
) -> list[str]:
    result = [
        command,
        "exec",
        *codex_bulk_context_args(),
        "--skip-git-repo-check",
        "--color",
        "never",
        "--sandbox",
        "read-only",
        "--cd",
        str(workdir),
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{effort}"',
        "--config",
        'approval_policy="never"',
        "--json",
        "-",
    ]
    if not persistent:
        result.insert(2, "--ephemeral")
    return result


def build_codex_resume_command(
    command: str,
    model: str,
    effort: str,
    session_id: str,
) -> list[str]:
    return [
        command,
        "exec",
        "resume",
        *codex_bulk_context_args(),
        "--skip-git-repo-check",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{effort}"',
        "--config",
        'approval_policy="never"',
        "--json",
        session_id,
        "-",
    ]


def build_codex_app_server_command(command: str) -> list[str]:
    result = [command, "app-server", "--strict-config", "--stdio"]
    for key, value in CODEX_CONTEXT_CONFIG.items():
        result.extend(("--config", f"{key}={json.dumps(value)}"))
    for feature in CODEX_DISABLED_FEATURES:
        result.extend(("--disable", feature))
    return result


def split_codex_protocol_prompt(prompt: str) -> tuple[str, str]:
    """Split one rendered prompt into immutable protocol and row-specific suffix."""
    if prompt.count(CODEX_SESSION_TURN_MARKER) != 1:
        raise ValueError(
            "protocol-root sessions require exactly one "
            f"{CODEX_SESSION_TURN_MARKER!r} marker in every rendered prompt"
        )
    prefix, suffix = prompt.split(CODEX_SESSION_TURN_MARKER)
    prefix = prefix.rstrip()
    suffix = suffix.lstrip()
    if not prefix or not suffix:
        raise ValueError("protocol-root prompt marker must separate nonempty prefix and suffix")
    return prefix, suffix


def parse_codex_events(stdout: str, stderr: str, index: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RequestFailure(
                f"row {index}: Codex stdout line {line_number} is not JSON: {line[:500]}",
                retryable=False,
            ) from error
        if not isinstance(event, dict):
            raise RequestFailure(
                f"row {index}: Codex stdout line {line_number} is not an object",
                retryable=False,
            )
        events.append(event)

    messages = [
        event["item"]["text"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "agent_message"
        and isinstance(event["item"].get("text"), str)
    ]
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if not messages or not completed:
        raise RequestFailure(
            f"row {index}: Codex emitted no completed assistant message and usage; stderr={stderr[:1000]}",
            retryable=False,
        )
    usage = completed[-1].get("usage")
    if not isinstance(usage, dict):
        raise RequestFailure(f"row {index}: Codex completion has no usage object", retryable=False)
    return {
        "content": [{"type": "text", "text": messages[-1]}],
        "usage": usage,
        "codex_events": events,
        "codex_stderr": stderr,
    }


def codex_session_id(response: dict[str, Any]) -> str:
    session_ids = [
        event.get("thread_id")
        for event in response.get("codex_events", [])
        if isinstance(event, dict)
        and event.get("type") == "thread.started"
        and isinstance(event.get("thread_id"), str)
    ]
    if not session_ids:
        raise RequestFailure("Codex response has no thread.started session id", retryable=False)
    if len(set(session_ids)) != 1:
        raise RequestFailure("Codex response contains conflicting session ids", retryable=False)
    return session_ids[0]


async def request_codex_one(
    semaphore: asyncio.Semaphore,
    *,
    command: list[str],
    prompt: str,
    timeout: float,
    retries: int,
    index: int,
    env: dict[str, str],
    input_token_limit: int = 0,
) -> tuple[dict[str, Any], float]:
    for attempt in range(retries + 1):
        started = time.perf_counter()
        process: asyncio.subprocess.Process | None = None
        try:
            async with semaphore:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=timeout,
                )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            if process.returncode != 0:
                raise codex_exit_failure(stdout, stderr, index=index, returncode=process.returncode)
            response = parse_codex_events(stdout, stderr, index)
            enforce_codex_input_limit(response, index=index, maximum=input_token_limit)
            return response, time.perf_counter() - started
        except asyncio.TimeoutError as error:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            last_error: Exception = RequestFailure(
                f"row {index}: Codex exceeded {timeout:g}s timeout",
                retryable=True,
            )
            if attempt == retries:
                raise last_error from error
        except RequestFailure as error:
            if not error.retryable or attempt == retries:
                raise
            last_error = error
        delay = min(30.0, 2.0**attempt)
        print(f"RETRY: row {index} attempt {attempt + 1}/{retries + 1}: {last_error}; sleep {delay:g}s")
        await asyncio.sleep(delay)
    raise AssertionError("retry loop must return or raise")


def codex_exit_failure(stdout: str, stderr: str, *, index: int, returncode: int) -> RequestFailure:
    """Preserve a Codex JSON-stream failure when stderr is empty."""
    combined = f"{stdout}\n{stderr}".strip()
    detail = stderr.strip() or stdout.strip() or "no diagnostic output"
    retryable = "no credits remaining" not in combined.lower()
    return RequestFailure(
        f"row {index}: Codex exited {returncode}: {detail[-2000:]}",
        retryable=retryable,
    )


def codex_document_attempts(
    parent_session_id: str | None,
    *,
    fork_retries: int,
    fresh_retries: int,
) -> list[tuple[str, str | None]]:
    """Name the bounded retry tree for one document segment."""
    if fork_retries < 0 or fresh_retries < 0:
        raise ValueError("document-session retry counts must be nonnegative")
    if parent_session_id is None:
        return [("fresh_start", None), *(("fresh_recovery", None),) * fresh_retries]
    return [
        ("continuation", parent_session_id),
        *(("same_context_fork", parent_session_id),) * fork_retries,
        *(("fresh_recovery", None),) * fresh_retries,
    ]


def annotation_base_spans(doc: dict[str, Any], guidance_field: str | None = None) -> list[Any]:
    """Return authoritative carriers directly or from the frozen guidance JSON."""
    base_spans = doc.get("base_spans")
    if isinstance(base_spans, list):
        return base_spans
    if guidance_field is not None:
        guidance = doc.get(guidance_field)
        if isinstance(guidance, str):
            try:
                guidance_payload = json.loads(guidance)
            except json.JSONDecodeError as error:
                raise ValueError(f"{guidance_field} is not JSON") from error
            base_spans = guidance_payload.get("base_spans") if isinstance(guidance_payload, dict) else None
            if isinstance(base_spans, list):
                return base_spans
    raise ValueError("json-subclasses requires authoritative base_spans on every input row")


def annotation_candidate_ledger(doc: dict[str, Any], guidance_field: str | None) -> list[Any]:
    """Return the controller-enumerated candidate ledger from frozen guidance JSON."""
    if guidance_field is None:
        raise ValueError("candidate-subclass annotation requires a guidance field")
    guidance = doc.get(guidance_field)
    if not isinstance(guidance, str):
        raise ValueError(f"{guidance_field} must be a JSON string")
    try:
        payload = json.loads(guidance)
    except json.JSONDecodeError as error:
        raise ValueError(f"{guidance_field} is not JSON") from error
    candidate_ledger = payload.get("candidate_ledger") if isinstance(payload, dict) else None
    if not isinstance(candidate_ledger, list):
        raise ValueError(f"{guidance_field} has no candidate_ledger list")
    return candidate_ledger


def codex_output_health(
    raw: str,
    text: str,
    tagset: set[str],
    fmt: str,
    *,
    allow_overlapping_spans: bool = False,
    base_spans: list[Any] | None = None,
    candidate_ledger: list[Any] | None = None,
    subclass_spec: SubclassSpec | None = None,
) -> dict[str, Any]:
    """Return auditable structural reasons for accepting or retrying one response."""
    if fmt in {"json-groups", "json-groups-lexical"}:
        grouped_parser = (
            parse_labels_grouped_lexical if fmt == "json-groups-lexical" else parse_labels_grouped
        )
        _predictions, stats = grouped_parser(raw, text, tagset)
        reasons = [
            name
            for name, count in stats.items()
            if count and name not in {"quarantined", "repaired_nfkc", "repaired_edit"}
        ]
        return {
            "unhealthy": bool(stats["quarantined"]),
            "reasons": reasons,
            "parse_stats": stats,
            "duplicate_typed_spans": False,
            "overlapping_spans": False,
        }
    elif fmt == "json-subclasses":
        if base_spans is None or subclass_spec is None:
            raise ValueError("json-subclasses health requires base spans and a subclass spec")
        predictions, subclass_spans, stats = parse_subclass_annotation(
            raw,
            text,
            base_spans,
            subclass_spec,
        )
        reasons = [name for name, count in stats.items() if count and name != "quarantined"]
        return {
            "unhealthy": bool(stats["quarantined"]),
            "reasons": reasons,
            "parse_stats": stats,
            "duplicate_typed_spans": False,
            "overlapping_spans": False,
            "primary_or_bernoulli_spans": len(predictions),
            "subclass_components": len(subclass_spans),
        }
    elif fmt == "json-candidate-subclasses":
        if base_spans is None or candidate_ledger is None or subclass_spec is None:
            raise ValueError(
                "json-candidate-subclasses health requires base spans, a candidate ledger, "
                "and a subclass spec"
            )
        predictions, subclass_spans, decisions, stats = parse_candidate_subclass_annotation(
            raw,
            text,
            base_spans,
            candidate_ledger,
            subclass_spec,
        )
        reasons = [name for name, count in stats.items() if count and name != "quarantined"]
        return {
            "unhealthy": bool(stats["quarantined"]),
            "reasons": reasons,
            "parse_stats": stats,
            "duplicate_typed_spans": False,
            "overlapping_spans": False,
            "primary_or_bernoulli_spans": len(predictions),
            "subclass_components": len(subclass_spans),
            "candidate_decisions": len(decisions),
        }
    elif fmt in {"json", "json-seq", "json-offsets"}:
        if fmt == "json":
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                items = None
            if not isinstance(items, list) or any(
                not isinstance(item, dict)
                or not isinstance(item.get("t"), str)
                or not item["t"]
                or not isinstance(item.get("type"), str)
                or type(item.get("n")) is not int
                or item["n"] < 1
                for item in items
            ):
                return {
                    "unhealthy": True,
                    "reasons": ["invalid_occurrence_json"],
                    "parse_stats": {"bad_json": 1},
                }
        sequence_parser = {
            "json": parse_labels,
            "json-seq": parse_labels_seq,
            "json-offsets": parse_labels_offsets,
        }[fmt]
        predictions, stats = sequence_parser(raw, text, tagset)
        typed_spans = sorted((item["start"], item["end"], item["label"]) for item in predictions)
        duplicate = len(typed_spans) != len(set(typed_spans))
        overlap = any(current[0] < previous[1] for previous, current in zip(typed_spans, typed_spans[1:]))
        reasons = [name for name, count in stats.items() if count]
        if duplicate:
            reasons.append("duplicate_typed_span")
        if overlap and not allow_overlapping_spans:
            reasons.append("overlapping_span")
        return {
            "unhealthy": bool(reasons),
            "reasons": reasons,
            "parse_stats": stats,
            "duplicate_typed_spans": duplicate,
            "overlapping_spans": overlap,
            "overlapping_spans_allowed": allow_overlapping_spans,
        }
    else:
        raise ValueError(f"document-session health is unsupported for format: {fmt}")


def codex_output_unhealthy(
    raw: str,
    text: str,
    tagset: set[str],
    fmt: str,
    *,
    allow_overlapping_spans: bool = False,
    base_spans: list[Any] | None = None,
    candidate_ledger: list[Any] | None = None,
    subclass_spec: SubclassSpec | None = None,
) -> bool:
    return bool(
        codex_output_health(
            raw,
            text,
            tagset,
            fmt,
            allow_overlapping_spans=allow_overlapping_spans,
            base_spans=base_spans,
            candidate_ledger=candidate_ledger,
            subclass_spec=subclass_spec,
        )["unhealthy"]
    )


async def request_api_annotation(
    session: Any,
    semaphore: asyncio.Semaphore,
    *,
    url: str,
    backend: str,
    model: str,
    prompt: str,
    max_tokens: int,
    effort: str,
    transport_retries: int,
    format_retries: int,
    index: int,
    doc: dict[str, Any],
    tagset: set[str],
    fmt: str,
    allow_overlapping_spans: bool = False,
    subclass_spec: SubclassSpec | None = None,
    format_retry_model: str | None = None,
    format_retry_effort: str | None = None,
    format_retry_prompt: str | None = None,
) -> tuple[dict[str, Any], float, None]:
    """Retry a stateless API annotation only when its parsed contract is unhealthy."""
    retry_policy = format_retry_policy(format_retry_model, format_retry_effort, format_retries, backend)
    responses = []
    total_latency = 0.0
    for _attempt in range(format_retries + 1):
        request_model = retry_policy["model"] if _attempt and retry_policy else model
        request_effort = retry_policy["effort"] if _attempt and retry_policy else effort
        request_prompt = format_retry_prompt if _attempt and format_retry_prompt is not None else prompt
        response, latency = await request_one(
            session,
            semaphore,
            url=url,
            payload=build_payload(request_model, request_prompt, max_tokens, request_effort, backend),
            retries=transport_retries,
            index=index,
        )
        total_latency += latency
        health = codex_output_health(
            response_text_or_empty(response),
            doc["text"],
            tagset,
            fmt,
            allow_overlapping_spans=allow_overlapping_spans,
            base_spans=doc.get("_subclass_base_spans"),
            candidate_ledger=doc.get("_subclass_candidate_ledger"),
            subclass_spec=subclass_spec,
        )
        capped = response.get("stop_reason") == "max_tokens" or any(
            choice.get("finish_reason") == "length" for choice in response.get("choices", [])
        )
        if capped:
            health["reasons"].append("unfinished_response")
            health["unhealthy"] = True
        response["annotation_request"] = {"model": request_model, "effort": request_effort}
        if format_retry_prompt is not None:
            response["annotation_request"]["prompt_sha256"] = hashlib.sha256(
                request_prompt.encode()
            ).hexdigest()
        response["annotation_health"] = health
        responses.append(response)
        if not health["unhealthy"]:
            break

    final_response = dict(responses[-1])
    if len(responses) > 1:
        final_response["format_recovery_responses"] = responses[:-1]
        final_response["aggregate_usage"] = {
            name: sum(usage_details(response)[name] for response in responses) for name in CODEX_USAGE_FIELDS
        }
    final_response["format_attempts"] = len(responses)
    if retry_policy is not None:
        final_response["format_retry_policy"] = retry_policy
    return final_response, total_latency, None


def format_retry_policy(model: str | None, effort: str | None, retries: int, backend: str) -> dict | None:
    """Freeze the optional single fresh request to a different annotation model."""
    if model is None:
        if effort is not None:
            raise ValueError("--format-retry-effort requires --format-retry-model")
        return None
    if backend not in {"anthropic", "openai"} or retries != 1 or effort is None:
        raise ValueError(
            "--format-retry-model requires a stateless API backend, --format-retries 1 "
            "and explicit --format-retry-effort"
        )
    return {"model": model, "effort": effort, "max_attempts": 1, "mode": "fresh_annotation"}


def require_retry_policy_resume(contract: dict, model: str, effort: str, policy: dict | None) -> None:
    """An existing raw prefix cannot acquire a different cross-model retry contract."""
    backend = contract["backend"]
    if policy is not None or backend.get("format_retry") is not None:
        if (contract["model"], backend["effort"], backend.get("format_retry")) != (model, effort, policy):
            raise ValueError("resume model/effort/retry policy differs from the frozen teacher contract")


def codex_compaction_event_types(response: dict[str, Any]) -> list[str]:
    """Name explicit compaction events so they stop and enter the attempt inventory."""
    events = response.get("codex_events", [])
    if not isinstance(events, list):
        return []
    result = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_name = str(event.get("type") or event.get("method") or "")
        if "compact" in event_name.lower():
            result.add(event_name)
        params = event.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        item_type = str(item.get("type") or "") if isinstance(item, dict) else ""
        if "compact" in item_type.lower():
            result.add(item_type)
    return sorted(result)


async def request_codex_document(
    entries: list[tuple[int, str, dict[str, Any]]],
    semaphore: asyncio.Semaphore,
    *,
    initial_command: list[str],
    command_name: str,
    model: str,
    effort: str,
    timeout: float,
    retries: int,
    fork_retries: int,
    recovery_retries: int,
    tagset: set[str],
    initial_session_id: str | None,
    env: dict[str, str],
    bootstrap_input_limit: int,
    context_input_limit: int,
    output_token_warning: int,
    fmt: str,
    allow_overlapping_spans: bool = False,
    subclass_spec: SubclassSpec | None = None,
) -> tuple[dict[int, tuple[dict[str, Any], float, dict[str, Any]]], str | None]:
    results = {}
    session_id = initial_session_id
    for index, prompt, doc in entries:
        responses = []
        total_latency = 0.0
        unhealthy = True
        output_warning_reached = False
        final_session_id = None
        parent_session_id = session_id
        attempts = codex_document_attempts(
            parent_session_id,
            fork_retries=fork_retries,
            fresh_retries=recovery_retries,
        )
        for attempt_kind, resume_session_id in attempts:
            command = (
                initial_command
                if resume_session_id is None
                else build_codex_resume_command(command_name, model, effort, resume_session_id)
            )
            response, latency = await request_codex_one(
                semaphore,
                command=command,
                prompt=prompt,
                timeout=timeout,
                retries=retries,
                index=index,
                env=env,
                input_token_limit=bootstrap_input_limit if resume_session_id is None else 0,
            )
            response["session_attempt"] = {
                "kind": attempt_kind,
                "parent_thread_id": resume_session_id,
            }
            responses.append(response)
            total_latency += latency
            final_session_id = codex_session_id(response)
            health = codex_output_health(
                response_text(response),
                doc["text"],
                tagset,
                fmt,
                allow_overlapping_spans=allow_overlapping_spans,
                base_spans=doc.get("_subclass_base_spans"),
                candidate_ledger=doc.get("_subclass_candidate_ledger"),
                subclass_spec=subclass_spec,
            )
            attempt_output_tokens = visible_output_tokens(response)
            output_warning_reached = bool(
                output_token_warning and attempt_output_tokens >= output_token_warning
            )
            compaction_event_types = codex_compaction_event_types(response)
            health_reasons = list(health["reasons"])
            if compaction_event_types:
                health_reasons.append("context_compaction_event")
            if output_warning_reached:
                health_reasons.append("output_token_warning")
            response["session_attempt"]["health"] = {
                **health,
                "unhealthy": bool(health_reasons),
                "reasons": health_reasons,
                "compaction_event_types": compaction_event_types,
                "output_token_warning": output_token_warning,
                "visible_output_tokens": attempt_output_tokens,
                "output_warning_reached": output_warning_reached,
            }
            unhealthy = bool(health_reasons)
            if not unhealthy:
                session_id = final_session_id
                break
            session_id = None

        final_response = dict(responses[-1])
        if len(responses) > 1:
            final_response["session_recovery_responses"] = responses[:-1]
            final_response["aggregate_usage"] = {
                name: sum(usage_details(response)[name] for response in responses)
                for name in CODEX_USAGE_FIELDS
            }
        context_input_tokens, output_tokens = usage_counts(responses[-1])
        final_visible_output_tokens = visible_output_tokens(responses[-1])
        context_limit_reached = bool(context_input_limit and context_input_tokens >= context_input_limit)
        output_warning_reached = bool(
            output_token_warning and final_visible_output_tokens >= output_token_warning
        )
        if context_limit_reached or output_warning_reached:
            # Keep a parser-accepted current row, but never use a context-sized
            # or abnormally long completion as the next segment's anchor.
            session_id = None
        session_metadata = {
            "thread_id": final_session_id,
            "prior_context_thread_id": parent_session_id,
            "history_reset": responses[-1]["session_attempt"]["kind"] in {"fresh_start", "fresh_recovery"},
            "recovery_attempts": len(responses) - 1,
            "context_input_tokens": context_input_tokens,
            "output_tokens": output_tokens,
            "visible_output_tokens": final_visible_output_tokens,
            "context_input_limit": context_input_limit,
            "context_limit_reached": context_limit_reached,
            "output_token_warning": output_token_warning,
            "output_warning_reached": output_warning_reached,
            "attempt_kinds": [response["session_attempt"]["kind"] for response in responses],
            "health_reasons": responses[-1]["session_attempt"]["health"]["reasons"],
            "compaction_event_types": responses[-1]["session_attempt"]["health"]["compaction_event_types"],
            "continuable": not unhealthy and not context_limit_reached,
        }
        results[index] = final_response, total_latency, session_metadata
    return results, session_id


def record_codex_attempt_health(
    response: dict[str, Any],
    doc: dict[str, Any],
    *,
    tagset: set[str],
    fmt: str,
    output_token_warning: int,
    allow_overlapping_spans: bool,
    subclass_spec: SubclassSpec | None = None,
) -> bool:
    """Attach structural/context health and return whether this attempt is rejected."""
    health = codex_output_health(
        response_text(response),
        doc["text"],
        tagset,
        fmt,
        allow_overlapping_spans=allow_overlapping_spans,
        base_spans=doc.get("_subclass_base_spans"),
        candidate_ledger=doc.get("_subclass_candidate_ledger"),
        subclass_spec=subclass_spec,
    )
    attempt_visible_output_tokens = visible_output_tokens(response)
    output_warning_reached = bool(
        output_token_warning and attempt_visible_output_tokens >= output_token_warning
    )
    compaction_event_types = codex_compaction_event_types(response)
    health_reasons = list(health["reasons"])
    attempt_failure = response.get("attempt_failure")
    if isinstance(attempt_failure, dict):
        failure_kind = attempt_failure.get("kind")
        if not isinstance(failure_kind, str) or not failure_kind:
            raise RequestFailure("Codex attempt failure has no kind", retryable=False)
        health_reasons.append(failure_kind)
    if compaction_event_types:
        health_reasons.append("context_compaction_event")
    if output_warning_reached:
        health_reasons.append("output_token_warning")
    response["session_attempt"]["health"] = {
        **health,
        "unhealthy": bool(health_reasons),
        "reasons": health_reasons,
        "compaction_event_types": compaction_event_types,
        "output_token_warning": output_token_warning,
        "visible_output_tokens": attempt_visible_output_tokens,
        "output_warning_reached": output_warning_reached,
    }
    return bool(health_reasons)


def codex_failed_attempt(kind: str, message: str) -> dict[str, Any]:
    """Return a parser-safe response that the session guard must reject."""
    return {
        "content": [{"type": "text", "text": "[]"}],
        "usage": {},
        "codex_events": [],
        "attempt_failure": {"kind": kind, "message": message},
    }


def codex_thread_store_connection_timeout(error: CodexAppServerError) -> bool:
    """Recognize the observed transient failure at the thread-fork boundary."""
    message = str(error)
    return all(
        fragment in message
        for fragment in (
            "app-server thread/fork failed:",
            "thread-store internal error:",
            "pool timed out while waiting for an open connection",
        )
    )


async def request_codex_forked_one(
    server: CodexAppServer,
    semaphore: asyncio.Semaphore,
    *,
    source_thread_id: str,
    prompt: str,
    command_workdir: Path,
    model: str,
    base_instructions: str,
    effort: str,
    timeout: float,
    index: int,
    attempt_kind: str,
) -> tuple[dict[str, Any], float, str | None]:
    """Fork an immutable source thread and run exactly one annotation turn."""
    from scripts.pii_codex_app_server import CodexAppServerError

    started = time.perf_counter()
    branch_thread_id = None
    try:
        async with semaphore:
            branch = await server.fork_thread(
                source_thread_id,
                cwd=str(command_workdir),
                model=model,
                base_instructions=base_instructions,
            )
            branch_thread_id = str(branch["id"])
            try:
                response = await server.run_turn(
                    branch_thread_id,
                    prompt,
                    cwd=str(command_workdir),
                    model=model,
                    effort=effort,
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                print(f"TIMEOUT: row {index} Codex turn exceeded {timeout:g}s")
                response = codex_failed_attempt(
                    "turn_timeout",
                    f"Codex turn exceeded {timeout:g}s",
                )
    except CodexAppServerError as error:
        if not codex_thread_store_connection_timeout(error):
            raise RequestFailure(f"row {index}: {error}", retryable=False) from error
        print(f"THREAD STORE TIMEOUT: row {index} Codex fork could not obtain a connection")
        response = codex_failed_attempt(
            "thread_store_connection_timeout",
            str(error),
        )
    response["session_attempt"] = {
        "kind": attempt_kind,
        "parent_thread_id": source_thread_id,
        "forked_thread_id": branch_thread_id,
    }
    return response, time.perf_counter() - started, branch_thread_id


async def request_codex_forked_document(
    entries: list[tuple[int, str, dict[str, Any]]],
    server: CodexAppServer,
    semaphore: asyncio.Semaphore,
    *,
    protocol_root_thread_id: str,
    command_workdir: Path,
    model: str,
    base_instructions: str,
    effort: str,
    timeout: float,
    tagset: set[str],
    initial_session_id: str | None,
    recovery_retries: int = 1,
    bootstrap_input_limit: int,
    context_input_limit: int,
    output_token_warning: int,
    fmt: str,
    allow_overlapping_spans: bool = False,
    subclass_spec: SubclassSpec | None = None,
) -> tuple[dict[int, tuple[dict[str, Any], float, dict[str, Any]]], str | None]:
    """Annotate one bounded part under its zero-or-one paragraph-replay budget."""
    if recovery_retries not in {0, 1}:
        raise ValueError("protocol-root recovery retries must be zero or one")

    async def annotate(
        entry: tuple[int, str, dict[str, Any]],
        source_thread_id: str,
        attempt_kind: str,
    ) -> tuple[dict[str, Any], float, str | None, bool]:
        index, prompt, doc = entry
        response, latency, thread_id = await request_codex_forked_one(
            server,
            semaphore,
            source_thread_id=source_thread_id,
            prompt=prompt,
            command_workdir=command_workdir,
            model=model,
            base_instructions=base_instructions,
            effort=effort,
            timeout=timeout,
            index=index,
            attempt_kind=attempt_kind,
        )
        enforce_codex_input_limit(
            response,
            index=index,
            maximum=bootstrap_input_limit if source_thread_id == protocol_root_thread_id else 0,
        )
        unhealthy = record_codex_attempt_health(
            response,
            doc,
            tagset=tagset,
            fmt=fmt,
            output_token_warning=output_token_warning,
            allow_overlapping_spans=allow_overlapping_spans,
            subclass_spec=subclass_spec,
        )
        return response, latency, thread_id, unhealthy

    def finish(
        index: int,
        responses: list[dict[str, Any]],
        total_latency: float,
        *,
        prior_context_thread_id: str | None,
        paragraph_replayed: bool,
        paragraph_replay_trigger_index: int | None,
    ) -> tuple[tuple[dict[str, Any], float, dict[str, Any]], str | None]:
        final_response = dict(responses[-1])
        if len(responses) > 1:
            final_response["session_recovery_responses"] = responses[:-1]
            final_response["aggregate_usage"] = {
                name: sum(usage_details(response)[name] for response in responses)
                for name in CODEX_USAGE_FIELDS
            }
        final_attempt = responses[-1]["session_attempt"]
        health = final_attempt["health"]
        unhealthy = bool(health["unhealthy"])
        context_input_tokens, output_tokens = usage_counts(responses[-1])
        final_visible_output_tokens = visible_output_tokens(responses[-1])
        context_limit_reached = bool(context_input_limit and context_input_tokens >= context_input_limit)
        output_warning_reached = bool(
            output_token_warning and final_visible_output_tokens >= output_token_warning
        )
        final_thread_id = final_attempt["forked_thread_id"]
        next_session_id = None if unhealthy or context_limit_reached else final_thread_id
        length_incident_eligible = bool(
            len(responses) == 2 and responses[0]["session_attempt"]["health"]["unhealthy"] and not unhealthy
        )
        session_metadata = {
            "thread_id": final_thread_id,
            "protocol_root_thread_id": protocol_root_thread_id,
            "prior_context_thread_id": prior_context_thread_id,
            "forked_from_thread_id": final_attempt["parent_thread_id"],
            "history_reset": final_attempt["parent_thread_id"] == protocol_root_thread_id,
            "recovery_attempts": len(responses) - 1,
            "paragraph_replayed": paragraph_replayed,
            "paragraph_replay_trigger_index": paragraph_replay_trigger_index,
            "context_input_tokens": context_input_tokens,
            "output_tokens": output_tokens,
            "visible_output_tokens": final_visible_output_tokens,
            "context_input_limit": context_input_limit,
            "context_limit_reached": context_limit_reached,
            "output_token_warning": output_token_warning,
            "output_warning_reached": output_warning_reached,
            "attempt_kinds": [response["session_attempt"]["kind"] for response in responses],
            "health_reasons": health["reasons"],
            "compaction_event_types": health["compaction_event_types"],
            "annotation_banned": unhealthy,
            "length_incident_eligible": length_incident_eligible,
            "continuable": next_session_id is not None,
        }
        return (final_response, total_latency, session_metadata), next_session_id

    paragraphs: list[list[tuple[int, str, dict[str, Any]]]] = []
    for entry in entries:
        plan = entry[2].get("_codex_session_plan")
        if not isinstance(plan, dict):
            raise ValueError(f"row {entry[0]} has no bounded document-session plan")
        if not paragraphs or plan.get("turn_in_paragraph") == 1:
            paragraphs.append([])
        paragraphs[-1].append(entry)

    results = {}
    session_id = initial_session_id
    for paragraph in paragraphs:
        original: dict[int, tuple[dict[str, Any], float, str | None]] = {}
        paragraph_start_session_id = session_id
        trigger_index = None
        for entry in paragraph:
            index = entry[0]
            prior_session_id = session_id
            source_thread_id = prior_session_id or protocol_root_thread_id
            attempt_kind = "continuation_fork" if prior_session_id else "protocol_root_fork"
            response, latency, thread_id, unhealthy = await annotate(
                entry,
                source_thread_id,
                attempt_kind,
            )
            original[index] = response, latency, prior_session_id
            context_input_tokens, _output_tokens = usage_counts(response)
            context_limit_reached = bool(context_input_limit and context_input_tokens >= context_input_limit)
            session_id = None if unhealthy or context_limit_reached else thread_id
            if unhealthy:
                trigger_index = index
                break

        if trigger_index is None:
            for entry in paragraph:
                index = entry[0]
                response, latency, prior_session_id = original[index]
                result, _next_session_id = finish(
                    index,
                    [response],
                    latency,
                    prior_context_thread_id=prior_session_id,
                    paragraph_replayed=False,
                    paragraph_replay_trigger_index=None,
                )
                results[index] = result
            continue

        if recovery_retries == 0:
            session_id = None
            for entry in paragraph:
                index = entry[0]
                if index in original:
                    response, latency, prior_session_id = original[index]
                else:
                    prior_session_id = session_id
                    source_thread_id = prior_session_id or protocol_root_thread_id
                    attempt_kind = "continuation_fork" if prior_session_id else "protocol_root_fork"
                    response, latency, _thread_id, _unhealthy = await annotate(
                        entry,
                        source_thread_id,
                        attempt_kind,
                    )
                result, session_id = finish(
                    index,
                    [response],
                    latency,
                    prior_context_thread_id=prior_session_id,
                    paragraph_replayed=False,
                    paragraph_replay_trigger_index=None,
                )
                results[index] = result
            continue

        session_id = None
        for replay_position, entry in enumerate(paragraph):
            index = entry[0]
            prior_session_id = session_id
            source_thread_id = prior_session_id or protocol_root_thread_id
            replay_kind = (
                "protocol_root_paragraph_replay"
                if replay_position == 0 or prior_session_id is None
                else "paragraph_replay_continuation_fork"
            )
            response, latency, thread_id, unhealthy = await annotate(
                entry,
                source_thread_id,
                replay_kind,
            )
            attempts = []
            total_latency = latency
            if index in original:
                old_response, old_latency, _old_prior = original[index]
                attempts.append(old_response)
                total_latency += old_latency
            attempts.append(response)
            if unhealthy and index not in original:
                response, recovery_latency, thread_id, unhealthy = await annotate(
                    entry,
                    protocol_root_thread_id,
                    "protocol_root_row_recovery",
                )
                attempts.append(response)
                total_latency += recovery_latency
            result, session_id = finish(
                index,
                attempts,
                total_latency,
                prior_context_thread_id=paragraph_start_session_id,
                paragraph_replayed=True,
                paragraph_replay_trigger_index=trigger_index,
            )
            results[index] = result
            if unhealthy:
                session_id = None
            elif session_id is not None:
                session_id = thread_id
    return results, session_id


CODEX_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def usage_details(response: dict[str, Any]) -> dict[str, int]:
    """Normalize total prompt volume and its cache subsets without changing raw usage."""
    usage = response.get("aggregate_usage") or response.get("usage") or {}
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        # Messages input_tokens excludes both cache reads and cache writes.
        cached = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        usage = {
            **usage,
            "input_tokens": int(usage.get("input_tokens") or 0) + cached + written,
            "cached_input_tokens": cached,
            "cache_write_input_tokens": written,
        }
    if "input_tokens" not in usage and "prompt_tokens" in usage:
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        usage = {
            "input_tokens": usage.get("prompt_tokens"),
            "cached_input_tokens": prompt_details.get("cached_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_output_tokens": completion_details.get("reasoning_tokens"),
        }
    return {name: int(usage.get(name) or 0) for name in CODEX_USAGE_FIELDS}


def usage_counts(response: dict[str, Any]) -> tuple[int, int]:
    usage = usage_details(response)
    return usage["input_tokens"], usage["output_tokens"]


def visible_output_tokens(response: dict[str, Any]) -> int:
    """Return completion tokens visible to the parser, excluding hidden reasoning."""
    usage = usage_details(response)
    return max(0, usage["output_tokens"] - usage["reasoning_output_tokens"])


def _paragraph_boundary_between(
    left: dict[str, Any],
    right: dict[str, Any],
    paragraph_field: str | None = None,
) -> bool:
    """Whether two pre-segmented rows meet at a visible paragraph boundary."""
    if paragraph_field is not None:
        return left[paragraph_field] != right[paragraph_field]
    left_text = str(left["text"])
    right_text = str(right["text"])
    return (
        left_text.endswith(("\n\n", "\r\n\r\n"))
        or right_text.startswith(("\n\n", "\r\n\r\n"))
        or (left_text.endswith(("\n", "\r\n")) and right_text.startswith(("\n", "\r\n")))
    )


def plan_document_session_parts(
    docs: list[dict[str, Any]],
    session_field: str,
    *,
    max_segments: int,
    max_source_chars: int,
    paragraph_field: str | None = None,
) -> list[dict[str, Any]]:
    """Partition each logical document into bounded persistent-session parts.

    Inputs are already offset-preserving segments.  We prefer the last paragraph
    edge that fits both hard limits and otherwise cut at the last admissible
    segment edge.  A single over-limit segment is refused rather than silently
    violating the absolute bound; its upstream segmenter must split it first.
    """
    if max_segments <= 0 or max_source_chars <= 0:
        raise ValueError("document-session segment and source-character limits must be positive")
    grouped: dict[str, list[int]] = {}
    for index, doc in enumerate(docs):
        value = doc.get(session_field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"input row {index} id={doc.get('id')!r} lacks nonempty string "
                f"session field {session_field!r}"
            )
        text_length = len(str(doc.get("text", "")))
        if text_length > max_source_chars:
            raise ValueError(
                f"input row {index} id={doc.get('id')!r} has {text_length} source characters, "
                f"above --session-max-source-chars={max_source_chars}; split that row first"
            )
        if paragraph_field is not None:
            paragraph = doc.get(paragraph_field)
            if not isinstance(paragraph, str) or not paragraph:
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks nonempty string "
                    f"paragraph field {paragraph_field!r}"
                )
        grouped.setdefault(value, []).append(index)

    result: list[dict[str, Any] | None] = [None] * len(docs)
    for value, indices in grouped.items():
        start = 0
        part = 0
        previous_split_kind: str | None = None
        while start < len(indices):
            hard_end = start
            source_chars = 0
            while hard_end < len(indices) and hard_end - start < max_segments:
                candidate_chars = source_chars + len(str(docs[indices[hard_end]]["text"]))
                if candidate_chars > max_source_chars:
                    break
                source_chars = candidate_chars
                hard_end += 1
            if hard_end == start:
                raise AssertionError("a prevalidated segment must fit an empty session part")

            end = hard_end
            split_kind = None
            if hard_end < len(indices):
                paragraph_ends = [
                    position
                    for position in range(start + 1, hard_end + 1)
                    if _paragraph_boundary_between(
                        docs[indices[position - 1]],
                        docs[indices[position]],
                        paragraph_field,
                    )
                ]
                if paragraph_ends:
                    end = paragraph_ends[-1]
                    split_kind = "paragraph_boundary"
                else:
                    split_kind = "hard_segment_boundary"

            selected = indices[start:end]
            part_chars = sum(len(str(docs[index]["text"])) for index in selected)
            source_chars_before_turn = 0
            turn_in_paragraph = 0
            paragraph_in_part = 0
            for turn, index in enumerate(selected, 1):
                if turn == 1 or _paragraph_boundary_between(
                    docs[selected[turn - 2]],
                    docs[index],
                    paragraph_field,
                ):
                    turn_in_paragraph = 1
                    paragraph_in_part += 1
                else:
                    turn_in_paragraph += 1
                row_chars = len(str(docs[index]["text"]))
                result[index] = {
                    "value": value,
                    "part": part,
                    "turn_in_part": turn,
                    "segments_in_part": len(selected),
                    "source_chars_in_part": part_chars,
                    "segments_before_turn": turn - 1,
                    "source_chars_before_turn": source_chars_before_turn,
                    "source_chars_through_turn": source_chars_before_turn + row_chars,
                    "paragraph_in_part": paragraph_in_part,
                    "turn_in_paragraph": turn_in_paragraph,
                    "paragraph_value": (
                        docs[index][paragraph_field] if paragraph_field is not None else None
                    ),
                    "split_before": (None if part == 0 or turn != 1 else previous_split_kind),
                }
                source_chars_before_turn += row_chars
            previous_split_kind = split_kind
            start = end
            part += 1

    if any(item is None for item in result):
        raise AssertionError("every input row must receive a document-session part")
    return [dict(item) for item in result if item is not None]


def paragraph_aligned_commit_ranges(
    docs: list[dict[str, Any]],
    start: int,
    wave: int,
    *,
    allow_transport_failure_prefix: bool = False,
) -> list[tuple[int, int]]:
    """Return ordered write ranges that never split one planned paragraph."""
    if not 0 <= start <= len(docs):
        raise ValueError("resume prefix is outside the input")
    if wave <= 0:
        raise ValueError("wave size must be positive")
    if start < len(docs) and not allow_transport_failure_prefix:
        first_plan = docs[start].get("_codex_session_plan")
        if not isinstance(first_plan, dict) or first_plan.get("turn_in_paragraph") != 1:
            raise ValueError(
                "protocol-root resume prefix ends inside a paragraph; "
                "truncate to the preceding paragraph boundary"
            )

    keys = []
    last_index: dict[tuple[str, int, int], int] = {}
    for index, doc in enumerate(docs):
        plan = doc.get("_codex_session_plan")
        if not isinstance(plan, dict):
            raise ValueError(f"input row {index} has no bounded session plan")
        key = (str(plan["value"]), int(plan["part"]), int(plan["paragraph_in_part"]))
        keys.append(key)
        last_index[key] = index

    ranges = []
    position = start
    while position < len(docs):
        end = min(position + wave, len(docs))
        while True:
            extended = max(last_index[key] + 1 for key in set(keys[position:end]))
            if extended <= end:
                break
            end = extended
        ranges.append((position, end))
        position = end
    return ranges


def summarize_document_session_parts(plans: list[dict[str, Any]]) -> dict[str, Any]:
    parts = {(plan["value"], plan["part"]) for plan in plans}
    return {
        "documents": len({plan["value"] for plan in plans}),
        "parts": len(parts),
        "paragraph_splits": sum(plan.get("split_before") == "paragraph_boundary" for plan in plans),
        "hard_splits": sum(plan.get("split_before") == "hard_segment_boundary" for plan in plans),
    }


def require_resume_pair(
    pred_path: Path, raw_path: Path, docs: list[dict[str, Any]], *, retry_policy: dict | None = None
) -> int:
    pred_exists = pred_path.exists()
    raw_exists = raw_path.exists()
    if pred_exists != raw_exists:
        raise ValueError("prediction and raw-response resume files must either both exist or both be absent")
    if not pred_exists:
        return 0
    pred_count = validated_resume_count(pred_path, docs)
    raw_count = validated_resume_count(raw_path, docs)
    if pred_count != raw_count:
        raise ValueError(f"resume files differ in length: predictions={pred_count}, raw={raw_count}")
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if json.loads(line)["response"].get("format_retry_policy") != retry_policy:
            raise ValueError("resume retry policy differs from the saved raw prefix")
    return pred_count


def require_session_event_prefix(
    event_path: Path,
    docs: list[dict[str, Any]],
    completed: int,
) -> None:
    if not event_path.is_file():
        raise ValueError(f"protocol-root resume requires existing session event table: {event_path}")
    selected: dict[int, str] = {}
    with event_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            index = row.get("input_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"{event_path}:{line_number}: invalid input_index")
            if not 0 <= index < completed:
                raise ValueError(
                    f"{event_path}:{line_number}: event index {index} is outside completed prefix {completed}"
                )
            if row.get("selected_attempt"):
                if index in selected:
                    raise ValueError(
                        f"{event_path}:{line_number}: duplicate selected attempt for row {index}"
                    )
                selected[index] = str(row.get("id"))
    expected = {index: str(doc["id"]) for index, doc in enumerate(docs[:completed])}
    if selected != expected:
        raise ValueError("session event table selected attempts do not match the completed input prefix")


def require_fresh_protocol_resume_paths(contract_path: Path, event_path: Path) -> None:
    """Require a new protocol ledger while retaining an existing data-row prefix."""
    existing = [str(path) for path in (contract_path, event_path) if path.exists()]
    if existing:
        raise ValueError(
            f"fresh protocol-root resume requires new prompt-contract and session-event paths: {existing}"
        )


def require_transport_failure_resume(raw_path: Path, completed: int) -> None:
    """Admit a mid-paragraph resume only from a verified recovered prefix."""
    if completed <= 0:
        raise ValueError("transport-failure resume requires a nonempty recovered prefix")
    last_session = None
    with raw_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= completed:
                break
            row = json.loads(line)
            session = row.get("session")
            if not isinstance(session, dict) or session.get("recovered_after_transport_failure") is not True:
                raise ValueError(
                    f"{raw_path}:{index + 1}: transport-failure resume requires a verified recovered row"
                )
            last_session = session
    if not isinstance(last_session, dict) or last_session.get("continuable") is not True:
        raise ValueError("transport-failure resume prefix has no healthy continuation thread")


def resumed_codex_sessions(
    raw_path: Path,
    docs: list[dict[str, Any]],
    completed: int,
    session_field: str,
) -> dict[str, str | None]:
    sessions: dict[str, str | None] = {}
    with raw_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= completed:
                break
            row = json.loads(line)
            session = row.get("session")
            value = str(docs[index][session_field])
            if (
                not isinstance(session, dict)
                or session.get("field") != session_field
                or session.get("value") != value
            ):
                raise ValueError(
                    f"{raw_path}:{index + 1}: missing session metadata for {session_field}={value!r}"
                )
            thread_id = session.get("thread_id")
            if session.get("continuable"):
                if not isinstance(thread_id, str) or not thread_id:
                    raise ValueError(f"{raw_path}:{index + 1}: continuable session has no thread id")
                sessions[value] = thread_id
            else:
                sessions[value] = None
    return sessions


def resumed_codex_session_parts(
    raw_path: Path,
    docs: list[dict[str, Any]],
    completed: int,
    session_field: str,
) -> dict[tuple[str, int], str | None]:
    """Recover the latest thread separately for each bounded document part."""
    sessions: dict[tuple[str, int], str | None] = {}
    with raw_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= completed:
                break
            row = json.loads(line)
            session = row.get("session")
            plan = docs[index].get("_codex_session_plan")
            value = str(docs[index][session_field])
            if not isinstance(plan, dict):
                raise ValueError(f"input row {index} has no bounded session plan")
            part = int(plan["part"])
            if (
                not isinstance(session, dict)
                or session.get("field") != session_field
                or session.get("value") != value
                or session.get("part") != part
            ):
                raise ValueError(
                    f"{raw_path}:{index + 1}: missing session metadata for "
                    f"{session_field}={value!r} part={part}"
                )
            thread_id = session.get("thread_id")
            key = (value, part)
            if session.get("continuable"):
                if not isinstance(thread_id, str) or not thread_id:
                    raise ValueError(f"{raw_path}:{index + 1}: continuable session has no thread id")
                sessions[key] = thread_id
            else:
                sessions[key] = None
    return sessions


def write_prompt_contract(
    path: Path,
    task_path: Path,
    examples_path: Path,
    task_template: str,
    examples: dict[str, Any],
    docs: list[dict[str, Any]],
    tags: list[str],
    fmt: str,
    language: str,
    model: str,
    *,
    url: str,
    effort: str,
    format_retries: int = 0,
    format_retry_model: str | None = None,
    format_retry_effort: str | None = None,
    backend: str = "anthropic",
    codex_command: str = "codex",
    codex_workdir: Path | None = None,
    codex_home: Path | None = None,
    codex_bootstrap_input_limit: int = 20_000,
    tag_catalog: dict[str, dict[str, str]] | None = None,
    tag_catalog_sha256: str | None = None,
    span_policy: str = "legacy",
    input_render: str = "plain",
    proposal_render: str = "markup",
    proposal_field: str | None = None,
    guidance_field: str | None = None,
    unicode_normalization: str = "none",
    language_rules: dict[str, list[str]] | None = None,
    language_rules_path: Path | None = None,
    session_field: str | None = None,
    session_paragraph_field: str | None = None,
    session_max_segments: int = 0,
    session_max_source_chars: int = 0,
    session_context_input_limit: int = 0,
    session_output_token_warning: int = 0,
    session_allow_overlapping_spans: bool = False,
    session_fork_retries: int = 0,
    session_recovery_retries: int = 0,
    protocol_root: dict[str, Any] | None = None,
    subclass_spec: SubclassSpec | None = None,
) -> None:
    contract = build_prompt_contract(
        task_template,
        examples,
        docs,
        tags,
        fmt,
        language,
        model,
        tag_catalog=tag_catalog,
        tag_catalog_sha256=tag_catalog_sha256,
        span_policy=span_policy,
        input_render=input_render,
        proposal_render=proposal_render,
        proposal_field=proposal_field,
        guidance_field=guidance_field,
        unicode_normalization=unicode_normalization,
        language_rules=language_rules,
        language_rules_sha256=(
            hashlib.sha256(language_rules_path.read_bytes()).hexdigest() if language_rules_path else None
        ),
    )
    contract["task_template_path"] = str(task_path.resolve())
    if language_rules_path is not None:
        contract["language_rules"]["path"] = str(language_rules_path.resolve())
    _, assembly = load_prompt_template(task_path, model)
    if assembly is not None:
        contract["template_assembly"] = assembly
        if format_retry_model is not None:
            _, retry_assembly = load_prompt_template(task_path, format_retry_model)
            contract["format_retry_template_assembly"] = retry_assembly
    contract["examples_path"] = str(examples_path.resolve())
    if subclass_spec is not None:
        contract["subclass_spec"] = {
            "path": str(subclass_spec.path.resolve()),
            "sha256": subclass_spec.sha256,
            "head_rows": subclass_spec.head_rows,
            "blocks": subclass_spec.config_blocks(),
        }
    if backend == "anthropic":
        contract["backend"] = {
            "kind": "anthropic_messages",
            "url": url,
            "effort": effort,
            "format_retries": format_retries,
        }
    elif backend == "openai":
        contract["backend"] = {
            "kind": "openai_chat_completions",
            "url": url,
            "effort": effort,
            "temperature": 0.0,
            "format_retries": format_retries,
        }
    else:
        if codex_workdir is None or codex_home is None:
            raise ValueError("Codex backend requires isolated home and work directories")
        contract["backend"] = {
            "kind": "codex_app_server" if protocol_root is not None else "codex_exec",
            "command": (f"{codex_command} app-server" if protocol_root is not None else codex_command),
            "workdir": str(codex_workdir.resolve()),
            "codex_home": str(codex_home.resolve()),
            "model": model,
            "effort": effort,
            "ephemeral": session_field is None,
            "ignore_user_config": protocol_root is None,
            "ignore_rules": protocol_root is None,
            "strict_config": True,
            "context_config": dict(CODEX_CONTEXT_CONFIG),
            "disabled_features": list(CODEX_DISABLED_FEATURES),
            "bootstrap_input_token_limit": codex_bootstrap_input_limit,
            "sandbox": "read-only",
            "approval_policy": "never",
        }
        if session_field is not None:
            plans = [doc.get("_codex_session_plan") for doc in docs]
            if not all(isinstance(plan, dict) for plan in plans):
                raise ValueError("Codex document sessions require a bounded part plan")
            contract["backend"]["session"] = {
                "group_field": session_field,
                "paragraph_field": session_paragraph_field,
                "order": "input order within each group",
                "output_unit": "one input segment",
                "max_segments_per_part": session_max_segments,
                "max_source_chars_per_part": session_max_source_chars,
                "context_input_token_limit": session_context_input_limit,
                "output_token_warning": session_output_token_warning,
                "output_token_warning_basis": ("visible completion tokens; hidden reasoning tokens excluded"),
                "allow_overlapping_spans": session_allow_overlapping_spans,
                "split_policy": (
                    "prefer the last paragraph-aligned segment edge within both hard limits; "
                    "otherwise split at the last admissible segment edge"
                ),
                "planned": summarize_document_session_parts(plans),
                "fork_retries_on_unhealthy_output": session_fork_retries,
                "fresh_retries_after_unhealthy_fork": session_recovery_retries,
            }
            if protocol_root is not None:
                contract["backend"]["session"]["paragraph_replay_policy"] = (
                    "stop on the first unhealthy continuation, discard that paragraph's "
                    "attempted outputs, and replay the full paragraph once from the immutable "
                    "protocol root; a repeated rejection bans that segment"
                    if session_recovery_retries == 1
                    else "do not replay an unhealthy turn; ban it, reset to the immutable "
                    "protocol root, and continue later segments without inheriting its context"
                )
                contract["backend"]["session"]["protocol_root"] = protocol_root
    retry_policy = format_retry_policy(format_retry_model, format_retry_effort, format_retries, backend)
    if retry_policy is not None:
        contract["backend"]["format_retry"] = retry_policy
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PROMPT: froze {len(contract['samples'])} rendered language samples -> {path}", flush=True)


async def ensure_codex_protocol_root(
    server: CodexAppServer,
    *,
    contract_path: Path,
    resume: bool,
    protocol_prefix: str,
    workdir: Path,
    model: str,
    effort: str,
    timeout: float,
) -> dict[str, Any]:
    prefix_sha256 = hashlib.sha256(protocol_prefix.encode()).hexdigest()
    if resume:
        if not contract_path.is_file():
            raise ValueError(f"protocol-root resume requires existing prompt contract: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        root = contract.get("backend", {}).get("session", {}).get("protocol_root")
        if not isinstance(root, dict):
            raise ValueError(f"prompt contract has no protocol-root state: {contract_path}")
        expected = {
            "base_instructions_sha256": prefix_sha256,
            "instruction_channel": "baseInstructions",
            "marker": CODEX_SESSION_TURN_MARKER,
            "model": model,
        }
        mismatches = {
            key: (root.get(key), value) for key, value in expected.items() if root.get(key) != value
        }
        if mismatches:
            raise ValueError(f"protocol-root resume identity changed: {mismatches}")
        thread_id = root.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("prompt contract protocol root has no thread id")
        thread = await server.read_thread(thread_id)
    else:
        staging_thread = await server.start_protocol_root(
            base_instructions=protocol_prefix,
            cwd=str(workdir),
            model=model,
        )
        staging_thread_id = str(staging_thread["id"])
        materialization = await server.run_turn(
            staging_thread_id,
            (
                "Controller guidance JSON string:\n"
                '{"base_annotation_status":"protocol materialization only",'
                '"base_spans":[],"offset_guide":[]}\n\n'
                'Input JSON string:\n""\n\nOutput:'
            ),
            cwd=str(workdir),
            model=model,
            effort=effort,
            timeout=timeout,
        )
        materialization_turn_id = str(materialization["app_server"]["turn_id"])
        thread = await server.fork_thread(
            staging_thread_id,
            cwd=str(workdir),
            model=model,
            before_turn_id=materialization_turn_id,
            base_instructions=protocol_prefix,
        )
        thread_id = str(thread["id"])
        materialization_text = response_text(materialization)
        root = {
            "thread_id": thread_id,
            "base_instructions_sha256": prefix_sha256,
            "instruction_channel": "baseInstructions",
            "marker": CODEX_SESSION_TURN_MARKER,
            "model": model,
            "contains_document_content": False,
            "materialization": {
                "strategy": "fork_before_synthetic_empty_input_turn",
                "staging_thread_id": staging_thread_id,
                "excluded_turn_id": materialization_turn_id,
                "response_text": materialization_text,
                "response_sha256": hashlib.sha256(materialization_text.encode()).hexdigest(),
                "usage": usage_details(materialization),
            },
            "fork_policy": "fork before every annotation turn; never append to the source anchor",
            "recovery_policy": (
                "on the first unhealthy continuation, replay its full paragraph once from "
                "the protocol root; a repeated rejection bans that segment"
            ),
        }
    turns = thread.get("turns", [])
    if not isinstance(turns, list) or turns:
        raise ValueError("protocol root must remain an empty-turn immutable thread")
    return dict(root)


def codex_attempt_event_rows(
    *,
    index: int,
    doc: dict[str, Any],
    response: dict[str, Any],
    session_metadata: dict[str, Any],
    task_template_path: Path,
    language: str,
    domain_field: str | None,
) -> list[dict[str, Any]]:
    """Flatten all retained attempts for later context-length analysis."""
    attempts = [*response.get("session_recovery_responses", []), response]
    plan = doc["_codex_session_plan"]
    final_rejected = bool(session_metadata["annotation_banned"])
    rows = []
    for attempt_number, attempt in enumerate(attempts, 1):
        attempt_metadata = attempt["session_attempt"]
        health = attempt_metadata["health"]
        direct_usage = attempt.get("usage") or {}
        usage = {name: int(direct_usage.get(name) or 0) for name in CODEX_USAGE_FIELDS}
        visible_tokens = max(0, usage["output_tokens"] - usage["reasoning_output_tokens"])
        rows.append(
            {
                "schema": "pii-codex-session-attempt-v1",
                "input_index": index,
                "id": doc["id"],
                "task_template": str(task_template_path.resolve()),
                "language": doc.get("lang", language),
                "domain": doc.get(domain_field) if domain_field is not None else None,
                "session_value": plan["value"],
                "session_part": plan["part"],
                "turn_in_part": plan["turn_in_part"],
                "paragraph_value": plan["paragraph_value"],
                "paragraph_in_part": plan["paragraph_in_part"],
                "turn_in_paragraph": plan["turn_in_paragraph"],
                "segments_before_turn": plan["segments_before_turn"],
                "source_chars_before_turn": plan["source_chars_before_turn"],
                "source_chars_through_turn": plan["source_chars_through_turn"],
                "attempt_number": attempt_number,
                "attempt_kind": attempt_metadata["kind"],
                "parent_thread_id": attempt_metadata["parent_thread_id"],
                "forked_thread_id": attempt_metadata["forked_thread_id"],
                "selected_attempt": attempt_number == len(attempts),
                "paragraph_replayed": session_metadata["paragraph_replayed"],
                "paragraph_replay_trigger_index": session_metadata["paragraph_replay_trigger_index"],
                "rejected": bool(health["unhealthy"]),
                "health_reasons": health["reasons"],
                "compaction_event_types": health["compaction_event_types"],
                **usage,
                "visible_output_tokens": visible_tokens,
                "final_annotation_banned": final_rejected,
                "exclude_from_length_fit": final_rejected,
                "length_incident_candidate": (
                    len(attempts) == 2
                    and attempt_number == 1
                    and bool(health["unhealthy"])
                    and not final_rejected
                ),
            }
        )
    return rows


async def run(args: argparse.Namespace) -> None:
    if args.codex_protocol_root:
        from scripts.pii_codex_app_server import CodexAppServer

    retry_policy = format_retry_policy(
        args.format_retry_model, args.format_retry_effort, args.format_retries, args.backend
    )
    if retry_policy is not None and not args.prompt_contract_out:
        raise ValueError("--format-retry-model requires --prompt-contract-out")
    prompt_dir = Path(PROMPT_DIR)
    task_path = Path(args.task_template or prompt_dir / "task.txt")
    examples_path = Path(args.examples or prompt_dir / "examples.json")
    task_template, template_assembly = load_prompt_template(task_path, args.model)
    retry_template = None
    if retry_policy is not None and template_assembly is not None:
        retry_template, _ = load_prompt_template(task_path, retry_policy["model"])
    if template_assembly is not None and args.resume:
        if not args.prompt_contract_out:
            raise ValueError("Resuming an included template requires its saved --prompt-contract-out")
        previous = json.loads(Path(args.prompt_contract_out).read_text())
        if previous.get("template_assembly") != template_assembly:
            raise ValueError("Prompt includes differ from the saved resume contract")
        if retry_policy is not None:
            _, retry_assembly = load_prompt_template(task_path, retry_policy["model"])
            if previous.get("format_retry_template_assembly") != retry_assembly:
                raise ValueError("Retry prompt includes differ from the saved resume contract")
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    tags = args.tags.split(",") if args.tags else CORE_TAGS
    tagset = set(tags)
    subclass_spec = load_subclass_spec(Path(args.subclass_spec)) if args.subclass_spec else None
    subclass_formats = {"json-subclasses", "json-candidate-subclasses"}
    if args.fmt in subclass_formats:
        if subclass_spec is None:
            raise SystemExit(f"--fmt {args.fmt} requires --subclass-spec")
        if task_template.count("{subclass_catalog}") != 1:
            raise ValueError(
                f"{args.fmt} task template requires exactly one {{subclass_catalog}} placeholder"
            )
        task_template = task_template.replace("{subclass_catalog}", render_subclass_catalog(subclass_spec))
        if retry_template is not None:
            if retry_template.count("{subclass_catalog}") != 1:
                raise ValueError("Retry task template requires exactly one {subclass_catalog} placeholder")
            retry_template = retry_template.replace(
                "{subclass_catalog}", render_subclass_catalog(subclass_spec)
            )
    elif subclass_spec is not None:
        raise SystemExit(
            "--subclass-spec is only valid with --fmt json-subclasses or json-candidate-subclasses"
        )
    # A class inventory the model has not seen before -- ontology v2 names such
    # as admin_area or record_identifier -- is under-specified by its name
    # alone, so the catalog carries each class's definition and an attested
    # surface into the prompt.
    tag_catalog = load_tag_catalog(args.tag_catalog) if args.tag_catalog else None
    language_rules = load_language_rules(args.language_rules) if args.language_rules else None
    if tag_catalog is not None:
        missing = [tag for tag in tags if tag not in tag_catalog]
        if missing:
            raise SystemExit(f"--tag-catalog defines no row for: {', '.join(missing)}")
    docs = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        docs = docs[: args.limit]
    evaluation_replay = None
    if args.verification_replay:
        receipts = (
            args.evaluation_replay_admission,
            args.reannotation_receipt,
            args.web_receipt,
            args.dedup_receipt,
        )
        if any(receipts):
            raise SystemExit("--verification-replay replaces admission receipts; pass none")
        # Opt-in, recorded bypass for re-running rows a published result already
        # annotated, to check that result; never an admission for training use.
        evaluation_replay = {
            "purpose": "verification_replay",
            "input_sha256": hashlib.sha256(Path(args.gold).read_bytes()).hexdigest(),
            "clearance": "No admission check; outputs verify a published result and are not training data.",
        }
    elif args.evaluation_replay_admission:
        evaluation_replay = require_evaluation_replay(
            Path(args.evaluation_replay_admission), Path(args.gold), docs
        )
    deduplication = None
    reannotation = None
    web_intake = None
    if args.reannotation_receipt:
        reannotation = require_training_reannotation(Path(args.reannotation_receipt), Path(args.gold), docs)
    elif args.web_receipt:
        web_intake = require_o4_web_intake(Path(args.web_receipt), docs)
    elif evaluation_replay is None:
        deduplication = require_annotation_dedup(
            Path(args.dedup_receipt) if args.dedup_receipt else None, Path(args.gold), docs
        )
    if not args.allow_example_fallback:
        require_language_specific_examples(examples, docs, args.lang)
    if subclass_spec is not None:
        for doc in docs:
            doc["_subclass_base_spans"] = annotation_base_spans(doc, args.guidance_field)
            if args.fmt == "json-candidate-subclasses":
                doc["_subclass_candidate_ledger"] = annotation_candidate_ledger(
                    doc,
                    args.guidance_field,
                )
    if args.fmt in {"json-groups", "json-groups-lexical"} and args.unicode_normalization != "NFKC":
        raise SystemExit("grouped JSON formats require --unicode-normalization NFKC")
    if args.unicode_normalization != "none":
        for index, doc in enumerate(docs):
            if doc["text"] != unicodedata.normalize(args.unicode_normalization, doc["text"]):
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} is not "
                    f"{args.unicode_normalization}-normalized at intake"
                )
    if args.proposal_field:
        for index, doc in enumerate(docs):
            proposals = doc.get(args.proposal_field)
            if not isinstance(proposals, list):
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks list proposal field "
                    f"{args.proposal_field!r}"
                )
    if args.source_paragraph_guidance:
        if args.guidance_field:
            raise SystemExit("--source-paragraph-guidance supplies the guidance field; omit --guidance-field")
        args.guidance_field = add_source_paragraph_guidance(docs, args.lang)
    if args.guidance_field:
        if "{guidance}" not in task_template:
            raise ValueError("--guidance-field requires a {guidance} task-template placeholder")
        for index, doc in enumerate(docs):
            guidance = doc.get(args.guidance_field)
            if args.guidance_field == "document_context" and isinstance(guidance, dict):
                if set(guidance) != {"before", "after"} or not all(
                    isinstance(value, str) for value in guidance.values()
                ):
                    raise ValueError("document_context must contain before/after text only")
                guidance = json.dumps(guidance, ensure_ascii=False)
                doc[args.guidance_field] = guidance
            if not isinstance(guidance, str) or not guidance:
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks nonempty string guidance "
                    f"field {args.guidance_field!r}"
                )
    if args.session_field:
        if args.backend != "codex" or args.fmt not in {
            "json-groups",
            "json-groups-lexical",
            "json-seq",
            "json-offsets",
            "json-subclasses",
            "json-candidate-subclasses",
        }:
            raise SystemExit(
                "--session-field currently requires --backend codex and a grouped or sequence JSON format"
            )
        plans = plan_document_session_parts(
            docs,
            args.session_field,
            max_segments=args.session_max_segments,
            max_source_chars=args.session_max_source_chars,
            paragraph_field=args.session_paragraph_field,
        )
        for doc, plan in zip(docs, plans, strict=True):
            doc["_codex_session_plan"] = plan

    pred_path = Path(args.out)
    raw_path = Path(args.raw_out or f"{args.out}.raw.jsonl")
    event_path = Path(args.session_event_out) if args.session_event_out else None
    fresh_protocol_resume = args.fresh_protocol_root_on_resume
    if args.resume:
        resume_count = require_resume_pair(pred_path, raw_path, docs, retry_policy=retry_policy)
        if args.prompt_contract_out and not fresh_protocol_resume:
            require_retry_policy_resume(
                json.loads(Path(args.prompt_contract_out).read_text()), args.model, args.effort, retry_policy
            )
        if fresh_protocol_resume:
            assert args.prompt_contract_out is not None
            assert event_path is not None
            require_fresh_protocol_resume_paths(Path(args.prompt_contract_out), event_path)
        elif event_path is not None:
            require_session_event_prefix(event_path, docs, resume_count)
        if args.resume_after_transport_failure:
            require_transport_failure_resume(raw_path, resume_count)
    else:
        output_paths = [pred_path, raw_path]
        if event_path is not None:
            output_paths.append(event_path)
        if args.codex_protocol_root and args.prompt_contract_out:
            output_paths.append(Path(args.prompt_contract_out))
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise ValueError(f"refusing to overwrite existing output without --resume: {existing}")
        resume_count = 0
    print(f"RESUME: retaining {resume_count}/{len(docs)} completed rows", flush=True)
    session_ids = (
        resumed_codex_session_parts(raw_path, docs, resume_count, args.session_field)
        if args.session_field and resume_count and not fresh_protocol_resume
        else {}
    )

    codex_workdir = Path(args.codex_workdir) if args.codex_workdir else None
    codex_home = Path(args.codex_home).expanduser() if args.codex_home else None
    codex_env: dict[str, str] | None = None
    if args.backend == "codex":
        if codex_workdir is None:
            raise SystemExit("--codex-workdir is required with --backend codex")
        if not codex_workdir.is_dir():
            raise SystemExit(f"--codex-workdir is not a directory: {codex_workdir}")
        if codex_home is None:
            raise SystemExit(
                "--codex-home is required with --backend codex; use a fresh isolated home "
                "authenticated only for this bulk run"
            )
        require_isolated_codex_home(codex_home)
        runtime_home = codex_home / "runtime-home"
        runtime_home.mkdir(exist_ok=True)
        codex_env = os.environ.copy()
        codex_env["CODEX_HOME"] = str(codex_home.resolve())
        codex_env["HOME"] = str(runtime_home.resolve())

    def render_templates(template: str) -> list[tuple[str, str]]:
        return [
            build_prompt(
                template,
                examples,
                doc.get("lang", args.lang),
                tags,
                doc["text"],
                args.fmt,
                proposals=(doc[args.proposal_field] if args.proposal_field else None),
                tag_catalog=tag_catalog,
                span_policy=args.span_policy,
                input_render=args.input_render,
                proposal_render=args.proposal_render,
                guidance=(doc[args.guidance_field] if args.guidance_field else None),
                language_rules=language_rules,
            )
            for doc in docs
        ]

    rendered_prompts = render_templates(task_template)
    protocol_root = None
    retry_prompts = None
    if retry_template is not None:
        retry_prompts = [prompt for _, prompt in render_templates(retry_template)]
    app_server_command = None
    if args.codex_protocol_root:
        if codex_workdir is None or codex_env is None:
            raise AssertionError("protocol-root mode requires initialized Codex paths")
        split_prompts = [
            (example_language, *split_codex_protocol_prompt(prompt))
            for example_language, prompt in rendered_prompts
        ]
        protocol_prefixes = {prefix for _language, prefix, _suffix in split_prompts}
        if len(protocol_prefixes) != 1:
            raise ValueError(
                "one protocol-root run requires one exact common prefix; split languages or prompt variants"
            )
        protocol_prefix = next(iter(protocol_prefixes))
        rendered_prompts = [(example_language, suffix) for example_language, _prefix, suffix in split_prompts]
        app_server_command = build_codex_app_server_command(args.codex_command)
        async with CodexAppServer(
            app_server_command,
            env=codex_env,
            request_timeout=args.timeout,
        ) as root_server:
            protocol_root = await ensure_codex_protocol_root(
                root_server,
                contract_path=Path(args.prompt_contract_out),
                resume=args.resume and not fresh_protocol_resume,
                protocol_prefix=protocol_prefix,
                workdir=codex_workdir,
                model=args.model,
                effort=args.effort,
                timeout=args.timeout,
            )

    if args.prompt_contract_out and (not args.resume or fresh_protocol_resume):
        write_prompt_contract(
            Path(args.prompt_contract_out),
            task_path,
            examples_path,
            task_template,
            examples,
            docs,
            tags,
            args.fmt,
            args.lang,
            args.model,
            url=args.url,
            effort=args.effort,
            format_retries=args.format_retries,
            format_retry_model=args.format_retry_model,
            format_retry_effort=args.format_retry_effort,
            backend=args.backend,
            codex_command=args.codex_command,
            codex_workdir=codex_workdir,
            codex_home=codex_home,
            codex_bootstrap_input_limit=args.codex_bootstrap_input_limit,
            tag_catalog=tag_catalog,
            tag_catalog_sha256=(
                None
                if args.tag_catalog is None
                else hashlib.sha256(Path(args.tag_catalog).read_bytes()).hexdigest()
            ),
            span_policy=args.span_policy,
            input_render=args.input_render,
            proposal_render=args.proposal_render,
            proposal_field=args.proposal_field,
            guidance_field=args.guidance_field,
            unicode_normalization=args.unicode_normalization,
            language_rules=language_rules,
            language_rules_path=Path(args.language_rules) if args.language_rules else None,
            session_field=args.session_field,
            session_paragraph_field=args.session_paragraph_field,
            session_max_segments=args.session_max_segments,
            session_max_source_chars=args.session_max_source_chars,
            session_context_input_limit=args.session_context_input_limit,
            session_output_token_warning=args.session_output_token_warning,
            session_allow_overlapping_spans=args.session_allow_overlapping_spans,
            session_fork_retries=args.session_fork_retries,
            session_recovery_retries=args.session_recovery_retries,
            protocol_root=protocol_root,
            subclass_spec=subclass_spec,
        )

    parser = {
        "json": parse_labels,
        "json-seq": parse_labels_seq,
        "json-offsets": parse_labels_offsets,
        "json-groups": parse_labels_grouped,
        "json-groups-lexical": parse_labels_grouped_lexical,
        "inline": parse_labels_inline,
        "redact": parse_labels_redact,
    }.get(args.fmt)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if event_path is not None:
        event_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    codex_command = (
        build_codex_command(
            args.codex_command,
            args.model,
            args.effort,
            codex_workdir,
            persistent=bool(args.session_field),
        )
        if codex_workdir is not None
        else None
    )
    parse_stats: dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    latencies: list[float] = []
    data_mode = "a" if resume_count else "w"
    event_mode = "w" if fresh_protocol_resume else data_mode
    async with AsyncExitStack() as stack:
        codex_app_server = None
        if args.backend in {"anthropic", "openai"}:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=args.timeout)
            connector = aiohttp.TCPConnector(force_close=True) if args.no_keepalive else None
            session = await stack.enter_async_context(
                aiohttp.ClientSession(timeout=timeout, connector=connector)
            )
        else:
            session = None
            if args.codex_protocol_root:
                assert app_server_command is not None
                assert codex_env is not None
                codex_app_server = await stack.enter_async_context(
                    CodexAppServer(
                        app_server_command,
                        env=codex_env,
                        request_timeout=args.timeout,
                    )
                )
        event_output = (
            stack.enter_context(event_path.open(event_mode, encoding="utf-8"))
            if event_path is not None
            else None
        )
        with (
            pred_path.open(data_mode, encoding="utf-8") as predictions,
            raw_path.open(data_mode, encoding="utf-8") as raw,
        ):
            commit_ranges = (
                paragraph_aligned_commit_ranges(
                    docs,
                    resume_count,
                    args.wave,
                    allow_transport_failure_prefix=args.resume_after_transport_failure,
                )
                if args.session_field and args.codex_protocol_root
                else [
                    (start, min(start + args.wave, len(docs)))
                    for start in range(resume_count, len(docs), args.wave)
                ]
            )
            for start, end in commit_ranges:
                chunk = docs[start:end]
                rendered = rendered_prompts[start : start + len(chunk)]
                if session is not None:
                    requests = [
                        request_api_annotation(
                            session,
                            semaphore,
                            url=args.url,
                            backend=args.backend,
                            model=args.model,
                            prompt=prompt,
                            max_tokens=args.max_new,
                            effort=args.effort,
                            transport_retries=args.retries,
                            format_retries=args.format_retries,
                            format_retry_model=args.format_retry_model,
                            format_retry_effort=args.format_retry_effort,
                            format_retry_prompt=(
                                retry_prompts[start + offset] if retry_prompts is not None else None
                            ),
                            index=start + offset,
                            doc=doc,
                            tagset=tagset,
                            fmt=args.fmt,
                            allow_overlapping_spans=args.session_allow_overlapping_spans,
                            subclass_spec=subclass_spec,
                        )
                        for offset, (doc, (_, prompt)) in enumerate(zip(chunk, rendered, strict=True))
                    ]
                    responses = list(await asyncio.gather(*requests))
                elif args.session_field:
                    assert codex_command is not None
                    assert codex_env is not None
                    document_entries: dict[tuple[str, int], list[tuple[int, str, dict[str, Any]]]] = {}
                    for offset, (doc, (_example_language, prompt)) in enumerate(
                        zip(chunk, rendered, strict=True)
                    ):
                        value = doc[args.session_field]
                        plan = doc["_codex_session_plan"]
                        key = (value, int(plan["part"]))
                        document_entries.setdefault(key, []).append((start + offset, prompt, doc))
                    if args.codex_protocol_root:
                        assert codex_app_server is not None
                        assert protocol_root is not None
                        assert codex_workdir is not None
                        document_requests = [
                            request_codex_forked_document(
                                entries,
                                codex_app_server,
                                semaphore,
                                protocol_root_thread_id=protocol_root["thread_id"],
                                command_workdir=codex_workdir,
                                model=args.model,
                                base_instructions=protocol_prefix,
                                effort=args.effort,
                                timeout=args.timeout,
                                tagset=tagset,
                                initial_session_id=session_ids.get(key),
                                recovery_retries=args.session_recovery_retries,
                                bootstrap_input_limit=args.codex_bootstrap_input_limit,
                                context_input_limit=args.session_context_input_limit,
                                output_token_warning=args.session_output_token_warning,
                                fmt=args.fmt,
                                allow_overlapping_spans=args.session_allow_overlapping_spans,
                                subclass_spec=subclass_spec,
                            )
                            for key, entries in document_entries.items()
                        ]
                    else:
                        document_requests = [
                            request_codex_document(
                                entries,
                                semaphore,
                                initial_command=codex_command,
                                command_name=args.codex_command,
                                model=args.model,
                                effort=args.effort,
                                timeout=args.timeout,
                                retries=args.retries,
                                fork_retries=args.session_fork_retries,
                                recovery_retries=args.session_recovery_retries,
                                tagset=tagset,
                                initial_session_id=session_ids.get(key),
                                env=codex_env,
                                bootstrap_input_limit=args.codex_bootstrap_input_limit,
                                context_input_limit=args.session_context_input_limit,
                                output_token_warning=args.session_output_token_warning,
                                fmt=args.fmt,
                                allow_overlapping_spans=args.session_allow_overlapping_spans,
                                subclass_spec=subclass_spec,
                            )
                            for key, entries in document_entries.items()
                        ]
                    document_results = await asyncio.gather(*document_requests)
                    responses_by_index = {}
                    for key, (results, final_session_id) in zip(
                        document_entries,
                        document_results,
                        strict=True,
                    ):
                        session_ids[key] = final_session_id
                        responses_by_index.update(results)
                    responses = [responses_by_index[start + offset] for offset in range(len(chunk))]
                else:
                    assert codex_command is not None
                    assert codex_env is not None
                    requests = [
                        request_codex_one(
                            semaphore,
                            command=codex_command,
                            prompt=prompt,
                            timeout=args.timeout,
                            retries=args.retries,
                            index=start + offset,
                            env=codex_env,
                            input_token_limit=args.codex_bootstrap_input_limit,
                        )
                        for offset, (_, prompt) in enumerate(rendered)
                    ]
                    responses = [
                        (response, latency, None) for response, latency in await asyncio.gather(*requests)
                    ]
                for row_offset, (
                    doc,
                    (example_language, prompt),
                    (response, latency, session_metadata),
                ) in enumerate(zip(chunk, rendered, responses, strict=True)):
                    text = response_text_or_empty(response)
                    alignment_repairs: list[dict[str, Any]] = []
                    label_sets: list[dict[str, Any]] = []
                    subclass_spans: list[dict[str, Any]] = []
                    candidate_decisions: list[dict[str, Any]] = []
                    if args.fmt in {"json-groups", "json-groups-lexical"}:
                        assert parser is not None
                        preds, stats = parser(
                            text,
                            doc["text"],
                            tagset,
                            alignment_repairs=alignment_repairs,
                            label_sets=label_sets,
                        )
                    elif args.fmt == "json-subclasses":
                        assert subclass_spec is not None
                        preds, subclass_spans, stats = parse_subclass_annotation(
                            text,
                            doc["text"],
                            doc["_subclass_base_spans"],
                            subclass_spec,
                        )
                    elif args.fmt == "json-candidate-subclasses":
                        assert subclass_spec is not None
                        preds, subclass_spans, candidate_decisions, stats = (
                            parse_candidate_subclass_annotation(
                                text,
                                doc["text"],
                                doc["_subclass_base_spans"],
                                doc["_subclass_candidate_ledger"],
                                subclass_spec,
                            )
                        )
                    else:
                        assert parser is not None
                        preds, stats = parser(text, doc["text"], tagset)
                    for key, value in stats.items():
                        parse_stats[key] = parse_stats.get(key, 0) + value
                    input_tokens, output_tokens = usage_counts(response)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    latencies.append(latency)
                    prediction_row = {
                        "id": doc["id"],
                        "preds": preds,
                        "parse_stats": stats,
                        "example_lang": example_language,
                        "latency_s": round(latency, 3),
                    }
                    annotation_health = response.get("annotation_health")
                    if "annotation_request" in response:
                        prediction_row["annotation_request"] = response["annotation_request"]
                    if isinstance(annotation_health, dict):
                        prediction_row["annotation_banned"] = bool(annotation_health.get("unhealthy"))
                        prediction_row["format_attempts"] = int(response["format_attempts"])
                        prediction_row["format_health_reasons"] = annotation_health.get("reasons", [])
                    if args.fmt in {"json-groups", "json-groups-lexical"}:
                        prediction_row["label_sets"] = label_sets
                        prediction_row["alignment_repairs"] = alignment_repairs
                    if args.fmt in subclass_formats:
                        prediction_row["subclass_spans"] = subclass_spans
                    if args.fmt == "json-candidate-subclasses":
                        prediction_row["candidate_decisions"] = candidate_decisions
                    if session_metadata is not None and "annotation_banned" in session_metadata:
                        prediction_row["annotation_banned"] = session_metadata["annotation_banned"]
                    predictions.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")
                    raw_row = {
                        "id": doc["id"],
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "deduplication_receipt": deduplication,
                        "reannotation_receipt": reannotation,
                        "evaluation_replay_admission": evaluation_replay,
                        "web_intake_receipt": web_intake,
                        "response": response,
                    }
                    if retry_prompts is not None:
                        raw_row["format_retry_prompt"] = retry_prompts[start + row_offset]
                    if session_metadata is not None:
                        plan = doc["_codex_session_plan"]
                        raw_row["session"] = {
                            "field": args.session_field,
                            "value": doc[args.session_field],
                            **plan,
                            **session_metadata,
                        }
                    raw.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                    if event_output is not None:
                        if session_metadata is None:
                            raise AssertionError("session event output requires session metadata")
                        for event_row in codex_attempt_event_rows(
                            index=start + row_offset,
                            doc=doc,
                            response=response,
                            session_metadata=session_metadata,
                            task_template_path=task_path,
                            language=args.lang,
                            domain_field=args.session_domain_field,
                        ):
                            event_output.write(json.dumps(event_row, ensure_ascii=False) + "\n")
                predictions.flush()
                raw.flush()
                if event_output is not None:
                    event_output.flush()
                completed = start + len(chunk)
                headline = (
                    f"{args.backend}-label {completed}/{len(docs)} parse-stats {parse_stats} "
                    f"tokens in={total_input_tokens} out={total_output_tokens}"
                )
                print(headline, flush=True)
                if headline_path := os.environ.get("AGENTCTL_HEADLINE_FILE"):
                    Path(headline_path).write_text(headline + "\n", encoding="utf-8")

    latencies.sort()
    print(f"DONE {len(docs)} docs -> {pred_path}; parse-stats {parse_stats}", flush=True)
    print(f"TOKENS input={total_input_tokens} output={total_output_tokens}", flush=True)
    if latencies:
        print(
            f"LATENCY per-doc s: mean {sum(latencies) / len(latencies):.2f} "
            f"p50 {latencies[len(latencies) // 2]:.2f} "
            f"p90 {latencies[int(0.9 * len(latencies))]:.2f} max {latencies[-1]:.2f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("anthropic", "openai", "codex"),
        default="anthropic",
    )
    parser.add_argument("--url", default="http://127.0.0.1:4141/v1/messages")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-workdir", default=None)
    parser.add_argument(
        "--codex-home",
        default=None,
        help=(
            "fresh isolated CODEX_HOME for bulk labeling; the child also receives an empty "
            "HOME below this directory so ~/.agents skills cannot enter the prompt"
        ),
    )
    parser.add_argument(
        "--codex-bootstrap-input-limit",
        type=int,
        default=20_000,
        help=(
            "fail a fresh Codex session above this input-token count before scaling; "
            "0 disables the guard, while document-session resumes are exempt"
        ),
    )
    parser.add_argument(
        "--codex-protocol-root",
        action="store_true",
        help=(
            "place the task prefix before the session-turn marker in an immutable app-server "
            "thread and explicitly fork it before every document turn"
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold", required=True, help="document JSONL; gold spans are ignored")
    intake = parser.add_mutually_exclusive_group(required=True)
    intake.add_argument(
        "--dedup-receipt",
        help="current partial-overlap receipt from pii_overlap_filter.py admit-annotation; --gold must be its retained file",
    )
    intake.add_argument(
        "--web-receipt",
        help="pii_fetch_web.py receipt: every row is O4's own hash-verified public web training text",
    )
    intake.add_argument(
        "--reannotation-receipt",
        help="paired training-reannotation receipt binding exact previously annotated text and shared row weight; grants no novelty or split admission",
    )
    intake.add_argument(
        "--verification-replay",
        action="store_true",
        help="skip admission checks to re-run rows a published result already annotated, recording "
        "the bypass on every raw row; outputs are not training data",
    )
    intake.add_argument(
        "--evaluation-replay-admission",
        help="existing Ont3 context admission; replay its exact evaluation file without new admission or training use",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--prompt-contract-out", default=None)
    parser.add_argument(
        "--task-template",
        default=None,
        help="task-template path; defaults to prompts/pii-label/task.txt",
    )
    parser.add_argument(
        "--examples",
        default=None,
        help="language-example JSON path; defaults to prompts/pii-label/examples.json",
    )
    parser.add_argument(
        "--allow-example-fallback",
        action="store_true",
        help="allow cross-language (normally English) shots only for historical replay",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--tags", default=None)
    parser.add_argument(
        "--tag-catalog",
        default=None,
        help="Markdown table of tag/definition/example rows to carry into the prompt",
    )
    parser.add_argument(
        "--language-rules",
        default=None,
        help=(
            "JSON object {language: [rule, ...]} rendered at the task template's "
            "{language_rules} placeholder for rows of that language only"
        ),
    )
    parser.add_argument(
        "--subclass-spec",
        default=None,
        help=(
            "closed subclass-family inventory used to render and validate "
            "--fmt json-subclasses or json-candidate-subclasses"
        ),
    )
    parser.add_argument("--span-policy", choices=SPAN_POLICIES, default="legacy")
    parser.add_argument("--fmt", choices=FORMATS, default="json-seq")
    parser.add_argument("--input-render", choices=("plain", "json-string"), default="plain")
    parser.add_argument("--proposal-render", choices=("markup", "list"), default="markup")
    parser.add_argument(
        "--proposal-field",
        default=None,
        help="JSONL field containing controller-validated span candidates for the prompt",
    )
    parser.add_argument(
        "--guidance-field",
        default=None,
        help="JSONL field containing non-source controller review guidance for the prompt",
    )
    parser.add_argument(
        "--source-paragraph-guidance",
        action="store_true",
        help="render the paper's ctx-v6 guidance: each row's annotation_guidance, else its "
        "document_context before/after text joined around the sentence as source paragraph",
    )
    parser.add_argument(
        "--unicode-normalization",
        choices=("none", "NFKC"),
        default="none",
        help="require intake text to already use this canonical Unicode coordinate system",
    )
    parser.add_argument(
        "--session-field",
        default=None,
        help=(
            "input metadata field grouping segments into persistent Codex conversations; "
            "use document_id for one session per intake document"
        ),
    )
    parser.add_argument(
        "--session-paragraph-field",
        default=None,
        help="optional input field identifying paragraph membership within each session document",
    )
    parser.add_argument(
        "--session-domain-field",
        default=None,
        help="optional input field copied into the flat session-attempt analysis table",
    )
    parser.add_argument(
        "--session-event-out",
        default=None,
        help="JSONL attempt/health/length table required by --codex-protocol-root",
    )
    parser.add_argument(
        "--session-max-segments",
        type=int,
        default=1,
        help=(
            "absolute maximum input segments in one persistent thread; the planner prefers "
            "a paragraph-aligned edge and otherwise cuts at this boundary; the safe default "
            "disables cross-segment persistence until a long-document pilot freezes a ceiling"
        ),
    )
    parser.add_argument(
        "--session-max-source-chars",
        type=int,
        default=2400,
        help="absolute source-character maximum in one persistent thread part",
    )
    parser.add_argument(
        "--session-context-input-limit",
        type=int,
        default=0,
        help=(
            "restart after a response reaches this measured context input-token count; "
            "0 leaves the preplanned segment/character limits as the absolute bounds"
        ),
    )
    parser.add_argument(
        "--session-output-token-warning",
        type=int,
        default=0,
        help=(
            "treat a completion at or above this visible output-token count (hidden reasoning "
            "excluded) as unhealthy, retry it, and refuse to continue from it; 0 disables "
            "this additional health signal"
        ),
    )
    parser.add_argument(
        "--session-allow-overlapping-spans",
        "--allow-overlapping-spans",
        action="store_true",
        help=(
            "allow intentional overlap in JSON occurrence/sequence/offset output while still "
            "rejecting duplicate typed spans and parse defects"
        ),
    )
    parser.add_argument(
        "--session-fork-retries",
        type=int,
        default=1,
        help="same-pre-segment-context sibling retries for an unhealthy continued turn",
    )
    parser.add_argument(
        "--session-recovery-retries",
        type=int,
        default=1,
        help="fresh-session retries after an unhealthy same-context sibling",
    )
    parser.add_argument("--effort", choices=EFFORT_LEVELS, default="low")
    parser.add_argument("--max-new", type=int, default=1536)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--wave", type=int, default=24, help="ordered commit group size")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--no-keepalive",
        action="store_true",
        help=(
            "close the HTTP connection after every request instead of pooling it; for a server "
            "that drops idle keep-alive connections between waves (vLLM behind an ssh tunnel), "
            "where every pooled connection is dead by the next wave and each retry burns one"
        ),
    )
    parser.add_argument(
        "--format-retries",
        type=int,
        default=0,
        help="fresh stateless API retries after structurally unhealthy parsed output",
    )
    parser.add_argument("--format-retry-model", help="model for the single fresh format/cap retry")
    parser.add_argument(
        "--format-retry-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max")
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fresh-protocol-root-on-resume",
        action="store_true",
        help=(
            "retain the prediction/raw row prefix but create a new Codex protocol root and new "
            "prompt-contract/session-event ledgers in the current isolated home"
        ),
    )
    parser.add_argument(
        "--resume-after-transport-failure",
        action="store_true",
        help=(
            "allow a verified transcript-recovered prefix to resume inside its planned paragraph; "
            "ordinary mid-paragraph resume remains rejected"
        ),
    )
    args = parser.parse_args()
    if args.wave <= 0:
        parser.error("--wave must be positive")
    if args.session_fork_retries < 0 or args.session_recovery_retries < 0:
        parser.error("session fork and fresh-recovery retry counts must be nonnegative")
    if args.retries < 0 or args.format_retries < 0:
        parser.error("transport and format retry counts must be nonnegative")
    if args.session_max_segments <= 0 or args.session_max_source_chars <= 0:
        parser.error("session segment and source-character limits must be positive")
    if args.session_context_input_limit < 0 or args.session_output_token_warning < 0:
        parser.error("session token limits must be nonnegative")
    if args.codex_bootstrap_input_limit < 0:
        parser.error("--codex-bootstrap-input-limit must be nonnegative")
    if args.codex_protocol_root:
        if args.backend != "codex" or not args.session_field:
            parser.error("--codex-protocol-root requires --backend codex and --session-field")
        if not args.prompt_contract_out or not args.session_event_out:
            parser.error("--codex-protocol-root requires --prompt-contract-out and --session-event-out")
        if args.session_fork_retries != 0 or args.session_recovery_retries not in {0, 1}:
            parser.error(
                "--codex-protocol-root requires no same-context fork and at most one clean retry: "
                "--session-fork-retries 0 --session-recovery-retries {0,1}"
            )
    elif args.session_event_out:
        parser.error("--session-event-out requires --codex-protocol-root")
    if args.resume_after_transport_failure and not (args.resume and args.codex_protocol_root):
        parser.error("--resume-after-transport-failure requires --resume and --codex-protocol-root")
    if args.fresh_protocol_root_on_resume and not (args.resume and args.codex_protocol_root):
        parser.error("--fresh-protocol-root-on-resume requires --resume and --codex-protocol-root")
    if args.session_paragraph_field and not args.session_field:
        parser.error("--session-paragraph-field requires --session-field")
    if args.session_domain_field and not args.session_event_out:
        parser.error("--session-domain-field requires --session-event-out")
    if args.session_allow_overlapping_spans and (
        args.fmt not in {"json", "json-seq", "json-offsets"}
        or (args.backend == "codex" and args.session_field is None)
    ):
        parser.error(
            "--allow-overlapping-spans requires an occurrence/sequence/offset JSON format; Codex also requires --session-field"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
