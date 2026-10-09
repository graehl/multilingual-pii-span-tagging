import asyncio
import sys

import pytest

from scripts.pii_codex_app_server import (
    CodexAppServer,
    CodexAppServerError,
    app_server_response,
)

FAKE_APP_SERVER = r"""
import json
import sys

threads = {}
active_turns = {}
missing_interrupt_acks = set()
missing_interrupt_completions = set()
uninterruptible_turns = set()
next_thread = 0
next_turn = 0
experimental_api = False

def send(message):
    print(json.dumps(message), flush=True)

for line in sys.stdin:
    message = json.loads(line)
    method = message["method"]
    request_id = message.get("id")
    params = message.get("params", {})
    if method == "initialize":
        experimental_api = params.get("capabilities", {}).get("experimentalApi", False)
        send({"id": request_id, "result": {"userAgent": "fake"}})
    elif method == "initialized":
        pass
    elif method == "thread/start":
        next_thread += 1
        thread_id = f"root-{next_thread}"
        thread = {
            "id": thread_id,
            "instructionSources": [],
            "turns": [],
            "baseInstructions": params["baseInstructions"],
        }
        threads[thread_id] = thread
        send({"id": request_id, "result": {"thread": thread}})
    elif method == "thread/read":
        send({"id": request_id, "result": {"thread": threads[params["threadId"]]}})
    elif method == "thread/fork":
        next_thread += 1
        source = params["threadId"]
        if not params.get("excludeTurns"):
            send({"id": request_id, "error": {"code": -32601, "message": "list_turns is not supported yet"}})
            continue
        if params.get("beforeTurnId") is not None and not experimental_api:
            send({"id": request_id, "error": {"code": -32600, "message": "experimentalApi required"}})
            continue
        if not threads[source].get("has_rollout"):
            send({"id": request_id, "error": {"code": -32600, "message": "no rollout found"}})
            continue
        thread_id = f"fork-{next_thread}"
        thread = dict(threads[source])
        turns = list(thread["turns"])
        if params.get("beforeTurnId") is not None:
            edge = next(
                index for index, turn in enumerate(turns) if turn["id"] == params["beforeTurnId"]
            )
            turns = turns[:edge]
        if params.get("baseInstructions") is not None:
            thread["baseInstructions"] = params["baseInstructions"]
        thread.update({
            "id": thread_id,
            "forkedFromId": source,
            "has_rollout": True,
            "instructionSources": [],
            "turns": turns,
        })
        threads[thread_id] = thread
        response_thread = dict(thread)
        response_thread["turns"] = []
        send({"id": request_id, "result": {"thread": response_thread}})
    elif method == "turn/start":
        next_turn += 1
        thread_id = params["threadId"]
        turn_id = f"turn-{next_turn}"
        threads[thread_id]["has_rollout"] = True
        turn = {"id": turn_id, "status": "inProgress"}
        threads[thread_id]["turns"].append(turn)
        active_turns[turn_id] = thread_id
        send({"id": request_id, "result": {"turn": {"id": turn_id, "status": "inProgress"}}})
        prompt = params["input"][0]["text"]
        if prompt == "stall-no-interrupt-ack":
            missing_interrupt_acks.add(turn_id)
            continue
        if prompt == "stall-no-interrupt-completion":
            missing_interrupt_completions.add(turn_id)
            continue
        if prompt == "stall-active":
            missing_interrupt_acks.add(turn_id)
            uninterruptible_turns.add(turn_id)
            continue
        if prompt == "stall":
            continue
        send({
            "method": "item/completed",
            "params": {
                "threadId": thread_id,
                "turnId": turn_id,
                "completedAtMs": 1,
                "item": {"id": "message-1", "type": "agentMessage", "text": "[]"},
            },
        })
        send({
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": thread_id,
                "turnId": turn_id,
                "tokenUsage": {
                    "last": {
                        "inputTokens": 120,
                        "cachedInputTokens": 90,
                        "cacheWriteInputTokens": 4,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 2,
                        "totalTokens": 125,
                    },
                    "total": {
                        "inputTokens": 120,
                        "cachedInputTokens": 90,
                        "cacheWriteInputTokens": 4,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 2,
                        "totalTokens": 125,
                    },
                },
            },
        })
        send({
            "method": "turn/completed",
            "params": {
                "threadId": thread_id,
                "turn": {
                    "id": turn_id,
                    "status": "completed",
                    "items": [],
                },
            },
        })
        turn["status"] = "completed"
        active_turns.pop(turn_id)
    elif method == "turn/interrupt":
        turn_id = params["turnId"]
        thread_id = active_turns[turn_id]
        turn = next(turn for turn in threads[thread_id]["turns"] if turn["id"] == turn_id)
        if turn_id in missing_interrupt_acks:
            if turn_id not in uninterruptible_turns:
                turn["status"] = "interrupted"
                active_turns.pop(turn_id)
            continue
        active_turns.pop(turn_id)
        turn["status"] = "interrupted"
        send({"id": request_id, "result": {}})
        if turn_id in missing_interrupt_completions:
            continue
        send({
            "method": "turn/completed",
            "params": {
                "threadId": thread_id,
                "turn": {
                    "id": turn_id,
                    "status": "interrupted",
                    "items": [],
                },
            },
        })
"""


def test_app_server_materializes_zero_turn_root_then_forks_and_runs_turn():
    async def exercise():
        async with CodexAppServer(
            [sys.executable, "-u", "-c", FAKE_APP_SERVER],
            env={},
            request_timeout=2,
        ) as server:
            root = await server.start_protocol_root(
                base_instructions="common protocol",
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            assert root["turns"] == []
            materialization = await server.run_turn(
                root["id"],
                "synthetic empty input",
                cwd="/tmp",
                model="gpt-5.6-luna",
                effort="low",
                timeout=2,
            )
            anchor = await server.fork_thread(
                root["id"],
                cwd="/tmp",
                model="gpt-5.6-luna",
                before_turn_id=materialization["app_server"]["turn_id"],
                base_instructions="common protocol",
            )
            assert anchor["turns"] == []
            assert anchor["baseInstructions"] == "common protocol"
            fork = await server.fork_thread(
                anchor["id"],
                cwd="/tmp",
                model="gpt-5.6-luna",
                base_instructions="common protocol",
            )
            response = await server.run_turn(
                fork["id"],
                "row-specific suffix",
                cwd="/tmp",
                model="gpt-5.6-luna",
                effort="low",
                timeout=2,
            )
            return root, anchor, fork, response

    root, anchor, fork, response = asyncio.run(exercise())

    assert anchor["forkedFromId"] == root["id"]
    assert fork["forkedFromId"] == anchor["id"]
    assert fork["baseInstructions"] == "common protocol"
    assert response["content"] == [{"type": "text", "text": "[]"}]
    assert response["usage"] == {
        "input_tokens": 120,
        "cached_input_tokens": 90,
        "cache_write_input_tokens": 4,
        "output_tokens": 3,
        "reasoning_output_tokens": 2,
    }
    assert response["app_server"]["thread_id"] == fork["id"]


def test_app_server_accepts_large_bounded_jsonl_response():
    large_response_server = FAKE_APP_SERVER.replace(
        '"turns": turns,',
        '"turns": turns, "copiedHistory": "x" * 70000,',
    )

    async def exercise():
        async with CodexAppServer(
            [sys.executable, "-u", "-c", large_response_server],
            env={},
            request_timeout=2,
        ) as server:
            root = await server.start_protocol_root(
                base_instructions="common protocol",
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            materialization = await server.run_turn(
                root["id"],
                "synthetic empty input",
                cwd="/tmp",
                model="gpt-5.6-luna",
                effort="low",
                timeout=2,
            )
            fork = await server.fork_thread(
                root["id"],
                cwd="/tmp",
                model="gpt-5.6-luna",
                before_turn_id=materialization["app_server"]["turn_id"],
            )
            return fork

    fork = asyncio.run(exercise())

    assert len(fork["copiedHistory"]) == 70000


def test_timed_out_turn_is_interrupted_before_a_fresh_fork_runs():
    async def exercise():
        async with CodexAppServer(
            [sys.executable, "-u", "-c", FAKE_APP_SERVER],
            env={},
            request_timeout=2,
        ) as server:
            root = await server.start_protocol_root(
                base_instructions="common protocol",
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            materialization = await server.run_turn(
                root["id"],
                "synthetic empty input",
                cwd="/tmp",
                model="gpt-5.6-luna",
                effort="low",
                timeout=2,
            )
            anchor = await server.fork_thread(
                root["id"],
                cwd="/tmp",
                model="gpt-5.6-luna",
                before_turn_id=materialization["app_server"]["turn_id"],
            )
            stalled = await server.fork_thread(
                anchor["id"],
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            with pytest.raises(asyncio.TimeoutError, match="was interrupted"):
                await server.run_turn(
                    stalled["id"],
                    "stall",
                    cwd="/tmp",
                    model="gpt-5.6-luna",
                    effort="low",
                    timeout=0.01,
                )
            retry = await server.fork_thread(
                anchor["id"],
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            return await server.run_turn(
                retry["id"],
                "healthy retry",
                cwd="/tmp",
                model="gpt-5.6-luna",
                effort="low",
                timeout=2,
            )

    response = asyncio.run(exercise())

    assert response["content"] == [{"type": "text", "text": "[]"}]


@pytest.mark.parametrize(
    "prompt",
    ["stall-no-interrupt-ack", "stall-no-interrupt-completion"],
)
def test_timed_out_turn_accepts_thread_read_terminal_confirmation(prompt):
    async def exercise():
        # Requests the fake server answers must not race host load; the unanswered
        # interrupt in the no-ack case costs this timeout once.
        async with CodexAppServer(
            [sys.executable, "-u", "-c", FAKE_APP_SERVER],
            env={},
            request_timeout=2,
        ) as server:
            root = await server.start_protocol_root(
                base_instructions="common protocol",
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            with pytest.raises(asyncio.TimeoutError, match="was interrupted"):
                await server.run_turn(
                    root["id"],
                    prompt,
                    cwd="/tmp",
                    model="gpt-5.6-luna",
                    effort="low",
                    timeout=0.01,
                )

    asyncio.run(exercise())


def test_timed_out_turn_rejects_thread_read_active_status():
    async def exercise():
        async with CodexAppServer(
            [sys.executable, "-u", "-c", FAKE_APP_SERVER],
            env={},
            request_timeout=2,
        ) as server:
            root = await server.start_protocol_root(
                base_instructions="common protocol",
                cwd="/tmp",
                model="gpt-5.6-luna",
            )
            with pytest.raises(CodexAppServerError, match="could not be interrupted"):
                await server.run_turn(
                    root["id"],
                    "stall-active",
                    cwd="/tmp",
                    model="gpt-5.6-luna",
                    effort="low",
                    timeout=0.01,
                )

    asyncio.run(exercise())


def test_app_server_response_rejects_noncompleted_turn():
    events = [
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread",
                "turn": {"id": "turn", "status": "failed", "error": {"message": "bad"}},
            },
        }
    ]

    with pytest.raises(CodexAppServerError, match="status 'failed'"):
        app_server_response(events, thread_id="thread", turn_id="turn", stderr="")
