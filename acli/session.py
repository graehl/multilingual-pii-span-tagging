from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from enum import Enum
from typing import Any, TextIO


class Format(str, Enum):
    COMPACT = "compact"
    PRETTY = "pretty"
    TOON = "toon"
    TEXT = "text"


_AGENT_ENV_NAMES = (
    "AGENTCTL_SESSION_ID",
    "AGENT_GUARD",
    "CLAUDECODE",
    "CI",
    "CODEX_THREAD_ID",
    "PI_CODING_AGENT",
)
_AGENT_ENV_PREFIXES = ("CLAUDE_CODE_",)


def _env_value_set(env: Mapping[str, str], name: str) -> bool:
    return bool(str(env.get(name, "")).strip())


def is_agent_session(
    stdout: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return True when the caller should receive compact machine output."""
    out = sys.stdout if stdout is None else stdout
    environ = os.environ if env is None else env
    isatty = getattr(out, "isatty", None)
    if not callable(isatty) or not isatty():
        return True
    if any(_env_value_set(environ, name) for name in _AGENT_ENV_NAMES):
        return True
    return any(
        name.startswith(prefix) and _env_value_set(environ, name)
        for name in environ
        for prefix in _AGENT_ENV_PREFIXES
    )


def _normalize_format(raw: Any) -> Format:
    if isinstance(raw, Format):
        return raw
    text = str(raw).strip().lower()
    if text in {"compact", "jsonl"}:
        return Format.COMPACT
    if text == "pretty":
        return Format.PRETTY
    if text == "toon":
        return Format.TOON
    if text == "text":
        return Format.TEXT
    raise ValueError(f"unknown output format {raw!r}")


def resolve_format(
    args: Any,
    stdout: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> Format:
    """Resolve explicit flags, then fall back to agent-session detection."""
    raw = getattr(args, "format", None)
    if raw:
        fmt = _normalize_format(raw)
    elif getattr(args, "acli_text", False):
        fmt = Format.TEXT
    else:
        fmt = Format.COMPACT if is_agent_session(stdout, env) else Format.PRETTY
    if fmt is Format.TOON and not bool(getattr(args, "acli_toon_allowed", False)):
        raise ValueError("--toon is only valid for table-producing subcommands")
    return fmt
