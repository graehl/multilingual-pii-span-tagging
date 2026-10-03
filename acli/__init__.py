"""Agent-first CLI helpers.

Small pure-function surface for CLIs that follow topics/acli-spec.md.
"""

from __future__ import annotations

from .args import (
    add_standard_args,
    argument_parser,
    candidates,
    capability_line,
    complete,
    hint,
    line_commentary,
    maybe_banner,
    maybe_complete,
    maybe_repl,
    set_completer,
)
from .commentary import commentary
from .emit import emit, emit_table, write_jsonl, write_pretty, write_toon_table
from .errors import ExitCode, die, error_envelope
from .session import Format, is_agent_session, resolve_format

__all__ = [
    "ExitCode",
    "Format",
    "add_standard_args",
    "argument_parser",
    "candidates",
    "capability_line",
    "commentary",
    "complete",
    "die",
    "emit",
    "emit_table",
    "error_envelope",
    "hint",
    "is_agent_session",
    "line_commentary",
    "maybe_banner",
    "maybe_complete",
    "maybe_repl",
    "resolve_format",
    "set_completer",
    "write_jsonl",
    "write_pretty",
    "write_toon_table",
]
