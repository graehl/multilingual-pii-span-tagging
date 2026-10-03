"""Load the shared Codex annotation transport from the user's agents checkout."""

import sys
from pathlib import Path

_SHARED_ROOT = Path.home() / "agents"
if not (_SHARED_ROOT / "document_annotation" / "codex.py").is_file():
    raise ImportError(
        "PII Codex annotation requires ~/agents/document_annotation/codex.py; "
        "install or transfer the shared agents checkout before running it"
    )
sys.path.insert(0, str(_SHARED_ROOT))

from document_annotation.codex import (  # noqa: E402
    APP_SERVER_JSONL_LIMIT_BYTES,
    TERMINAL_TURN_STATUSES,
    TURN_INTERRUPT_TIMEOUT_SECONDS,
    CodexAppServer,
    CodexAppServerError,
    app_server_response,
    app_server_usage,
)

__all__ = [
    "APP_SERVER_JSONL_LIMIT_BYTES",
    "TERMINAL_TURN_STATUSES",
    "TURN_INTERRUPT_TIMEOUT_SECONDS",
    "CodexAppServer",
    "CodexAppServerError",
    "app_server_response",
    "app_server_usage",
]
