from __future__ import annotations

import json
import sys
from enum import IntEnum
from typing import Any, NoReturn, TextIO


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    DATA = 3
    NOT_FOUND = 4
    CONFLICT = 5
    UNAVAILABLE = 69
    SOFTWARE = 70
    TEMPORARY = 75


def error_envelope(
    message: str,
    code: ExitCode | int = ExitCode.USAGE,
    *,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exit_code = int(code)
    try:
        name = ExitCode(exit_code).name.lower()
    except ValueError:
        name = "error"
    error = {"code": name, "message": message}
    if detail:
        error["detail"] = detail
    return {"ok": False, "exit_code": exit_code, "error": error}


def die(
    message: str,
    code: ExitCode | int = ExitCode.USAGE,
    *,
    detail: dict[str, Any] | None = None,
    out: TextIO = sys.stderr,
) -> NoReturn:
    print(
        json.dumps(
            error_envelope(message, code, detail=detail),
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=out,
    )
    raise SystemExit(int(code))
