from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Iterable
from typing import Any, TextIO

from .commentary import prepare_commentary
from .session import Format

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_NUMBERISH_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_DELIMITERS = {",", "\t", "|"}


def write_jsonl(
    value: Any, out: TextIO = sys.stdout, *, commentary: bool = True
) -> None:
    """Write compact JSON Lines.

    A nonempty list/tuple is treated as rows; an empty one emits [].
    Any other value is one JSONL record.
    """
    rows = value if isinstance(value, (list, tuple)) and value else [value]
    for row in rows:
        prepared, found = prepare_commentary(row, include=commentary)
        if not commentary and isinstance(row, dict) and set(row) == {"_acli"}:
            continue
        # Member order is the tool's to choose (commentary keeps it by
        # contract), so no record is re-sorted, with or without commentary.
        out.write(json.dumps(prepared, separators=(",", ":"), allow_nan=False) + "\n")
        if found:
            out.flush()


def write_pretty(
    value: Any, out: TextIO = sys.stdout, *, commentary: bool = True
) -> None:
    """Write human fallback JSON."""
    prepared, found = prepare_commentary(value, include=commentary)
    encoded = json.dumps(prepared, indent=2, allow_nan=False)
    out.write(encoded + "\n")
    if found:
        out.flush()


def _validate_key(text: str, kind: str) -> None:
    if not _KEY_RE.match(text):
        raise ValueError(f"TOON {kind} must match {_KEY_RE.pattern}: {text!r}")


def _escape_string(text: str) -> str:
    pieces: list[str] = []
    for ch in text:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            raise ValueError("TOON strings must not contain lone surrogates")
        if ch == "\\":
            pieces.append("\\\\")
        elif ch == '"':
            pieces.append('\\"')
        elif ch == "\n":
            pieces.append("\\n")
        elif ch == "\r":
            pieces.append("\\r")
        elif ch == "\t":
            pieces.append("\\t")
        elif ord(ch) < 0x20:
            pieces.append(f"\\u{ord(ch):04x}")
        else:
            pieces.append(ch)
    return '"' + "".join(pieces) + '"'


def _must_quote(text: str, delimiter: str) -> bool:
    if text == "" or text != text.strip():
        return True
    if text in {"true", "false", "null"}:
        return True
    if _NUMBERISH_RE.match(text):
        return True
    if text.startswith("-"):
        return True
    if delimiter and delimiter in text:
        return True
    return any(ch in text for ch in ':"\\[]{}') or any(ord(ch) < 0x20 for ch in text)


def _toon_scalar(value: Any, delimiter: str) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        raise TypeError("TOON flat table values must be scalars")
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOON flat table floats must be finite")
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    text = str(value)
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        raise ValueError("TOON strings must not contain lone surrogates")
    return _escape_string(text) if _must_quote(text, delimiter) else text


def write_toon_table(
    rows: Iterable[dict[str, Any]],
    columns: Iterable[str] | None = None,
    out: TextIO = sys.stdout,
    *,
    name: str = "items",
    delimiter: str = ",",
) -> None:
    """Write the flat uniform TOON table subset.

    The subset is intentionally narrow: one array of scalar-valued objects
    with a fixed column set. Nested or non-uniform input raises instead of
    silently falling back to a broader format.
    """
    if delimiter not in _DELIMITERS:
        raise ValueError("TOON delimiter must be comma, tab, or pipe")
    _validate_key(name, "table name")
    materialized = list(rows)
    if columns is None:
        if not materialized:
            raise ValueError("TOON columns are required for an empty table")
        ordered_columns = list(materialized[0].keys())
    else:
        ordered_columns = [str(col) for col in columns]
    for col in ordered_columns:
        _validate_key(col, "column")
    expected = set(ordered_columns)
    for row in materialized:
        if not isinstance(row, dict):
            raise TypeError("TOON table rows must be dictionaries")
        if set(row.keys()) != expected:
            raise ValueError("TOON table rows must have a uniform column set")

    delim_marker = "" if delimiter == "," else delimiter
    header_fields = delimiter.join(ordered_columns)
    out.write(f"{name}[{len(materialized)}{delim_marker}]{{{header_fields}}}:\n")
    for row in materialized:
        values = [_toon_scalar(row[col], delimiter) for col in ordered_columns]
        out.write(f"  {delimiter.join(values)}\n")


def emit_table(
    rows: list[dict[str, Any]],
    columns: Iterable[str],
    fmt: Format | str,
    out: TextIO = sys.stdout,
    *,
    name: str,
) -> None:
    """Emit result rows that share a declared column set.

    Rows may carry only the keys their own outcome produced (a failure has
    `reason`, not `method`); TOON is a uniform table by contract, so each
    row is projected onto `columns` there. In the line formats zero rows is
    ambiguous with no output, so an empty table becomes the definitive
    empty record `{"count": 0, "of": <name>}` (topics/acli.md).
    """
    ordered_columns = [str(col) for col in columns]
    resolved = _resolve(fmt)
    if resolved is Format.TOON:
        table = [{col: row.get(col, "") for col in ordered_columns} for row in rows]
        emit({"rows": table, "columns": ordered_columns, "name": name}, resolved, out)
    elif not rows and resolved in {Format.COMPACT, Format.TEXT}:
        write_jsonl({"count": 0, "of": name}, out)
    else:
        emit(rows, resolved, out)


def _resolve(fmt: Format | str) -> Format:
    if isinstance(fmt, Format):
        return fmt
    if str(fmt).strip().lower() == "jsonl":
        return Format.COMPACT
    return Format(fmt)


def emit(
    value: Any,
    fmt: Format | str,
    out: TextIO = sys.stdout,
    *,
    text: str | None = None,
    commentary: bool = True,
) -> None:
    resolved = _resolve(fmt)
    if resolved is Format.COMPACT:
        write_jsonl(value, out, commentary=commentary)
        return
    if resolved is Format.TEXT and text is None:
        # The text preference without a renderer falls back to JSONL.
        write_jsonl(value, out, commentary=commentary)
        return
    if resolved is Format.PRETTY:
        write_pretty(value, out, commentary=commentary)
        return
    prepared, found = prepare_commentary(value, include=False)
    if commentary and found:
        raise ValueError(
            "commentary requires JSON/JSONL; use --json, --pretty, or --no-commentary"
        )
    value = prepared
    if resolved is Format.TEXT and text is not None:
        out.write(text.rstrip("\n") + "\n")
    elif resolved is Format.TOON:
        if isinstance(value, dict) and "rows" in value:
            write_toon_table(
                value["rows"],
                value.get("columns"),
                out,
                name=value.get("name", "items"),
                delimiter=value.get("delimiter", ","),
            )
        else:
            write_toon_table(value, out=out)
    else:
        raise ValueError(f"unknown output format {fmt!r}")
