from __future__ import annotations

from typing import Any


def _items(metadata: Any) -> list[dict[str, str]]:
    if not isinstance(metadata, dict) or set(metadata) != {"commentary"}:
        raise ValueError("_acli must contain only a commentary array")
    items = metadata["commentary"]
    if not isinstance(items, list) or not items:
        raise ValueError("_acli.commentary must be a nonempty array")
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"text"}
            or not isinstance(item["text"], str)
            or not item["text"].strip()
        ):
            raise ValueError("each commentary item must contain nonempty text")
    return items


def commentary(*texts: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return commentary on an object, or a standalone commentary record."""
    added = _items({"commentary": [{"text": text} for text in texts]})
    if value is not None and not isinstance(value, dict):
        raise TypeError("commentary value must be a JSON object")
    result = dict(value) if value is not None else {}
    previous = _items(result["_acli"]) if "_acli" in result else []
    result["_acli"] = {"commentary": [*previous, *added]}
    return result


def prepare_commentary(value: Any, *, include: bool) -> tuple[Any, bool]:
    """Validate reserved metadata and optionally remove it without mutation."""
    found = False
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key == "_acli":
                _items(child)
                found = True
                if include:
                    result[key] = child
            else:
                result[key], nested = prepare_commentary(child, include=include)
                found |= nested
        return result, found
    if isinstance(value, (list, tuple)):
        rows = []
        for child in value:
            row, nested = prepare_commentary(child, include=include)
            rows.append(row)
            found |= nested
        return rows, found
    return value, False
