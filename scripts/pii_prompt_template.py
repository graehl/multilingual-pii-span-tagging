"""Assemble shared annotation templates with explicit model-conditional includes."""

from __future__ import annotations

import fnmatch
import hashlib
import re
import shlex
from pathlib import Path

_INCLUDE = re.compile(r"\{\{(include(?:-model)?)\s+([^{}]+)\}\}")


def load_prompt_template(path: Path, model: str) -> tuple[str, dict | None]:
    """Expand template assets before inserting any document or example text."""
    files: dict[str, str] = {}
    includes = []

    def expand(source: Path, ancestors: tuple[Path, ...]) -> str:
        source = source.resolve()
        if source in ancestors:
            raise ValueError(f"Cyclic prompt include: {source}")
        raw = source.read_bytes()
        files[str(source)] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

        def include(match: re.Match[str]) -> str:
            kind, arguments = match.groups()
            words = shlex.split(arguments)
            conditional = kind == "include-model"
            if len(words) != (2 if conditional else 1):
                raise ValueError(f"Invalid prompt include in {source}: {match.group()}")
            pattern = words[0] if conditional else None
            if conditional and not model:
                raise ValueError("Model-conditional prompt requires an explicit model identity")
            selected = pattern is None or fnmatch.fnmatchcase(model.casefold(), pattern.casefold())
            target = (source.parent / words[-1]).resolve()
            includes.append(
                {"parent": str(source), "path": str(target), "model_pattern": pattern, "selected": selected}
            )
            return expand(target, (*ancestors, source)) if selected else ""

        rendered = _INCLUDE.sub(include, text)
        if "{{include" in rendered:
            raise ValueError(f"Malformed or unresolved prompt include in {source}")
        return rendered

    template = expand(path, ())
    if not includes:
        return template, None
    return template, {
        "schema": "pii-prompt-includes/v1",
        "model": model,
        "files": files,
        "includes": includes,
        "assembled_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }
