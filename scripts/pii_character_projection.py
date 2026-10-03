#!/usr/bin/env python3
"""Fixed and train-derived Unicode character projections for PII sidecars."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

CHARACTER_PADDING_ID = 0
CHARACTER_BEGIN_ID = 1
CHARACTER_END_ID = 2
CHARACTER_UNKNOWN_ID = 3
CHARACTER_RESERVED_IDS = 4
SCRIPT_V1_DATA_PATH = Path(__file__).parent / "vendor/script_bpe_v1/script_encoding_v1.json"
SCRIPT_V1_DATA_SHA256 = "895fc5cd38a93f8d64509215d2741a4f93f2df722fd5404c8fd27bc9774121fc"
CUSTOM_PROJECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CharacterPairProjection:
    """Map each Unicode scalar to two compact categorical IDs."""

    name: str
    primary_vocab_size: int
    secondary_vocab_size: int
    character_pairs: Mapping[str, tuple[int, int]]
    metadata: Mapping[str, object]

    def pair(self, character: str) -> tuple[int, int]:
        if len(character) != 1:
            raise ValueError("character projection requires exactly one Unicode scalar")
        return self.character_pairs.get(
            character,
            (CHARACTER_UNKNOWN_ID, CHARACTER_UNKNOWN_ID),
        )

    @staticmethod
    def special_pair(identifier: int) -> tuple[int, int]:
        if identifier not in {
            CHARACTER_PADDING_ID,
            CHARACTER_BEGIN_ID,
            CHARACTER_END_ID,
            CHARACTER_UNKNOWN_ID,
        }:
            raise ValueError(f"unsupported special character identifier {identifier}")
        return identifier, identifier


def diagonal_character_ids(projection: CharacterPairProjection, text: str) -> list[int]:
    """Project text to one ID per codepoint, requiring identical component IDs.

    A tokenizer-free character model drives both learned component embedding
    tables from a single ID stream, so that stream reproduces the model's
    two-component input exactly only when every assigned scalar uses the same
    primary and secondary ID.  Unassigned scalars share the reserved unknown
    pair and are therefore always diagonal.
    """
    identifiers: list[int] = []
    for character in text:
        primary, secondary = projection.pair(character)
        if primary != secondary:
            raise ValueError(
                f"U+{ord(character):04X} projects to different component IDs "
                f"({primary}, {secondary}); a one-ID character stream requires "
                f"a diagonal projection, and {projection.name!r} is not one"
            )
        identifiers.append(primary)
    return identifiers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def character_projection_coverage(
    projection: CharacterPairProjection,
    texts: Iterable[str],
) -> dict[str, int | float]:
    """Summarize assigned and unknown character mass without retaining text."""
    assigned_occurrences = 0
    unknown_occurrences = 0
    assigned_characters: set[str] = set()
    unknown_characters: set[str] = set()
    for text in texts:
        for character in text:
            if character in projection.character_pairs:
                assigned_occurrences += 1
                assigned_characters.add(character)
            else:
                unknown_occurrences += 1
                unknown_characters.add(character)
    total_occurrences = assigned_occurrences + unknown_occurrences
    return {
        "total_occurrences": total_occurrences,
        "assigned_occurrences": assigned_occurrences,
        "unknown_occurrences": unknown_occurrences,
        "unknown_occurrence_rate": (unknown_occurrences / total_occurrences if total_occurrences else 0.0),
        "assigned_distinct_characters": len(assigned_characters),
        "unknown_distinct_characters": len(unknown_characters),
    }


def build_literal_character_projection(
    texts: Iterable[str],
    *,
    name: str = "literal-train-v1",
    training_text_sha256: str | None = None,
) -> CharacterPairProjection:
    """Assign one deterministic categorical ID to each train-observed scalar."""
    occurrences = 0
    characters: set[str] = set()
    for text_index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"literal projection text {text_index} is not a string")
        occurrences += len(text)
        characters.update(text)
    ordered = sorted(characters, key=ord)
    for character in ordered:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"literal projection contains surrogate U+{codepoint:04X}")
    character_pairs = {
        character: (identifier, identifier)
        for identifier, character in enumerate(ordered, start=CHARACTER_RESERVED_IDS)
    }
    vocab_size = CHARACTER_RESERVED_IDS + len(character_pairs)
    metadata: dict[str, object] = {
        "projection": "train-only literal Unicode-scalar identity",
        "version": 1,
        "assigned_characters": len(character_pairs),
        "training_character_occurrences": occurrences,
        "reserved_ids": CHARACTER_RESERVED_IDS,
        "unknown_policy": "unassigned scalars share the reserved unknown pair",
    }
    if training_text_sha256 is not None:
        metadata["training_text_sha256"] = training_text_sha256
    return CharacterPairProjection(
        name=name,
        primary_vocab_size=vocab_size,
        secondary_vocab_size=vocab_size,
        character_pairs=MappingProxyType(character_pairs),
        metadata=MappingProxyType(metadata),
    )


@cache
def load_script_v1_projection(path: Path = SCRIPT_V1_DATA_PATH) -> CharacterPairProjection:
    """Load the collision-free pair mapping released with SCRIPT v1."""
    path = path.resolve()
    observed_sha256 = _sha256(path)
    if path == SCRIPT_V1_DATA_PATH.resolve() and observed_sha256 != SCRIPT_V1_DATA_SHA256:
        raise ValueError(f"vendored SCRIPT v1 projection hash mismatch: {observed_sha256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "1.0":
        raise ValueError(f"expected SCRIPT projection version 1.0, got {payload.get('version')!r}")
    num_blocks = payload.get("num_blocks")
    num_index_tokens = payload.get("num_index_tokens")
    blocks = payload.get("blocks")
    if (
        not isinstance(num_blocks, int)
        or num_blocks <= 0
        or not isinstance(num_index_tokens, int)
        or num_index_tokens <= 0
        or not isinstance(blocks, list)
        or len(blocks) != num_blocks
    ):
        raise ValueError("SCRIPT v1 projection has invalid block or index counts")

    character_pairs: dict[str, tuple[int, int]] = {}
    for block_index, block in enumerate(blocks):
        if not isinstance(block, list) or len(block) != 5 or not isinstance(block[4], str):
            raise ValueError(f"SCRIPT v1 block {block_index} has an invalid record")
        characters = block[4]
        if not characters or len(characters) > num_index_tokens:
            raise ValueError(f"SCRIPT v1 block {block_index} has an invalid width")
        primary_id = CHARACTER_RESERVED_IDS + block_index
        for character_index, character in enumerate(characters):
            if character in character_pairs:
                raise ValueError(f"SCRIPT v1 assigns U+{ord(character):04X} more than once")
            character_pairs[character] = (
                primary_id,
                CHARACTER_RESERVED_IDS + character_index,
            )

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("SCRIPT v1 projection lacks source provenance")
    metadata: dict[str, object] = {
        "projection": "SCRIPT v1 block/index pair",
        "version": payload["version"],
        "path": str(path),
        "sha256": observed_sha256,
        "assigned_characters": len(character_pairs),
        "block_ids": num_blocks,
        "index_ids": num_index_tokens,
        "reserved_ids": CHARACTER_RESERVED_IDS,
        "unknown_policy": "unassigned scalars share the reserved unknown pair",
        "source": source,
    }
    return CharacterPairProjection(
        name="script-v1",
        primary_vocab_size=CHARACTER_RESERVED_IDS + num_blocks,
        secondary_vocab_size=CHARACTER_RESERVED_IDS + num_index_tokens,
        character_pairs=MappingProxyType(character_pairs),
        metadata=MappingProxyType(metadata),
    )


@cache
def load_script_v1_character_groups(
    path: Path = SCRIPT_V1_DATA_PATH,
) -> Mapping[str, tuple[str, str]]:
    """Return SCRIPT v1's script/supercategory compatibility group per scalar."""
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("SCRIPT v1 projection lacks its block table")
    groups: dict[str, tuple[str, str]] = {}
    for block_index, block in enumerate(blocks):
        if (
            not isinstance(block, list)
            or len(block) != 5
            or not isinstance(block[1], str)
            or not isinstance(block[2], str)
            or not isinstance(block[4], str)
        ):
            raise ValueError(f"SCRIPT v1 block {block_index} has an invalid record")
        group = block[1], block[2]
        for character in block[4]:
            previous = groups.setdefault(character, group)
            if previous != group:
                raise ValueError(f"SCRIPT v1 assigns U+{ord(character):04X} to incompatible groups")
    return MappingProxyType(groups)


@cache
def load_character_projection(path: Path) -> CharacterPairProjection:
    """Load a generated character-pair projection and validate its ID space."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CUSTOM_PROJECTION_SCHEMA_VERSION:
        raise ValueError(
            "expected character projection schema "
            f"{CUSTOM_PROJECTION_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    name = payload.get("name")
    primary_vocab_size = payload.get("primary_vocab_size")
    secondary_vocab_size = payload.get("secondary_vocab_size")
    records = payload.get("character_pairs")
    metadata = payload.get("metadata")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(primary_vocab_size, int)
        or primary_vocab_size <= CHARACTER_RESERVED_IDS
        or not isinstance(secondary_vocab_size, int)
        or secondary_vocab_size <= CHARACTER_RESERVED_IDS
        or not isinstance(records, list)
        or not isinstance(metadata, dict)
    ):
        raise ValueError("generated character projection has invalid top-level fields")

    character_pairs: dict[str, tuple[int, int]] = {}
    used_pairs: set[tuple[int, int]] = set()
    for record_index, record in enumerate(records):
        if (
            not isinstance(record, list)
            or len(record) != 3
            or not isinstance(record[0], int)
            or not isinstance(record[1], int)
            or not isinstance(record[2], int)
        ):
            raise ValueError(f"character-pair record {record_index} is invalid")
        codepoint, primary_id, secondary_id = record
        if not 0 <= codepoint <= 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"character-pair record {record_index} is not a Unicode scalar")
        pair = primary_id, secondary_id
        if not (
            CHARACTER_RESERVED_IDS <= primary_id < primary_vocab_size
            and CHARACTER_RESERVED_IDS <= secondary_id < secondary_vocab_size
        ):
            raise ValueError(f"character-pair record {record_index} has an out-of-range ID")
        character = chr(codepoint)
        if character in character_pairs or pair in used_pairs:
            raise ValueError(f"character-pair record {record_index} is not collision-free")
        character_pairs[character] = pair
        used_pairs.add(pair)

    artifact_metadata = {
        **metadata,
        "path": str(path),
        "sha256": _sha256(path),
        "assigned_characters": len(character_pairs),
        "reserved_ids": CHARACTER_RESERVED_IDS,
        "unknown_policy": "unassigned scalars share the reserved unknown pair",
    }
    return CharacterPairProjection(
        name=name,
        primary_vocab_size=primary_vocab_size,
        secondary_vocab_size=secondary_vocab_size,
        character_pairs=MappingProxyType(character_pairs),
        metadata=MappingProxyType(artifact_metadata),
    )


def write_character_projection(projection: CharacterPairProjection, path: Path) -> dict[str, object]:
    """Serialize a generated projection without retaining source text."""
    path = path.resolve()
    payload = {
        "schema_version": CUSTOM_PROJECTION_SCHEMA_VERSION,
        "name": projection.name,
        "primary_vocab_size": projection.primary_vocab_size,
        "secondary_vocab_size": projection.secondary_vocab_size,
        "character_pairs": [
            [ord(character), *pair]
            for character, pair in sorted(
                projection.character_pairs.items(),
                key=lambda item: ord(item[0]),
            )
        ],
        "metadata": dict(projection.metadata),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "assigned_characters": len(projection.character_pairs),
    }
