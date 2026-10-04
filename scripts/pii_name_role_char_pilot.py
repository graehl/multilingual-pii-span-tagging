#!/usr/bin/env python3
"""Compare a cased-bigram name-role baseline with a compact character CNN."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from scripts.pii_character_projection import (
    CHARACTER_RESERVED_IDS,
    CHARACTER_UNKNOWN_ID,
    CharacterPairProjection,
    load_script_v1_projection,
)
from scripts.pii_language_policy import load_core_language_policy, validate_core_language_support
from trainlib import ValidationController, WeightedLengthBatchSampler, fork_rng

SCHEMA = "pii-name-role-character-pilot-v1"
ROLES = ("given", "family")
ROLE_IDS = {role: index for index, role in enumerate(ROLES)}
ROLE_BITS = {"given": 1, "family": 2}
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
TRUNC = "<TRUNC>"
TYPED_UNKNOWN_CHARACTERS = (
    "<UNK_LATIN_UPPER>",
    "<UNK_LATIN_LOWER>",
    "<UNK_LETTER>",
    "<UNK_MARK>",
    "<UNK_DIGIT>",
    "<UNK_PUNCT>",
    "<UNK_SYMBOL>",
    "<UNK_SPACE>",
    "<UNK_OTHER>",
)
SCRIPT_BLOCK_UNKNOWN = "<UNK_SCRIPT_UNASSIGNED>"
UNKNOWN_CHARACTER_BACKOFFS = ("typed", "script-block")
UPPERCASE_PUBLISHER_SOURCES = frozenset({"us-census-2010", "insee-prenoms-2024"})
DEFAULT_LANGUAGE_ROUND = Path(__file__).with_name("pii_name_role_language_round_v2.yaml")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """The source checkout's commit; None when this code runs from an extracted archive, not a checkout."""
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return None
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def normalize_surface(surface: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", surface).split())


def split_key(surface: str) -> str:
    return normalize_surface(surface).casefold()


def base_language(language: str) -> str:
    return language.split("-", 1)[0].lower()


def has_mixed_case(surface: str) -> bool:
    return any(character.isupper() for character in surface) and any(
        character.islower() for character in surface
    )


def natural_case(surface: str) -> str:
    """Preserve informative mixed case; otherwise use a title-like display form."""
    if has_mixed_case(surface):
        return surface
    return surface.title()


def script_block_unknown_token(
    character: str,
    projection: CharacterPairProjection | None = None,
) -> str:
    projection = projection or load_script_v1_projection()
    primary_id, _secondary_id = projection.pair(character)
    if primary_id == CHARACTER_UNKNOWN_ID:
        return SCRIPT_BLOCK_UNKNOWN
    block_index = primary_id - CHARACTER_RESERVED_IDS
    block_count = projection.primary_vocab_size - CHARACTER_RESERVED_IDS
    if not 0 <= block_index < block_count:
        raise ValueError(f"SCRIPT block index out of range: {block_index}")
    return f"<UNK_SCRIPT_BLOCK_{block_index:03d}>"


def unknown_character_tokens(backoff: str) -> tuple[str, ...]:
    if backoff == "typed":
        return TYPED_UNKNOWN_CHARACTERS
    if backoff == "script-block":
        projection = load_script_v1_projection()
        block_count = projection.primary_vocab_size - CHARACTER_RESERVED_IDS
        return (
            *(f"<UNK_SCRIPT_BLOCK_{index:03d}>" for index in range(block_count)),
            SCRIPT_BLOCK_UNKNOWN,
        )
    raise ValueError(f"unknown character backoff {backoff!r}")


def unknown_character_token(
    character: str,
    backoff: str = "typed",
    *,
    script_projection: CharacterPairProjection | None = None,
) -> str:
    if backoff == "script-block":
        return script_block_unknown_token(character, script_projection)
    if backoff != "typed":
        raise ValueError(f"unknown character backoff {backoff!r}")
    category = unicodedata.category(character)
    name = unicodedata.name(character, "")
    if "LATIN" in name and category == "Lu":
        return "<UNK_LATIN_UPPER>"
    if "LATIN" in name and category == "Ll":
        return "<UNK_LATIN_LOWER>"
    if category.startswith("L"):
        return "<UNK_LETTER>"
    if category.startswith("M"):
        return "<UNK_MARK>"
    if category.startswith("N"):
        return "<UNK_DIGIT>"
    if category.startswith("P"):
        return "<UNK_PUNCT>"
    if category.startswith("S"):
        return "<UNK_SYMBOL>"
    if character.isspace():
        return "<UNK_SPACE>"
    return "<UNK_OTHER>"


def stable_bucket(value: str, modulus: int = 10_000) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def split_name(key: str) -> str:
    bucket = stable_bucket(key)
    if bucket < 8000:
        return "train"
    if bucket < 9000:
        return "development"
    return "test"


@dataclass(frozen=True)
class Example:
    surface: str
    key: str
    role: str
    language: str
    supervision_weight: float = 1.0
    supervision_tier: str = "unspecified"
    sources: tuple[str, ...] = ()


def _read_context_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"surface", "role", "context_language"}
            if not isinstance(row, dict) or not required <= set(row):
                raise ValueError(f"{path}:{line_number}: invalid name-context row")
            if row["role"] not in ROLE_BITS:
                raise ValueError(f"{path}:{line_number}: unsupported role {row['role']!r}")
            supervision_weight = float(row.get("supervision_weight", 1.0))
            if not math.isfinite(supervision_weight) or supervision_weight <= 0:
                raise ValueError(f"{path}:{line_number}: supervision_weight must be positive and finite")
            row["supervision_weight"] = supervision_weight
            yield row


def load_unambiguous_examples(path: Path) -> tuple[list[Example], dict]:
    """Keep one surface per normalized-key/language after global role disambiguation."""
    role_masks: dict[str, int] = {}
    preferred_case: dict[str, str] = {}
    rows = 0
    for row in _read_context_rows(path):
        surface = normalize_surface(str(row["surface"]))
        key = split_key(surface)
        if not key:
            continue
        role_masks[key] = role_masks.get(key, 0) | ROLE_BITS[str(row["role"])]
        source = str(row.get("source", ""))
        if source not in UPPERCASE_PUBLISHER_SOURCES and has_mixed_case(surface):
            candidate = natural_case(surface)
            current = preferred_case.get(key)
            if current is None or candidate < current:
                preferred_case[key] = candidate
        rows += 1

    representatives: dict[tuple[str, str], Example] = {}
    representative_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    excluded_ambiguous_rows = 0
    publisher_case_matched = 0
    publisher_case_title_fallback = 0
    for row in _read_context_rows(path):
        surface = normalize_surface(str(row["surface"]))
        key = split_key(surface)
        if not key or role_masks.get(key) not in ROLE_BITS.values():
            excluded_ambiguous_rows += 1
            continue
        source = str(row.get("source", ""))
        if source in UPPERCASE_PUBLISHER_SOURCES:
            matched = preferred_case.get(key)
            if matched is None:
                surface = natural_case(surface)
                publisher_case_title_fallback += 1
            else:
                surface = matched
                publisher_case_matched += 1
        else:
            surface = natural_case(surface)
        language = base_language(str(row["context_language"]))
        role = str(row["role"])
        source = str(row.get("source", "unspecified"))
        candidate = Example(
            surface=surface,
            key=key,
            role=role,
            language=language,
            supervision_weight=float(row["supervision_weight"]),
            supervision_tier=str(row.get("supervision_tier", "unspecified")),
            sources=(source,),
        )
        identity = (key, language)
        representative_sources[identity].add(source)
        current = representatives.get(identity)
        if (
            current is None
            or candidate.supervision_weight > current.supervision_weight
            or (
                candidate.supervision_weight == current.supervision_weight
                and (candidate.surface, candidate.supervision_tier)
                < (current.surface, current.supervision_tier)
            )
        ):
            representatives[identity] = candidate

    examples = sorted(
        (
            Example(
                surface=example.surface,
                key=example.key,
                role=example.role,
                language=example.language,
                supervision_weight=example.supervision_weight,
                supervision_tier=example.supervision_tier,
                sources=tuple(sorted(representative_sources[identity])),
            )
            for identity, example in representatives.items()
        ),
        key=lambda item: (item.key, item.language, item.surface),
    )
    return examples, {
        "input_rows": rows,
        "normalized_keys": len(role_masks),
        "ambiguous_role_keys": sum(mask == 3 for mask in role_masks.values()),
        "excluded_ambiguous_rows": excluded_ambiguous_rows,
        "unambiguous_key_language_examples": len(examples),
        "publisher_uppercase_sources": sorted(UPPERCASE_PUBLISHER_SOURCES),
        "publisher_case_matched_from_other_sources": publisher_case_matched,
        "publisher_case_title_fallback": publisher_case_title_fallback,
    }


def cap_partition(examples: Sequence[Example], cap_per_cell: int) -> list[Example]:
    if cap_per_cell <= 0:
        return list(examples)
    cells: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        cells[(example.language, example.role)].append(example)
    selected = []
    for cell in sorted(cells):
        ranked = sorted(
            cells[cell],
            key=lambda item: (
                -item.supervision_weight,
                stable_bucket(f"rank\0{item.key}\0{item.language}", 1 << 63),
                item.key,
            ),
        )
        selected.extend(ranked[:cap_per_cell])
    return sorted(selected, key=lambda item: (item.key, item.language))


def partition_examples(
    examples: Sequence[Example],
    *,
    train_cap_per_cell: int,
    evaluation_cap_per_cell: int,
) -> dict[str, list[Example]]:
    raw: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        raw[split_name(example.key)].append(example)
    return {
        "train": cap_partition(raw["train"], train_cap_per_cell),
        "development": cap_partition(raw["development"], evaluation_cap_per_cell),
        "test": cap_partition(raw["test"], evaluation_cap_per_cell),
    }


def cell_counts(examples: Sequence[Example]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        counts[example.language][example.role] += 1
    return {language: dict(sorted(values.items())) for language, values in sorted(counts.items())}


def load_core_languages(path: Path) -> list[str]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(item["code"]) for item in value["languages"]]


def legacy_sampling_plan(
    examples: Sequence[Example],
    *,
    language_round: Path,
    expansion_mass: float,
) -> tuple[list[float], dict[str, float], dict]:
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        cells[(example.language, example.role)].append(index)
    languages = sorted({language for language, _role in cells})
    core = load_core_languages(language_round)
    missing_core_cells = {
        language: [role for role in ROLES if not cells.get((language, role))]
        for language in core
        if any(not cells.get((language, role)) for role in ROLES)
    }
    if missing_core_cells:
        raise ValueError(f"core languages lack a role cell: {missing_core_cells}")
    extras = sorted(set(languages) - set(core))
    if not 0 <= expansion_mass < 1:
        raise ValueError("expansion mass must be in [0, 1)")
    if not extras:
        expansion_mass = 0.0
    core_mass = 1.0 - expansion_mass
    language_mass = {language: core_mass / len(core) for language in core}
    language_mass.update({language: expansion_mass / len(extras) for language in extras})
    weights = [0.0] * len(examples)
    cell_weight_sums = {}
    for language in languages:
        roles = [role for role in ROLES if cells.get((language, role))]
        for role in roles:
            indices = cells[(language, role)]
            total_supervision_weight = sum(examples[index].supervision_weight for index in indices)
            cell_weight_sums[f"{language}:{role}"] = total_supervision_weight
            for index in indices:
                weights[index] = (
                    language_mass[language]
                    / len(roles)
                    * examples[index].supervision_weight
                    / total_supervision_weight
                )
    if any(weight <= 0 for weight in weights):
        raise ValueError("every training example must receive positive sampler mass")
    receipt = {
        **validate_core_language_support(language_mass, language_round=language_round),
        "weighting_order": [
            "fixed language mass",
            "equal mass over available roles within language",
            "relative supervision weight within language-role cell",
        ],
        "cell_supervision_weight_sums": cell_weight_sums,
    }
    return weights, language_mass, receipt


def sampling_plan(
    examples: Sequence[Example],
    *,
    language_round: Path,
    maximum_support_multiplier: float,
    full_support_effective_cell_weight: float,
) -> tuple[list[float], dict[str, float], dict]:
    if not math.isfinite(maximum_support_multiplier) or maximum_support_multiplier < 1:
        raise ValueError("maximum support multiplier must be finite and at least one")
    if not math.isfinite(full_support_effective_cell_weight) or full_support_effective_cell_weight <= 0:
        raise ValueError("full-support effective cell weight must be positive and finite")
    policy = load_core_language_policy(language_round)
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        cells[(example.language, example.role)].append(index)
    languages = sorted({language for language, _role in cells})
    unknown_languages = sorted(set(languages) - set(policy["core_languages"]))
    if unknown_languages:
        raise ValueError(f"training data contains languages outside the declared round: {unknown_languages}")

    effective_cell_weight = {
        (language, role): sum(examples[index].supervision_weight for index in indices)
        for (language, role), indices in cells.items()
    }
    support_multipliers = {}
    for language in languages:
        balanced_support = min(
            (effective_cell_weight.get((language, role), 0.0) for role in ROLES),
            default=0.0,
        )
        support_fraction = min(1.0, balanced_support / full_support_effective_cell_weight)
        support_multipliers[language] = 1 + support_fraction * (maximum_support_multiplier - 1)
    raw_language_mass = {
        language: policy["importance_weights"][language] * support_multipliers[language]
        for language in languages
    }
    total_language_mass = sum(raw_language_mass.values())
    language_mass = {language: mass / total_language_mass for language, mass in raw_language_mass.items()}
    below_minimum = {
        language: mass
        for language, mass in language_mass.items()
        if mass + 1e-12 < policy["minimum_expected_share"]
    }
    if below_minimum:
        raise ValueError(
            "represented-language expected sampler floor failed: "
            f"below={below_minimum} minimum={policy['minimum_expected_share']:.6f}"
        )
    void_languages = sorted(set(policy["core_languages"]) - set(languages))
    full_core_validation = (
        None
        if void_languages
        else validate_core_language_support(language_mass, language_round=language_round)
    )

    weights = [0.0] * len(examples)
    for language in languages:
        roles = [role for role in ROLES if cells.get((language, role))]
        for role in roles:
            indices = cells[(language, role)]
            total_supervision_weight = effective_cell_weight[(language, role)]
            for index in indices:
                weights[index] = (
                    language_mass[language]
                    / len(roles)
                    * examples[index].supervision_weight
                    / total_supervision_weight
                )
    if any(weight <= 0 for weight in weights):
        raise ValueError("every training example must receive positive sampler mass")
    receipt = {
        "status": (
            "verified_represented_sampler_with_voids"
            if void_languages
            else "verified_full_language_round_sampler"
        ),
        "language_round": {
            "path": policy["path"],
            "sha256": policy["sha256"],
            "round_id": policy["round_id"],
        },
        "importance_weights": {language: policy["importance_weights"][language] for language in languages},
        "minimum_expected_represented_language_share": policy["minimum_expected_share"],
        "full_core_validation": full_core_validation,
        "support_multiplier": {
            "maximum": maximum_support_multiplier,
            "full_support_effective_cell_weight": full_support_effective_cell_weight,
            "per_language": support_multipliers,
        },
        "void_languages": void_languages,
        "weighting_order": [
            "declared 4x/2x/1x language importance",
            "bounded support multiplier over balanced effective role support",
            "normalization over non-void languages",
            "equal mass over available roles within language",
            "relative supervision weight within language-role cell",
        ],
        "effective_cell_weight": {
            f"{language}:{role}": value for (language, role), value in sorted(effective_cell_weight.items())
        },
    }
    return weights, language_mass, receipt


def case_variants(
    surface: str,
    *,
    natural_weight: float,
    uppercase_weight: float,
    lowercase_weight: float,
) -> tuple[tuple[str, float], ...]:
    weights = (natural_weight, uppercase_weight, lowercase_weight)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("case intent weights must be finite and nonnegative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("case intent weights must sum to one")
    combined: dict[str, float] = {}
    for variant, weight in zip(
        (natural_case(surface), surface.upper(), surface.lower()),
        weights,
        strict=True,
    ):
        combined[variant] = combined.get(variant, 0.0) + weight
    return tuple((variant, weight) for variant, weight in combined.items() if weight > 0)


def cased_bigrams(surface: str) -> list[tuple[str, str]]:
    characters = [BOS, *surface, EOS]
    return list(zip(characters, characters[1:]))


@dataclass
class BigramModel:
    class_counts: list[float]
    totals: list[float]
    counts: list[Counter[tuple[str, str]]]
    vocabulary: set[tuple[str, str]]
    alpha: float

    @classmethod
    def fit(
        cls,
        examples: Sequence[Example],
        example_weights: Sequence[float],
        *,
        alpha: float,
        case_weights: tuple[float, float, float],
    ) -> BigramModel:
        class_counts = [0.0] * len(ROLES)
        totals = [0.0] * len(ROLES)
        counts = [Counter() for _ in ROLES]
        vocabulary: set[tuple[str, str]] = set()
        for example, weight in zip(examples, example_weights, strict=True):
            role_id = ROLE_IDS[example.role]
            variants = case_variants(
                example.surface,
                natural_weight=case_weights[0],
                uppercase_weight=case_weights[1],
                lowercase_weight=case_weights[2],
            )
            class_counts[role_id] += float(weight)
            for variant, case_weight in variants:
                variant_weight = float(weight) * case_weight
                for bigram in cased_bigrams(variant):
                    counts[role_id][bigram] += variant_weight
                    totals[role_id] += variant_weight
                    vocabulary.add(bigram)
        return cls(class_counts, totals, counts, vocabulary, alpha)

    def predict_one(self, surface: str) -> int:
        vocabulary_size = len(self.vocabulary) + 1
        total_class_mass = sum(self.class_counts)
        scores = []
        for role_id in range(len(ROLES)):
            score = math.log(self.class_counts[role_id] / total_class_mass)
            denominator = self.totals[role_id] + self.alpha * vocabulary_size
            for bigram in cased_bigrams(surface):
                score += math.log((self.counts[role_id].get(bigram, 0.0) + self.alpha) / denominator)
            scores.append(score)
        return max(range(len(scores)), key=scores.__getitem__)

    def predict(self, surfaces: Sequence[str]) -> list[int]:
        return [self.predict_one(surface) for surface in surfaces]

    def save_gzip(self, path: Path) -> None:
        value = {
            "schema": SCHEMA,
            "kind": "cased-bigram-naive-bayes",
            "roles": list(ROLES),
            "alpha": self.alpha,
            "class_counts": self.class_counts,
            "totals": self.totals,
            "counts": [
                [[left, right, count] for (left, right), count in sorted(role_counts.items())]
                for role_counts in self.counts
            ],
        }
        payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
                output.write(payload)


def build_character_vocabulary(
    examples: Sequence[Example],
    *,
    rare_character_maximum_count: int,
    rare_character_backoff_probability: float,
    maximum_characters: int,
    case_weights: tuple[float, float, float],
    unknown_character_backoff: str,
) -> tuple[list[str], dict[str, int]]:
    if rare_character_maximum_count < -1:
        raise ValueError("rare character maximum count must be at least -1")
    if not 0 <= rare_character_backoff_probability <= 1:
        raise ValueError("rare character backoff probability must be in [0, 1]")
    natural_counts: Counter[str] = Counter()
    candidate_characters: set[str] = set()
    for example in examples:
        natural_counts.update(natural_case(example.surface))
        for surface, _weight in case_variants(
            example.surface,
            natural_weight=case_weights[0],
            uppercase_weight=case_weights[1],
            lowercase_weight=case_weights[2],
        ):
            candidate_characters.update(surface)
    rare_characters = {
        character
        for character in candidate_characters
        if natural_counts.get(character, 0) <= rare_character_maximum_count
    }
    if rare_character_backoff_probability == 1:
        candidate_characters -= rare_characters
    characters = [
        character
        for character in sorted(
            candidate_characters,
            key=lambda item: (-natural_counts.get(item, 0), item),
        )
    ]
    if maximum_characters:
        characters = characters[:maximum_characters]
    vocabulary = [
        PAD,
        BOS,
        EOS,
        TRUNC,
        *unknown_character_tokens(unknown_character_backoff),
        *characters,
    ]
    return vocabulary, {character: index for index, character in enumerate(vocabulary)}


def character_backoff_plan(
    examples: Sequence[Example],
    *,
    rare_character_maximum_count: int,
    rare_character_backoff_probability: float,
    frequent_character_backoff_probability: float,
    unknown_character_backoff: str,
    vocabulary: Sequence[str],
) -> tuple[frozenset[str], dict]:
    if not 0 <= frequent_character_backoff_probability <= 1:
        raise ValueError("frequent character backoff probability must be in [0, 1]")
    natural_counts: Counter[str] = Counter()
    for example in examples:
        natural_counts.update(natural_case(example.surface))
    rare_characters = frozenset(
        character for character, count in natural_counts.items() if count <= rare_character_maximum_count
    )
    literal_characters = {item for item in vocabulary if len(item) == 1}
    script_blocks = set()
    unassigned_characters = 0
    if unknown_character_backoff == "script-block":
        for character in rare_characters:
            token = script_block_unknown_token(character)
            if token == SCRIPT_BLOCK_UNKNOWN:
                unassigned_characters += 1
            else:
                script_blocks.add(token)
    receipt = {
        "count_source": "normalized and publisher-case-reconstructed natural training surfaces",
        "rare_character_maximum_count": rare_character_maximum_count,
        "rare_character_backoff_probability": rare_character_backoff_probability,
        "frequent_character_backoff_probability": frequent_character_backoff_probability,
        "natural_codepoint_occurrences": sum(natural_counts.values()),
        "natural_distinct_codepoints": len(natural_counts),
        "rare_distinct_codepoints": len(rare_characters),
        "rare_codepoint_occurrences": sum(natural_counts[item] for item in rare_characters),
        "retained_literal_codepoints": len(literal_characters),
        "unknown_character_backoff": unknown_character_backoff,
        "fallback_tokens": len(unknown_character_tokens(unknown_character_backoff)),
        "rare_script_blocks": len(script_blocks) if unknown_character_backoff == "script-block" else None,
        "rare_unassigned_codepoints": (
            unassigned_characters if unknown_character_backoff == "script-block" else None
        ),
        "hard_replacement": rare_character_backoff_probability == 1,
    }
    if unknown_character_backoff == "script-block":
        receipt["script_projection"] = dict(load_script_v1_projection().metadata)
    return rare_characters, receipt


def encode_surface(
    surface: str,
    vocabulary: dict[str, int],
    max_characters: int,
    *,
    unknown_character_backoff: str = "typed",
    rare_characters: frozenset[str] = frozenset(),
    rare_character_backoff_probability: float = 0.0,
    frequent_character_backoff_probability: float = 0.0,
    backoff_rng: random.Random | None = None,
    script_projection: CharacterPairProjection | None = None,
) -> list[int]:
    if max_characters < 5:
        raise ValueError("max_characters must be at least five")
    if not 0 <= rare_character_backoff_probability <= 1:
        raise ValueError("rare character backoff probability must be in [0, 1]")
    if not 0 <= frequent_character_backoff_probability <= 1:
        raise ValueError("frequent character backoff probability must be in [0, 1]")
    if (
        rare_character_backoff_probability not in {0.0, 1.0}
        or frequent_character_backoff_probability not in {0.0, 1.0}
    ) and backoff_rng is None:
        raise ValueError("fractional character backoff requires a random generator")
    characters = list(surface)
    content_limit = max_characters - 2
    if len(characters) > content_limit:
        left = (content_limit - 1) // 2
        right = content_limit - 1 - left
        characters = [*characters[:left], TRUNC, *characters[-right:]]
    ids = [vocabulary[BOS]]
    for character in characters:
        if character == TRUNC:
            ids.append(vocabulary[TRUNC])
            continue
        probability = (
            rare_character_backoff_probability
            if character in rare_characters
            else frequent_character_backoff_probability
        )
        replace = probability == 1 or (
            probability > 0 and backoff_rng is not None and backoff_rng.random() < probability
        )
        if replace or character not in vocabulary:
            token = unknown_character_token(
                character,
                unknown_character_backoff,
                script_projection=script_projection,
            )
            ids.append(vocabulary[token])
        else:
            ids.append(vocabulary[character])
    ids.append(vocabulary[EOS])
    ids.extend([vocabulary[PAD]] * (max_characters - len(ids)))
    return ids


class NameRoleDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        *,
        character_ids: dict[str, int],
        language_ids: dict[str, int],
        max_characters: int,
        case_weights: tuple[float, float, float],
        unknown_character_backoff: str,
        rare_characters: frozenset[str],
        rare_character_backoff_probability: float,
        frequent_character_backoff_probability: float,
        seed: int,
    ) -> None:
        self.examples = list(examples)
        self.character_ids = character_ids
        self.language_ids = language_ids
        self.max_characters = max_characters
        self.case_weights = case_weights
        self.unknown_character_backoff = unknown_character_backoff
        self.rare_characters = rare_characters
        self.rare_character_backoff_probability = rare_character_backoff_probability
        self.frequent_character_backoff_probability = frequent_character_backoff_probability
        self.case_rng = fork_rng(seed, "name-role-case-augmentation")
        self.backoff_rng = fork_rng(seed, "name-role-character-backoff")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        variants = case_variants(
            example.surface,
            natural_weight=self.case_weights[0],
            uppercase_weight=self.case_weights[1],
            lowercase_weight=self.case_weights[2],
        )
        surface = self.case_rng.choices(
            [variant for variant, _weight in variants],
            weights=[weight for _variant, weight in variants],
            k=1,
        )[0]
        characters = torch.tensor(
            encode_surface(
                surface,
                self.character_ids,
                self.max_characters,
                unknown_character_backoff=self.unknown_character_backoff,
                rare_characters=self.rare_characters,
                rare_character_backoff_probability=self.rare_character_backoff_probability,
                frequent_character_backoff_probability=self.frequent_character_backoff_probability,
                backoff_rng=self.backoff_rng,
            ),
            dtype=torch.int64,
        )
        language = torch.zeros(len(self.language_ids), dtype=torch.float32)
        language[self.language_ids[example.language]] = 1.0
        target = torch.tensor(ROLE_IDS[example.role], dtype=torch.int64)
        return characters, language, target


class NameRoleCharCNN(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        language_count: int,
        embedding_dim: int,
        convolution_channels: int,
        language_dim: int,
        language_dropout: float,
        language_noise: float,
        shared_mixture_alpha: float,
    ) -> None:
        super().__init__()
        if not 0 <= language_dropout < 1:
            raise ValueError("language dropout must be in [0, 1)")
        if not 0 <= language_noise <= 1:
            raise ValueError("language noise must be in [0, 1]")
        if not 0 <= shared_mixture_alpha <= 1:
            raise ValueError("shared mixture alpha must be in [0, 1]")
        self.language_dropout = language_dropout
        self.language_noise = language_noise
        self.shared_mixture_alpha = shared_mixture_alpha
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, convolution_channels, kernel_size, bias=False)
            for kernel_size in (2, 3, 4)
        )
        self.language_projection = nn.Linear(language_count, language_dim, bias=False)
        character_dim = 3 * convolution_channels
        self.shared_classifier = nn.Linear(character_dim, len(ROLES))
        self.conditioned_classifier = nn.Linear(character_dim + language_dim, len(ROLES))

    def forward(
        self,
        char_ids: torch.Tensor,
        language_probs: torch.Tensor,
        mixture_alpha: float | None = None,
    ) -> torch.Tensor:
        embedded = self.embedding(char_ids).transpose(1, 2)
        pooled = [torch.relu(convolution(embedded)).amax(dim=2) for convolution in self.convolutions]
        characters = torch.cat(pooled, dim=1)
        if self.training and self.language_dropout > 0:
            keep = torch.rand((language_probs.shape[0], 1), device=language_probs.device)
            language_probs = language_probs * (keep >= self.language_dropout)
        if self.training and self.language_noise > 0:
            random_probs = torch.rand_like(language_probs)
            random_probs = random_probs / random_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            noise = self.language_noise * torch.rand(
                (language_probs.shape[0], 1),
                device=language_probs.device,
            )
            has_language = language_probs.sum(dim=1, keepdim=True) > 0
            language_probs = torch.where(
                has_language,
                (1 - noise) * language_probs + noise * random_probs,
                language_probs,
            )
        language = torch.tanh(self.language_projection(language_probs))
        shared = torch.softmax(self.shared_classifier(characters), dim=1)
        conditioned = torch.softmax(
            self.conditioned_classifier(torch.cat([characters, language], dim=1)),
            dim=1,
        )
        alpha = self.shared_mixture_alpha if mixture_alpha is None else mixture_alpha
        if isinstance(alpha, torch.Tensor):
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(1)
            if alpha.ndim != 2 or alpha.shape[1] != 1:
                raise ValueError("tensor mixture alpha must have shape [batch, 1]")
        elif not 0 <= alpha <= 1:
            raise ValueError("mixture alpha must be in [0, 1]")
        probabilities = alpha * shared + (1 - alpha) * conditioned
        return probabilities.clamp_min(1e-12).log()


def model_compatibility_signature(config: dict) -> dict:
    backoff = config["character_backoff"]
    script_projection = backoff.get("script_projection", {})
    return {
        "schema": config["schema"],
        "roles": config["roles"],
        "character_vocabulary": config["character_vocabulary"],
        "languages": config["languages"],
        "max_characters": config["max_characters"],
        "embedding_dim": config["embedding_dim"],
        "convolution_channels": config["convolution_channels"],
        "language_dim": config["language_dim"],
        "shared_mixture_alpha": config["shared_mixture_alpha"],
        "rare_character_maximum_count": backoff["rare_character_maximum_count"],
        "rare_character_backoff_probability": backoff["rare_character_backoff_probability"],
        "frequent_character_backoff_probability": backoff["frequent_character_backoff_probability"],
        "unknown_character_backoff": backoff["unknown_character_backoff"],
        "script_projection_sha256": script_projection.get("sha256"),
    }


def load_initial_model(
    model: NameRoleCharCNN,
    output: Path,
    expected_config: dict,
) -> dict:
    output = output.resolve()
    config_path = output / "config.json"
    model_path = output / "model.safetensors"
    parent_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = model_compatibility_signature(expected_config)
    observed = model_compatibility_signature(parent_config)
    differences = [key for key in expected if expected[key] != observed[key]]
    if differences:
        raise ValueError(f"incompatible initial model fields: {', '.join(differences)}")
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    return {
        "output": str(output),
        "model": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "compatibility_fields": sorted(expected),
    }


def metric_counts(targets: Sequence[int], predictions: Sequence[int]) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("target and prediction lengths differ")
    per_role = {}
    f1s = []
    recalls = []
    for role_id, role in enumerate(ROLES):
        true_positive = sum(t == role_id and p == role_id for t, p in zip(targets, predictions))
        predicted = sum(p == role_id for p in predictions)
        gold = sum(t == role_id for t in targets)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_role[role] = {
            "tp": true_positive,
            "predicted": predicted,
            "gold": gold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if gold:
            f1s.append(f1)
            recalls.append(recall)
    return {
        "rows": len(targets),
        "accuracy": sum(t == p for t, p in zip(targets, predictions)) / len(targets),
        "macro_f1_observed_roles": sum(f1s) / len(f1s),
        "balanced_accuracy_observed_roles": sum(recalls) / len(recalls),
        "per_role": per_role,
    }


def score_examples(examples: Sequence[Example], predictions: Sequence[int]) -> dict:
    targets = [ROLE_IDS[example.role] for example in examples]
    overall = metric_counts(targets, predictions)
    per_language = {}
    for language in sorted({example.language for example in examples}):
        indices = [index for index, example in enumerate(examples) if example.language == language]
        per_language[language] = metric_counts(
            [targets[index] for index in indices],
            [predictions[index] for index in indices],
        )
    dual_role = [
        value["macro_f1_observed_roles"]
        for value in per_language.values()
        if all(value["per_role"][role]["gold"] for role in ROLES)
    ]
    overall["dual_role_language_macro_f1"] = sum(dual_role) / len(dual_role) if dual_role else None
    overall["dual_role_languages"] = len(dual_role)
    overall["per_language"] = per_language
    return overall


def case_view(surface: str, mode: str) -> str:
    if mode == "natural":
        return natural_case(surface)
    if mode == "upper":
        return surface.upper()
    if mode == "lower":
        return surface.lower()
    raise ValueError(f"unknown case view {mode!r}")


def batched_model_predictions(
    model: NameRoleCharCNN,
    examples: Sequence[Example],
    *,
    character_ids: dict[str, int],
    language_ids: dict[str, int],
    max_characters: int,
    unknown_character_backoff: str,
    batch_size: int,
    surface_mode: str,
    use_language: bool,
    mixture_alpha: float,
) -> list[int]:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for begin in range(0, len(examples), batch_size):
            batch = examples[begin : begin + batch_size]
            surfaces = [case_view(example.surface, surface_mode) for example in batch]
            characters = torch.tensor(
                [
                    encode_surface(
                        surface,
                        character_ids,
                        max_characters,
                        unknown_character_backoff=unknown_character_backoff,
                    )
                    for surface in surfaces
                ],
                dtype=torch.int64,
            )
            languages = torch.zeros((len(batch), len(language_ids)), dtype=torch.float32)
            if use_language:
                for index, example in enumerate(batch):
                    languages[index, language_ids[example.language]] = 1.0
            predictions.extend(
                model(characters, languages, mixture_alpha=mixture_alpha).argmax(dim=1).tolist()
            )
    return predictions


def evaluate_model(
    model: NameRoleCharCNN,
    examples: Sequence[Example],
    **kwargs,
) -> dict:
    alpha = model.shared_mixture_alpha
    modes = {
        "natural_language": ("natural", True, alpha),
        "natural_shared_only": ("natural", False, 1.0),
        "natural_conditioned_only": ("natural", True, 0.0),
        "upper_language": ("upper", True, alpha),
        "lower_language": ("lower", True, alpha),
    }
    results = {}
    for key, (surface_mode, use_language, mixture_alpha) in modes.items():
        predictions = batched_model_predictions(
            model,
            examples,
            surface_mode=surface_mode,
            use_language=use_language,
            mixture_alpha=mixture_alpha,
            **kwargs,
        )
        results[key] = score_examples(examples, predictions)
    return results


def evaluate_bigram(model: BigramModel, examples: Sequence[Example]) -> dict:
    results = {}
    for surface_mode in ("natural", "upper", "lower"):
        surfaces = [case_view(example.surface, surface_mode) for example in examples]
        results[surface_mode] = score_examples(examples, model.predict(surfaces))
    return results


class _ValidationHooks:
    """Minimal trainlib ValidationController host for an in-memory pilot model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.best_state: dict[str, torch.Tensor] | None = None

    def _update_val_cycle_progress(self, running_loss: float) -> None:
        del running_loss

    def _maybe_adjust_patience_lr_rebound(self, running_loss: float) -> None:
        del running_loss

    def _try_patience_lr_anneal(self, running_loss: float) -> bool:
        del running_loss
        return False

    def _reset_val_cadence(self, step: int) -> None:
        del step

    def _resume_recent_checkpoint_capture(self, *, reason: str) -> None:
        del reason

    def _save_val_best(self) -> None:
        self.best_state = copy.deepcopy(self.model.state_dict())

    def _write_val_decode_snapshot(self, event: str, *, running_loss: float | None = None) -> None:
        del event, running_loss


def train_model(
    model: NameRoleCharCNN,
    train_dataset: NameRoleDataset,
    sampler_weights: Sequence[float],
    development: Sequence[Example],
    *,
    character_ids: dict[str, int],
    language_ids: dict[str, int],
    max_characters: int,
    unknown_character_backoff: str,
    batch_size: int,
    samples_per_epoch: int,
    maximum_epochs: int,
    patience: int,
    learning_rate: float,
    length_window_steps: int,
    seed: int,
    protect_initial: bool,
) -> tuple[NameRoleCharCNN, list[dict], int, dict | None]:
    sampler = WeightedLengthBatchSampler(
        lengths=[min(len(example.surface) + 2, max_characters) for example in train_dataset.examples],
        weights=sampler_weights,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        epoch_examples=samples_per_epoch,
        length_window_steps=length_window_steps,
        seed=seed,
        # The recorded name-kind recipe predates the carried default.
        draw_policy="systematic",
    )
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum_epochs)
    controller = ValidationController(val_patience=patience, val_min_delta=0.0)
    validation_hooks = _ValidationHooks(model)
    history = []
    steps = 0
    initial_validation = None
    if protect_initial:
        initial_validation = evaluate_model(
            model,
            development,
            character_ids=character_ids,
            language_ids=language_ids,
            max_characters=max_characters,
            unknown_character_backoff=unknown_character_backoff,
            batch_size=batch_size,
        )
        initial_score = initial_validation["natural_language"]["macro_f1_observed_roles"]
        controller.observe(
            1.0 - initial_score,
            step=0,
            cycles_complete=0,
            hooks=validation_hooks,
        )
        print(
            json.dumps(
                {
                    "phase": "initial-validation",
                    "development_natural_language_macro_f1": initial_score,
                }
            ),
            flush=True,
        )
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        loss_sum = 0.0
        rows = 0
        started = time.perf_counter()
        for characters, languages, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(characters, languages)
            loss = nn.functional.nll_loss(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            rows += len(targets)
            steps += 1
        scheduler.step()
        development_result = evaluate_model(
            model,
            development,
            character_ids=character_ids,
            language_ids=language_ids,
            max_characters=max_characters,
            unknown_character_backoff=unknown_character_backoff,
            batch_size=batch_size,
        )
        score = development_result["natural_language"]["macro_f1_observed_roles"]
        epoch_result = {
            "epoch": epoch,
            "steps": steps,
            "rows": rows,
            "train_loss": loss_sum / rows,
            "learning_rate": scheduler.get_last_lr()[0],
            "development_natural_language_macro_f1": score,
            "development_natural_shared_only_macro_f1": development_result["natural_shared_only"][
                "macro_f1_observed_roles"
            ],
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_result)
        print(json.dumps({"phase": "train", **epoch_result}), flush=True)
        should_stop = controller.observe(
            1.0 - score,
            step=steps,
            cycles_complete=epoch,
            hooks=validation_hooks,
        )
        if should_stop:
            break
    if validation_hooks.best_state is None:
        raise RuntimeError("training produced no selected state")
    model.load_state_dict(validation_hooks.best_state)
    return model, history, steps, initial_validation


def selected_epoch(history: Sequence[dict], initial_validation: dict | None) -> int:
    candidates = [
        (item["development_natural_language_macro_f1"], -item["epoch"], item["epoch"]) for item in history
    ]
    if initial_validation is not None:
        candidates.append(
            (
                initial_validation["natural_language"]["macro_f1_observed_roles"],
                0,
                0,
            )
        )
    return max(candidates)[2]


def _timed_samples(operation, repetitions: int) -> list[float]:
    operation()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return samples


def benchmark(
    model: NameRoleCharCNN,
    bigram: BigramModel,
    examples: Sequence[Example],
    *,
    character_ids: dict[str, int],
    language_ids: dict[str, int],
    max_characters: int,
    unknown_character_backoff: str,
    repetitions: int,
) -> dict:
    sample = list(examples[: min(1024, len(examples))])
    one = sample[:1]
    operations = {
        "bigram_python_batch1": lambda: bigram.predict([natural_case(one[0].surface)]),
        "bigram_python_batch1024": lambda: bigram.predict(
            [natural_case(example.surface) for example in sample]
        ),
        "torch_encode_and_forward_batch1": lambda: batched_model_predictions(
            model,
            one,
            character_ids=character_ids,
            language_ids=language_ids,
            max_characters=max_characters,
            unknown_character_backoff=unknown_character_backoff,
            batch_size=1,
            surface_mode="natural",
            use_language=True,
            mixture_alpha=model.shared_mixture_alpha,
        ),
        "torch_encode_and_forward_batch1024": lambda: batched_model_predictions(
            model,
            sample,
            character_ids=character_ids,
            language_ids=language_ids,
            max_characters=max_characters,
            unknown_character_backoff=unknown_character_backoff,
            batch_size=len(sample),
            surface_mode="natural",
            use_language=True,
            mixture_alpha=model.shared_mixture_alpha,
        ),
    }
    result = {}
    for name, operation in operations.items():
        samples = _timed_samples(operation, repetitions)
        rows = 1 if name.endswith("batch1") else len(sample)
        result[name] = {
            "raw_seconds": samples,
            "median_microseconds_per_name": 1e6 * sorted(samples)[len(samples) // 2] / rows,
            "rows_per_call": rows,
        }
    return result


def host_snapshot() -> dict:
    load = os.getloadavg()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "load_average": list(load),
    }


def validate_output_directory(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(path)
    unexpected = [item for item in path.iterdir() if not item.name.endswith(".meta.md")]
    if unexpected:
        raise FileExistsError(f"output directory contains prior artifacts: {unexpected}")


def fit_evaluate(args: argparse.Namespace) -> None:
    validate_output_directory(args.output)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.training_threads)
    before = host_snapshot()

    examples, loading = load_unambiguous_examples(args.context_jsonl)
    partitions = partition_examples(
        examples,
        train_cap_per_cell=args.train_cap_per_language_role,
        evaluation_cap_per_cell=args.evaluation_cap_per_language_role,
    )
    split_keys = {name: {example.key for example in rows} for name, rows in partitions.items()}
    if any(
        split_keys[left] & split_keys[right]
        for left, right in (("train", "development"), ("train", "test"), ("development", "test"))
    ):
        raise ValueError("normalized surface keys overlap partitions")

    if args.expansion_mass is None:
        sampler_weights, language_mass, language_receipt = sampling_plan(
            partitions["train"],
            language_round=args.language_round,
            maximum_support_multiplier=args.maximum_support_multiplier,
            full_support_effective_cell_weight=args.full_support_effective_cell_weight,
        )
    else:
        sampler_weights, language_mass, language_receipt = legacy_sampling_plan(
            partitions["train"],
            language_round=args.language_round,
            expansion_mass=args.expansion_mass,
        )
    rare_character_maximum_count = args.rare_character_maximum_count
    if args.minimum_character_count is not None:
        if args.minimum_character_count < 0:
            raise ValueError("minimum character count must be nonnegative")
        rare_character_maximum_count = args.minimum_character_count - 1
    case_weights = (
        args.case_natural_weight,
        args.case_uppercase_weight,
        args.case_lowercase_weight,
    )
    character_vocabulary, character_ids = build_character_vocabulary(
        partitions["train"],
        rare_character_maximum_count=rare_character_maximum_count,
        rare_character_backoff_probability=args.rare_character_backoff_probability,
        maximum_characters=args.maximum_vocabulary_characters,
        case_weights=case_weights,
        unknown_character_backoff=args.unknown_character_backoff,
    )
    rare_characters, character_backoff_receipt = character_backoff_plan(
        partitions["train"],
        rare_character_maximum_count=rare_character_maximum_count,
        rare_character_backoff_probability=args.rare_character_backoff_probability,
        frequent_character_backoff_probability=args.frequent_character_backoff_probability,
        unknown_character_backoff=args.unknown_character_backoff,
        vocabulary=character_vocabulary,
    )
    languages = sorted({example.language for example in examples})
    language_ids = {language: index for index, language in enumerate(languages)}

    bigram = BigramModel.fit(
        partitions["train"],
        sampler_weights,
        alpha=args.bigram_alpha,
        case_weights=case_weights,
    )
    bigram_development = evaluate_bigram(bigram, partitions["development"])

    train_dataset = NameRoleDataset(
        partitions["train"],
        character_ids=character_ids,
        language_ids=language_ids,
        max_characters=args.max_characters,
        case_weights=case_weights,
        unknown_character_backoff=args.unknown_character_backoff,
        rare_characters=rare_characters,
        rare_character_backoff_probability=args.rare_character_backoff_probability,
        frequent_character_backoff_probability=args.frequent_character_backoff_probability,
        seed=args.seed,
    )
    model = NameRoleCharCNN(
        vocabulary_size=len(character_vocabulary),
        language_count=len(language_ids),
        embedding_dim=args.embedding_dim,
        convolution_channels=args.convolution_channels,
        language_dim=args.language_dim,
        language_dropout=args.language_dropout,
        language_noise=args.language_noise,
        shared_mixture_alpha=args.shared_mixture_alpha,
    )
    config = {
        "schema": SCHEMA,
        "roles": list(ROLES),
        "character_vocabulary": character_vocabulary,
        "languages": languages,
        "max_characters": args.max_characters,
        "embedding_dim": args.embedding_dim,
        "convolution_channels": args.convolution_channels,
        "language_dim": args.language_dim,
        "language_dropout": args.language_dropout,
        "language_noise": args.language_noise,
        "shared_mixture_alpha": args.shared_mixture_alpha,
        "character_backoff": character_backoff_receipt,
        "probability_mixture": "alpha*p(role|name) + (1-alpha)*p(role|name,language_probs)",
        "nation_input": False,
        "language_input": "base-language probability vector; all-zero means unavailable",
        "semantic_scope": (
            "lexical given-versus-family evidence; middle/neither remain sequence-grammar outcomes"
        ),
    }
    initialization = None
    if args.init_from_output is not None:
        if args.output.resolve() == args.init_from_output.resolve():
            raise ValueError("continuation output must differ from its initialization output")
        initialization = load_initial_model(model, args.init_from_output, config)
    model, history, steps, initial_validation = train_model(
        model,
        train_dataset,
        sampler_weights,
        partitions["development"],
        character_ids=character_ids,
        language_ids=language_ids,
        max_characters=args.max_characters,
        unknown_character_backoff=args.unknown_character_backoff,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        length_window_steps=args.length_window_steps,
        seed=args.seed,
        protect_initial=initialization is not None,
    )
    development = evaluate_model(
        model,
        partitions["development"],
        character_ids=character_ids,
        language_ids=language_ids,
        max_characters=args.max_characters,
        unknown_character_backoff=args.unknown_character_backoff,
        batch_size=args.batch_size,
    )
    test = evaluate_model(
        model,
        partitions["test"],
        character_ids=character_ids,
        language_ids=language_ids,
        max_characters=args.max_characters,
        unknown_character_backoff=args.unknown_character_backoff,
        batch_size=args.batch_size,
    )
    bigram_test = evaluate_bigram(bigram, partitions["test"])

    args.output.mkdir(parents=True, exist_ok=True)
    bigram_path = args.output / "bigram-model.json.gz"
    model_path = args.output / "model.safetensors"
    config_path = args.output / "config.json"
    result_path = args.output / "result.json"
    bigram.save_gzip(bigram_path)
    save_file(model.state_dict(), model_path)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    torch.set_num_threads(args.inference_threads)
    timings = benchmark(
        model,
        bigram,
        partitions["test"],
        character_ids=character_ids,
        language_ids=language_ids,
        max_characters=args.max_characters,
        unknown_character_backoff=args.unknown_character_backoff,
        repetitions=args.benchmark_repetitions,
    )
    result = {
        "schema": SCHEMA,
        "status": "completed_pre_onnx_comparison",
        "git_commit": git_commit(),
        "seed": args.seed,
        "input": {
            "path": str(args.context_jsonl.resolve()),
            "sha256": file_sha256(args.context_jsonl),
            **loading,
        },
        "partition": {
            "rule": "sha256(NFKC-whitespace-folded-casefolded-surface), 80/10/10",
            "counts": {name: len(rows) for name, rows in partitions.items()},
            "cells": {name: cell_counts(rows) for name, rows in partitions.items()},
            "normalized_key_overlap": 0,
        },
        "sampling": {
            "expected_language_mass": language_mass,
            "samples_per_epoch": args.samples_per_epoch,
            "receipt": language_receipt,
        },
        "training": {
            "maximum_epochs": args.maximum_epochs,
            "patience": args.patience,
            "case_intent_weights": {
                "natural_or_reconstructed": args.case_natural_weight,
                "all_uppercase": args.case_uppercase_weight,
                "all_lowercase": args.case_lowercase_weight,
            },
            "uppercase_publisher_policy": (
                "match casefolded mixed-case resources, else title-case; raw Census/INSEE uppercase excluded"
            ),
            "trainlib": {
                "weighted_length_batch_sampler": True,
                "validation_controller": True,
                "length_window_steps": args.length_window_steps,
            },
            "optimizer": {
                "name": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": 1e-4,
                "schedule": "cosine annealing to zero over maximum_epochs",
            },
            "initialization": initialization,
            "selected_epoch": selected_epoch(history, initial_validation),
            "completed_epochs": len(history),
            "optimizer_steps": steps,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "character_backoff": character_backoff_receipt,
            "history": history,
        },
        "metrics": {
            "selection_metric": "development.natural_language.macro_f1_observed_roles",
            "primary_test_metric": "test.natural_language.macro_f1_observed_roles",
            "initial_checkpoint_development": initial_validation,
            "bigram_development": bigram_development,
            "neural_development": development,
            "bigram_test": bigram_test,
            "neural_test": test,
        },
        "artifacts": {
            "bigram_model": {
                "path": str(bigram_path.resolve()),
                "bytes": bigram_path.stat().st_size,
                "sha256": file_sha256(bigram_path),
                "compression": "gzip",
            },
            "neural_model": {
                "path": str(model_path.resolve()),
                "bytes": model_path.stat().st_size,
                "sha256": file_sha256(model_path),
            },
            "neural_config": {
                "path": str(config_path.resolve()),
                "bytes": config_path.stat().st_size,
                "sha256": file_sha256(config_path),
            },
        },
        "performance": {
            "training_threads": args.training_threads,
            "inference_threads": args.inference_threads,
            "repetitions": args.benchmark_repetitions,
            "before": before,
            "after": host_snapshot(),
            "timings": timings,
            "grade": "diagnostic local shared-host comparison",
        },
        "decision_rule": {
            "export_onnx_only_if": "neural natural-case held-out macro F1 exceeds the bigram baseline",
            "onnx_exported": False,
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": "complete",
                "result": str(result_path.resolve()),
                "bigram_test_natural_macro_f1": bigram_test["natural"]["macro_f1_observed_roles"],
                "neural_test_natural_language_macro_f1": test["natural_language"]["macro_f1_observed_roles"],
                "neural_test_natural_shared_only_macro_f1": test["natural_shared_only"][
                    "macro_f1_observed_roles"
                ],
            }
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-evaluate")
    fit.add_argument("--context-jsonl", type=Path, required=True)
    fit.add_argument("--language-round", type=Path, default=DEFAULT_LANGUAGE_ROUND)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument(
        "--init-from-output",
        type=Path,
        help=(
            "initialize from a prior output directory after exact architecture, vocabulary, "
            "language, and character-backoff compatibility checks"
        ),
    )
    fit.add_argument("--seed", type=int, default=155)
    fit.add_argument("--train-cap-per-language-role", type=int, default=40_000)
    fit.add_argument("--evaluation-cap-per-language-role", type=int, default=10_000)
    fit.add_argument(
        "--expansion-mass",
        type=float,
        default=None,
        help="legacy core-versus-expansion allocation; omit for importance/support weighting",
    )
    fit.add_argument("--maximum-support-multiplier", type=float, default=2.0)
    fit.add_argument("--full-support-effective-cell-weight", type=float, default=1000.0)
    fit.add_argument("--bigram-alpha", type=float, default=0.1)
    fit.add_argument("--case-natural-weight", type=float, default=0.7)
    fit.add_argument("--case-uppercase-weight", type=float, default=0.2)
    fit.add_argument("--case-lowercase-weight", type=float, default=0.1)
    count_group = fit.add_mutually_exclusive_group()
    count_group.add_argument(
        "--rare-character-maximum-count",
        type=int,
        default=1,
        help="natural-training count k eligible for block backoff; -1 disables",
    )
    count_group.add_argument(
        "--minimum-character-count",
        type=int,
        default=None,
        help="compatibility spelling: literal threshold N means k=N-1",
    )
    fit.add_argument(
        "--rare-character-backoff-probability",
        type=float,
        default=1.0,
        help="sample-time replacement probability for codepoints with count <= k",
    )
    fit.add_argument(
        "--frequent-character-backoff-probability",
        type=float,
        default=0.0,
        help="optional separate replacement probability for codepoints with count > k",
    )
    fit.add_argument(
        "--unknown-character-backoff",
        choices=UNKNOWN_CHARACTER_BACKOFFS,
        default="script-block",
        help="fallback token family for replaced, pruned, or unseen codepoints",
    )
    fit.add_argument(
        "--maximum-vocabulary-characters",
        type=int,
        default=0,
        help="zero retains every eligible literal after hard rare-character backoff",
    )
    fit.add_argument("--max-characters", type=int, default=48)
    fit.add_argument("--embedding-dim", type=int, default=24)
    fit.add_argument("--convolution-channels", type=int, default=32)
    fit.add_argument("--language-dim", type=int, default=8)
    fit.add_argument("--language-dropout", type=float, default=0.25)
    fit.add_argument("--language-noise", type=float, default=0.15)
    fit.add_argument(
        "--shared-mixture-alpha",
        type=float,
        default=0.2,
        help="unconditional p(role|name) mass; default leaves 80%% language-conditioned mass",
    )
    fit.add_argument("--batch-size", type=int, default=1024)
    fit.add_argument("--length-window-steps", type=int, default=8)
    fit.add_argument("--samples-per-epoch", type=int, default=120_000)
    fit.add_argument("--maximum-epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=5)
    fit.add_argument("--learning-rate", type=float, default=0.002)
    fit.add_argument("--training-threads", type=int, default=8)
    fit.add_argument("--inference-threads", type=int, default=1)
    fit.add_argument("--benchmark-repetitions", type=int, default=5)
    fit.set_defaults(func=fit_evaluate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
