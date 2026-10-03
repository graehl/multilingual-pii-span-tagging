#!/usr/bin/env python3
"""Propose full person-name carriers and component subspans for annotation QC."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

CONFIG_SCHEMA = "pii-name-annotation-qc-profiles"
CONFIG_VERSIONS = frozenset({1, 2, 3, 4})
OUTPUT_SCHEMA = "pii-name-annotation-qc"
OUTPUT_VERSION = 1
COMPONENT_VALUES = frozenset({"given_name", "middle_name", "family_name", "Q"})
COMPONENT_ORDERS = frozenset({"any", "given_first", "family_first"})
ORDER_PREFERENCES = frozenset({"given_first", "family_first"})
PARTITION_POLICIES = frozenset(
    {
        "all_family_prefix_and_suffix_partitions",
        "default_plus_lexicon_supported_family_prefix_and_suffix_partitions",
    }
)
DEFAULT_COMPONENT_SYMBOLS = {
    "given_name": "G",
    "middle_name": "M",
    "family_name": "F",
}
DEFAULT_IGNORED_COMPONENT_VALUES = frozenset({"Q"})
DEFAULT_SEQUENCE_PATTERN = "(?:G|M|F|GM?F|FGM?)"
DEFAULT_NAME_ATOM_JOINER_RANGES = ((39, 39), (45, 45), (700, 700), (8217, 8217))
PROFILE_KEYS = frozenset(
    {
        "maximum_given_spaces",
        "maximum_family_spaces",
        "middle_possible",
        "component_order",
        "partition_policy",
        "allow_comma_inversion",
        "max_join_gap_chars",
        "max_name_atoms",
        "minimum_assignment_margin",
        "retain_given_initial_with_middle",
        "weights",
        "family_particles",
        "honorifics",
        "middle_suffixes",
        "compact_script",
        "compact_family_lengths",
        "default_compact_family_length",
        "given_lexicons",
        "family_lexicons",
    }
)
REQUIRED_PROFILE_KEYS = PROFILE_KEYS - {
    "partition_policy",
    "middle_suffixes",
    "maximum_given_spaces",
    "maximum_family_spaces",
    "middle_possible",
    "retain_given_initial_with_middle",
}
# Version-4 configurations written before the compound-family and patronymic
# rules omit these keys; a complete profile must still define the legacy set.
LEGACY_WEIGHT_KEYS = frozenset({"order", "comma", "source_component", "lexicon", "compact_default"})
WEIGHT_KEYS = LEGACY_WEIGHT_KEYS | {"compound_family", "middle_suffix", "middle_presence"}
# Scripts whose names are written without word spacing, so a configured honorific
# may be attached directly to the name atom it follows (王医生, 山田さん, 김민수님).
ATTACHED_HONORIFIC_SCRIPTS = frozenset({"Han", "Japanese", "Hangul"})
ANNOTATION_CONTRACT_KEYS = frozenset(
    {
        "carrier_labels",
        "component_labels",
        "suppress_name_when_exactly_relabeled_as",
        "projection_values",
    }
)
DEFAULT_SECTION_KEYS = frozenset({"grammar", "limits", "scoring", "lexicons"})
GRAMMAR_KEYS = frozenset({"component_order", "allow_comma_inversion", "compact_name"})
EXPLICIT_GRAMMAR_KEYS = frozenset(
    {"order_preference", "partition_policy", "allow_comma_inversion", "compact_name"}
)
# Optional so every existing grammar stays valid without restating the default.
OPTIONAL_GRAMMAR_KEYS = frozenset(
    {
        "maximum_given_spaces",
        "maximum_family_spaces",
        "middle_possible",
    }
)
# Whether the language has a middle-name slot at all. Spanish has none: a nombre may be
# compound, and the surplus atoms belong to it, so assigning them a middle produces a
# component the convention does not have. A language that says so never generates a shape
# with a middle, rather than generating one and repairing it afterwards.
COMPACT_NAME_KEYS = frozenset({"script", "family_lengths", "default_family_length"})
REQUIRED_LIMIT_KEYS = frozenset({"max_join_gap_chars", "max_name_atoms"})
# Per-language component width bounds: the widest given/family surface (in interior
# spaces) the neural scorer has training support for. Wider candidates are extrapolation
# and cost scoring time, so a language may cap them here; unset means no cap.
LIMIT_KEYS = REQUIRED_LIMIT_KEYS | {"maximum_given_spaces", "maximum_family_spaces"}
REQUIRED_SCORING_KEYS = frozenset({"minimum_assignment_margin", "weights"})
SCORING_KEYS = REQUIRED_SCORING_KEYS | {"retain_given_initial_with_middle"}
LEGACY_LEXICON_KEYS = frozenset({"family_particles", "honorifics", "given_files", "family_files"})
LEXICON_KEYS = LEGACY_LEXICON_KEYS | {"middle_suffixes"}
LANGUAGE_SECTION_KEYS = frozenset({"grammar", "limits", "scoring", "lexicons"})
GRAMMAR_SCRIPT_KEYS = frozenset({"positive", "negative"})
NameKindScorer = Callable[[list[str]], list[dict[str, float]]]


@dataclass(frozen=True)
class SourceSpan:
    index: int
    start: int
    end: int
    label: str
    score: float | None
    raw: Any


@dataclass(frozen=True)
class KnownComponent:
    start: int
    end: int
    value: str
    source: str


@dataclass(frozen=True)
class NameAtom:
    start: int
    end: int
    text: str
    key: str


@dataclass(frozen=True)
class ComponentDraft:
    start: int
    end: int
    value: str


@dataclass(frozen=True)
class AssignmentCandidate:
    kind: str
    order: str
    components: tuple[ComponentDraft, ...]
    score: float
    evidence: tuple[str, ...]
    # atoms in the given component; the tie-break below the score
    given_run: int = 1


@dataclass(frozen=True)
class GrammarUnicodeRanges:
    positive: tuple[tuple[int, int], ...]
    negative: tuple[tuple[int, int], ...]

    def accepts(self, text: str, start: int, end: int) -> bool:
        for character in text[start:end]:
            codepoint = ord(character)
            if not _in_ranges(codepoint, self.positive) or _in_ranges(codepoint, self.negative):
                return False
        return True


@dataclass(frozen=True)
class Profile:
    name: str
    language_rule: str
    grammar_script: str | None
    component_order: str
    order_preference: str | None
    partition_policy: str
    maximum_given_spaces: int
    maximum_family_spaces: int
    middle_possible: bool
    allow_comma_inversion: bool
    max_join_gap_chars: int
    max_name_atoms: int
    minimum_assignment_margin: float
    retain_given_initial_with_middle: bool
    weights: dict[str, float]
    family_particles: frozenset[str]
    honorifics: frozenset[str]
    middle_suffixes: frozenset[str]
    compact_script: str | None
    compact_family_lengths: tuple[int, ...]
    default_compact_family_length: int | None
    given_names: frozenset[str]
    family_names: frozenset[str]
    component_symbols: dict[str, str]
    ignored_component_values: frozenset[str]
    sequence_pattern: str
    name_atom_joiner_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NameQcConfig:
    path: Path
    carrier_labels: frozenset[str]
    component_labels: dict[str, str]
    name_suppression_labels: frozenset[str]
    projection_values: dict[str, frozenset[str]]
    default_profile: dict[str, Any]
    default_grammar: str
    profile_overrides: dict[str, dict[str, Any]]
    language_profiles: dict[str, str]
    language_overrides: dict[str, dict[str, Any]]
    grammar_scripts: dict[str, GrammarUnicodeRanges]
    language_grammar_scripts: dict[str, str]
    component_symbols: dict[str, str]
    ignored_component_values: frozenset[str]
    sequence_pattern: str
    name_atom_joiner_ranges: tuple[tuple[int, int], ...]
    name_kind_model: dict[str, Any] | None


def normalized_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _nonnegative_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    if value < 0:
        raise ValueError(f"{where} must be nonnegative")
    return float(value)


def _positive_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{where} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{where} must not contain duplicates")
    return value


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def _unicode_ranges(raw: Any, where: str, *, nonempty: bool) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list) or (nonempty and not raw) or len(raw) % 2:
        requirement = "nonempty inclusive endpoint pairs" if nonempty else "inclusive endpoint pairs"
        raise ValueError(f"{where} must contain {requirement}")
    endpoints = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0x10FFFF:
            raise ValueError(f"{where} must contain integer Unicode endpoints")
        endpoints.append(value)
    ranges = tuple(zip(endpoints[::2], endpoints[1::2]))
    if any(start > end for start, end in ranges):
        raise ValueError(f"{where} contains a descending range")
    return ranges


def _grammar_unicode_ranges(raw: Any, where: str) -> GrammarUnicodeRanges:
    if not isinstance(raw, dict) or set(raw) != GRAMMAR_SCRIPT_KEYS:
        raise ValueError(f"{where} must define positive and negative ranges")
    return GrammarUnicodeRanges(
        positive=_unicode_ranges(raw["positive"], f"{where}.positive", nonempty=True),
        negative=_unicode_ranges(raw["negative"], f"{where}.negative", nonempty=False),
    )


def _validate_profile(raw: dict[str, Any], where: str, *, complete: bool) -> None:
    if not isinstance(raw, dict) or set(raw) - PROFILE_KEYS:
        raise ValueError(f"{where} has unsupported fields")
    if complete and not REQUIRED_PROFILE_KEYS <= set(raw):
        raise ValueError(f"{where} must define every profile field")
    if "component_order" in raw and raw["component_order"] not in COMPONENT_ORDERS:
        raise ValueError(f"{where}.component_order is invalid")
    for key in ("maximum_given_spaces", "maximum_family_spaces"):
        if key in raw:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{where}.{key} must be a nonnegative integer")
    for key in ("middle_possible", "retain_given_initial_with_middle"):
        if key in raw and not isinstance(raw[key], bool):
            raise ValueError(f"{where}.{key} must be a boolean")
    if "partition_policy" in raw and raw["partition_policy"] not in PARTITION_POLICIES:
        raise ValueError(f"{where}.partition_policy is invalid")
    if "allow_comma_inversion" in raw and not isinstance(raw["allow_comma_inversion"], bool):
        raise ValueError(f"{where}.allow_comma_inversion must be boolean")
    for field in ("max_join_gap_chars", "max_name_atoms"):
        if field in raw:
            _positive_integer(raw[field], f"{where}.{field}")
    if "minimum_assignment_margin" in raw:
        _nonnegative_number(raw["minimum_assignment_margin"], f"{where}.minimum_assignment_margin")
    if "weights" in raw:
        weights = raw["weights"]
        if not isinstance(weights, dict) or set(weights) - WEIGHT_KEYS:
            raise ValueError(f"{where}.weights has unsupported fields")
        if complete and not LEGACY_WEIGHT_KEYS <= set(weights):
            raise ValueError(f"{where}.weights must define every weight")
        for key, value in weights.items():
            _nonnegative_number(value, f"{where}.weights.{key}")
    for field in (
        "family_particles",
        "honorifics",
        "middle_suffixes",
        "given_lexicons",
        "family_lexicons",
    ):
        if field in raw:
            _string_list(raw[field], f"{where}.{field}")
    if "compact_script" in raw and raw["compact_script"] not in {None, "Han", "Hangul"}:
        raise ValueError(f"{where}.compact_script must be null, Han, or Hangul")
    if "compact_family_lengths" in raw:
        lengths = raw["compact_family_lengths"]
        if not isinstance(lengths, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in lengths
        ):
            raise ValueError(f"{where}.compact_family_lengths must contain positive integers")
        if len(lengths) != len(set(lengths)):
            raise ValueError(f"{where}.compact_family_lengths must not contain duplicates")
    if "default_compact_family_length" in raw and raw["default_compact_family_length"] is not None:
        _positive_integer(
            raw["default_compact_family_length"],
            f"{where}.default_compact_family_length",
        )


def _annotation_contract(
    raw: Any,
    where: str,
) -> tuple[frozenset[str], dict[str, str], frozenset[str], dict[str, frozenset[str]]]:
    if not isinstance(raw, dict) or set(raw) != ANNOTATION_CONTRACT_KEYS:
        raise ValueError(f"{where} has an invalid shape")
    carrier_labels = frozenset(_string_list(raw["carrier_labels"], f"{where}.carrier_labels"))
    component_labels = raw["component_labels"]
    if (
        not isinstance(component_labels, dict)
        or set(component_labels.values()) != {"given_name", "middle_name", "family_name"}
        or any(not isinstance(key, str) or not key for key in component_labels)
    ):
        raise ValueError(f"{where}.component_labels must map labels onto all three component values")
    if not set(component_labels) <= carrier_labels:
        raise ValueError(f"{where}: component labels must also be carrier labels")
    suppression = frozenset(
        _string_list(
            raw["suppress_name_when_exactly_relabeled_as"],
            f"{where}.suppress_name_when_exactly_relabeled_as",
        )
    )
    if suppression & carrier_labels:
        raise ValueError(f"{where}: name suppression labels cannot also be carrier labels")
    projection_values = raw["projection_values"]
    if not isinstance(projection_values, dict) or not projection_values:
        raise ValueError(f"{where}.projection_values must be a nonempty object")
    projections = {}
    for field, values in projection_values.items():
        if not isinstance(field, str) or not field:
            raise ValueError(f"{where}: projection field names must be nonempty strings")
        projections[field] = frozenset(_string_list(values, f"{where}.projection_values.{field}"))
    return carrier_labels, dict(component_labels), suppression, projections


def _normalized_language_key(value: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a nonempty language key")
    key = normalized_key(value).replace("_", "-")
    if not key:
        raise ValueError(f"{where} normalizes to an empty language key")
    return key


def _merge_profile(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if key == "weights":
            target["weights"].update(value)
        else:
            target[key] = value


def _grammar_profile(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != GRAMMAR_KEYS:
        raise ValueError(f"{where} must define component_order, allow_comma_inversion, and compact_name")
    compact = raw["compact_name"]
    if compact is not None and (not isinstance(compact, dict) or set(compact) != COMPACT_NAME_KEYS):
        raise ValueError(f"{where}.compact_name has an invalid shape")
    result = {
        "component_order": raw["component_order"],
        "partition_policy": (
            "all_family_prefix_and_suffix_partitions"
            if raw["component_order"] == "any"
            else "default_plus_lexicon_supported_family_prefix_and_suffix_partitions"
        ),
        "allow_comma_inversion": raw["allow_comma_inversion"],
        "compact_script": None if compact is None else compact["script"],
        "compact_family_lengths": [] if compact is None else compact["family_lengths"],
        "default_compact_family_length": None if compact is None else compact["default_family_length"],
    }
    _validate_profile(result, where, complete=False)
    return result


def _explicit_grammar_profile(raw: Any, where: str) -> dict[str, Any]:
    if (
        not isinstance(raw, dict)
        or not EXPLICIT_GRAMMAR_KEYS <= set(raw)
        or set(raw) - EXPLICIT_GRAMMAR_KEYS - OPTIONAL_GRAMMAR_KEYS
    ):
        raise ValueError(
            f"{where} must define order_preference, partition_policy, allow_comma_inversion, and compact_name"
        )
    order_preference = raw["order_preference"]
    if order_preference is not None and order_preference not in ORDER_PREFERENCES:
        raise ValueError(f"{where}.order_preference is invalid")
    result = _grammar_profile(
        {
            "component_order": order_preference or "any",
            "allow_comma_inversion": raw["allow_comma_inversion"],
            "compact_name": raw["compact_name"],
        },
        where,
    )
    result["partition_policy"] = raw["partition_policy"]
    for key in OPTIONAL_GRAMMAR_KEYS:
        if key in raw:
            result[key] = raw[key]
    _validate_profile(result, where, complete=False)
    return result


def _settings_profile(raw: Any, where: str, *, complete: bool) -> dict[str, Any]:
    allowed = {"limits", "scoring", "lexicons"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError(f"{where} has unsupported fields")
    if complete and set(raw) != allowed:
        raise ValueError(f"{where} must define limits, scoring, and lexicons")
    result: dict[str, Any] = {}
    if "limits" in raw:
        limits = raw["limits"]
        if not isinstance(limits, dict) or set(limits) - LIMIT_KEYS:
            raise ValueError(f"{where}.limits has unsupported fields")
        if complete and not REQUIRED_LIMIT_KEYS <= set(limits):
            raise ValueError(f"{where}.limits must define every limit")
        for key in ("maximum_given_spaces", "maximum_family_spaces"):
            value = limits.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{where}.limits.{key} must be a nonnegative integer")
        result.update(limits)
    if "scoring" in raw:
        scoring = raw["scoring"]
        if not isinstance(scoring, dict) or set(scoring) - SCORING_KEYS:
            raise ValueError(f"{where}.scoring has unsupported fields")
        if complete and not REQUIRED_SCORING_KEYS <= set(scoring):
            raise ValueError(f"{where}.scoring must define margin and weights")
        if "minimum_assignment_margin" in scoring:
            result["minimum_assignment_margin"] = scoring["minimum_assignment_margin"]
        if "retain_given_initial_with_middle" in scoring:
            result["retain_given_initial_with_middle"] = scoring["retain_given_initial_with_middle"]
        if "weights" in scoring:
            weights = scoring["weights"]
            if not isinstance(weights, dict) or set(weights) - WEIGHT_KEYS:
                raise ValueError(f"{where}.scoring.weights has unsupported fields")
            if complete and not LEGACY_WEIGHT_KEYS <= set(weights):
                raise ValueError(f"{where}.scoring.weights must define every weight")
            result["weights"] = dict(weights)
    if "lexicons" in raw:
        lexicons = raw["lexicons"]
        if not isinstance(lexicons, dict) or set(lexicons) - LEXICON_KEYS:
            raise ValueError(f"{where}.lexicons has unsupported fields")
        if complete and not LEGACY_LEXICON_KEYS <= set(lexicons):
            raise ValueError(f"{where}.lexicons must define every lexicon field")
        field_map = {
            "family_particles": "family_particles",
            "honorifics": "honorifics",
            "middle_suffixes": "middle_suffixes",
            "given_files": "given_lexicons",
            "family_files": "family_lexicons",
        }
        for source, target in field_map.items():
            if source in lexicons:
                result[target] = lexicons[source]
    _validate_profile(result, where, complete=False)
    return result


def _component_sequence_contract(
    payload: dict[str, Any],
    *,
    explicit: bool,
) -> tuple[dict[str, str], frozenset[str], str]:
    if not explicit:
        return (
            dict(DEFAULT_COMPONENT_SYMBOLS),
            DEFAULT_IGNORED_COMPONENT_VALUES,
            DEFAULT_SEQUENCE_PATTERN,
        )
    symbols = payload["symbols"]
    ignored = frozenset(_string_list(payload["ignored_values"], "ignored_values"))
    modeled = COMPONENT_VALUES - ignored
    if (
        not isinstance(symbols, dict)
        or set(symbols) != modeled
        or any(
            not isinstance(symbol, str) or len(symbol) != 1 or not symbol.isascii() or not symbol.isalnum()
            for symbol in symbols.values()
        )
        or len(set(symbols.values())) != len(symbols)
    ):
        raise ValueError(
            "symbols must map every non-ignored component value onto one unique ASCII alphanumeric"
        )
    if ignored != {"Q"}:
        raise ValueError("ignored_values must contain exactly Q")
    pattern = payload["sequence_pattern"]
    if not isinstance(pattern, str) or not pattern or len(pattern) > 512:
        raise ValueError("sequence_pattern must be a nonempty bounded regular expression")
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"sequence_pattern is not a valid regular expression: {error}") from error
    semantics = payload["sequence_pattern_semantics"]
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("sequence_pattern_semantics must be nonempty")
    return dict(symbols), ignored, pattern


def postprocessor_identity(config_path: str | Path) -> dict[str, Any]:
    """A digest naming exactly which postprocessor produced a set of name components.

    Rows that receive machine-supplied components record this so a later reader can tell
    them from annotated ones without consulting anything outside the corpus. Hashing the
    profile alone would miss a changed character model, since the profile only names it by
    a relative path, so every file the profile points at is folded in too and listed.
    """
    config_path = Path(config_path)
    digest = hashlib.sha256()
    covered = []
    for path in [config_path, *_referenced_model_files(config_path)]:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
        covered.append(path.name)
    return {
        "source": "name-kind-postprocessor",
        "profile": config_path.name,
        "sha256": digest.hexdigest(),
        "covers": covered,
    }


def _referenced_model_files(config_path: Path) -> list[Path]:
    """The character-model files a profile names, in a stable order."""
    profile = json.loads(config_path.read_text(encoding="utf-8"))
    contract = profile.get("name_kind_model")
    if not isinstance(contract, dict) or not contract.get("config_path"):
        return []
    model_config = config_path.parent / str(contract["config_path"])
    if not model_config.is_file():
        return []
    found = [model_config]
    declared = json.loads(model_config.read_text(encoding="utf-8"))
    # A weights reference may be a bare string or an object with a path, as the ONNX entry
    # is. Missing it would leave a changed model invisible to the digest, which is the one
    # thing the digest exists to catch.
    candidates = []
    for value in declared.values():
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, dict) and isinstance(value.get("path"), str):
            candidates.append(value["path"])
    for value in candidates:
        if not value or Path(value).is_absolute():
            continue
        sibling = model_config.parent / value
        if sibling.is_file() and sibling not in found:
            found.append(sibling)
    return found


def _name_kind_model_contract(raw: Any) -> dict[str, Any]:
    keys = {
        "candidate_scope",
        "config_path",
        "role_components",
        "score_transform",
        "score_weight",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("name_kind_model has an invalid shape")
    if (
        raw["candidate_scope"] != "component subspans permitted by the selected language grammar"
        or raw["score_transform"] != "centered_log_odds"
    ):
        raise ValueError("name_kind_model has an unsupported scoring contract")
    config_path = raw["config_path"]
    if not isinstance(config_path, str) or not config_path or Path(config_path).is_absolute():
        raise ValueError("name_kind_model.config_path must be a nonempty relative path")
    role_components = raw["role_components"]
    if (
        not isinstance(role_components, dict)
        or len(role_components) != 2
        or any(
            not isinstance(role, str) or not role or not isinstance(component, str)
            for role, component in role_components.items()
        )
        or set(role_components.values()) != {"given_name", "family_name"}
    ):
        raise ValueError("name_kind_model.role_components must map two roles one-to-one")
    _nonnegative_number(raw["score_weight"], "name_kind_model.score_weight")
    return dict(raw)


def _load_v1_config(path: Path, payload: dict[str, Any]) -> NameQcConfig:
    required = {
        "schema",
        "version",
        "carrier_labels",
        "component_labels",
        "suppress_name_when_exactly_relabeled_as",
        "projection_values",
        "default_profile",
        "profiles",
        "language_profiles",
    }
    if set(payload) != required:
        raise ValueError("version-1 name QC config has an invalid top-level shape")
    carrier_labels = frozenset(_string_list(payload["carrier_labels"], "carrier_labels"))
    component_labels = payload["component_labels"]
    if (
        not isinstance(component_labels, dict)
        or set(component_labels.values()) != {"given_name", "middle_name", "family_name"}
        or any(not isinstance(key, str) or not key for key in component_labels)
    ):
        raise ValueError("component_labels must map labels onto all three component values")
    if not set(component_labels) <= carrier_labels:
        raise ValueError("component labels must also be carrier labels")
    name_suppression_labels = frozenset(
        _string_list(
            payload["suppress_name_when_exactly_relabeled_as"],
            "suppress_name_when_exactly_relabeled_as",
        )
    )
    if name_suppression_labels & carrier_labels:
        raise ValueError("name suppression labels cannot also be carrier labels")
    projection_values = payload["projection_values"]
    if not isinstance(projection_values, dict) or not projection_values:
        raise ValueError("projection_values must be a nonempty object")
    closed_projections = {}
    for field, values in projection_values.items():
        if not isinstance(field, str) or not field:
            raise ValueError("projection field names must be nonempty strings")
        closed_projections[field] = frozenset(_string_list(values, f"projection_values.{field}"))
    default_profile = payload["default_profile"]
    _validate_profile(default_profile, "default_profile", complete=True)
    profiles = payload["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("profiles must be a nonempty object")
    for name, raw in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("profile names must be nonempty strings")
        _validate_profile(raw, f"profiles.{name}", complete=False)
    language_profiles = payload["language_profiles"]
    if not isinstance(language_profiles, dict) or any(
        not isinstance(language, str) or not language or profile not in profiles
        for language, profile in language_profiles.items()
    ):
        raise ValueError("language_profiles must map language keys onto declared profiles")
    return NameQcConfig(
        path=path.resolve(),
        carrier_labels=carrier_labels,
        component_labels=dict(component_labels),
        name_suppression_labels=name_suppression_labels,
        projection_values=closed_projections,
        default_profile=dict(default_profile),
        default_grammar="default",
        profile_overrides={name: dict(raw) for name, raw in profiles.items()},
        language_profiles={
            normalized_key(key).replace("_", "-"): value for key, value in language_profiles.items()
        },
        language_overrides={},
        grammar_scripts={},
        language_grammar_scripts={},
        component_symbols=dict(DEFAULT_COMPONENT_SYMBOLS),
        ignored_component_values=DEFAULT_IGNORED_COMPONENT_VALUES,
        sequence_pattern=DEFAULT_SEQUENCE_PATTERN,
        name_atom_joiner_ranges=DEFAULT_NAME_ATOM_JOINER_RANGES,
        name_kind_model=None,
    )


def _load_modern_config(path: Path, payload: dict[str, Any]) -> NameQcConfig:
    version = payload["version"]
    explicit = version == 4
    required = {"schema", "version", "annotation_contract", "defaults", "grammars", "languages"}
    allowed = required | {"grammar_scripts", "name_atom_joiners"}
    if version in {3, 4}:
        required |= {"grammar_scripts", "name_kind_model"}
        allowed |= {"name_kind_model"}
    if explicit:
        sequence_keys = {
            "symbols",
            "ignored_values",
            "name_atom_joiners",
            "sequence_pattern",
            "sequence_pattern_semantics",
        }
        required |= sequence_keys
        allowed |= sequence_keys
    if not required <= set(payload) or set(payload) - allowed:
        raise ValueError(f"version-{version} name QC config has an invalid top-level shape")
    component_symbols, ignored_component_values, sequence_pattern = _component_sequence_contract(
        payload,
        explicit=explicit,
    )
    name_atom_joiner_ranges = (
        _unicode_ranges(
            payload["name_atom_joiners"],
            "name_atom_joiners",
            nonempty=True,
        )
        if "name_atom_joiners" in payload
        else DEFAULT_NAME_ATOM_JOINER_RANGES
    )
    name_kind_model = _name_kind_model_contract(payload["name_kind_model"]) if version in {3, 4} else None
    carrier_labels, component_labels, suppression, projections = _annotation_contract(
        payload["annotation_contract"],
        "annotation_contract",
    )
    grammars = payload["grammars"]
    if not isinstance(grammars, dict) or not grammars:
        raise ValueError("grammars must be a nonempty object")
    grammar_profiles = {}
    for name, raw in grammars.items():
        if not isinstance(name, str) or not name:
            raise ValueError("grammar names must be nonempty strings")
        grammar_profiles[name] = (
            _explicit_grammar_profile(raw, f"grammars.{name}")
            if explicit
            else _grammar_profile(raw, f"grammars.{name}")
        )
    defaults = payload["defaults"]
    if not isinstance(defaults, dict) or set(defaults) != DEFAULT_SECTION_KEYS:
        raise ValueError("defaults must define grammar, limits, scoring, and lexicons")
    default_grammar = defaults["grammar"]
    if default_grammar not in grammar_profiles:
        raise ValueError("defaults.grammar must name a declared grammar")
    default_profile = _settings_profile(
        {key: defaults[key] for key in ("limits", "scoring", "lexicons")},
        "defaults",
        complete=True,
    )
    _merge_profile(default_profile, grammar_profiles[default_grammar])
    _validate_profile(default_profile, "defaults", complete=True)
    grammar_scripts = {}
    if "grammar_scripts" in payload:
        raw_scripts = payload["grammar_scripts"]
        if not isinstance(raw_scripts, dict) or not raw_scripts:
            raise ValueError("grammar_scripts must be a nonempty object")
        for name, raw in raw_scripts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("grammar_scripts keys must be nonempty strings")
            grammar_scripts[name] = _grammar_unicode_ranges(raw, f"grammar_scripts.{name}")
    languages = payload["languages"]
    if not isinstance(languages, dict):
        raise ValueError("languages must be an object")
    language_profiles = {}
    language_overrides = {}
    language_grammar_scripts = {}
    for raw_language, raw in languages.items():
        language = _normalized_language_key(raw_language, "languages key")
        if language in language_profiles:
            raise ValueError(f"languages contains duplicate normalized key {language!r}")
        allowed_language_keys = LANGUAGE_SECTION_KEYS | ({"grammar_script"} if grammar_scripts else set())
        if (
            not isinstance(raw, dict)
            or set(raw) - allowed_language_keys
            or "grammar" not in raw
            or raw["grammar"] not in grammar_profiles
        ):
            raise ValueError(f"languages.{raw_language} must name a declared grammar")
        language_profiles[language] = raw["grammar"]
        grammar_script = raw.get("grammar_script")
        if version in {3, 4} and grammar_script is None:
            raise ValueError(f"languages.{raw_language} must name a grammar_script")
        if grammar_script is not None:
            if not isinstance(grammar_script, str) or grammar_script not in grammar_scripts:
                raise ValueError(f"languages.{raw_language} names an unknown grammar_script")
            language_grammar_scripts[language] = grammar_script
        language_overrides[language] = _settings_profile(
            {key: value for key, value in raw.items() if key not in {"grammar", "grammar_script"}},
            f"languages.{raw_language}",
            complete=False,
        )
    return NameQcConfig(
        path=path.resolve(),
        carrier_labels=carrier_labels,
        component_labels=component_labels,
        name_suppression_labels=suppression,
        projection_values=projections,
        default_profile=default_profile,
        default_grammar=default_grammar,
        profile_overrides=grammar_profiles,
        language_profiles=language_profiles,
        language_overrides=language_overrides,
        grammar_scripts=grammar_scripts,
        language_grammar_scripts=language_grammar_scripts,
        component_symbols=component_symbols,
        ignored_component_values=ignored_component_values,
        sequence_pattern=sequence_pattern,
        name_atom_joiner_ranges=name_atom_joiner_ranges,
        name_kind_model=name_kind_model,
    )


def load_config(path: Path) -> NameQcConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("name QC config must be an object")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("version") not in CONFIG_VERSIONS:
        raise ValueError("unsupported name QC config schema or version")
    if payload["version"] == 1:
        return _load_v1_config(path, payload)
    return _load_modern_config(path, payload)


def _lexicon(path: Path) -> frozenset[str]:
    values = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            value = line.rstrip("\r\n")
            if not value or value.lstrip().startswith("#"):
                continue
            value = value.split("\t", 1)[0]
            key = normalized_key(value)
            if not key:
                raise ValueError(f"{path}:{line_number}: empty normalized lexicon entry")
            values.add(key)
    return frozenset(values)


class NameAnnotationQc:
    def __init__(
        self,
        config: NameQcConfig,
        *,
        given_lexicons: Iterable[Path] = (),
        family_lexicons: Iterable[Path] = (),
        name_kind_deployment: Any | None = None,
        mode: str = "insert-name-kinds",
        max_name_gap_chars: int = 5,
        apply_name_grammar: bool = True,
        override_name_kinds: bool = False,
    ) -> None:
        if mode not in {"insert-name-kinds", "infer-person-name"}:
            raise ValueError(f"unknown name processor mode {mode!r}")
        if (
            isinstance(max_name_gap_chars, bool)
            or not isinstance(max_name_gap_chars, int)
            or max_name_gap_chars < 0
        ):
            raise ValueError("max_name_gap_chars must be a nonnegative integer")
        self.mode = mode
        self.max_name_gap_chars = max_name_gap_chars
        self.apply_name_grammar = apply_name_grammar
        if override_name_kinds and mode != "insert-name-kinds":
            raise ValueError("override_name_kinds requires insert-name-kinds mode")
        self.override_name_kinds = override_name_kinds
        self.config = config
        self.extra_given_lexicons = tuple(path.resolve() for path in given_lexicons)
        self.extra_family_lexicons = tuple(path.resolve() for path in family_lexicons)
        self._profile_cache: dict[str, Profile] = {}
        self._lexicon_cache: dict[Path, frozenset[str]] = {}
        self._name_kind_deployment = name_kind_deployment
        self._name_kind_role_components: dict[str, str] = {}
        self._name_kind_score_weight = 0.0
        if config.name_kind_model is not None and mode == "insert-name-kinds":
            model_contract = config.name_kind_model
            if self._name_kind_deployment is None:
                from scripts.pii_name_role_publish import NameKindOnnxDeployment

                model_path = config.path.parent / model_contract["config_path"]
                self._name_kind_deployment = NameKindOnnxDeployment.load(model_path)
            self._name_kind_role_components = dict(model_contract["role_components"])
            roles = self._name_kind_deployment.config.get("roles")
            if (
                not isinstance(roles, list)
                or len(roles) != len(self._name_kind_role_components)
                or set(roles) != set(self._name_kind_role_components)
            ):
                raise ValueError("name-kind deployment roles disagree with name_kind_model.role_components")
            self._name_kind_score_weight = float(model_contract["score_weight"])

    def _name_kind_scores(
        self,
        surfaces: list[str],
        language: str,
    ) -> list[dict[str, float]]:
        if self._name_kind_deployment is None:
            return []
        _, log_probabilities = self._name_kind_deployment.infer(surfaces, language)
        roles = self._name_kind_deployment.config["roles"]
        if len(log_probabilities) != len(surfaces):
            raise ValueError("name-kind deployment returned the wrong batch size")
        result = []
        for row in log_probabilities:
            if len(row) != len(roles):
                raise ValueError("name-kind deployment returned the wrong role width")
            values = [float(value) for value in row]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("name-kind deployment returned a nonfinite score")
            center = sum(values) / len(values)
            result.append(
                {self._name_kind_role_components[role]: value - center for role, value in zip(roles, values)}
            )
        return result

    def _load_lexicons(self, values: list[str], extra: tuple[Path, ...]) -> frozenset[str]:
        paths = [self.config.path.parent / value for value in values]
        paths.extend(extra)
        result = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in self._lexicon_cache:
                self._lexicon_cache[resolved] = _lexicon(resolved)
            result.update(self._lexicon_cache[resolved])
        return frozenset(result)

    def profile(self, language: str) -> Profile:
        if not isinstance(language, str) or not language:
            raise ValueError("language must be a nonempty string")
        language_key = normalized_key(language).replace("_", "-")
        base_language = language_key.split("-", 1)[0]
        direct_grammar = language if language in self.config.profile_overrides else None
        if base_language == "und":
            language_rule = "default"
            profile_name = self.config.default_grammar
            direct_grammar = None
        elif direct_grammar is not None:
            language_rule = direct_grammar
            profile_name = direct_grammar
        else:
            language_rule = (
                language_key
                if language_key in self.config.language_profiles
                else base_language
                if base_language in self.config.language_profiles
                else "default"
            )
            profile_name = self.config.language_profiles.get(
                language_rule,
                self.config.default_grammar,
            )
        cache_key = f"{language_rule}:{profile_name}:{language_key}"
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]
        merged = dict(self.config.default_profile)
        merged["weights"] = dict(merged["weights"])
        if profile_name in self.config.profile_overrides:
            _merge_profile(merged, self.config.profile_overrides[profile_name])
        if direct_grammar is None and language_rule in self.config.language_overrides:
            _merge_profile(merged, self.config.language_overrides[language_rule])
        compact_script = merged["compact_script"]
        compact_lengths = tuple(merged["compact_family_lengths"])
        compact_default = merged["default_compact_family_length"]
        if compact_script is None and (compact_lengths or compact_default is not None):
            raise ValueError(f"profile {profile_name!r} defines compact lengths without a script")
        if compact_script is not None and (not compact_lengths or compact_default not in compact_lengths):
            raise ValueError(f"profile {profile_name!r} must declare a default among its compact lengths")
        component_order = merged["component_order"]
        partition_policy = merged.get(
            "partition_policy",
            "all_family_prefix_and_suffix_partitions"
            if component_order == "any"
            else "default_plus_lexicon_supported_family_prefix_and_suffix_partitions",
        )
        profile = Profile(
            name=profile_name,
            language_rule=language_rule,
            grammar_script=(
                None
                if direct_grammar is not None
                else self.config.language_grammar_scripts.get(language_rule)
            ),
            component_order=component_order,
            order_preference=None if component_order == "any" else component_order,
            partition_policy=partition_policy,
            # Spaces permitted inside a component handed to the neural scorer. The
            # default is the carrier's own atom limit, which imposes nothing: multi-atom
            # given and family runs have always been evaluated, so a bound here is a new
            # restriction and belongs to languages that ask for one. Unrelated to
            # max_name_atoms, which bounds the whole carrier rather than one component.
            maximum_given_spaces=int(merged.get("maximum_given_spaces", merged["max_name_atoms"])),
            maximum_family_spaces=int(merged.get("maximum_family_spaces", merged["max_name_atoms"])),
            middle_possible=bool(merged.get("middle_possible", True)),
            allow_comma_inversion=merged["allow_comma_inversion"],
            max_join_gap_chars=merged["max_join_gap_chars"],
            max_name_atoms=merged["max_name_atoms"],
            minimum_assignment_margin=float(merged["minimum_assignment_margin"]),
            retain_given_initial_with_middle=merged.get("retain_given_initial_with_middle", False),
            weights={key: float(value) for key, value in merged["weights"].items()},
            family_particles=frozenset(normalized_key(value) for value in merged["family_particles"]),
            honorifics=frozenset(normalized_key(value) for value in merged["honorifics"]),
            middle_suffixes=frozenset(normalized_key(value) for value in merged.get("middle_suffixes", [])),
            compact_script=compact_script,
            compact_family_lengths=compact_lengths,
            default_compact_family_length=compact_default,
            given_names=self._load_lexicons(merged["given_lexicons"], self.extra_given_lexicons),
            family_names=self._load_lexicons(merged["family_lexicons"], self.extra_family_lexicons),
            component_symbols=dict(self.config.component_symbols),
            ignored_component_values=self.config.ignored_component_values,
            sequence_pattern=self.config.sequence_pattern,
            name_atom_joiner_ranges=self.config.name_atom_joiner_ranges,
        )
        self._profile_cache[cache_key] = profile
        return profile

    def analyze(
        self,
        *,
        row_id: str,
        text: str,
        language: str,
        grammar_language: str | None = None,
        annotations: Any,
        subclass_annotations: Any = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            raise ValueError(f"{row_id}: text must be a string")
        selected_profile = self.profile(grammar_language or language)
        default_profile = self.profile("und")
        spans = _source_spans(text, annotations, self.config, row_id)
        known_components = _known_components(
            text,
            spans,
            subclass_annotations,
            self.config,
            row_id,
        )
        if self.override_name_kinds:
            known_components = []
        suppressed_name_indices = _suppressed_name_indices(spans, self.config)
        if self.mode == "infer-person-name":
            return self._infer_person_names(
                row_id,
                text,
                language,
                grammar_language,
                spans,
                known_components,
                selected_profile,
                default_profile,
                suppressed_name_indices,
                subclass_annotations,
            )
        clusters = _name_clusters(
            text,
            spans,
            self.config,
            selected_profile,
            default_profile,
            suppressed_name_indices,
        )
        if self.override_name_kinds:
            # Change component assignments independently inside retained carriers.
            clusters = [
                [span]
                for cluster in clusters
                for span in cluster
                if span.label not in self.config.component_labels
            ]
        proposal_profiles = [
            _effective_profile(
                text,
                min(span.start for span in cluster),
                max(span.end for span in cluster),
                selected_profile,
                default_profile,
                self.config,
            )
            for cluster in clusters
        ]
        proposals = [
            _proposal(
                text,
                cluster,
                known_components,
                profile,
                (
                    (lambda surfaces: self._name_kind_scores(surfaces, language))
                    if self._name_kind_deployment is not None
                    else None
                ),
                self._name_kind_score_weight,
            )
            for cluster, profile in zip(clusters, proposal_profiles)
        ]
        findings = [finding for proposal in proposals for finding in proposal.pop("findings")]
        name_indices = {
            *suppressed_name_indices,
            *(index for cluster in clusters for index in (span.index for span in cluster)),
        }
        candidate_primary = [_canonical_primary(span) for span in spans if span.index not in name_indices]
        candidate_primary.extend(proposal["carrier"] for proposal in proposals)
        candidate_primary.sort(key=lambda span: (span["start"], span["end"], span["label"]))
        candidate_subclasses = [
            component for proposal in proposals for component in proposal["subclass_spans"]
        ]
        candidate_subclasses.extend(
            dict(item) for item in (subclass_annotations or []) if item.get("family") != "name_component"
        )
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
        return {
            "schema": OUTPUT_SCHEMA,
            "version": OUTPUT_VERSION,
            "id": row_id,
            "language": language,
            "grammar_language": grammar_language or language,
            "profile": selected_profile.name,
            "grammar": selected_profile.name,
            "language_rule": selected_profile.language_rule,
            "grammar_script": selected_profile.grammar_script,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_annotations": annotations,
            "source_subclass_annotations": subclass_annotations,
            "proposals": proposals,
            "findings": findings,
            "summary": {
                "name_groups": len(proposals),
                "finding_counts": counts,
                "review_required": sum(proposal["review_required"] for proposal in proposals),
            },
            "candidate": {
                "status": "review_required"
                if any(proposal["review_required"] for proposal in proposals)
                else "proposed",
                "primary_spans": candidate_primary,
                "subclass_spans": candidate_subclasses,
            },
        }

    def _infer_person_names(
        self,
        row_id: str,
        text: str,
        language: str,
        grammar_language: str | None,
        spans: list[SourceSpan],
        known: list[KnownComponent],
        selected: Profile,
        default: Profile,
        suppressed: set[int],
        subclass_annotations: Any,
    ) -> dict[str, Any]:
        """Infer containers from predictions only; grammar can veto a gap-budget join."""
        components = [k for k in known if k.value != "Q"]
        names = [
            s
            for s in spans
            if _is_name_span(s, self.config) and s.index not in suppressed and s.label != "name_prefix"
        ]
        for k in components:
            if not any(s.start == k.start and s.end == k.end for s in names):
                names.append(SourceSpan(len(spans) + len(names), k.start, k.end, k.value, None, None))
        names.sort(key=lambda s: (s.start, -s.end, s.index))
        name_indices = {s.index for s in names} | suppressed
        barriers = [
            s for s in spans if s.index not in name_indices and s.label not in {"O", "Q", "name_prefix"}
        ]
        groups: list[list[SourceSpan]] = []
        budgets: list[int] = []
        guard_components = list(components)
        for span in names:
            if span.label in self.config.component_labels or span.label in COMPONENT_VALUES:
                continue
            if any(span.start <= k.start and k.end <= span.end for k in components):
                continue
            profile = _effective_profile(text, span.start, span.end, selected, default, self.config)
            inferred, _, _, _, _ = _assign_components(text, span.start, span.end, [], profile)
            guard_components.extend(
                KnownComponent(k.start, k.end, k.value, "container_grammar_guard")
                for k in inferred
                if k.value != "Q"
            )

        def grammar_allows(group: list[SourceSpan]) -> bool:
            start, end = min(s.start for s in group), max(s.end for s in group)
            profile = _effective_profile(text, start, end, selected, default, self.config)
            relevant = sorted(
                {(k.start, k.end, k.value) for k in guard_components if start <= k.start and k.end <= end}
            )
            encoded = ""
            last_end = start
            for begin, finish, value in relevant:
                symbol = profile.component_symbols[value]
                # Consecutive tokens of one component form one grammar symbol.
                if not encoded.endswith(symbol) or text[last_end:begin].strip():
                    encoded += symbol
                last_end = finish
            if encoded and re.fullmatch(profile.sequence_pattern, encoded) is None:
                return False
            return (
                len(_name_atoms(text, start, end, profile.name_atom_joiner_ranges)) <= profile.max_name_atoms
            )

        index = 0
        while index < len(names):
            candidate = [names[index]]
            end = names[index].end
            total_gap = best_gap = 0
            best_end = index + 1
            for next_index in range(index + 1, len(names)):
                span = names[next_index]
                gap = max(0, span.start - end)
                wholes = [
                    s
                    for s in [*candidate, span]
                    if s.label not in self.config.component_labels and s.label not in COMPONENT_VALUES
                ]
                separate_wholes = any(
                    a.end <= b.start or b.end <= a.start for a, b in itertools.combinations(wholes, 2)
                )
                if span.start >= end and (
                    total_gap + gap > self.max_name_gap_chars
                    or any(s.start < span.start and s.end > end for s in barriers)
                    or separate_wholes
                    or "\n" in text[end : span.start]
                    or "\r" in text[end : span.start]
                ):
                    break
                candidate.append(span)
                total_gap += gap
                contained = span.end <= end
                end = max(end, span.end)
                # A legal complete name may have an incomplete prefix (G M -> G M F).
                # Keep looking within the gap budget rather than rejecting that prefix.
                if contained or not self.apply_name_grammar or grammar_allows(candidate):
                    best_end, best_gap = next_index + 1, total_gap
            groups.append(names[index:best_end])
            budgets.append(best_gap)
            index = best_end
        primary = [_canonical_primary(s) for s in spans if s.index not in name_indices]
        subclasses = [dict(item) for item in (subclass_annotations or [])]
        proposals = []
        for group, gap in zip(groups, budgets):
            start, end = min(s.start for s in group), max(s.end for s in group)
            carrier = {"start": start, "end": end, "label": "person_name"}
            primary.append(carrier)
            parts = [
                {
                    "start": k.start,
                    "end": k.end,
                    "family": "name_component",
                    "value": k.value,
                    "carrier_start": start,
                    "carrier_end": end,
                    "type": "person_name",
                }
                for k in components
                if start <= k.start and k.end <= end
            ]
            for item in subclasses:
                if item.get("family") == "name_component" and start <= item["start"] < item["end"] <= end:
                    item.update(carrier_start=start, carrier_end=end, type="person_name")
            existing_parts = {
                (k.get("start"), k.get("end"), k.get("family"), k.get("value")) for k in subclasses
            }
            subclasses.extend(
                k for k in parts if (k["start"], k["end"], k["family"], k["value"]) not in existing_parts
            )
            proposals.append(
                {"carrier": carrier, "subclass_spans": parts, "gap_characters": gap, "review_required": False}
            )
        primary.sort(key=lambda s: (s["start"], s["end"], s["label"]))
        return {
            "schema": OUTPUT_SCHEMA,
            "version": OUTPUT_VERSION,
            "id": row_id,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "language": language,
            "grammar_language": grammar_language or language,
            "profile": selected.name,
            "language_rule": selected.language_rule,
            "mode": self.mode,
            "max_name_gap_chars": self.max_name_gap_chars,
            "apply_name_grammar": self.apply_name_grammar,
            "source_annotations": [s.raw for s in spans],
            "proposals": proposals,
            "findings": [],
            "summary": {"name_groups": len(groups), "finding_counts": {}, "review_required": 0},
            "candidate": {"status": "proposed", "primary_spans": primary, "subclass_spans": subclasses},
        }

    def adjusted_inference_fields(
        self,
        *,
        row_id: str,
        text: str,
        language: str,
        grammar_language: str | None = None,
        predictions: Any,
        subclass_annotations: Any = None,
    ) -> dict[str, Any]:
        """Return optional final-stage primary spans and categorical name sidecars."""
        analysis = self.analyze(
            row_id=row_id,
            text=text,
            language=language,
            grammar_language=grammar_language,
            annotations=predictions,
            subclass_annotations=subclass_annotations,
        )
        return self.inference_fields(analysis)

    def inference_fields(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Expose an analyzed candidate in the ordinary prediction wire format."""
        candidate = analysis["candidate"]
        primary_spans = [
            {key: span[key] for key in ("start", "end", "label", "score") if key in span}
            for span in candidate["primary_spans"]
        ]
        return {
            "preds": primary_spans,
            "subclass_spans": candidate["subclass_spans"],
            "name_oracle": {
                "mode": self.mode,
                "max_name_gap_chars": self.max_name_gap_chars,
                "apply_name_grammar": self.apply_name_grammar,
                "override_name_kinds": self.override_name_kinds,
                "config_sha256": hashlib.sha256(self.config.path.read_bytes()).hexdigest(),
                "profile": analysis["profile"],
                "language_rule": analysis["language_rule"],
                "status": candidate["status"],
                "review_required": analysis["summary"]["review_required"],
            },
        }

    def assign_expected_components(
        self,
        *,
        text: str,
        language: str,
        values: tuple[str, ...],
        grammar_language: str | None = None,
        sequence_pattern: str | None = None,
    ) -> dict[str, Any]:
        """Assign exact subspans under a supplied legal component sequence."""
        if not text or not values or any(value not in COMPONENT_VALUES for value in values):
            raise ValueError("expected name components require text and known component values")
        profile = self.profile(grammar_language or language)
        encoded = "".join(
            profile.component_symbols[value]
            for value in values
            if value not in profile.ignored_component_values
        )
        if re.fullmatch(sequence_pattern or profile.sequence_pattern, encoded) is None:
            return {"status": "illegal_sequence", "components": [], "candidate_count": 0}
        atoms = _name_atoms(text, 0, len(text), profile.name_atom_joiner_ranges)
        units = _conditioned_component_units(text, atoms, profile)
        candidates: list[AssignmentCandidate] = []
        if len(units) >= len(values):
            for cuts in itertools.combinations(range(1, len(units)), len(values) - 1):
                boundaries = (0, *cuts, len(units))
                components = tuple(
                    ComponentDraft(
                        units[boundaries[index]].start,
                        units[boundaries[index + 1] - 1].end,
                        value,
                    )
                    for index, value in enumerate(values)
                )
                non_q = [value for value in values if value not in profile.ignored_component_values]
                order = (
                    "family_first"
                    if "family_name" in non_q
                    and "given_name" in non_q
                    and non_q.index("family_name") < non_q.index("given_name")
                    else "given_first"
                )
                candidates.append(
                    _score_candidate(
                        text,
                        "conditioned_sequence",
                        order,
                        components,
                        [],
                        profile,
                    )
                )
        if self._name_kind_deployment is not None and candidates:
            candidates = _apply_name_kind_scores(
                text,
                candidates,
                lambda surfaces: self._name_kind_scores(surfaces, language),
                self._name_kind_score_weight,
                profile,
            )
        if not candidates:
            return {"status": "unresolved", "components": [], "candidate_count": 0}
        best = max(candidates, key=lambda candidate: (candidate.score, candidate.kind))
        return {
            "status": "assigned",
            "candidate_count": len(candidates),
            "score": best.score,
            "components": [
                {
                    "start": component.start,
                    "end": component.end,
                    "family": "name_component",
                    "value": component.value,
                    "text": text[component.start : component.end],
                }
                for component in best.components
            ],
        }


def _span_parts(item: Any, index: int, row_id: str) -> tuple[int, int, str, float | None]:
    if isinstance(item, dict) and isinstance(item.get("span"), list) and len(item["span"]) == 2:
        start, end = item["span"]
        label = item.get("class")
        score = item.get("score")
    elif isinstance(item, dict):
        start, end = item.get("start"), item.get("end")
        label = item.get("label", item.get("type"))
        score = item.get("score")
    elif isinstance(item, (list, tuple)) and len(item) == 3:
        start, end, label = item
        score = None
    else:
        raise ValueError(f"{row_id}: annotation {index} has an unsupported shape")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not isinstance(label, str)
        or not label
    ):
        raise ValueError(f"{row_id}: annotation {index} has invalid fields")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{row_id}: annotation {index} has an invalid score")
        score = float(score)
    return start, end, label, score


def _source_spans(
    text: str,
    annotations: Any,
    config: NameQcConfig,
    row_id: str,
) -> list[SourceSpan]:
    if not isinstance(annotations, list):
        raise ValueError(f"{row_id}: annotations must be a list")
    spans = []
    for index, item in enumerate(annotations):
        start, end, label, score = _span_parts(item, index, row_id)
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{row_id}: annotation {index} is outside the text")
        if isinstance(item, dict):
            surface = item.get("text", item.get("surface"))
            if surface is not None and surface != text[start:end]:
                raise ValueError(f"{row_id}: annotation {index} text does not match its interval")
        spans.append(SourceSpan(index, start, end, label, score, item))
    return spans


def _is_name_span(span: SourceSpan, config: NameQcConfig) -> bool:
    if span.label in config.carrier_labels:
        return True
    if not isinstance(span.raw, dict):
        return False
    return any(span.raw.get(field) in values for field, values in config.projection_values.items())


def _known_components(
    text: str,
    spans: list[SourceSpan],
    subclass_annotations: Any,
    config: NameQcConfig,
    row_id: str,
) -> list[KnownComponent]:
    known = [
        KnownComponent(span.start, span.end, config.component_labels[span.label], "primary")
        for span in spans
        if span.label in config.component_labels
    ]
    if subclass_annotations is None:
        return known
    if not isinstance(subclass_annotations, list):
        raise ValueError(f"{row_id}: subclass annotations must be a list")
    for index, item in enumerate(subclass_annotations):
        if not isinstance(item, dict) or item.get("family") != "name_component":
            continue
        start, end, value = item.get("start"), item.get("end"), item.get("value")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or value not in COMPONENT_VALUES
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"{row_id}: subclass annotation {index} is invalid")
        surface = item.get("t", item.get("text"))
        if surface is not None and surface != text[start:end]:
            raise ValueError(f"{row_id}: subclass annotation {index} text does not match")
        known.append(KnownComponent(start, end, value, "subclass"))
    return list({(item.start, item.end, item.value): item for item in known}.values())


def _join_kind(gap: str, profile: Profile) -> str | None:
    if len(gap) > profile.max_join_gap_chars or "\n" in gap or "\r" in gap:
        return None
    if gap and gap.isspace():
        return "whitespace"
    if profile.allow_comma_inversion and gap.count(",") == 1 and not gap.replace(",", "").strip():
        return "comma"
    return None


def _name_clusters(
    text: str,
    spans: list[SourceSpan],
    config: NameQcConfig,
    selected_profile: Profile,
    default_profile: Profile,
    suppressed_name_indices: set[int],
) -> list[list[SourceSpan]]:
    names = sorted(
        (span for span in spans if _is_name_span(span, config) and span.index not in suppressed_name_indices),
        key=lambda span: (span.start, -span.end, span.index),
    )
    # A whole-name span (person_name or a projection of it) is the tagger's
    # boundary decision: a cluster never extends past it and two of them never
    # merge. Separately labeled components still coalesce, including across a
    # comma, and comma inversion still applies inside one whole name.
    clusters: list[list[SourceSpan]] = []
    bounds: list[tuple[int, int] | None] = []
    for span in names:
        whole = _is_whole_name(span, config)
        own_bound = (span.start, span.end) if whole else None
        if not clusters:
            clusters.append([span])
            bounds.append(own_bound)
            continue
        bound = bounds[-1]
        if bound is not None:
            bounded = bound[0] <= span.start and span.end <= bound[1]
        else:
            bounded = not whole or all(
                span.start <= item.start and item.end <= span.end for item in clusters[-1]
            )
        if not bounded:
            clusters.append([span])
            bounds.append(own_bound)
            continue
        current_end = max(item.end for item in clusters[-1])
        candidate_end = max(current_end, span.end)
        profile = _effective_profile(
            text,
            min(item.start for item in clusters[-1]),
            candidate_end,
            selected_profile,
            default_profile,
            config,
        )
        join_kind = _join_kind(text[current_end : span.start], profile)
        current_start = min(item.start for item in clusters[-1])
        comma_is_inverted_name = (
            join_kind == "comma"
            and len(
                _name_atoms(
                    text,
                    current_start,
                    current_end,
                    profile.name_atom_joiner_ranges,
                )
            )
            == 1
        )
        if span.start <= current_end or join_kind == "whitespace" or comma_is_inverted_name:
            clusters[-1].append(span)
            bounds[-1] = bound or own_bound
        else:
            clusters.append([span])
            bounds.append(own_bound)
    return clusters


def _is_whole_name(span: SourceSpan, config: NameQcConfig) -> bool:
    """A name carrier the tagger emitted as a whole, as opposed to a component or prefix."""
    return (
        span.label not in config.component_labels
        and span.label != "name_prefix"
        and _is_name_span(span, config)
    )


def _effective_profile(
    text: str,
    start: int,
    end: int,
    selected: Profile,
    default: Profile,
    config: NameQcConfig,
) -> Profile:
    if selected.grammar_script is None:
        return selected
    ranges = config.grammar_scripts[selected.grammar_script]
    return selected if ranges.accepts(text, start, end) else default


def _suppressed_name_indices(spans: list[SourceSpan], config: NameQcConfig) -> set[int]:
    suppressing_intervals = {
        (span.start, span.end) for span in spans if span.label in config.name_suppression_labels
    }
    return {
        span.index
        for span in spans
        if _is_name_span(span, config) and (span.start, span.end) in suppressing_intervals
    }


def _is_letter_or_mark(character: str) -> bool:
    return unicodedata.category(character)[0] in {"L", "M"}


def _name_atoms(
    text: str,
    start: int,
    end: int,
    joiner_ranges: tuple[tuple[int, int], ...] = DEFAULT_NAME_ATOM_JOINER_RANGES,
) -> list[NameAtom]:
    atoms = []
    position = start
    while position < end:
        if not _is_letter_or_mark(text[position]):
            position += 1
            continue
        atom_start = position
        position += 1
        while position < end:
            character = text[position]
            if _is_letter_or_mark(character):
                position += 1
                continue
            if (
                _in_ranges(ord(character), joiner_ranges)
                and position + 1 < end
                and _is_letter_or_mark(text[position - 1])
                and _is_letter_or_mark(text[position + 1])
            ):
                position += 1
                continue
            break
        atom_text = text[atom_start:position]
        atoms.append(NameAtom(atom_start, position, atom_text, normalized_key(atom_text)))
    return atoms


def _conditioned_component_units(
    text: str,
    atoms: list[NameAtom],
    profile: Profile,
) -> list[NameAtom]:
    """Expose compact-script codepoints while preserving other name atoms."""
    units = []
    for atom in atoms:
        if profile.compact_script is None or not all(
            _script_character(character, profile.compact_script) for character in atom.text
        ):
            units.append(atom)
            continue
        units.extend(
            NameAtom(
                atom.start + offset,
                atom.start + offset + 1,
                character,
                normalized_key(character),
            )
            for offset, character in enumerate(atom.text)
        )
    return units


def _script_character(character: str, script: str) -> bool:
    codepoint = ord(character)
    if script == "Han":
        return (
            0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0x20000 <= codepoint <= 0x2FA1F
        )
    if script == "Hangul":
        return 0x1100 <= codepoint <= 0x11FF or 0x3130 <= codepoint <= 0x318F or 0xAC00 <= codepoint <= 0xD7AF
    return False


def _component(start_atom: NameAtom, end_atom: NameAtom, value: str) -> ComponentDraft:
    return ComponentDraft(start_atom.start, end_atom.end, value)


def _default_family_last_start(atoms: list[NameAtom], profile: Profile) -> int:
    family_start = len(atoms) - 1
    while family_start > 1 and atoms[family_start - 1].key in profile.family_particles:
        family_start -= 1
    return family_start


def _family_last_components(
    atoms: list[NameAtom],
    family_start: int,
    given: int,
) -> tuple[ComponentDraft, ...]:
    """Given run of `given` atoms, the interior as a middle, then the family run.

    The run's length is a candidate the scorer decides, not a rule. A model whose
    training data holds compound given surfaces can judge "José María" as one given;
    hardcoding the boundary would deny it the chance.
    """
    components = [_component(atoms[0], atoms[given - 1], "given_name")]
    if given < family_start:
        components.append(_component(atoms[given], atoms[family_start - 1], "middle_name"))
    components.append(_component(atoms[family_start], atoms[-1], "family_name"))
    return tuple(components)


def _given_run_kind(kind: str, given: int) -> str:
    """Name the candidate so a one-atom given keeps its historical kind string."""
    return f"{kind}:given{given}" if given > 1 else kind


def _default_family_first_end(atoms: list[NameAtom], profile: Profile) -> int:
    family_end = 0
    if atoms[0].key in profile.family_particles:
        while family_end + 1 < len(atoms) - 1 and atoms[family_end].key in profile.family_particles:
            family_end += 1
    return family_end


def _family_first_components(
    atoms: list[NameAtom],
    family_end: int,
    given: int,
) -> tuple[ComponentDraft, ...]:
    """Family run, then a given run of `given` atoms, then the interior as a middle."""
    given_start = family_end + 1
    given_end = given_start + given
    components = [_component(atoms[0], atoms[family_end], "family_name")]
    components.append(_component(atoms[given_start], atoms[given_end - 1], "given_name"))
    if given_end < len(atoms):
        components.append(_component(atoms[given_end], atoms[-1], "middle_name"))
    return tuple(components)


def _comma_split(
    text: str,
    start: int,
    end: int,
    atoms: list[NameAtom],
) -> int | None:
    """Index of the first name atom after a single separating comma.

    The caller builds the same shapes from it that it builds for family_first.
    """
    comma_positions = [index for index in range(start, end) if text[index] == ","]
    if len(comma_positions) != 1:
        return None
    comma = comma_positions[0]
    before = [index for index, atom in enumerate(atoms) if atom.end <= comma]
    after = [index for index, atom in enumerate(atoms) if atom.start > comma]
    if not before or not after:
        return None
    return after[0]


def _component_lexicon_key(text: str, component: ComponentDraft) -> str:
    return normalized_key(text[component.start : component.end])


def _score_candidate(
    text: str,
    kind: str,
    order: str,
    components: tuple[ComponentDraft, ...],
    known: list[KnownComponent],
    profile: Profile,
    given_run: int = 1,
) -> AssignmentCandidate:
    score = 0.0
    evidence = []
    middle_weight = profile.weights.get("middle_presence", 0.0)
    if middle_weight and any(component.value == "middle_name" for component in components):
        score += middle_weight
        evidence.append("middle_presence")
    if profile.order_preference is not None and order == profile.order_preference:
        score += profile.weights["order"]
        evidence.append(f"language_order:{order}")
    if kind.startswith("comma_inversion"):
        score += profile.weights["comma"]
        evidence.append("comma_inversion")
    for component in components:
        if component.value == "Q":
            continue
        for annotation in known:
            if annotation.end <= component.start or component.end <= annotation.start:
                continue
            if annotation.value == component.value:
                score += profile.weights["source_component"]
                evidence.append(f"source_agrees:{component.value}")
            elif annotation.value != "Q":
                score -= profile.weights["source_component"]
                evidence.append(f"source_conflicts:{annotation.value}->{component.value}")
        key = _component_lexicon_key(text, component)
        if component.value == "given_name":
            if key in profile.given_names:
                score += profile.weights["lexicon"]
                evidence.append(f"given_lexicon:{key}")
            if key in profile.family_names:
                score -= profile.weights["lexicon"]
                evidence.append(f"family_lexicon_conflict:{key}")
        elif component.value == "family_name":
            core = key.split()[-1]
            family_atoms = _name_atoms(
                text,
                component.start,
                component.end,
                profile.name_atom_joiner_ranges,
            )
            family_hits = [atom.key for atom in family_atoms if atom.key in profile.family_names]
            if key in profile.family_names or core in profile.family_names or family_hits:
                score += profile.weights["lexicon"]
                score += max(0, len(family_hits) - 1) * profile.weights["lexicon"] / 2
                evidence.append(f"family_lexicon:{key}")
            if key in profile.given_names:
                score -= profile.weights["lexicon"]
                evidence.append(f"given_lexicon_conflict:{key}")
            # Languages whose convention is a two-part surname (Spanish paternal +
            # maternal) prefer reading a multi-atom tail as one family name over
            # a middle name plus a one-atom family name.
            compound_weight = profile.weights.get("compound_family", 0.0)
            if (
                compound_weight
                and sum(1 for atom in family_atoms if atom.key not in profile.family_particles) >= 2
            ):
                score += compound_weight
                evidence.append("compound_family")
        suffix_weight = profile.weights.get("middle_suffix", 0.0)
        if suffix_weight and profile.middle_suffixes:
            # A patronymic is morphologically marked (Russian -овна/-ович); such an
            # atom is a middle name, and reading it as given or family is penalised.
            suffixed = any(key.endswith(suffix) for suffix in profile.middle_suffixes)
            if suffixed and component.value == "middle_name":
                score += suffix_weight
                evidence.append(f"middle_suffix:{key}")
            elif suffixed and component.value in {"given_name", "family_name"}:
                score -= suffix_weight
                evidence.append(f"middle_suffix_conflict:{component.value}:{key}")
    return AssignmentCandidate(kind, order, components, score, tuple(evidence), given_run)


def _split_attached_honorifics(atoms: list[NameAtom], profile: Profile) -> list[NameAtom]:
    """Detach a configured honorific written directly after a name atom.

    Only scripts without word spacing qualify, and only honorifics written in
    such a script: a Latin honorific never splits a Latin atom. The remaining
    name part must keep at least one character. The longest matching honorific
    wins, so a configured 선생님 is preferred over its own suffix 님.
    """
    if profile.grammar_script not in ATTACHED_HONORIFIC_SCRIPTS:
        return atoms
    suffixes = sorted(
        (key for key in profile.honorifics if not any("a" <= c <= "z" for c in key)),
        key=len,
        reverse=True,
    )
    if not suffixes:
        return atoms
    result = []
    for atom in atoms:
        if atom.key in profile.honorifics:
            result.append(atom)
            continue
        split_key = next(
            (key for key in suffixes if atom.key.endswith(key) and len(atom.key) > len(key)),
            None,
        )
        if split_key is None:
            result.append(atom)
            continue
        cut = atom.end - len(split_key)
        head = atom.text[: cut - atom.start]
        tail = atom.text[cut - atom.start :]
        if normalized_key(tail) != split_key or not head:
            result.append(atom)
            continue
        result.append(NameAtom(atom.start, cut, head, normalized_key(head)))
        result.append(NameAtom(cut, atom.end, tail, split_key))
    return result


def _compact_assignment(
    text: str,
    atom: NameAtom,
    known: list[KnownComponent],
    profile: Profile,
) -> list[AssignmentCandidate]:
    if (
        profile.compact_script is None
        or not profile.compact_family_lengths
        or not all(_script_character(character, profile.compact_script) for character in atom.text)
    ):
        return []
    candidates = []
    for family_length in profile.compact_family_lengths:
        if family_length >= len(atom.text):
            continue
        split = atom.start + family_length
        components = (
            ComponentDraft(atom.start, split, "family_name"),
            ComponentDraft(split, atom.end, "given_name"),
        )
        candidate = _score_candidate(
            text,
            f"compact_{profile.compact_script.casefold()}",
            "family_first",
            components,
            known,
            profile,
        )
        score = candidate.score
        evidence = list(candidate.evidence)
        if family_length == profile.default_compact_family_length:
            score += profile.weights["compact_default"]
            evidence.append(f"compact_default_family_length:{family_length}")
        candidates.append(
            AssignmentCandidate(candidate.kind, candidate.order, components, score, tuple(evidence))
        )
    return candidates


def _mononym_assignment(
    text: str,
    atom: NameAtom,
    known: list[KnownComponent],
    profile: Profile,
    *,
    allow_neural_candidates: bool = False,
) -> list[AssignmentCandidate]:
    key = atom.key
    explicit_values = {
        component.value for component in known if component.start == atom.start and component.end == atom.end
    }
    supported = set()
    if allow_neural_candidates:
        supported.update({"given_name", "family_name"})
    if key in profile.given_names:
        supported.add("given_name")
    if key in profile.family_names:
        supported.add("family_name")
    if profile.weights["source_component"] > 0:
        supported.update(explicit_values & {"given_name", "family_name", "middle_name"})
    return [
        _score_candidate(
            text,
            "mononym",
            profile.component_order,
            (ComponentDraft(atom.start, atom.end, value),),
            known,
            profile,
        )
        for value in sorted(supported)
    ]


def _candidate_sequence_allowed(candidate: AssignmentCandidate, profile: Profile) -> bool:
    encoded = "".join(
        profile.component_symbols[component.value]
        for component in candidate.components
        if component.value not in profile.ignored_component_values
    )
    return re.fullmatch(profile.sequence_pattern, encoded) is not None


def _apply_name_kind_scores(
    text: str,
    candidates: list[AssignmentCandidate],
    scorer: NameKindScorer,
    score_weight: float,
    profile: Profile,
) -> list[AssignmentCandidate]:
    # The per-language space bounds say nothing about how long a component may be. They
    # bound only what the character model is asked to judge: a surface wider than the
    # model's training distribution contributes no lexical evidence rather than
    # unreliable evidence, and the component is still offered, scored by the
    # deterministic weights alone.
    def within_bound(component: ComponentDraft) -> bool:
        limit = (
            profile.maximum_given_spaces if component.value == "given_name" else profile.maximum_family_spaces
        )
        atoms = _name_atoms(text, component.start, component.end, profile.name_atom_joiner_ranges)
        return len(atoms) <= limit + 1

    intervals = sorted(
        {
            (component.start, component.end)
            for candidate in candidates
            for component in candidate.components
            if component.value in {"given_name", "family_name"} and within_bound(component)
        }
    )
    if not intervals:
        return candidates
    scores = scorer([text[start:end] for start, end in intervals])
    if len(scores) != len(intervals):
        raise ValueError("name-kind scorer returned the wrong batch size")
    by_interval = dict(zip(intervals, scores))
    result = []
    for candidate in candidates:
        score = candidate.score
        evidence = list(candidate.evidence)
        for component in candidate.components:
            if component.value not in {"given_name", "family_name"} or not within_bound(component):
                continue
            component_scores = by_interval[(component.start, component.end)]
            if component.value not in component_scores:
                raise ValueError(f"name-kind scorer omitted component role {component.value!r}")
            contribution = score_weight * component_scores[component.value]
            if not math.isfinite(contribution):
                raise ValueError("name-kind scorer returned a nonfinite contribution")
            score += contribution
            evidence.append(f"name_kind:{component.value}:{contribution:+.6g}")
        result.append(
            AssignmentCandidate(
                candidate.kind,
                candidate.order,
                candidate.components,
                score,
                tuple(evidence),
                candidate.given_run,
            )
        )
    return result


def _assign_components(
    text: str,
    start: int,
    end: int,
    known: list[KnownComponent],
    profile: Profile,
    name_kind_scorer: NameKindScorer | None = None,
    name_kind_score_weight: float = 0.0,
) -> tuple[list[ComponentDraft], AssignmentCandidate | None, float, bool, str | None]:
    atoms = _split_attached_honorifics(
        _name_atoms(text, start, end, profile.name_atom_joiner_ranges), profile
    )
    honorifics = [atom for atom in atoms if atom.key in profile.honorifics]
    name_atoms = [atom for atom in atoms if atom.key not in profile.honorifics]
    q_components = [ComponentDraft(atom.start, atom.end, "Q") for atom in honorifics]
    if not name_atoms:
        return q_components, None, 0.0, True, "no_name_atoms"
    if len(name_atoms) > profile.max_name_atoms:
        return q_components, None, 0.0, True, "too_many_name_atoms"
    candidates = []
    if len(name_atoms) == 1:
        candidates.extend(_compact_assignment(text, name_atoms[0], known, profile))
        if not candidates:
            candidates.extend(
                _mononym_assignment(
                    text,
                    name_atoms[0],
                    known,
                    profile,
                    allow_neural_candidates=name_kind_scorer is not None,
                )
            )
    else:
        family_last_starts = {_default_family_last_start(name_atoms, profile)}
        family_first_ends = {_default_family_first_end(name_atoms, profile)}
        if profile.partition_policy == "all_family_prefix_and_suffix_partitions":
            family_last_starts.update(range(1, len(name_atoms)))
            family_first_ends.update(range(len(name_atoms) - 1))
        for family_start in range(1, len(name_atoms)):
            family_atoms = name_atoms[family_start:]
            family_key = normalized_key(text[family_atoms[0].start : family_atoms[-1].end])
            if family_key in profile.family_names or all(
                atom.key in profile.family_names or atom.key in profile.family_particles
                for atom in family_atoms
            ):
                family_last_starts.add(family_start)
        for family_end in range(0, len(name_atoms) - 1):
            family_atoms = name_atoms[: family_end + 1]
            family_key = normalized_key(text[family_atoms[0].start : family_atoms[-1].end])
            if family_key in profile.family_names or all(
                atom.key in profile.family_names or atom.key in profile.family_particles
                for atom in family_atoms
            ):
                family_first_ends.add(family_end)

        # Every given-run length is a candidate, so the run's boundary is scored rather
        # than fixed by the grammar. A language with no middle-name slot simply never
        # offers the shapes that would produce one, which is why the selected components
        # need no merging afterwards. The per-language space bounds do not appear here:
        # they say what the scorer may be asked about, not how long a component may be.
        def given_runs(interior: int) -> range:
            # A run longer than one atom exists only for the scorer to judge, so offer
            # the alternatives only when a scorer is loaded. Without one the reading is
            # the historical single-atom given, except where the grammar has no middle
            # slot and the interior must therefore belong to the given run.
            first = interior if not profile.middle_possible else 1
            last = interior if name_kind_scorer is not None else first
            return range(first, last + 1)

        candidates.extend(
            _score_candidate(
                text,
                _given_run_kind(f"family_last:{family_start}", given),
                "given_first",
                _family_last_components(name_atoms, family_start, given),
                known,
                profile,
                given,
            )
            for family_start in sorted(family_last_starts)
            for given in given_runs(family_start)
        )
        # A carrier can be all surname in any language: "van Gogh", two apellidos, or
        # "آل مكتوم" standing alone. Spanish clinical records make it common by separating
        # "Nombre:" from "Apellidos:", but nothing about it is Spanish. Without this the
        # only legal multi-atom readings invent a given name out of surname material. It
        # is a candidate rather than a rule, so the scorer still prefers a real given name
        # where the lexicon or order supports one.
        # Offered only where the lexicon supports reading every atom as surname
        # material, the same test the other family candidates use. Unconditional, it
        # displaces clear given-plus-family readings such as "William Tambellini";
        # gated, it stays reachable for two apellidos standing alone.
        whole_key = normalized_key(text[name_atoms[0].start : name_atoms[-1].end])
        if whole_key in profile.family_names or all(
            atom.key in profile.family_names or atom.key in profile.family_particles for atom in name_atoms
        ):
            candidates.append(
                _score_candidate(
                    text,
                    "family_only",
                    "given_first",
                    (_component(name_atoms[0], name_atoms[-1], "family_name"),),
                    known,
                    profile,
                    0,
                )
            )
        candidates.extend(
            _score_candidate(
                text,
                _given_run_kind(f"family_first:{family_end}", given),
                "family_first",
                _family_first_components(name_atoms, family_end, given),
                known,
                profile,
                given,
            )
            for family_end in sorted(family_first_ends)
            for given in given_runs(len(name_atoms) - family_end - 1)
        )
        if profile.allow_comma_inversion:
            split = _comma_split(text, start, end, name_atoms)
            if split is not None:
                candidates.extend(
                    _score_candidate(
                        text,
                        _given_run_kind("comma_inversion", given),
                        "family_first",
                        _family_first_components(name_atoms, split - 1, given),
                        known,
                        profile,
                        given,
                    )
                    for given in given_runs(len(name_atoms) - split)
                )
    had_candidates = bool(candidates)
    # A component is built from the first to the last of its name atoms and so would
    # swallow an honorific standing between them ("his brother Mr. B. Bodén" read as
    # given "his", middle "brother Mr. B"). Such a carrier holds a reference or two
    # people, not one name; no component may overlap a Q atom.
    candidates = [
        candidate
        for candidate in candidates
        if not any(
            honorific.start < component.end and component.start < honorific.end
            for component in candidate.components
            for honorific in honorifics
        )
    ]
    candidates = [candidate for candidate in candidates if _candidate_sequence_allowed(candidate, profile)]
    if not candidates:
        return (
            q_components,
            None,
            0.0,
            True,
            "no_legal_component_sequence" if had_candidates else "unresolved_mononym",
        )
    if name_kind_scorer is not None:
        candidates = _apply_name_kind_scores(
            text,
            candidates,
            name_kind_scorer,
            name_kind_score_weight,
            profile,
        )
    ranked = sorted(candidates, key=lambda candidate: (candidate.score, candidate.kind), reverse=True)
    best = ranked[0]
    margin = best.score - ranked[1].score if len(ranked) > 1 else best.score
    review_required = margin < profile.minimum_assignment_margin
    retain_given_initial = (
        profile.retain_given_initial_with_middle
        and any(component.value == "middle_name" for component in best.components)
        and any(
            atom.start >= component.start
            and atom.end <= component.end
            and len(atom.key) > 1
            and atom.key not in profile.family_particles
            for component in best.components
            if component.value == "family_name"
            for atom in name_atoms
        )
    )
    selected_components = []
    unresolved_initial = False
    for component in best.components:
        key = _component_lexicon_key(text, component)
        independently_known = (component.value == "given_name" and key in profile.given_names) or (
            component.value == "family_name" and key in profile.family_names
        )
        looks_like_initial = len(key) == 1 and (
            text[component.end : component.end + 1] == "."
            or (profile.compact_script is None and text[component.start : component.end].isupper())
        )
        if (
            component.value in {"given_name", "family_name"}
            and looks_like_initial
            and not independently_known
            and not (component.value == "given_name" and retain_given_initial)
        ):
            unresolved_initial = True
            continue
        selected_components.append(component)
    if unresolved_initial:
        review_required = True
    return (
        sorted([*q_components, *selected_components], key=lambda component: (component.start, component.end)),
        best,
        margin,
        review_required,
        "unresolved_edge_initial"
        if unresolved_initial
        else "low_assignment_margin"
        if review_required
        else None,
    )


def _maximal_interval_count(spans: list[SourceSpan]) -> int:
    intervals = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if intervals and span.start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], span.end))
        else:
            intervals.append((span.start, span.end))
    return len(intervals)


def _component_status(
    component: ComponentDraft,
    known: list[KnownComponent],
) -> tuple[str, list[KnownComponent]]:
    exact = [item for item in known if item.start == component.start and item.end == component.end]
    if any(item.value == component.value for item in exact):
        return "agree", exact
    conflicting = [
        item
        for item in known
        if item.end > component.start and component.end > item.start and item.value != component.value
    ]
    if exact or conflicting:
        return "conflict", exact or conflicting
    return "missing", []


def _proposal(
    text: str,
    cluster: list[SourceSpan],
    all_known: list[KnownComponent],
    profile: Profile,
    name_kind_scorer: NameKindScorer | None = None,
    name_kind_score_weight: float = 0.0,
) -> dict[str, Any]:
    start = min(span.start for span in cluster)
    end = max(span.end for span in cluster)
    known = [item for item in all_known if item.start < end and start < item.end]
    components, assignment, margin, review_required, unresolved = _assign_components(
        text,
        start,
        end,
        known,
        profile,
        name_kind_scorer,
        name_kind_score_weight,
    )
    scores = [span.score for span in cluster if span.score is not None]
    carrier = {
        "start": start,
        "end": end,
        "label": "person_name",
        "text": text[start:end],
    }
    if scores:
        carrier["score"] = min(scores)
    findings = []
    has_full_carrier = any(
        span.start == start and span.end == end and span.label == "person_name" for span in cluster
    )
    if _maximal_interval_count(cluster) > 1 and not has_full_carrier:
        findings.append(
            {
                "kind": "oversplit_person_name",
                "span": [start, end],
                "text": text[start:end],
                "source_indices": sorted(span.index for span in cluster),
            }
        )
    subclass_spans = []
    component_details = []
    for component in components:
        status, compared = _component_status(component, known)
        detail = {
            "span": [component.start, component.end],
            "family": "name_component",
            "value": component.value,
            "text": text[component.start : component.end],
            "status": status,
        }
        if compared:
            detail["compared_to"] = [
                {
                    "span": [item.start, item.end],
                    "value": item.value,
                    "source": item.source,
                }
                for item in compared
            ]
        component_details.append(detail)
        subclass_spans.append(
            {
                "carrier_start": start,
                "carrier_end": end,
                "type": "person_name",
                "start": component.start,
                "end": component.end,
                "family": "name_component",
                "value": component.value,
            }
        )
        if status == "missing":
            findings.append(
                {
                    "kind": "missing_name_component_detail",
                    "carrier_span": [start, end],
                    "span": [component.start, component.end],
                    "value": component.value,
                    "text": text[component.start : component.end],
                }
            )
        elif status == "conflict":
            review_required = True
            findings.append(
                {
                    "kind": "name_component_disagreement",
                    "carrier_span": [start, end],
                    "span": [component.start, component.end],
                    "proposed_value": component.value,
                    "text": text[component.start : component.end],
                }
            )
    if unresolved:
        findings.append(
            {
                "kind": "ambiguous_name_components",
                "span": [start, end],
                "text": text[start:end],
                "reason": unresolved,
            }
        )
    return {
        "grammar": profile.name,
        "language_rule": profile.language_rule,
        "grammar_script": profile.grammar_script,
        "carrier": carrier,
        "source_indices": sorted(span.index for span in cluster),
        "source_labels": [span.label for span in sorted(cluster, key=lambda span: span.index)],
        "action": "coalesce" if _maximal_interval_count(cluster) > 1 else "retain_carrier",
        "assignment": None
        if assignment is None
        else {
            "kind": assignment.kind,
            "order": assignment.order,
            "score": assignment.score,
            "runner_up_margin": margin,
            "evidence": list(assignment.evidence),
        },
        "review_required": review_required,
        "components": component_details,
        "subclass_spans": subclass_spans,
        "findings": findings,
    }


def _canonical_primary(span: SourceSpan) -> dict[str, Any]:
    result = {"start": span.start, "end": span.end, "label": span.label}
    if span.score is not None:
        result["score"] = span.score
    return result


def _open_output(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdout, False
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", encoding="utf-8"), True


def _write_rows(path: str, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    output, close = _open_output(path)
    try:
        for row in materialized:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        if close:
            output.close()


def _json_lines(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def _record_annotations(row: dict[str, Any], field: str, row_id: str) -> Any:
    if field != "auto":
        if field not in row:
            raise ValueError(f"{row_id}: missing annotations field {field!r}")
        return row[field]
    for candidate in ("preds", "spans", "annotations", "primary_spans"):
        if candidate in row:
            return row[candidate]
    raise ValueError(f"{row_id}: no supported annotations field found")


def _engine(args: argparse.Namespace) -> NameAnnotationQc:
    return NameAnnotationQc(
        load_config(Path(args.config)),
        given_lexicons=(Path(path) for path in args.given_lexicon),
        family_lexicons=(Path(path) for path in args.family_lexicon),
        mode=getattr(args, "mode", "insert-name-kinds"),
        max_name_gap_chars=getattr(args, "max_name_gap_chars", 5),
        apply_name_grammar=not getattr(args, "no_name_grammar", False),
        override_name_kinds=getattr(args, "override_name_kinds", False),
    )


def audit_native_arrays(args: argparse.Namespace) -> None:
    engine = _engine(args)
    texts = Path(args.texts).read_text(encoding="utf-8").splitlines()
    predictions = _json_lines(Path(args.predictions))
    if len(texts) != len(predictions):
        raise ValueError(f"text/prediction line counts differ: {len(texts)} != {len(predictions)}")
    _write_rows(
        args.output,
        (
            engine.analyze(
                row_id=str(index),
                text=text,
                language=args.language,
                grammar_language=getattr(args, "grammar_language", None),
                annotations=annotations,
            )
            for index, (text, annotations) in enumerate(zip(texts, predictions))
        ),
    )


def _raw_name_components(row: dict[str, Any], text: str, config: NameQcConfig) -> tuple[list[dict], dict]:
    """Recover only valid name tokens; parent intervals are rebuilt by the name processor."""
    from scripts.pii_subclass import _load_strict_json_response

    choices = row["response"]["choices"]
    if len(choices) != 1:
        raise ValueError(f"{row['id']}: raw-name input requires exactly one response choice")
    choice = choices[0]
    status = {"components": 0, "rejected": None}
    if choice["finish_reason"] != "stop":
        return [], {**status, "rejected": "incomplete_response"}
    try:
        payload = _load_strict_json_response(choice["message"]["content"])
    except json.JSONDecodeError:
        return [], {**status, "rejected": "bad_json"}
    if not isinstance(payload, dict) or not isinstance(payload.get("subclass_spans"), list):
        return [], {**status, "rejected": "invalid_subclass_array"}
    parts = [
        p
        for p in payload["subclass_spans"]
        if isinstance(p, dict) and p.get("family") == "name_component" and p.get("value") != "Q"
    ]
    if any("t" not in p for p in parts):
        return [], {**status, "rejected": "invalid_name_component"}
    try:
        known = _known_components(text, [], parts, config, str(row["id"]))
    except ValueError:
        return [], {**status, "rejected": "invalid_name_component"}
    ordered = sorted(known, key=lambda p: (p.start, p.end, p.value))
    if any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
        return [], {**status, "rejected": "conflicting_name_components"}
    # Positive name kinds imply person_name regardless of the claimed parent's
    # type or coordinates. Q is not a positive component and cannot invalidate
    # that implication. Actual component offsets, surfaces and values must pass.
    normalized = [
        {"start": p.start, "end": p.end, "family": "name_component", "value": p.value} for p in ordered
    ]
    return normalized, {**status, "components": len(normalized)}


def audit_records(args: argparse.Namespace) -> None:
    engine = _engine(args)
    records = _json_lines(Path(args.input))
    raw_by_id = None
    if getattr(args, "raw_name_input", None):
        if engine.mode != "infer-person-name":
            raise ValueError("--raw-name-input requires --mode infer-person-name")
        raw_records = _json_lines(Path(args.raw_name_input))
        raw_by_id = {str(row["id"]): row for row in raw_records}
        if len(raw_by_id) != len(raw_records):
            raise ValueError("--raw-name-input contains duplicate ids")
    source_by_id = None
    if args.source_input:
        source_records = _json_lines(Path(args.source_input))
        if any(not isinstance(row, dict) for row in source_records):
            raise ValueError(f"{args.source_input}: every row must be an object")
        source_by_id = {str(row.get(args.id_field)): row for row in source_records}
        if len(source_by_id) != len(source_records) or "None" in source_by_id:
            raise ValueError(f"{args.source_input}: source ids must be present and unique")
    output_rows = []
    seen_ids = set()
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{args.input}:{index + 1}: row is not an object")
        row_id = str(row.get(args.id_field, index))
        source_row = source_by_id.get(row_id) if source_by_id is not None else row
        if source_row is None:
            raise ValueError(f"{row_id}: no matching source record")
        if row_id in seen_ids:
            raise ValueError(f"{args.input}: duplicate prediction id {row_id!r}")
        seen_ids.add(row_id)
        language = args.language or source_row.get(args.language_field) or row.get(args.language_field)
        if not isinstance(language, str) or not language:
            raise ValueError(
                f"{row_id}: explicit BCP 47 tag is required via --language or field {args.language_field!r}"
            )
        text = source_row.get(args.text_field, row.get(args.text_field))
        if not isinstance(text, str):
            raise ValueError(f"{row_id}: text is required")
        annotations = _record_annotations(row, args.annotations_field, row_id)
        if source_by_id is not None and not getattr(args, "source_text_only", False):
            source_annotations = _record_annotations(
                source_row,
                args.source_annotations_field,
                row_id,
            )
            if not isinstance(source_annotations, list) or not isinstance(annotations, list):
                raise ValueError(f"{row_id}: source and predicted annotations must be lists")
            annotations = [*source_annotations, *annotations]
        subclass_annotations = row.get(args.subclass_field) if args.subclass_field else None
        recovery = None
        if raw_by_id is not None:
            if row_id not in raw_by_id:
                raise ValueError(f"{row_id}: missing --raw-name-input row")
            parts, recovery = _raw_name_components(raw_by_id[row_id], text, engine.config)
            subclass_annotations = [
                p for p in (subclass_annotations or []) if p.get("family") != "name_component"
            ] + parts
            if recovery["rejected"]:
                print(f"[name-recovery] {row_id}: {recovery['rejected']}", file=sys.stderr)
        analysis = engine.analyze(
            row_id=row_id,
            text=text,
            language=language,
            grammar_language=getattr(args, "grammar_language", None),
            annotations=annotations,
            subclass_annotations=subclass_annotations,
        )
        if getattr(args, "output_view", "audit") == "predictions":
            output_rows.append(
                {
                    "id": row_id,
                    "text": text,
                    "bcp47": language,
                    **engine.inference_fields(analysis),
                }
            )
        else:
            output_rows.append(analysis)
        if recovery is not None:
            output_rows[-1]["raw_name_recovery"] = recovery
    if raw_by_id is not None and seen_ids != set(raw_by_id):
        raise ValueError("--raw-name-input and prediction membership differ")
    if source_by_id is not None and seen_ids != set(source_by_id):
        missing = sorted(set(source_by_id) - seen_ids)
        raise ValueError(f"prediction input omits {len(missing)} source ids; first is {missing[0]!r}")
    _write_rows(args.output, output_rows)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--override-name-kinds",
        action="store_true",
        help="replace all supplied name-kind labels in insert-name-kinds mode",
    )
    parser.add_argument(
        "--mode", choices=("insert-name-kinds", "infer-person-name"), default="insert-name-kinds"
    )
    parser.add_argument(
        "--max-name-gap-chars",
        type=int,
        default=5,
        help="total intervening O characters allowed per inferred name",
    )
    parser.add_argument(
        "--no-name-grammar",
        action="store_true",
        help="disable component-sequence vetoes in infer-person-name mode",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("pii_name_annotation_qc_profiles_v2.json")),
    )
    parser.add_argument("--given-lexicon", action="append", default=[])
    parser.add_argument("--family-lexicon", action="append", default=[])
    parser.add_argument("--output", required=True, help="JSONL output path, or - for stdout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    native_parser = subparsers.add_parser(
        "native",
        help="audit parallel text lines and native serving JSON-array lines",
    )
    _common_arguments(native_parser)
    native_parser.add_argument("--texts", required=True)
    native_parser.add_argument("--predictions", required=True)
    native_parser.add_argument("--language", required=True)
    native_parser.add_argument("--grammar-language")
    native_parser.set_defaults(func=audit_native_arrays)
    records_parser = subparsers.add_parser(
        "records",
        help="audit JSONL annotation records containing text and spans",
    )
    _common_arguments(records_parser)
    records_parser.add_argument("--input", required=True)
    records_parser.add_argument(
        "--output-view",
        choices=("audit", "predictions"),
        default="audit",
        help="emit the audit or usable primary predictions with preserved refinement spans",
    )
    records_parser.add_argument(
        "--source-input",
        help="optional JSONL source packet supplying text, language, and base spans by id",
    )
    records_parser.add_argument("--language", help="fixed BCP 47 tag for every input row")
    records_parser.add_argument(
        "--grammar-language",
        help="optional fixed BCP 47 tag selecting the grammar independently",
    )
    records_parser.add_argument("--id-field", default="id")
    records_parser.add_argument(
        "--language-field",
        default="bcp47",
        help="row field containing the required BCP 47 tag (default: bcp47)",
    )
    records_parser.add_argument("--text-field", default="text")
    records_parser.add_argument("--annotations-field", default="auto")
    records_parser.add_argument("--source-annotations-field", default="base_spans")
    records_parser.add_argument(
        "--source-text-only",
        action="store_true",
        help="read source text and language by id without importing any source annotations",
    )
    records_parser.add_argument("--subclass-field", default="subclass_spans")
    records_parser.add_argument(
        "--raw-name-input",
        help="OpenAI-compatible raw JSONL: recover valid name components before whole-response quarantine; infer-person-name only",
    )
    records_parser.set_defaults(func=audit_records)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
