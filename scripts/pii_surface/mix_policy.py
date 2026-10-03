"""Validated per-language, per-family surface-realization policy."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import yaml

try:
    from pii_projector import Tagset
except ModuleNotFoundError:  # Imported as scripts.pii_surface.mix_policy in tests.
    from scripts.pii_projector import Tagset

SURFACE_EXPERT_ROOTS = ("person_name", "organization", "occupation", "location")
_TAGSET = Tagset()


def surface_ontology_group(tag: str) -> str | None:
    """Return the nearest surface-family root in the canonical ontology."""
    node = tag.casefold()
    if node not in _TAGSET.nodes:
        return None
    return next(
        (ancestor for ancestor in _TAGSET.ancestors(node) if ancestor in SURFACE_EXPERT_ROOTS),
        None,
    )


def surface_expert_group(tag: str) -> str | None:
    """Return an ontology family only for non-format learned surfaces."""
    node = tag.casefold()
    if node not in _TAGSET.nodes or _TAGSET.coarse(node) == "format":
        return None
    return surface_ontology_group(tag)


LEARNED_OPEN_CLASS_TAGS = frozenset(
    node.upper() for node in _TAGSET.nodes if surface_expert_group(node) is not None
)
NAME_TAGS = frozenset(tag for tag in LEARNED_OPEN_CLASS_TAGS if surface_expert_group(tag) == "person_name")
ORGANIZATION_TAGS = frozenset(
    tag for tag in LEARNED_OPEN_CLASS_TAGS if surface_expert_group(tag) == "organization"
)
OCCUPATION_TAGS = frozenset(
    tag for tag in LEARNED_OPEN_CLASS_TAGS if surface_expert_group(tag) == "occupation"
)
LOCATION_TAGS = frozenset(
    node.upper() for node in _TAGSET.nodes if surface_ontology_group(node) == "location"
)


def surface_family(tag: str) -> str:
    if tag in NAME_TAGS:
        return "name"
    if tag in ORGANIZATION_TAGS:
        return "organization"
    if tag in OCCUPATION_TAGS:
        return "occupation"
    if tag in LOCATION_TAGS:
        return "location"
    return "other"


@dataclass(frozen=True)
class CellPolicy:
    natural_surface_rate: float = 0.5
    natural_min_distinct_full_rate: int = 100
    natural_count_temperature: float = 0.5
    native_name_rate: float = 0.8
    non_empirical_fallback: str = "generator"
    fresh_faker_beta: float | None = None
    replacement_char_alpha: float | None = None
    # Temporal tags do not go through the natural-surface path at all: a date's
    # value has to survive for coherence, so the filler re-renders its surface
    # through the locale profile instead. That happens whenever a locale profile
    # is loaded, independent of natural_surface_rate, which means a recipe that
    # opts only ORGANIZATION into realization still gets every date rewritten.
    # This is the switch for it. It defaults true because every recipe written
    # before it relied on that behaviour.
    localize_temporal_surfaces: bool = True

    def validate(self, where: str) -> None:
        if not isinstance(self.natural_surface_rate, (int, float)) or isinstance(
            self.natural_surface_rate, bool
        ):
            raise ValueError(f"{where}.natural_surface_rate must be numeric")
        if not 0 <= self.natural_surface_rate <= 1:
            raise ValueError(f"{where}.natural_surface_rate must be in [0, 1]")
        if not isinstance(self.natural_min_distinct_full_rate, int) or isinstance(
            self.natural_min_distinct_full_rate, bool
        ):
            raise ValueError(f"{where}.natural_min_distinct_full_rate must be an integer")
        if self.natural_min_distinct_full_rate <= 0:
            raise ValueError(f"{where}.natural_min_distinct_full_rate must be positive")
        if not isinstance(self.natural_count_temperature, (int, float)) or isinstance(
            self.natural_count_temperature, bool
        ):
            raise ValueError(f"{where}.natural_count_temperature must be numeric")
        if not 0 <= self.natural_count_temperature <= 1:
            raise ValueError(f"{where}.natural_count_temperature must be in [0, 1]")
        if not isinstance(self.native_name_rate, (int, float)) or isinstance(self.native_name_rate, bool):
            raise ValueError(f"{where}.native_name_rate must be numeric")
        if not 0 <= self.native_name_rate <= 1:
            raise ValueError(f"{where}.native_name_rate must be in [0, 1]")
        if not isinstance(self.localize_temporal_surfaces, bool):
            raise ValueError(f"{where}.localize_temporal_surfaces must be true or false")
        if self.non_empirical_fallback not in {"generator", "source"}:
            raise ValueError(f"{where}.non_empirical_fallback must be 'generator' or 'source'")
        for field in ("fresh_faker_beta", "replacement_char_alpha"):
            value = getattr(self, field)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1
            ):
                raise ValueError(f"{where}.{field} must be null or numeric in [0, 1]")
        if (self.fresh_faker_beta is None) != (self.replacement_char_alpha is None):
            raise ValueError(
                f"{where}.fresh_faker_beta and replacement_char_alpha must be configured together"
            )


POLICY_FIELDS = frozenset(CellPolicy.__dataclass_fields__)


def overlay(base: CellPolicy, raw: Any, where: str) -> CellPolicy:
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    unknown = set(raw) - POLICY_FIELDS
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")
    updated = replace(base, **raw)
    updated.validate(where)
    return updated


class SurfaceMixPolicy:
    def __init__(self, path: Path, raw: dict[str, Any]):
        if raw.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported schema_version")
        version = raw.get("recipe_version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"{path}: recipe_version must be a nonempty string")
        allowed = {"schema_version", "recipe_version", "status", "defaults", "families", "tags", "languages"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown top-level keys: {', '.join(sorted(unknown))}")
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.version = version
        self.status = raw.get("status")
        self.raw = raw
        self.default = overlay(CellPolicy(), raw.get("defaults"), "defaults")
        self.families = self._overlays(self.default, raw.get("families"), "families")
        self.tags = self._overlays(self.default, raw.get("tags"), "tags")
        languages = raw.get("languages") or {}
        if not isinstance(languages, dict):
            raise ValueError(f"{path}: languages must be an object")
        self.languages = languages
        for language, language_raw in languages.items():
            if not isinstance(language_raw, dict):
                raise ValueError(f"languages.{language} must be an object")
            unknown_language = set(language_raw) - {"defaults", "families", "tags"}
            if unknown_language:
                raise ValueError(
                    f"languages.{language} has unknown keys: {', '.join(sorted(unknown_language))}"
                )
            language_default = overlay(
                self.default, language_raw.get("defaults"), f"languages.{language}.defaults"
            )
            self._overlays(language_default, language_raw.get("families"), f"languages.{language}.families")
            self._overlays(language_default, language_raw.get("tags"), f"languages.{language}.tags")

    @staticmethod
    def _overlays(base: CellPolicy, raw: Any, where: str) -> dict[str, CellPolicy]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be an object")
        return {str(key): overlay(base, value, f"{where}.{key}") for key, value in raw.items()}

    @classmethod
    def load(cls, path: Path) -> SurfaceMixPolicy:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: policy must be a YAML object")
        return cls(path, raw)

    @cache
    def for_tag(self, language: str, tag: str) -> CellPolicy:
        policy = self.default
        if family_raw := (self.raw.get("families") or {}).get(surface_family(tag)):
            policy = overlay(policy, family_raw, f"families.{surface_family(tag)}")
        if tag_raw := (self.raw.get("tags") or {}).get(tag):
            policy = overlay(policy, tag_raw, f"tags.{tag}")
        language_raw = self.languages.get(language) or {}
        policy = overlay(policy, language_raw.get("defaults"), f"languages.{language}.defaults")
        if family_raw := (language_raw.get("families") or {}).get(surface_family(tag)):
            policy = overlay(policy, family_raw, f"languages.{language}.families.{surface_family(tag)}")
        if tag_raw := (language_raw.get("tags") or {}).get(tag):
            policy = overlay(policy, tag_raw, f"languages.{language}.tags.{tag}")
        return policy

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "recipe_version": self.version,
            "status": self.status,
            "resolved_default": asdict(self.default),
        }
