#!/usr/bin/env python3
"""Validate core-language sampling for joint and explicitly routed PII models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LANGUAGE_ROUND_PATH = Path(__file__).with_name("pii_language_round.yaml")
DEFAULT_MINIMUM_CORE_SHARE = 0.01


REPO_ROOT = Path(__file__).resolve().parents[1]


def recorded_path(path: Path) -> str:
    """How a path should appear inside a saved config or receipt.

    Repository-relative when the file lives in the repository, because an absolute path
    records where one machine happened to keep its checkout and is wrong everywhere else:
    the workers hold the same tree under a different home, so an absolute record cannot be
    resolved there. Readers resolve a relative value against the repository root. A file
    outside the repository has no such frame and stays absolute.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core_language_policy(path: str | Path = DEFAULT_LANGUAGE_ROUND_PATH) -> dict[str, Any]:
    """Load the core inventory, minimum share, and relative importance weights."""
    policy_path = Path(path)
    value = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("languages"), list):
        raise ValueError(f"language round must contain a languages list: {policy_path}")
    languages = []
    for index, item in enumerate(value["languages"]):
        code = item.get("code") if isinstance(item, dict) else None
        if not isinstance(code, str) or not code:
            raise ValueError(f"language round entry {index} lacks a nonempty code: {policy_path}")
        languages.append(code)
    if len(languages) != len(set(languages)):
        raise ValueError(f"language round contains duplicate language codes: {policy_path}")
    minimum_share = float(
        value.get("training_mix", {}).get(
            "minimum_core_language_share",
            DEFAULT_MINIMUM_CORE_SHARE,
        )
    )
    if not math.isfinite(minimum_share) or not 0 < minimum_share < 1:
        raise ValueError(f"minimum core-language share must be in (0, 1): {minimum_share}")
    if len(languages) * minimum_share > 1 + 1e-12:
        raise ValueError(
            f"{len(languages)} core languages at share {minimum_share} exceed total sampler mass"
        )
    importance = value.get("language_importance", {})
    if not isinstance(importance, dict):
        raise ValueError(f"language_importance must be a mapping: {policy_path}")
    configured_weights = importance.get("weights", {})
    if not isinstance(configured_weights, dict):
        raise ValueError(f"language_importance.weights must be a mapping: {policy_path}")
    unlisted_weight = float(importance.get("unlisted_language_weight", 1.0))
    if not math.isfinite(unlisted_weight) or unlisted_weight <= 0:
        raise ValueError(f"unlisted language importance must be positive: {unlisted_weight}")
    importance_weights = {
        language: float(configured_weights.get(language, unlisted_weight)) for language in languages
    }
    invalid_weights = {
        language: weight
        for language, weight in importance_weights.items()
        if not math.isfinite(weight) or weight <= 0
    }
    if invalid_weights:
        raise ValueError(f"language importance weights must be positive and finite: {invalid_weights}")
    return {
        "path": recorded_path(policy_path),
        "sha256": file_sha256(policy_path),
        "round_id": value.get("round_id"),
        "core_languages": languages,
        "minimum_expected_share": minimum_share,
        "importance_weights": importance_weights,
    }


def allocate_importance_language_counts(
    total_rows: int,
    *,
    language_round: str | Path = DEFAULT_LANGUAGE_ROUND_PATH,
) -> dict[str, int]:
    """Allocate an exact row budget by declared language importance."""
    if total_rows <= 0:
        raise ValueError("total rows must be positive")
    policy = load_core_language_policy(language_round)
    weights = policy["importance_weights"]
    total_weight = sum(weights.values())
    quotas = {language: total_rows * weight / total_weight for language, weight in weights.items()}
    counts = {language: math.floor(quota) for language, quota in quotas.items()}
    remainder = total_rows - sum(counts.values())
    order = {language: index for index, language in enumerate(policy["core_languages"])}
    ranked = sorted(
        weights,
        key=lambda language: (-(quotas[language] - counts[language]), order[language]),
    )
    for language in ranked[:remainder]:
        counts[language] += 1
    validate_core_language_support(counts, language_round=language_round)
    return counts


def language_shares(
    languages: Iterable[str],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Normalize row languages and optional sampler weights into expected shares."""
    language_list = list(languages)
    weight_list = [1.0] * len(language_list) if weights is None else list(weights)
    if len(language_list) != len(weight_list):
        raise ValueError("language and sampler-weight lengths differ")
    totals: dict[str, float] = {}
    for language, raw_weight in zip(language_list, weight_list, strict=True):
        if not isinstance(language, str) or not language:
            raise ValueError(f"training row lacks a nonempty language: {language!r}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid sampler weight for {language!r}: {raw_weight!r}")
        totals[language] = totals.get(language, 0.0) + weight
    total = sum(totals.values())
    if total <= 0:
        raise ValueError("language sampler has no positive mass")
    return {language: mass / total for language, mass in sorted(totals.items()) if mass > 0}


def _route_scope(
    path: str | Path,
    component: str,
    core_languages: set[str],
) -> dict[str, Any]:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes = value.get("routes") if isinstance(value, dict) else None
    if not isinstance(routes, dict) or not routes:
        raise ValueError(f"language-support split manifest must contain routes: {manifest_path}")
    route_languages: dict[str, list[str]] = {}
    route_masses: dict[str, float] = {}
    for route, details in routes.items():
        languages = details.get("languages") if isinstance(details, dict) else None
        if not isinstance(route, str) or not route or not isinstance(languages, list):
            raise ValueError(f"invalid language route {route!r} in {manifest_path}")
        if any(not isinstance(language, str) or not language for language in languages):
            raise ValueError(f"route {route!r} has an invalid language code in {manifest_path}")
        if len(languages) != len(set(languages)):
            raise ValueError(f"route {route!r} repeats a language in {manifest_path}")
        route_languages[route] = languages
        mass = float(details.get("joint_sampling_mass", float("nan")))
        if not math.isfinite(mass) or not 0 < mass <= 1:
            raise ValueError(f"route {route!r} lacks a valid joint_sampling_mass in {manifest_path}")
        route_masses[route] = mass
    if component not in route_languages:
        raise ValueError(f"language-support split component {component!r} is absent from {manifest_path}")
    if abs(sum(route_masses.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"language route joint_sampling_mass values sum to {sum(route_masses.values())}, expected 1"
        )
    assignments: dict[str, list[str]] = {}
    for route, languages in route_languages.items():
        for language in languages:
            assignments.setdefault(language, []).append(route)
    aggregate = set(assignments)
    missing = sorted(core_languages - aggregate)
    if missing:
        raise ValueError(
            f"language-support split manifest omits core languages: {missing}; manifest={manifest_path}"
        )
    multiply_routed = {
        language: routes
        for language, routes in assignments.items()
        if language in core_languages and len(routes) != 1
    }
    if multiply_routed:
        raise ValueError(
            "each core language must have exactly one route for aggregate exposure accounting: "
            f"{multiply_routed}"
        )
    return {
        "path": recorded_path(manifest_path),
        "sha256": file_sha256(manifest_path),
        "version": value.get("version"),
        "component": component,
        "component_languages": route_languages[component],
        "component_joint_sampling_mass": route_masses[component],
        "required_components": sorted(route_languages),
        "aggregate_core_languages": sorted(core_languages & aggregate),
    }


def validate_core_language_support(
    expected_shares: Mapping[str, float],
    *,
    language_round: str | Path = DEFAULT_LANGUAGE_ROUND_PATH,
    split_manifest: str | Path | None = None,
    split_component: str | None = None,
) -> dict[str, Any]:
    """Fail unless one model, or its declared routed component, meets the core floor."""
    if (split_manifest is None) != (split_component is None):
        raise ValueError("language-support split requires both a route manifest and a component name")
    policy = load_core_language_policy(language_round)
    shares = language_shares(expected_shares.keys(), list(expected_shares.values()))
    core = set(policy["core_languages"])
    route = None
    if split_manifest is None:
        mode = "joint"
        required = core
        aggregate_equivalent_shares = shares
    else:
        mode = "routed_component"
        route = _route_scope(split_manifest, str(split_component), core)
        declared = set(route["component_languages"])
        required = core & declared
        undeclared_observed = sorted(
            language for language in core - declared if shares.get(language, 0.0) > 0
        )
        if undeclared_observed:
            raise ValueError(
                f"routed component {split_component!r} samples undeclared core languages: "
                f"{undeclared_observed}"
            )
        aggregate_equivalent_shares = {
            language: share * route["component_joint_sampling_mass"] for language, share in shares.items()
        }
    minimum = policy["minimum_expected_share"]
    missing = sorted(language for language in required if shares.get(language, 0.0) == 0)
    below = {
        language: aggregate_equivalent_shares[language]
        for language in sorted(required)
        if language in aggregate_equivalent_shares and aggregate_equivalent_shares[language] + 1e-12 < minimum
    }
    if missing or below:
        raise ValueError(
            "core-language expected sampler floor failed: "
            f"missing={missing} below={below} minimum={minimum:.6f} mode={mode}"
        )
    receipt = {
        "schema_version": 1,
        "status": "verified_expected_sampler",
        "mode": mode,
        "language_round": policy,
        "required_core_languages_for_model": sorted(required),
        "component_expected_language_shares": shares,
        "aggregate_equivalent_expected_language_shares": aggregate_equivalent_shares,
        "extra_languages": sorted(set(shares) - core),
        "missing_core_languages": missing,
        "below_minimum_core_languages": below,
    }
    if route is not None:
        receipt["route"] = route
        receipt["aggregate_completion_rule"] = (
            "a coverage claim requires a successful training receipt for every required component"
        )
    return receipt
