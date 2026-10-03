"""Build and sample predicate-aware entity-surface pools for ont3.

The pool is deliberately separate from the legacy language/tag pool.  A
predicate draw must preserve every known intrinsic carrier target, keep
contextual predicates out of surface selection, and records the support and
backoff route used for the draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

POLICY_SCHEMA = "pii-predicate-surface-realization-policy"
POLICY_VERSION = 1
POOL_ENTRY_SCHEMA = "pii-predicate-surface-pool-entry"
POOL_ENTRY_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _finite_probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{where} must be finite and in [0, 1]")
    return result


def _string_set(value: Any, where: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a nonempty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{where} must contain nonempty strings")
    result = frozenset(value)
    if len(result) != len(value):
        raise ValueError(f"{where} contains duplicates")
    return result


@dataclass(frozen=True)
class OriginSamplingRoute:
    surface_origin: str
    probability: float
    min_distinct_support: int


@dataclass(frozen=True)
class PredicateSurfacePolicy:
    version: str
    full_pool_probability: float
    min_distinct_support: int
    count_temperature: float
    real_original_probability: float
    real_original_origins: frozenset[str]
    admitted_non_original_origins: frozenset[str]
    origin_sampling_routes: tuple[OriginSamplingRoute, ...]
    intrinsic_predicates: frozenset[str]
    contextual_predicates: frozenset[str]
    predicate_weights: Mapping[str, float]
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "PredicateSurfacePolicy":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema",
            "version",
            "policy_version",
            "status",
            "full_pool_probability",
            "min_distinct_support",
            "count_temperature",
            "real_original_probability",
            "real_original_origins",
            "admitted_non_original_origins",
            "intrinsic_predicates",
            "contextual_predicates",
            "predicate_weights",
        }
        optional = {"origin_sampling_routes"}
        if (
            not isinstance(payload, dict)
            or not required <= set(payload)
            or set(payload) - required - optional
        ):
            raise ValueError(f"{path}: predicate-surface policy fields do not match schema")
        if payload["schema"] != POLICY_SCHEMA or payload["version"] != POLICY_VERSION:
            raise ValueError(f"{path}: unsupported predicate-surface policy")
        policy_version = payload["policy_version"]
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError(f"{path}: policy_version must be nonempty")
        if not isinstance(payload["status"], str) or not payload["status"]:
            raise ValueError(f"{path}: status must be nonempty")
        minimum = payload["min_distinct_support"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
            raise ValueError(f"{path}: min_distinct_support must be a positive integer")
        intrinsic = _string_set(payload["intrinsic_predicates"], "intrinsic_predicates")
        contextual = _string_set(payload["contextual_predicates"], "contextual_predicates")
        if overlap := intrinsic & contextual:
            raise ValueError(f"{path}: predicates cannot be intrinsic and contextual: {sorted(overlap)}")
        raw_weights = payload["predicate_weights"]
        if not isinstance(raw_weights, dict) or set(raw_weights) != set(intrinsic):
            raise ValueError(f"{path}: predicate_weights must name every intrinsic predicate exactly")
        weights = {}
        for predicate, value in raw_weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}: predicate weight for {predicate} must be numeric")
            weight = float(value)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{path}: predicate weight for {predicate} must be positive")
            weights[predicate] = weight
        real_origins = _string_set(payload["real_original_origins"], "real_original_origins")
        non_original_origins = _string_set(
            payload["admitted_non_original_origins"], "admitted_non_original_origins"
        )
        if overlap := real_origins & non_original_origins:
            raise ValueError(f"{path}: surface-origin gates overlap: {sorted(overlap)}")
        raw_origin_routes = payload.get("origin_sampling_routes", [])
        if not isinstance(raw_origin_routes, list):
            raise ValueError(f"{path}: origin_sampling_routes must be a list")
        origin_routes = []
        seen_route_origins = set()
        for index, raw_route in enumerate(raw_origin_routes):
            where = f"{path}: origin_sampling_routes[{index}]"
            if not isinstance(raw_route, dict) or set(raw_route) != {
                "surface_origin",
                "probability",
                "min_distinct_support",
            }:
                raise ValueError(f"{where}: fields do not match schema")
            origin = raw_route["surface_origin"]
            if not isinstance(origin, str) or not origin:
                raise ValueError(f"{where}: surface_origin must be nonempty")
            if origin not in non_original_origins:
                raise ValueError(f"{where}: surface_origin must be an admitted non-original origin")
            if origin in seen_route_origins:
                raise ValueError(f"{where}: duplicate surface_origin {origin!r}")
            seen_route_origins.add(origin)
            route_minimum = raw_route["min_distinct_support"]
            if isinstance(route_minimum, bool) or not isinstance(route_minimum, int) or route_minimum <= 0:
                raise ValueError(f"{where}: min_distinct_support must be a positive integer")
            origin_routes.append(
                OriginSamplingRoute(
                    surface_origin=origin,
                    probability=_finite_probability(raw_route["probability"], f"{where}.probability"),
                    min_distinct_support=route_minimum,
                )
            )
        if sum(route.probability for route in origin_routes) > 1.0:
            raise ValueError(f"{path}: origin_sampling_routes probabilities sum above one")
        return cls(
            version=policy_version,
            full_pool_probability=_finite_probability(
                payload["full_pool_probability"], "full_pool_probability"
            ),
            min_distinct_support=minimum,
            count_temperature=_finite_probability(payload["count_temperature"], "count_temperature"),
            real_original_probability=_finite_probability(
                payload["real_original_probability"], "real_original_probability"
            ),
            real_original_origins=real_origins,
            admitted_non_original_origins=non_original_origins,
            origin_sampling_routes=tuple(origin_routes),
            intrinsic_predicates=intrinsic,
            contextual_predicates=contextual,
            predicate_weights=weights,
            path=path.resolve(),
            sha256=file_sha256(path),
        )

    @property
    def all_predicates(self) -> frozenset[str]:
        return self.intrinsic_predicates | self.contextual_predicates

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "policy_version": self.version,
            "full_pool_probability": self.full_pool_probability,
            "min_distinct_support": self.min_distinct_support,
            "count_temperature": self.count_temperature,
            "real_original_probability": self.real_original_probability,
            "origin_sampling_routes": [
                {
                    "surface_origin": route.surface_origin,
                    "probability": route.probability,
                    "min_distinct_support": route.min_distinct_support,
                }
                for route in self.origin_sampling_routes
            ],
            "intrinsic_predicates": sorted(self.intrinsic_predicates),
            "contextual_predicates": sorted(self.contextual_predicates),
            "predicate_weights": dict(sorted(self.predicate_weights.items())),
        }

    def validate_channels(self, channels: Iterable[str]) -> None:
        active = frozenset(channels)
        if active != self.all_predicates:
            missing = sorted(active - self.all_predicates)
            extra = sorted(self.all_predicates - active)
            raise ValueError(
                "predicate-surface policy does not partition the active predicate head: "
                f"missing={missing}, extra={extra}"
            )


def _validate_relative_attrs(
    attrs: Any,
    *,
    surface_length: int,
    intrinsic: frozenset[str],
    where: str,
) -> dict[str, tuple[tuple[int, int], ...]]:
    if not isinstance(attrs, dict):
        raise ValueError(f"{where}: intrinsic_attrs must be an object")
    if unknown := set(attrs) - set(intrinsic):
        raise ValueError(f"{where}: non-intrinsic predicate attrs: {sorted(unknown)}")
    result = {}
    for predicate, intervals in attrs.items():
        if not isinstance(intervals, list):
            raise ValueError(f"{where}: {predicate} intervals must be a list")
        normalized = []
        for index, interval in enumerate(intervals):
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or isinstance(interval[0], bool)
                or not isinstance(interval[0], int)
                or isinstance(interval[1], bool)
                or not isinstance(interval[1], int)
            ):
                raise ValueError(f"{where}: {predicate}[{index}] must be an integer interval")
            start, end = interval
            if not 0 <= start < end <= surface_length:
                raise ValueError(f"{where}: {predicate}[{index}] escapes the surface")
            if normalized and normalized[-1][1] > start:
                raise ValueError(f"{where}: {predicate} intervals overlap or are unsorted")
            normalized.append((start, end))
        result[predicate] = tuple(normalized)
    return result


@dataclass(frozen=True)
class PredicatePoolEntry:
    value: str
    normalized: str
    count: int
    entry_id: str
    provenance: str
    language: str
    primary_type: str
    surface_origin: str
    intrinsic_attrs: Mapping[str, tuple[tuple[int, int], ...]]
    objective_weights: Mapping[str, float]
    source_counts: Mapping[str, int]
    source_record_hashes: tuple[str, ...]
    predicate_seed_methods: tuple[str, ...]

    @property
    def positive_signature(self) -> frozenset[str]:
        return frozenset(predicate for predicate, intervals in self.intrinsic_attrs.items() if intervals)

    @property
    def known_intrinsic(self) -> frozenset[str]:
        return frozenset(self.intrinsic_attrs)


@dataclass(frozen=True)
class PredicateSurfaceDraw:
    entry: PredicatePoolEntry
    receipt: Mapping[str, Any]


def _distinct_support(entries: Iterable[PredicatePoolEntry]) -> int:
    return len({entry.normalized for entry in entries})


def _carrier_state(
    attrs: Mapping[str, Iterable[Iterable[int]]], policy: PredicateSurfacePolicy
) -> dict[str, bool]:
    unknown = set(attrs) - set(policy.all_predicates)
    if unknown:
        raise ValueError(f"carrier names predicates absent from the realization policy: {sorted(unknown)}")
    return {
        predicate: bool(tuple(intervals))
        for predicate, intervals in attrs.items()
        if predicate in policy.intrinsic_predicates
    }


def _compatible(entry: PredicatePoolEntry, carrier_state: Mapping[str, bool]) -> bool:
    return all(
        predicate in entry.known_intrinsic and (predicate in entry.positive_signature) is expected_positive
        for predicate, expected_positive in carrier_state.items()
    )


class PredicateSurfacePool:
    def __init__(
        self,
        entries: Iterable[PredicatePoolEntry],
        *,
        version: str,
        policy_sha256: str,
        path: Path,
        sha256: str,
    ):
        self.version = version
        self.policy_sha256 = policy_sha256
        self.path = path.resolve()
        self.sha256 = sha256
        groups: dict[tuple[str, str, str], list[PredicatePoolEntry]] = defaultdict(list)
        origin_groups: dict[tuple[str, str, str], list[PredicatePoolEntry]] = defaultdict(list)
        for entry in entries:
            origin_bucket = "real_original" if entry.surface_origin == "real_original" else "non_original"
            groups[(entry.language, entry.primary_type, origin_bucket)].append(entry)
            origin_groups[(entry.language, entry.primary_type, entry.surface_origin)].append(entry)
        self.groups = {key: tuple(values) for key, values in groups.items()}
        self.origin_groups = {key: tuple(values) for key, values in origin_groups.items()}

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        policy: PredicateSurfacePolicy,
        split: str = "train",
    ) -> "PredicateSurfacePool":
        path = Path(path)
        entries = []
        versions = set()
        policy_hashes = set()
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                where = f"{path}:{line_number}"
                if row.get("split") != split:
                    continue
                if row.get("schema") != POOL_ENTRY_SCHEMA or row.get("schema_version") != POOL_ENTRY_VERSION:
                    raise ValueError(f"{where}: unsupported predicate-surface pool entry")
                string_fields = (
                    "pool_version",
                    "policy_sha256",
                    "entry_id",
                    "lang",
                    "primary_type",
                    "value",
                    "normalized",
                    "surface_origin",
                )
                if any(not isinstance(row.get(field), str) or not row[field] for field in string_fields):
                    raise ValueError(f"{where}: malformed predicate-surface string field")
                count = row.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{where}: count must be a positive integer")
                origin = row["surface_origin"]
                admitted_origins = policy.real_original_origins | policy.admitted_non_original_origins
                if origin not in admitted_origins:
                    raise ValueError(f"{where}: surface origin {origin!r} is not admitted by policy")
                attrs = _validate_relative_attrs(
                    row.get("intrinsic_attrs"),
                    surface_length=len(row["value"]),
                    intrinsic=policy.intrinsic_predicates,
                    where=where,
                )
                positive_signature = sorted(predicate for predicate, intervals in attrs.items() if intervals)
                known_negative = sorted(predicate for predicate, intervals in attrs.items() if not intervals)
                if row.get("positive_signature") != positive_signature:
                    raise ValueError(f"{where}: positive_signature does not match intrinsic_attrs")
                if row.get("known_negative_intrinsic") != known_negative:
                    raise ValueError(f"{where}: known_negative_intrinsic does not match intrinsic_attrs")
                objective_weights = row.get("objective_weights")
                if not isinstance(objective_weights, dict) or not {"O", "other"} <= set(objective_weights):
                    raise ValueError(f"{where}: objective_weights must contain O and other")
                if unknown := set(objective_weights) - {"O", "other", *attrs}:
                    raise ValueError(f"{where}: objective_weights names unknown attrs: {sorted(unknown)}")
                for name, value in objective_weights.items():
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError(f"{where}: objective weight {name} must be nonnegative")
                source_counts = row.get("source_counts")
                source_hashes = row.get("source_record_hashes")
                methods = row.get("predicate_seed_methods")
                if not isinstance(source_counts, dict) or any(
                    not isinstance(name, str)
                    or not name
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for name, value in source_counts.items()
                ):
                    raise ValueError(f"{where}: source_counts is malformed")
                if not isinstance(source_hashes, list) or any(
                    not isinstance(value, str) or not value for value in source_hashes
                ):
                    raise ValueError(f"{where}: source_record_hashes is malformed")
                if not isinstance(methods, list) or any(
                    not isinstance(value, str) or not value for value in methods
                ):
                    raise ValueError(f"{where}: predicate_seed_methods is malformed")
                versions.add(row["pool_version"])
                policy_hashes.add(row["policy_sha256"])
                entries.append(
                    PredicatePoolEntry(
                        value=row["value"],
                        normalized=row["normalized"],
                        count=count,
                        entry_id=row["entry_id"],
                        provenance=f"predicate-pool:{row['pool_version']}:{row['entry_id']}",
                        language=row["lang"],
                        primary_type=row["primary_type"],
                        surface_origin=origin,
                        intrinsic_attrs=attrs,
                        objective_weights={name: float(value) for name, value in objective_weights.items()},
                        source_counts=dict(source_counts),
                        source_record_hashes=tuple(source_hashes),
                        predicate_seed_methods=tuple(methods),
                    )
                )
        if len(versions) != 1:
            raise ValueError(f"{path}: expected one pool version, found {sorted(versions)}")
        if policy_hashes != {policy.sha256}:
            raise ValueError(
                f"{path}: pool policy hash {sorted(policy_hashes)} does not match {policy.sha256}"
            )
        return cls(
            entries,
            version=versions.pop(),
            policy_sha256=policy.sha256,
            path=path,
            sha256=file_sha256(path),
        )

    def _origin_entries(
        self,
        language: str,
        primary_type: str,
        rng: random.Random,
        policy: PredicateSurfacePolicy,
    ) -> tuple[tuple[PredicatePoolEntry, ...], str, bool]:
        real = self.groups.get((language, primary_type, "real_original"), ())
        other = self.groups.get((language, primary_type, "non_original"), ())
        requested_real = rng.random() < policy.real_original_probability
        if requested_real and real:
            return real, "real_original", False
        if other:
            return other, "non_original", requested_real
        return (), "none", requested_real

    def draw(
        self,
        language: str,
        primary_type: str,
        carrier_attrs: Mapping[str, Iterable[Iterable[int]]],
        rng: random.Random,
        *,
        policy: PredicateSurfacePolicy,
    ) -> tuple[PredicateSurfaceDraw | None, dict[str, Any]]:
        if policy.sha256 != self.policy_sha256:
            raise ValueError("predicate-surface pool and policy are not bound to the same hash")
        carrier_state = _carrier_state(carrier_attrs, policy)
        active_signature = frozenset(predicate for predicate, positive in carrier_state.items() if positive)
        requested_origin_route = None
        requested_origin_support = 0
        requested_origin_compatible_support = 0
        if policy.origin_sampling_routes:
            route_gate = rng.random()
            cumulative_probability = 0.0
            for route in policy.origin_sampling_routes:
                cumulative_probability += route.probability
                if route_gate < cumulative_probability:
                    requested_origin_route = route
                    break
        if requested_origin_route is not None:
            origin_entries = self.origin_groups.get(
                (language, primary_type, requested_origin_route.surface_origin), ()
            )
            origin_compatible = tuple(entry for entry in origin_entries if _compatible(entry, carrier_state))
            requested_origin_support = _distinct_support(origin_entries)
            requested_origin_compatible_support = _distinct_support(origin_compatible)
            if requested_origin_compatible_support >= requested_origin_route.min_distinct_support:
                selected = rng.choices(
                    origin_compatible,
                    weights=[entry.count**policy.count_temperature for entry in origin_compatible],
                    k=1,
                )[0]
                receipt = {
                    "policy_version": policy.version,
                    "pool_version": self.version,
                    "language": language,
                    "primary_type": primary_type,
                    "requested_active_signature": sorted(active_signature),
                    "origin_bucket": requested_origin_route.surface_origin,
                    "origin_gate_backoff": False,
                    "origin_distinct_support": requested_origin_support,
                    "compatible_distinct_support": requested_origin_compatible_support,
                    "min_distinct_support": requested_origin_route.min_distinct_support,
                    "declared_origin_route": requested_origin_route.surface_origin,
                    "declared_origin_probability": requested_origin_route.probability,
                    "selected_route": "declared_surface_origin",
                    "selected_predicate": None,
                    "selected_entry_id": selected.entry_id,
                    "selected_surface_origin": selected.surface_origin,
                    "selected_predicate_seed_methods": list(selected.predicate_seed_methods),
                    "selected_source_counts": dict(sorted(selected.source_counts.items())),
                    "route_distinct_support": requested_origin_compatible_support,
                    "backoff": False,
                }
                return PredicateSurfaceDraw(entry=selected, receipt=receipt), receipt
        origin_entries, origin_bucket, origin_backoff = self._origin_entries(
            language, primary_type, rng, policy
        )
        compatible = tuple(entry for entry in origin_entries if _compatible(entry, carrier_state))
        compatible_support = _distinct_support(compatible)
        base_receipt: dict[str, Any] = {
            "policy_version": policy.version,
            "pool_version": self.version,
            "language": language,
            "primary_type": primary_type,
            "requested_active_signature": sorted(active_signature),
            "origin_bucket": origin_bucket,
            "origin_gate_backoff": origin_backoff,
            "origin_distinct_support": _distinct_support(origin_entries),
            "compatible_distinct_support": compatible_support,
            "min_distinct_support": policy.min_distinct_support,
            "declared_origin_route": (
                requested_origin_route.surface_origin if requested_origin_route is not None else None
            ),
            "declared_origin_probability": (
                requested_origin_route.probability if requested_origin_route is not None else None
            ),
            "declared_origin_distinct_support": requested_origin_support,
            "declared_origin_compatible_distinct_support": requested_origin_compatible_support,
            "declared_origin_gate_backoff": requested_origin_route is not None,
        }
        if not compatible:
            receipt = {
                **base_receipt,
                "selected_route": "source_fallback_no_compatible_surface",
                "selected_predicate": None,
                "selected_entry_id": None,
                "route_distinct_support": 0,
                "backoff": True,
            }
            return None, receipt
        if compatible_support < policy.min_distinct_support:
            receipt = {
                **base_receipt,
                "selected_route": "source_fallback_insufficient_compatible_support",
                "selected_predicate": None,
                "selected_entry_id": None,
                "route_distinct_support": compatible_support,
                "backoff": True,
            }
            return None, receipt

        route = "full_primary_pool"
        selected_predicate = None
        candidates = compatible
        backoff = False
        exact = tuple(entry for entry in compatible if entry.positive_signature == active_signature)
        exact_support = _distinct_support(exact)
        individual_support = {
            predicate: _distinct_support(
                entry for entry in compatible if predicate in entry.positive_signature
            )
            for predicate in sorted(active_signature)
        }
        if rng.random() >= policy.full_pool_probability and active_signature:
            if exact_support >= policy.min_distinct_support:
                route = "exact_predicate_signature"
                candidates = exact
            else:
                supported = [
                    predicate
                    for predicate in sorted(active_signature)
                    if individual_support[predicate] >= policy.min_distinct_support
                ]
                if supported:
                    selected_predicate = rng.choices(
                        supported,
                        weights=[policy.predicate_weights[predicate] for predicate in supported],
                        k=1,
                    )[0]
                    route = "individual_predicate"
                    candidates = tuple(
                        entry for entry in compatible if selected_predicate in entry.positive_signature
                    )
                    backoff = True
                else:
                    backoff = True
        selected = rng.choices(
            candidates,
            weights=[entry.count**policy.count_temperature for entry in candidates],
            k=1,
        )[0]
        receipt = {
            **base_receipt,
            "selected_route": route,
            "selected_predicate": selected_predicate,
            "selected_entry_id": selected.entry_id,
            "selected_surface_origin": selected.surface_origin,
            "selected_predicate_seed_methods": list(selected.predicate_seed_methods),
            "selected_source_counts": dict(sorted(selected.source_counts.items())),
            "route_distinct_support": _distinct_support(candidates),
            "exact_signature_distinct_support": exact_support,
            "individual_predicate_distinct_support": individual_support,
            "backoff": backoff,
        }
        return PredicateSurfaceDraw(entry=selected, receipt=receipt), receipt

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "pool_version": self.version,
            "policy_sha256": self.policy_sha256,
            "groups": len(self.groups),
            "entries": sum(len(entries) for entries in self.groups.values()),
            "real_original_entries": sum(
                len(entries) for key, entries in self.groups.items() if key[2] == "real_original"
            ),
            "surface_origin_entries": dict(
                sorted(
                    Counter(
                        entry.surface_origin for entries in self.groups.values() for entry in entries
                    ).items()
                )
            ),
        }


@dataclass
class _Aggregate:
    count: int
    source_counts: Counter[str]
    source_record_hashes: set[str]
    predicate_seed_methods: set[str]


def _validate_source_predicate_span(
    predicate_span: Any,
    *,
    text: str,
    primary_spans: set[tuple[int, int, str]],
    policy: PredicateSurfacePolicy,
    where: str,
) -> tuple[tuple[int, int, str], dict[str, list[list[int]]], dict[str, float]]:
    if not isinstance(predicate_span, dict) or set(predicate_span) not in (
        {"start", "end", "type", "attrs"},
        {"start", "end", "type", "attrs", "objective_weights"},
    ):
        raise ValueError(f"{where}: invalid predicate span fields")
    start, end, primary_type = (
        predicate_span["start"],
        predicate_span["end"],
        predicate_span["type"],
    )
    identity = (start, end, primary_type)
    if identity not in primary_spans:
        raise ValueError(f"{where}: predicate span does not match a primary span")
    attrs = predicate_span["attrs"]
    if not isinstance(attrs, dict) or not attrs:
        raise ValueError(f"{where}: attrs must be a nonempty object")
    if unknown := set(attrs) - set(policy.all_predicates):
        raise ValueError(f"{where}: predicates absent from policy: {sorted(unknown)}")
    normalized_attrs = {}
    for predicate, intervals in attrs.items():
        if not isinstance(intervals, list):
            raise ValueError(f"{where}: {predicate} intervals must be a list")
        normalized = []
        for index, interval in enumerate(intervals):
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or isinstance(interval[0], bool)
                or not isinstance(interval[0], int)
                or isinstance(interval[1], bool)
                or not isinstance(interval[1], int)
            ):
                raise ValueError(f"{where}: {predicate}[{index}] must be an integer interval")
            interval_start, interval_end = interval
            if not start <= interval_start < interval_end <= end:
                raise ValueError(f"{where}: {predicate}[{index}] escapes its primary span")
            if normalized and normalized[-1][1] > interval_start:
                raise ValueError(f"{where}: {predicate} intervals overlap or are unsorted")
            normalized.append([interval_start, interval_end])
        normalized_attrs[predicate] = normalized
    raw_weights = predicate_span.get("objective_weights", {"O": 1.0, "other": 1.0})
    if not isinstance(raw_weights, dict) or not {"O", "other"} <= set(raw_weights):
        raise ValueError(f"{where}: objective_weights must contain O and other")
    if unknown := set(raw_weights) - {"O", "other", *attrs}:
        raise ValueError(f"{where}: objective_weights names unknown attrs: {sorted(unknown)}")
    weights = {}
    for name, value in raw_weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{where}: objective weight {name} must be nonnegative")
        weights[name] = float(value)
    return identity, normalized_attrs, weights


def build_predicate_surface_pool(
    input_paths: Iterable[Path],
    *,
    policy: PredicateSurfacePolicy,
    pool_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not pool_version:
        raise ValueError("pool_version must be nonempty")
    input_paths = tuple(Path(path) for path in input_paths)
    aggregates: dict[str, tuple[dict[str, Any], _Aggregate]] = {}
    rows = 0
    primary_spans_seen = 0
    contextual_attrs_ignored = Counter()
    origins = Counter()
    for path in input_paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                text = row.get("text")
                language = row.get("lang")
                spans = row.get("spans")
                if not isinstance(text, str) or not isinstance(language, str) or not language:
                    raise ValueError(f"{path}:{line_number}: row requires text and lang")
                if not isinstance(spans, list):
                    raise ValueError(f"{path}:{line_number}: spans must be a list")
                primary_spans = set()
                for index, span in enumerate(spans):
                    if (
                        not isinstance(span, list)
                        or len(span) != 3
                        or isinstance(span[0], bool)
                        or not isinstance(span[0], int)
                        or isinstance(span[1], bool)
                        or not isinstance(span[1], int)
                        or not isinstance(span[2], str)
                        or not 0 <= span[0] < span[1] <= len(text)
                    ):
                        raise ValueError(f"{path}:{line_number}:spans[{index}] is invalid")
                    primary_spans.add((span[0], span[1], span[2]))
                raw_predicate_spans = row.get("predicate_spans", ())
                if not isinstance(raw_predicate_spans, (list, tuple)):
                    raise ValueError(f"{path}:{line_number}: predicate_spans must be a list")
                predicate_by_primary = {}
                for index, predicate_span in enumerate(raw_predicate_spans):
                    identity, attrs, weights = _validate_source_predicate_span(
                        predicate_span,
                        text=text,
                        primary_spans=primary_spans,
                        policy=policy,
                        where=f"{path}:{line_number}:predicate_spans[{index}]",
                    )
                    if identity in predicate_by_primary:
                        raise ValueError(f"{path}:{line_number}: duplicate predicate primary span")
                    predicate_by_primary[identity] = (attrs, weights)
                origin = row.get("surface_origin")
                if not isinstance(origin, str) or not origin:
                    raise ValueError(f"{path}:{line_number}: surface_origin is required")
                admitted_origins = policy.real_original_origins | policy.admitted_non_original_origins
                if origin not in admitted_origins:
                    raise ValueError(f"{path}:{line_number}: unrecognized surface_origin {origin!r}")
                origins[origin] += 1
                source_name = str(row.get("src") or row.get("source") or "unknown")
                source_hash = str(row.get("seed_id") or semantic_sha256(row))
                predicate_seed = row.get("predicate_seed")
                method = (
                    str(predicate_seed.get("method"))
                    if isinstance(predicate_seed, dict) and predicate_seed.get("method")
                    else "unspecified"
                )
                for start, end, primary_type in sorted(primary_spans):
                    primary_spans_seen += 1
                    attrs, weights = predicate_by_primary.get(
                        (start, end, primary_type), ({}, {"O": 0.0, "other": 0.0})
                    )
                    contextual_attrs_ignored.update(set(attrs) & set(policy.contextual_predicates))
                    intrinsic_attrs = {
                        predicate: [
                            [interval_start - start, interval_end - start]
                            for interval_start, interval_end in intervals
                        ]
                        for predicate, intervals in attrs.items()
                        if predicate in policy.intrinsic_predicates
                    }
                    intrinsic_weights = {"O": float(weights["O"]), "other": float(weights["other"])}
                    intrinsic_weights.update(
                        {
                            predicate: float(weights[predicate])
                            for predicate in intrinsic_attrs
                            if predicate in weights
                        }
                    )
                    value = text[start:end]
                    base = {
                        "schema": POOL_ENTRY_SCHEMA,
                        "schema_version": POOL_ENTRY_VERSION,
                        "pool_version": pool_version,
                        "policy_sha256": policy.sha256,
                        "split": "train",
                        "lang": language,
                        "primary_type": primary_type,
                        "value": value,
                        "normalized": normalized_surface(value),
                        "surface_origin": origin,
                        "intrinsic_attrs": intrinsic_attrs,
                        "objective_weights": intrinsic_weights,
                        "positive_signature": sorted(
                            predicate for predicate, intervals in intrinsic_attrs.items() if intervals
                        ),
                        "known_negative_intrinsic": sorted(
                            predicate for predicate, intervals in intrinsic_attrs.items() if not intervals
                        ),
                    }
                    key = semantic_sha256(base)
                    if key not in aggregates:
                        aggregates[key] = (
                            base,
                            _Aggregate(
                                count=0,
                                source_counts=Counter(),
                                source_record_hashes=set(),
                                predicate_seed_methods=set(),
                            ),
                        )
                    _base, aggregate = aggregates[key]
                    aggregate.count += 1
                    aggregate.source_counts[source_name] += 1
                    aggregate.source_record_hashes.add(source_hash)
                    aggregate.predicate_seed_methods.add(method)
    entries = []
    for key in sorted(aggregates):
        base, aggregate = aggregates[key]
        entry_id = semantic_sha256([pool_version, key])[:20]
        entries.append(
            {
                **base,
                "entry_id": entry_id,
                "count": aggregate.count,
                "source_counts": dict(sorted(aggregate.source_counts.items())),
                "source_record_hashes": sorted(aggregate.source_record_hashes),
                "predicate_seed_methods": sorted(aggregate.predicate_seed_methods),
            }
        )
    support_by_group = Counter()
    distinct_by_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    support_by_predicate: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    support_by_signature: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for entry in entries:
        group = (entry["lang"], entry["primary_type"], entry["surface_origin"])
        support_by_group[group] += entry["count"]
        distinct_by_group[group].add(entry["normalized"])
        for predicate in entry["positive_signature"]:
            support_by_predicate[(entry["lang"], entry["primary_type"], predicate)].add(entry["normalized"])
        signature = "+".join(entry["positive_signature"]) or "<none>"
        support_by_signature[(entry["lang"], entry["primary_type"], signature)].add(entry["normalized"])
    report = {
        "schema": "pii-predicate-surface-pool-report",
        "version": 1,
        "status": "materialized_not_quality_evidence",
        "pool_version": pool_version,
        "policy": policy.receipt(),
        "inputs": [{"path": str(Path(path)), "sha256": file_sha256(Path(path))} for path in input_paths],
        "rows": rows,
        "primary_spans": primary_spans_seen,
        "entries": len(entries),
        "surface_origins": dict(sorted(origins.items())),
        "contextual_attrs_excluded_from_surface_index": dict(sorted(contextual_attrs_ignored.items())),
        "groups": [
            {
                "lang": language,
                "primary_type": primary_type,
                "surface_origin": origin,
                "observed_count": support_by_group[(language, primary_type, origin)],
                "distinct_values": len(distinct_by_group[(language, primary_type, origin)]),
            }
            for language, primary_type, origin in sorted(support_by_group)
        ],
        "predicate_distinct_support": [
            {
                "lang": language,
                "primary_type": primary_type,
                "predicate": predicate,
                "distinct_values": len(values),
            }
            for (language, primary_type, predicate), values in sorted(support_by_predicate.items())
        ],
        "signature_distinct_support": [
            {
                "lang": language,
                "primary_type": primary_type,
                "signature": signature,
                "distinct_values": len(values),
            }
            for (language, primary_type, signature), values in sorted(support_by_signature.items())
        ],
    }
    return entries, report


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--pool-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.exists() or args.report.exists():
            raise FileExistsError(args.out if args.out.exists() else args.report)
        policy = PredicateSurfacePolicy.load(args.policy)
        entries, report = build_predicate_surface_pool(
            args.input,
            policy=policy,
            pool_version=args.pool_version,
        )
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, entries)
    report["output"] = {
        "path": str(args.out),
        "sha256": file_sha256(args.out),
        "entries": len(entries),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"entries": len(entries), "rows": report["rows"]}, sort_keys=True))
    print(args.out)
    print(args.report)


if __name__ == "__main__":
    main()
