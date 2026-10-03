"""Deterministic surface realization after trainer row selection.

The batch-sampler wrapper assigns each sampled occurrence a stable draw nonce.
``SampleTimeSurfaceRealizer`` consumes that nonce immediately before
tokenization, so replaying one stored carrier can expose the model to a new
surface without making the result depend on DataLoader worker scheduling.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    from pii_instantiate_transport import Filler
    from pii_locale_render import LocaleRenderer
    from pii_seed_tree import SeedTree
    from pii_surface_pool import SurfacePool

    from pii_surface.mix_policy import SurfaceMixPolicy
    from pii_surface.predicate_pool import PredicateSurfacePolicy, PredicateSurfacePool
    from pii_surface.rerealize import (
        file_sha256,
        placeholderize_row,
        realize_placeholder_row,
        surface_generator_receipt,
    )
except ModuleNotFoundError:  # Imported as scripts.pii_surface.sample_time in tests.
    from scripts.pii_instantiate_transport import Filler
    from scripts.pii_locale_render import LocaleRenderer
    from scripts.pii_seed_tree import SeedTree
    from scripts.pii_surface.mix_policy import SurfaceMixPolicy
    from scripts.pii_surface.predicate_pool import PredicateSurfacePolicy, PredicateSurfacePool
    from scripts.pii_surface.rerealize import (
        file_sha256,
        placeholderize_row,
        realize_placeholder_row,
        surface_generator_receipt,
    )
    from scripts.pii_surface_pool import SurfacePool


class SurfaceDrawBatchSampler:
    """Attach a monotone draw nonce to every emitted sample request."""

    def __init__(self, batch_sampler: Any):
        self.batch_sampler = batch_sampler
        self.batch_size = batch_sampler.batch_size
        self.drop_last = getattr(batch_sampler, "drop_last", False)
        self._next_draw_nonce = 0

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[list[tuple[int, bool | None, int]]]:
        for batch in self.batch_sampler:
            requests = []
            for item in batch:
                if isinstance(item, tuple):
                    if len(item) != 2 or not isinstance(item[1], bool):
                        raise ValueError(f"invalid wrapped sample request {item!r}")
                    index, sampled_variant = item
                else:
                    index, sampled_variant = item, None
                requests.append((int(index), sampled_variant, self._next_draw_nonce))
                self._next_draw_nonce += 1
            yield requests

    def summary(self) -> str:
        return "surface_draw_nonce=monotone-sampled-occurrence-v1 " + self.batch_sampler.summary()


class SampleTimeSurfaceRealizer:
    """Realize eligible exact-span rows from one immutable configuration."""

    def __init__(
        self,
        *,
        surface_pool_path: Path | None = None,
        surface_pool_split: str = "train",
        surface_recipe_path: Path | None = None,
        locale_profile_path: Path | None = None,
        materialization_version: str,
        seed: int,
        row_string_equals: Mapping[str, str],
        link_repeated_entities: bool = True,
        reject_unlocalized_categorical: bool = False,
        context_generator_path: Path | None = None,
        context_generator_rate: float = 1.0,
        context_generator_route: str = "auto",
        predicate_surface_pool_path: Path | None = None,
        predicate_surface_policy_path: Path | None = None,
    ):
        if surface_pool_split != "train":
            raise ValueError("sample-time training realization requires the train surface-pool split")
        if not materialization_version:
            raise ValueError("sample-time materialization version must be nonempty")
        if not row_string_equals:
            raise ValueError("sample-time realization requires at least one exact row predicate")
        legacy_configuration = (surface_pool_path, surface_recipe_path, locale_profile_path)
        if any(value is not None for value in legacy_configuration) and not all(
            value is not None for value in legacy_configuration
        ):
            raise ValueError("legacy surface realization requires pool, recipe, and locale profile")
        if (predicate_surface_pool_path is None) != (predicate_surface_policy_path is None):
            raise ValueError("predicate surface realization requires both pool and policy")
        if not any(value is not None for value in (*legacy_configuration, predicate_surface_pool_path)):
            raise ValueError("sample-time realization requires a legacy or predicate surface pool")
        self.surface_pool_path = Path(surface_pool_path) if surface_pool_path is not None else None
        self.surface_pool_split = surface_pool_split
        self.surface_recipe_path = Path(surface_recipe_path) if surface_recipe_path is not None else None
        self.locale_profile_path = Path(locale_profile_path) if locale_profile_path is not None else None
        self.predicate_surface_pool_path = (
            Path(predicate_surface_pool_path) if predicate_surface_pool_path is not None else None
        )
        self.predicate_surface_policy_path = (
            Path(predicate_surface_policy_path) if predicate_surface_policy_path is not None else None
        )
        self.materialization_version = materialization_version
        self.seed_tree = SeedTree(int(seed))
        self.row_string_equals = dict(
            sorted((str(key), str(value)) for key, value in row_string_equals.items())
        )
        self.link_repeated_entities = bool(link_repeated_entities)
        self.reject_unlocalized_categorical = bool(reject_unlocalized_categorical)
        self.context_generator_path = (
            Path(context_generator_path) if context_generator_path is not None else None
        )
        self.context_generator_rate = float(context_generator_rate)
        self.context_generator_route = str(context_generator_route)
        if not 0.0 <= self.context_generator_rate <= 1.0:
            raise ValueError("context generator rate must be in [0, 1]")
        if self.context_generator_path is None and self.context_generator_rate != 1.0:
            raise ValueError("context generator rate requires a context generator")
        if self.context_generator_path is None and self.context_generator_route != "auto":
            raise ValueError("context generator route requires a context generator")
        if self.context_generator_path is not None and self.surface_pool_path is None:
            raise ValueError("context generator realization requires the legacy surface configuration")
        self.surface_pool = (
            SurfacePool.load(self.surface_pool_path, split=self.surface_pool_split)
            if self.surface_pool_path is not None
            else None
        )
        self.surface_policy = (
            SurfaceMixPolicy.load(self.surface_recipe_path) if self.surface_recipe_path is not None else None
        )
        self.predicate_surface_policy = (
            PredicateSurfacePolicy.load(self.predicate_surface_policy_path)
            if self.predicate_surface_policy_path is not None
            else None
        )
        self.predicate_surface_pool = (
            PredicateSurfacePool.load(
                self.predicate_surface_pool_path,
                policy=self.predicate_surface_policy,
                split="train",
            )
            if self.predicate_surface_pool_path is not None
            else None
        )
        if self.context_generator_path is None:
            self.context_generator = None
        elif (self.context_generator_path / "config.json").is_file():
            if self.context_generator_route != "auto":
                raise ValueError("a single context generator does not accept a language route mode")
            try:
                from pii_surface.context_generator import LoadedContextSurfaceGenerator
            except ModuleNotFoundError:
                from scripts.pii_surface.context_generator import LoadedContextSurfaceGenerator

            self.context_generator = LoadedContextSurfaceGenerator(self.context_generator_path)
        else:
            try:
                from pii_surface.language_conditioning_bundle import (
                    INTRINSIC_WINNER,
                    LoadedLanguageConditioningSurfaceGenerator,
                )
            except ModuleNotFoundError:
                from scripts.pii_surface.language_conditioning_bundle import (
                    INTRINSIC_WINNER,
                    LoadedLanguageConditioningSurfaceGenerator,
                )

            route_mode = (
                INTRINSIC_WINNER if self.context_generator_route == "auto" else self.context_generator_route
            )
            self.context_generator = LoadedLanguageConditioningSurfaceGenerator(
                self.context_generator_path,
                route_mode=route_mode,
            )
        self._fillers: dict[str, Filler] = {}

    def eligible(self, row: Mapping[str, Any]) -> bool:
        return all(
            isinstance(row.get(field), str) and row[field] == expected
            for field, expected in self.row_string_equals.items()
        )

    def _filler(self, language: str) -> Filler:
        filler = self._fillers.get(language)
        if filler is None:
            filler = Filler(
                language,
                self.seed_tree.root_seed,
                surface_pool=self.surface_pool,
                reject_unlocalized_categorical=self.reject_unlocalized_categorical,
                locale_renderer=(
                    LocaleRenderer(language, self.locale_profile_path)
                    if self.locale_profile_path is not None
                    else None
                ),
                surface_policy=self.surface_policy,
            )
            self._fillers[language] = filler
        return filler

    def realize(self, row: Mapping[str, Any], draw_nonce: int) -> dict[str, Any]:
        if draw_nonce < 0:
            raise ValueError("surface draw nonce must be nonnegative")
        source_row = dict(row)
        if not self.eligible(source_row):
            return source_row
        if not source_row.get("spans"):
            return source_row
        placeholder_row = placeholderize_row(
            source_row,
            link_repeated_entities=self.link_repeated_entities,
            opaque_entity_slots=self.predicate_surface_pool is not None,
        )
        return realize_placeholder_row(
            source_row,
            placeholder_row,
            filler=self._filler(placeholder_row["lang"]),
            seed_tree=self.seed_tree,
            realization_index=draw_nonce,
            materialization_version=self.materialization_version,
            partition_role="training",
            surface_generator=self.context_generator,
            surface_generator_rate=self.context_generator_rate,
            predicate_surface_pool=self.predicate_surface_pool,
            predicate_surface_policy=self.predicate_surface_policy,
        )

    def eligibility_summary(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_language = Counter(str(row.get("lang") or "<unknown>") for row in rows if self.eligible(row))
        return {
            "eligible_rows": sum(by_language.values()),
            "eligible_rows_by_language": dict(sorted(by_language.items())),
            "total_rows": len(rows),
        }

    def receipt(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timing": "after-sampler-selection-before-tokenization",
            "draw_identity": "monotone-sampled-occurrence-v1",
            "materialization_version": self.materialization_version,
            "seed": self.seed_tree.root_seed,
            "row_string_equals": self.row_string_equals,
            "link_repeated_entities": self.link_repeated_entities,
            "reject_unlocalized_categorical": self.reject_unlocalized_categorical,
            "length_binning": "stored-row-token-length-estimate; realized drift is not re-binned",
            "surface_pool": (
                {
                    "path": str(self.surface_pool_path),
                    "sha256": file_sha256(self.surface_pool_path),
                    "split": self.surface_pool_split,
                    "version": self.surface_pool.version,
                }
                if self.surface_pool is not None
                else None
            ),
            "surface_recipe": self.surface_policy.receipt() if self.surface_policy is not None else None,
            "locale_profile": (
                {
                    "path": str(self.locale_profile_path),
                    "sha256": file_sha256(self.locale_profile_path),
                }
                if self.locale_profile_path is not None
                else None
            ),
            "predicate_surface_pool": (
                self.predicate_surface_pool.receipt() if self.predicate_surface_pool is not None else None
            ),
            "predicate_surface_policy": (
                self.predicate_surface_policy.receipt() if self.predicate_surface_policy is not None else None
            ),
            "context_surface_generator": (
                surface_generator_receipt(self.context_generator, self.context_generator_rate)
            ),
            **self.eligibility_summary(rows),
        }
