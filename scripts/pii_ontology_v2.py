#!/usr/bin/env python
"""Fail-closed loader and validator for the PII target ontology v2.

The spec lives in scripts/pii_tagset_v2.yaml. Version 1
(scripts/pii_tagset.yaml) stays immutable and is the authority for which
source schemas and labels exist; this module validates that v2 covers it
exactly.

Reusable API for later discrimination/relabel code:

  load_ontology()                      -> cached OntologyV2
  ontology.primary_types               -> the 29 BIOES primary classes
  ontology.bioes_labels()              -> O plus B/I/E/S per primary class
  ontology.canonical_fallback(node)    -> v2 projection of a v1 canonical node
  ontology.canonical_acceptable(node)  -> its fixed-span retyping candidates
  ontology.source_fallback(schema, label) / source_acceptable(schema, label)
  ontology.source_labels(schema)       -> the v1 compatibility label surface
  ontology.source_extension_labels(schema) -> declared beyond-v1 labels
  ontology.source_route(schema, label) / source_promotion(schema, label)
  ontology.channels_for(primary_type)  -> applicable attribute channels
  ontology.validate_dataset_metadata(metadata)
  ontology.validate_record(row, ...)   -> per-row gold contract
  ontology.validate_jsonl(path, ...)   -> whole-file gold contract
  ontology.admit_remap(fallback, acceptable, strengths) -> frozen admission rule

Self-test and file validation: python scripts/pii_ontology_v2.py [--jsonl PATH]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ONTOLOGY_PATH = Path(__file__).with_name("pii_tagset_v2.yaml")
COMPATIBILITY_TAGSET_PATH = Path(__file__).with_name("pii_tagset.yaml")
GOLD_SCHEMA_PATH = Path(__file__).with_name("pii_gold_v2.schema.json")

ONTOLOGY_VERSION = "pii-ontology-v2"
SCHEMA_VERSION = "pii-gold-v2"
PROJECTION_VERSION = "pii-v1-to-v2-projection-v2"
REMAP_ADMISSION_VERSION = "pii-ontology-v2-remap-admission-v1"

OUTSIDE = "O"
BIOES_PREFIXES = ("B", "I", "E", "S")
PRIMARY_COUNT = 29
CANONICAL_V1_NODE_COUNT = 116
ATTR_CHANNELS = ("care_provider", "family_name")
ATTR_SEMANTICS_VERSION = 1
ATTR_VALUE_KIND = "character_extents"
DATASET_METADATA_FIELDS = ("ontology_version", "schema_version", "attr_channels")
REGISTRY_FIELDS = ("applicable_types", "semantics_version", "value_kind")

# The proposal artifact merged money into `quantity` and so admitted 28
# classes; the adopted Q6 decision separates `monetary_amount`.
PROPOSAL_CLASS_COUNT = 28
ADMISSION_PRONGS = ("R", "A", "S")
COMPATIBILITY_RELATION_KINDS = ("bucket", "functional", "hierarchy", "indeterminacy")
WARM_INIT_CHECKPOINTS = (250, 500, 1000, 1500)

# A source extension declares a qualified-gold label the v1 tagset never mapped.
# Its route says how a later relabeler may resolve it; its promotion says whether
# it may be used at all. Both vocabularies are closed so a typo cannot widen them.
EXTENSION_ROUTES = (
    "composition",
    "constrained_discrimination",
    "currently_unrepresentable",
    "deterministic",
    "independent_review",
    "outside",
)
EXTENSION_PROMOTIONS = ("evaluation_only", "training_eligible", "withheld")
EXTENSION_PROVENANCE_KINDS = ("onboarding_manifest", "raw_scan")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTENSION_EVIDENCE = "research/pii/frontier/evidence/ontology-v2-source-extension-evidence-v1.json"

# ln(3) mean-logit units are required to retype a fixed source span away from
# its declared fallback. Retaining the fallback needs no margin.
REMAP_MARGIN_NATS = math.log(3.0)


class OntologyError(ValueError):
    """A spec, dataset-metadata, or gold-record contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OntologyError(message)


def _is_int(value: Any) -> bool:
    """Integers only: booleans are not accepted as offsets."""
    return isinstance(value, int) and not isinstance(value, bool)


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects a repeated mapping key at any nesting depth.

    PyYAML silently keeps the last of a repeated key, so a duplicated block
    parses cleanly and a later edit to the shadowed copy is discarded without a
    word. This is the same safe constructor set, only stricter.
    """

    # Set on a per-parse subclass so a duplicate-key error can name the caller's
    # source rather than PyYAML's generic stream label.
    ontology_source: str | None = None

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        # Check the node's own keys before flatten_mapping runs: after merge-key
        # expansion a repeat is the documented YAML override of an anchor, not a
        # duplicate, so checking afterwards would reject legitimate `<<:` reuse.
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                mark = key_node.start_mark
                where = self.ontology_source or mark.name
                raise OntologyError(
                    f"{where}:{mark.line + 1}: duplicate YAML key {key!r};"
                    " the later value would silently shadow the earlier one"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _strict_load(text: str, source: str) -> Any:
    """Parse with the strict loader, naming ``source`` in every error it raises.

    The duplicate-key error is an OntologyError raised from inside the loader,
    so it never passes through the YAMLError handler; the loader has to be told
    the caller's name for it rather than falling back to PyYAML's stream label.
    """
    loader = type("_SourcedStrictLoader", (_StrictSafeLoader,), {"ontology_source": source})
    try:
        return yaml.load(text, Loader=loader)
    except yaml.YAMLError as error:
        raise OntologyError(f"{source}: invalid YAML: {error}") from None


def load_yaml(path: Path | str) -> Any:
    """Parse a YAML file with the canonical strict-but-safe loader used here."""
    path = Path(path)
    return _strict_load(path.read_text(encoding="utf-8"), str(path))


def parse_yaml(text: str, *, source: str = "<string>") -> Any:
    """Parse a YAML string with the same strict loader (for tests and callers)."""
    return _strict_load(text, source)


class OntologyV2:
    """The validated v2 spec plus the operations later stages need."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        compatibility_tagset: Mapping[str, Any] | None = None,
        source: str = str(ONTOLOGY_PATH),
    ) -> None:
        self.source = source
        self.spec = spec
        self._validate(compatibility_tagset)

    # ---- spec validation -------------------------------------------------

    def _check(self, condition: bool, message: str) -> None:
        _require(condition, f"{self.source}: {message}")

    def _validate(self, compatibility_tagset: Mapping[str, Any] | None) -> None:
        spec = self.spec
        self._check(isinstance(spec, Mapping), "expected a YAML object")
        self._check(spec.get("version") == 2, f"version must be 2, got {spec.get('version')!r}")
        self._check(spec.get("ontology_version") == ONTOLOGY_VERSION, "ontology_version drift")
        self._check(spec.get("schema_version") == SCHEMA_VERSION, "schema_version drift")
        self._check(spec.get("projection_version") == PROJECTION_VERSION, "projection_version drift")
        self.projection_version = PROJECTION_VERSION
        self._check(spec.get("span_encoding") == "BIOES", "primary spans must be BIOES")
        self._check(spec.get("outside_label") == OUTSIDE, "outside_label must be O")

        self.families = dict(spec["families"])
        self.actions = dict(spec["actions"])
        self._check(bool(self.families) and bool(self.actions), "families and actions are required")
        for name, description in list(self.families.items()) + list(self.actions.items()):
            self._check(
                isinstance(name, str) and isinstance(description, str) and description.strip() != "",
                f"metadata entry {name!r} needs a nonempty description",
            )
        self.admission_prongs = dict(spec["admission_prongs"])
        self._check(
            tuple(sorted(self.admission_prongs)) == tuple(sorted(ADMISSION_PRONGS)),
            f"admission_prongs must be exactly {list(ADMISSION_PRONGS)}",
        )
        self._validate_adoption_note(spec["adoption_note"])

        primary = spec["primary"]
        self._check(isinstance(primary, Mapping), "primary must be a mapping")
        self._check(
            len(primary) == PRIMARY_COUNT,
            f"expected {PRIMARY_COUNT} primary classes, got {len(primary)}",
        )
        self.primary = {str(name): dict(entry) for name, entry in primary.items()}
        self.primary_types = tuple(sorted(self.primary))
        self._check(OUTSIDE not in self.primary, "O is a sentinel target, not a primary class")
        for name, entry in self.primary.items():
            self._check(
                entry.get("family") in self.families,
                f"primary {name}: unknown family {entry.get('family')!r}",
            )
            self._check(
                entry.get("action") in self.actions,
                f"primary {name}: unknown action {entry.get('action')!r}",
            )
            prongs = entry.get("prongs")
            self._check(
                isinstance(prongs, list) and bool(prongs),
                f"primary {name}: prongs must be a nonempty list",
            )
            prongs = list(prongs or ())
            self._check(
                len(set(prongs)) == len(prongs),
                f"primary {name}: duplicate admission prongs",
            )
            unknown_prongs = sorted(set(prongs) - set(self.admission_prongs))
            self._check(
                not unknown_prongs,
                f"primary {name}: unknown admission prongs: {', '.join(unknown_prongs)}",
            )
            definition = entry.get("definition")
            self._check(
                isinstance(definition, str) and definition.strip() != "",
                f"primary {name}: missing definition",
            )
        used_families = {entry["family"] for entry in self.primary.values()}
        unused = sorted(set(self.families) - used_families)
        self._check(not unused, f"families no primary class uses: {', '.join(unused)}")
        used_actions = {entry["action"] for entry in self.primary.values()}
        unused_actions = sorted(set(self.actions) - used_actions)
        self._check(not unused_actions, f"actions no primary class uses: {', '.join(unused_actions)}")
        self._targets = set(self.primary_types) | {OUTSIDE}

        self._validate_attr_channels(spec["attr_channels"])
        self._validate_predicate_contract(spec["predicate_contract"])
        self._validate_gold_metadata_contract(spec["gold_metadata_contract"])

        self.old_v1 = {
            str(node): self._entry(f"old_v1.{node}", entry) for node, entry in spec["old_v1"].items()
        }
        self._check(
            len(self.old_v1) == CANONICAL_V1_NODE_COUNT,
            f"expected {CANONICAL_V1_NODE_COUNT} canonical v1 nodes, got {len(self.old_v1)}",
        )

        self.sources = {
            str(schema): {
                str(label): self._entry(f"sources.{schema}.{label}", entry)
                for label, entry in mapping.items()
            }
            for schema, mapping in spec["sources"].items()
        }
        for schema, mapping in self.sources.items():
            self._check(bool(mapping), f"sources.{schema}: empty label space")

        self._validate_remap_admission(spec["remap_admission"])
        self._validate_methodology(spec["compatibility_scoring"], spec["warm_init"])
        self._validate_compatibility(compatibility_tagset)

    def _validate_adoption_note(self, note: Any) -> None:
        """The 28-vs-29 provenance must stay explicit rather than implied."""
        self._check(isinstance(note, Mapping), "adoption_note must be a mapping")
        self._check(
            note.get("proposal_class_count") == PROPOSAL_CLASS_COUNT,
            f"adoption_note.proposal_class_count must be {PROPOSAL_CLASS_COUNT}",
        )
        self._check(
            note.get("adopted_class_count") == PRIMARY_COUNT,
            f"adoption_note.adopted_class_count must be {PRIMARY_COUNT}",
        )
        self._check(
            note.get("proposal_merged_money_into_quantity") is True,
            "adoption_note must record that the proposal merged money into quantity",
        )
        self._check(
            note.get("difference") == "q6_separates_monetary_amount_from_quantity",
            "adoption_note must record the Q6 separation as the difference",
        )
        self._check(
            note.get("q6_currency_designator_alone") == OUTSIDE,
            "adoption_note must record that a bare currency designator is O",
        )
        self.adoption_note = dict(note)

    def _entry(self, where: str, entry: Any) -> tuple[str, tuple[str, ...]]:
        self._check(isinstance(entry, Mapping), f"{where}: expected a mapping")
        acceptable = entry.get("acceptable")
        fallback = entry.get("fallback")
        self._check(
            isinstance(acceptable, list) and bool(acceptable),
            f"{where}: acceptable must be a nonempty list",
        )
        unknown = sorted(set(acceptable) - self._targets)
        self._check(not unknown, f"{where}: acceptable targets outside primary+O: {', '.join(unknown)}")
        self._check(
            len(set(acceptable)) == len(acceptable),
            f"{where}: duplicate acceptable targets",
        )
        self._check(fallback in acceptable, f"{where}: fallback {fallback!r} is not in its acceptable set")
        return str(fallback), tuple(acceptable)

    def _validate_attr_channels(self, registry: Any) -> None:
        self._check(isinstance(registry, Mapping), "attr_channels must be a mapping")
        self._check(
            tuple(sorted(registry)) == ATTR_CHANNELS,
            f"attr_channels must be exactly {list(ATTR_CHANNELS)}, got {sorted(registry)}",
        )
        self.attr_channels = {}
        for channel, entry in registry.items():
            where = f"attr_channels.{channel}"
            self._check(isinstance(entry, Mapping), f"{where}: expected a mapping")
            applicable = entry.get("applicable_types")
            self._check(
                isinstance(applicable, list) and bool(applicable),
                f"{where}: applicable_types must be a nonempty list",
            )
            unknown = sorted(set(applicable) - set(self.primary_types))
            self._check(not unknown, f"{where}: unknown applicable types: {', '.join(unknown)}")
            self._check(
                entry.get("semantics_version") == ATTR_SEMANTICS_VERSION,
                f"{where}: semantics_version must be {ATTR_SEMANTICS_VERSION}",
            )
            self._check(
                entry.get("value_kind") == ATTR_VALUE_KIND,
                f"{where}: value_kind must be {ATTR_VALUE_KIND!r}, got {entry.get('value_kind')!r}",
            )
            definition = entry.get("definition")
            self._check(
                isinstance(definition, str) and definition.strip() != "",
                f"{where}: missing definition",
            )
            self.attr_channels[str(channel)] = {
                "applicable_types": tuple(sorted(applicable)),
                "semantics_version": int(entry["semantics_version"]),
                "value_kind": str(entry["value_kind"]),
            }
        self._channels_by_type: dict[str, tuple[str, ...]] = {}
        for channel, entry in sorted(self.attr_channels.items()):
            for primary_type in entry["applicable_types"]:
                self._channels_by_type.setdefault(primary_type, ())
                self._channels_by_type[primary_type] += (channel,)

    def _validate_predicate_contract(self, contract: Any) -> None:
        """Raw logits are per model position; only decoded output is per span."""
        self._check(isinstance(contract, Mapping), "predicate_contract must be a mapping")
        bioes_label_count = 1 + len(BIOES_PREFIXES) * PRIMARY_COUNT
        expected = {
            "objective": "independent_masked_bce",
            "encoding": "not_bioes",
            "token_supervision": "existential_over_characters",
            "raw_output_granularity": "model_position",
            "primary_logit_shape": ["B", "T", bioes_label_count],
            "predicate_logit_shape": ["B", "T", "K"],
            "logit_axes": ["batch", "position", "channel"],
            "position_granularity": "declared_by_producer_metadata",
            "position_granularity_values": ["token", "character"],
        }
        for field, value in expected.items():
            self._check(
                contract.get(field) == value,
                f"predicate_contract.{field} must be {value!r}, got {contract.get(field)!r}",
            )
        self._check(
            "primary_span" not in contract.get("logit_axes", ()),
            "raw logit axes are positions, never decoded primary spans",
        )
        self._check(
            "predicate_logit_axes" not in contract,
            "predicate_logit_axes is superseded by logit_axes; a stale copy would contradict it",
        )
        evaluation = contract["evaluation"]
        self._check(
            evaluation.get("predicate_scope") == "known_applicable_gold_primary_spans",
            "predicate evaluation is restricted to known applicable gold primary spans",
        )
        self._check(evaluation.get("missing_gold_key") == "excluded", "a missing gold key must be excluded")
        self._check(
            evaluation.get("primary_span_metrics_use_predicates") is False,
            "primary span metrics never use predicate values",
        )
        self.predicate_contract = dict(contract)

    def _validate_gold_metadata_contract(self, contract: Any) -> None:
        self._check(isinstance(contract, Mapping), "gold_metadata_contract must be a mapping")
        self._check(
            tuple(contract.get("required_fields", ())) == DATASET_METADATA_FIELDS,
            f"gold metadata must require {list(DATASET_METADATA_FIELDS)}",
        )
        self._check(contract.get("registry_match") == "exact", "registry_match must be exact")
        self._check(
            tuple(contract.get("registry_fields", ())) == REGISTRY_FIELDS,
            f"registry_fields must be {list(REGISTRY_FIELDS)}",
        )
        self._check(contract.get("unknown_row_fields") == "preserved", "unknown row fields must be preserved")
        self.gold_metadata_contract = dict(contract)
        self.forbid_overlapping_spans = contract.get("primary_span_overlap") == "forbidden"

    def _validate_remap_admission(self, admission: Any) -> None:
        self._check(isinstance(admission, Mapping), "remap_admission must be a mapping")
        self._check(admission.get("version") == REMAP_ADMISSION_VERSION, "remap_admission version drift")
        threshold = admission.get("margin_threshold_nats")
        self._check(
            isinstance(threshold, float) and threshold == REMAP_MARGIN_NATS,
            f"remap_admission margin must be ln(3) = {REMAP_MARGIN_NATS!r}, got {threshold!r}",
        )
        self._check(admission.get("margin_threshold_expression") == "ln(3)", "margin expression drift")
        self._check(admission.get("ties") == "retain_fallback", "ties must retain the fallback")
        self._check(
            admission.get("fallback_winner_requires_margin") is False,
            "a fallback winner needs no margin",
        )
        self._check(
            admission.get("argmax_restricted_to") == "declared_acceptable_set",
            "the argmax is restricted to the declared acceptable set",
        )
        self._check(admission.get("logits") == "raw", "candidate strength uses raw logits")
        self._check(admission.get("status") == "proposal_only", "model output remains a proposal")
        self._check(
            admission.get("evaluation_promotion_automatic") is False,
            "evaluation promotion is never automatic",
        )
        self.remap_admission = dict(admission)

    def _validate_methodology(self, scoring: Any, warm_init: Any) -> None:
        """Frozen scoring and warm-init methodology, recorded and pinned here."""
        self._check(isinstance(scoring, Mapping), "compatibility_scoring must be a mapping")
        self._check(
            list(scoring.get("primary_matching", ())) == ["exact", "symmetric_80pct_overlap"],
            "primary matching is exact plus symmetric 80% overlap",
        )
        self._check(
            scoring.get("assignment") == "maximum_cardinality_one_to_one",
            "matching is a maximum-cardinality one-to-one assignment",
        )
        criterion = scoring.get("overlap_criterion")
        self._check(
            isinstance(criterion, str) and criterion.strip() != "",
            "compatibility_scoring must state the symmetric overlap criterion",
        )
        kinds = scoring.get("compatibility_relation_kinds")
        self._check(
            isinstance(kinds, list)
            and len(set(kinds)) == len(kinds)
            and tuple(sorted(kinds)) == COMPATIBILITY_RELATION_KINDS,
            f"compatibility_relation_kinds must be exactly {list(COMPATIBILITY_RELATION_KINDS)},"
            f" got {kinds!r}",
        )
        semantics = scoring.get("compatibility_relation_semantics")
        self._check(isinstance(semantics, Mapping), "each relation kind needs a stated meaning")
        self._check(
            tuple(sorted(semantics)) == COMPATIBILITY_RELATION_KINDS,
            "compatibility_relation_semantics must cover exactly the declared relation kinds",
        )
        for kind, meaning in semantics.items():
            self._check(
                isinstance(meaning, str) and meaning.strip() != "",
                f"compatibility_relation_semantics.{kind}: missing meaning",
            )
        self._check(
            scoring.get("permanent_v2_gold_scoring_uses_compatibility") is False,
            "permanent v2 gold scoring never scores through the compatibility relation",
        )
        self.compatibility_scoring = dict(scoring)

        self._check(isinstance(warm_init, Mapping), "warm_init must be a mapping")
        expected = {
            "one_to_one_rows": "copy",
            "merged_rows": "mass_weighted",
            "merged_bias": "contributor_log_sum_exp",
            "merged_bias_formula": "b_new = logsumexp(b_1, ..., b_n)",
            "merged_bias_rationale": None,
            "merged_rows_formula": None,
            "outside_row": "copy",
            "split_rows": "duplicate_with_symmetry_breaking",
            "symmetry_breaking": "deliberate_for_future_splits",
            "symmetry_breaking_rationale": None,
            "new_predicate_heads": "initialized_separately",
            "matched_control": "same_v11_encoder_with_fresh_or_refitted_v2_head",
            "encoder_schedule": "frozen_for_initial_head_fit_then_unfrozen",
            "adoption_criterion": ("better_at_early_checkpoints_and_no_worse_at_the_matched_final_rung"),
        }
        for field, value in expected.items():
            found = warm_init.get(field)
            if value is None:
                self._check(
                    isinstance(found, str) and found.strip() != "",
                    f"warm_init.{field}: a stated formula or rationale is required",
                )
            else:
                self._check(
                    found == value,
                    f"warm_init.{field} must be {value!r}, got {found!r}",
                )
        self._check(
            tuple(warm_init.get("checkpoints", ())) == WARM_INIT_CHECKPOINTS,
            f"warm_init.checkpoints must be {list(WARM_INIT_CHECKPOINTS)},"
            f" got {warm_init.get('checkpoints')!r}",
        )
        pilot = warm_init.get("matched_pilot")
        self._check(isinstance(pilot, Mapping), "warm_init.matched_pilot must be a mapping")
        self._check(
            pilot.get("first_stage") == "fit_new_head_with_encoder_frozen",
            "the matched pilot fits the new head with the encoder frozen first",
        )
        self._check(
            pilot.get("adopt_warm_init_only_if_matched_pilot_supports_it") is True,
            "warm init is adopted only if its matched pilot supports it",
        )
        self._check(
            bool(pilot.get("must_match")),
            "the matched pilot must declare what it holds fixed",
        )
        self.warm_init = dict(warm_init)

    def _validate_compatibility(self, compatibility_tagset: Mapping[str, Any] | None) -> None:
        declared = self.spec["compatibility_tagset"]
        self._check(declared.get("version") == 1, "the compatibility tagset must be version 1")
        path = Path(declared["path"])
        if compatibility_tagset is None:
            candidate = path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
            if not candidate.exists():
                candidate = COMPATIBILITY_TAGSET_PATH
            compatibility_tagset = load_yaml(candidate)
        if not isinstance(compatibility_tagset, Mapping):
            self._check(False, f"{path}: expected a YAML object")
            return

        v1_nodes = set(compatibility_tagset["nodes"])
        missing = sorted(v1_nodes - set(self.old_v1))
        extra = sorted(set(self.old_v1) - v1_nodes)
        self._check(
            not missing, f"old_v1 is missing {len(missing)} canonical nodes: {', '.join(missing[:6])}"
        )
        self._check(not extra, f"old_v1 declares unknown canonical nodes: {', '.join(extra[:6])}")

        v1_sources = compatibility_tagset["sources"]
        missing_schemas = sorted(set(v1_sources) - set(self.sources))
        extra_schemas = sorted(set(self.sources) - set(v1_sources))
        self._check(not missing_schemas, f"sources is missing schemas: {', '.join(missing_schemas)}")
        self._check(not extra_schemas, f"sources declares unknown schemas: {', '.join(extra_schemas)}")
        for schema, mapping in v1_sources.items():
            missing_labels = sorted(set(mapping) - set(self.sources[schema]))
            extra_labels = sorted(set(self.sources[schema]) - set(mapping))
            self._check(
                not missing_labels,
                f"sources.{schema}: missing labels: {', '.join(missing_labels[:6])}",
            )
            self._check(
                not extra_labels,
                f"sources.{schema}: unknown labels: {', '.join(extra_labels[:6])}",
            )

        self._validate_derivation(compatibility_tagset["nodes"], v1_sources)
        self._validate_source_extensions(v1_sources)
        self._validate_composition_contracts()
        self._validate_source_metadata_channels()

    def _validate_derivation(self, v1_nodes: Mapping[str, Any], v1_sources: Mapping[str, Any]) -> None:
        """Recompute every acceptable set from v1 and reject any difference.

        The explicit sets in the YAML stay authoritative for consumers, but a
        set that no longer follows from its declared construction is drift, so
        it fails here rather than silently outliving the projection it came
        from.
        """
        construction = self.spec["acceptable_set_construction"]
        self._check(isinstance(construction, Mapping), "acceptable_set_construction must be a mapping")
        self._check(
            construction.get("rule") == "functional_projection_plus_descendant_images_plus_curated_links",
            "acceptable_set_construction declares a rule this loader does not implement",
        )
        self._check(
            construction.get("ancestor_images_added") is False,
            "ancestor images are deliberately not added",
        )
        self._check(
            construction.get("curated_links_must_be_effective") is True
            and construction.get("source_overrides_must_be_effective") is True,
            "redundant curated links and overrides must be rejected, not tolerated",
        )

        children: dict[str, list[str]] = {node: [] for node in v1_nodes}
        for node, entry in v1_nodes.items():
            parent = entry.get("parent") if isinstance(entry, Mapping) else None
            if parent is None:
                continue
            self._check(parent in children, f"old_v1.{node}: unknown v1 parent {parent!r}")
            children[str(parent)].append(str(node))

        def descendants(node: str) -> set[str]:
            seen: set[str] = set()
            stack = list(children[node])
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(children[current])
            return seen

        curated = construction.get("curated_links") or {}
        self._check(isinstance(curated, Mapping), "curated_links must be a mapping")
        for node, targets in curated.items():
            self._check(node in self.old_v1, f"curated_links.{node}: unknown v1 node")
            self._check(
                isinstance(targets, list) and bool(targets),
                f"curated_links.{node}: targets must be a nonempty list",
            )
            self._check(
                len(set(targets)) == len(targets),
                f"curated_links.{node}: duplicate targets",
            )
            unknown = sorted(set(targets) - self._targets)
            self._check(
                not unknown,
                f"curated_links.{node}: targets outside primary+O: {', '.join(unknown)}",
            )

        for node, (fallback, acceptable) in self.old_v1.items():
            mechanical = {fallback} | {self.old_v1[child][0] for child in descendants(node)}
            links = set(curated.get(node, ()))
            if links:
                self._check(
                    bool(links - mechanical),
                    f"curated_links.{node}: adds nothing beyond the mechanical set; a redundant"
                    " link would keep a stale set alive after a projection changes",
                )
            self._check(
                set(acceptable) == mechanical | links,
                f"old_v1.{node}: acceptable set drifted from its declared construction;"
                f" expected {sorted(mechanical | links)}, got {sorted(acceptable)}",
            )

        # An override may adjust the acceptable set, the fallback, or both, but
        # only for the one named source cell. Shared canonical-node fallbacks are
        # never edited to repair a single corpus.
        overrides: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
        declared_overrides = construction.get("source_overrides") or []
        self._check(isinstance(declared_overrides, list), "source_overrides must be a list")
        for index, override in enumerate(declared_overrides):
            where = f"source_overrides[{index}]"
            self._check(isinstance(override, Mapping), f"{where}: expected a mapping")
            unknown_fields = sorted(set(override) - {"schema", "label", "acceptable", "fallback", "reason"})
            self._check(not unknown_fields, f"{where}: unknown fields: {', '.join(unknown_fields)}")
            schema, label = override.get("schema"), override.get("label")
            self._check(
                schema in v1_sources and label in v1_sources[schema],
                f"{where}: {schema}.{label} is not a v1 source label",
            )
            self._check((schema, label) not in overrides, f"{where}: duplicate override for {schema}.{label}")
            reason = override.get("reason")
            self._check(
                isinstance(reason, str) and reason.strip() != "",
                f"{where}: an override needs a stated reason",
            )
            node = str(v1_sources[schema][label])
            canonical_fallback, canonical_acceptable = self.old_v1[node]

            # Either half may be omitted, and the omitted half keeps its
            # canonical value: a corpus whose only defect is the default answer
            # must be able to say so without restating a set it did not change.
            self._check(
                "fallback" in override or "acceptable" in override,
                f"{where}: an override must declare a fallback, an acceptable set, or both",
            )
            targets = override.get("acceptable", list(canonical_acceptable))
            self._check(
                isinstance(targets, list) and bool(targets) and len(set(targets)) == len(targets),
                f"{where}: acceptable must be a nonempty duplicate-free list",
            )
            unknown = sorted(set(targets) - self._targets)
            self._check(not unknown, f"{where}: targets outside primary+O: {', '.join(unknown)}")
            fallback = override.get("fallback", canonical_fallback)
            self._check(
                fallback in targets,
                f"{where}: fallback {fallback!r} is not in the override's acceptable set",
            )
            self._check(
                set(targets) != set(canonical_acceptable) or fallback != canonical_fallback,
                f"{where}: override equals the canonical projection for {node}, so it changes nothing",
            )
            overrides[(str(schema), str(label))] = (str(fallback), tuple(targets))

        for schema, mapping in v1_sources.items():
            for label, node in mapping.items():
                node = str(node)
                self._check(node in self.old_v1, f"sources.{schema}.{label}: unknown v1 node {node!r}")
                canonical = self.old_v1[node]
                fallback, acceptable = self.sources[schema][label]
                expected_fallback, expected_acceptable = overrides.get((schema, label), canonical)
                self._check(
                    set(acceptable) == set(expected_acceptable),
                    f"sources.{schema}.{label}: acceptable set is neither its canonical set for"
                    f" {node} nor a declared override; expected {sorted(expected_acceptable)},"
                    f" got {sorted(acceptable)}",
                )
                self._check(
                    fallback == expected_fallback,
                    f"sources.{schema}.{label}: fallback {fallback!r} is neither the canonical"
                    f" fallback for {node} nor a declared override; expected"
                    f" {expected_fallback!r}",
                )
                self._check(
                    fallback in acceptable,
                    f"sources.{schema}.{label}: fallback {fallback!r} left its acceptable set",
                )
        self.source_overrides = {cell: value for cell, value in overrides.items()}

        money_labels = construction.get("explicit_money_labels") or []
        self._check(isinstance(money_labels, list) and bool(money_labels), "explicit_money_labels missing")
        for name in money_labels:
            projects_to_money = self.old_v1.get(str(name), (None,))[0] == "monetary_amount" or any(
                self.sources[schema][name][0] == "monetary_amount"
                for schema in self.sources
                if name in self.sources[schema]
            )
            self._check(
                projects_to_money,
                f"explicit_money_labels: {name!r} is not a v1 node or source label projecting to"
                " monetary_amount",
            )

    # ---- source extensions ----------------------------------------------

    def _resolve_repo_path(self, declared: Any, where: str) -> Path:
        """Resolve a declared artifact path, or reject it.

        A declaration may only name a tracked file inside this repository. An
        absolute path, a traversal that escapes the repository, or anything that
        normalizes into the private ``tasks/`` tree is rejected: a spec that
        could cite a private or external file would appear to satisfy the
        evidence checks while being uncheckable by anyone else.
        """
        self._check(
            isinstance(declared, str) and declared.strip() != "",
            f"{where}: an artifact path must be a nonempty string",
        )
        candidate = Path(str(declared))
        self._check(
            not candidate.is_absolute(),
            f"{where}: {declared!r} must be repository-relative, not absolute",
        )
        repo = REPO_ROOT.resolve()
        resolved = (repo / candidate).resolve()
        self._check(
            resolved != repo and resolved.is_relative_to(repo),
            f"{where}: {declared!r} resolves outside the repository",
        )
        self._check(
            not resolved.is_relative_to(repo / "tasks"),
            f"{where}: {declared!r} resolves into the private tasks/ tree, which a tracked spec may not cite",
        )
        return resolved

    def _extension_evidence(self, path: Path) -> Mapping[str, Any]:
        if path not in self._evidence_cache:
            self._check(path.exists(), f"source_extensions: missing evidence artifact {path}")
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                self._check(False, f"{path}: invalid JSON: {error}")
                loaded = {}
            self._check(isinstance(loaded, Mapping), f"{path}: expected a JSON object")
            self._evidence_cache[path] = loaded
        return self._evidence_cache[path]

    def _validate_source_extensions(self, v1_sources: Mapping[str, Any]) -> None:
        """Validate the beyond-v1 label registry, failing closed on every axis.

        This is a separate declared surface: the exact v1 schema/label coverage
        check above still runs untouched over ``sources``. What makes an entry
        here a real extension rather than drift is that a tracked manifest or a
        tracked evidence artifact independently records the same label and the
        same counts.
        """
        self._evidence_cache: dict[Path, Mapping[str, Any]] = {}
        self._required_extension_labels: dict[str, set[str]] = {}
        self._v1_sources = v1_sources
        self.source_extensions: dict[str, dict[str, dict[str, Any]]] = {}
        declared = self.spec.get("source_extensions") or {}
        self._check(isinstance(declared, Mapping), "source_extensions must be a mapping")

        for schema, entry in declared.items():
            where = f"source_extensions.{schema}"
            self._check(isinstance(entry, Mapping), f"{where}: expected a mapping")
            self._check(schema in v1_sources, f"{where}: {schema!r} is not an existing v1 schema")
            unknown_fields = sorted(set(entry) - {"provenance", "labels"})
            self._check(not unknown_fields, f"{where}: unknown fields: {', '.join(unknown_fields)}")

            provenance = entry.get("provenance")
            self._check(isinstance(provenance, Mapping), f"{where}.provenance: expected a mapping")
            kind = provenance.get("kind")
            self._check(
                kind in EXTENSION_PROVENANCE_KINDS,
                f"{where}.provenance: kind must be one of {list(EXTENSION_PROVENANCE_KINDS)}, got {kind!r}",
            )
            corroborated = self._extension_corroboration(str(schema), provenance, where)

            # Emptiness is left to the completeness guard below, which names the
            # labels that went missing rather than only that something did.
            labels = entry.get("labels")
            self._check(isinstance(labels, Mapping), f"{where}.labels: expected a mapping")
            labels = labels or {}
            declared_labels: dict[str, dict[str, Any]] = {}
            for label, spec in labels.items():
                declared_labels[str(label)] = self._validate_extension_label(
                    str(schema), str(label), spec, v1_sources, corroborated
                )

            # Equality, not containment: a later deletion would otherwise drop a
            # recovered qualified-gold label back to a silent drop while leaving
            # the registry valid, which is the exact failure this file exists to
            # end.
            required = self._required_extension_labels[str(schema)]
            missing = sorted(required - set(declared_labels))
            self._check(
                not missing,
                f"{where}: the evidence records {len(required)} recovered labels for this schema"
                f" but the registry omits {', '.join(missing)}; declare every recovered label or"
                " remove it from the evidence artifact",
            )
            self.source_extensions[str(schema)] = declared_labels

    def _extension_corroboration(
        self, schema: str, provenance: Mapping[str, Any], where: str
    ) -> Mapping[str, Mapping[str, Any]]:
        """Return the labels this entry must declare and the counts it must match."""
        declared_artifact = provenance.get("evidence_artifact", DEFAULT_EXTENSION_EVIDENCE)
        evidence_path = self._resolve_repo_path(declared_artifact, f"{where}.provenance")
        evidence = self._extension_evidence(evidence_path)
        recorded = (evidence.get("sources") or {}).get(schema)
        self._check(
            isinstance(recorded, Mapping),
            f"{where}: {evidence_path} records no evidence for {schema!r}",
        )
        recorded = dict(recorded or {})
        self._check(
            recorded.get("kind") == provenance.get("kind"),
            f"{where}.provenance: kind {provenance.get('kind')!r} disagrees with the evidence"
            f" artifact's {recorded.get('kind')!r}",
        )

        if provenance.get("kind") == "onboarding_manifest":
            declared_manifest = provenance.get("manifest")
            self._check(
                declared_manifest is not None,
                f"{where}.provenance: a manifest-backed entry must name its manifest",
            )
            manifest_path = self._resolve_repo_path(declared_manifest, f"{where}.provenance")
            self._check(manifest_path.exists(), f"{where}.provenance: missing {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._check(
                manifest.get("source_schema") == schema,
                f"{where}.provenance: {manifest_path} declares source_schema"
                f" {manifest.get('source_schema')!r}, not {schema!r}",
            )
            self._check(
                str(declared_manifest) == str(recorded.get("manifest")),
                f"{where}.provenance: manifest path disagrees with the evidence artifact",
            )
            ignored = {str(name) for name in manifest.get("ignored_source_labels") or ()}
            self._check(
                ignored == set(recorded.get("ignored_source_labels") or ()),
                f"{where}.provenance: the manifest's ignored_source_labels disagree with the"
                " evidence artifact",
            )
            required = ignored
        else:
            observed = {str(name) for name in recorded.get("observed_labels") or ()}
            self._check(
                bool(observed),
                f"{where}.provenance: a raw-scan entry needs an observed label inventory in the"
                " evidence artifact",
            )
            undeclared = recorded.get("undeclared_labels")
            self._check(
                isinstance(undeclared, list) and bool(undeclared),
                f"{where}.provenance: a raw-scan entry needs the evidence artifact's list of"
                " labels the v1 mapping never declared",
            )
            required = {str(name) for name in undeclared or ()}
            outside_scan = sorted(required - observed)
            self._check(
                not outside_scan,
                f"{where}.provenance: undeclared labels absent from the scan's observed"
                f" inventory: {', '.join(outside_scan)}",
            )
            already_v1 = sorted(required & set(self._v1_sources.get(schema, {})))
            self._check(
                not already_v1,
                f"{where}.provenance: {', '.join(already_v1)} is already mapped by v1, so it is"
                " not an undeclared label",
            )
        self._required_extension_labels[schema] = required

        counts = recorded.get("ignored_label_counts")
        self._check(
            isinstance(counts, Mapping) and bool(counts),
            f"{where}: the evidence artifact records no per-label counts for {schema!r}",
        )
        return dict(counts or {})

    def _validate_extension_label(
        self,
        schema: str,
        label: str,
        spec: Any,
        v1_sources: Mapping[str, Any],
        corroborated: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        where = f"source_extensions.{schema}.{label}"
        self._check(isinstance(spec, Mapping), f"{where}: expected a mapping")
        unknown_fields = sorted(
            set(spec) - {"fallback", "acceptable", "route", "promotion", "evidence", "reason"}
        )
        self._check(not unknown_fields, f"{where}: unknown fields: {', '.join(unknown_fields)}")
        self._check(
            label not in v1_sources[schema],
            f"{where}: {label!r} is already a v1 source label; declare it in sources, not source_extensions",
        )
        self._check(
            label in self._required_extension_labels.get(schema, set()),
            f"{where}: {label!r} is not recorded as a recovered label for {schema};"
            " an undeclared label here would be drift, not an extension",
        )

        fallback, acceptable = self._entry(where, spec)
        route, promotion = spec.get("route"), spec.get("promotion")
        self._check(
            route in EXTENSION_ROUTES,
            f"{where}: route must be one of {list(EXTENSION_ROUTES)}, got {route!r}",
        )
        self._check(
            promotion in EXTENSION_PROMOTIONS,
            f"{where}: promotion must be one of {list(EXTENSION_PROMOTIONS)}, got {promotion!r}",
        )
        reason = spec.get("reason")
        self._check(
            isinstance(reason, str) and reason.strip() != "",
            f"{where}: an extension needs a stated reason",
        )

        evidence = spec.get("evidence")
        self._check(isinstance(evidence, Mapping), f"{where}.evidence: expected a mapping")
        spans, by_split = evidence.get("spans"), evidence.get("by_split")
        self._check(_is_int(spans) and spans >= 0, f"{where}.evidence: spans must be a nonnegative integer")
        self._check(
            isinstance(by_split, Mapping) and bool(by_split),
            f"{where}.evidence: by_split must be a nonempty mapping",
        )
        for split, count in (by_split or {}).items():
            self._check(
                _is_int(count) and count >= 0,
                f"{where}.evidence.by_split.{split}: must be a nonnegative integer",
            )
        self._check(
            spans == sum((by_split or {}).values()),
            f"{where}.evidence: spans {spans!r} does not equal the sum of by_split",
        )

        recorded = corroborated.get(label)
        self._check(
            isinstance(recorded, Mapping),
            f"{where}: the evidence artifact records no counts for this label",
        )
        recorded = dict(recorded or {})
        self._check(
            recorded.get("spans") == spans and dict(recorded.get("by_split") or {}) == dict(by_split or {}),
            f"{where}.evidence: counts disagree with the evidence artifact"
            f" ({recorded.get('spans')!r} / {recorded.get('by_split')!r})",
        )
        return {
            "fallback": fallback,
            "acceptable": acceptable,
            "route": str(route),
            "promotion": str(promotion),
            "evidence": {"spans": int(spans), "by_split": dict(sorted((by_split or {}).items()))},
        }

    def _validate_composition_contracts(self) -> None:
        """Structure-only boundary transforms, recorded for the future relabeler."""
        declared = self.spec.get("composition_contracts") or {}
        self._check(isinstance(declared, Mapping), "composition_contracts must be a mapping")
        for name, entry in declared.items():
            where = f"composition_contracts.{name}"
            self._check(isinstance(entry, Mapping), f"{where}: expected a mapping")
            schema = entry.get("schema")
            self._check(schema in self.sources, f"{where}: unknown source schema {schema!r}")
            labels = entry.get("labels") or ([entry["label"]] if "label" in entry else [])
            labels = list(labels) + ([entry["coarse_label"]] if "coarse_label" in entry else [])
            self._check(bool(labels), f"{where}: a contract must name the labels it governs")
            for label in labels:
                self._check(
                    label in self.sources[schema] or label in self.source_extensions.get(schema, {}),
                    f"{where}: {schema}.{label} is neither a source label nor a declared extension",
                )
            for field in ("contract", "requires", "without_structure", "note"):
                value = entry.get(field)
                self._check(
                    isinstance(value, str) and value.strip() != "",
                    f"{where}: {field} must be a nonempty string",
                )
            self._check(
                entry.get("creates_new_primary_or_predicate") is False,
                f"{where}: a composition contract may never create a primary class or predicate",
            )
        self.composition_contracts = {str(name): dict(entry) for name, entry in declared.items()}

    def _validate_source_metadata_channels(self) -> None:
        """Source-side provenance channels, and where they may discriminate."""
        declared = self.spec.get("source_metadata_channels") or {}
        self._check(isinstance(declared, Mapping), "source_metadata_channels must be a mapping")
        for schema, channels in declared.items():
            self._check(schema in self.sources, f"source_metadata_channels: unknown schema {schema!r}")
            self._check(
                isinstance(channels, Mapping) and bool(channels),
                f"source_metadata_channels.{schema}: expected a nonempty mapping",
            )
            for channel, entry in channels.items():
                where = f"source_metadata_channels.{schema}.{channel}"
                self._check(isinstance(entry, Mapping), f"{where}: expected a mapping")
                self._check(
                    channel not in self.attr_channels,
                    f"{where}: a source metadata channel may not shadow a predicate channel",
                )
                self._check(
                    entry.get("creates_predicate_channel") is False,
                    f"{where}: source metadata never becomes a predicate channel",
                )
                values = entry.get("values")
                self._check(
                    isinstance(values, list) and bool(values) and len(set(values)) == len(values),
                    f"{where}: values must be a nonempty duplicate-free list",
                )
                note = entry.get("note")
                self._check(
                    isinstance(note, str) and note.strip() != "",
                    f"{where}: a metadata channel needs a stated note",
                )
                for label, rules in (entry.get("deterministic_discriminator") or {}).items():
                    self._check(
                        label in self.sources[schema],
                        f"{where}: discriminator label {schema}.{label} does not exist",
                    )
                    acceptable = set(self.sources[schema][label][1])
                    for value, target in (rules or {}).items():
                        self._check(
                            value in values,
                            f"{where}.{label}: {value!r} is not a declared channel value",
                        )
                        self._check(
                            target in acceptable,
                            f"{where}.{label}: {value!r} maps to {target!r}, which is outside the"
                            f" declared acceptable set for {schema}.{label}",
                        )
        self.source_metadata_channels = {
            str(schema): {str(channel): dict(entry) for channel, entry in channels.items()}
            for schema, channels in declared.items()
        }

    # ---- inventory -------------------------------------------------------

    def bioes_labels(self) -> list[str]:
        """O first, then B/I/E/S per primary class in sorted class order."""
        labels = [OUTSIDE]
        for primary_type in self.primary_types:
            labels.extend(f"{prefix}-{primary_type}" for prefix in BIOES_PREFIXES)
        return labels

    def label_ids(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.bioes_labels())}

    def family_of(self, primary_type: str) -> str:
        return self._primary_entry(primary_type)["family"]

    def action_of(self, primary_type: str) -> str:
        return self._primary_entry(primary_type)["action"]

    def _primary_entry(self, primary_type: str) -> dict[str, Any]:
        try:
            return self.primary[primary_type]
        except KeyError:
            raise OntologyError(f"unknown primary type {primary_type!r}") from None

    def channels_for(self, primary_type: str) -> tuple[str, ...]:
        """Attribute channels applicable to a primary type, in sorted order."""
        self._primary_entry(primary_type)
        return self._channels_by_type.get(primary_type, ())

    # ---- projection ------------------------------------------------------

    def canonical_fallback(self, node: str) -> str:
        return self._canonical(node)[0]

    def canonical_acceptable(self, node: str) -> tuple[str, ...]:
        return self._canonical(node)[1]

    def _canonical(self, node: str) -> tuple[str, tuple[str, ...]]:
        try:
            return self.old_v1[node]
        except KeyError:
            raise OntologyError(f"unknown canonical v1 node {node!r}") from None

    def source_fallback(self, schema: str, label: str) -> str:
        return self._source(schema, label)[0]

    def source_acceptable(self, schema: str, label: str) -> tuple[str, ...]:
        return self._source(schema, label)[1]

    def source_schemas(self) -> tuple[str, ...]:
        return tuple(self.sources)

    def source_labels(self, schema: str) -> tuple[str, ...]:
        """The v1 compatibility label surface: exactly the labels v1 declared."""
        try:
            return tuple(self.sources[schema])
        except KeyError:
            raise OntologyError(f"unknown source schema {schema!r}") from None

    def source_extension_labels(self, schema: str) -> tuple[str, ...]:
        """Labels declared beyond v1 for this schema, in sorted order."""
        if schema not in self.sources:
            raise OntologyError(f"unknown source schema {schema!r}")
        return tuple(sorted(self.source_extensions.get(schema, {})))

    def is_source_extension(self, schema: str, label: str) -> bool:
        return label in self.source_extensions.get(schema, {})

    def source_route(self, schema: str, label: str) -> str:
        """How a later relabeler may resolve this cell.

        A legacy v1 label has no declared route; it follows the ordinary
        fixed-span admission rule, reported here as ``constrained_discrimination``
        when it has alternatives and ``deterministic`` when it does not.
        """
        entry = self._extension(schema, label)
        if entry is not None:
            return entry["route"]
        _, acceptable = self._source(schema, label)
        return "deterministic" if len(acceptable) == 1 else "constrained_discrimination"

    def source_promotion(self, schema: str, label: str) -> str | None:
        """Declared promotion for a source extension, or None for a legacy label.

        A declared extension returns one of ``EXTENSION_PROMOTIONS``. A legacy v1
        label returns ``None``: a label mapping says what a span means, never
        which corpus stratum it came from. The 24 v1 schemas mix qualified human
        gold with publisher-synthetic, teacher-replay, authored-template and
        translated sources, so inferring training eligibility from schema
        membership would recreate exactly the gold/replay conflation this review
        was opened to correct. Callers must take legacy corpus eligibility from
        independent dataset/stratum provenance.
        """
        entry = self._extension(schema, label)
        if entry is not None:
            return entry["promotion"]
        self._source(schema, label)
        return None

    def source_extension_evidence(self, schema: str, label: str) -> dict[str, Any]:
        entry = self._extension(schema, label)
        if entry is None:
            raise OntologyError(f"{schema}.{label} is not a declared source extension")
        return dict(entry["evidence"])

    def _extension(self, schema: str, label: str) -> dict[str, Any] | None:
        if schema not in self.sources:
            raise OntologyError(f"unknown source schema {schema!r}")
        return self.source_extensions.get(schema, {}).get(label)

    def _source(self, schema: str, label: str) -> tuple[str, tuple[str, ...]]:
        mapping = self.sources.get(schema)
        if mapping is None:
            raise OntologyError(f"unknown source schema {schema!r}")
        entry = mapping.get(label)
        if entry is not None:
            return entry
        extension = self.source_extensions.get(schema, {}).get(label)
        if extension is not None:
            return extension["fallback"], extension["acceptable"]
        raise OntologyError(f"{schema}: unknown label {label!r}")

    # ---- gold records ----------------------------------------------------

    def validate_dataset_metadata(self, metadata: Any, *, where: str = "dataset metadata") -> None:
        """Require declared versions and exact attr_channels registry equality."""
        _require(isinstance(metadata, Mapping), f"{where}: expected a mapping")
        for field in DATASET_METADATA_FIELDS:
            _require(field in metadata, f"{where}: missing {field}")
        _require(
            metadata["ontology_version"] == ONTOLOGY_VERSION,
            f"{where}: ontology_version {metadata['ontology_version']!r} != {ONTOLOGY_VERSION!r}",
        )
        _require(
            metadata["schema_version"] == SCHEMA_VERSION,
            f"{where}: schema_version {metadata['schema_version']!r} != {SCHEMA_VERSION!r}",
        )
        declared = metadata["attr_channels"]
        _require(isinstance(declared, Mapping), f"{where}: attr_channels must be a mapping")
        undeclared = sorted(set(declared) - set(self.attr_channels))
        absent = sorted(set(self.attr_channels) - set(declared))
        _require(not undeclared, f"{where}: undeclared attribute channels: {', '.join(undeclared)}")
        _require(not absent, f"{where}: attribute channels not declared: {', '.join(absent)}")
        for channel, entry in declared.items():
            _require(isinstance(entry, Mapping), f"{where}: attr_channels.{channel} must be a mapping")
            expected = self.attr_channels[channel]
            _require(
                tuple(sorted(entry.get("applicable_types", ()))) == expected["applicable_types"],
                f"{where}: attr_channels.{channel} applicable_types drift",
            )
            _require(
                entry.get("semantics_version") == expected["semantics_version"],
                f"{where}: attr_channels.{channel} semantics_version drift",
            )
            _require(
                entry.get("value_kind") == expected["value_kind"],
                f"{where}: attr_channels.{channel} value_kind must be"
                f" {expected['value_kind']!r}, got {entry.get('value_kind')!r}",
            )

    def validate_record(self, row: Any, *, where: str = "record") -> dict[str, Any]:
        """Validate one gold record and return it unchanged.

        Unknown top-level fields are preserved: they are provenance, not an
        error. ``attrs`` stays optional so v1 readers can ignore it.
        """
        _require(isinstance(row, Mapping), f"{where}: expected a mapping")
        for field in ("id", "text", "spans"):
            _require(field in row, f"{where}: missing {field}")
        _require(isinstance(row["id"], str) and row["id"] != "", f"{where}: id must be a nonempty string")
        text = row["text"]
        _require(isinstance(text, str), f"{where}: text must be a string")
        spans = row["spans"]
        _require(isinstance(spans, list), f"{where}: spans must be a list")

        length = len(text)
        previous_key: tuple[int, int, str] | None = None
        previous_end = -1
        for index, span in enumerate(spans):
            start, end, primary_type = self._validate_span(span, length, where=f"{where}: span {index}")
            key = (start, end, primary_type)
            _require(
                previous_key is None or previous_key <= key,
                f"{where}: span {index}: unsorted spans at {start}",
            )
            if self.forbid_overlapping_spans:
                _require(start >= previous_end, f"{where}: span {index}: overlapping spans at {start}")
            previous_key = key
            previous_end = max(previous_end, end)
        return dict(row)

    def _validate_span(self, span: Any, text_length: int, *, where: str) -> tuple[int, int, str]:
        _require(isinstance(span, Mapping), f"{where}: expected a {{start, end, type, attrs?}} mapping")
        unknown = sorted(set(span) - {"start", "end", "type", "attrs"})
        _require(not unknown, f"{where}: unknown span fields: {', '.join(unknown)}")
        for field in ("start", "end", "type"):
            _require(field in span, f"{where}: missing {field}")
        start, end, primary_type = span["start"], span["end"], span["type"]
        _require(_is_int(start) and _is_int(end), f"{where}: offsets must be integers")
        _require(
            0 <= start < end <= text_length,
            f"{where}: invalid span [{start}, {end}) for text length {text_length}",
        )
        _require(
            primary_type in self.primary,
            f"{where}: unknown primary type {primary_type!r}",
        )
        if "attrs" in span:
            self._validate_attrs(span["attrs"], primary_type, start, end, where=where)
        return start, end, primary_type

    def _validate_attrs(
        self,
        attrs: Any,
        primary_type: str,
        span_start: int,
        span_end: int,
        *,
        where: str,
    ) -> None:
        _require(isinstance(attrs, Mapping), f"{where}: attrs must be a mapping")
        applicable = self.channels_for(primary_type)
        for channel, intervals in attrs.items():
            _require(channel in self.attr_channels, f"{where}: undeclared attribute channel {channel!r}")
            _require(
                channel in applicable,
                f"{where}: channel {channel!r} does not apply to {primary_type}",
            )
            _require(isinstance(intervals, list), f"{where}: attrs.{channel} must be a list of intervals")
            seen: list[tuple[int, int]] = []
            for interval in intervals:
                _require(
                    isinstance(interval, (list, tuple)) and len(interval) == 2,
                    f"{where}: attrs.{channel}: expected [start, end] intervals",
                )
                start, end = interval
                _require(
                    _is_int(start) and _is_int(end),
                    f"{where}: attrs.{channel}: interval offsets must be integers",
                )
                _require(start < end, f"{where}: attrs.{channel}: empty interval [{start}, {end})")
                _require(
                    span_start <= start and end <= span_end,
                    f"{where}: attrs.{channel}: interval [{start}, {end}) escapes the "
                    f"span [{span_start}, {span_end})",
                )
                for other_start, other_end in seen:
                    _require(
                        end <= other_start or start >= other_end,
                        f"{where}: attrs.{channel}: overlapping intervals at {start}",
                    )
                seen.append((start, end))

    def validate_jsonl(
        self,
        path: Path | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        require_metadata: bool = True,
    ) -> dict[str, Any]:
        """Validate a gold JSONL file and its dataset metadata sidecar."""
        path = Path(path)
        if metadata is None and require_metadata:
            metadata = load_dataset_metadata(path)
        if metadata is not None:
            self.validate_dataset_metadata(metadata, where=str(dataset_metadata_path(path)))

        records = 0
        spans = 0
        attributed_spans = 0
        seen_ids: set[str] = set()
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                where = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise OntologyError(f"{where}: invalid JSON: {error}") from None
                self.validate_record(row, where=where)
                _require(row["id"] not in seen_ids, f"{where}: duplicate record id {row['id']!r}")
                seen_ids.add(row["id"])
                records += 1
                spans += len(row["spans"])
                attributed_spans += sum(1 for span in row["spans"] if span.get("attrs"))
        _require(records > 0, f"{path}: empty gold corpus")
        return {
            "path": str(path),
            "records": records,
            "spans": spans,
            "attributed_spans": attributed_spans,
        }

    # ---- frozen remap admission -----------------------------------------

    def admit_remap(
        self,
        fallback: str,
        acceptable: Sequence[str],
        strengths: Mapping[str, float],
        *,
        threshold: float = REMAP_MARGIN_NATS,
    ) -> dict[str, Any]:
        """Apply the frozen fixed-span admission rule to candidate strengths.

        ``strengths`` are mean-over-included-tokens candidate strengths, one
        per acceptable target. Every acceptable candidate must be a primary
        class or O, listed once. The argmax is restricted to ``acceptable``;
        changing away from ``fallback`` needs a margin of at least ln(3)
        mean-logit units; ties retain the fallback; a singleton acceptable set
        is deterministic rather than model-supported. The result is a
        proposal, never an automatic promotion.

        ``threshold`` exists only so a caller can state the frozen value
        explicitly. Any other value is rejected: the margin is part of the
        frozen rule, not a caller's tuning knob.
        """
        _require(bool(acceptable), "admit_remap: acceptable must be nonempty")
        unknown = sorted(set(acceptable) - self._targets)
        _require(
            not unknown,
            f"admit_remap: acceptable targets outside primary+O: {', '.join(unknown)}",
        )
        _require(len(set(acceptable)) == len(acceptable), "admit_remap: duplicate acceptable targets")
        _require(fallback in acceptable, f"admit_remap: fallback {fallback!r} is not acceptable")
        _require(
            threshold == REMAP_MARGIN_NATS,
            f"admit_remap: the ln(3) margin is frozen at {REMAP_MARGIN_NATS!r};"
            f" {threshold!r} cannot override it",
        )
        record: dict[str, Any] = {
            "fallback": fallback,
            "acceptable": list(acceptable),
            "threshold": threshold,
            "status": self.remap_admission["status"],
        }
        if len(acceptable) == 1:
            return {
                **record,
                "label": fallback,
                "winner": fallback,
                "runner_up": None,
                "winner_strength": None,
                "runner_up_strength": None,
                "margin": None,
                "changed": False,
                "decision": "singleton_acceptable_set",
            }
        missing = sorted(set(acceptable) - set(strengths))
        extra = sorted(set(strengths) - set(acceptable))
        _require(not missing, f"admit_remap: no strength for {', '.join(missing)}")
        _require(not extra, f"admit_remap: strengths outside the acceptable set: {', '.join(extra)}")
        for candidate, strength in strengths.items():
            _require(
                isinstance(strength, (int, float))
                and not isinstance(strength, bool)
                and math.isfinite(strength),
                f"admit_remap: strength for {candidate!r} must be a finite number",
            )
        # Ties retain the fallback, then resolve by name so the rule is deterministic.
        ranked = sorted(
            acceptable, key=lambda candidate: (-strengths[candidate], candidate != fallback, candidate)
        )
        winner, runner_up = ranked[0], ranked[1]
        margin = strengths[winner] - strengths[runner_up]
        changed = winner != fallback and margin >= threshold
        return {
            **record,
            "label": winner if changed else fallback,
            "winner": winner,
            "runner_up": runner_up,
            "winner_strength": strengths[winner],
            "runner_up_strength": strengths[runner_up],
            "margin": margin,
            "changed": changed,
            "decision": (
                "retained_fallback"
                if winner == fallback
                else "admitted_change"
                if changed
                else "margin_below_threshold"
            ),
        }


def dataset_metadata_path(path: Path | str) -> Path:
    """Sidecar holding a gold file's ontology/schema declaration."""
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def load_dataset_metadata(path: Path | str) -> dict[str, Any]:
    """Read the ontology declaration for a gold file.

    The sidecar may nest it under ``ontology``; otherwise the sidecar object
    is the declaration itself.
    """
    sidecar = dataset_metadata_path(path)
    _require(sidecar.exists(), f"{sidecar}: missing dataset metadata sidecar")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    _require(isinstance(metadata, Mapping), f"{sidecar}: expected a JSON object")
    nested = metadata.get("ontology")
    return dict(nested) if isinstance(nested, Mapping) else dict(metadata)


def validate_spec(
    spec: Mapping[str, Any],
    *,
    compatibility_tagset: Mapping[str, Any] | None = None,
    source: str = str(ONTOLOGY_PATH),
) -> OntologyV2:
    """Validate an already-parsed v2 spec, raising on the first violation."""
    return OntologyV2(spec, compatibility_tagset=compatibility_tagset, source=source)


@lru_cache(maxsize=None)
def load_ontology(path: Path | str = ONTOLOGY_PATH) -> OntologyV2:
    """Load and validate the committed v2 spec (cached per path)."""
    path = Path(path)
    spec = load_yaml(path)
    if not isinstance(spec, Mapping):
        raise OntologyError(f"{path}: expected a YAML object")
    return OntologyV2(spec, source=str(path))


def primary_types() -> tuple[str, ...]:
    return load_ontology().primary_types


@lru_cache(maxsize=None)
def load_gold_schema(path: Path | str = GOLD_SCHEMA_PATH) -> dict[str, Any]:
    """Load the JSON Schema for one gold record (cached per path)."""
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(schema, dict), f"{path}: expected a JSON object")
    return schema


def validate_gold_schema(
    schema: Mapping[str, Any] | None = None,
    ontology: OntologyV2 | None = None,
    *,
    where: str = str(GOLD_SCHEMA_PATH),
) -> None:
    """Require the record schema and the ontology to agree on the inventory."""
    schema = load_gold_schema() if schema is None else schema
    ontology = load_ontology() if ontology is None else ontology
    _require(schema.get("x-ontology-version") == ONTOLOGY_VERSION, f"{where}: ontology_version drift")
    _require(schema.get("x-schema-version") == SCHEMA_VERSION, f"{where}: schema_version drift")
    definitions = schema["$defs"]
    _require(
        tuple(definitions["primary_type"]["enum"]) == ontology.primary_types,
        f"{where}: primary_type enum does not match the adopted inventory",
    )
    _require(
        tuple(sorted(definitions["attrs"]["properties"])) == tuple(sorted(ontology.attr_channels)),
        f"{where}: attrs channels do not match the attribute registry",
    )
    _require(
        definitions["attrs"].get("additionalProperties") is False,
        f"{where}: attrs must reject undeclared channels",
    )
    _require(
        definitions["span"].get("additionalProperties") is False,
        f"{where}: span must reject unknown fields",
    )
    _require(
        definitions["record"].get("additionalProperties") is not False,
        f"{where}: unknown top-level row fields must be preserved",
    )
    metadata = definitions["dataset_metadata"]
    _require(
        tuple(metadata["required"]) == DATASET_METADATA_FIELDS,
        f"{where}: dataset metadata must require {list(DATASET_METADATA_FIELDS)}",
    )


def bioes_labels() -> list[str]:
    return load_ontology().bioes_labels()


def validate_record(row: Any, *, where: str = "record") -> dict[str, Any]:
    return load_ontology().validate_record(row, where=where)


def validate_dataset_metadata(metadata: Any, *, where: str = "dataset metadata") -> None:
    load_ontology().validate_dataset_metadata(metadata, where=where)


def validate_jsonl(
    path: Path | str,
    *,
    metadata: Mapping[str, Any] | None = None,
    require_metadata: bool = True,
) -> dict[str, Any]:
    return load_ontology().validate_jsonl(path, metadata=metadata, require_metadata=require_metadata)


def _self_test(ontology: OntologyV2) -> int:
    checks: list[tuple[Any, Any]] = [
        (len(ontology.primary_types), PRIMARY_COUNT),
        (len(ontology.bioes_labels()), 1 + 4 * PRIMARY_COUNT),
        (ontology.bioes_labels()[0], OUTSIDE),
        (len(ontology.old_v1), CANONICAL_V1_NODE_COUNT),
        (ontology.canonical_fallback("salary"), "monetary_amount"),
        (ontology.canonical_fallback("quantity"), "quantity"),
        (ontology.canonical_fallback("contact"), OUTSIDE),
        (ontology.canonical_fallback("name_prefix"), "person_name"),
        (ontology.canonical_fallback("currency_designator"), OUTSIDE),
        (set(ontology.canonical_acceptable("currency_designator")), {"monetary_amount", OUTSIDE}),
        (ontology.canonical_fallback("biometric_identifier"), "record_identifier"),
        # Proposal mappings the parent review corrected.
        (ontology.canonical_fallback("county"), "locality"),
        (ontology.canonical_fallback("airport_code"), "locality"),
        (ontology.canonical_fallback("timezone"), "admin_area"),
        (set(ontology.canonical_acceptable("region")), {"admin_area", "locality"}),
        (
            set(ontology.canonical_acceptable("financial")),
            {
                "bank_account_number",
                "bank_routing_code",
                "monetary_amount",
                "organization",
                "payment_card_data",
                OUTSIDE,
            },
        ),
        (
            ontology.source_acceptable("openai_8", "account_number"),
            ("bank_account_number", "government_id", "payment_card_data", "record_identifier"),
        ),
        (
            set(ontology.source_acceptable("spy_7", "ID_NUM")),
            {"bank_account_number", "government_id", "payment_card_data", "record_identifier"},
        ),
        (ontology.source_acceptable("openmed_54", "COUNTY"), ("locality",)),
        (set(ontology.actions), {"GENERALIZE", "KEEP", "MASK", "PSEUDONYM", "SUPPRESS"}),
        (ontology.attr_channels["family_name"]["value_kind"], ATTR_VALUE_KIND),
        (ontology.attr_channels["care_provider"]["value_kind"], ATTR_VALUE_KIND),
        (ontology.source_acceptable("wojood_nested", "MONEY"), ("monetary_amount",)),
        (ontology.source_acceptable("ai4privacy_new", "SALARY"), ("monetary_amount",)),
        # Qualified-gold delta: overrides that change the acceptable set only.
        (
            ontology.source_acceptable("tab_8", "CODE"),
            ("government_id", "phone_number", "record_identifier"),
        ),
        (ontology.source_fallback("tab_8", "CODE"), "record_identifier"),
        (set(ontology.source_acceptable("tab_8", "QUANTITY")), {"quantity", "monetary_amount"}),
        (set(ontology.source_acceptable("mapa_coarse", "AMOUNT")), {"quantity", "monetary_amount"}),
        (OUTSIDE in ontology.source_acceptable("tab_8", "DATETIME"), True),
        # ... and the four that also move a fallback, for one source cell only.
        (ontology.source_fallback("tab_8", "MISC"), OUTSIDE),
        (ontology.canonical_fallback("misc_identifier"), "record_identifier"),
        (ontology.source_fallback("meddocan_phi", "FAMILIARES_SUJETO_ASISTENCIA"), "demographic_attribute"),
        (ontology.canonical_fallback("relative_name"), "person_name"),
        (ontology.source_fallback("meddocan_phi", "TERRITORIO"), "locality"),
        (ontology.canonical_fallback("location"), "location"),
        (ontology.source_fallback("multigrascco", "HEALTH_FCLT"), "demographic_attribute"),
        (ontology.canonical_fallback("healthcare_org"), "organization"),
        (ontology.source_fallback("multigrascco", "DIRECT_ID_ADDRESS"), "demographic_attribute"),
        (ontology.canonical_fallback("address"), "street_address"),
        # Source extensions are a separate surface and never touch v1 coverage.
        (
            ontology.source_labels("hiner_original"),
            ("LANGUAGE", "LOCATION", "ORGANIZATION", "PERSON", "RELIGION"),
        ),
        (
            ontology.source_extension_labels("hiner_original"),
            ("FESTIVAL", "GAME", "LITERATURE", "MISC", "NUMEX", "TIMEX"),
        ),
        (ontology.source_extension_labels("klue_ner"), ("DT", "QT", "TI")),
        (len(ontology.source_extension_labels("wojood_nested")), 10),
        (ontology.source_extension_labels("multigrascco"), ("RELATIVE_TIME",)),
        (ontology.source_fallback("hiner_original", "NUMEX"), OUTSIDE),
        (ontology.source_route("hiner_original", "TIMEX"), "constrained_discrimination"),
        (ontology.source_route("wojood_nested", "CURR"), "composition"),
        (ontology.source_promotion("multigrascco", "RELATIVE_TIME"), "evaluation_only"),
        (ontology.source_promotion("hiner_original", "PERSON"), None),
        (ontology.source_promotion("nemotron_pii", "date"), None),
        (ontology.is_source_extension("hiner_original", "PERSON"), False),
        (ontology.source_extension_evidence("hiner_original", "NUMEX")["spans"], 24289),
        (ontology.source_extension_evidence("klue_ner", "QT")["spans"], 14868),
        (
            sum(
                ontology.source_extension_evidence("wojood_nested", label)["spans"]
                for label in ontology.source_extension_labels("wojood_nested")
            ),
            775,
        ),
        # Nothing recovered here is promotable in this contract.
        (
            {
                ontology.source_promotion(schema, label)
                for schema in ("hiner_original", "klue_ner", "wojood_nested")
                for label in ontology.source_extension_labels(schema)
            },
            {"withheld"},
        ),
        (ontology.channels_for("person_name"), ATTR_CHANNELS),
        (ontology.channels_for("email"), ()),
        (ontology.remap_admission["margin_threshold_nats"], REMAP_MARGIN_NATS),
        # Source metadata stays provenance, never a third predicate channel.
        (
            ontology.source_metadata_channels["tab_8"]["confidential_status"]["deterministic_discriminator"][
                "DEM"
            ]["HEALTH"],
            "health_condition",
        ),
        (tuple(sorted(ontology.attr_channels)), ATTR_CHANNELS),
    ]
    for got, want in checks:
        assert got == want, f"got {got!r}, want {want!r}"

    admission = ontology.admit_remap(
        "record_identifier",
        ontology.canonical_acceptable("misc_identifier"),
        {
            target: (2.0 if target == "government_id" else 0.0)
            for target in ontology.canonical_acceptable("misc_identifier")
        },
    )
    assert admission["label"] == "government_id" and admission["changed"], admission
    narrow = ontology.admit_remap(
        "record_identifier",
        ontology.canonical_acceptable("misc_identifier"),
        {
            target: (1.0 if target == "government_id" else 0.0)
            for target in ontology.canonical_acceptable("misc_identifier")
        },
    )
    assert narrow["label"] == "record_identifier" and not narrow["changed"], narrow

    # The hardened guards fire before the singleton shortcut.
    for bad_call in (
        lambda: ontology.admit_remap("locality", ["locality", "not_a_class"], {"locality": 0.0}),
        lambda: ontology.admit_remap("not_a_class", ["not_a_class"], {}),
        lambda: ontology.admit_remap("locality", ["locality", "locality"], {"locality": 0.0}),
        lambda: ontology.admit_remap(
            "record_identifier",
            ontology.canonical_acceptable("misc_identifier"),
            dict.fromkeys(ontology.canonical_acceptable("misc_identifier"), 0.0),
            threshold=0.0,
        ),
    ):
        try:
            bad_call()
        except OntologyError:
            continue
        raise AssertionError("admit_remap accepted an input its contract forbids")

    record = {
        "id": "self-test-1",
        "text": "Dr. Ada Lovelace treated Alan.",
        "spans": [
            {
                "start": 0,
                "end": 17,
                "type": "person_name",
                "attrs": {"family_name": [[8, 17]], "care_provider": [[0, 17]]},
            },
            {"start": 25, "end": 29, "type": "person_name", "attrs": {"care_provider": []}},
        ],
        "provenance": "self-test",
    }
    validated = ontology.validate_record(record)
    assert validated["provenance"] == "self-test", validated
    validate_gold_schema(ontology=ontology)
    return len(checks) + 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=ONTOLOGY_PATH)
    parser.add_argument("--jsonl", type=Path, action="append", default=[], help="gold file to validate")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="validate rows without requiring a dataset metadata sidecar",
    )
    parser.add_argument("--report", type=Path, help="write a JSON validation report")
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    label_count = len(ontology.bioes_labels())
    source_labels = sum(len(mapping) for mapping in ontology.sources.values())
    extensions = sum(len(mapping) for mapping in ontology.source_extensions.values())
    print(
        f"ontology v2 OK: {len(ontology.primary_types)} primary classes, {label_count} BIOES labels, "
        f"{len(ontology.old_v1)} canonical v1 nodes, {len(ontology.sources)} source schemas, "
        f"{source_labels} source labels, {extensions} declared source extensions in "
        f"{len(ontology.source_extensions)} schemas"
    )
    passed = _self_test(ontology)
    print(f"{passed} self-tests passed")

    files = [ontology.validate_jsonl(path, require_metadata=not args.no_metadata) for path in args.jsonl]
    for summary in files:
        print(
            f"{summary['path']}: {summary['records']} records, {summary['spans']} spans, "
            f"{summary['attributed_spans']} with attributes"
        )
    if args.report:
        report = {
            "schema": SCHEMA_VERSION,
            "ontology": {
                "path": str(args.ontology),
                "ontology_version": ONTOLOGY_VERSION,
                "projection_version": ontology.projection_version,
                "primary_classes": list(ontology.primary_types),
                "bioes_labels": label_count,
            },
            "files": files,
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
