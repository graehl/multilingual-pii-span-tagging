"""Focused contract tests for the PII target ontology v2."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.pii_ontology_v2 import (
    ATTR_CHANNELS,
    CANONICAL_V1_NODE_COUNT,
    COMPATIBILITY_TAGSET_PATH,
    EXTENSION_PROMOTIONS,
    EXTENSION_ROUTES,
    ONTOLOGY_PATH,
    ONTOLOGY_VERSION,
    OUTSIDE,
    PRIMARY_COUNT,
    PROJECTION_VERSION,
    REMAP_ADMISSION_VERSION,
    REMAP_MARGIN_NATS,
    SCHEMA_VERSION,
    OntologyError,
    load_gold_schema,
    load_ontology,
    load_yaml,
    parse_yaml,
    validate_gold_schema,
    validate_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

ADOPTED_PRIMARY_INVENTORY = (
    "admin_area",
    "age",
    "bank_account_number",
    "bank_routing_code",
    "credential",
    "date",
    "date_of_birth",
    "demographic_attribute",
    "device_identifier",
    "email",
    "government_id",
    "gps_coordinates",
    "health_condition",
    "ip_address",
    "locality",
    "location",
    "monetary_amount",
    "organization",
    "payment_card_data",
    "person_name",
    "phone_number",
    "postal_code",
    "protected_attribute",
    "quantity",
    "record_identifier",
    "street_address",
    "time",
    "url",
    "username",
)

VALID_METADATA = {
    "ontology_version": ONTOLOGY_VERSION,
    "schema_version": SCHEMA_VERSION,
    "attr_channels": {
        "family_name": {
            "applicable_types": ["person_name"],
            "semantics_version": 1,
            "value_kind": "character_extents",
        },
        "care_provider": {
            "applicable_types": ["person_name"],
            "semantics_version": 1,
            "value_kind": "character_extents",
        },
    },
}


@pytest.fixture(scope="module")
def ontology():
    return load_ontology()


@pytest.fixture(scope="module")
def v1_tagset():
    return yaml.safe_load(COMPATIBILITY_TAGSET_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def spec():
    return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))


def ontology_canonical(spec, node):
    """The canonical acceptable set a spec declares for a v1 node."""
    return spec["old_v1"][node]["acceptable"]


def record(**overrides):
    row = {
        "id": "row-1",
        "text": "Dr. Ada Lovelace met Alan Turing.",
        "spans": [
            {"start": 0, "end": 16, "type": "person_name"},
            {"start": 21, "end": 32, "type": "person_name"},
        ],
    }
    row.update(overrides)
    return row


# ---- inventory -----------------------------------------------------------


def test_primary_inventory_is_exactly_the_adopted_29(ontology):
    assert ontology.primary_types == ADOPTED_PRIMARY_INVENTORY
    assert len(ontology.primary_types) == PRIMARY_COUNT == 29
    assert OUTSIDE not in ontology.primary


def test_quantity_and_monetary_amount_are_both_primary(ontology):
    assert "quantity" in ontology.primary
    assert "monetary_amount" in ontology.primary
    assert ontology.family_of("monetary_amount") == "financial"
    assert ontology.action_of("monetary_amount") == "GENERALIZE"
    assert ontology.family_of("quantity") == "other"
    assert ontology.action_of("quantity") == "GENERALIZE"
    assert "non-currency" in ontology.primary["quantity"]["definition"]
    assert "cryptocurrency" in ontology.primary["monetary_amount"]["definition"]


def test_adoption_note_records_the_28_to_29_difference(ontology):
    note = ontology.adoption_note
    assert note["proposal_class_count"] == 28
    assert note["adopted_class_count"] == PRIMARY_COUNT == 29
    assert note["proposal_merged_money_into_quantity"] is True
    assert note["difference"] == "q6_separates_monetary_amount_from_quantity"
    assert note["q6_currency_designator_alone"] == OUTSIDE


@pytest.mark.parametrize(
    "field, value",
    [
        ("proposal_class_count", 29),
        ("adopted_class_count", 28),
        ("proposal_merged_money_into_quantity", False),
        ("difference", "something_else"),
    ],
)
def test_adoption_note_drift_is_rejected(spec, field, value):
    spec["adoption_note"][field] = value
    with pytest.raises(OntologyError, match="adoption_note"):
        validate_spec(spec)


def test_actions_are_the_proposals_five_and_surrogate_is_gone(ontology):
    assert set(ontology.actions) == {"SUPPRESS", "PSEUDONYM", "MASK", "GENERALIZE", "KEEP"}
    assert "SURROGATE" not in ontology.actions
    assert "SURROGATE" not in {ontology.action_of(name) for name in ontology.primary_types}


def test_families_are_the_proposals_twelve(ontology):
    assert set(ontology.families) == {
        "person",
        "attribute",
        "organization",
        "location",
        "contact",
        "digital",
        "authority_id",
        "org_id",
        "financial",
        "other",
        "credential",
        "date_time",
    }


@pytest.mark.parametrize(
    "primary_type, family, action",
    [
        ("admin_area", "location", "KEEP"),
        ("bank_account_number", "financial", "MASK"),
        ("bank_routing_code", "financial", "KEEP"),
        ("device_identifier", "digital", "SUPPRESS"),
        ("government_id", "authority_id", "MASK"),
        ("health_condition", "attribute", "SUPPRESS"),
        ("ip_address", "digital", "GENERALIZE"),
        ("location", "location", "PSEUDONYM"),
        ("payment_card_data", "financial", "MASK"),
        ("person_name", "person", "PSEUDONYM"),
        ("phone_number", "contact", "MASK"),
        ("protected_attribute", "attribute", "SUPPRESS"),
        ("record_identifier", "org_id", "PSEUDONYM"),
        ("time", "date_time", "KEEP"),
    ],
)
def test_proposal_family_and_action_assignments(ontology, primary_type, family, action):
    assert ontology.family_of(primary_type) == family
    assert ontology.action_of(primary_type) == action


@pytest.mark.parametrize(
    "primary_type, prongs",
    [
        ("person_name", ["R", "A"]),
        ("date_of_birth", ["R", "A"]),
        ("username", ["R"]),
        ("device_identifier", ["R"]),
        ("monetary_amount", ["A"]),
        ("quantity", ["A"]),
        ("admin_area", ["A"]),
    ],
)
def test_admission_prongs_are_recorded(ontology, primary_type, prongs):
    assert ontology.primary[primary_type]["prongs"] == prongs


def test_unknown_admission_prong_is_rejected(spec):
    spec["primary"]["person_name"]["prongs"] = ["R", "Z"]
    with pytest.raises(OntologyError, match="unknown admission prongs: Z"):
        validate_spec(spec)


def test_an_unused_family_is_rejected(spec):
    spec["families"]["ghost"] = "a family no class uses"
    with pytest.raises(OntologyError, match="families no primary class uses: ghost"):
        validate_spec(spec)


def test_an_unused_action_is_rejected(spec):
    spec["actions"]["SURROGATE"] = "a reintroduced action no class uses"
    with pytest.raises(OntologyError, match="actions no primary class uses: SURROGATE"):
        validate_spec(spec)


def test_bioes_labels_are_o_plus_four_per_class(ontology):
    labels = ontology.bioes_labels()
    assert labels[0] == OUTSIDE
    assert len(labels) == 1 + 4 * PRIMARY_COUNT == 117
    assert len(set(labels)) == len(labels)
    assert labels[1:5] == ["B-admin_area", "I-admin_area", "E-admin_area", "S-admin_area"]
    assert ontology.label_ids()["O"] == 0


def test_family_and_action_metadata_are_declared(ontology):
    for primary_type in ontology.primary_types:
        assert ontology.family_of(primary_type) in ontology.families
        assert ontology.action_of(primary_type) in ontology.actions
    with pytest.raises(OntologyError, match="unknown primary type"):
        ontology.family_of("family_name")


# ---- v1 coverage ---------------------------------------------------------


def test_all_116_canonical_v1_nodes_are_covered_exactly(ontology, v1_tagset):
    assert set(ontology.old_v1) == set(v1_tagset["nodes"])
    assert len(ontology.old_v1) == CANONICAL_V1_NODE_COUNT == 116


def test_every_current_source_label_is_covered_exactly(ontology, v1_tagset):
    assert set(ontology.sources) == set(v1_tagset["sources"])
    for schema, mapping in v1_tagset["sources"].items():
        assert set(ontology.source_labels(schema)) == set(mapping), schema
    assert sum(len(mapping) for mapping in ontology.sources.values()) == 711


def test_missing_canonical_node_is_rejected(spec):
    spec["old_v1"].pop("person_name")
    with pytest.raises(OntologyError, match="expected 116 canonical v1 nodes"):
        validate_spec(spec)


def test_extra_source_label_is_rejected(spec):
    spec["sources"]["tab_8"]["NOT_A_LABEL"] = {"fallback": "O", "acceptable": ["O"]}
    with pytest.raises(OntologyError, match="sources.tab_8: unknown labels: NOT_A_LABEL"):
        validate_spec(spec)


def test_missing_source_label_is_rejected(spec):
    spec["sources"]["tab_8"].pop("CODE")
    with pytest.raises(OntologyError, match="sources.tab_8: missing labels: CODE"):
        validate_spec(spec)


# ---- acceptable sets and fallbacks ---------------------------------------


def test_every_acceptable_set_is_nonempty_and_within_primary_plus_outside(ontology):
    targets = set(ontology.primary_types) | {OUTSIDE}
    entries = list(ontology.old_v1.items())
    entries += [
        (f"{schema}.{label}", entry)
        for schema, mapping in ontology.sources.items()
        for label, entry in mapping.items()
    ]
    for where, (fallback, acceptable) in entries:
        assert acceptable, where
        assert set(acceptable) <= targets, where
        assert fallback in acceptable, where


def test_acceptable_target_outside_the_inventory_is_rejected(spec):
    spec["old_v1"]["person_name"]["acceptable"] = ["person_name", "family_name"]
    with pytest.raises(OntologyError, match="acceptable targets outside primary\\+O: family_name"):
        validate_spec(spec)


def test_fallback_outside_its_acceptable_set_is_rejected(spec):
    spec["sources"]["tab_8"]["ORG"]["fallback"] = "government_id"
    with pytest.raises(OntologyError, match="fallback 'government_id' is not in its acceptable set"):
        validate_spec(spec)


def test_unknown_family_is_rejected(spec):
    spec["primary"]["age"]["family"] = "not_a_family"
    with pytest.raises(OntologyError, match="primary age: unknown family"):
        validate_spec(spec)


def test_unknown_action_is_rejected(spec):
    spec["primary"]["age"]["action"] = "REDACT"
    with pytest.raises(OntologyError, match="primary age: unknown action"):
        validate_spec(spec)


def test_primary_count_drift_is_rejected(spec):
    spec["primary"].pop("quantity")
    with pytest.raises(OntologyError, match="expected 29 primary classes, got 28"):
        validate_spec(spec)


# ---- Q6 monetary/currency and other frozen projections -------------------


def test_money_labels_project_to_monetary_amount(ontology):
    assert ontology.canonical_fallback("monetary_amount") == "monetary_amount"
    assert ontology.canonical_fallback("salary") == "monetary_amount"
    assert ontology.source_acceptable("wojood_nested", "MONEY") == ("monetary_amount",)
    assert ontology.source_acceptable("ai4privacy_new", "SALARY") == ("monetary_amount",)
    assert ontology.source_fallback("authored_v1", "monetary_amount") == "monetary_amount"


def test_bare_currency_designator_is_outside_with_a_monetary_alternative(ontology):
    assert ontology.canonical_fallback("currency_designator") == OUTSIDE
    assert set(ontology.canonical_acceptable("currency_designator")) == {"monetary_amount", OUTSIDE}
    assert ontology.source_fallback("openmed_54", "CURRENCYCODE") == OUTSIDE
    assert set(ontology.source_acceptable("openmed_54", "CURRENCYCODE")) == {
        "monetary_amount",
        OUTSIDE,
    }


def test_quantity_is_kept_and_stays_ambiguous_with_money(ontology):
    assert ontology.canonical_fallback("quantity") == "quantity"
    assert set(ontology.canonical_acceptable("quantity")) == {
        "quantity",
        "monetary_amount",
        "postal_code",
        "age",
    }


def test_other_superseding_projections(ontology):
    assert ontology.canonical_fallback("contact") == OUTSIDE
    assert set(ontology.canonical_acceptable("contact")) == {"email", "phone_number", "url", OUTSIDE}
    assert ontology.canonical_fallback("name_prefix") == "person_name"
    assert set(ontology.canonical_acceptable("name_prefix")) == {"person_name", OUTSIDE}
    assert ontology.canonical_fallback("biometric_identifier") == "record_identifier"
    assert ontology.canonical_fallback("clinician_name") == "person_name"
    assert ontology.canonical_fallback("family_name") == "person_name"
    assert ontology.canonical_fallback("city") == "locality"
    assert ontology.canonical_fallback("location") == "location"


def test_payment_card_brands_project_to_organization(ontology):
    assert ontology.projection_version == PROJECTION_VERSION
    assert ontology.canonical_fallback("payment_card") == "organization"
    assert set(ontology.canonical_acceptable("payment_card")) == {
        "organization",
        "payment_card_data",
    }
    assert ontology.source_fallback("fastino_42", "payment_card") == "payment_card_data"
    assert set(ontology.source_acceptable("fastino_42", "payment_card")) == {
        "payment_card_data",
    }
    for schema in ("openmed_54", "ai4privacy_200k"):
        assert ontology.source_fallback(schema, "CREDITCARDISSUER") == "organization"
        assert ontology.source_acceptable(schema, "CREDITCARDISSUER") == ("organization",)

    assert ontology.canonical_fallback("card_number") == "payment_card_data"
    assert ontology.source_fallback("openmed_54", "CREDITCARD") == "payment_card_data"
    assert ontology.source_fallback("ai4privacy_200k", "CREDITCARDNUMBER") == "payment_card_data"


def test_proposal_mapping_for_county_airport_code_and_timezone(ontology):
    # A county is a sub-state area and an airport code stands for a settlement;
    # a time zone stands for an administrative area at or above state level.
    assert ontology.canonical_acceptable("county") == ("locality",)
    assert ontology.canonical_acceptable("airport_code") == ("locality",)
    assert ontology.canonical_acceptable("timezone") == ("admin_area",)
    for schema, label in (
        ("openmed_54", "COUNTY"),
        ("ai4privacy_200k", "COUNTY"),
        ("authored_v1", "county"),
        ("nemotron_pii", "county"),
    ):
        assert ontology.source_acceptable(schema, label) == ("locality",), (schema, label)
    assert ontology.source_acceptable("ai4privacy_new", "AIRPORTCODE") == ("locality",)
    assert ontology.source_acceptable("ai4privacy_new", "TIMEZONE") == ("admin_area",)


def test_region_gains_locality_from_its_county_descendant(ontology):
    assert ontology.canonical_fallback("region") == "admin_area"
    assert set(ontology.canonical_acceptable("region")) == {"admin_area", "locality"}
    assert set(ontology.source_acceptable("fastino_42", "state_or_region")) == {
        "admin_area",
        "locality",
    }
    # location's own set is unchanged: it already covered both rungs.
    assert set(ontology.canonical_acceptable("location")) == {
        "admin_area",
        "gps_coordinates",
        "locality",
        "location",
        "postal_code",
        "street_address",
    }


def test_no_ancestor_images_are_added_merely_for_being_ancestors(ontology):
    # given_name's parent projects to person_name too, but nothing coarser leaks in.
    assert ontology.canonical_acceptable("given_name") == ("person_name",)
    assert ontology.canonical_acceptable("iban") == ("bank_account_number",)


# ---- source-specific overrides -------------------------------------------


@pytest.mark.parametrize(
    "schema,label",
    [
        ("authored_v1", "order_number"),
        ("authored_v1", "reservation_number"),
        ("multigrascco", "ID"),
    ],
)
def test_narrowed_identifier_labels(ontology, schema, label):
    assert ontology.source_acceptable(schema, label) == ("record_identifier",)
    assert ontology.source_fallback(schema, label) == "record_identifier"


@pytest.mark.parametrize("schema,label", [("tab_8", "QUANTITY"), ("mapa_coarse", "AMOUNT")])
def test_generic_amount_labels_stay_ambiguous(ontology, schema, label):
    assert set(ontology.source_acceptable(schema, label)) == {"quantity", "monetary_amount"}


def test_broad_sets_are_preserved_for_deliberately_coarse_labels(ontology):
    misc = set(ontology.canonical_acceptable("misc_identifier"))
    assert {"record_identifier", "government_id", "username", "device_identifier"} <= misc
    other = set(ontology.source_acceptable("multigrascco", "OTHER"))
    assert len(other) > 1


def test_source_label_inherits_its_node_set_when_not_overridden(ontology):
    assert ontology.source_acceptable("openmed_54", "FIRSTNAME") == ontology.canonical_acceptable(
        "given_name"
    )


def test_canonical_financial_keeps_its_outside_candidate(ontology):
    # currency_designator is a financial descendant that projects to O, so O is
    # mechanical here rather than curated.
    assert ontology.canonical_fallback("financial") == "bank_account_number"
    assert set(ontology.canonical_acceptable("financial")) == {
        "bank_account_number",
        "bank_routing_code",
        "monetary_amount",
        "organization",
        "payment_card_data",
        OUTSIDE,
    }
    assert "quantity" not in ontology.canonical_acceptable("financial")


def test_explicit_account_number_label_drops_the_currency_and_amount_candidates(ontology):
    assert ontology.source_fallback("openai_8", "account_number") == "bank_account_number"
    assert set(ontology.source_acceptable("openai_8", "account_number")) == {
        "bank_account_number",
        "government_id",
        "payment_card_data",
        "record_identifier",
    }
    assert ontology.canonical_acceptable("account_number") == ("bank_account_number",)


def test_spy_generic_id_number_spans_four_identifier_kinds(ontology):
    assert ontology.source_fallback("spy_7", "ID_NUM") == "government_id"
    assert set(ontology.source_acceptable("spy_7", "ID_NUM")) == {
        "government_id",
        "record_identifier",
        "bank_account_number",
        "payment_card_data",
    }


# ---- fail-closed derivation of every acceptable set ----------------------


def test_canonical_derivation_drift_is_rejected(spec):
    spec["old_v1"]["county"] = {"fallback": "admin_area", "acceptable": ["admin_area"]}
    with pytest.raises(OntologyError, match="old_v1.region: acceptable set drifted"):
        validate_spec(spec)


def test_a_stray_canonical_candidate_is_rejected(spec):
    spec["old_v1"]["region"]["acceptable"] = ["admin_area", "locality", "location"]
    with pytest.raises(OntologyError, match="old_v1.region: acceptable set drifted"):
        validate_spec(spec)


def test_source_derivation_drift_without_an_override_is_rejected(spec):
    spec["sources"]["tab_8"]["DEM"] = {
        "fallback": "demographic_attribute",
        "acceptable": ["demographic_attribute"],
    }
    with pytest.raises(OntologyError, match="sources.tab_8.DEM: acceptable set is neither"):
        validate_spec(spec)


def test_dropping_an_override_while_keeping_its_effect_is_rejected(spec):
    construction = spec["acceptable_set_construction"]
    construction["source_overrides"] = [
        override for override in construction["source_overrides"] if override["label"] != "ID_NUM"
    ]
    with pytest.raises(OntologyError, match="sources.spy_7.ID_NUM: acceptable set is neither"):
        validate_spec(spec)


def test_a_source_fallback_that_leaves_its_canonical_projection_is_rejected(spec):
    spec["sources"]["openai_8"]["account_number"]["fallback"] = "payment_card_data"
    with pytest.raises(OntologyError, match="is neither the canonical fallback"):
        validate_spec(spec)


def test_a_redundant_curated_link_is_rejected(spec):
    spec["acceptable_set_construction"]["curated_links"]["contact"] = [
        "email",
        "phone_number",
        "url",
        OUTSIDE,
    ]
    with pytest.raises(OntologyError, match="curated_links.contact: adds nothing"):
        validate_spec(spec)


def test_a_curated_link_on_an_unknown_node_is_rejected(spec):
    spec["acceptable_set_construction"]["curated_links"]["not_a_node"] = [OUTSIDE]
    with pytest.raises(OntologyError, match="curated_links.not_a_node: unknown v1 node"):
        validate_spec(spec)


def test_an_ineffective_source_override_is_rejected(spec):
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "tab_8",
            "label": "ORG",
            "acceptable": ["organization"],
            "reason": "already the canonical projection for organization",
        }
    )
    with pytest.raises(OntologyError, match="override equals the canonical projection"):
        validate_spec(spec)


def test_an_override_may_omit_acceptable_and_move_only_the_fallback(spec):
    canonical = ontology_canonical(spec, "person_attribute")
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "tab_8",
            "label": "DEM",
            "fallback": "health_condition",
            "reason": "a fallback-only override must not have to restate the canonical set",
        }
    )
    spec["sources"]["tab_8"]["DEM"]["fallback"] = "health_condition"
    ontology = validate_spec(spec)
    assert ontology.source_fallback("tab_8", "DEM") == "health_condition"
    # The omitted half kept its canonical value rather than narrowing.
    assert set(ontology.source_acceptable("tab_8", "DEM")) == set(canonical)
    assert ontology.canonical_fallback("person_attribute") == "demographic_attribute"


def test_an_override_may_omit_fallback_and_move_only_the_acceptable_set(spec):
    narrowed = ["location", "locality"]
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "tab_8",
            "label": "LOC",
            "acceptable": narrowed,
            "reason": "an acceptable-only override keeps the canonical fallback",
        }
    )
    spec["sources"]["tab_8"]["LOC"]["acceptable"] = narrowed
    ontology = validate_spec(spec)
    assert ontology.source_fallback("tab_8", "LOC") == "location"
    assert set(ontology.source_acceptable("tab_8", "LOC")) == set(narrowed)


def test_an_override_declaring_neither_field_is_rejected(spec):
    spec["acceptable_set_construction"]["source_overrides"].append(
        {"schema": "tab_8", "label": "ORG", "reason": "declares nothing at all"}
    )
    with pytest.raises(OntologyError, match="must declare a fallback, an acceptable set, or both"):
        validate_spec(spec)


def test_a_fallback_only_override_still_has_to_match_its_source_cell(spec):
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "tab_8",
            "label": "DEM",
            "fallback": "health_condition",
            "reason": "the source cell is deliberately left disagreeing",
        }
    )
    with pytest.raises(OntologyError, match="is neither the canonical fallback"):
        validate_spec(spec)


def test_an_override_fallback_outside_its_own_acceptable_set_is_rejected(spec):
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "tab_8",
            "label": "ORG",
            "fallback": "locality",
            "acceptable": ["organization", "location"],
            "reason": "the declared fallback is not among the declared targets",
        }
    )
    with pytest.raises(OntologyError, match="fallback 'locality' is not in the override"):
        validate_spec(spec)


def test_an_override_with_an_unknown_field_is_rejected(spec):
    spec["acceptable_set_construction"]["source_overrides"][0]["notes"] = "typo for reason"
    with pytest.raises(OntologyError, match=r"source_overrides\[0\]: unknown fields: notes"):
        validate_spec(spec)


def test_a_duplicate_source_override_is_rejected(spec):
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "multigrascco",
            "label": "ID",
            "acceptable": ["government_id"],
            "reason": "a second, conflicting override",
        }
    )
    with pytest.raises(OntologyError, match="duplicate override for multigrascco.ID"):
        validate_spec(spec)


def test_an_override_for_a_label_v1_does_not_have_is_rejected(spec):
    spec["acceptable_set_construction"]["source_overrides"].append(
        {
            "schema": "tab_8",
            "label": "NOT_A_LABEL",
            "acceptable": ["record_identifier"],
            "reason": "no such label",
        }
    )
    with pytest.raises(OntologyError, match="is not a v1 source label"):
        validate_spec(spec)


def test_declared_construction_rule_drift_is_rejected(spec):
    spec["acceptable_set_construction"]["ancestor_images_added"] = True
    with pytest.raises(OntologyError, match="ancestor images are deliberately not added"):
        validate_spec(spec)


def test_unknown_schema_and_label_lookups_fail(ontology):
    with pytest.raises(OntologyError, match="unknown source schema"):
        ontology.source_fallback("not_a_schema", "X")
    with pytest.raises(OntologyError, match="tab_8: unknown label"):
        ontology.source_fallback("tab_8", "NOT_A_LABEL")
    with pytest.raises(OntologyError, match="unknown canonical v1 node"):
        ontology.canonical_fallback("not_a_node")


# ---- attribute channels --------------------------------------------------


def test_exactly_two_channels_applicable_only_to_person_name(ontology):
    assert tuple(sorted(ontology.attr_channels)) == ATTR_CHANNELS
    for entry in ontology.attr_channels.values():
        assert entry["applicable_types"] == ("person_name",)
        assert entry["semantics_version"] == 1
        assert entry["value_kind"] == "character_extents"
    assert ontology.channels_for("person_name") == ATTR_CHANNELS
    assert ontology.channels_for("email") == ()


@pytest.mark.parametrize("value", [None, "labels", "spans"])
def test_attr_channel_value_kind_drift_in_the_spec_is_rejected(spec, value):
    entry = spec["attr_channels"]["family_name"]
    if value is None:
        entry.pop("value_kind")
    else:
        entry["value_kind"] = value
    with pytest.raises(OntologyError, match="value_kind must be 'character_extents'"):
        validate_spec(spec)


def test_registry_fields_must_include_value_kind(spec):
    spec["gold_metadata_contract"]["registry_fields"] = ["applicable_types", "semantics_version"]
    with pytest.raises(OntologyError, match="registry_fields must be"):
        validate_spec(spec)


def test_no_honorific_channel_is_adopted(ontology):
    assert "honorific" not in ontology.attr_channels


def test_attrs_positive_negative_and_unknown(ontology):
    positive = record(
        spans=[
            {
                "start": 0,
                "end": 16,
                "type": "person_name",
                "attrs": {"family_name": [[8, 16]], "care_provider": [[0, 16]]},
            }
        ]
    )
    ontology.validate_record(positive)

    negative = record(spans=[{"start": 0, "end": 16, "type": "person_name", "attrs": {"family_name": []}}])
    ontology.validate_record(negative)

    unknown = record(spans=[{"start": 0, "end": 16, "type": "person_name", "attrs": {}}])
    ontology.validate_record(unknown)


def test_multiple_disjoint_intervals_are_valid(ontology):
    row = record(
        spans=[
            {
                "start": 0,
                "end": 16,
                "type": "person_name",
                "attrs": {"family_name": [[4, 7], [8, 16]]},
            }
        ]
    )
    ontology.validate_record(row)


def test_attrs_absence_is_backward_compatible(ontology):
    row = ontology.validate_record(record())
    assert all("attrs" not in span for span in row["spans"])


def test_unknown_top_level_fields_are_preserved(ontology):
    row = ontology.validate_record(record(lang="nl", span_provenance=[{"generator": "x"}]))
    assert row["lang"] == "nl"
    assert row["span_provenance"] == [{"generator": "x"}]


@pytest.mark.parametrize(
    "attrs,message",
    [
        ({"honorific": [[0, 3]]}, "undeclared attribute channel"),
        ({"family_name": [[8, 20]]}, "escapes the span"),
        ({"family_name": [[8, 8]]}, "empty interval"),
        ({"family_name": [[8, 12], [10, 16]]}, "overlapping intervals"),
        ({"family_name": [[8, 16], [8, 16]]}, "overlapping intervals"),
        ({"family_name": [[True, 16]]}, "interval offsets must be integers"),
        ({"family_name": [[8]]}, r"expected \[start, end\] intervals"),
        ({"family_name": [8, 16]}, r"expected \[start, end\] intervals"),
        ({"family_name": "8-16"}, "must be a list of intervals"),
        ({"family_name": True}, "must be a list of intervals"),
    ],
)
def test_malformed_attrs_are_rejected(ontology, attrs, message):
    row = record(spans=[{"start": 0, "end": 16, "type": "person_name", "attrs": attrs}])
    with pytest.raises(OntologyError, match=message):
        ontology.validate_record(row)


def test_attrs_on_an_inapplicable_type_are_rejected(ontology):
    row = record(
        text="ada@example.com",
        spans=[{"start": 0, "end": 15, "type": "email", "attrs": {"family_name": [[0, 3]]}}],
    )
    with pytest.raises(OntologyError, match="does not apply to email"):
        ontology.validate_record(row)


# ---- record contract -----------------------------------------------------


def test_unknown_primary_type_is_rejected(ontology):
    row = record(spans=[{"start": 0, "end": 16, "type": "family_name"}])
    with pytest.raises(OntologyError, match="unknown primary type 'family_name'"):
        ontology.validate_record(row)


def test_outside_is_not_a_span_type(ontology):
    row = record(spans=[{"start": 0, "end": 16, "type": "O"}])
    with pytest.raises(OntologyError, match="unknown primary type 'O'"):
        ontology.validate_record(row)


def test_overlapping_primary_spans_are_rejected(ontology):
    assert ontology.forbid_overlapping_spans
    row = record(
        spans=[
            {"start": 0, "end": 16, "type": "person_name"},
            {"start": 4, "end": 16, "type": "person_name"},
        ]
    )
    with pytest.raises(OntologyError, match="overlapping spans"):
        ontology.validate_record(row)


def test_unsorted_spans_are_rejected(ontology):
    row = record(
        spans=[
            {"start": 21, "end": 32, "type": "person_name"},
            {"start": 0, "end": 16, "type": "person_name"},
        ]
    )
    with pytest.raises(OntologyError, match="unsorted spans"):
        ontology.validate_record(row)


@pytest.mark.parametrize(
    "span,message",
    [
        ({"start": -1, "end": 16, "type": "person_name"}, "invalid span"),
        ({"start": 16, "end": 16, "type": "person_name"}, "invalid span"),
        ({"start": 0, "end": 400, "type": "person_name"}, "invalid span"),
        ({"start": True, "end": 16, "type": "person_name"}, "offsets must be integers"),
        ({"start": 0.0, "end": 16, "type": "person_name"}, "offsets must be integers"),
        ({"start": 0, "end": 16}, "missing type"),
        ({"start": 0, "end": 16, "type": "person_name", "label": "x"}, "unknown span fields: label"),
    ],
)
def test_malformed_spans_are_rejected(ontology, span, message):
    with pytest.raises(OntologyError, match=message):
        ontology.validate_record(record(spans=[span]))


def test_legacy_triple_spans_are_rejected(ontology):
    with pytest.raises(OntologyError, match=r"expected a \{start, end, type, attrs\?\} mapping"):
        ontology.validate_record(record(spans=[[0, 16, "person_name"]]))


@pytest.mark.parametrize(
    "row,message",
    [
        ({"text": "x", "spans": []}, "missing id"),
        ({"id": "a", "spans": []}, "missing text"),
        ({"id": "a", "text": "x"}, "missing spans"),
        ({"id": "", "text": "x", "spans": []}, "id must be a nonempty string"),
        ({"id": "a", "text": 7, "spans": []}, "text must be a string"),
        ({"id": "a", "text": "x", "spans": {}}, "spans must be a list"),
    ],
)
def test_malformed_records_are_rejected(ontology, row, message):
    with pytest.raises(OntologyError, match=message):
        ontology.validate_record(row)


# ---- dataset metadata ----------------------------------------------------


def test_valid_dataset_metadata(ontology):
    ontology.validate_dataset_metadata(deepcopy(VALID_METADATA))


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda meta: meta.pop("attr_channels"), "missing attr_channels"),
        (lambda meta: meta.update(ontology_version="pii-ontology-v1"), "ontology_version"),
        (lambda meta: meta.update(schema_version="pii-gold-v1"), "schema_version"),
        (
            lambda meta: meta["attr_channels"].update(
                honorific={
                    "applicable_types": ["person_name"],
                    "semantics_version": 1,
                    "value_kind": "character_extents",
                }
            ),
            "undeclared attribute channels: honorific",
        ),
        (
            lambda meta: meta["attr_channels"].pop("care_provider"),
            "attribute channels not declared: care_provider",
        ),
        (
            lambda meta: meta["attr_channels"]["family_name"].update(
                applicable_types=["person_name", "email"]
            ),
            "applicable_types drift",
        ),
        (
            lambda meta: meta["attr_channels"]["family_name"].update(semantics_version=2),
            "semantics_version drift",
        ),
        (
            lambda meta: meta["attr_channels"]["family_name"].update(value_kind="labels"),
            "value_kind must be 'character_extents'",
        ),
        (
            lambda meta: meta["attr_channels"]["care_provider"].pop("value_kind"),
            "value_kind must be 'character_extents'",
        ),
    ],
)
def test_dataset_metadata_registry_mismatch_is_rejected(ontology, mutate, message):
    metadata = deepcopy(VALID_METADATA)
    mutate(metadata)
    with pytest.raises(OntologyError, match=message):
        ontology.validate_dataset_metadata(metadata)


def test_attr_channel_registry_drift_in_the_spec_is_rejected(spec):
    spec["attr_channels"]["honorific"] = {
        "applicable_types": ["person_name"],
        "semantics_version": 1,
        "value_kind": "character_extents",
        "definition": "not adopted",
    }
    with pytest.raises(OntologyError, match="attr_channels must be exactly"):
        validate_spec(spec)


def test_attr_channel_applicability_drift_in_the_spec_is_rejected(spec):
    spec["attr_channels"]["family_name"]["applicable_types"] = ["person_name", "not_a_type"]
    with pytest.raises(OntologyError, match="unknown applicable types: not_a_type"):
        validate_spec(spec)


# ---- frozen remap admission ----------------------------------------------


def test_remap_admission_metadata_is_frozen(ontology):
    admission = ontology.remap_admission
    assert admission["version"] == REMAP_ADMISSION_VERSION
    assert admission["margin_threshold_nats"] == REMAP_MARGIN_NATS == pytest.approx(1.0986122886681098)
    assert admission["margin_threshold_expression"] == "ln(3)"
    assert admission["ties"] == "retain_fallback"
    assert admission["fallback_winner_requires_margin"] is False
    assert admission["singleton_acceptable_set"] == "deterministic_not_model_supported"
    assert admission["argmax_restricted_to"] == "declared_acceptable_set"
    assert admission["token_inclusion"] == "every_nonempty_offset_token_overlapping_the_span"
    assert admission["candidate_strength"] == "mean_over_included_tokens_of_max_bioes_column_logit"
    assert admission["strength_columns"] == ["B", "I", "E", "S"]
    assert admission["logits"] == "raw"
    assert admission["status"] == "proposal_only"
    assert admission["evaluation_promotion_automatic"] is False
    assert set(admission["record_fields"]) >= {
        "winner",
        "runner_up",
        "winner_strength",
        "runner_up_strength",
        "margin",
        "threshold",
        "fallback",
        "model_identity",
        "token_offsets",
        "route_provenance",
    }


def test_remap_admission_threshold_drift_is_rejected(spec):
    spec["remap_admission"]["margin_threshold_nats"] = 1.0
    with pytest.raises(OntologyError, match="margin must be ln\\(3\\)"):
        validate_spec(spec)


def test_admit_remap_requires_the_margin_to_leave_the_fallback(ontology):
    acceptable = ontology.canonical_acceptable("misc_identifier")
    below = ontology.admit_remap(
        "record_identifier",
        acceptable,
        {target: (1.0 if target == "government_id" else 0.0) for target in acceptable},
    )
    assert below["label"] == "record_identifier"
    assert below["winner"] == "government_id"
    assert below["changed"] is False
    assert below["decision"] == "margin_below_threshold"

    above = ontology.admit_remap(
        "record_identifier",
        acceptable,
        {target: (REMAP_MARGIN_NATS if target == "government_id" else 0.0) for target in acceptable},
    )
    assert above["label"] == "government_id"
    assert above["changed"] is True
    assert above["margin"] == pytest.approx(REMAP_MARGIN_NATS)
    assert above["threshold"] == REMAP_MARGIN_NATS
    assert above["status"] == "proposal_only"


def test_admit_remap_retains_the_fallback_without_a_margin(ontology):
    acceptable = ontology.canonical_acceptable("misc_identifier")
    result = ontology.admit_remap(
        "record_identifier",
        acceptable,
        {target: (0.001 if target == "record_identifier" else 0.0) for target in acceptable},
    )
    assert result["label"] == "record_identifier"
    assert result["decision"] == "retained_fallback"
    assert result["margin"] == pytest.approx(0.001)


def test_admit_remap_ties_retain_the_fallback(ontology):
    acceptable = ontology.canonical_acceptable("misc_identifier")
    result = ontology.admit_remap("record_identifier", acceptable, {target: 1.0 for target in acceptable})
    assert result["winner"] == "record_identifier"
    assert result["label"] == "record_identifier"
    assert result["margin"] == 0.0


def test_admit_remap_singleton_is_deterministic(ontology):
    result = ontology.admit_remap("record_identifier", ("record_identifier",), {})
    assert result["decision"] == "singleton_acceptable_set"
    assert result["label"] == "record_identifier"
    assert result["margin"] is None


@pytest.mark.parametrize(
    "fallback,acceptable,strengths,message",
    [
        ("email", ("person_name", "O"), {"person_name": 1.0, "O": 0.0}, "is not acceptable"),
        ("person_name", ("person_name", "O"), {"person_name": 1.0}, "no strength for O"),
        (
            "person_name",
            ("person_name", "O"),
            {"person_name": 1.0, "O": 0.0, "email": 3.0},
            "strengths outside the acceptable set",
        ),
        (
            "person_name",
            ("person_name", "O"),
            {"person_name": float("inf"), "O": 0.0},
            "must be a finite number",
        ),
        ("person_name", ("person_name", "O"), {"person_name": True, "O": 0.0}, "must be a finite number"),
        (
            "person_name",
            ("person_name", "family_name"),
            {"person_name": 1.0, "family_name": 0.0},
            r"acceptable targets outside primary\+O: family_name",
        ),
        ("not_a_type", ("not_a_type",), {}, r"acceptable targets outside primary\+O: not_a_type"),
        (
            "person_name",
            ("person_name", "person_name"),
            {"person_name": 1.0},
            "duplicate acceptable targets",
        ),
        ("person_name", (), {}, "acceptable must be nonempty"),
    ],
)
def test_admit_remap_rejects_malformed_candidates(ontology, fallback, acceptable, strengths, message):
    with pytest.raises(OntologyError, match=message):
        ontology.admit_remap(fallback, acceptable, strengths)


@pytest.mark.parametrize("acceptable", [("record_identifier",), ("record_identifier", "government_id")])
def test_admit_remap_rejects_a_threshold_override(ontology, acceptable):
    """The singleton path must not slip past the frozen-threshold check."""
    strengths = {target: 0.0 for target in acceptable}
    with pytest.raises(OntologyError, match=r"the ln\(3\) margin is frozen"):
        ontology.admit_remap("record_identifier", acceptable, strengths, threshold=0.5)


def test_admit_remap_accepts_the_frozen_threshold_stated_explicitly(ontology):
    result = ontology.admit_remap(
        "record_identifier", ("record_identifier",), {}, threshold=REMAP_MARGIN_NATS
    )
    assert result["decision"] == "singleton_acceptable_set"


@pytest.mark.parametrize(
    "fallback,acceptable",
    [
        ("not_a_type", ("not_a_type",)),
        ("family_name", ("family_name",)),
    ],
)
def test_admit_remap_rejects_an_invalid_singleton(ontology, fallback, acceptable):
    with pytest.raises(OntologyError, match=r"acceptable targets outside primary\+O"):
        ontology.admit_remap(fallback, acceptable, {})


def test_admit_remap_rejects_a_singleton_whose_fallback_is_not_its_member(ontology):
    with pytest.raises(OntologyError, match="is not acceptable"):
        ontology.admit_remap("email", ("person_name",), {})


# ---- predicate and scoring metadata --------------------------------------


def test_predicate_contract_is_recorded(ontology):
    contract = ontology.predicate_contract
    assert contract["objective"] == "independent_masked_bce"
    assert contract["encoding"] == "not_bioes"
    assert contract["token_supervision"] == "existential_over_characters"
    assert contract["evaluation"]["missing_gold_key"] == "excluded"
    assert contract["evaluation"]["primary_span_metrics_use_predicates"] is False


def test_raw_logits_are_per_model_position_not_per_decoded_span(ontology):
    contract = ontology.predicate_contract
    assert contract["raw_output_granularity"] == "model_position"
    assert contract["primary_logit_shape"] == ["B", "T", 117]
    assert contract["predicate_logit_shape"] == ["B", "T", "K"]
    assert contract["logit_axes"] == ["batch", "position", "channel"]
    assert "primary_span" not in contract["logit_axes"]
    assert "predicate_logit_axes" not in contract
    assert contract["position_granularity"] == "declared_by_producer_metadata"
    assert contract["position_granularity_values"] == ["token", "character"]
    # Interpreted output is still spans plus per-span character extents.
    assert "Primary spans" in contract["interpreted_output"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "softmax_cross_entropy"),
        ("predicate_logit_shape", ["B", "P", "K"]),
        ("primary_logit_shape", ["B", "P", 117]),
        ("logit_axes", ["batch", "primary_span", "channel"]),
        ("raw_output_granularity", "primary_span"),
        ("position_granularity", "token"),
    ],
)
def test_predicate_contract_drift_is_rejected(spec, field, value):
    spec["predicate_contract"][field] = value
    with pytest.raises(OntologyError, match=f"predicate_contract.{field}"):
        validate_spec(spec)


def test_a_stale_per_span_axis_declaration_is_rejected(spec):
    spec["predicate_contract"]["predicate_logit_axes"] = ["batch", "primary_span", "channel"]
    with pytest.raises(OntologyError, match="predicate_logit_axes is superseded"):
        validate_spec(spec)


def test_compatibility_and_warm_init_metadata(ontology):
    scoring = ontology.compatibility_scoring
    assert scoring["primary_matching"] == ["exact", "symmetric_80pct_overlap"]
    assert scoring["assignment"] == "maximum_cardinality_one_to_one"
    assert scoring["permanent_v2_gold_scoring_uses_compatibility"] is False
    assert set(scoring["compatibility_relation_kinds"]) == {
        "functional",
        "hierarchy",
        "indeterminacy",
        "bucket",
    }
    assert set(scoring["compatibility_relation_semantics"]) == set(scoring["compatibility_relation_kinds"])
    assert "0.8" in scoring["overlap_criterion"]
    warm = ontology.warm_init
    assert warm["merged_rows"] == "mass_weighted"
    assert warm["merged_bias"] == "contributor_log_sum_exp"
    assert warm["merged_bias_formula"] == "b_new = logsumexp(b_1, ..., b_n)"
    assert "combined prior" in warm["merged_bias_rationale"]
    assert "support-mass-weighted" in warm["merged_rows_formula"]
    assert warm["outside_row"] == "copy"
    assert warm["split_rows"] == "duplicate_with_symmetry_breaking"
    assert warm["new_predicate_heads"] == "initialized_separately"
    assert warm["matched_control"] == "same_v11_encoder_with_fresh_or_refitted_v2_head"
    assert warm["checkpoints"] == [250, 500, 1000, 1500]
    assert warm["encoder_schedule"] == "frozen_for_initial_head_fit_then_unfrozen"
    assert warm["adoption_criterion"] == "better_at_early_checkpoints_and_no_worse_at_the_matched_final_rung"
    assert set(warm["matched_pilot"]["must_match"]) == {
        "data",
        "order",
        "optimizer",
        "schedule",
        "budget",
        "seeds",
    }
    assert warm["matched_pilot"]["first_stage"] == "fit_new_head_with_encoder_frozen"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("primary_matching", ["exact"], "exact plus symmetric 80% overlap"),
        ("assignment", "greedy", "maximum-cardinality one-to-one"),
        (
            "compatibility_relation_kinds",
            ["functional", "hierarchy", "indeterminacy"],
            "compatibility_relation_kinds must be exactly",
        ),
        (
            "compatibility_relation_kinds",
            ["functional", "hierarchy", "indeterminacy", "bucket", "other"],
            "compatibility_relation_kinds must be exactly",
        ),
        ("permanent_v2_gold_scoring_uses_compatibility", True, "never scores through"),
        ("overlap_criterion", "", "symmetric overlap criterion"),
    ],
)
def test_compatibility_scoring_drift_is_rejected(spec, field, value, message):
    spec["compatibility_scoring"][field] = value
    with pytest.raises(OntologyError, match=re.escape(message)):
        validate_spec(spec)


def test_relation_semantics_must_cover_the_declared_kinds(spec):
    spec["compatibility_scoring"]["compatibility_relation_semantics"].pop("bucket")
    with pytest.raises(OntologyError, match="must cover exactly the declared relation kinds"):
        validate_spec(spec)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("merged_rows", "averaged", "warm_init.merged_rows must be"),
        ("merged_bias", "averaged", "warm_init.merged_bias must be"),
        ("merged_bias_formula", "b_new = mean(b_1, ..., b_n)", "warm_init.merged_bias_formula"),
        ("merged_rows_formula", "", "a stated formula or rationale is required"),
        ("split_rows", "duplicate", "warm_init.split_rows must be"),
        ("outside_row", "reinitialized", "warm_init.outside_row must be"),
        ("checkpoints", [250, 500, 1000], "warm_init.checkpoints must be"),
        ("encoder_schedule", "always_unfrozen", "warm_init.encoder_schedule must be"),
        ("matched_control", "a_different_encoder", "warm_init.matched_control must be"),
        ("adoption_criterion", "better_on_average", "warm_init.adoption_criterion must be"),
    ],
)
def test_warm_init_drift_is_rejected(spec, field, value, message):
    spec["warm_init"][field] = value
    with pytest.raises(OntologyError, match=re.escape(message)):
        validate_spec(spec)


def test_attribute_provenance_hints_are_recorded(ontology):
    hints = ontology.spec["attribute_provenance_hints"]
    assert hints["family_name"]["known_positive_whole_extent"] == ["family_name"]
    assert set(hints["family_name"]["known_negative"]) == {"given_name", "middle_name", "name_prefix"}
    assert hints["care_provider"]["known_positive_whole_extent"] == ["clinician_name"]
    assert set(hints["care_provider"]["known_negative"]) == {"patient_name", "relative_name"}


# ---- gold JSON Schema ----------------------------------------------------


def test_gold_schema_matches_the_ontology(ontology):
    validate_gold_schema(ontology=ontology)
    schema = load_gold_schema()
    assert tuple(schema["$defs"]["primary_type"]["enum"]) == ADOPTED_PRIMARY_INVENTORY
    assert schema["$defs"]["record"]["additionalProperties"] is True
    assert schema["$defs"]["span"]["additionalProperties"] is False
    assert schema["$defs"]["attrs"]["additionalProperties"] is False


def test_gold_schema_inventory_drift_is_rejected(ontology):
    schema = deepcopy(load_gold_schema())
    schema["$defs"]["primary_type"]["enum"].append("family_name")
    with pytest.raises(OntologyError, match="primary_type enum does not match"):
        validate_gold_schema(schema, ontology)


def test_gold_schema_internal_references_resolve():
    schema = load_gold_schema()
    definitions = schema["$defs"]

    def references(node):
        if isinstance(node, dict):
            if isinstance(node.get("$ref"), str):
                yield node["$ref"]
            for value in node.values():
                yield from references(value)
        elif isinstance(node, list):
            for value in node:
                yield from references(value)

    seen = set(references(schema))
    assert seen
    for reference in seen:
        assert reference.startswith("#/$defs/"), reference
        assert reference.split("/")[-1] in definitions, reference


# ---- loader and CLI smoke ------------------------------------------------


def gold_file(tmp_path: Path, *, metadata=None) -> Path:
    path = tmp_path / "gold.jsonl"
    rows = [
        record(id="row-1"),
        record(
            id="row-2",
            text="Betaal EUR 300 aan Ada Lovelace.",
            spans=[
                {"start": 7, "end": 14, "type": "monetary_amount"},
                {
                    "start": 19,
                    "end": 31,
                    "type": "person_name",
                    "attrs": {"family_name": [[23, 31]], "care_provider": []},
                },
            ],
        ),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    sidecar = path.with_name(path.name + ".meta.json")
    sidecar.write_text(
        json.dumps({"ontology": deepcopy(VALID_METADATA) if metadata is None else metadata}),
        encoding="utf-8",
    )
    return path


def test_validate_jsonl_accepts_a_conforming_file(ontology, tmp_path):
    summary = ontology.validate_jsonl(gold_file(tmp_path))
    assert summary["records"] == 2
    assert summary["spans"] == 4
    assert summary["attributed_spans"] == 1


def test_validate_jsonl_requires_matching_metadata(ontology, tmp_path):
    metadata = deepcopy(VALID_METADATA)
    metadata["schema_version"] = "pii-gold-v1"
    path = gold_file(tmp_path, metadata=metadata)
    with pytest.raises(OntologyError, match="schema_version"):
        ontology.validate_jsonl(path)


def test_validate_jsonl_reports_the_offending_line(ontology, tmp_path):
    path = gold_file(tmp_path)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record(id="row-3", spans=[{"start": 0, "end": 400, "type": "date"}])) + "\n")
    with pytest.raises(OntologyError, match=r"gold\.jsonl:3: span 0: invalid span"):
        ontology.validate_jsonl(path)


def test_validate_jsonl_rejects_duplicate_ids(ontology, tmp_path):
    path = gold_file(tmp_path)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record(id="row-1")) + "\n")
    with pytest.raises(OntologyError, match="duplicate record id"):
        ontology.validate_jsonl(path)


def test_missing_metadata_sidecar_is_rejected(ontology, tmp_path):
    path = gold_file(tmp_path)
    path.with_name(path.name + ".meta.json").unlink()
    with pytest.raises(OntologyError, match="missing dataset metadata sidecar"):
        ontology.validate_jsonl(path)
    assert ontology.validate_jsonl(path, require_metadata=False)["records"] == 2


def test_loader_is_cached(ontology):
    assert load_ontology() is ontology


def test_cli_self_test_and_file_validation(tmp_path):
    path = gold_file(tmp_path)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "pii_ontology_v2.py"),
            "--jsonl",
            str(path),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "29 primary classes, 117 BIOES labels" in result.stdout
    assert "self-tests passed" in result.stdout
    assert "2 records, 4 spans" in result.stdout
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["ontology"]["primary_classes"] == list(ADOPTED_PRIMARY_INVENTORY)
    assert written["files"][0]["records"] == 2


# ---- qualified-gold delta: per-source overrides ---------------------------


@pytest.mark.parametrize(
    "schema,label,fallback,node,node_fallback",
    [
        ("tab_8", "MISC", OUTSIDE, "misc_identifier", "record_identifier"),
        (
            "meddocan_phi",
            "FAMILIARES_SUJETO_ASISTENCIA",
            "demographic_attribute",
            "relative_name",
            "person_name",
        ),
        ("meddocan_phi", "TERRITORIO", "locality", "location", "location"),
        ("multigrascco", "HEALTH_FCLT", "demographic_attribute", "healthcare_org", "organization"),
        (
            "multigrascco",
            "DIRECT_ID_ADDRESS",
            "demographic_attribute",
            "address",
            "street_address",
        ),
    ],
)
def test_a_source_override_moves_one_cell_and_never_its_shared_v1_node(
    ontology, schema, label, fallback, node, node_fallback
):
    assert ontology.source_fallback(schema, label) == fallback
    assert ontology.canonical_fallback(node) == node_fallback


def test_tab_code_gains_the_guidelines_authority_and_contact_candidates(ontology):
    assert ontology.source_acceptable("tab_8", "CODE") == (
        "government_id",
        "phone_number",
        "record_identifier",
    )
    assert ontology.source_fallback("tab_8", "CODE") == "record_identifier"


def test_tab_datetime_can_reach_outside_for_its_duration_share(ontology):
    assert OUTSIDE in ontology.source_acceptable("tab_8", "DATETIME")
    assert ontology.source_fallback("tab_8", "DATETIME") == "date"


def test_quantity_no_longer_claims_generic_durations(ontology):
    definition = ontology.primary["quantity"]["definition"]
    assert "duration" in definition
    assert re.search(r"durations?.{0,80}outside this de-identification ontology", definition)


# ---- source extensions ---------------------------------------------------

EXTENSION_SCHEMAS = ("hiner_original", "klue_ner", "multigrascco", "wojood_nested")


def test_extensions_do_not_touch_the_v1_compatibility_surface(ontology, v1_tagset):
    for schema in EXTENSION_SCHEMAS:
        assert set(ontology.source_labels(schema)) == set(v1_tagset["sources"][schema])
        assert not set(ontology.source_extension_labels(schema)) & set(v1_tagset["sources"][schema])


@pytest.mark.parametrize(
    "schema,labels",
    [
        ("hiner_original", ("FESTIVAL", "GAME", "LITERATURE", "MISC", "NUMEX", "TIMEX")),
        ("klue_ner", ("DT", "QT", "TI")),
        ("multigrascco", ("RELATIVE_TIME",)),
    ],
)
def test_declared_extension_inventory(ontology, schema, labels):
    assert ontology.source_extension_labels(schema) == labels


def test_wojood_declares_all_ten_ignored_labels_totalling_775_spans(ontology):
    labels = ontology.source_extension_labels("wojood_nested")
    assert len(labels) == 10
    total = sum(ontology.source_extension_evidence("wojood_nested", label)["spans"] for label in labels)
    assert total == 775


def test_hiner_split_preserving_counts_match_the_audit(ontology):
    evidence = {
        label: ontology.source_extension_evidence("hiner_original", label)
        for label in ontology.source_extension_labels("hiner_original")
    }
    assert evidence["NUMEX"]["spans"] == 24289
    assert evidence["TIMEX"]["spans"] == 18412
    named = ("FESTIVAL", "GAME", "LITERATURE", "MISC")
    assert sum(evidence[label]["spans"] for label in named) == 8524
    for label, entry in evidence.items():
        assert entry["spans"] == sum(entry["by_split"].values()), label


def test_klue_split_preserving_counts_match_the_audit(ontology):
    spans = {
        label: ontology.source_extension_evidence("klue_ner", label)["spans"]
        for label in ontology.source_extension_labels("klue_ner")
    }
    assert spans == {"DT": 10341, "QT": 14868, "TI": 2565}


def test_multigrascco_relative_time_is_evaluation_only(ontology):
    assert ontology.source_promotion("multigrascco", "RELATIVE_TIME") == "evaluation_only"
    assert ontology.source_route("multigrascco", "RELATIVE_TIME") == "currently_unrepresentable"
    assert ontology.source_extension_evidence("multigrascco", "RELATIVE_TIME")["spans"] == 5444


def test_no_recovered_label_is_training_eligible_in_this_contract(ontology):
    for schema in ("hiner_original", "klue_ner", "wojood_nested"):
        for label in ontology.source_extension_labels(schema):
            assert ontology.source_promotion(schema, label) == "withheld", f"{schema}.{label}"


def test_legacy_labels_get_a_derived_route_but_no_promotion(ontology):
    assert ontology.source_route("tab_8", "PERSON") == "deterministic"
    assert ontology.source_route("tab_8", "DEM") == "constrained_discrimination"
    assert not ontology.is_source_extension("tab_8", "PERSON")


@pytest.mark.parametrize(
    "schema,label",
    [
        ("nemotron_pii", "date"),  # publisher-synthetic
        ("authored_v1", "date"),  # project-authored template
        ("spy_7", "ID_NUM"),  # released synthetic
        ("tab_8", "PERSON"),  # qualified human gold
        ("meddocan_phi", "NOMBRE_SUJETO_ASISTENCIA"),  # qualified human gold
        ("multigrascco", "NAME_DOCTOR"),  # evaluation gold
    ],
)
def test_no_legacy_label_acquires_training_eligibility_from_its_mapping(ontology, schema, label):
    # A label mapping says what a span means, never which corpus stratum it came
    # from. The 24 v1 schemas mix qualified gold with publisher-synthetic,
    # authored-template and translated sources, so promotion must stay undeclared.
    assert ontology.source_promotion(schema, label) is None


def test_promotion_is_undeclared_for_every_legacy_cell_in_every_schema(ontology):
    for schema in ontology.source_schemas():
        for label in ontology.source_labels(schema):
            assert ontology.source_promotion(schema, label) is None, f"{schema}.{label}"


def test_undeclared_legacy_promotion_is_not_a_closed_vocabulary_value():
    assert None not in EXTENSION_PROMOTIONS
    assert "unknown" not in EXTENSION_PROMOTIONS
    assert set(EXTENSION_PROMOTIONS) == {"evaluation_only", "training_eligible", "withheld"}


def test_extension_lookups_resolve_but_typos_still_fail(ontology):
    assert ontology.source_fallback("hiner_original", "NUMEX") == OUTSIDE
    assert set(ontology.source_acceptable("hiner_original", "NUMEX")) == {OUTSIDE, "quantity"}
    with pytest.raises(OntologyError, match="hiner_original: unknown label 'NUMEXX'"):
        ontology.source_fallback("hiner_original", "NUMEXX")
    with pytest.raises(OntologyError, match="unknown source schema"):
        ontology.source_extension_labels("not_a_schema")
    with pytest.raises(OntologyError, match="is not a declared source extension"):
        ontology.source_extension_evidence("tab_8", "PERSON")


def test_every_extension_route_and_promotion_is_in_the_closed_vocabulary(ontology):
    for schema, labels in ontology.source_extensions.items():
        for label, entry in labels.items():
            assert entry["route"] in EXTENSION_ROUTES, f"{schema}.{label}"
            assert entry["promotion"] in EXTENSION_PROMOTIONS, f"{schema}.{label}"
            assert entry["fallback"] in entry["acceptable"]
            assert set(entry["acceptable"]) <= set(ADOPTED_PRIMARY_INVENTORY) | {OUTSIDE}


def extension(spec, schema="hiner_original", label="FESTIVAL"):
    return spec["source_extensions"][schema]["labels"][label]


def test_an_extension_for_an_unknown_schema_is_rejected(spec):
    spec["source_extensions"]["not_a_schema"] = deepcopy(spec["source_extensions"]["hiner_original"])
    with pytest.raises(OntologyError, match="is not an existing v1 schema"):
        validate_spec(spec)


def test_an_extension_that_shadows_a_v1_label_is_rejected(spec):
    labels = spec["source_extensions"]["hiner_original"]["labels"]
    labels["PERSON"] = deepcopy(labels["FESTIVAL"])
    with pytest.raises(OntologyError, match="declare it in sources, not"):
        validate_spec(spec)


def test_a_label_absent_from_the_manifest_ignored_list_is_rejected_as_drift(spec):
    labels = spec["source_extensions"]["hiner_original"]["labels"]
    labels["FESTIVALL"] = labels.pop("FESTIVAL")
    with pytest.raises(OntologyError, match="would be drift, not an extension"):
        validate_spec(spec)


# ---- completeness guard --------------------------------------------------


@pytest.mark.parametrize(
    "schema,label",
    [
        ("hiner_original", "NUMEX"),
        ("hiner_original", "FESTIVAL"),
        ("klue_ner", "QT"),
        ("wojood_nested", "CURR"),
        ("multigrascco", "RELATIVE_TIME"),
    ],
)
def test_dropping_a_recovered_label_from_the_registry_is_rejected(spec, schema, label):
    del spec["source_extensions"][schema]["labels"][label]
    with pytest.raises(OntologyError, match=f"the registry omits {label}"):
        validate_spec(spec)


def test_dropping_several_recovered_labels_is_rejected(spec):
    labels = spec["source_extensions"]["wojood_nested"]["labels"]
    for label in ("PERCENT", "PRODUCT", "QUANTITY"):
        del labels[label]
    with pytest.raises(OntologyError, match="registry omits PERCENT, PRODUCT, QUANTITY"):
        validate_spec(spec)


def test_an_extra_label_beyond_the_recovered_set_is_rejected(spec):
    labels = spec["source_extensions"]["klue_ner"]["labels"]
    labels["ZZ"] = deepcopy(labels["TI"])
    with pytest.raises(OntologyError, match="'ZZ' is not recorded as a recovered label"):
        validate_spec(spec)


def test_the_declared_registry_equals_the_recovered_set_for_every_schema(ontology):
    artifact = json.loads(
        (
            REPO_ROOT / "research/pii/frontier/evidence/ontology-v2-source-extension-evidence-v1.json"
        ).read_text(encoding="utf-8")
    )
    for schema, entry in artifact["sources"].items():
        if entry["kind"] == "onboarding_manifest":
            recovered = set(entry["ignored_source_labels"])
        else:
            recovered = set(entry["undeclared_labels"])
        assert set(ontology.source_extension_labels(schema)) == recovered, schema


def test_a_raw_scan_undeclared_label_must_be_in_its_observed_inventory(spec):
    artifact = json.loads(
        (
            REPO_ROOT / "research/pii/frontier/evidence/ontology-v2-source-extension-evidence-v1.json"
        ).read_text(encoding="utf-8")
    )
    artifact["sources"]["multigrascco"]["observed_labels"] = ["DATE"]
    forged = REPO_ROOT / "research/pii/frontier/evidence/forged-extension-evidence-test.json"
    forged.write_text(json.dumps(artifact), encoding="utf-8")
    try:
        spec["source_extensions"]["multigrascco"]["provenance"]["evidence_artifact"] = str(
            forged.relative_to(REPO_ROOT)
        )
        with pytest.raises(OntologyError, match="absent from the scan's observed inventory"):
            validate_spec(spec)
    finally:
        forged.unlink()


def test_extension_counts_that_disagree_with_the_evidence_artifact_are_rejected(spec):
    entry = extension(spec)
    entry["evidence"] = {"spans": 267, "by_split": {"test": 40, "train": 197, "validation": 30}}
    with pytest.raises(OntologyError, match="counts disagree with the evidence artifact"):
        validate_spec(spec)


def test_extension_counts_must_sum_to_their_splits(spec):
    extension(spec)["evidence"]["spans"] = 265
    with pytest.raises(OntologyError, match="does not equal the sum of by_split"):
        validate_spec(spec)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("route", "handwaving", "route must be one of"),
        ("promotion", "definitely_fine", "promotion must be one of"),
        ("reason", "  ", "needs a stated reason"),
        ("fallback", "not_a_class", "fallback 'not_a_class' is not in its acceptable set"),
    ],
)
def test_extension_field_vocabularies_are_closed(spec, field, value, message):
    extension(spec)[field] = value
    with pytest.raises(OntologyError, match=message):
        validate_spec(spec)


def test_an_extension_with_an_unknown_field_is_rejected(spec):
    extension(spec)["note"] = "typo for reason"
    with pytest.raises(OntologyError, match="unknown fields: note"):
        validate_spec(spec)


@pytest.mark.parametrize(
    "declared,message",
    [
        ("tasks/ignored-gold-label-audit.json", "resolves into the private tasks/ tree"),
        ("data/../tasks/ignored-gold-label-audit.json", "resolves into the private tasks/ tree"),
        ("scripts/x/../../tasks/audit.json", "resolves into the private tasks/ tree"),
        ("/etc/passwd", "must be repository-relative, not absolute"),
        (str(REPO_ROOT / "tasks" / "audit.json"), "must be repository-relative, not absolute"),
        ("../outside-the-repo/manifest.json", "resolves outside the repository"),
        ("data/../../escape/manifest.json", "resolves outside the repository"),
        ("", "must be a nonempty string"),
        ("   ", "must be a nonempty string"),
        (17, "must be a nonempty string"),
    ],
)
def test_a_declared_artifact_path_must_stay_inside_the_repository(spec, declared, message):
    spec["source_extensions"]["hiner_original"]["provenance"]["manifest"] = declared
    with pytest.raises(OntologyError, match=re.escape(message)):
        validate_spec(spec)


def test_the_evidence_artifact_path_is_guarded_the_same_way(spec):
    spec["source_extensions"]["multigrascco"]["provenance"]["evidence_artifact"] = (
        "research/../tasks/ignored-gold-label-audit.json"
    )
    with pytest.raises(OntologyError, match="resolves into the private tasks/ tree"):
        validate_spec(spec)


def test_an_ordinary_tracked_path_still_resolves(spec):
    # The guard must not be so tight that a normal declaration stops working.
    spec["source_extensions"]["hiner_original"]["provenance"]["manifest"] = (
        "data/pii-onboarded/./hiner/manifest.json"
    )
    with pytest.raises(OntologyError, match="manifest path disagrees with the evidence artifact"):
        validate_spec(spec)
    assert load_ontology().source_extension_labels("hiner_original")


def test_a_manifest_whose_source_schema_disagrees_is_rejected(spec):
    spec["source_extensions"]["hiner_original"]["provenance"]["manifest"] = (
        "data/pii-onboarded/klue-ner/manifest.json"
    )
    with pytest.raises(OntologyError, match="declares source_schema 'klue_ner', not"):
        validate_spec(spec)


def test_a_provenance_kind_outside_the_vocabulary_is_rejected(spec):
    spec["source_extensions"]["hiner_original"]["provenance"]["kind"] = "vibes"
    with pytest.raises(OntologyError, match="kind must be one of"):
        validate_spec(spec)


def test_no_declared_artifact_path_points_into_the_private_task_tree(spec):
    # Prose may name the rejected tree; a declared path value may not be in it.
    declared = []
    for entry in spec["source_extensions"].values():
        provenance = entry["provenance"]
        declared += [provenance.get("manifest"), provenance.get("evidence_artifact")]
    declared = [str(value) for value in declared if value is not None]
    assert declared
    for value in declared:
        resolved = (REPO_ROOT / value).resolve()
        assert resolved.is_relative_to(REPO_ROOT.resolve()), value
        assert not resolved.is_relative_to(REPO_ROOT.resolve() / "tasks"), value


def test_the_evidence_artifact_names_no_private_task_path():
    path = REPO_ROOT / "research/pii/frontier/evidence/ontology-v2-source-extension-evidence-v1.json"
    assert "tasks/" not in path.read_text(encoding="utf-8")


def test_the_evidence_artifact_carries_counts_but_no_surface_text():
    artifact = json.loads(
        (
            REPO_ROOT / "research/pii/frontier/evidence/ontology-v2-source-extension-evidence-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["redaction"] == "counts_and_label_names_only"
    for schema, entry in artifact["sources"].items():
        assert entry["ignored_label_counts"], schema
        for label, counts in entry["ignored_label_counts"].items():
            assert counts["spans"] == sum(counts["by_split"].values()), f"{schema}.{label}"
        assert not {"surfaces", "examples", "text", "spans_text"} & set(entry), schema


# ---- composition contracts and source metadata ---------------------------


def test_composition_contracts_never_create_a_primary_or_predicate(ontology):
    assert ontology.composition_contracts
    for name, entry in ontology.composition_contracts.items():
        assert entry["creates_new_primary_or_predicate"] is False, name
    assert len(ontology.primary_types) == PRIMARY_COUNT
    assert tuple(sorted(ontology.attr_channels)) == ATTR_CHANNELS


def test_the_three_structure_only_composition_contracts_are_recorded(ontology):
    contracts = ontology.composition_contracts
    assert contracts["multigrascco_name_title"]["without_structure"] == "independent_review"
    assert contracts["wojood_nested_amount_components"]["labels"] == ["CURR", "UNIT"]
    mapa = contracts["mapa_fine_person_components"]
    assert mapa["fine_title_and_role_stay_inside_person_span"] is True
    assert mapa["admit_fine_layer_only_siblings_wholesale"] is False
    assert "205541" in mapa["provenance_warning"] and "205542" in mapa["provenance_warning"]


def test_a_composition_contract_that_claims_a_new_head_is_rejected(spec):
    spec["composition_contracts"]["multigrascco_name_title"]["creates_new_primary_or_predicate"] = True
    with pytest.raises(OntologyError, match="may never create a primary class or predicate"):
        validate_spec(spec)


def test_a_composition_contract_on_an_unknown_label_is_rejected(spec):
    spec["composition_contracts"]["multigrascco_name_title"]["label"] = "NOT_A_LABEL"
    with pytest.raises(OntologyError, match="is neither a source label nor a declared extension"):
        validate_spec(spec)


def test_tab_confidential_status_is_provenance_not_a_third_predicate(ontology):
    channel = ontology.source_metadata_channels["tab_8"]["confidential_status"]
    assert channel["creates_predicate_channel"] is False
    assert "confidential_status" not in ontology.attr_channels
    assert tuple(sorted(ontology.attr_channels)) == ATTR_CHANNELS
    discriminator = channel["deterministic_discriminator"]["DEM"]
    assert discriminator["HEALTH"] == "health_condition"
    assert set(discriminator[value] for value in ("ETHNIC", "POLITICS", "BELIEF", "SEX")) == {
        "protected_attribute"
    }
    assert set(discriminator.values()) <= set(ontology.source_acceptable("tab_8", "DEM"))


def test_tab_identifier_type_is_recorded_as_provenance(ontology):
    channel = ontology.source_metadata_channels["tab_8"]["identifier_type"]
    assert set(channel["values"]) == {"DIRECT", "NO_MASK", "QUASI"}
    assert channel["creates_predicate_channel"] is False


def test_a_metadata_channel_that_claims_to_be_a_predicate_is_rejected(spec):
    spec["source_metadata_channels"]["tab_8"]["confidential_status"]["creates_predicate_channel"] = True
    with pytest.raises(OntologyError, match="never becomes a predicate channel"):
        validate_spec(spec)


def test_a_discriminator_target_outside_the_acceptable_set_is_rejected(spec):
    spec["source_metadata_channels"]["tab_8"]["confidential_status"]["deterministic_discriminator"]["DEM"][
        "HEALTH"
    ] = "organization"
    with pytest.raises(OntologyError, match="which is outside the declared acceptable set"):
        validate_spec(spec)


def test_a_metadata_channel_may_not_shadow_a_predicate_channel(spec):
    channels = spec["source_metadata_channels"]["tab_8"]
    channels["family_name"] = deepcopy(channels["identifier_type"])
    with pytest.raises(OntologyError, match="may not shadow a predicate channel"):
        validate_spec(spec)


# ---- strict YAML parsing -------------------------------------------------


def test_duplicate_yaml_keys_are_rejected_at_the_top_level():
    with pytest.raises(OntologyError, match=r"^probe\.yaml:2: duplicate YAML key 'version';"):
        parse_yaml("version: 2\nversion: 3\n", source="probe.yaml")


def test_duplicate_yaml_keys_are_rejected_at_any_nesting_depth():
    text = "a:\n  b:\n    c: 1\n    d: 2\n    c: 3\n"
    with pytest.raises(OntologyError, match=r"^deep\.yaml:5: duplicate YAML key 'c';"):
        parse_yaml(text, source="deep.yaml")


def test_parse_yaml_falls_back_to_its_default_source_label():
    with pytest.raises(OntologyError, match=r"^<string>:2: duplicate YAML key 'a';"):
        parse_yaml("a: 1\na: 2\n")


def test_load_yaml_names_the_file_in_a_duplicate_key_error(tmp_path):
    path = tmp_path / "dupe.yaml"
    path.write_text("outer:\n  inner: 1\n  inner: 2\n", encoding="utf-8")
    with pytest.raises(OntologyError, match=rf"^{re.escape(str(path))}:3: duplicate YAML key 'inner';"):
        load_yaml(path)


def test_a_malformed_file_also_names_its_path(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(OntologyError, match=rf"^{re.escape(str(path))}: invalid YAML:"):
        load_yaml(path)


def test_merge_keys_still_work_and_an_anchor_override_is_not_a_duplicate():
    text = "base: &base\n  a: 1\n  b: 2\nderived:\n  <<: *base\n  b: 3\n"
    assert parse_yaml(text)["derived"] == {"a": 1, "b": 3}


def test_the_committed_spec_and_v1_tagset_both_parse_without_duplicate_keys():
    assert load_yaml(ONTOLOGY_PATH)["version"] == 2
    assert load_yaml(COMPATIBILITY_TAGSET_PATH)["sources"]


def test_the_gold_metadata_contract_declares_each_key_once():
    text = ONTOLOGY_PATH.read_text(encoding="utf-8")
    block = text.split("\ngold_metadata_contract:\n", 1)[1].split("\n\n", 1)[0]
    keys = [line.split(":", 1)[0].strip() for line in block.splitlines() if line.startswith("  ")]
    assert len(keys) == len(set(keys)), keys
