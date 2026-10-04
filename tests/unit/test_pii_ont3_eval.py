from pathlib import Path

import numpy as np
import pytest

from scripts.pii_ont3_eval import (
    _decode_window,
    _load_bcp47_sidecar,
    _load_id_list,
    _normalize_reference_logit_biases,
    _parse_reference_logit_bias,
    _select_rows_by_ids,
    gold_views,
    logit_diagnostic_token,
    score_rows,
    select_decoded_conditioned_predicate_logits,
)
from scripts.pii_subclass import load_subclass_spec

SUBCLASS_SPEC = Path("scripts/pii_subclass_families_v3.json")
REFERENCE_FORM_SPEC = Path("scripts/pii_subclass_families_v4.json")


@pytest.mark.parametrize(
    "label,end,expected_fp",
    [("person_name", 7, 0), ("organization", 7, 1), ("person_name", 6, 1)],
)
def test_optional_reference_is_neutral_but_strict_view_requires_reference(label, end, expected_fp):
    gold = [
        {
            "id": "optional",
            "text": "witness met Anna",
            "spans": [(0, 7, "person_reference"), (12, 16, "person_name")],
        }
    ]
    spans = [{"start": 0, "end": end, "label": label}, {"start": 12, "end": 16, "label": "person_name"}]
    pred = [
        {"id": "optional", "reference_aware_preds": spans, "named_only_preds": spans, "predicate_tokens": []}
    ]
    result = score_rows(gold, pred, [], {})
    for field in ("named_only_primary_exact", "optional_reference_primary_exact"):
        assert {k: result[field][k] for k in ("tp", "fp", "fn")} == {"tp": 1, "fp": expected_fp, "fn": 0}
    assert {k: result["reference_aware_primary_exact"][k] for k in ("tp", "fp", "fn")} == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
    }


def test_bcp47_sidecar_is_canonical_and_line_aligned(tmp_path):
    sidecar = tmp_path / "eval.bcp"
    sidecar.write_text("EN-us\nund\nzh-hant-tw\n", encoding="utf-8")

    assert _load_bcp47_sidecar(sidecar, 3) == ["en-US", "und", "zh-Hant-TW"]

    with pytest.raises(ValueError, match="has 3 rows; expected 2"):
        _load_bcp47_sidecar(sidecar, 2)


def test_bcp47_sidecar_rejects_blank_or_non_bcp47_rows(tmp_path):
    sidecar = tmp_path / "eval.bcp"
    sidecar.write_text("en\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported BCP 47"):
        _load_bcp47_sidecar(sidecar, 2)


def test_id_list_selection_preserves_source_order_and_rejects_bad_membership(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("row-c\nrow-a\n", encoding="utf-8")
    selected_ids = _load_id_list(ids)
    rows = [{"id": "row-a"}, {"id": "row-b"}, {"id": "row-c"}]

    assert _select_rows_by_ids(rows, selected_ids, source="rows.jsonl") == [
        {"id": "row-a"},
        {"id": "row-c"},
    ]

    with pytest.raises(ValueError, match="missing selected ids"):
        _select_rows_by_ids(rows, ["row-z"], source="rows.jsonl")

    ids.write_text("row-a\nrow-a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate row ids"):
        _load_id_list(ids)


def _fixture():
    predicate_names = [
        "given_name",
        "middle_name",
        "family_name",
        "witness_or_bystander",
    ]
    applicable = {
        "given_name": frozenset({"person_name"}),
        "middle_name": frozenset({"person_name"}),
        "family_name": frozenset({"person_name"}),
        "witness_or_bystander": frozenset({"person_name", "person_reference"}),
    }
    gold = {
        "id": "row-1",
        "text": "Anna Smith met witness",
        "base_spans": [{"start": 0, "end": 10, "type": "person_name", "surface": "Anna Smith"}],
        "preds": [
            {"start": 0, "end": 4, "label": "given_name"},
            {"start": 5, "end": 10, "label": "family_name"},
            {"start": 15, "end": 22, "label": "person_reference"},
            {"start": 15, "end": 22, "label": "witness_or_bystander"},
        ],
    }
    prediction = {
        "id": "row-1",
        "reference_aware_preds": [
            {"start": 0, "end": 10, "label": "person_name"},
            {"start": 15, "end": 22, "label": "person_reference"},
        ],
        "named_only_preds": [{"start": 0, "end": 10, "label": "person_name"}],
        "predicate_tokens": [
            {
                "start": 0,
                "end": 4,
                "token_id": 10,
                "active": ["given_name", "witness_or_bystander"],
            },
            {
                "start": 5,
                "end": 10,
                "token_id": 11,
                "active": ["family_name"],
            },
            {
                "start": 15,
                "end": 22,
                "token_id": 12,
                "active": ["witness_or_bystander"],
            },
        ],
    }
    return predicate_names, applicable, gold, prediction


def test_gold_views_replaces_same_boundary_primary_and_keeps_named_only_base():
    predicate_names, _applicable, gold, _prediction = _fixture()
    gold["base_spans"].append({"start": 15, "end": 22, "type": "person_name", "surface": "witness"})

    reference_aware, named_only, predicates = gold_views(gold, predicate_names)

    assert reference_aware == [
        (0, 10, "person_name"),
        (15, 22, "person_reference"),
    ]
    assert named_only == [
        (0, 10, "person_name"),
        (15, 22, "person_name"),
    ]
    assert predicates == [
        (0, 4, "given_name"),
        (5, 10, "family_name"),
        (15, 22, "witness_or_bystander"),
    ]


@pytest.mark.parametrize("predict_reference", [False, True])
def test_score_ignores_conflicting_reference_and_its_supervision(predict_reference):
    gold = {
        "id": "nested",
        "text": "Google's lawyer",
        "spans": [[0, 15, "person_reference"], [0, 6, "organization"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 15,
                "type": "person_reference",
                "attrs": {"legal_professional": [[0, 15]]},
                "objective_weights": {"legal_professional": 1.0},
            }
        ],
    }
    gold["subclass_spans"] = [
        {
            "carrier_start": 0,
            "carrier_end": 15,
            "start": 0,
            "end": 15,
            "type": "person_reference",
            "family": "reference_form",
            "value": "descriptive",
        }
    ]
    entity = {"start": 0, "end": 6, "label": "organization"}
    prediction = {
        "id": "nested",
        "reference_aware_preds": [entity]
        + ([{"start": 0, "end": 15, "label": "person_reference"}] if predict_reference else []),
        "named_only_preds": [entity],
        "predicate_tokens": [],
        "subclass_spans": gold["subclass_spans"],
        "subclass_spans_oracle_carriers": gold["subclass_spans"],
    }
    channels = ["legal_professional"]
    applicable = {"legal_professional": frozenset({"person_reference"})}
    result = score_rows([gold], [prediction], channels, applicable, load_subclass_spec(REFERENCE_FORM_SPEC))
    assert result["reference_aware_primary_exact"]["f1"] == 1.0
    assert result["reference_exact"]["micro"]["gold"] == 0
    assert result["reference_exact"]["micro"]["predicted"] == 0
    assert result["predicate_token_cells"]["known_cells"] == 0
    assert result["subclass_exact"]["known_carrier_families"] == 0
    assert gold_views(gold, channels) == ([(0, 6, "organization")], [(0, 6, "organization")], [])
    assert len(gold["spans"]) == 2

    prediction["reference_aware_preds"] = prediction["reference_aware_preds"][1:]
    missing = score_rows([gold], [prediction], channels, applicable)
    assert missing["reference_aware_primary_exact"]["fn"] == 1
    prediction["reference_aware_preds"].append({"start": 7, "end": 13, "label": "organization"})
    wrong = score_rows([gold], [prediction], channels, applicable)
    assert wrong["reference_aware_primary_exact"]["fp"] == 1


def test_legacy_reference_predicate_inside_a_retained_entity_is_ignored():
    gold = [
        {
            "id": "nested",
            "text": "Google's lawyer",
            "base_spans": [{"start": 0, "end": 6, "type": "organization"}],
            "preds": [
                {"start": 0, "end": 15, "label": "person_reference"},
                {"start": 0, "end": 6, "label": "legal_professional"},
            ],
        },
        {
            "id": "isolated",
            "text": "lawyer",
            "base_spans": [],
            "preds": [
                {"start": 0, "end": 6, "label": "person_reference"},
                {"start": 0, "end": 6, "label": "legal_professional"},
            ],
        },
    ]
    predictions = [
        {
            "id": row["id"],
            "reference_aware_preds": [],
            "named_only_preds": [],
            "predicate_tokens": [{"start": 0, "end": 6, "token_id": 1, "active": []}],
        }
        for row in gold
    ]
    result = score_rows(
        gold,
        predictions,
        ["legal_professional"],
        {
            "legal_professional": frozenset({"person_reference"}),
        },
    )
    assert result["predicate_token_cells"]["known_cells"] == 1
    assert result["predicate_exact"]["micro"]["gold"] == 1


def test_score_rows_separates_primary_reference_and_predicate_objectives():
    predicate_names, applicable, gold, prediction = _fixture()

    result = score_rows([gold], [prediction], predicate_names, applicable)

    assert result["reference_exact"]["micro"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "gold": 1,
        "predicted": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert result["named_only_primary_exact"]["f1"] == 1.0
    assert result["predicate_token_cells"]["known_cells"] == 9
    assert result["predicate_token_cells"]["micro"]["tp"] == 3
    assert result["predicate_token_cells"]["micro"]["fp"] == 1
    assert result["predicate_token_cells"]["micro"]["fn"] == 0
    assert result["predicate_exact"]["micro"]["tp"] == 3
    assert result["predicate_exact"]["micro"]["fp"] == 1
    assert result["predicate_exact"]["micro"]["fn"] == 0
    assert result["predicate_token_cells"]["supported_channels"] == [
        "family_name",
        "given_name",
        "middle_name",
        "witness_or_bystander",
    ]


def test_score_rows_excludes_both_conflicting_legacy_reference_carriers():
    predicate_names = ["family_member"]
    applicable = {"family_member": frozenset({"person_reference"})}
    gold = {
        "id": "nested-row",
        "text": "Her child",
        "base_spans": [{"start": 0, "end": 9, "type": "person_name", "surface": "Her child"}],
        "preds": [
            {"start": 0, "end": 3, "label": "person_reference"},
            {"start": 0, "end": 3, "label": "family_member"},
            {"start": 0, "end": 9, "label": "person_reference"},
            {"start": 0, "end": 9, "label": "family_member"},
        ],
    }
    prediction = {
        "id": "nested-row",
        "reference_aware_preds": [
            {"start": 0, "end": 3, "label": "person_reference"},
            {"start": 0, "end": 9, "label": "person_reference"},
        ],
        "named_only_preds": [
            {"start": 0, "end": 9, "label": "person_name"},
        ],
        "predicate_tokens": [
            {"start": 0, "end": 3, "token_id": 10, "active": ["family_member"]},
            {"start": 4, "end": 9, "token_id": 11, "active": ["family_member"]},
        ],
    }

    result = score_rows([gold], [prediction], predicate_names, applicable)

    assert result["reference_exact"]["micro"]["gold"] == 0
    assert result["reference_exact"]["micro"]["predicted"] == 0
    assert result["predicate_token_cells"]["known_cells"] == 0
    assert result["predicate_exact"]["micro"]["gold"] == 0
    assert result["reference_overlap_projection"]["excluded_targets"] == 2
    assert result["named_only_primary_exact"]["f1"] == 1.0


def test_conditioned_predicates_follow_projected_semantic_type_and_gate_o():
    logits = np.asarray(
        [
            [[1.0, 2.0], [11.0, 12.0]],
            [[3.0, 4.0], [13.0, 14.0]],
            [[5.0, 6.0], [15.0, 16.0]],
            [[7.0, 8.0], [17.0, 18.0]],
        ]
    )

    selected = select_decoded_conditioned_predicate_logits(
        logits,
        ["B-person_name", "E-person_name", "S-person_reference", "O"],
        ["person_name", "person_reference"],
    )

    assert selected[0].tolist() == [1.0, 2.0]
    assert selected[1].tolist() == [3.0, 4.0]
    assert selected[2].tolist() == [15.0, 16.0]
    assert selected[3] is None


def test_reference_logit_penalty_can_restore_o_without_changing_zero_cost():
    logits = np.asarray([[0.0, 1.0]])
    id2label = {0: "O", 1: "S-person_reference"}

    unpenalized, _stats, _labels = _decode_window(logits, [(0, 4)], id2label)
    penalized, _stats, labels = _decode_window(
        logits,
        [(0, 4)],
        id2label,
        penalized_columns=[1],
        column_penalty=2.0,
    )

    assert unpenalized == [{"start": 0, "end": 4, "label": "person_reference"}]
    assert penalized == []
    assert labels == ["O"]


def test_reference_logit_bias_is_type_specific_and_zero_is_exact():
    logits = np.asarray([[1.0, 2.0, 1.75]])
    id2label = {
        0: "O",
        1: "S-person_reference",
        2: "S-organization_reference",
    }

    baseline, _stats, baseline_labels = _decode_window(logits, [(0, 4)], id2label)
    zero, _stats, zero_labels = _decode_window(
        logits,
        [(0, 4)],
        id2label,
        column_biases={1: 0.0, 2: 0.0},
    )
    biased, _stats, biased_labels = _decode_window(
        logits,
        [(0, 4)],
        id2label,
        column_biases={2: 0.5},
    )

    assert zero == baseline
    assert zero_labels == baseline_labels
    assert biased == [{"start": 0, "end": 4, "label": "organization_reference"}]
    assert biased_labels == ["S-organization_reference"]


def test_reference_logit_bias_cli_and_normalization():
    assert _parse_reference_logit_bias("person_reference=1.25") == (
        "person_reference",
        1.25,
    )
    assert _normalize_reference_logit_biases({"person_reference": 1.25}) == {
        "organization_reference": 0.0,
        "person_reference": 1.25,
    }
    with pytest.raises(ValueError, match="unknown reference-logit bias"):
        _normalize_reference_logit_biases({"person": 1.0})


def test_logit_diagnostic_token_keeps_added_heads_and_strongest_incumbent():
    result = logit_diagnostic_token(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([[[5.0, 6.0]], [[7.0, 8.0]]]).reshape(2, 2),
        id2label={
            0: "O",
            1: "S-person_name",
            2: "S-person_reference",
            3: "S-organization_reference",
        },
        predicate_names=["patient", "legal_professional"],
        condition_types=["person_name", "person_reference"],
    )

    assert result == {
        "reference_label_logits": {
            "S-person_reference": 3.0,
            "S-organization_reference": 4.0,
        },
        "best_nonreference": {"label": "S-person_name", "logit": 2.0},
        "nonreference_topk": [
            {"label": "S-person_name", "logit": 2.0},
            {"label": "O", "logit": 1.0},
        ],
        "conditioned_predicate_logits": {
            "person_name": {"patient": 5.0, "legal_professional": 6.0},
            "person_reference": {"patient": 7.0, "legal_professional": 8.0},
        },
    }


def test_materialized_gold_respects_masks_and_scores_categorical_subclasses():
    predicate_names = ["care_provider", "patient"]
    applicable = {
        "care_provider": frozenset({"person_name", "person_reference"}),
        "patient": frozenset({"person_name", "person_reference"}),
    }
    gold = {
        "id": "materialized-row",
        "text": "Anna Smith in Paris",
        "spans": [[0, 10, "person_name"], [14, 19, "locality"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 10,
                "type": "person_name",
                "attrs": {"care_provider": []},
                "objective_weights": {"care_provider": 1.0, "patient": 0.0},
            }
        ],
        "subclass_spans": [
            {
                "carrier_start": 0,
                "carrier_end": 10,
                "type": "person_name",
                "start": 0,
                "end": 4,
                "family": "name_component",
                "value": "given_name",
            },
            {
                "carrier_start": 0,
                "carrier_end": 10,
                "type": "person_name",
                "start": 5,
                "end": 10,
                "family": "name_component",
                "value": "family_name",
            },
        ],
    }
    name_components = gold["subclass_spans"]
    prediction = {
        "id": gold["id"],
        "reference_aware_preds": [
            {"start": 0, "end": 10, "label": "person_name"},
            {"start": 14, "end": 19, "label": "locality"},
        ],
        "named_only_preds": [
            {"start": 0, "end": 10, "label": "person_name"},
            {"start": 14, "end": 19, "label": "locality"},
        ],
        "predicate_tokens": [
            {"start": 0, "end": 4, "token_id": 10, "active": []},
            {"start": 5, "end": 10, "token_id": 11, "active": []},
        ],
        "subclass_spans": [
            *name_components,
            {
                "carrier_start": 14,
                "carrier_end": 19,
                "type": "locality",
                "start": 14,
                "end": 19,
                "family": "place_coarseness",
                "value": "Q",
            },
        ],
        "subclass_spans_oracle_carriers": [
            *name_components,
            {
                "carrier_start": 14,
                "carrier_end": 19,
                "type": "locality",
                "start": 14,
                "end": 19,
                "family": "place_coarseness",
                "value": "Q",
            },
        ],
    }

    result = score_rows(
        [gold],
        [prediction],
        predicate_names,
        applicable,
        load_subclass_spec(SUBCLASS_SPEC),
    )

    assert result["predicate_token_cells"]["supported_channels"] == ["care_provider"]
    assert result["predicate_token_cells"]["known_cells"] == 2
    assert result["subclass_exact"]["supported_families"] == ["name_component"]
    assert result["subclass_exact"]["known_carrier_families"] == 1
    assert result["subclass_exact"]["end_to_end"]["micro"]["f1"] == 1.0
    assert result["subclass_exact"]["oracle_carrier"]["micro"]["f1"] == 1.0


def test_oracle_carrier_subclass_score_separates_head_from_primary_recall():
    spec = load_subclass_spec(SUBCLASS_SPEC)
    component = {
        "carrier_start": 0,
        "carrier_end": 4,
        "type": "person_name",
        "start": 0,
        "end": 4,
        "family": "name_component",
        "value": "given_name",
    }
    gold = {
        "id": "missed-carrier",
        "text": "Anna",
        "spans": [[0, 4, "person_name"]],
        "predicate_spans": [],
        "subclass_spans": [component],
    }
    prediction = {
        "id": gold["id"],
        "reference_aware_preds": [],
        "named_only_preds": [],
        "predicate_tokens": [{"start": 0, "end": 4, "token_id": 10, "active": []}],
        "subclass_spans": [],
        "subclass_spans_oracle_carriers": [component],
    }

    result = score_rows([gold], [prediction], [], {}, spec)

    assert result["subclass_exact"]["end_to_end"]["micro"]["f1"] == 0.0
    assert result["subclass_exact"]["oracle_carrier"]["micro"]["f1"] == 1.0


@pytest.mark.parametrize("reverse", [False, True])
def test_permissive_name_score_credits_split_given_spans_symmetrically(reverse):
    text = "Mary Ann Smith"

    def components(parts):
        return [
            dict(
                carrier_start=0,
                carrier_end=len(text),
                type="person_name",
                start=a,
                end=b,
                family="name_component",
                value=v,
            )
            for a, b, v in parts
        ]

    split = components([(0, 4, "given_name"), (5, 8, "given_name"), (9, 14, "family_name")])
    joined = components([(0, 8, "given_name"), (9, 14, "family_name")])
    gold = dict(
        id="partition",
        text=text,
        spans=[[0, len(text), "person_name"]],
        subclass_spans=joined if reverse else split,
    )
    prediction = dict(
        id="partition",
        reference_aware_preds=[dict(start=0, end=len(text), label="person_name")],
        named_only_preds=[dict(start=0, end=len(text), label="person_name")],
        predicate_tokens=[],
        subclass_spans=split if reverse else joined,
        subclass_spans_oracle_carriers=split if reverse else joined,
    )
    result = score_rows([gold], [prediction], [], {}, load_subclass_spec(SUBCLASS_SPEC))
    assert result["subclass_exact"]["end_to_end"]["micro"]["f1"] < 1
    assert result["name_component_permissive"]["end_to_end"]["f1"] == 1
    assert result["name_component_permissive"]["oracle_carrier"]["f1"] == 1
    prediction["subclass_spans"][0] = {**prediction["subclass_spans"][0], "value": "family_name"}
    assert (
        score_rows([gold], [prediction], [], {}, load_subclass_spec(SUBCLASS_SPEC))[
            "name_component_permissive"
        ]["end_to_end"]["f1"]
        < 1
    )


def test_permissive_name_projection_keeps_material_and_label_barriers():
    from scripts.pii_subclass import permissive_name_components

    spans = [(0, 4, "given_name"), (6, 9, "given_name")]
    assert permissive_name_components("Mary, Ann", spans) == set(spans)
    assert permissive_name_components("MaryX Ann", spans) == set(spans)
    spans = [(0, 4, "given_name"), (5, 8, "middle_name"), (9, 12, "given_name")]
    assert permissive_name_components("Mary Sue Ann", spans) == set(spans)
    overlapping = [(0, 6, "given_name"), (5, 8, "given_name")]
    assert permissive_name_components("Mary Ann", overlapping) == set(overlapping)
    conflicting = [(0, 1, "family_name"), (0, 2, "given_name")]
    assert permissive_name_components("琪琪", conflicting) == set(conflicting)


def test_missing_gold_subclasses_are_unknown_not_negative():
    gold = {
        "id": "pre-subclass-pilot",
        "text": "Anna",
        "spans": [[0, 4, "person_name"]],
        "predicate_spans": [],
    }
    prediction = {
        "id": gold["id"],
        "reference_aware_preds": [{"start": 0, "end": 4, "label": "person_name"}],
        "named_only_preds": [{"start": 0, "end": 4, "label": "person_name"}],
        "predicate_tokens": [{"start": 0, "end": 4, "token_id": 10, "active": []}],
        "subclass_spans": [],
        "subclass_spans_oracle_carriers": [],
    }

    result = score_rows(
        [gold],
        [prediction],
        [],
        {},
        load_subclass_spec(SUBCLASS_SPEC),
    )

    assert result["subclass_exact"]["known_carrier_families"] == 0
    assert result["subclass_exact"]["supported_families"] == []


def test_routine_projection_excludes_reviewed_bare_pronoun_and_attached_predicates():
    gold = {
        "id": "bare-pronoun",
        "text": "He spoke.",
        "spans": [[0, 2, "person_reference"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 2,
                "type": "person_reference",
                "attrs": {"witness_or_bystander": [[0, 2]]},
                "objective_weights": {"witness_or_bystander": 1.0},
            }
        ],
        "reference_form_spans": [
            {
                "carrier_start": 0,
                "carrier_end": 2,
                "type": "person_reference",
                "start": 0,
                "end": 2,
                "family": "reference_form",
                "value": "bare_pronoun",
            }
        ],
    }
    prediction = {
        "id": "bare-pronoun",
        "reference_aware_preds": [{"start": 0, "end": 2, "label": "person_reference"}],
        "named_only_preds": [],
        "predicate_tokens": [{"start": 0, "end": 2, "token_id": 10, "active": ["witness_or_bystander"]}],
        "subclass_spans": [],
        "subclass_spans_oracle_carriers": [],
    }
    predicate_names = ["witness_or_bystander"]
    applicable = {"witness_or_bystander": frozenset({"person_reference"})}
    spec = load_subclass_spec(REFERENCE_FORM_SPEC)

    all_references = score_rows([gold], [prediction], predicate_names, applicable, spec)
    routine = score_rows(
        [gold],
        [prediction],
        predicate_names,
        applicable,
        spec,
        routine_reference_projection=True,
    )

    assert all_references["reference_exact"]["micro"]["tp"] == 1
    assert routine["reference_exact"]["micro"]["gold"] == 0
    assert routine["reference_exact"]["micro"]["predicted"] == 0
    assert routine["predicate_token_cells"]["supported_channels"] == []
    assert routine["predicate_token_cells"]["known_cells"] == 0
    assert routine["reference_form_projection"] == {
        "mode": "routine_non_bare",
        "excluded_values": ["bare_pronoun"],
        "excluded_carriers": 1,
        "excluded_carriers_by_form": {"bare_pronoun": 1},
    }


def test_routine_projection_requires_complete_reviewed_reference_forms():
    predicate_names, applicable, gold, prediction = _fixture()
    gold["reference_form_spans"] = []

    with pytest.raises(ValueError, match="incomplete reference_form coverage"):
        score_rows(
            [gold],
            [prediction],
            predicate_names,
            applicable,
            load_subclass_spec(REFERENCE_FORM_SPEC),
            routine_reference_projection=True,
        )


def test_score_rows_breaks_the_primary_objective_down_by_type():
    """A single micro F1 cannot say whether an arm closed a deficit on one type.

    The data-admission and surface-realization arms are aimed at organization and
    date specifically, so the score has to report those separately or the arms
    cannot be read at all.
    """
    predicate_names, applicable, gold, prediction = _fixture()

    result = score_rows([gold], [prediction], predicate_names, applicable)
    by_type = result["primary_exact_by_type"]

    # The fixture's reference-aware view holds one person_name and one
    # person_reference, and each type is counted only against its own gold.
    assert set(by_type) == {"person_name", "person_reference"}
    assert by_type["person_name"]["gold"] == 1
    assert by_type["person_reference"]["gold"] == 1
    assert (
        sum(entry["gold"] for entry in by_type.values()) == (result["reference_aware_primary_exact"]["gold"])
    )
    assert sum(entry["tp"] for entry in by_type.values()) == (result["reference_aware_primary_exact"]["tp"])
