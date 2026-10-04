"""What happens at the step the old ontology's head loses its authority.

A dual-head run is two phases with a boundary in the middle: before it, two
heads and a moving blend; after it, one head over the new ontology. Three
things have to be true at that boundary, and none of them shows up in a loss
curve if it is wrong:

* the objective does not move -- a zero-weight head was already contributing no
  gradient, so retiring it must be free rather than a silent change of target;
* the retired head is really gone -- no forward pass, no optimizer state, and
  no rows in anything saved afterwards, which is what makes the post-fade
  checkpoint a different kind of artifact from the pre-fade ones; and
* the surviving parameters keep their optimizer state, so the second phase
  continues rather than restarting its moments.

The learning-rate restart is tested here too, because it is the other half of
"the run changes phase": without it the new head trains alone on the decayed
tail of a schedule chosen for the whole run.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import AutoModel, BertConfig, TrainingArguments, get_scheduler

from scripts.pii_dual_head import (
    MAP_EXTENSION_SCHEMA,
    MAP_SCHEMA,
    ONTOLOGY_EXTENSION_SCHEMA,
    V1_TO_V2,
    V2_TO_V1,
    bioes_structure_cost_matrix,
    build_affine_projection,
    build_correctness_map,
    continuation_label_mask,
    dual_head_loss,
    head_entity_labels,
    head_loss,
    head_loss_terms,
    initialize_affine_classifier,
    load_correctness_map,
    mapped_structure_cost_rows,
    new_space_labels,
    parse_transition_schedule,
    project_affine_parameters,
    span_risk_gate,
    transition_completion_step,
    transition_weight,
)
from scripts.pii_encoder_train import (
    BOUNDARY2ID,
    DualHeadLossMixin,
    SpanDataset,
    annotated_boundary_loss,
    build_dual_head_span_metric_report,
    entity_dice_loss,
    entity_presence_loss,
    expand_mapped_single_head_successor,
    nonnegative_pu_entity_loss,
    partial_o_loss,
    phase_restart_scheduler,
    resume_extension_restart_step,
    validate_dual_head_resume,
    validate_mapped_single_head_checkpoint,
)
from scripts.pii_layered_head_model import (
    CONCAT_HEAD_ARCHITECTURE,
    LayerConcatForTokenClassification,
)
from scripts.pii_projector import Tagset

OLD_LABELS = new_space_labels(["city", "contact"])
NEW_LABELS = new_space_labels(["email", "locality", "phone_number"])

MAP_DOCUMENT = {
    "schema": MAP_SCHEMA,
    "ontology": {"sha256": "test", "primary_types": ["email", "locality", "phone_number"]},
    "v1_to_v2": {
        "city": {
            "kind": "canonical",
            "fallback": "locality",
            "accepted": ["locality"],
            "origin": {"locality": "definitional"},
        },
        "contact": {
            "kind": "canonical",
            "fallback": "O",
            "accepted": ["O", "email", "phone_number"],
            "origin": {"email": "definitional", "phone_number": "definitional", "O": "definitional"},
        },
        "fixture.X": {
            "kind": "source",
            "fallback": "email",
            "accepted": ["email"],
            "origin": {"email": "test"},
        },
    },
    "v2_to_v1": {},
    "v1_to_v2_source_receipt": "test-only",
}


def correctness():
    return build_correctness_map(MAP_DOCUMENT, OLD_LABELS, NEW_LABELS, map_path="<test>")


def successor_correctness():
    parent = correctness()
    document = json.loads(json.dumps(MAP_DOCUMENT))
    document["ontology"]["primary_types"] = [
        *MAP_DOCUMENT["ontology"]["primary_types"],
        "person_reference",
        "organization_reference",
    ]
    document["legacy_outside_unknown_primary_types"] = [
        "person_reference",
        "organization_reference",
    ]
    document["_artifact_sha256"] = "successor-map"
    document["_parent_map_sha256"] = parent.map_sha256
    document["_parent_ontology_sha256"] = parent.ontology_sha256
    document["_parent_primary_types"] = MAP_DOCUMENT["ontology"]["primary_types"]
    labels = new_space_labels(document["ontology"]["primary_types"])
    return build_correctness_map(document, OLD_LABELS, labels, map_path="<successor-test>")


class ProjectionTagset:
    sources = {"fixture": {"X": "city"}}
    cuts = {"redaction_20_v1": {"groups": ["all"]}}

    def cut_targets(self, cut):
        assert cut == "redaction_20_v1"
        return {"all"}

    def project_canonical_cut(self, source, cut):
        assert cut == "redaction_20_v1"
        assert source in {"city", "contact"}
        return "all"


def projection_counts():
    label_payload = json.dumps(tuple(OLD_LABELS), ensure_ascii=False, separators=(",", ":"))
    support = {
        "city": ["email", "locality"],
        "contact": ["O", "email", "phone_number"],
    }
    origins = {
        "city": {
            "email": ["source:fixture.X"],
            "locality": ["canonical"],
        },
        "contact": {
            "O": ["canonical"],
            "email": ["canonical"],
            "phone_number": ["canonical"],
        },
    }
    return {
        "schema": "pii-ontology-v1-v2-joint-counts",
        "schema_version": 1,
        "audit": {"status": "passed"},
        "counting_contract": {"raw_counts": True},
        "hard_map": {
            "sha256": "map",
            "accepted_support": support,
            "accepted_support_origins": origins,
        },
        "v1_head": {
            "labels": OLD_LABELS,
            "class_order": ["city", "contact"],
            "label_order_sha256": hashlib.sha256(label_payload.encode()).hexdigest(),
        },
        "v1_tagset": {"sha256": "tagset"},
        "v2_ontology": {
            "sha256": "test",
            "primary_types": ["email", "locality", "phone_number"],
        },
        "counts": {
            "city": {"email": 3, "locality": 3},
            "contact": {"O": 1, "email": 3, "phone_number": 0},
        },
        "totals": {"counted_spans": 10, "off_support_spans": 0},
    }


def affine_projection(direction=V1_TO_V2, *, counts=True, alpha=0.0):
    return build_affine_projection(
        MAP_DOCUMENT,
        OLD_LABELS,
        NEW_LABELS,
        tagset=cast(Tagset, ProjectionTagset()),
        direction=direction,
        counts_document=projection_counts() if counts else None,
        add_k=1.0,
        alpha=alpha,
        map_sha256="map",
        tagset_sha256="tagset",
    )


def test_no_count_projection_is_uniform_and_preserves_o_and_bioes() -> None:
    projection = affine_projection(counts=False)
    old = {label: index for index, label in enumerate(OLD_LABELS)}
    new = {label: index for index, label in enumerate(NEW_LABELS)}

    assert projection.coefficients[old["O"], new["O"]] == 1
    assert torch.count_nonzero(projection.coefficients[old["O"]]) == 1
    assert projection.coefficients[old["B-contact"], new["O"]] == pytest.approx(1 / 3)
    assert projection.coefficients[old["B-contact"], new["B-email"]] == pytest.approx(1 / 3)
    assert projection.coefficients[old["B-contact"], new["B-phone_number"]] == pytest.approx(1 / 3)
    assert projection.coefficients[old["B-contact"], new["I-email"]] == 0
    torch.testing.assert_close(
        projection.coefficients.sum(dim=1),
        torch.ones(len(OLD_LABELS), dtype=torch.float64),
    )


def test_count_projection_and_p20_blend_use_pseudocount_formula() -> None:
    direct = affine_projection()
    blended = affine_projection(alpha=5.0)
    old = {label: index for index, label in enumerate(OLD_LABELS)}
    new = {label: index for index, label in enumerate(NEW_LABELS)}

    assert direct.coefficients[old["S-contact"], new["S-email"]] == pytest.approx(4 / 7)
    assert direct.coefficients[old["S-contact"], new["O"]] == pytest.approx(2 / 7)
    assert direct.coefficients[old["S-contact"], new["S-phone_number"]] == pytest.approx(1 / 7)
    assert not torch.equal(blended.coefficients, direct.coefficients)
    torch.testing.assert_close(
        blended.coefficients.sum(dim=1),
        torch.ones(len(OLD_LABELS), dtype=torch.float64),
    )


def test_reverse_projection_is_transpose_supported_but_source_normalized() -> None:
    projection = affine_projection(V2_TO_V1)
    source = {label: index for index, label in enumerate(NEW_LABELS)}
    target = {label: index for index, label in enumerate(OLD_LABELS)}

    assert projection.coefficients[source["O"], target["O"]] == 1
    assert projection.coefficients[source["B-email"], target["B-city"]] == pytest.approx(0.5)
    assert projection.coefficients[source["B-email"], target["B-contact"]] == pytest.approx(0.5)
    assert projection.coefficients[source["B-email"], target["O"]] == 0
    torch.testing.assert_close(
        projection.coefficients.sum(dim=1),
        torch.ones(len(NEW_LABELS), dtype=torch.float64),
    )


def test_affine_tensor_initialization_matches_independent_reconstruction() -> None:
    projection = affine_projection(alpha=2.0)
    source = torch.nn.Linear(3, len(OLD_LABELS), bias=True)
    target = torch.nn.Linear(3, len(NEW_LABELS), bias=True)
    with torch.no_grad():
        source.weight.copy_(torch.arange(len(OLD_LABELS) * 3).reshape(-1, 3))
        source.bias.copy_(torch.arange(len(OLD_LABELS)))
    expected_weight = projection.coefficients.T @ source.weight.detach().double()
    expected_bias = projection.coefficients.T @ source.bias.detach().double()

    initialize_affine_classifier(target, source, projection)
    projected_weight, projected_bias = project_affine_parameters(projection, source.weight, source.bias)

    torch.testing.assert_close(target.weight, expected_weight.to(target.weight))
    torch.testing.assert_close(target.bias, expected_bias.to(target.bias))
    torch.testing.assert_close(target.weight, projected_weight)
    torch.testing.assert_close(target.bias, projected_bias)


def test_loaded_correctness_map_binds_artifact_and_ontology_hashes(tmp_path) -> None:
    data = json.dumps(MAP_DOCUMENT, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path = tmp_path / "map.json"
    path.write_bytes(data)

    document = load_correctness_map(path)
    mapping = build_correctness_map(document, OLD_LABELS, NEW_LABELS, map_path=str(path))

    assert mapping.map_sha256 == hashlib.sha256(data).hexdigest()
    assert mapping.ontology_sha256 == "test"
    assert mapping.map_sha256 != mapping.ontology_sha256


def test_successor_overlay_binds_parent_and_makes_predecessor_o_ambiguous(tmp_path) -> None:
    base_data = json.dumps(MAP_DOCUMENT, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    base_path = tmp_path / "base-map.json"
    base_path.write_bytes(base_data)
    extension = {
        "schema": ONTOLOGY_EXTENSION_SCHEMA,
        "schema_version": 1,
        "ontology_version": "test-reference-successor-v1",
        "base_ontology": {"sha256": "test"},
        "append_primary_types": [
            {"name": "person_reference"},
            {"name": "organization_reference"},
        ],
        "legacy_outside_unknown_primary_types": [
            "person_reference",
            "organization_reference",
        ],
    }
    extension_data = json.dumps(extension, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    extension_path = tmp_path / "extension.json"
    extension_path.write_bytes(extension_data)
    overlay = {
        "schema": MAP_EXTENSION_SCHEMA,
        "schema_version": 1,
        "base_map": {
            "path": base_path.name,
            "sha256": hashlib.sha256(base_data).hexdigest(),
        },
        "ontology_extension": {
            "path": extension_path.name,
            "sha256": hashlib.sha256(extension_data).hexdigest(),
        },
    }
    overlay_data = json.dumps(overlay, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_bytes(overlay_data)

    document = load_correctness_map(overlay_path)
    successor_labels = new_space_labels(document["ontology"]["primary_types"])
    mapping = build_correctness_map(document, OLD_LABELS, successor_labels, map_path=str(overlay_path))
    allowed = {
        successor_labels[index]
        for index in mapping.old_to_new[mapping.old_outside_id].nonzero().flatten().tolist()
    }

    assert mapping.map_sha256 == hashlib.sha256(overlay_data).hexdigest()
    assert mapping.parent_map_sha256 == hashlib.sha256(base_data).hexdigest()
    assert mapping.parent_ontology_sha256 == "test"
    assert mapping.parent_new_labels == tuple(NEW_LABELS)
    assert allowed == {
        "O",
        *{f"{prefix}-person_reference" for prefix in "BIES"},
        *{f"{prefix}-organization_reference" for prefix in "BIES"},
    }


class Tokenizer:
    """Four tokens: [CLS], the two words of ``Ada Diaz``, [SEP]."""

    def __call__(self, text, **kwargs):
        assert text == "Ada Diaz"
        return {
            "input_ids": [0, 1, 2, 3],
            "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
        }


def test_a_batch_routes_old_new_and_annotated_only_rows_without_inventing_o() -> None:
    old_ids = {label: index for index, label in enumerate(OLD_LABELS)}
    new_ids = {label: index for index, label in enumerate(NEW_LABELS)}
    rows = [
        {"text": "Ada Diaz", "spans": [[0, 3, "city"]], "label_space": "v1"},
        {"text": "Ada Diaz", "spans": [[0, 3, "locality"]], "label_space": "v2"},
        {
            "text": "Ada Diaz",
            "spans": [[0, 3, "contact"]],
            "label_space": "v1",
            "supervision": "annotated_spans_only",
        },
    ]
    items = SpanDataset(
        rows,
        Tokenizer(),
        old_ids,
        max_len=32,
        secondary_label2id=new_ids,
    )

    old_row, new_row, teacher_row = (items[index] for index in range(len(rows)))
    assert old_row["labels"] == [-100, old_ids["S-city"], old_ids["O"], -100]
    assert old_row["secondary_labels"] == [-100, -100, new_ids["O"], -100]
    assert new_row["labels"] == [-100, -100, old_ids["O"], -100]
    assert new_row["secondary_labels"] == [-100, new_ids["S-locality"], new_ids["O"], -100]
    assert teacher_row["labels"] == [-100, old_ids["S-contact"], -100, -100]
    assert teacher_row["secondary_labels"] == [-100, -100, -100, -100]

    # The new head sees old-space entities through the allowed-set map, its own
    # labels directly, and no background target at all from annotated-only rows.
    old_targets = torch.tensor([item["labels"] for item in (old_row, new_row, teacher_row)])
    new_targets = torch.tensor([item["secondary_labels"] for item in (old_row, new_row, teacher_row)])
    logits = torch.zeros(3, 4, len(NEW_LABELS))
    _, covered = head_loss(logits, new_targets, old_targets, correctness().old_to_new)
    assert covered == 5


@pytest.mark.parametrize("label_space,span_type", [("v1", "city"), ("v2", "locality")])
def test_predecessor_complete_rows_mask_successor_reference_negatives(
    label_space,
    span_type,
) -> None:
    mapping = successor_correctness()
    old_ids = {label: index for index, label in enumerate(OLD_LABELS)}
    new_ids = {label: index for index, label in enumerate(mapping.new_labels)}
    row = {
        "text": "Ada Diaz",
        "spans": [[0, 3, span_type]],
        "label_space": label_space,
        "unknown_primary_types": ["person_reference", "organization_reference"],
    }
    item = SpanDataset(
        [row],
        Tokenizer(),
        old_ids,
        max_len=32,
        secondary_label2id=new_ids,
        include_complete_presence_labels=True,
        legacy_outside_unknown_primary_types=mapping.legacy_outside_unknown_primary_types,
    )[0]

    assert item["labels"][2] == old_ids["O"]
    assert item["secondary_labels"][2] == -100
    assert item["complete_presence_labels"][2] == -100
    mapped_targets = head_entity_labels(
        torch.tensor([item["secondary_labels"]]),
        torch.tensor([item["labels"]]),
        own_o_label_id=mapping.new_outside_id,
        cross_o_label_id=mapping.old_outside_id,
        cross_membership=mapping.old_to_new,
    )
    assert mapped_targets[0, 2] == -100


def test_predecessor_unknown_types_must_exactly_match_successor_map() -> None:
    mapping = successor_correctness()
    row = {
        "text": "Ada Diaz",
        "spans": [[0, 3, "city"]],
        "unknown_primary_types": ["person_reference"],
    }
    dataset = SpanDataset(
        [row],
        Tokenizer(),
        {label: index for index, label in enumerate(OLD_LABELS)},
        max_len=32,
        secondary_label2id={label: index for index, label in enumerate(mapping.new_labels)},
        legacy_outside_unknown_primary_types=mapping.legacy_outside_unknown_primary_types,
    )

    with pytest.raises(ValueError, match="must exactly match"):
        dataset[0]


def test_v1_only_validation_uses_mapped_loss_without_hard_v2_span_metrics() -> None:
    mapping = correctness()

    assert build_dual_head_span_metric_report(mapping, [{"label_space": "v1"}, {}]) is None
    assert (
        build_dual_head_span_metric_report(
            mapping,
            [{"label_space": "v1"}, {"label_space": "v2"}],
        )
        is not None
    )


def two_head_model():
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=len(OLD_LABELS),
        id2label=dict(enumerate(OLD_LABELS)),
        label2id={label: index for index, label in enumerate(OLD_LABELS)},
    )
    config.architectures = [LayerConcatForTokenClassification.__name__]
    config.pii_head_architecture = CONCAT_HEAD_ARCHITECTURE
    config.pii_encoder_layers = [2, 4]
    config.pii_token_offsets = [0]
    config.pii_head_kind = "affine"
    config.pii_head_rank = 3
    config.pii_classifier_dropout = 0.0
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config))
    model.attach_secondary_head(NEW_LABELS)
    return model


def batch():
    return {
        "input_ids": torch.tensor([[1, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def test_retiring_the_old_head_leaves_a_single_head_new_ontology_tagger() -> None:
    model = two_head_model()
    promoted = model.secondary_classifier

    retired = model.retire_primary_head()

    assert model.secondary_classifier is None
    assert model.classifier is promoted, "the new head takes over the tagger's own output"
    assert model.config.num_labels == len(NEW_LABELS)
    assert model.num_labels == len(NEW_LABELS)
    assert model.config.id2label == dict(enumerate(NEW_LABELS))
    assert model.config.pii_secondary_labels is None
    assert len(retired) == 2, "the old head's weight and bias are handed back to the caller"

    saved = model.state_dict()
    assert not [key for key in saved if key.startswith("secondary_classifier")], (
        "a checkpoint written after the transition must carry no slot for the retired head"
    )
    assert saved["classifier.weight"].shape[0] == len(NEW_LABELS)

    output = model(**batch())
    assert output.logits.shape == (1, 4, len(NEW_LABELS))
    assert getattr(output, "secondary_logits", None) is None, "the retired head is not computed"


def stamp_dual_head_resume_identity(model, mapping) -> None:
    model.config.pii_dual_head_map_sha256 = mapping.map_sha256
    model.config.pii_dual_head_ontology_sha256 = mapping.ontology_sha256
    model.config.pii_dual_head_old_weight_schedule = "constant:0.0"
    model.config.pii_dual_head_init = "fallback"
    model.config.pii_dual_head_eval_weight = 0.0
    model.config.pii_dual_head_retirement_step = 0
    model.config.pii_dual_head_lr_restart_step = 0
    model.config.pii_dual_head_horizon = 1500
    model.config.pii_o_token_loss_weight = 1.0
    model.config.pii_entity_dice_loss_weight = 0.0
    model.config.pii_partial_o_loss_weight = 0.0
    model.config.pii_partial_expected_entity_ratio_lower_width = 0.1
    model.config.pii_complete_presence_loss_weight = 0.0
    model.config.pii_complete_boundary_loss_weight = 0.0


def test_retired_checkpoint_resumes_as_its_exact_new_ontology_shape(tmp_path) -> None:
    mapping = correctness()
    model = two_head_model()
    stamp_dual_head_resume_identity(model, mapping)
    model.retire_primary_head()
    model.config.pii_dual_head_retired_at_step = 0
    model.save_pretrained(tmp_path)

    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path)

    assert restored.classifier.out_features == len(NEW_LABELS)
    assert validate_dual_head_resume(
        restored,
        mapping,
        OLD_LABELS,
        NEW_LABELS,
        old_weight_schedule="constant:0.0",
        initialization="fallback",
        eval_weight=0.0,
        retirement_step=0,
        lr_restart_step=0,
        horizon=1500,
    )


def test_retired_head_starts_a_new_mapped_supervision_stage_without_reattachment() -> None:
    mapping = correctness()
    model = two_head_model()
    stamp_dual_head_resume_identity(model, mapping)
    model.retire_primary_head()
    model.config.pii_dual_head_retired_at_step = 0
    classifier = model.classifier

    validate_mapped_single_head_checkpoint(
        model,
        mapping,
        NEW_LABELS,
        exact_resume=False,
        old_weight_schedule="constant:0.0",
        eval_weight=0.0,
        retirement_step=0,
        horizon=2000,
    )

    assert model.classifier is classifier
    assert model.secondary_classifier is None
    assert model.config.id2label == dict(enumerate(NEW_LABELS))


def test_native_head_explicitly_binds_a_map_without_changing_weights() -> None:
    mapping = correctness()
    model = two_head_model()
    model.retire_primary_head()
    model.config.pii_native_new_label_space = True
    before = {name: value.clone() for name, value in model.state_dict().items()}
    kwargs = dict(
        exact_resume=False,
        old_weight_schedule="constant:0.0",
        eval_weight=0.0,
        retirement_step=0,
        horizon=2000,
    )
    with pytest.raises(ValueError, match="old head is retired"):
        validate_mapped_single_head_checkpoint(model, mapping, NEW_LABELS, **kwargs)
    validate_mapped_single_head_checkpoint(model, mapping, NEW_LABELS, bind_native_map=True, **kwargs)
    assert model.config.pii_mapped_single_head_origin == "native"
    assert getattr(model.config, "pii_dual_head_retired_at_step", None) is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="unmapped native checkpoint"):
        validate_mapped_single_head_checkpoint(model, mapping, NEW_LABELS, bind_native_map=True, **kwargs)


def test_native_binding_rejects_a_different_output_inventory() -> None:
    model = two_head_model()
    model.retire_primary_head()
    model.config.pii_native_new_label_space = True
    with pytest.raises(ValueError, match="requested new-ontology labels"):
        validate_mapped_single_head_checkpoint(
            model,
            correctness(),
            list(reversed(NEW_LABELS)),
            exact_resume=False,
            old_weight_schedule="constant:0.0",
            eval_weight=0.0,
            retirement_step=0,
            horizon=2000,
            bind_native_map=True,
        )
    assert getattr(model.config, "pii_mapped_single_head_origin", None) is None


def test_successor_expansion_exactly_preserves_parent_and_predicate_rows(tmp_path) -> None:
    parent_mapping = correctness()
    successor_mapping = successor_correctness()
    model = two_head_model()
    stamp_dual_head_resume_identity(model, parent_mapping)
    model.retire_primary_head()
    model.config.pii_dual_head_retired_at_step = 0
    model.attach_predicate_head(["care_provider"])
    parent_weight = model.classifier.weight.detach().clone()
    parent_bias = model.classifier.bias.detach().clone()
    predicate_weight = model.predicate_classifier.weight.detach().clone()
    predicate_bias = model.predicate_classifier.bias.detach().clone()

    copied = expand_mapped_single_head_successor(
        model,
        successor_mapping,
        successor_mapping.new_labels,
    )

    assert copied == len(NEW_LABELS)
    assert model.classifier.out_features == len(successor_mapping.new_labels)
    torch.testing.assert_close(model.classifier.weight[:copied], parent_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.classifier.bias[:copied], parent_bias, rtol=0, atol=0)
    torch.testing.assert_close(model.predicate_classifier.weight, predicate_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.predicate_classifier.bias, predicate_bias, rtol=0, atol=0)
    assert model.config.pii_successor_parent_output_rows == len(NEW_LABELS)
    assert model.config.pii_successor_appended_output_rows == 8

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path)
    torch.testing.assert_close(restored.classifier.weight[:copied], parent_weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.classifier.bias[:copied], parent_bias, rtol=0, atol=0)
    torch.testing.assert_close(restored.predicate_classifier.weight, predicate_weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.predicate_classifier.bias, predicate_bias, rtol=0, atol=0)


def test_mapped_single_head_resume_binds_its_new_stage_horizon() -> None:
    mapping = correctness()
    model = two_head_model()
    stamp_dual_head_resume_identity(model, mapping)
    model.retire_primary_head()
    model.config.pii_dual_head_retired_at_step = 0
    model.config.pii_dual_head_mapped_single_head = True

    validate_mapped_single_head_checkpoint(
        model,
        mapping,
        NEW_LABELS,
        exact_resume=True,
        old_weight_schedule="constant:0.0",
        eval_weight=0.0,
        retirement_step=0,
        horizon=1500,
    )
    with pytest.raises(ValueError, match="pii_dual_head_horizon"):
        validate_mapped_single_head_checkpoint(
            model,
            mapping,
            NEW_LABELS,
            exact_resume=True,
            old_weight_schedule="constant:0.0",
            eval_weight=0.0,
            retirement_step=0,
            horizon=1200,
        )
    with pytest.raises(ValueError, match="pii_o_token_loss_weight"):
        validate_mapped_single_head_checkpoint(
            model,
            mapping,
            NEW_LABELS,
            exact_resume=True,
            old_weight_schedule="constant:0.0",
            eval_weight=0.0,
            retirement_step=0,
            horizon=1500,
            o_token_loss_weight=2.0,
        )
    with pytest.raises(ValueError, match="pii_entity_dice_loss_weight"):
        validate_mapped_single_head_checkpoint(
            model,
            mapping,
            NEW_LABELS,
            exact_resume=True,
            old_weight_schedule="constant:0.0",
            eval_weight=0.0,
            retirement_step=0,
            horizon=1500,
            entity_dice_loss_weight=0.1,
        )
    with pytest.raises(ValueError, match="pii_complete_boundary_loss_weight"):
        validate_mapped_single_head_checkpoint(
            model,
            mapping,
            NEW_LABELS,
            exact_resume=True,
            old_weight_schedule="constant:0.0",
            eval_weight=0.0,
            retirement_step=0,
            horizon=1500,
            complete_boundary_loss_weight=0.1,
        )
    with pytest.raises(ValueError, match="--dual-head-mapped-single-head"):
        validate_dual_head_resume(
            model,
            mapping,
            OLD_LABELS,
            NEW_LABELS,
            old_weight_schedule="constant:0.0",
            initialization="fallback",
            eval_weight=0.0,
            retirement_step=0,
            lr_restart_step=0,
            horizon=1500,
        )


def test_mapped_single_head_resume_can_explicitly_extend_its_horizon() -> None:
    mapping = correctness()
    model = two_head_model()
    stamp_dual_head_resume_identity(model, mapping)
    model.retire_primary_head()
    model.config.pii_dual_head_retired_at_step = 0
    model.config.pii_dual_head_mapped_single_head = True

    restart_step = resume_extension_restart_step(model.config, 2500, enabled=True)
    assert restart_step == 1500
    validate_mapped_single_head_checkpoint(
        model,
        mapping,
        NEW_LABELS,
        exact_resume=True,
        old_weight_schedule="constant:0.0",
        eval_weight=0.0,
        retirement_step=0,
        horizon=2500,
        resume_from_horizon=restart_step,
    )

    with pytest.raises(ValueError, match="greater than the checkpoint horizon"):
        resume_extension_restart_step(model.config, 1500, enabled=True)
    with pytest.raises(ValueError, match="pii_dual_head_horizon"):
        validate_mapped_single_head_checkpoint(
            model,
            mapping,
            NEW_LABELS,
            exact_resume=True,
            old_weight_schedule="constant:0.0",
            eval_weight=0.0,
            retirement_step=0,
            horizon=2500,
            resume_from_horizon=1400,
        )


def test_dual_head_resume_rejects_a_changed_trajectory_horizon() -> None:
    mapping = correctness()
    model = two_head_model()
    stamp_dual_head_resume_identity(model, mapping)

    with pytest.raises(ValueError, match="pii_dual_head_horizon"):
        validate_dual_head_resume(
            model,
            mapping,
            OLD_LABELS,
            NEW_LABELS,
            old_weight_schedule="constant:0.0",
            initialization="fallback",
            eval_weight=0.0,
            retirement_step=0,
            lr_restart_step=0,
            horizon=1200,
        )


def test_retiring_frees_the_old_head_and_keeps_the_survivors_optimizer_state() -> None:
    model = two_head_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(**batch())
    (output.logits.sum() + output.secondary_logits.sum()).backward()
    optimizer.step()
    survivor = model.secondary_classifier.weight
    survivor_moment = optimizer.state[survivor]["exp_avg"].clone()
    old_head = list(model.classifier.parameters())
    assert all(parameter in optimizer.state for parameter in old_head)

    retired = model.retire_primary_head()
    dropped = {id(parameter) for parameter in retired}
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in dropped]
    for parameter in retired:
        optimizer.state.pop(parameter, None)

    tracked = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not tracked & dropped, "the retired head must not keep a slot in the optimizer"
    assert id(model.classifier.weight) in tracked, "the promoted head keeps training"
    assert torch.equal(optimizer.state[model.classifier.weight]["exp_avg"], survivor_moment), (
        "the surviving head keeps the moments it accumulated before the transition"
    )
    optimizer.zero_grad()
    model(**batch()).logits.sum().backward()
    optimizer.step()


def test_retirement_does_not_change_the_objective_it_inherits() -> None:
    """At weight zero the blend already *is* the surviving head's loss."""
    mapping = correctness()
    torch.manual_seed(0)
    old_logits = torch.randn(1, 4, len(OLD_LABELS))
    new_logits = torch.randn(1, 4, len(NEW_LABELS))
    s_city = OLD_LABELS.index("S-city")
    locality = NEW_LABELS.index("S-locality")
    old_labels = torch.tensor([[s_city, -100, 0, -100]])
    new_labels = torch.tensor([[-100, locality, 0, -100]])

    blended, _ = dual_head_loss(old_logits, new_logits, old_labels, new_labels, mapping, 0.0)
    total, covered = head_loss(new_logits, new_labels, old_labels, mapping.old_to_new)

    assert covered == 3
    assert torch.allclose(blended, total / covered, atol=1e-6)


def test_retirement_preserves_o_weighted_marginal_objective() -> None:
    mapping = correctness()
    torch.manual_seed(0)
    old_logits = torch.randn(1, 4, len(OLD_LABELS))
    new_logits = torch.randn(1, 4, len(NEW_LABELS))
    old_labels = torch.tensor([[OLD_LABELS.index("S-city"), -100, mapping.old_outside_id, -100]])
    new_labels = torch.tensor([[-100, NEW_LABELS.index("S-locality"), mapping.new_outside_id, -100]])

    blended, telemetry = dual_head_loss(
        old_logits,
        new_logits,
        old_labels,
        new_labels,
        mapping,
        0.0,
        o_token_weight=2.0,
    )
    unweighted, _ = dual_head_loss(
        old_logits,
        new_logits,
        old_labels,
        new_labels,
        mapping,
        0.0,
    )
    terms = head_loss_terms(
        new_logits,
        new_labels,
        old_labels,
        mapping.old_to_new,
        own_o_label_id=mapping.new_outside_id,
        cross_o_label_id=mapping.old_outside_id,
        o_token_weight=2.0,
    )

    assert terms.covered == 3
    assert terms.normalizer == 4
    assert telemetry["dual_new_tokens"] == 3
    assert telemetry["dual_new_loss_normalizer"] == 4
    assert telemetry["dual_o_token_weight"] == 2
    assert not torch.allclose(blended, unweighted)
    assert torch.allclose(blended, terms.total / terms.normalizer, atol=1e-6)


def test_mapped_loss_applies_per_token_primary_objective_weights() -> None:
    mapping = correctness()
    torch.manual_seed(4)
    logits = torch.randn(1, 3, len(NEW_LABELS))
    old_labels = torch.tensor([[OLD_LABELS.index("S-city"), -100, -100]])
    new_labels = torch.tensor([[-100, NEW_LABELS.index("S-locality"), -100]])
    weights = torch.tensor([[0.5, 1.0, 0.0]])

    terms = head_loss_terms(
        logits,
        new_labels,
        old_labels,
        mapping.old_to_new,
        token_objective_weights=weights,
    )
    first = head_loss_terms(
        logits[:, :1],
        new_labels[:, :1],
        old_labels[:, :1],
        mapping.old_to_new,
    )
    second = head_loss_terms(
        logits[:, 1:2],
        new_labels[:, 1:2],
        old_labels[:, 1:2],
        mapping.old_to_new,
    )

    assert terms.covered == 2
    assert terms.normalizer == pytest.approx(1.5)
    torch.testing.assert_close(terms.total, 0.5 * first.total + second.total)

    with pytest.raises(ValueError, match="unsupervised tokens"):
        head_loss_terms(
            logits,
            new_labels,
            old_labels,
            mapping.old_to_new,
            token_objective_weights=torch.tensor([[0.5, 1.0, 0.1]]),
        )


def test_entity_targets_merge_direct_and_mapped_supervision_without_unmasking() -> None:
    mapping = correctness()
    old_labels = torch.tensor([[OLD_LABELS.index("S-city"), mapping.old_outside_id, -100, -100]])
    new_labels = torch.tensor([[-100, -100, NEW_LABELS.index("S-locality"), -100]])

    targets = head_entity_labels(
        new_labels,
        old_labels,
        own_o_label_id=mapping.new_outside_id,
        cross_o_label_id=mapping.old_outside_id,
    )

    assert targets.tolist() == [[1, 0, 1, -100]]


def test_mapped_single_head_applies_entity_dice_to_its_product_logits() -> None:
    mapping = correctness()
    torch.manual_seed(0)
    logits = torch.randn(1, 4, len(NEW_LABELS), requires_grad=True)
    old_labels = torch.tensor([[mapping.old_outside_id, OLD_LABELS.index("S-city"), -100, -100]])
    new_labels = torch.tensor([[-100, -100, NEW_LABELS.index("S-locality"), -100]])

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            assert set(inputs) == {"input_ids"}
            return SimpleNamespace(logits=logits, secondary_logits=None)

    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    trainer.entity_dice_weight = 0.1
    loss = trainer.compute_loss(
        FixedModel(),
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": old_labels,
            "secondary_labels": new_labels,
        },
    )
    terms = head_loss_terms(
        logits,
        new_labels,
        old_labels,
        mapping.old_to_new,
    )
    entity_labels = head_entity_labels(
        new_labels,
        old_labels,
        own_o_label_id=mapping.new_outside_id,
        cross_o_label_id=mapping.old_outside_id,
    )
    expected = (terms.total / terms.normalizer + 0.1 * entity_dice_loss(logits, entity_labels)) / 1.1

    torch.testing.assert_close(loss, expected.to(logits.dtype))
    assert trainer.dual_head_telemetry["dual_entity_dice_weight"] == 0.1
    assert trainer.dual_head_telemetry["dual_entity_dice_tokens"] == 3


def test_mapped_single_head_consumes_primary_objective_weights() -> None:
    mapping = correctness()
    torch.manual_seed(5)
    logits = torch.randn(1, 3, len(NEW_LABELS), requires_grad=True)
    old_labels = torch.tensor([[OLD_LABELS.index("S-city"), -100, -100]])
    new_labels = torch.tensor([[-100, NEW_LABELS.index("S-locality"), -100]])
    weights = torch.tensor([[0.85, 1.0, 0.0]])

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            assert set(inputs) == {"input_ids"}
            return SimpleNamespace(logits=logits, secondary_logits=None)

    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    loss = trainer.compute_loss(
        FixedModel(),
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "labels": old_labels,
            "primary_objective_weights": weights,
            "secondary_labels": new_labels,
        },
    )
    expected = head_loss_terms(
        logits,
        new_labels,
        old_labels,
        mapping.old_to_new,
        token_objective_weights=weights,
    )

    torch.testing.assert_close(loss, (expected.total / expected.normalizer).to(logits.dtype))
    assert trainer.dual_head_telemetry["dual_new_loss_normalizer"] == pytest.approx(1.85)


@pytest.mark.parametrize("use_margin,use_risk", [(True, False), (False, True), (True, True)])
def test_retired_head_applies_bioes_controls_only_during_training(use_margin, use_risk) -> None:
    mapping = correctness()
    torch.manual_seed(42)
    logits = torch.randn(1, 4, len(NEW_LABELS), requires_grad=True)
    old = torch.tensor([[OLD_LABELS.index("S-city"), -100, -100, -100]])
    new = torch.tensor([[-100, NEW_LABELS.index("S-locality"), 0, -100]])
    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    if use_margin:
        trainer.bioes_structure_cost_matrix = bioes_structure_cost_matrix(
            NEW_LABELS, boundary_cost=4, type_cost=2
        )
        trainer.bioes_mapped_cost_rows = mapped_structure_cost_rows(
            trainer.bioes_structure_cost_matrix, mapping.old_to_new
        )
    if use_risk:
        trainer.bioes_risk_weight = 1
        trainer.bioes_risk_threshold = 1
        trainer.bioes_risk_scale = 0.5
        trainer.bioes_continuation = continuation_label_mask(NEW_LABELS)
        trainer.bioes_incompatible = ~torch.eye(len(NEW_LABELS), dtype=torch.bool)

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            return SimpleNamespace(logits=logits, secondary_logits=None)

    model = FixedModel()
    inputs = {"input_ids": torch.ones(1, 4, dtype=torch.long), "labels": old, "secondary_labels": new}
    trained = trainer.compute_loss(model, dict(inputs))
    trained_gradient = torch.autograd.grad(trained, logits, retain_graph=True)[0]
    model.training = False
    evaluated = trainer.compute_loss(model, dict(inputs))
    ordinary = head_loss_terms(logits, new, old, mapping.old_to_new)
    torch.testing.assert_close(evaluated, ordinary.total / ordinary.normalizer)
    assert not torch.allclose(trained_gradient, torch.autograd.grad(evaluated, logits)[0])
    assert not torch.allclose(trained, evaluated)


def test_span_risk_does_not_cross_batch_rows_or_depend_on_batching() -> None:
    logits = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], [[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    labels = torch.tensor([[0, 1], [0, 0]])
    options = dict(outside_id=0, threshold=1.0, scale=0.5, weight=1.0)
    incompatible = ~torch.eye(3, dtype=torch.bool)
    continuation = torch.tensor([False, False, True])
    batched = span_risk_gate(logits, labels, incompatible, continuation, **options)
    separate = torch.cat(
        [
            span_risk_gate(logits[i : i + 1], labels[i : i + 1], incompatible, continuation, **options)
            for i in range(2)
        ]
    )
    torch.testing.assert_close(batched, separate)
    labels[0, 0] = 2  # A clipped span still starts a fresh row/segment.
    assert torch.isfinite(span_risk_gate(logits, labels, incompatible, continuation, **options)).all()


def test_mapped_single_head_applies_presence_ce_only_on_complete_tokens() -> None:
    mapping = correctness()
    torch.manual_seed(1)
    logits = torch.randn(1, 4, len(NEW_LABELS), requires_grad=True)
    old_labels = torch.tensor([[mapping.old_outside_id, OLD_LABELS.index("S-city"), -100, -100]])
    new_labels = torch.tensor([[-100, -100, NEW_LABELS.index("S-locality"), -100]])
    presence_labels = torch.tensor([[0, 1, -100, -100]])

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            assert set(inputs) == {"input_ids"}
            return SimpleNamespace(logits=logits, secondary_logits=None)

    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    trainer.complete_presence_loss_weight = 0.5
    loss = trainer.compute_loss(
        FixedModel(),
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": old_labels,
            "secondary_labels": new_labels,
            "complete_presence_labels": presence_labels,
        },
    )
    terms = head_loss_terms(logits, new_labels, old_labels, mapping.old_to_new)
    presence = entity_presence_loss(logits, presence_labels, mapping.new_outside_id)
    expected = (terms.total / terms.normalizer + 0.5 * presence) / 1.5

    torch.testing.assert_close(loss, expected.to(logits.dtype))
    assert trainer.dual_head_telemetry["dual_complete_presence_weight"] == 0.5
    assert trainer.dual_head_telemetry["dual_complete_presence_tokens"] == 2


def test_mapped_single_head_applies_complete_boundary_ce_only_on_entity_tokens() -> None:
    mapping = correctness()
    torch.manual_seed(2)
    logits = torch.randn(1, 4, len(NEW_LABELS), requires_grad=True)
    old_labels = torch.tensor([[mapping.old_outside_id, OLD_LABELS.index("S-city"), -100, -100]])
    new_labels = torch.tensor([[-100, -100, NEW_LABELS.index("S-locality"), -100]])
    boundary_labels = torch.tensor([[-100, BOUNDARY2ID["S"], BOUNDARY2ID["S"], -100]])

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            assert set(inputs) == {"input_ids"}
            return SimpleNamespace(logits=logits, secondary_logits=None)

    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    trainer.complete_boundary_loss_weight = 0.25
    loss = trainer.compute_loss(
        FixedModel(),
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": old_labels,
            "secondary_labels": new_labels,
            "complete_boundary_labels": boundary_labels,
        },
    )
    terms = head_loss_terms(logits, new_labels, old_labels, mapping.old_to_new)
    boundary = annotated_boundary_loss(logits, boundary_labels, NEW_LABELS)
    expected = (terms.total / terms.normalizer + 0.25 * boundary) / 1.25

    torch.testing.assert_close(loss, expected.to(logits.dtype))
    assert trainer.dual_head_telemetry["dual_complete_boundary_weight"] == 0.25
    assert trainer.dual_head_telemetry["dual_complete_boundary_tokens"] == 2


def test_mapped_single_head_partial_o_weight_is_training_only() -> None:
    mapping = correctness()
    torch.manual_seed(2)
    logits = torch.randn(1, 3, len(NEW_LABELS), requires_grad=True)
    old_labels = torch.tensor([[-100, -100, -100]])
    new_labels = torch.tensor([[NEW_LABELS.index("S-locality"), -100, -100]])
    consistency_mask = torch.tensor([[0, 1, 1]])

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            assert set(inputs) == {"input_ids"}
            return SimpleNamespace(logits=logits, secondary_logits=None)

    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    trainer.partial_o_loss_weight = 0.2
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "labels": old_labels,
        "secondary_labels": new_labels,
        "consistency_mask": consistency_mask,
    }
    loss = trainer.compute_loss(FixedModel(), dict(inputs))
    terms = head_loss_terms(logits, new_labels, old_labels, mapping.old_to_new)
    unmarked_o = partial_o_loss(logits, consistency_mask, mapping.new_outside_id)
    expected = (terms.total / terms.normalizer + 0.2 * unmarked_o) / 1.2

    torch.testing.assert_close(loss, expected.to(logits.dtype))
    assert trainer.dual_head_telemetry["dual_partial_o_weight"] == 0.2
    assert trainer.dual_head_telemetry["dual_partial_o_tokens"] == 2

    FixedModel.training = False
    eval_loss = trainer.compute_loss(FixedModel(), dict(inputs))
    torch.testing.assert_close(eval_loss, (terms.total / terms.normalizer).to(logits.dtype))


def test_mapped_single_head_pu_preserves_known_span_type_loss() -> None:
    mapping = correctness()
    torch.manual_seed(3)
    logits = torch.randn(1, 3, len(NEW_LABELS), requires_grad=True)
    old_labels = torch.tensor([[-100, -100, -100]])
    known_type = NEW_LABELS.index("S-locality")
    new_labels = torch.tensor([[known_type, -100, -100]])
    positive_mask = torch.tensor([[1, 0, 0]])
    consistency_mask = torch.tensor([[0, 1, 1]])
    group_ids = torch.tensor([0])
    class_priors = torch.tensor([0.25])

    class FixedModel:
        training = True

        def __call__(self, **inputs):
            assert set(inputs) == {"input_ids"}
            return SimpleNamespace(logits=logits, secondary_logits=None)

    trainer = DualHeadLossMixin()
    trainer.correctness_map = mapping
    trainer.dual_head_retired = True
    trainer.partial_entity_pu_loss_weight = 0.4
    loss = trainer.compute_loss(
        FixedModel(),
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "labels": old_labels,
            "secondary_labels": new_labels,
            "consistency_mask": consistency_mask,
            "partial_entity_positive_mask": positive_mask,
            "partial_entity_pu_group": group_ids,
            "partial_entity_pu_prior": class_priors,
        },
    )

    typed = head_loss_terms(logits, new_labels, old_labels, mapping.old_to_new)
    pu = nonnegative_pu_entity_loss(
        logits,
        positive_mask,
        consistency_mask,
        group_ids,
        class_priors,
        mapping.new_outside_id,
    )
    expected = (typed.total / typed.normalizer + 0.4 * pu.loss) / 1.4
    typed_only_gradient = torch.autograd.grad(typed.total / typed.normalizer, logits, retain_graph=True)[0]
    loss.backward()

    torch.testing.assert_close(loss, expected.to(logits.dtype))
    assert torch.count_nonzero(typed_only_gradient[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert trainer.dual_head_telemetry["dual_new_tokens"] == 1
    assert trainer.dual_head_telemetry["dual_partial_entity_pu_positive_tokens"] == 1
    assert trainer.dual_head_telemetry["dual_partial_entity_pu_unlabeled_tokens"] == 2


def test_a_compressed_fade_retires_the_old_head_where_it_reaches_zero() -> None:
    schedule = parse_transition_schedule("linear:1.0:0.0:0.2")
    assert transition_completion_step(schedule, 1500) == 300
    assert transition_weight(schedule, 299, 1500) > 0
    assert transition_weight(schedule, 300, 1500) == 0.0


def scheduler_factors(restart_step: int, total: int, sched: str = "cosine") -> list[float]:
    args = TrainingArguments(
        output_dir="/tmp/pii-dual-head-schedule-test",
        lr_scheduler_type=sched,
        warmup_ratio=0.1,
        learning_rate=1.0,
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = phase_restart_scheduler(args, optimizer, total, restart_step)
    return [scheduler.lr_lambdas[0](step) for step in range(total)]


def test_the_learning_rate_restarts_where_the_transition_ends() -> None:
    total, restart = 100, 40
    factors = scheduler_factors(restart, total)

    assert factors[0] == pytest.approx(0.0), "the first phase warms up from zero"
    assert factors[restart] == pytest.approx(0.0), "so does the phase after the handover"
    assert factors[restart + 4] > factors[restart - 1], (
        "the point of restarting is that the second phase gets a full rate again, "
        "not the decayed tail of the first"
    )
    assert max(factors[restart:]) == pytest.approx(1.0, abs=1e-6)
    assert factors[-1] < 0.05, "each phase still decays to the end of its own segment"


def test_horizon_extension_preserves_the_old_schedule_then_restarts_it() -> None:
    args = TrainingArguments(
        output_dir="/tmp/pii-resume-extension-schedule-test",
        lr_scheduler_type="cosine",
        learning_rate=1.0,
    )
    old_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    old_scheduler = get_scheduler(
        "cosine",
        optimizer=old_optimizer,
        num_warmup_steps=0,
        num_training_steps=100,
    )
    extended_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    extended_scheduler = phase_restart_scheduler(args, extended_optimizer, 200, 100)
    old_factor = old_scheduler.lr_lambdas[0]
    extended_factor = extended_scheduler.lr_lambdas[0]

    for step in (0, 1, 50, 99):
        assert extended_factor(step) == pytest.approx(old_factor(step))
        assert extended_factor(step + 100) == pytest.approx(old_factor(step))


def test_a_restart_outside_the_horizon_is_refused() -> None:
    for restart in (0, 100, 140):
        with pytest.raises(ValueError, match="horizon"):
            scheduler_factors(restart, 100)
