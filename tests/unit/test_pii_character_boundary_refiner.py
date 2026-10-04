import pytest
import torch
from safetensors.torch import save_file

from scripts.pii_character_boundary_refiner import (
    HashedBoundaryRanker,
    boundary_feature_ids,
    boundary_refiner_deployment_config,
    boundary_training_groups,
    bracket_imbalance,
    document_bootstrap_f1_standard_error,
    export_boundary_refiner,
    group_batch,
    proposal_span_groups,
    refine_row_spans,
    select_earliest_epoch_within_one_standard_error,
    select_largest_penalty_within_one_standard_error,
    span_group_batch,
    span_pair_feature_ids,
    span_training_groups,
    strip_whitespace_edges,
    validate_fit_output_directory,
)


def test_boundary_features_are_stable_and_sensitive_to_local_context():
    first = boundary_feature_ids("abc.", 3, "end", 2, buckets=1024)
    second = boundary_feature_ids("abc.", 3, "end", 2, buckets=1024)
    changed = boundary_feature_ids("abc,", 3, "end", 2, buckets=1024)

    assert first == second
    assert first != changed
    assert all(0 <= feature < 1024 for feature in first)


def test_zero_ranker_scores_every_candidate_equally():
    model = HashedBoundaryRanker(128)
    features = torch.randint(0, 128, (4, 3, 22))

    assert torch.equal(model(features), torch.zeros(4, 3))


def test_training_groups_do_not_make_another_true_endpoint_negative():
    rows = [
        {
            "id": "row",
            "lang": "en",
            "text": "abcd",
            "spans": [[0, 2, "person_name"], [2, 4, "organization"]],
        }
    ]
    groups = boundary_training_groups(rows)
    first_end = next(group for group in groups if group.position == 2 and group.endpoint == "end")
    second_start = next(group for group in groups if group.position == 2 and group.endpoint == "start")

    assert first_end.candidates[first_end.target_index] == 2
    assert second_start.candidates[second_start.target_index] == 2


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_training_batch_supports_clipped_and_full_radius_candidates(radius):
    rows = [{"id": "r", "lang": "en", "text": "Alice went away.", "spans": [[0, 5, "person_name"]]}]
    groups = boundary_training_groups(rows, radius)
    features, mask, targets = group_batch(
        groups, rows, {"person_name": 0}, context=3, buckets=1024, device=torch.device("cpu")
    )
    assert features.shape[1] == 2 * radius + 1
    assert mask.sum(dim=1).tolist() == [len(g.candidates) for g in groups]
    model = HashedBoundaryRanker(1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = model.weights.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(features).masked_fill(~mask, -float("inf")), targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(initial, model.weights.weight.detach())


def test_refinement_moves_character_endpoints_without_changing_count_or_type():
    predictions = [
        {"start": 0, "end": 4, "label": "person_name"},
        {"start": 5, "end": 8, "label": "organization"},
    ]
    targets = {
        (0, "start", "person_name"),
        (3, "end", "person_name"),
        (5, "start", "organization"),
        (9, "end", "organization"),
    }

    refined, telemetry = refine_row_spans(
        "abc. def!",
        predictions,
        lambda position, endpoint, label, _source: float((position, endpoint, label) in targets),
    )

    assert refined == [
        {"start": 0, "end": 3, "label": "person_name"},
        {"start": 5, "end": 9, "label": "organization"},
    ]
    assert telemetry["proposals"] == 2
    assert telemetry["changed_spans"] == 2


def test_refinement_identity_tie_break_preserves_original_spans():
    predictions = [{"start": 1, "end": 4, "label": "person_name"}]

    refined, telemetry = refine_row_spans(" abc ", predictions, lambda *_args: 0.0)

    assert refined == predictions
    assert telemetry["changed_spans"] == 0


def test_refinement_can_move_inside_an_existing_overlap_component():
    predictions = [
        {"start": 0, "end": 3, "label": "person_name"},
        {"start": 2, "end": 5, "label": "organization"},
    ]

    refined, telemetry = refine_row_spans(
        "abcde",
        predictions,
        lambda position, endpoint, label, _source: float(
            (position, endpoint, label) == (2, "end", "person_name")
        ),
    )

    assert refined == [
        {"start": 0, "end": 2, "label": "person_name"},
        {"start": 2, "end": 5, "label": "organization"},
    ]
    assert telemetry["overlapping_source_rows"] == 1
    assert telemetry["overlapping_refined_rows"] == 0


def test_refinement_does_not_converge_distinct_proposals_to_a_duplicate():
    predictions = [
        {"start": 0, "end": 3, "label": "person_name"},
        {"start": 1, "end": 4, "label": "person_name"},
    ]

    refined, telemetry = refine_row_spans(
        "abcd",
        predictions,
        lambda position, endpoint, _label, _source: float((position, endpoint) in {(0, "start"), (3, "end")}),
    )

    assert len({(span["start"], span["end"], span["label"]) for span in refined}) == 2
    assert telemetry["duplicate_source_spans"] == 0
    assert telemetry["duplicate_refined_spans"] == 0


def test_movement_penalty_requires_a_sufficient_score_gain():
    predictions = [{"start": 1, "end": 4, "label": "person_name"}]

    refined, telemetry = refine_row_spans(
        " abc ",
        predictions,
        lambda position, endpoint, _label, _source: 0.1 if (position, endpoint) == (0, "start") else 0.0,
        movement_penalty=0.2,
    )

    assert refined == predictions
    assert telemetry["changed_spans"] == 0


def test_paired_movement_penalty_prefers_symmetric_trim_over_equal_scoring_shift():
    # "1 (2000)." with a proposal on "(2000)": shifting left to " (2000" and
    # trimming to "2000" both move two characters; the shift scores slightly higher.
    text = "1 (2000)."
    predictions = [{"start": 2, "end": 8, "label": "date"}]
    scores = {(1, "start"): 2.93, (3, "start"): 2.92, (7, "end"): 3.69}

    def score(position, endpoint, _label, _source):
        return scores.get((position, endpoint), 0.0)

    independent, _ = refine_row_spans(text, predictions, score, movement_penalty=2.0)
    paired, _ = refine_row_spans(text, predictions, score, movement_penalty=2.0, paired_movement_penalty=1.0)
    unchanged, _ = refine_row_spans(
        text, predictions, score, movement_penalty=2.0, paired_movement_penalty=4.0
    )

    assert independent == [{"start": 1, "end": 7, "label": "date"}]
    assert paired == [{"start": 3, "end": 7, "label": "date"}]
    assert unchanged == independent


def test_feature_template_two_extends_template_one_with_endpoint_conjoined_characters():
    start = boundary_feature_ids("1 (2000)", 1, "start", 0, buckets=1 << 18, template=2)
    end = boundary_feature_ids("1 (2000)", 1, "end", 0, buckets=1 << 18, template=2)

    assert start[:22] == boundary_feature_ids("1 (2000)", 1, "start", 0, buckets=1 << 18)
    assert len(start) == 27
    # Template 1 shares the right-character unigram across endpoints; template 2 does not.
    assert len(set(start[22:]) & set(end[22:])) == 0
    with pytest.raises(ValueError):
        boundary_feature_ids("abc", 1, "start", 0, template=99)


def test_feature_template_three_adds_endpoint_conjoined_skip_grams():
    text = "1 (2000)."
    three = boundary_feature_ids(text, 3, "start", 0, buckets=1 << 18, template=3)
    assert three[:27] == boundary_feature_ids(text, 3, "start", 0, buckets=1 << 18, template=2)
    assert len(three) == 35
    assert three[27:] != boundary_feature_ids(text, 3, "end", 0, buckets=1 << 18, template=3)[27:]


def test_feature_template_four_is_language_conjoined():
    text = "김성주 원장은 말했다"
    korean = boundary_feature_ids(text, 6, "end", 0, buckets=1 << 18, template=4, language="ko")
    other = boundary_feature_ids(text, 6, "end", 0, buckets=1 << 18, template=4, language="ja")
    assert korean[:35] == boundary_feature_ids(text, 6, "end", 0, buckets=1 << 18, template=3)
    assert len(korean) == 41
    assert korean[35:39] != other[35:39]
    assert korean[39:] == other[39:]
    with pytest.raises(ValueError):
        boundary_feature_ids(text, 6, "end", 0, template=4)


def test_bracket_imbalance_counts_unmatched_brackets_by_depth():
    assert bracket_imbalance("(2000)") == 0
    assert bracket_imbalance(" (2000") == 1
    assert bracket_imbalance("5 UE 2259/91)") == 1
    assert bracket_imbalance("オリヒロ（株）") == 0
    assert bracket_imbalance(")(") == 2


def test_bracket_penalty_prefers_balanced_trim_over_one_sided_shift():
    text = "1 (2000)."
    predictions = [{"start": 2, "end": 8, "label": "date"}]
    scores = {(1, "start"): 2.93, (3, "start"): 2.92, (7, "end"): 3.69}

    def score(position, endpoint, _label, _source):
        return scores.get((position, endpoint), 0.0)

    plain, _ = refine_row_spans(text, predictions, score, movement_penalty=2.0)
    balanced, _ = refine_row_spans(text, predictions, score, movement_penalty=2.0, bracket_penalty=1.0)

    assert plain == [{"start": 1, "end": 7, "label": "date"}]
    assert balanced == [{"start": 3, "end": 7, "label": "date"}]


def test_pair_score_is_added_to_candidate_spans():
    predictions = [{"start": 2, "end": 8, "label": "date"}]
    refined, _ = refine_row_spans(
        "1 (2000).",
        predictions,
        lambda *_args: 0.0,
        pair_score=lambda start, end, _label: 1.0 if (start, end) == (3, 7) else 0.0,
    )
    assert refined == [{"start": 3, "end": 7, "label": "date"}]


def test_span_groups_and_batch_score_joint_candidates():
    rows = [{"id": "r", "lang": "pt", "text": "1 (2000).", "spans": [[3, 7, "date"]]}]
    groups = span_training_groups(rows, 1)
    assert len(groups) == 1
    group = groups[0]
    assert group.candidates[group.target_index] == (3, 7)
    assert len(group.candidates) == 9
    features, mask, targets = span_group_batch(
        groups, rows, {"date": 0}, context=3, buckets=1 << 18, device=torch.device("cpu"), template=1
    )
    assert features.shape == (1, 9, 22 + 22 + len(span_pair_feature_ids("1 (2000).", 3, 7, 0)))
    assert mask.all() and targets.tolist() == [group.target_index]


def test_word_guard_never_trims_letters_but_may_trim_punctuation():
    predictions = [{"start": 0, "end": 7, "label": "person_name"}]

    def prefer_trim(position, endpoint, _label, _source):
        return 5.0 if (position, endpoint) in {(6, "end"), (1, "start")} else 0.0

    free, _ = refine_row_spans("길이 원장, x", predictions, prefer_trim)
    guarded, _ = refine_row_spans("길이 원장, x", predictions, prefer_trim, protect_word_characters=True)
    assert free == [{"start": 1, "end": 6, "label": "person_name"}]
    # Dropping the leading Hangul syllable is forbidden; dropping the comma is not.
    assert guarded == [{"start": 0, "end": 6, "label": "person_name"}]


def test_proposal_groups_surround_the_proposal_and_target_reachable_gold():
    rows = [{"id": "r", "lang": "pt", "text": "1 (2000).", "spans": [[3, 7, "date"]]}]
    predictions = [{"id": "r", "preds": [{"start": 2, "end": 8, "label": "date"}]}]
    exact = [{"id": "r", "preds": [{"start": 3, "end": 7, "label": "date"}]}]
    far = [{"id": "r", "preds": [{"start": 0, "end": 1, "label": "date"}]}]

    (group,) = proposal_span_groups(rows, predictions, 1)
    assert group.source == (2, 8)
    assert group.candidates[group.target_index] == (3, 7)
    assert (1, 9) in group.candidates and (2, 8) in group.candidates
    (stay,) = proposal_span_groups(rows, exact, 1)
    assert stay.candidates[stay.target_index] == stay.source == (3, 7)
    assert proposal_span_groups(rows, far, 1) == []


def test_collapsed_whitespace_window_reaches_past_space_runs():
    plain = boundary_feature_ids("ab   (x", 5, "start", 0, buckets=1 << 18)
    collapsed = boundary_feature_ids("ab   (x", 5, "start", 0, buckets=1 << 18, collapse_whitespace=True)
    single = boundary_feature_ids("ab (x", 3, "start", 0, buckets=1 << 18)
    assert collapsed == single
    assert collapsed != plain


def test_strip_whitespace_edges_keeps_count_and_all_space_spans():
    rows = [{"id": "r", "text": "  Ann  , x"}]
    predictions = [
        {
            "id": "r",
            "preds": [{"start": 0, "end": 7, "label": "person_name"}, {"start": 0, "end": 2, "label": "x"}],
        }
    ]
    (stripped,) = strip_whitespace_edges(rows, predictions)
    assert stripped["preds"] == [
        {"start": 2, "end": 5, "label": "person_name"},
        {"start": 0, "end": 2, "label": "x"},
    ]


def test_endpoint_radii_limit_each_end_separately():
    predictions = [{"start": 2, "end": 6, "label": "date"}]
    targets = {(0, "start"), (4, "end")}

    def score(position, endpoint, _label, _source):
        return 5.0 if (position, endpoint) in targets else 0.0

    both, _ = refine_row_spans("abcdefgh", predictions, score, radius=2)
    start_only, _ = refine_row_spans("abcdefgh", predictions, score, radius=2, endpoint_radii=(2, 1))
    assert both == [{"start": 0, "end": 4, "label": "date"}]
    assert start_only == [{"start": 0, "end": 6, "label": "date"}]
    config = {
        "schema_version": 1,
        "context": 3,
        "hash_buckets": 1024,
        "radius": 2,
        "endpoint_radii": [2, 1],
        "labels": ["date"],
        "supported_languages": ["en"],
    }
    deployed = boundary_refiner_deployment_config(config)
    assert deployed["start_offsets"] == [-2, -1, 0, 1, 2]
    assert deployed["end_offsets"] == [-1, 0, 1]


def test_deployment_config_refuses_decoder_extensions():
    config = {
        "schema_version": 1,
        "context": 3,
        "hash_buckets": 1024,
        "radius": 1,
        "labels": ["date"],
        "supported_languages": ["en"],
    }
    boundary_refiner_deployment_config(config)
    with pytest.raises(ValueError):
        boundary_refiner_deployment_config({**config, "paired_movement_penalty": 1.0})
    with pytest.raises(ValueError):
        boundary_refiner_deployment_config({**config, "feature_template": 2})
    with pytest.raises(ValueError):
        boundary_refiner_deployment_config({**config, "bracket_penalty": 1.0})
    with pytest.raises(ValueError):
        boundary_refiner_deployment_config({**config, "pair_features": True})
    with pytest.raises(ValueError):
        boundary_refiner_deployment_config({**config, "protect_word_characters": True})


def test_one_standard_error_rule_selects_earliest_practical_tie():
    curve = [
        {"epoch": 0, "selector": {"f1": 0.80}},
        {"epoch": 1, "selector": {"f1": 0.84}},
        {"epoch": 2, "selector": {"f1": 0.85}},
    ]

    selection = select_earliest_epoch_within_one_standard_error(curve, 0.015)

    assert selection["numeric_best_epoch"] == 2
    assert selection["eligible_epochs"] == [1, 2]
    assert selection["selected_epoch"] == 1


def test_one_standard_error_rule_selects_largest_practical_penalty():
    curve = [
        {"movement_penalty": 0.0, "selector": {"f1": 0.85}},
        {"movement_penalty": 0.5, "selector": {"f1": 0.845}},
        {"movement_penalty": 1.0, "selector": {"f1": 0.82}},
    ]

    selection = select_largest_penalty_within_one_standard_error(curve, 0.01)

    assert selection["numeric_best_penalty"] == 0.0
    assert selection["eligible_penalties"] == [0.0, 0.5]
    assert selection["selected_penalty"] == 0.5


def test_document_bootstrap_noise_estimate_is_seeded():
    rows = [
        {"id": "a", "spans": [[0, 1, "person_name"]]},
        {"id": "b", "spans": [[0, 1, "person_name"]]},
    ]
    predictions = [
        {"id": "a", "preds": [{"start": 0, "end": 1, "label": "person_name"}]},
        {"id": "b", "preds": []},
    ]

    first = document_bootstrap_f1_standard_error(rows, predictions, samples=100, seed=155)
    second = document_bootstrap_f1_standard_error(rows, predictions, samples=100, seed=155)

    assert first == second
    assert first > 0


def test_fit_accepts_only_tracker_metadata_in_precreated_output(tmp_path):
    output = tmp_path / "refiner"
    output.mkdir()
    (output / "model.safetensors.meta.md").write_text("tracker metadata")

    validate_fit_output_directory(output)

    (output / "model.safetensors").write_bytes(b"prior model")
    try:
        validate_fit_output_directory(output)
    except FileExistsError as error:
        assert "prior artifacts" in str(error)
    else:
        raise AssertionError("prior refiner artifacts must block fitting")


def test_onnx_export_matches_endpoint_scores(tmp_path):
    pytest.importorskip("onnxruntime")
    refiner = tmp_path / "refiner"
    refiner.mkdir()
    model = HashedBoundaryRanker(128)
    generator = torch.Generator().manual_seed(155)
    with torch.no_grad():
        model.weights.weight.copy_(torch.rand(128, 1, generator=generator) - 0.5)
    save_file(model.state_dict(), refiner / "model.safetensors")
    (refiner / "config.json").write_text(
        """{
  "context": 3,
  "hash_buckets": 128,
  "labels": ["locality", "monetary_amount", "person_name", "phone_number", "postal_code"],
  "movement_penalty": 2.0,
  "radius": 1,
  "schema": "pii-character-boundary-refiner-v1",
  "schema_version": 1,
  "supported_languages": ["en"]
}\n"""
    )

    receipt = export_boundary_refiner(refiner, tmp_path / "export")

    assert receipt["validation"]["requests"] == 5
    assert receipt["validation"]["max_absolute_error"] <= 1e-6
    assert (tmp_path / "export/boundary-refiner.onnx").is_file()
    assert (tmp_path / "export/boundary-refiner.json").is_file()
    assert (tmp_path / "export/boundary-refiner.README.md").is_file()
    assert "boundary-refiner.README.md" in receipt["outputs"]
