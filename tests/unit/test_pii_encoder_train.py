import json
import math
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts import pii_encoder_train
from scripts.pii_bioes import count_bioes_violations
from scripts.pii_character_projection import build_literal_character_projection
from scripts.pii_encoder_train import (
    ANNOTATED_SPANS_ONLY,
    BOUNDARY2ID,
    ContinuousCharacterDataCollator,
    FrozenClassifierPrefixCallback,
    JointMlmDataCollator,
    LearningCurveCallback,
    OTokenLossWeightMixin,
    PartialLabelDataCollator,
    SpanDataset,
    WeightedSamplingTrainer,
    annotated_boundary_loss,
    collapse_bioes_logits,
    combine_objective_sampling_weights,
    declared_subclass_exposure_targets,
    differential_learning_rate_groups,
    encoder_parameter_prior_anchors,
    encoder_parameter_prior_loss,
    entity_dice_loss,
    entity_presence_log_odds,
    entity_presence_loss,
    expand_output_labels,
    expected_entity_ratio_loss,
    freeze_encoder_except_top_layers,
    freeze_encoder_parameters,
    inherited_output_parameter_prior_anchors,
    inherited_output_parameter_prior_loss,
    last_trainable_encoder_matrix,
    load_encoder_prior_model,
    loss_logit_gradient_norm,
    masked_language_model_loss,
    nonnegative_pu_entity_loss,
    o_token_weighted_trainer_class,
    o_weighted_token_classification_loss,
    parent_presence_kl_loss,
    parse_language_shares,
    parse_mlm_pool_probabilities,
    parse_partial_negative_sources,
    parse_train_pool_specs,
    partial_boundary_retention_loss,
    partial_negative_loss,
    partial_o_loss,
    per_row_o_weighted_token_classification_loss,
    physical_batch_objective_loss,
    projected_warm_start_cut_groups,
    projected_warm_start_labels,
    rebalance_family_share_within_groups,
    record_training_objective_config,
    replay_rows_at_probability,
    resolve_physical_batch_mlm_probabilities,
    resolve_training_seeds,
    resolve_window_limits,
    sampling_plan_for_rows,
    sampling_weights_and_pool_keys_for_rows,
    sampling_weights_for_rows,
    save_initialized_checkpoint,
    span_metrics,
    successor_parent_output_rows,
    symmetric_token_kl_loss,
    token_classification_loss,
    window_mlm_replay_records,
    window_records,
)


def test_training_windowing_preserves_exact_resume_policy():
    resolve = pii_encoder_train.resolve_training_windowing
    assert resolve(None) == "token-capacity"
    assert resolve(None, resume_config=SimpleNamespace()) == "character"
    saved = SimpleNamespace(pii_training_windowing="token-capacity")
    assert resolve(None, resume_config=saved) == "token-capacity"
    with pytest.raises(ValueError, match="exact resume"):
        resolve("character", resume_config=saved)


def test_training_window_limit_can_differ_without_changing_validation_limit():
    assert resolve_window_limits(900, 2048) == (2048, 900)
    assert resolve_window_limits(900, 0) == (900, 900)
    with pytest.raises(ValueError, match="--max-chars must be positive"):
        resolve_window_limits(0, 2048)
    with pytest.raises(ValueError, match="--max-train-chars must be nonnegative"):
        resolve_window_limits(900, -1)


def test_reference_projection_keeps_nested_names_and_partial_membership(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {
        "text": "Google's lawyer",
        "spans": [[0, 15, "person_reference"], [0, 6, "organization"]],
        "primary_span_objective_weights": [2.0, 3.0],
        "label_space": "v1",
        "lang": "en",
        "sampling_pool": "fixed",
        "predicate_spans": [
            {"start": 0, "end": 15, "type": "person_reference", "attrs": {}},
            {"start": 0, "end": 6, "type": "organization", "attrs": {}},
        ],
    }
    partial = {
        "text": "the lawyer",
        "spans": [[0, 10, "person_reference"]],
        "supervision": "annotated_spans_only",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(partial) + "\n")
    original = window_records(path, 900)
    assert original[0]["spans"] == [[0, 6, "organization"]]
    assert original[0]["primary_span_objective_weights"] == [3.0]
    assert [span["type"] for span in original[0]["predicate_spans"]] == ["organization"]
    assert original[1]["spans"] == partial["spans"]

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == row["text"]
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 6), (6, 15), (0, 0)],
            }

    item = SpanDataset([original[0]], Tokenizer(), {"O": 0, "S-organization": 1}, max_len=32)[0]
    assert item["labels"] == [-100, 1, 0, -100]
    projected = window_records(path, 900, defer_references=True)
    assert len(projected) == len(original) == 2
    assert projected[0]["spans"] == [[0, 6, "organization"]]
    assert projected[0]["primary_span_objective_weights"] == [3.0]
    assert projected[0]["deferred_reference_spans"] == [[0, 15, "person_reference"]]
    assert [span["type"] for span in projected[0]["predicate_spans"]] == ["organization"]
    assert projected[0]["sampling_pool"] == original[0]["sampling_pool"]
    assert projected[0]["supervision"] == "complete"
    assert projected[1]["spans"] == []
    assert projected[1]["supervision"] == "annotated_spans_only"
    assert window_records(path, 900) == original


def test_training_seed_tree_forks_model_sampling_replay_and_validation():
    seeds = resolve_training_seeds(None, pii_encoder_train.REPRODUCIBILITY_CONFIG_PATH, 23)

    assert seeds["mode"] == "pii-seed-tree-v1"
    assert seeds["root"] == 23
    assert len({seeds["model"], seeds["sampling"], seeds["replay"], seeds["validation"]}) == 4
    assert resolve_training_seeds(29, pii_encoder_train.REPRODUCIBILITY_CONFIG_PATH) == {
        "mode": "legacy-single-seed",
        "model": 29,
        "sampling": 29,
        "replay": 29,
        "validation": 29,
        "validation_pinned": False,
    }


def test_a_pinned_validation_seed_survives_a_change_of_run_seed():
    runs = [
        resolve_training_seeds(seed, pii_encoder_train.REPRODUCIBILITY_CONFIG_PATH, None, 7)
        for seed in (29, 31)
    ]

    # The model and the data order move with the run seed; the choice of what is scored
    # does not, which is what makes two seeds of one recipe comparable.
    assert [run["model"] for run in runs] == [29, 31]
    assert [run["sampling"] for run in runs] == [29, 31]
    assert {run["validation"] for run in runs} == {7}
    assert all(run["validation_pinned"] for run in runs)


def test_a_pinned_validation_seed_also_overrides_the_seed_tree():
    tree = resolve_training_seeds(None, pii_encoder_train.REPRODUCIBILITY_CONFIG_PATH, 23)
    pinned = resolve_training_seeds(None, pii_encoder_train.REPRODUCIBILITY_CONFIG_PATH, 23, 7)

    assert pinned["validation"] == 7
    assert pinned["validation"] != tree["validation"]
    assert (pinned["model"], pinned["sampling"]) == (tree["model"], tree["sampling"])


def test_validation_selection_is_seeded_and_not_order_sensitive_by_default():
    rows = [{"id": index} for index in range(20)]

    selected = pii_encoder_train.select_validation_rows(rows, limit=5, seed=23)

    assert selected == pii_encoder_train.select_validation_rows(rows, limit=5, seed=23)
    assert selected != rows[:5]
    assert (
        pii_encoder_train.select_validation_rows(
            rows,
            limit=5,
            seed=23,
            policy="head",
        )
        == rows[:5]
    )


def test_validation_selection_receipt_identifies_exact_draw():
    universe = [
        {"id": 0, "lang": "en", "src": "a", "label_space": "v1"},
        {
            "id": 1,
            "lang": "fr",
            "src": "b",
            "label_space": "v2",
            "supervision": "annotated_spans_only",
        },
        {"id": 2, "lang": "en", "src": "a", "label_space": "v1"},
    ]
    selected = [universe[2], universe[1]]

    receipt = pii_encoder_train.validation_selection_receipt(
        universe,
        selected,
        policy="shuffle",
        seed=31,
        limit=2,
    )

    assert receipt["schema_version"] == 2
    assert receipt["universe_windows"] == 3
    assert receipt["selected_windows"] == 2
    assert receipt["universe_supervision"] == {"annotated_spans_only": 1, "complete": 2}
    assert receipt["selected_supervision"] == {"annotated_spans_only": 1, "complete": 1}
    assert receipt["universe_label_spaces"] == {"v1": 2, "v2": 1}
    assert receipt["selected_label_spaces"] == {"v1": 1, "v2": 1}
    assert receipt["selected_languages"] == {"en": 1, "fr": 1}
    assert receipt["selected_sources"] == {"a": 1, "b": 1}
    assert len(receipt["selected_sha256"]) == 64


def test_extra_save_steps_callback_does_not_change_other_steps() -> None:
    callback = pii_encoder_train.ExtraSaveStepsCallback([125, 375])
    control = SimpleNamespace(should_save=False)

    assert callback.on_step_end(None, SimpleNamespace(global_step=124), control).should_save is False
    assert callback.on_step_end(None, SimpleNamespace(global_step=125), control).should_save is True

    control.should_save = False
    assert callback.on_step_end(None, SimpleNamespace(global_step=250), control).should_save is False
    assert callback.on_step_end(None, SimpleNamespace(global_step=375), control).should_save is True


def test_extra_save_steps_callback_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="positive optimizer-step"):
        pii_encoder_train.ExtraSaveStepsCallback([0])


def test_stop_after_step_callback_requests_terminal_checkpoint_and_eval() -> None:
    callback = pii_encoder_train.StopAfterStepCallback(250)
    control = SimpleNamespace(
        should_evaluate=False,
        should_save=False,
        should_training_stop=False,
    )

    callback.on_step_end(None, SimpleNamespace(global_step=249), control)
    assert control.should_evaluate is False
    assert control.should_save is False
    assert control.should_training_stop is False

    callback.on_step_end(None, SimpleNamespace(global_step=250), control)
    assert control.should_evaluate is True
    assert control.should_save is True
    assert control.should_training_stop is True


def test_stop_after_step_callback_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError, match="positive optimizer-step"):
        pii_encoder_train.StopAfterStepCallback(0)


def test_final_selection_callback_requests_aligned_terminal_eval_and_save() -> None:
    callback = pii_encoder_train.FinalSelectionCheckpointCallback()
    control = SimpleNamespace(should_evaluate=False, should_save=False)

    callback.on_step_end(None, SimpleNamespace(global_step=9, max_steps=10), control)
    assert control.should_evaluate is False
    assert control.should_save is False

    callback.on_step_end(None, SimpleNamespace(global_step=10, max_steps=10), control)
    assert control.should_evaluate is True
    assert control.should_save is True


def test_effective_training_step_preserves_selected_trajectory_offset() -> None:
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=3),
        args=SimpleNamespace(pii_step_offset=8750),
    )

    assert pii_encoder_train.effective_training_step(trainer) == 8753


def test_explicit_final_checkpoint_rejects_best_only_retention(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_encoder_train.py",
            "--data",
            str(tmp_path / "unopened-data"),
            "--out",
            str(tmp_path / "out"),
            "--save-limit",
            "1",
            "--keep-final-checkpoint",
        ],
    )
    with pytest.raises(SystemExit) as error:
        pii_encoder_train.main()
    assert error.value.code == 2
    assert "--keep-final-checkpoint requires --save-limit >= 2" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_context_mixture_cli_requires_context_field(monkeypatch, tmp_path, capsys):
    data = tmp_path / "unopened-data"
    data.mkdir()
    (data / "train.jsonl").write_text("{}\n")
    (data / "labels.json").write_text(json.dumps({"labels": ["person_name"]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_encoder_train.py",
            "--data",
            str(tmp_path / "unopened-data"),
            "--out",
            str(tmp_path / "out"),
            "--context-configurations",
            "[[-1,0,1],[-1,0],[0]]",
        ],
    )
    with pytest.raises(SystemExit) as error:
        pii_encoder_train.main()
    assert error.value.code == 2
    assert "--context-configurations requires --context-field" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_seeds_before_loading_tokenizer_or_model(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n")
    (data_dir / "labels.json").write_text(json.dumps({"labels": ["name"]}))
    events = []

    monkeypatch.setattr(pii_encoder_train, "set_seed", lambda seed: events.append(("seed", seed)))

    def stop_at_tokenizer(model_name):
        events.append(("tokenizer", model_name))
        raise RuntimeError("stop after tokenizer")

    monkeypatch.setattr(pii_encoder_train.AutoTokenizer, "from_pretrained", stop_at_tokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_encoder_train.py",
            "--data",
            str(data_dir),
            "--out",
            str(tmp_path / "out"),
            "--model",
            "test/model",
            "--seed",
            "23",
        ],
    )

    with pytest.raises(RuntimeError, match="stop after tokenizer"):
        pii_encoder_train.main()

    assert events == [("seed", 23), ("tokenizer", "test/model")]


def test_external_resume_requires_two_checkpoint_slots(monkeypatch, tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n")
    (data_dir / "labels.json").write_text(json.dumps({"labels": ["name"]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_encoder_train.py",
            "--data",
            str(data_dir),
            "--out",
            str(tmp_path / "new-stage-output"),
            "--model",
            "test/model",
            "--resume-from-checkpoint",
            str(tmp_path / "prior-stage" / "checkpoint-500"),
            "--save-limit",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        pii_encoder_train.main()

    assert "best-checkpoint rotation can delete the new resume milestone" in capsys.readouterr().err


def test_dual_head_accepts_entity_dice_after_marginal_composition(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n")
    (data_dir / "labels.json").write_text(json.dumps({"labels": ["name"]}))
    map_path = tmp_path / "map.json"
    map_path.write_text("{}")

    def stop_at_tokenizer(model_name):
        raise RuntimeError(f"stop after tokenizer: {model_name}")

    monkeypatch.setattr(pii_encoder_train.AutoTokenizer, "from_pretrained", stop_at_tokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_encoder_train.py",
            "--data",
            str(data_dir),
            "--out",
            str(tmp_path / "out"),
            "--model",
            "test/model",
            "--head-kind",
            "affine",
            "--dual-head-map",
            str(map_path),
            "--entity-dice-loss",
            "0.1",
            "--max-steps",
            "2",
        ],
    )

    with pytest.raises(RuntimeError, match="stop after tokenizer"):
        pii_encoder_train.main()


def test_entity_pu_cli_requires_frozen_prior_and_excludes_masked_o(monkeypatch, tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n")
    (data_dir / "labels.json").write_text(json.dumps({"labels": ["name"]}))
    map_path = tmp_path / "map.json"
    map_path.write_text("{}")
    prior_path = tmp_path / "prior.json"
    prior_path.write_text("{}")
    base = [
        "pii_encoder_train.py",
        "--data",
        str(data_dir),
        "--out",
        str(tmp_path / "out"),
        "--model",
        "test/model",
        "--dual-head-map",
        str(map_path),
        "--partial-entity-pu-loss",
        "0.2",
    ]

    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="2"):
        pii_encoder_train.main()
    assert "requires an existing --partial-entity-pu-prior" in capsys.readouterr().err

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--partial-entity-pu-prior", str(prior_path), "--partial-o-loss", "0.2"],
    )
    with pytest.raises(SystemExit, match="2"):
        pii_encoder_train.main()
    assert "matched alternatives" in capsys.readouterr().err


def test_save_initialized_checkpoint_records_control_and_rejects_overwrite(tmp_path) -> None:
    class Saveable:
        def __init__(self, filename):
            self.filename = filename
            self.config = SimpleNamespace()

        def save_pretrained(self, output):
            (output / self.filename).write_text("saved")

    model = Saveable("model.marker")
    tokenizer = Saveable("tokenizer.marker")
    provenance = {"schema_version": 1, "seed": 101, "bioes_outputs": 349}
    output = tmp_path / "initialized"

    save_initialized_checkpoint(model, tokenizer, output, provenance)

    assert model.config.pii_initialized_only is True
    assert model.config.pii_initialization_control == provenance
    assert json.loads((output / "pii_initialization_control.json").read_text()) == provenance
    assert (output / "model.marker").is_file()
    assert (output / "tokenizer.marker").is_file()
    with pytest.raises(FileExistsError, match="unexpected entries"):
        save_initialized_checkpoint(model, tokenizer, output, provenance)


def test_freeze_encoder_parameters_leaves_task_head_trainable() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 4)
            self.classifier = nn.Linear(4, 2)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()

    summary = freeze_encoder_parameters(model)

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert summary["encoder_parameters"] == sum(parameter.numel() for parameter in model.encoder.parameters())
    assert summary["trainable_parameters"] == sum(
        parameter.numel() for parameter in model.classifier.parameters()
    )


def test_frozen_classifier_prefix_survives_gradients_and_weight_decay_bit_exact() -> None:
    model = nn.Module()
    model.classifier = nn.Linear(3, 4)
    model.config = SimpleNamespace(pii_successor_parent_output_rows=2)
    callback = FrozenClassifierPrefixCallback(model, successor_parent_output_rows(model))
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.1, weight_decay=0.2)
    inherited_weight = model.classifier.weight[:2].detach().clone()
    inherited_bias = model.classifier.bias[:2].detach().clone()
    appended_weight = model.classifier.weight[2:].detach().clone()
    callback.on_train_begin(None, None, None)

    model.classifier(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    callback.on_optimizer_step(None, None, None, optimizer=optimizer)

    assert torch.equal(model.classifier.weight[:2], inherited_weight)
    assert torch.equal(model.classifier.bias[:2], inherited_bias)
    assert not torch.equal(model.classifier.weight[2:], appended_weight)
    assert torch.count_nonzero(optimizer.state[model.classifier.weight]["exp_avg"][:2]) == 0
    assert torch.count_nonzero(optimizer.state[model.classifier.bias]["exp_avg"][:2]) == 0
    callback.on_train_end(None, None, None)


def test_successor_parent_output_rows_requires_a_strict_declared_prefix() -> None:
    model = nn.Module()
    model.classifier = nn.Linear(3, 4)
    model.config = SimpleNamespace(pii_successor_parent_output_rows=4)

    with pytest.raises(ValueError, match="strict classifier prefix"):
        successor_parent_output_rows(model)


def test_freeze_encoder_except_top_layers_trains_only_top_and_head() -> None:
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Linear(3, 3)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(nn.Linear(3, 3) for _ in range(4))

    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.classifier = nn.Linear(3, 2)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()

    summary = freeze_encoder_except_top_layers(model, 2)

    assert all(not parameter.requires_grad for parameter in model.encoder.embeddings.parameters())
    assert all(
        not parameter.requires_grad
        for layer in model.encoder.encoder.layer[:2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.encoder.encoder.layer[2:]
        for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert summary["encoder_layers"] == 4
    assert summary["trainable_top_encoder_layers"] == 2


def test_top_encoder_layer_trains_while_inherited_classifier_prefix_stays_exact() -> None:
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Linear(3, 3)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(nn.Linear(3, 3) for _ in range(2))

        def forward(self, inputs):
            hidden = self.embeddings(inputs)
            for layer in self.encoder.layer:
                hidden = layer(hidden)
            return hidden

    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.classifier = nn.Linear(3, 4)
            self.config = SimpleNamespace(pii_successor_parent_output_rows=2)

        @property
        def base_model(self):
            return self.encoder

        def forward(self, inputs):
            return self.classifier(self.encoder(inputs))

    model = TokenClassifier()
    freeze_encoder_except_top_layers(model, 1)
    callback = FrozenClassifierPrefixCallback(model, successor_parent_output_rows(model))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
        weight_decay=0.2,
    )
    inherited_weight = model.classifier.weight[:2].detach().clone()
    inherited_bias = model.classifier.bias[:2].detach().clone()
    appended_weight = model.classifier.weight[2:].detach().clone()
    frozen_layer_weight = model.encoder.encoder.layer[0].weight.detach().clone()
    trainable_layer_weight = model.encoder.encoder.layer[1].weight.detach().clone()
    callback.on_train_begin(None, None, None)

    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    callback.on_optimizer_step(None, None, None, optimizer=optimizer)

    assert torch.equal(model.classifier.weight[:2], inherited_weight)
    assert torch.equal(model.classifier.bias[:2], inherited_bias)
    assert not torch.equal(model.classifier.weight[2:], appended_weight)
    assert torch.equal(model.encoder.encoder.layer[0].weight, frozen_layer_weight)
    assert not torch.equal(model.encoder.encoder.layer[1].weight, trainable_layer_weight)
    callback.on_train_end(None, None, None)


def test_freeze_encoder_except_top_layers_can_train_full_stack() -> None:
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Linear(3, 3)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(nn.Linear(3, 3) for _ in range(4))

    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.classifier = nn.Linear(3, 2)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()

    summary = freeze_encoder_except_top_layers(model, 4)

    assert all(not parameter.requires_grad for parameter in model.encoder.embeddings.parameters())
    assert all(
        parameter.requires_grad for layer in model.encoder.encoder.layer for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert summary["encoder_layers"] == 4
    assert summary["trainable_top_encoder_layers"] == 4
    assert summary["frozen_encoder_parameters"] == sum(
        parameter.numel() for parameter in model.encoder.embeddings.parameters()
    )


def test_span_metrics_scores_exact_typed_bioes_spans_and_ignores_special_tokens() -> None:
    id2label = {
        0: "O",
        1: "B-name",
        2: "I-name",
        3: "E-name",
        4: "S-name",
    }
    labels = np.array(
        [
            [-100, 1, 3, 0, -100],
            [-100, 0, 4, 0, -100],
        ]
    )
    predicted_ids = np.array(
        [
            [4, 1, 3, 0, 4],
            [4, 0, 0, 0, 4],
        ]
    )
    logits = np.eye(len(id2label))[predicted_ids]

    metrics = span_metrics((logits, labels), id2label)
    preprocessed_metrics = span_metrics((predicted_ids, labels), id2label)

    assert metrics["span_true_positives"] == 1
    assert metrics["span_predictions"] == 1
    assert metrics["span_gold"] == 2
    assert metrics["span_precision"] == 1.0
    assert metrics["span_recall"] == 0.5
    assert metrics["span_f1"] == 2 / 3
    assert metrics["raw_bioes_invalid_constraints"] == 0
    assert metrics["raw_bioes_invalid_rate"] == 0.0
    assert metrics["raw_bioes_sequences_with_invalid"] == 0
    assert preprocessed_metrics == metrics


def test_bioes_violation_count_covers_start_transitions_types_and_end() -> None:
    assert count_bioes_violations(["O", "S-name", "O"]) == (0, 4)
    assert count_bioes_violations(["O", "I-name", "O"]) == (2, 4)
    assert count_bioes_violations(["B-name", "E-name"]) == (0, 3)
    assert count_bioes_violations(["B-name", "I-name"]) == (1, 3)
    assert count_bioes_violations(["B-name", "E-other"]) == (1, 3)


def test_span_metrics_reports_raw_bioes_illegality_before_permissive_repair() -> None:
    id2label = {
        0: "O",
        1: "B-name",
        2: "I-name",
        3: "E-name",
        4: "S-name",
    }
    labels = np.array([[-100, 0, 4, 0, -100]])
    predicted_ids = np.array([[4, 0, 2, 0, 4]])

    metrics = span_metrics((predicted_ids, labels), id2label)

    assert metrics["span_predictions"] == 1
    assert metrics["raw_bioes_invalid_constraints"] == 2
    assert metrics["raw_bioes_constraints"] == 4
    assert metrics["raw_bioes_invalid_rate"] == 0.5
    assert metrics["raw_bioes_sequences_with_invalid"] == 1
    assert metrics["raw_bioes_sequence_invalid_rate"] == 1.0


def test_persist_validation_selection_reports_whether_it_established_the_draw(tmp_path) -> None:
    receipt = {"selected_sha256": "abc", "policy": "shuffle"}
    path, established = pii_encoder_train.persist_validation_selection(tmp_path, receipt, resume=False)
    assert established and path.is_file()
    _, established_again = pii_encoder_train.persist_validation_selection(tmp_path, receipt, resume=True)
    assert not established_again
    with pytest.raises(ValueError):
        pii_encoder_train.persist_validation_selection(tmp_path, {"selected_sha256": "def"}, resume=True)


def test_a_new_validation_draw_discards_the_resumed_best_checkpoint() -> None:
    callback = pii_encoder_train.NewValidationDrawCallback()
    state = SimpleNamespace(best_metric=0.9598, best_model_checkpoint="/old/checkpoint-7750")
    control = object()

    assert callback.on_train_begin(None, state, control) is control
    assert state.best_metric is None
    assert state.best_model_checkpoint is None


def test_a_fresh_run_has_no_resumed_best_checkpoint_to_discard() -> None:
    callback = pii_encoder_train.NewValidationDrawCallback()
    state = SimpleNamespace(best_metric=None, best_model_checkpoint=None)
    control = object()

    assert callback.on_train_begin(None, state, control) is control
    assert state.best_metric is None


def test_learning_curve_callback_persists_budget_and_quality_point(tmp_path) -> None:
    callback = LearningCurveCallback(train_windows=100)
    callback.on_train_begin(SimpleNamespace(output_dir=str(tmp_path)), None, None)
    callback.on_evaluate(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=12, epoch=1.25),
        None,
        metrics={"eval_loss": 0.3, "eval_span_f1": 0.7},
    )

    point = json.loads((tmp_path / "learning_curve.jsonl").read_text())
    assert point["step"] == 12
    assert point["epoch"] == 1.25
    assert point["windows_seen"] == 125
    assert point["eval_loss"] == 0.3
    assert point["eval_span_f1"] == 0.7
    assert point["elapsed_s"] >= 0
    assert point["peak_memory_allocated_mib"] >= 0
    assert point["peak_memory_reserved_mib"] >= 0


def test_rolling_train_loss_uses_step_weighted_sampler_epoch_window(tmp_path) -> None:
    callback = pii_encoder_train.RollingTrainLossCallback(sampling_epoch_steps=4)
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=0, epoch=0.0, log_history=[])
    callback.on_train_begin(args, state, None)

    observed = []
    for step, loss in ((2, 1.0), (4, 3.0), (8, 5.0)):
        state.global_step = step
        state.epoch = step / 4
        logs = {"loss": loss}
        callback.on_log(args, state, None, logs=logs)
        observed.append((logs["rolling_sampling_epoch_loss"], logs["rolling_sampling_epoch_coverage"]))

    assert observed == [(1.0, 0.5), (2.0, 1.0), (5.0, 1.0)]
    points = [json.loads(line) for line in (tmp_path / "training_loss_curve.jsonl").read_text().splitlines()]
    assert [point["step"] for point in points] == [2, 4, 8]
    assert points[-1]["rolling_sampling_epoch_loss"] == 5.0


def test_victory_lap_arguments_restart_scaled_base_schedule_for_one_epoch(tmp_path) -> None:
    base = pii_encoder_train.TrainingArguments(
        output_dir=str(tmp_path / "main"),
        learning_rate=3e-4,
        num_train_epochs=3.0,
        max_steps=100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )
    base.pii_encoder_learning_rate = 1e-5
    base.pii_character_learning_rate = None
    base.pii_character_output_learning_rate = None

    lap = pii_encoder_train.victory_lap_training_arguments(
        base,
        output_dir=tmp_path / "lap",
        learning_rate_scale=0.1,
        selected_step=8750,
        validation_seed=47,
        validation_windows=8000,
    )

    assert lap.learning_rate == pytest.approx(3e-5)
    assert lap.pii_encoder_learning_rate == pytest.approx(1e-6)
    assert lap.num_train_epochs == 1.0
    assert lap.max_steps == -1
    assert lap.lr_scheduler_type == base.lr_scheduler_type
    assert lap.warmup_ratio == base.warmup_ratio
    assert lap.eval_strategy == "no"
    assert lap.save_strategy == "no"
    assert lap.load_best_model_at_end is False
    assert lap.pii_step_offset == 8750
    assert lap.data_seed == 47
    assert lap.sampling_epoch_examples == 8000


def test_default_victory_lap_scale_is_two_thirds_of_smoked_rate() -> None:
    assert pii_encoder_train.DEFAULT_VICTORY_LAP_LR_SCALE == pytest.approx(0.1 * 2 / 3)


def test_lap_only_selection_binds_checkpoint_and_validation_draw(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-3500"
    checkpoint.mkdir()
    receipt_path = tmp_path / "victory_lap.json"
    receipt = {
        "status": "completed",
        "selected_checkpoint": str(checkpoint),
        "selected_step": 3500,
        "selected_eval_loss": 0.1,
        "selected_metric_name": "eval_span_f1",
        "selected_metric_value": 0.82,
        "validation_selected_sha256": "abc123",
        "base_learning_rates": {
            "head": 3e-4,
            "encoder": 1e-5,
            "character": None,
            "character_output": None,
        },
        "base_scheduler": "cosine",
        "warmup_ratio": 0.03,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert (
        pii_encoder_train.load_victory_lap_only_selection(
            receipt_path,
            init_checkpoint=checkpoint,
            validation_sha256="abc123",
        )
        == receipt
    )

    with pytest.raises(ValueError, match="validation draw differs"):
        pii_encoder_train.load_victory_lap_only_selection(
            receipt_path,
            init_checkpoint=checkpoint,
            validation_sha256="different",
        )

    other_checkpoint = tmp_path / "checkpoint-4000"
    other_checkpoint.mkdir()
    with pytest.raises(ValueError, match="initialization differs"):
        pii_encoder_train.load_victory_lap_only_selection(
            receipt_path,
            init_checkpoint=other_checkpoint,
            validation_sha256="abc123",
        )

    base = pii_encoder_train.TrainingArguments(
        output_dir=str(tmp_path / "lap"),
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
    )
    base.pii_encoder_learning_rate = 1e-5
    base.pii_character_learning_rate = None
    base.pii_character_output_learning_rate = None
    pii_encoder_train.validate_victory_lap_only_configuration(receipt, base)

    base.learning_rate = 2e-4
    with pytest.raises(ValueError, match="base learning rates differ"):
        pii_encoder_train.validate_victory_lap_only_configuration(receipt, base)


def test_learning_curve_integrates_actual_optimizer_learning_rates(tmp_path) -> None:
    callback = LearningCurveCallback(train_windows=100)
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=2, epoch=0.5)
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}, {"lr": 1e-3}])
    callback.on_train_begin(args, state, None)
    callback.on_optimizer_step(args, state, None, optimizer=optimizer)
    optimizer.param_groups[0]["lr"] = 5e-4
    optimizer.param_groups[1]["lr"] = 5e-4
    callback.on_optimizer_step(args, state, None, optimizer=optimizer)
    callback.on_evaluate(args, state, None, metrics={})

    point = json.loads((tmp_path / "learning_curve.jsonl").read_text())
    assert point["schema_version"] == 2
    assert point["learning_rate"] == 5e-4
    assert point["learning_rate_by_group"] == [5e-4, 5e-4]
    assert point["learning_rate_integral"] == pytest.approx(1.5e-3)
    assert point["learning_rate_integral_by_group"] == pytest.approx([1.5e-3, 1.5e-3])
    assert point["learning_rate_group_names"] == ["group-0", "group-1"]


def test_differential_learning_rate_groups_separate_encoder_and_head() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 4)
            self.classifier = nn.Linear(4, 2)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()
    groups = differential_learning_rate_groups(
        model,
        {"encoder.weight", "classifier.weight"},
        head_lr=3e-4,
        encoder_lr=3e-6,
        weight_decay=0.01,
    )

    assert [group["group_name"] for group in groups] == [
        "encoder-decay",
        "encoder-no-decay",
        "head-decay",
        "head-no-decay",
    ]
    assert [group["lr"] for group in groups] == [3e-6, 3e-6, 3e-4, 3e-4]
    assert [group["weight_decay"] for group in groups] == [0.01, 0.0, 0.01, 0.0]
    assert sum(len(group["params"]) for group in groups) == 4


def test_differential_learning_rate_groups_isolate_character_encoder() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 4)
            self.character_encoder = nn.Linear(3, 4)
            self.classifier = nn.Linear(4, 2)
            self.character_classifier = nn.Linear(4, 2, bias=False)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()
    groups = differential_learning_rate_groups(
        model,
        {
            "encoder.weight",
            "character_encoder.weight",
            "classifier.weight",
            "character_classifier.weight",
        },
        head_lr=3e-6,
        encoder_lr=5e-7,
        character_lr=3e-4,
        character_output_lr=1e-4,
        weight_decay=0.01,
    )

    assert [group["group_name"] for group in groups] == [
        "encoder-decay",
        "encoder-no-decay",
        "character-decay",
        "character-no-decay",
        "character-output-no-decay",
        "head-decay",
        "head-no-decay",
    ]
    assert [group["lr"] for group in groups] == [5e-7, 5e-7, 3e-4, 3e-4, 1e-4, 3e-6, 3e-6]
    assert [group["weight_decay"] for group in groups] == [0.01, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0]
    assert sum(len(group["params"]) for group in groups) == 7


def test_differential_learning_rate_groups_allow_frozen_encoder_for_character_training() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 4)
            self.character_encoder = nn.Linear(3, 4)
            self.classifier = nn.Linear(4, 2)
            self.character_classifier = nn.Linear(4, 2, bias=False)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    groups = differential_learning_rate_groups(
        model,
        {
            "character_encoder.weight",
            "classifier.weight",
            "character_classifier.weight",
        },
        head_lr=3e-6,
        encoder_lr=3e-6,
        character_lr=3e-4,
        character_output_lr=1e-4,
        weight_decay=0.01,
        require_encoder_parameters=False,
    )

    assert [group["group_name"] for group in groups] == [
        "character-decay",
        "character-no-decay",
        "character-output-no-decay",
        "head-decay",
        "head-no-decay",
    ]
    assert sum(len(group["params"]) for group in groups) == 5


def test_learning_curve_resume_preserves_prior_active_time(tmp_path) -> None:
    args = SimpleNamespace(output_dir=str(tmp_path))
    first = LearningCurveCallback(train_windows=100)
    first.on_train_begin(args, None, None)
    first.on_evaluate(args, SimpleNamespace(global_step=10, epoch=1.0), None, metrics={})
    first_point = json.loads((tmp_path / "learning_curve.jsonl").read_text().splitlines()[-1])

    resumed = LearningCurveCallback(train_windows=100, resume=True)
    resumed.on_train_begin(args, None, None)
    resumed.on_evaluate(args, SimpleNamespace(global_step=20, epoch=2.0), None, metrics={})
    points = [json.loads(line) for line in (tmp_path / "learning_curve.jsonl").read_text().splitlines()]

    assert len(points) == 2
    assert points[-1]["elapsed_s"] >= first_point["elapsed_s"]
    assert points[-1]["step"] == 20
    assert points[-1]["windows_seen"] == 200


def test_training_objective_config_clears_warm_start_metadata() -> None:
    config = SimpleNamespace(
        pii_annotated_boundary_loss_weight=0.01,
        pii_partial_negative_loss_weight=0.1,
        pii_partial_negative_sources={"legacy": ["name"]},
    )
    args = SimpleNamespace(
        annotated_boundary_loss=0.0,
        annotated_boundary_telemetry=False,
        partial_boundary_retention_loss=0.0,
        partial_boundary_retention_temperature=1.0,
        partial_o_loss=0.0,
        partial_entity_pu_loss=0.0,
        partial_entity_pu_prior=None,
        partial_entity_pu_positive_margin=None,
        partial_expected_entity_ratio_loss=0.0,
        partial_expected_entity_ratio_prior=None,
        partial_expected_entity_ratio_lower_width=0.1,
        partial_parent_presence_kl=0.0,
        partial_negative_loss=0.0,
        coarse_agreement_weight_20=0.0,
        coarse_agreement_weight_9=0.0,
        coarse_agreement_weight_presence_20=0.0,
        coarse_agreement_reduction="sum",
        fine_label_loss_weight=1.0,
        o_token_loss_weight=1.0,
        entity_dice_loss=0.0,
        complete_presence_loss=0.0,
        complete_family_loss=0.0,
        complete_boundary_loss=0.0,
        reference_primary_positive_loss_weight=0.0,
        partial_primary_objective_weight=1.0,
        native_new_label_space=False,
        logical_step_objective_normalization=False,
        predicate_loss_weight=0.0,
        reference_type_residual_loss_weight=0.0,
        subclass_loss_weight=0.0,
        rdrop_alpha=0.0,
        mlm_replay_data="",
        mlm_replay_prob=0.0,
        mlm_replay_gold_policy="scale",
        mlm_physical_batch_prob=0.0,
        mlm_pool_probabilities={},
        mlm_loss_weight=1.0,
        mlm_probability=0.15,
        mlm_head_model="",
        model="test/model",
        use_mlm_objective=False,
        document_start_marker=False,
        accumulation_loss="mean",
        max_grad_norm=1.0,
        batch_formation="band-random",
        batch_padding_budget=0.10,
        draw_policy="carried",
        lr=2e-5,
        encoder_lr=None,
        continuous_character_lr=None,
        continuous_character_output_lr=None,
        continuous_character_auxiliary_loss_weight=0.0,
        continuous_character_auxiliary_loss_fade_steps=0,
        encoder_prior_checkpoint="",
        encoder_prior_weight=0.0,
        encoder_prior_start_step=0,
        inherited_output_prior_checkpoint=None,
        inherited_output_prior_weight=0.0,
        inherited_output_prior_start_step=0,
        val_selection="shuffle",
        max_val_windows=8000,
        selection_metric="loss",
        patience=8,
        early_stopping_threshold=0.001,
        train_loss_window_epochs=1.0,
        victory_lap_lr_scale=0.1,
    )

    record_training_objective_config(config, args, {})

    assert config.pii_initialized_only is False
    assert config.pii_annotated_boundary_loss_weight == 0.0
    assert config.pii_annotated_boundary_telemetry is False
    assert config.pii_partial_boundary_retention_loss_weight == 0.0
    assert config.pii_partial_o_loss_weight == 0.0
    assert config.pii_partial_entity_pu_loss_weight == 0.0
    assert config.pii_partial_entity_pu_prior is None
    assert config.pii_partial_entity_pu_prior_sha256 is None
    assert config.pii_partial_entity_pu_positive_margin is None
    assert config.pii_partial_expected_entity_ratio_loss_weight == 0.0
    assert config.pii_partial_expected_entity_ratio_prior is None
    assert config.pii_partial_expected_entity_ratio_prior_sha256 is None
    assert config.pii_partial_expected_entity_ratio_lower_width == 0.1
    assert config.pii_partial_parent_presence_kl_weight == 0.0
    assert config.pii_partial_negative_loss_weight == 0.0
    assert config.pii_partial_negative_sources == {}
    assert config.pii_coarse_agreement_weight_20 == 0.0
    assert config.pii_coarse_agreement_weight_9 == 0.0
    assert config.pii_coarse_agreement_weight_presence_20 == 0.0
    assert config.pii_coarse_agreement_reduction == "sum"
    assert config.pii_fine_label_loss_weight == 1.0
    assert config.pii_o_token_loss_weight == 1.0
    assert config.pii_entity_dice_loss_weight == 0.0
    assert config.pii_complete_presence_loss_weight == 0.0
    assert config.pii_complete_family_loss_weight == 0.0
    assert config.pii_complete_boundary_loss_weight == 0.0
    assert config.pii_reference_primary_positive_loss_weight == 0.0
    assert config.pii_partial_primary_objective_weight == 1.0
    assert config.pii_logical_step_objective_normalization is False
    assert config.pii_predicate_loss_weight == 0.0
    assert config.pii_reference_type_residual_loss_weight == 0.0
    assert config.pii_rdrop_alpha == 0.0
    assert config.pii_mlm_replay_probability == 0.0
    assert config.pii_mlm_replay_gold_policy == "scale"
    assert config.pii_mlm_physical_batch_probability == 0.0
    assert config.pii_mlm_pool_probabilities == {}
    assert config.pii_mlm_head_model is None
    assert config.pii_head_learning_rate == 2e-5
    assert config.pii_encoder_learning_rate is None
    assert config.pii_character_output_learning_rate is None
    assert config.pii_character_auxiliary_loss_weight == 0.0
    assert config.pii_character_auxiliary_loss_fade_steps == 0
    assert config.pii_character_learning_rate is None
    assert config.pii_encoder_prior_checkpoint is None
    assert config.pii_encoder_prior_weight == 0.0
    assert config.pii_encoder_prior_start_step == 0
    assert config.pii_inherited_output_prior_checkpoint is None
    assert config.pii_inherited_output_prior_weight == 0.0
    assert config.pii_inherited_output_prior_start_step == 0
    assert config.pii_validation_selection_policy == "shuffle"
    assert config.pii_validation_selection_limit == 8000
    assert config.pii_selection_metric == "loss"
    assert config.pii_early_stopping_patience == 8
    assert config.pii_early_stopping_threshold == 0.001
    assert config.pii_train_loss_window_epochs == 1.0
    assert config.pii_victory_lap_lr_scale == 0.1
    assert config.pii_victory_lap is None


def test_encoder_parameter_prior_anchors_only_trainable_encoder_parameters() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
            self.classifier = nn.Linear(4, 2)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()
    anchor = TokenClassifier()
    model.encoder[0].requires_grad_(False)
    with torch.no_grad():
        anchor.encoder[1].weight.fill_(2.0)

    anchors = encoder_parameter_prior_anchors(model, anchor)

    assert set(anchors) == {"1.weight", "1.bias"}
    assert anchors["1.weight"].device.type == "cpu"
    assert torch.all(anchors["1.weight"] == 2.0)


def test_encoder_parameter_prior_loss_is_zero_at_anchor_and_tracks_encoder_only() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2, bias=False)
            self.classifier = nn.Linear(2, 1, bias=False)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()
    anchor = TokenClassifier()
    anchor.load_state_dict(model.state_dict())
    anchors = encoder_parameter_prior_anchors(model, anchor)

    zero, count = encoder_parameter_prior_loss(model, anchors)
    with torch.no_grad():
        model.encoder.weight[0, 0] += 2.0
        model.classifier.weight.add_(10.0)
    moved, moved_count = encoder_parameter_prior_loss(model, anchors)

    assert zero.item() == 0.0
    assert moved.item() == pytest.approx(2.0)
    assert count == moved_count == 4


def test_inherited_output_prior_tracks_only_declared_classifier_prefix() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(3, 4)

    model = TokenClassifier()
    anchor = TokenClassifier()
    anchor.load_state_dict(model.state_dict())
    anchors = inherited_output_parameter_prior_anchors(model, anchor, 2)

    zero, count = inherited_output_parameter_prior_loss(model, anchors, 2)
    with torch.no_grad():
        model.classifier.weight[0, 1] += 2.0
        model.classifier.bias[1] += 3.0
        model.classifier.weight[3].add_(100.0)
        model.classifier.bias[3] += 100.0
    moved, moved_count = inherited_output_parameter_prior_loss(model, anchors, 2)

    assert zero.item() == 0.0
    assert moved.item() == pytest.approx(6.5)
    assert count == moved_count == 8


def test_inherited_output_prior_rejects_nonprefix_and_shape_mismatch() -> None:
    model = SimpleNamespace(classifier=nn.Linear(3, 4))
    anchor = SimpleNamespace(classifier=nn.Linear(3, 5))

    with pytest.raises(ValueError, match="rows must be"):
        inherited_output_parameter_prior_anchors(model, model, 4)
    with pytest.raises(ValueError, match="weight shape mismatch"):
        inherited_output_parameter_prior_anchors(model, anchor, 2)


def test_encoder_prior_loading_preserves_training_rng_stream(monkeypatch) -> None:
    marker = object()

    def consuming_loader(checkpoint):
        assert checkpoint == "anchor"
        assert torch.rand(1).numel() == 1
        assert np.random.random() >= 0
        assert pii_encoder_train.random.random() >= 0
        return marker

    monkeypatch.setattr(pii_encoder_train, "load_local_token_classifier", consuming_loader)
    torch.manual_seed(47)
    np.random.seed(53)
    pii_encoder_train.random.seed(59)
    expected_torch = torch.rand(1)
    expected_numpy = np.random.random()
    expected_python = pii_encoder_train.random.random()
    torch.manual_seed(47)
    np.random.seed(53)
    pii_encoder_train.random.seed(59)

    loaded = load_encoder_prior_model("anchor")

    assert loaded is marker
    torch.testing.assert_close(torch.rand(1), expected_torch)
    assert np.random.random() == expected_numpy
    assert pii_encoder_train.random.random() == expected_python


def test_mlm_windows_preserve_language_mass_and_credit_each_segment(tmp_path) -> None:
    packet = tmp_path / "natural.jsonl"
    packet.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "id": "en-long",
                    "text": "First. Second.",
                    "lang": "en",
                    "sampling_weight": 0.75,
                },
                {
                    "id": "en-short",
                    "text": "Third.",
                    "lang": "en",
                    "sampling_weight": 0.25,
                },
                {
                    "id": "de-short",
                    "text": "Vierte.",
                    "lang": "de",
                    "sampling_weight": 1.0,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    tokenizer = lambda text, **_kwargs: {"input_ids": list(range(len(text) + 2))}
    windows, windowing = window_mlm_replay_records(
        packet,
        max_chars=8,
        tokenizer=tokenizer,
        max_tokens=100,
    )

    language_mass = Counter()
    document_mass = Counter()
    for row in windows:
        language_mass[row["lang"]] += row["sampling_weight"]
        document_mass[row["source_document_id"]] += row["sampling_weight"]
    assert language_mass == pytest.approx({"de": 1.0, "en": 1.0})
    assert document_mass["en-long"] / document_mass["en-short"] == pytest.approx(6.0)
    assert all(row["objective"] == "mlm" for row in windows)
    assert windowing["adaptive_token_splits"] == 0


def test_mlm_windows_refine_token_overflow_without_length_normalization(tmp_path) -> None:
    packet = tmp_path / "natural.jsonl"
    packet.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "id": "en-long",
                    "text": "abcdefghij",
                    "lang": "en",
                    "sampling_weight": 0.5,
                },
                {
                    "id": "en-short",
                    "text": "klmno",
                    "lang": "en",
                    "sampling_weight": 0.5,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = lambda text, **_kwargs: {"input_ids": list(range(len(text) + 2))}

    windows, windowing = window_mlm_replay_records(
        packet,
        max_chars=10,
        tokenizer=tokenizer,
        max_tokens=7,
    )

    document_mass = Counter()
    for row in windows:
        document_mass[row["source_document_id"]] += row["sampling_weight"]
    assert len(windows) == 3
    assert document_mass["en-long"] / document_mass["en-short"] == pytest.approx(2.0)
    assert windowing == {
        "documents": 2,
        "initial_segments": 2,
        "initial_segments_by_language": {"en": 2},
        "final_segments": 3,
        "final_segments_by_language": {"en": 3},
        "adaptive_token_splits": 1,
        "adaptive_token_splits_by_language": {"en": 1},
        "maximum_segment_tokens": 7,
        "max_chars": 10,
        "max_tokens": 7,
        "segment_weighting": (
            "one source-row weight occurrence per final segment, then per-language mass renormalization; "
            "no token-count adjustment"
        ),
    }


def test_joint_objective_sampling_weights_have_exact_requested_mass() -> None:
    weights = combine_objective_sampling_weights(
        supervised_weights=[2.0, 1.0],
        supervised_count=2,
        mlm_weights=[3.0, 1.0],
        mlm_probability=0.15,
    )

    assert sum(weights[:2]) == pytest.approx(0.85)
    assert sum(weights[2:]) == pytest.approx(0.15)
    assert weights[0] / weights[1] == pytest.approx(2.0)
    assert weights[2] / weights[3] == pytest.approx(3.0)


def test_mlm_pool_probability_parser_is_explicit_and_bounded() -> None:
    assert parse_mlm_pool_probabilities(["pretraining=1", "gold=0.05"]) == {
        "pretraining": 1.0,
        "gold": 0.05,
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_mlm_pool_probabilities(["gold=.1", "gold=.2"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_mlm_pool_probabilities(["gold=1.1"])


def test_mlm_pool_probability_resolution_uses_only_positive_mass() -> None:
    probabilities, pool_mass = resolve_physical_batch_mlm_probabilities(
        ["gold", "teacher", "unused"],
        [0.8, 0.2, 0.0],
        default_probability=0.05,
        overrides={"teacher": 0.7},
    )

    assert probabilities == {"gold": 0.05, "teacher": 0.7}
    assert pool_mass == {"gold": 0.8, "teacher": 0.2, "unused": 0.0}
    with pytest.raises(ValueError, match="no positive sampling mass"):
        resolve_physical_batch_mlm_probabilities(
            ["gold", "unused"],
            [1.0, 0.0],
            default_probability=0.0,
            overrides={"unused": 1.0},
        )


def test_gold_rebalance_preserves_language_mass_and_joint_gold_share() -> None:
    rows = [
        {"lang": "en", "sampling_family": "gold"},
        {"lang": "en", "sampling_family": "synthetic"},
        {"lang": "es", "sampling_family": "gold"},
        {"lang": "es", "sampling_family": "synthetic"},
        {"lang": "de", "sampling_family": "synthetic"},
    ]
    weights = [0.12, 0.18, 0.03, 0.02, 0.65]
    mlm_probability = 0.15
    adjusted, evidence = rebalance_family_share_within_groups(
        rows,
        weights,
        group_field="lang",
        family_field="sampling_family",
        family_value="gold",
        target_share=0.15 / (1 - mlm_probability),
    )

    for language in ("de", "en", "es"):
        prior_mass = sum(weight for row, weight in zip(rows, weights) if row["lang"] == language)
        adjusted_mass = sum(weight for row, weight in zip(rows, adjusted) if row["lang"] == language)
        assert adjusted_mass == pytest.approx(prior_mass)
    adjusted_gold = sum(weight for row, weight in zip(rows, adjusted) if row["sampling_family"] == "gold")
    assert adjusted_gold / sum(adjusted) == pytest.approx(0.15 / 0.85)
    joint = combine_objective_sampling_weights(
        adjusted,
        len(adjusted),
        [1.0, 1.0],
        mlm_probability,
    )
    assert sum(
        weight for row, weight in zip(rows, joint[: len(rows)]) if row["sampling_family"] == "gold"
    ) == pytest.approx(0.15)
    assert evidence["prior_share"] == pytest.approx(0.15)
    assert evidence["achieved_share"] == pytest.approx(0.15 / 0.85)
    assert evidence["odds_multiplier"] > 1


def test_gold_rebalance_rejects_target_outside_language_support() -> None:
    rows = [
        {"lang": "en", "sampling_family": "gold"},
        {"lang": "de", "sampling_family": "synthetic"},
    ]

    with pytest.raises(ValueError, match="infeasible"):
        rebalance_family_share_within_groups(
            rows,
            [0.2, 0.8],
            group_field="lang",
            family_field="sampling_family",
            family_value="gold",
            target_share=0.25,
        )


def test_joint_mlm_collator_routes_unmarked_evaluation_rows_to_tag_collator() -> None:
    collator = JointMlmDataCollator(
        tokenizer=None,
        tag_collator=lambda rows: {"seen": rows},
        mlm_probability=0.15,
        replay_seed=17,
    )
    features = [{"input_ids": [1, 2], "labels": [0, 1]}]

    batch = collator(features)

    assert batch == {"seen": features}


def test_joint_mlm_collator_wraps_marked_tag_only_training_batch() -> None:
    collator = JointMlmDataCollator(
        tokenizer=None,
        tag_collator=lambda rows: {"seen": rows},
        mlm_probability=0.15,
        replay_seed=17,
    )
    features = [{"input_ids": [1, 2], "labels": [0, 1], "pii_objective": "tag"}]

    batch = collator(features)

    assert batch == {"tag_batch": {"seen": [{"input_ids": [1, 2], "labels": [0, 1]}]}}


def test_joint_mlm_collator_rejects_mixed_physical_batch() -> None:
    collator = JointMlmDataCollator(
        tokenizer=None,
        tag_collator=lambda rows: {"seen": rows},
        mlm_probability=0.15,
        replay_seed=17,
    )
    features = [
        {"input_ids": [1, 2], "labels": [0, 1], "pii_objective": "tag"},
        {"input_ids": [1, 3], "pii_objective": "mlm"},
    ]

    with pytest.raises(ValueError, match="must not mix"):
        collator(features)


def test_joint_mlm_collator_forces_one_target_when_dynamic_masking_selects_none() -> None:
    tokenizer = SimpleNamespace(all_special_ids=[0, 2], mask_token_id=99)
    collator = JointMlmDataCollator(
        tokenizer=tokenizer,
        tag_collator=None,
        mlm_probability=0.15,
        replay_seed=17,
    )
    collator._fallback_generator = torch.Generator().manual_seed(23)
    collator._mlm_collator = lambda rows: {
        "input_ids": torch.tensor([[0, 7, 2], [0, 8, 2]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "labels": torch.full((2, 3), -100),
    }

    batch = collator([{"input_ids": [0, 7, 2], "pii_objective": "mlm"}])["mlm_batch"]

    masked = torch.nonzero(batch["labels"] != -100, as_tuple=False)
    assert masked.shape == (1, 2)
    row, column = masked[0]
    assert batch["input_ids"][row, column] == tokenizer.mask_token_id
    assert batch["labels"][row, column] in (7, 8)
    assert column == 1


def test_gradient_probe_prefers_last_encoder_layer_over_pooler() -> None:
    class BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
            self.pooler = nn.Linear(3, 3)

    model = nn.Module()
    model.base_model = BaseModel()

    name, parameter = last_trainable_encoder_matrix(model)

    assert name == "encoder.layer.1.weight"
    assert parameter is model.base_model.encoder.layer[1].weight


def test_masked_language_model_loss_uses_task_encoder_and_fixed_head() -> None:
    class Encoder(nn.Module):
        def forward(self, input_ids, attention_mask, return_dict):
            del attention_mask
            assert return_dict is True
            return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 3).float())

    class TaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

        @property
        def base_model(self):
            return self.encoder

    head = nn.Linear(3, 4, bias=False)
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 2, -100]]),
    }

    loss, logits, masked_tokens = masked_language_model_loss(TaskModel(), head, batch)

    expected = torch.nn.functional.cross_entropy(logits[:, 1, :], torch.tensor([2]))
    torch.testing.assert_close(loss, expected)
    assert logits.shape == (1, 3, 4)
    assert masked_tokens == 1


def test_physical_batch_objective_loss_keeps_frequency_and_weight_independent() -> None:
    tag = torch.tensor(2.0)
    mlm = torch.tensor(3.0)
    physical_losses = [
        physical_batch_objective_loss(tag, None, 0.1),
        physical_batch_objective_loss(None, mlm, 0.1),
        physical_batch_objective_loss(None, mlm, 0.1),
        physical_batch_objective_loss(tag, None, 0.1),
    ]

    logical_loss = torch.stack(physical_losses).mean()

    assert logical_loss.item() == pytest.approx((2 * 2.0 + 2 * 0.1 * 3.0) / 4)
    with pytest.raises(ValueError, match="cannot carry both"):
        physical_batch_objective_loss(tag, mlm, 0.1)


def test_replay_rows_reaches_requested_share_deterministically() -> None:
    primary = [{"id": index} for index in range(90)]
    replay = [{"id": index} for index in range(7)]

    selected = replay_rows_at_probability(primary, replay, probability=0.1, seed=23)

    assert len(selected) == 10
    assert len(selected) / (len(primary) + len(selected)) == 0.1
    assert selected == replay_rows_at_probability(primary, replay, probability=0.1, seed=23)
    assert len({row["id"] for row in selected[:7]}) == 7


def test_replay_rows_stratifies_language_floor_without_changing_replay_share() -> None:
    primary = [{"id": index, "lang": "fr"} for index in range(40)]
    replay = [
        *[{"id": f"en-{index}", "lang": "en"} for index in range(50)],
        *[{"id": f"de-{index}", "lang": "de"} for index in range(50)],
    ]

    selected = replay_rows_at_probability(
        primary,
        replay,
        probability=0.6,
        seed=23,
        minimum_language_shares={"en": 0.25},
    )
    mixed = primary + selected

    assert len(selected) == 60
    assert len(selected) / len(mixed) == 0.6
    assert sum(row["lang"] == "en" for row in mixed) / len(mixed) == 0.25
    assert selected == replay_rows_at_probability(
        primary,
        replay,
        probability=0.6,
        seed=23,
        minimum_language_shares={"en": 0.25},
    )


def test_replay_rows_rejects_impossible_language_floor() -> None:
    primary = [{"lang": "fr"} for _ in range(90)]
    replay = [{"lang": "en"}]

    with pytest.raises(ValueError, match="replay budget"):
        replay_rows_at_probability(
            primary,
            replay,
            probability=0.1,
            seed=23,
            minimum_language_shares={"en": 0.25},
        )


def test_replay_rows_enforces_language_floor_within_replay_tranche() -> None:
    primary = [{"id": index, "lang": "sv"} for index in range(40)]
    replay = [
        *[{"id": f"en-{index}", "lang": "en"} for index in range(20)],
        *[{"id": f"sv-{index}", "lang": "sv"} for index in range(2)],
        *[{"id": f"de-{index}", "lang": "de"} for index in range(20)],
    ]

    selected = replay_rows_at_probability(
        primary,
        replay,
        probability=0.5,
        seed=31,
        minimum_language_shares={"en": 0.2},
        minimum_replay_language_shares={"en": 0.25, "sv": 0.3},
    )

    assert len(selected) == 40
    assert sum(row["lang"] == "en" for row in selected) >= 10
    assert sum(row["lang"] == "sv" for row in selected) >= 12
    assert sum(row["lang"] == "en" for row in primary + selected) >= 16
    assert selected == replay_rows_at_probability(
        primary,
        replay,
        probability=0.5,
        seed=31,
        minimum_language_shares={"en": 0.2},
        minimum_replay_language_shares={"en": 0.25, "sv": 0.3},
    )


def test_replay_rows_rejects_combined_floors_beyond_replay_budget() -> None:
    primary = [{"lang": "fr"} for _ in range(40)]
    replay = [{"lang": "en"}, {"lang": "de"}]

    with pytest.raises(ValueError, match="replay budget"):
        replay_rows_at_probability(
            primary,
            replay,
            probability=0.5,
            seed=31,
            minimum_language_shares={"en": 0.4},
            minimum_replay_language_shares={"de": 0.3},
        )


def test_parse_language_shares_rejects_duplicates_and_excess_total() -> None:
    assert parse_language_shares(["en=0.25", "de=0.10"]) == {"en": 0.25, "de": 0.1}
    with pytest.raises(ValueError, match="duplicate"):
        parse_language_shares(["en=0.25", "en=0.30"])
    with pytest.raises(ValueError, match="sum to more than 1"):
        parse_language_shares(["en=0.75", "de=0.50"])


def test_parse_partial_negative_sources_rejects_ambiguous_declarations() -> None:
    assert parse_partial_negative_sources(["aqmar-natural=organization, location,person_name"]) == {
        "aqmar-natural": ("organization", "location", "person_name")
    }
    with pytest.raises(ValueError, match="SOURCE=TYPE,TYPE"):
        parse_partial_negative_sources(["aqmar-natural"])
    with pytest.raises(ValueError, match="duplicate partial-negative source"):
        parse_partial_negative_sources(["aqmar-natural=location", "aqmar-natural=person_name"])
    with pytest.raises(ValueError, match="duplicate type"):
        parse_partial_negative_sources(["aqmar-natural=location,location"])


def test_window_records_preserves_language_for_post_windowing_mix(tmp_path) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": "First. Second.",
                "lang": "en",
                "spans": [],
                "src": "gold",
                "mix_source": "primary",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = window_records(corpus, max_chars=8)

    assert len(rows) == 2
    assert {row["lang"] for row in rows} == {"en"}
    assert {row["supervision"] for row in rows} == {"complete"}
    assert {row["src"] for row in rows} == {"gold"}
    assert {row["mix_source"] for row in rows} == {"primary"}


def test_window_records_never_splits_a_labeled_span(tmp_path) -> None:
    corpus = tmp_path / "train.jsonl"
    text = "Before very-long-entity after. Tail."
    start = text.index("very-long-entity")
    end = start + len("very-long-entity")
    corpus.write_text(
        json.dumps(
            {
                "id": "protected-span",
                "text": text,
                "lang": "en",
                "spans": [[start, end, "name"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = window_records(corpus, max_chars=start + 4)

    entity_windows = [row for row in rows if row["spans"]]
    assert len(entity_windows) == 1
    entity_window = entity_windows[0]
    assert entity_window["text"][entity_window["spans"][0][0] : entity_window["spans"][0][1]] == (
        "very-long-entity"
    )


def test_window_records_rebases_predicates_and_protects_ignored_spans(tmp_path) -> None:
    corpus = tmp_path / "train.jsonl"
    text = "Lead sentence. Lewis Stott met the treating nurse."
    name_start = text.index("Lewis")
    name_end = name_start + len("Lewis Stott")
    reference_start = text.index("the treating nurse")
    reference_end = reference_start + len("the treating nurse")
    corpus.write_text(
        json.dumps(
            {
                "text": text,
                "lang": "en",
                "spans": [[name_start, name_end, "person_name"]],
                "primary_span_objective_weights": [0.85],
                "predicate_spans": [
                    {
                        "start": name_start,
                        "end": name_end,
                        "type": "person_name",
                        "attrs": {
                            "given_name": [[name_start, name_start + 5]],
                            "family_name": [[name_end - 5, name_end]],
                        },
                        "objective_weights": {
                            "O": 0.25,
                            "other": 0.5,
                            "given_name": 1.0,
                        },
                    }
                ],
                "subclass_spans": [
                    {
                        "carrier_start": name_start,
                        "carrier_end": name_end,
                        "type": "person_name",
                        "start": name_start,
                        "end": name_start + 5,
                        "family": "name_component",
                        "value": "given_name",
                        "objective_weight": 0.8,
                        "learning_weight": 0.5,
                    },
                    {
                        "carrier_start": name_start,
                        "carrier_end": name_end,
                        "type": "person_name",
                        "start": name_end - 5,
                        "end": name_end,
                        "family": "name_component",
                        "value": "family_name",
                    },
                ],
                "surface_origin": "teacher_generated",
                "predicate_seed": {"method": "reviewed_teacher_direct"},
                "seed_id": "predicate-row-1",
                "ignored_spans": [[reference_start, reference_end, "person_reference"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = window_records(corpus, max_chars=20)
    name_row = next(row for row in rows if row["spans"])
    predicate = name_row["predicate_spans"][0]
    reference_row = next(row for row in rows if row.get("ignored_spans"))

    assert name_row["text"][predicate["start"] : predicate["end"]] == "Lewis Stott"
    assert (
        name_row["text"][predicate["attrs"]["given_name"][0][0] : predicate["attrs"]["given_name"][0][1]]
        == "Lewis"
    )
    assert (
        name_row["text"][predicate["attrs"]["family_name"][0][0] : predicate["attrs"]["family_name"][0][1]]
        == "Stott"
    )
    assert predicate["objective_weights"] == {
        "O": 0.25,
        "other": 0.5,
        "given_name": 1.0,
    }
    given_name, family_name = name_row["subclass_spans"]
    assert name_row["text"][given_name["start"] : given_name["end"]] == "Lewis"
    assert name_row["text"][family_name["start"] : family_name["end"]] == "Stott"
    assert (given_name["carrier_start"], given_name["carrier_end"]) == (
        predicate["start"],
        predicate["end"],
    )
    assert given_name["objective_weight"] == 0.8
    assert given_name["learning_weight"] == 0.5
    assert name_row["primary_span_objective_weights"] == [0.85]
    assert name_row["surface_origin"] == "teacher_generated"
    assert name_row["predicate_seed"] == {"method": "reviewed_teacher_direct"}
    assert name_row["seed_id"] == "predicate-row-1"
    ignored = reference_row["ignored_spans"][0]
    assert reference_row["text"][ignored[0] : ignored[1]] == "the treating nurse"


def test_window_records_preserves_sampling_fields(tmp_path) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": "Ada",
                "lang": "en",
                "spans": [[0, 3, "name"]],
                "sampling_pool": "gold-en",
                "sampling_family": "gold",
                "sampling_weight": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = window_records(corpus, max_chars=100)

    assert rows[0]["sampling_pool"] == "gold-en"
    assert rows[0]["sampling_family"] == "gold"
    assert rows[0]["sampling_weight"] == 0.25


def test_sampling_config_compiles_exact_match_pools_to_row_weights(tmp_path) -> None:
    rows = [
        {"lang": "en", "mix_source": "primary"},
        {"lang": "en", "mix_source": "primary"},
        {"lang": "en", "mix_source": "admitted"},
        {"lang": "de", "mix_source": "admitted"},
    ]
    config = tmp_path / "sampling.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pools": [
                    {
                        "name": "primary-en",
                        "weight": 0.5,
                        "match": {"lang": "en", "mix_source": "primary"},
                    },
                    {
                        "name": "admitted-en",
                        "weight": 0.3,
                        "match": {"lang": "en", "mix_source": "admitted"},
                    },
                    {
                        "name": "admitted-de",
                        "weight": 0.2,
                        "match": {"lang": "de", "mix_source": "admitted"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    weights = sampling_weights_for_rows(rows, config)
    routed_weights, pool_keys = sampling_weights_and_pool_keys_for_rows(rows, config)

    assert weights == pytest.approx([0.25, 0.25, 0.3, 0.2])
    assert routed_weights == pytest.approx(weights)
    assert pool_keys == ["primary-en", "primary-en", "admitted-en", "admitted-de"]


def test_sampling_config_derives_disjoint_predicate_exposure_rows(tmp_path) -> None:
    def predicate_row(seed_id, primary_type, *channels):
        return {
            "text": "x",
            "spans": [[0, 1, primary_type]],
            "lang": "en",
            "sampling_pool": "predicate-v2",
            "seed_id": seed_id,
            "predicate_spans": [
                {
                    "start": 0,
                    "end": 1,
                    "type": primary_type,
                    "attrs": {channel: [[0, 1]] for channel in channels},
                }
            ],
        }

    rows = [
        {"text": "complete", "spans": [], "lang": "en", "sampling_pool": "data"},
        predicate_row("a", "person_reference", "alpha"),
        predicate_row("b", "person_reference", "beta"),
        predicate_row("c", "organization_reference", "alpha"),
        predicate_row("d", "organization_reference", "beta"),
        predicate_row("e", "person_name", "alpha", "beta"),
    ]
    config = tmp_path / "sampling.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predicate_exposure_strata": {
                    "schema": "pii-predicate-exposure-strata",
                    "schema_version": 1,
                    "sampling_pool": "predicate-v2",
                    "field": "sampling_predicate_exposure",
                    "background": "background",
                    "primary_types": ["person_reference", "organization_reference"],
                    "predicate_channels": ["alpha", "beta"],
                },
                "pools": [
                    {"name": "complete", "weight": 0.5, "match": {"sampling_pool": "data"}},
                    *[
                        {
                            "name": target,
                            "weight": 0.1,
                            "match": {"sampling_predicate_exposure": target},
                        }
                        for target in (
                            "primary:person_reference",
                            "primary:organization_reference",
                            "predicate:alpha",
                            "predicate:beta",
                        )
                    ],
                    {
                        "name": "background-en",
                        "weight": 0.1,
                        "match": {
                            "lang": "en",
                            "sampling_pool": "predicate-v2",
                            "sampling_predicate_exposure": "background",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    weights, pool_keys, derivation = sampling_plan_for_rows(rows, config)

    assert sum(weights) == pytest.approx(1.0)
    assert pool_keys[0] == "complete"
    assert Counter(pool_keys[1:]) == {
        "background-en": 1,
        "predicate:alpha": 1,
        "predicate:beta": 1,
        "primary:organization_reference": 1,
        "primary:person_reference": 1,
    }
    receipt = derivation["predicate_exposure_strata"]
    assert receipt["reserved_rows"] == 4
    assert receipt["background_rows"] == 1
    assert len({target["row_identity"] for target in receipt["targets"].values()}) == 4
    assert sampling_plan_for_rows(rows, config)[2] == derivation


def test_declared_subclass_exposure_targets_reads_sampling_pools(tmp_path) -> None:
    config = tmp_path / "sampling.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pools": [
                    {
                        "name": "given",
                        "weight": 0.5,
                        "match": {"sampling_refinement_exposure": "subclass:name_component=given_name"},
                    },
                    {
                        "name": "predicate",
                        "weight": 0.5,
                        "match": {"sampling_refinement_exposure": "bernoulli:patient"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert declared_subclass_exposure_targets(config) == ("name_component=given_name",)


def test_predicate_exposure_derivation_refuses_one_row_for_two_targets(tmp_path) -> None:
    rows = [
        {
            "text": "x",
            "spans": [[0, 1, "person_reference"]],
            "lang": "en",
            "sampling_pool": "predicate-v2",
            "seed_id": "only",
            "predicate_spans": [
                {
                    "start": 0,
                    "end": 1,
                    "type": "person_reference",
                    "attrs": {"alpha": [[0, 1]]},
                }
            ],
        }
    ]
    config = tmp_path / "sampling.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predicate_exposure_strata": {
                    "schema": "pii-predicate-exposure-strata",
                    "schema_version": 1,
                    "sampling_pool": "predicate-v2",
                    "field": "sampling_predicate_exposure",
                    "background": "background",
                    "primary_types": ["person_reference"],
                    "predicate_channels": ["alpha"],
                },
                "pools": [
                    {
                        "name": "unused",
                        "weight": 1.0,
                        "match": {"sampling_pool": "predicate-v2"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be assigned distinct rows"):
        sampling_plan_for_rows(rows, config)


def test_sampling_config_can_preserve_intake_row_mass_after_sentence_split(tmp_path) -> None:
    rows = [
        {"sampling_pool": "mapped", "sampling_intake_factor": 0.5},
        {"sampling_pool": "mapped", "sampling_intake_factor": 0.5},
        {"sampling_pool": "mapped", "sampling_intake_factor": 1.0},
        {"sampling_pool": "native", "sampling_intake_factor": 1.0},
    ]
    config = tmp_path / "sampling.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "example_factor_field": "sampling_intake_factor",
                "pools": [
                    {"name": "mapped", "weight": 0.75, "match": {"sampling_pool": "mapped"}},
                    {"name": "native", "weight": 0.25, "match": {"sampling_pool": "native"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    weights = sampling_weights_for_rows(rows, config)

    assert weights == pytest.approx([0.1875, 0.1875, 0.375, 0.25])
    assert sum(weights[:2]) == pytest.approx(weights[2])


def test_inline_sampling_weights_are_all_or_none() -> None:
    assert sampling_weights_for_rows([{"sampling_weight": 2}, {"sampling_weight": 1}]) == [2.0, 1.0]
    assert sampling_weights_for_rows([{}, {}]) is None
    with pytest.raises(ValueError, match="every training row"):
        sampling_weights_for_rows([{"sampling_weight": 1}, {}])


def test_train_pool_specs_support_simple_weights_or_external_config() -> None:
    assert parse_train_pool_specs(["admitted=/data/admitted.jsonl:0.2"]) == [
        ("admitted", "/data/admitted.jsonl", 0.2)
    ]
    assert parse_train_pool_specs(["admitted=/data/admitted.jsonl"], sampling_config=True) == [
        ("admitted", "/data/admitted.jsonl", None)
    ]
    with pytest.raises(ValueError, match="requires :WEIGHT"):
        parse_train_pool_specs(["admitted=/data/admitted.jsonl"])
    with pytest.raises(ValueError, match="omit :WEIGHT"):
        parse_train_pool_specs(["admitted=/data/admitted.jsonl:0.2"], sampling_config=True)


def test_weighted_trainer_overrides_batch_sampler_and_emits_requested_mass(tmp_path) -> None:
    class ToyDataset(torch.utils.data.Dataset):
        sampling_weights = [0.1, 0.1, 0.4, 0.4]

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"row_index": index}

        def token_lengths(self):
            return [4, 8, 12, 16]

    args = pii_encoder_train.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=11,
    )
    args.sampling_epoch_examples = 20
    args.sampling_length_window_steps = 2
    trainer = WeightedSamplingTrainer(
        model=nn.Linear(1, 1),
        args=args,
        train_dataset=ToyDataset(),
        data_collator=lambda features: {
            "row_index": torch.tensor([feature["row_index"] for feature in features])
        },
    )

    observed = Counter()
    for batch in trainer.get_train_dataloader():
        observed.update(int(index) for index in batch["row_index"])

    assert sum(observed.values()) == 20
    assert sum(observed[index] for index in (0, 1)) == 4
    assert sum(observed[index] for index in (2, 3)) == 16


def test_weighted_trainer_routes_homogeneous_pool_and_objective_batches(tmp_path) -> None:
    class ToyDataset(torch.utils.data.Dataset):
        sampling_weights = [0.125] * 8
        sampling_pool_keys = ["tag"] * 4 + ["mlm"] * 4
        mlm_batch_probabilities = {"tag": 0.0, "mlm": 1.0}

        def __len__(self):
            return 8

        def __getitem__(self, sampled_index):
            index, mlm = sampled_index
            return {
                "row_index": index,
                "pool": int(index >= 4),
                "mlm": int(mlm),
            }

        def token_lengths(self):
            return [4, 5, 6, 7, 20, 21, 22, 23]

    args = pii_encoder_train.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=11,
    )
    args.sampling_epoch_examples = 40
    args.sampling_length_bucket_width = 0
    trainer = WeightedSamplingTrainer(
        model=nn.Linear(1, 1),
        args=args,
        train_dataset=ToyDataset(),
        data_collator=lambda features: {
            key: torch.tensor([feature[key] for feature in features]) for key in ("row_index", "pool", "mlm")
        },
    )

    batches = list(trainer.get_train_dataloader())

    assert len(batches) == 20
    assert all(len(set(batch["pool"].tolist())) == 1 for batch in batches)
    assert all(len(set(batch["mlm"].tolist())) == 1 for batch in batches)
    assert all(batch["pool"][0] == batch["mlm"][0] for batch in batches)


def test_window_records_drops_unlabeled_partial_annotation_windows(tmp_path) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": "Ada",
                        "lang": "en",
                        "spans": [[0, 3, "name"]],
                        "supervision": ANNOTATED_SPANS_ONLY,
                    }
                ),
                json.dumps(
                    {
                        "text": "No annotated entity",
                        "lang": "en",
                        "spans": [],
                        "supervision": ANNOTATED_SPANS_ONLY,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = window_records(corpus, max_chars=100)

    assert len(rows) == 1
    assert rows[0]["text"] == "Ada"
    assert rows[0]["supervision"] == ANNOTATED_SPANS_ONLY


def test_span_dataset_masks_unannotated_partial_tokens_but_not_complete_tokens() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    label2id = {"O": 0, "S-name": 1}
    partial = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
            }
        ],
        Tokenizer(),
        label2id,
        max_len=32,
    )[0]
    complete = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]]}],
        Tokenizer(),
        label2id,
        max_len=32,
    )[0]

    assert partial["labels"] == [-100, 1, -100, -100]
    assert complete["labels"] == [-100, 1, 0, -100]
    assert "boundary_labels" not in partial
    assert "boundary_labels" not in complete


def test_predicate_targets_decompose_a_whole_name_without_false_outside_labels() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "My name is Lewis Francis William Stott"
            return {
                "input_ids": list(range(9)),
                "offset_mapping": [
                    (0, 0),
                    (0, 2),
                    (3, 7),
                    (8, 10),
                    (11, 16),
                    (17, 24),
                    (25, 32),
                    (33, 38),
                    (0, 0),
                ],
            }

    spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    item = SpanDataset(
        [
            {
                "text": "My name is Lewis Francis William Stott",
                "spans": [[11, 38, "person_name"]],
                "predicate_spans": [
                    {
                        "start": 11,
                        "end": 38,
                        "type": "person_name",
                        "attrs": {
                            "given_name": [[11, 16]],
                            "family_name": [[33, 38]],
                        },
                        "objective_weights": {
                            "O": 0.25,
                            "other": 0.5,
                            "given_name": 1.0,
                        },
                    }
                ],
            }
        ],
        Tokenizer(),
        {
            "O": 0,
            "B-person_name": 1,
            "I-person_name": 2,
            "E-person_name": 3,
        },
        max_len=32,
        predicate_spec=spec,
    )[0]

    unknown = [-100.0, -100.0, -100.0]
    assert item["predicate_labels"] == [
        unknown,
        unknown,
        unknown,
        unknown,
        [1.0, 0.0, -100.0],
        [0.0, 0.0, -100.0],
        [0.0, 0.0, -100.0],
        [0.0, 1.0, -100.0],
        unknown,
    ]
    no_weight = [0.0, 0.0, 0.0]
    assert item["predicate_weights"] == [
        no_weight,
        no_weight,
        no_weight,
        no_weight,
        [1.0, 0.25, 0.0],
        [1.0, 0.25, 0.0],
        [1.0, 0.25, 0.0],
        [1.0, 0.5, 0.0],
        no_weight,
    ]


def test_predicate_spec_accepts_legacy_and_declared_channel_token_rules(tmp_path) -> None:
    legacy = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    declared_channel = pii_encoder_train.load_predicate_spec(
        pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH.with_name(
            "pii_ont3_bernoulli_predicate_channels_v1.json"
        )
    )

    assert legacy.channels == ("given_name", "family_name", "care_provider")
    assert declared_channel.channels == (
        "care_provider",
        "patient",
        "family_member",
        "witness_or_bystander",
        "investigator_or_law_enforcement",
        "legal_professional",
    )

    invalid = json.loads(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH.read_text(encoding="utf-8"))
    invalid["target_encoding"]["token_rule"] = "Any token near a carrier is known."
    invalid_path = tmp_path / "invalid-predicate-spec.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="masked token contract"):
        pii_encoder_train.load_predicate_spec(invalid_path)


def test_care_provider_predicate_activates_for_person_references_only_when_gold_type_matches() -> None:
    spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    row = {
        "text": "the treating nurse",
        "spans": [[0, 18, "person_reference"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 18,
                "type": "person_reference",
                "attrs": {"care_provider": [[0, 18]]},
            }
        ],
    }
    labels = pii_encoder_train.predicate_token_labels(
        row,
        [(0, 0), (0, 3), (4, 12), (13, 18), (0, 0)],
        spec,
    )
    assert [target[2] for target in labels] == [-100.0, 1.0, 1.0, 1.0, -100.0]

    row["predicate_spans"][0]["attrs"]["care_provider"] = []
    labels = pii_encoder_train.predicate_token_labels(
        row,
        [(0, 0), (0, 3), (4, 12), (13, 18), (0, 0)],
        spec,
    )
    assert [target[2] for target in labels] == [-100.0, 0.0, 0.0, 0.0, -100.0]

    row["spans"] = [[0, 18, "organization"]]
    with pytest.raises(ValueError, match="does not match an activating primary span"):
        pii_encoder_train.predicate_token_labels(
            row,
            [(0, 0), (0, 3), (4, 12), (13, 18), (0, 0)],
            spec,
        )


def test_predicate_supervision_emits_one_semantic_gold_condition_per_carrier_token() -> None:
    spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    row = {
        "text": "Dr Delgado met witness",
        "spans": [[0, 10, "person_name"], [15, 22, "person_reference"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 10,
                "type": "person_name",
                "attrs": {"care_provider": [[0, 10]]},
            },
            {
                "start": 15,
                "end": 22,
                "type": "person_reference",
                "attrs": {"care_provider": [[15, 22]]},
            },
        ],
    }

    labels, weights, condition_ids = pii_encoder_train.predicate_token_supervision(
        row,
        [(0, 0), (0, 2), (3, 10), (11, 14), (15, 22), (0, 0)],
        spec,
        condition_type2id={"person_name": 0, "person_reference": 1},
    )

    assert condition_ids == [-1, 0, 0, -1, 1, -1]
    assert any(weight > 0 for weight in weights[1])
    assert any(weight > 0 for weight in weights[4])
    assert all(label == -100.0 for label in labels[3])


def test_predicate_supervision_retains_masked_targets_with_zero_objective_weight() -> None:
    spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    row = {
        "text": "Dr Delgado",
        "spans": [[0, 10, "person_name"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 10,
                "type": "person_name",
                "attrs": {"care_provider": [[0, 10]]},
            }
        ],
    }

    labels, weights = pii_encoder_train.predicate_token_supervision(
        row,
        [(0, 0), (0, 2), (3, 10), (0, 0)],
        spec,
        objective_mask_channels=frozenset({"care_provider"}),
    )

    channel = spec.channels.index("care_provider")
    assert labels[1][channel] == 1.0
    assert labels[2][channel] == 1.0
    assert weights[1][channel] == 0.0
    assert weights[2][channel] == 0.0


def test_predicate_supervision_rejects_unknown_objective_mask_channel() -> None:
    spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)

    with pytest.raises(ValueError, match="unknown predicate objective mask"):
        pii_encoder_train.predicate_token_supervision(
            {"text": "x", "spans": [], "predicate_spans": []},
            [(0, 1)],
            spec,
            objective_mask_channels=frozenset({"not_a_channel"}),
        )


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ({"O": 1.0}, "must contain O and other"),
        ({"O": 1.0, "other": 1.0, "family_name": 1.0}, "unknown targets"),
        ({"O": -1.0, "other": 1.0}, "finite and nonnegative"),
    ],
)
def test_predicate_weight_profiles_fail_closed(profile, message) -> None:
    spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    row = {
        "text": "Dr Delgado",
        "spans": [[0, 10, "person_name"]],
        "predicate_spans": [
            {
                "start": 0,
                "end": 10,
                "type": "person_name",
                "attrs": {"care_provider": [[0, 10]]},
                "objective_weights": profile,
            }
        ],
    }

    with pytest.raises(ValueError, match=message):
        pii_encoder_train.predicate_token_supervision(
            row,
            [(0, 0), (0, 2), (3, 10), (0, 0)],
            spec,
        )


def test_span_dataset_projects_primary_span_objective_weights_to_tokens() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "She spoke"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 9), (0, 0)],
            }

    rows = [
        {
            "text": "She spoke",
            "spans": [[0, 3, "person_reference"]],
            "primary_span_objective_weights": [0.85],
            "supervision": ANNOTATED_SPANS_ONLY,
        },
        {
            "text": "She spoke",
            "spans": [[0, 3, "person_reference"]],
            "supervision": "complete",
        },
    ]
    dataset = SpanDataset(
        rows,
        Tokenizer(),
        {"O": 0, "S-person_reference": 1},
        max_len=16,
    )

    assert dataset[0]["primary_objective_weights"] == [0.0, 0.85, 0.0, 0.0]
    assert dataset[1]["primary_objective_weights"] == [0.0, 1.0, 1.0, 0.0]

    balanced = SpanDataset(
        rows,
        Tokenizer(),
        {"O": 0, "S-person_reference": 1},
        max_len=16,
        primary_type_learning_weights={"person_reference": 4.0},
    )
    assert balanced[0]["primary_objective_weights"] == [0.0, 3.4, 0.0, 0.0]
    assert balanced[1]["primary_objective_weights"] == [0.0, 4.0, 1.0, 0.0]

    attenuated = SpanDataset(
        rows,
        Tokenizer(),
        {"O": 0, "S-person_reference": 1},
        max_len=16,
        primary_type_learning_weights={"person_reference": 4.0},
        partial_primary_objective_weight=0.25,
    )
    assert attenuated[0]["primary_objective_weights"] == [0.0, 0.85, 0.0, 0.0]
    assert attenuated[1]["primary_objective_weights"] == [0.0, 4.0, 1.0, 0.0]

    complete_primary_control = SpanDataset(
        rows,
        Tokenizer(),
        {"O": 0, "S-person_reference": 1},
        max_len=16,
        partial_primary_objective_weight=0.0,
    )
    assert complete_primary_control[0]["primary_objective_weights"] == [0.0, 0.0, 0.0, 0.0]
    assert complete_primary_control[1]["primary_objective_weights"] == [0.0, 1.0, 1.0, 0.0]


def test_span_dataset_rejects_partial_row_without_token_aligned_span() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    dataset = SpanDataset(
        [
            {
                "id": "teacher-row-7",
                "text": "Ada note",
                "spans": [[20, 23, "name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
            }
        ],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
    )

    with pytest.raises(
        ValueError,
        match="annotated-spans-only row 'teacher-row-7' has no token-aligned span after truncation",
    ):
        dataset[0]


def test_span_dataset_can_render_same_stored_row_as_tag_or_mlm_view() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            result = {"input_ids": [0, 1, 2, 3]}
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = [(0, 0), (0, 3), (4, 8), (0, 0)]
            return result

    dataset = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]]}],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
        joint_objectives=True,
    )

    tag_view = dataset[(0, False)]
    mlm_view = dataset[(0, True)]

    assert tag_view["pii_objective"] == "tag"
    assert tag_view["labels"] == [-100, 1, 0, -100]
    assert mlm_view == {"input_ids": [0, 1, 2, 3], "pii_objective": "mlm"}


def test_span_dataset_projects_raw_characters_onto_token_offsets() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    projection = build_literal_character_projection(["Ada note"])
    item = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]]}],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
        character_projection=projection,
    )[0]

    assert len(item["character_ids"]) == len("Ada note")
    assert item["character_mask"] == [True] * len("Ada note")
    assert item["token_offsets"] == [[0, 0], [0, 3], [4, 8], [0, 0]]
    assert item["character_ids"][3] == list(projection.pair(" "))


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_continuous_character_collator_pads_both_lattices(padding_side) -> None:
    class Tokenizer:
        pass

    tokenizer = Tokenizer()
    tokenizer.padding_side = padding_side

    def base(features):
        target = max(len(feature["input_ids"]) for feature in features)
        padded = []
        for feature in features:
            padding = [0] * (target - len(feature["input_ids"]))
            padded.append(
                feature["input_ids"] + padding if padding_side == "right" else padding + feature["input_ids"]
            )
        return {"input_ids": torch.tensor(padded)}

    collator = ContinuousCharacterDataCollator(tokenizer, base)
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "character_ids": [[4, 4], [5, 5]],
                "character_mask": [True, True],
                "token_offsets": [[0, 0], [0, 2], [0, 0]],
            },
            {
                "input_ids": [1, 3],
                "character_ids": [[6, 6]],
                "character_mask": [True],
                "token_offsets": [[0, 1], [0, 0]],
            },
        ]
    )

    assert batch["character_ids"].tolist() == [[[4, 4], [5, 5]], [[6, 6], [0, 0]]]
    assert batch["character_mask"].tolist() == [[True, True], [True, False]]
    expected_second_offsets = [[0, 0], [0, 1], [0, 0]] if padding_side == "left" else [[0, 1], [0, 0], [0, 0]]
    assert batch["token_offsets"][1].tolist() == expected_second_offsets


def test_span_dataset_realizes_selected_occurrence_before_tokenization() -> None:
    class Realizer:
        def __init__(self):
            self.calls = []

        def realize(self, row, draw_nonce):
            self.calls.append((row["text"], draw_nonce))
            return {**row, "text": "Grace note", "spans": [[0, 5, "name"]]}

    class Tokenizer:
        def __init__(self):
            self.texts = []

        def __call__(self, text, **kwargs):
            self.texts.append(text)
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 5), (6, 10), (0, 0)],
            }

    realizer = Realizer()
    tokenizer = Tokenizer()
    dataset = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]]}],
        tokenizer,
        {"O": 0, "S-name": 1},
        max_len=32,
        surface_realizer=realizer,
    )

    item = dataset[(0, None, 17)]

    assert realizer.calls == [("Ada note", 17)]
    assert tokenizer.texts == ["Grace note"]
    assert item["labels"] == [-100, 1, 0, -100]
    with pytest.raises(ValueError, match="requires a sampler draw nonce"):
        dataset[0]


def test_span_dataset_emits_boundary_labels_only_for_partial_span_tokens() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    label2id = {"O": 0, "S-name": 1}
    rows = [
        {
            "text": "Ada note",
            "spans": [[0, 3, "name"]],
            "supervision": ANNOTATED_SPANS_ONLY,
        }
    ]
    partial = SpanDataset(
        rows,
        Tokenizer(),
        label2id,
        max_len=32,
        include_boundary_labels=True,
    )[0]
    complete = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]]}],
        Tokenizer(),
        label2id,
        max_len=32,
        include_boundary_labels=True,
    )[0]

    assert partial["boundary_labels"] == [-100, BOUNDARY2ID["S"], -100, -100]
    assert complete["boundary_labels"] == [-100, -100, -100, -100]


def test_span_dataset_masks_complete_tokens_with_ignored_annotations() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada met applicant"
            return {
                "input_ids": [0, 1, 2, 3, 4],
                "offset_mapping": [(0, 0), (0, 3), (4, 7), (8, 17), (0, 0)],
            }

    item = SpanDataset(
        [
            {
                "text": "Ada met applicant",
                "spans": [[0, 3, "name"]],
                "ignored_spans": [[8, 17, "person_reference"]],
            }
        ],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
        include_complete_presence_labels=True,
    )[0]

    assert item["labels"] == [-100, 1, 0, -100, -100]
    assert item["complete_presence_labels"] == [-100, 1, 0, -100, -100]


def test_span_dataset_rejects_ignored_annotation_overlapping_supervision() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": [0, 1, 2],
                "offset_mapping": [(0, 0), (0, 3), (0, 0)],
            }

    dataset = SpanDataset(
        [
            {
                "text": "Ada",
                "spans": [[0, 3, "name"]],
                "ignored_spans": [[0, 3, "person_reference"]],
            }
        ],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
    )

    with pytest.raises(ValueError, match="overlaps supervised name span"):
        dataset[0]


def test_span_dataset_emits_retention_mask_only_outside_partial_gold() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    label2id = {"O": 0, "S-name": 1}
    partial = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
            }
        ],
        Tokenizer(),
        label2id,
        max_len=32,
        include_consistency_mask=True,
    )[0]
    complete = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]]}],
        Tokenizer(),
        label2id,
        max_len=32,
        include_consistency_mask=True,
    )[0]

    assert partial["consistency_mask"] == [0, 0, 1, 0]
    assert complete["consistency_mask"] == [0, 0, 0, 0]


def test_span_dataset_emits_pu_masks_for_mapped_partial_spans() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    assignments = {("teacher", "en"): 0.25}
    partial = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "person_name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
                "label_space": "v2",
                "lang": "en",
                "mix_source": "teacher",
            }
        ],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
        secondary_label2id={"O": 0, "S-person_name": 1},
        include_consistency_mask=True,
        partial_entity_pu_assignments=assignments,
    )[0]
    complete = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]], "lang": "en", "src": "gold"}],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
        include_consistency_mask=True,
        partial_entity_pu_assignments=assignments,
    )[0]

    assert partial["secondary_labels"] == [-100, 1, -100, -100]
    assert partial["partial_entity_positive_mask"] == [0, 1, 0, 0]
    assert partial["consistency_mask"] == [0, 0, 1, 0]
    assert partial["partial_entity_pu_group"] == 0
    assert partial["partial_entity_pu_prior"] == 0.25
    assert complete["partial_entity_positive_mask"] == [0, 0, 0, 0]
    assert complete["partial_entity_pu_group"] == -1
    assert complete["partial_entity_pu_prior"] == 0.0


def test_span_dataset_emits_expected_entity_ratio_masks_and_prior_metadata() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    assignments = {("teacher", "en"): 0.25}
    item = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "person_name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
                "label_space": "v2",
                "lang": "en",
                "mix_source": "teacher",
            }
        ],
        Tokenizer(),
        {"O": 0, "S-name": 1},
        max_len=32,
        secondary_label2id={"O": 0, "S-person_name": 1},
        include_consistency_mask=True,
        partial_entity_ratio_assignments=assignments,
    )[0]

    assert item["partial_entity_positive_mask"] == [0, 1, 0, 0]
    assert item["consistency_mask"] == [0, 0, 1, 0]
    assert item["partial_entity_ratio_group"] == 0
    assert item["partial_entity_ratio_prior"] == 0.25
    assert "partial_entity_pu_group" not in item


def test_partial_label_collator_preserves_pu_row_metadata_and_pads_masks() -> None:
    collator = PartialLabelDataCollator.__new__(PartialLabelDataCollator)
    collator.tokenizer = SimpleNamespace(padding_side="right")
    collator.base = lambda features: {
        "input_ids": torch.tensor(
            [feature["input_ids"] + [0] * (3 - len(feature["input_ids"])) for feature in features]
        )
    }

    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "partial_entity_positive_mask": [0, 1],
                "partial_entity_pu_group": 3,
                "partial_entity_pu_prior": 0.2,
            },
            {
                "input_ids": [1, 2, 3],
                "partial_entity_positive_mask": [0, 1, 0],
                "partial_entity_pu_group": 4,
                "partial_entity_pu_prior": 0.3,
            },
        ]
    )

    assert batch["partial_entity_positive_mask"].tolist() == [[0, 1, 0], [0, 1, 0]]
    assert batch["partial_entity_pu_group"].dtype == torch.long
    assert batch["partial_entity_pu_group"].tolist() == [3, 4]
    assert batch["partial_entity_pu_prior"].dtype == torch.float
    assert batch["partial_entity_pu_prior"].tolist() == pytest.approx([0.2, 0.3])


def test_partial_label_collator_preserves_expected_entity_ratio_metadata() -> None:
    collator = PartialLabelDataCollator.__new__(PartialLabelDataCollator)
    collator.tokenizer = SimpleNamespace(padding_side="right")
    collator.base = lambda features: {
        "input_ids": torch.tensor(
            [feature["input_ids"] + [0] * (3 - len(feature["input_ids"])) for feature in features]
        )
    }

    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "partial_entity_positive_mask": [0, 1],
                "partial_entity_ratio_group": 3,
                "partial_entity_ratio_prior": 0.2,
            },
            {
                "input_ids": [1, 2, 3],
                "partial_entity_positive_mask": [0, 1, 0],
                "partial_entity_ratio_group": 4,
                "partial_entity_ratio_prior": 0.3,
            },
        ]
    )

    assert batch["partial_entity_positive_mask"].tolist() == [[0, 1, 0], [0, 1, 0]]
    assert batch["partial_entity_ratio_group"].dtype == torch.long
    assert batch["partial_entity_ratio_group"].tolist() == [3, 4]
    assert batch["partial_entity_ratio_prior"].dtype == torch.float
    assert batch["partial_entity_ratio_prior"].tolist() == pytest.approx([0.2, 0.3])


def test_partial_label_collator_pads_predicate_matrices_as_unknown() -> None:
    collator = PartialLabelDataCollator.__new__(PartialLabelDataCollator)
    collator.tokenizer = SimpleNamespace(padding_side="right")
    collator.base = lambda features: {
        "input_ids": torch.tensor(
            [feature["input_ids"] + [0] * (3 - len(feature["input_ids"])) for feature in features]
        )
    }

    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "predicate_labels": [[-100.0, -100.0], [1.0, 0.0]],
                "predicate_weights": [[0.0, 0.0], [1.0, 0.25]],
                "reference_type_labels": [[-100.0, -100.0], [0.0, 1.0]],
                "reference_type_weights": [[0.0, 0.0], [0.5, 2.0]],
                "predicate_condition_ids": [-1, 2],
            },
            {
                "input_ids": [1, 2, 3],
                "predicate_labels": [[-100.0, -100.0], [0.0, 1.0], [-100.0, -100.0]],
                "predicate_weights": [[0.0, 0.0], [0.25, 0.5], [0.0, 0.0]],
                "reference_type_labels": [
                    [-100.0, -100.0],
                    [1.0, 0.0],
                    [-100.0, -100.0],
                ],
                "reference_type_weights": [[0.0, 0.0], [1.0, 0.75], [0.0, 0.0]],
                "predicate_condition_ids": [-1, 1, -1],
            },
        ]
    )

    assert batch["predicate_labels"].tolist() == [
        [[-100.0, -100.0], [1.0, 0.0], [-100.0, -100.0]],
        [[-100.0, -100.0], [0.0, 1.0], [-100.0, -100.0]],
    ]
    assert batch["predicate_weights"].tolist() == [
        [[0.0, 0.0], [1.0, 0.25], [0.0, 0.0]],
        [[0.0, 0.0], [0.25, 0.5], [0.0, 0.0]],
    ]
    assert batch["predicate_condition_ids"].tolist() == [[-1, 2, -1], [-1, 1, -1]]
    assert batch["reference_type_labels"].tolist() == [
        [[-100.0, -100.0], [0.0, 1.0], [-100.0, -100.0]],
        [[-100.0, -100.0], [1.0, 0.0], [-100.0, -100.0]],
    ]
    assert batch["reference_type_weights"].tolist() == [
        [[0.0, 0.0], [0.5, 2.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.75], [0.0, 0.0]],
    ]


def test_reference_type_residual_supervision_distinguishes_complete_partial_and_legacy() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Alice met counsel"
            return {
                "input_ids": [0, 1, 2, 3, 4],
                "offset_mapping": [(0, 0), (0, 5), (6, 9), (10, 17), (0, 0)],
            }

    old_labels = {"O": 0, "S-person_name": 1}
    new_labels = {
        "O": 0,
        "S-person_name": 1,
        "S-organization_reference": 2,
        "S-person_reference": 3,
    }
    reference_types = ("organization_reference", "person_reference")
    common = {
        "text": "Alice met counsel",
        "spans": [[0, 5, "person_reference"], [10, 17, "person_name"]],
        "label_space": "v2",
        "primary_span_objective_weights": [0.5, 0.8],
    }

    complete = SpanDataset(
        [common],
        Tokenizer(),
        old_labels,
        max_len=32,
        secondary_label2id=new_labels,
        reference_type_residual_types=reference_types,
    )[0]
    partial = SpanDataset(
        [{**common, "supervision": ANNOTATED_SPANS_ONLY}],
        Tokenizer(),
        old_labels,
        max_len=32,
        secondary_label2id=new_labels,
        reference_type_residual_types=reference_types,
    )[0]
    legacy = SpanDataset(
        [
            {
                "text": "Alice met counsel",
                "spans": [[0, 5, "person_name"]],
                "unknown_primary_types": list(reference_types),
                "primary_span_objective_weights": [0.7],
            }
        ],
        Tokenizer(),
        old_labels,
        max_len=32,
        secondary_label2id=new_labels,
        legacy_outside_unknown_primary_types=reference_types,
        reference_type_residual_types=reference_types,
    )[0]

    assert complete["reference_type_labels"] == [
        [-100.0, -100.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-100.0, -100.0],
    ]
    assert complete["reference_type_weights"] == [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.8, 0.8],
        [0.0, 0.0],
    ]
    assert partial["reference_type_labels"] == [
        [-100.0, -100.0],
        [0.0, 1.0],
        [-100.0, -100.0],
        [0.0, 0.0],
        [-100.0, -100.0],
    ]
    assert legacy["reference_type_labels"] == [
        [-100.0, -100.0],
        [0.0, 0.0],
        [-100.0, -100.0],
        [-100.0, -100.0],
        [-100.0, -100.0],
    ]
    assert legacy["reference_type_weights"][1] == [0.7, 0.7]


def test_span_dataset_emits_entity_presence_only_for_complete_rows() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    label2id = {"O": 0, "S-name": 1}
    common = {
        "text": "Ada note",
        "spans": [[0, 3, "name"]],
    }
    complete = SpanDataset(
        [common],
        Tokenizer(),
        label2id,
        max_len=32,
        include_complete_presence_labels=True,
    )[0]
    partial = SpanDataset(
        [{**common, "supervision": ANNOTATED_SPANS_ONLY}],
        Tokenizer(),
        label2id,
        max_len=32,
        include_complete_presence_labels=True,
    )[0]

    assert complete["complete_presence_labels"] == [-100, 1, 0, -100]
    assert partial["complete_presence_labels"] == [-100, -100, -100, -100]


def test_span_dataset_emits_family_targets_only_for_complete_rows() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    label2id = {"O": 0, "S-name": 1}
    common = {
        "text": "Ada note",
        "spans": [[0, 3, "name"]],
    }
    complete = SpanDataset(
        [common],
        Tokenizer(),
        label2id,
        max_len=32,
        include_complete_family_labels=True,
        family_target_by_own_label={1: 3},
        family_target_by_secondary_label={},
    )[0]
    masked = SpanDataset(
        [common],
        Tokenizer(),
        label2id,
        max_len=32,
        include_complete_family_labels=True,
        family_target_by_own_label={},
        family_target_by_secondary_label={},
    )[0]
    partial = SpanDataset(
        [{**common, "supervision": ANNOTATED_SPANS_ONLY}],
        Tokenizer(),
        label2id,
        max_len=32,
        include_complete_family_labels=True,
        family_target_by_own_label={1: 3},
        family_target_by_secondary_label={},
    )[0]

    assert complete["complete_family_labels"] == [-100, 3, 0, -100]
    assert masked["complete_family_labels"] == [-100, -100, 0, -100]
    assert partial["complete_family_labels"] == [-100, -100, -100, -100]

    with pytest.raises(ValueError, match="family target tables"):
        SpanDataset(
            [common],
            Tokenizer(),
            label2id,
            max_len=32,
            include_complete_family_labels=True,
        )


def test_span_dataset_emits_complete_entity_boundaries_from_product_labels() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    direct = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "locality"]]}],
        Tokenizer(),
        {"O": 0, "S-locality": 1},
        max_len=32,
        include_complete_boundary_labels=True,
    )[0]
    mapped = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "locality"]],
                "label_space": "v2",
            }
        ],
        Tokenizer(),
        {"O": 0, "S-city": 1},
        max_len=32,
        secondary_label2id={"O": 0, "S-locality": 1},
        include_complete_boundary_labels=True,
    )[0]
    partial = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "locality"]],
                "supervision": ANNOTATED_SPANS_ONLY,
            }
        ],
        Tokenizer(),
        {"O": 0, "S-locality": 1},
        max_len=32,
        include_complete_boundary_labels=True,
    )[0]

    expected = [-100, BOUNDARY2ID["S"], -100, -100]
    assert direct["complete_boundary_labels"] == expected
    assert mapped["complete_boundary_labels"] == expected
    assert partial["complete_boundary_labels"] == [-100, -100, -100, -100]


def test_span_dataset_emits_source_group_only_outside_matching_partial_gold() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Ada note"
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],
            }

    label2id = {"O": 0, "S-name": 1}
    source_groups = {"aqmar-natural": 0}
    matching = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
                "src": "aqmar-natural",
            }
        ],
        Tokenizer(),
        label2id,
        max_len=32,
        partial_negative_source_groups=source_groups,
    )[0]
    different_source = SpanDataset(
        [
            {
                "text": "Ada note",
                "spans": [[0, 3, "name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
                "src": "other",
            }
        ],
        Tokenizer(),
        label2id,
        max_len=32,
        partial_negative_source_groups=source_groups,
    )[0]
    complete = SpanDataset(
        [{"text": "Ada note", "spans": [[0, 3, "name"]], "src": "aqmar-natural"}],
        Tokenizer(),
        label2id,
        max_len=32,
        partial_negative_source_groups=source_groups,
    )[0]

    assert matching["partial_negative_groups"] == [-1, -1, 0, -1]
    assert different_source["partial_negative_groups"] == [-1, -1, -1, -1]
    assert complete["partial_negative_groups"] == [-1, -1, -1, -1]


def test_collapsed_boundary_probabilities_equal_summed_fine_probabilities() -> None:
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-location",
        "I-location",
        "E-location",
        "S-location",
    ]
    logits = torch.tensor([[[0.1, 0.2, -0.4, 0.7, 0.3, -0.2, 0.6, 0.5, -0.1]]])

    collapsed = collapse_bioes_logits(logits, labels)
    fine_probabilities = torch.softmax(logits.float(), dim=-1)
    expected = torch.stack(
        [
            fine_probabilities[..., [0]].sum(dim=-1),
            fine_probabilities[..., [1, 5]].sum(dim=-1),
            fine_probabilities[..., [2, 6]].sum(dim=-1),
            fine_probabilities[..., [3, 7]].sum(dim=-1),
            fine_probabilities[..., [4, 8]].sum(dim=-1),
        ],
        dim=-1,
    )

    assert torch.allclose(torch.softmax(collapsed, dim=-1), expected)


def test_annotated_boundary_loss_masks_unannotated_and_complete_tokens() -> None:
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    logits = torch.tensor(
        [
            [[0.0, 2.0, 0.0, 0.0, 0.0], [10.0, -10.0, -10.0, -10.0, -10.0]],
            [[10.0, -10.0, -10.0, -10.0, -10.0], [10.0, -10.0, -10.0, -10.0, -10.0]],
        ],
        requires_grad=True,
    )
    boundary_labels = torch.tensor(
        [
            [BOUNDARY2ID["B"], -100],
            [-100, -100],
        ]
    )

    loss = annotated_boundary_loss(logits, boundary_labels, labels)
    expected = torch.nn.functional.cross_entropy(
        collapse_bioes_logits(logits[:, :1], labels)[0],
        torch.tensor([BOUNDARY2ID["B"]]),
    )
    loss.backward()

    assert torch.allclose(loss, expected)
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_annotated_boundary_loss_is_exact_zero_without_partial_tokens() -> None:
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    logits = torch.randn(2, 3, len(labels), requires_grad=True)
    boundary_labels = torch.full((2, 3), -100)

    loss = annotated_boundary_loss(logits, boundary_labels, labels)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_partial_boundary_retention_kl_only_updates_unannotated_student_token() -> None:
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    student = torch.tensor(
        [[[2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[0.0, 2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    consistency_mask = torch.tensor([[0, 1]])

    loss = partial_boundary_retention_loss(
        student,
        teacher,
        consistency_mask,
        labels,
        temperature=2.0,
    )
    loss.backward()

    assert loss.item() > 0
    assert torch.count_nonzero(student.grad[0, 0]) == 0
    assert torch.count_nonzero(student.grad[0, 1]) > 0
    assert teacher.grad is None


def test_partial_boundary_retention_is_zero_without_unannotated_tokens() -> None:
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    student = torch.randn(2, 3, len(labels), requires_grad=True)
    teacher = torch.randn_like(student)

    loss = partial_boundary_retention_loss(
        student,
        teacher,
        torch.zeros(2, 3, dtype=torch.long),
        labels,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(student.grad) == 0


def test_partial_o_loss_only_updates_unannotated_tokens_toward_o() -> None:
    logits = torch.tensor(
        [[[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[0, 1, 0]])

    loss = partial_o_loss(logits, mask, o_label_id=0)
    loss.backward()

    assert loss.item() > 0
    assert logits.grad[0, 1, 0] < 0
    assert torch.count_nonzero(logits.grad[0, [0, 2]]) == 0


def test_partial_o_loss_is_zero_without_unannotated_tokens() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)

    loss = partial_o_loss(logits, torch.zeros(2, 3, dtype=torch.long), o_label_id=0)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_partial_negative_loss_only_rejects_covered_types_on_selected_tokens() -> None:
    logits = torch.zeros(1, 2, 6, requires_grad=True)
    group_ids = torch.tensor([[-1, 0]])
    prohibited_groups = [[1, 2, 3, 4]]

    loss = partial_negative_loss(logits, group_ids, prohibited_groups)
    loss.backward()
    base_loss = loss.detach()

    prohibited_high = logits.detach().clone()
    prohibited_high[0, 1, 1] = 3.0
    unrelated_high = logits.detach().clone()
    unrelated_high[0, 1, 5] = 3.0

    assert partial_negative_loss(prohibited_high, group_ids, prohibited_groups) > base_loss
    assert partial_negative_loss(unrelated_high, group_ids, prohibited_groups) < base_loss
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1]) > 0


def test_partial_negative_loss_is_zero_without_matching_tokens() -> None:
    logits = torch.randn(2, 3, 6, requires_grad=True)

    loss = partial_negative_loss(logits, torch.full((2, 3), -1), [[1, 2, 3, 4]])
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_loss_logit_gradient_norm_does_not_accumulate_or_consume_graph() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))

    measured = loss_logit_gradient_norm(loss, logits)

    assert measured > 0
    assert logits.grad is None
    loss.backward()
    assert torch.count_nonzero(logits.grad) > 0


def test_token_classification_loss_exposes_returned_logit_gradient() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, -100], [2, 4, 3]])

    loss = token_classification_loss(logits, labels)
    measured = loss_logit_gradient_norm(loss, logits)

    assert measured > 0
    assert torch.isfinite(loss)
    assert logits.grad is None


def test_character_auxiliary_loss_fades_without_reaching_incumbent_logits() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            loss = token_classification_loss(model.logits, inputs["labels"])
            outputs = SimpleNamespace(
                logits=model.logits,
                character_logits=model.character_logits,
            )
            return (loss, outputs) if return_outputs else loss

    class CharacterTrainer(pii_encoder_train.CharacterAuxiliaryLossMixin, BaseTrainer):
        pass

    model = SimpleNamespace(
        training=True,
        logits=torch.tensor([[[0.0, 2.0]]], requires_grad=True),
        character_logits=torch.tensor([[[2.0, 0.0]]], requires_grad=True),
    )
    labels = torch.tensor([[1]])
    trainer = CharacterTrainer()
    trainer.state = SimpleNamespace(global_step=0)
    trainer.character_auxiliary_loss_weight = 0.5
    trainer.character_auxiliary_loss_fade_steps = 10

    start = trainer.compute_loss(model, {"labels": labels})
    expected_start = token_classification_loss(model.logits, labels) + 0.5 * token_classification_loss(
        model.character_logits,
        labels,
    )
    torch.testing.assert_close(start, expected_start)

    trainer.state.global_step = 5
    midpoint = trainer.compute_loss(model, {"labels": labels})
    expected_midpoint = token_classification_loss(
        model.logits,
        labels,
    ) + 0.25 * token_classification_loss(model.character_logits, labels)
    torch.testing.assert_close(midpoint, expected_midpoint)

    trainer.state.global_step = 10
    end = trainer.compute_loss(model, {"labels": labels})
    torch.testing.assert_close(end, token_classification_loss(model.logits, labels))


def test_coarse_agreement_can_replace_fine_label_loss() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            loss = token_classification_loss(model.logits, inputs["labels"])
            outputs = SimpleNamespace(logits=model.logits)
            return (loss, outputs) if return_outputs else loss

    class CoarseTrainer(pii_encoder_train.CoarseAgreementMixin, BaseTrainer):
        pass

    logits = torch.tensor([[[0.0, -4.0, 2.0]]], requires_grad=True)
    model = SimpleNamespace(training=True, logits=logits)
    labels = torch.tensor([[1]])
    trainer = CoarseTrainer()
    trainer.coarse_agreement = [(1.0, [[0], [1, 2]], torch.tensor([0, 1, 1]))]
    trainer.fine_label_loss_weight = 0.0

    loss = trainer.compute_loss(model, {"labels": labels})
    expected = pii_encoder_train.coarse_agreement_loss(
        logits,
        labels,
        [[0], [1, 2]],
        torch.tensor([0, 1, 1]),
    )

    torch.testing.assert_close(loss, expected)
    assert loss < token_classification_loss(logits, labels)

    model.training = False
    eval_loss = trainer.compute_loss(model, {"labels": labels})
    torch.testing.assert_close(eval_loss, expected)


def test_default_coarse_agreement_blend_preserves_historical_training_loss() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            loss = token_classification_loss(model.logits, inputs["labels"])
            outputs = SimpleNamespace(logits=model.logits)
            return (loss, outputs) if return_outputs else loss

    class CoarseTrainer(pii_encoder_train.CoarseAgreementMixin, BaseTrainer):
        pass

    logits = torch.tensor([[[0.0, -4.0, 2.0]]])
    model = SimpleNamespace(training=True, logits=logits)
    labels = torch.tensor([[1]])
    trainer = CoarseTrainer()
    trainer.coarse_agreement = [(1.0, [[0], [1, 2]], torch.tensor([0, 1, 1]))]

    loss = trainer.compute_loss(model, {"labels": labels})
    expected = (
        token_classification_loss(logits, labels)
        + pii_encoder_train.coarse_agreement_loss(
            logits,
            labels,
            [[0], [1, 2]],
            torch.tensor([0, 1, 1]),
        )
    ) / 2

    torch.testing.assert_close(loss, expected)


def test_mean_coarse_agreement_does_not_reward_duplicate_components() -> None:
    logits = torch.zeros((1, 1, 3))
    groups = [[0], [1, 2]]
    remap = torch.tensor([0, 1, 1])

    singleton_target = pii_encoder_train.coarse_agreement_loss(
        logits,
        torch.tensor([[0]]),
        groups,
        remap,
        normalize_groups=True,
    )
    duplicate_target = pii_encoder_train.coarse_agreement_loss(
        logits,
        torch.tensor([[1]]),
        groups,
        remap,
        normalize_groups=True,
    )

    torch.testing.assert_close(singleton_target, duplicate_target)
    torch.testing.assert_close(singleton_target, torch.tensor(math.log(2.0)))


def test_entity_dice_loss_rewards_entity_geometry_and_ignores_masked_tokens() -> None:
    labels = torch.tensor([[0, 1, -100]])
    good_logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]]], requires_grad=True)
    bad_logits = torch.tensor([[[-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0]]])

    good_loss = entity_dice_loss(good_logits, labels)
    bad_loss = entity_dice_loss(bad_logits, labels)
    good_loss.backward()

    assert good_loss < bad_loss
    assert torch.count_nonzero(good_logits.grad[0, :2]) > 0
    assert torch.count_nonzero(good_logits.grad[0, 2]) == 0


def test_entity_dice_loss_penalizes_false_entities_on_empty_supervision() -> None:
    labels = torch.zeros(1, 3, dtype=torch.long)
    all_o = torch.tensor([[[8.0, -8.0]] * 3])
    all_entity = torch.tensor([[[-8.0, 8.0]] * 3])

    assert entity_dice_loss(all_o, labels) < entity_dice_loss(all_entity, labels)


def test_entity_presence_loss_projects_all_fine_entity_mass_and_masks_unknowns() -> None:
    labels = torch.tensor([[0, 1, -100]])
    good_logits = torch.tensor(
        [[[8.0, -8.0, -8.0], [-8.0, 7.0, 7.0], [-8.0, 8.0, 8.0]]],
        requires_grad=True,
    )
    bad_logits = torch.tensor([[[-8.0, 8.0, 8.0], [8.0, -8.0, -8.0], [8.0, -8.0, -8.0]]])

    good_loss = entity_presence_loss(good_logits, labels, o_label_id=0)
    bad_loss = entity_presence_loss(bad_logits, labels, o_label_id=0)
    good_loss.backward()

    assert good_loss < bad_loss
    assert torch.count_nonzero(good_logits.grad[0, :2]) > 0
    assert torch.count_nonzero(good_logits.grad[0, 2]) == 0


def test_nonnegative_pu_entity_loss_clips_negative_risk_and_ignores_complete_rows() -> None:
    logits = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[-8.0, 8.0], [-8.0, 8.0]],
        ],
        requires_grad=True,
    )
    positive_mask = torch.tensor([[1, 0], [0, 0]])
    unlabeled_mask = torch.tensor([[0, 1], [0, 0]])

    risk = nonnegative_pu_entity_loss(
        logits,
        positive_mask,
        unlabeled_mask,
        torch.tensor([0, -1]),
        torch.tensor([0.25, 0.0]),
        o_label_id=0,
    )
    expected = 0.25 * torch.nn.functional.softplus(torch.tensor(-1.0))
    risk.loss.backward()

    torch.testing.assert_close(entity_presence_log_odds(logits.detach(), 0)[0], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(risk.loss, expected)
    assert risk.negative_risk.item() == 0.0
    assert risk.unclipped_negative_risk.item() < 0.0
    assert risk.positive_tokens == risk.unlabeled_tokens == risk.groups == risk.clipped_groups == 1
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_pu_positive_margin_stops_positive_pressure_after_margin() -> None:
    logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]], requires_grad=True)
    risk = nonnegative_pu_entity_loss(
        logits,
        torch.tensor([[1, 0]]),
        torch.tensor([[0, 1]]),
        torch.tensor([0]),
        torch.tensor([0.2]),
        o_label_id=0,
        positive_margin=1.0,
    )
    risk.loss.backward()

    assert risk.loss.item() == 0.0
    assert risk.clipped_groups == 1
    assert torch.count_nonzero(logits.grad) == 0


def test_expected_entity_ratio_loss_is_zero_inside_interval_and_masks_complete_rows() -> None:
    probabilities = torch.tensor(
        [
            [[0.8, 0.1, 0.1], [0.8, 0.15, 0.05], [0.1, 0.45, 0.45]],
            [[0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45]],
        ]
    )
    logits = probabilities.log().requires_grad_()
    ratio = expected_entity_ratio_loss(
        logits,
        torch.tensor([[1, 0, 0], [0, 0, 0]]),
        torch.tensor([[0, 1, 0], [0, 0, 0]]),
        torch.tensor([0, -1]),
        torch.tensor([0.25, 0.0]),
        o_label_id=0,
        lower_width=0.1,
    )
    ratio.loss.backward()

    assert ratio.loss.item() == 0.0
    assert ratio.predicted_ratio.item() == pytest.approx(0.2)
    assert ratio.lower_ratio.item() == pytest.approx(0.15)
    assert ratio.upper_ratio.item() == pytest.approx(0.25)
    assert ratio.tokens == 2
    assert ratio.rows == 1
    assert torch.count_nonzero(logits.grad) == 0


def test_expected_entity_ratio_loss_pushes_every_entity_row_in_the_needed_direction() -> None:
    positive_mask = torch.tensor([[1, 0]])
    unlabeled_mask = torch.tensor([[0, 1]])
    group_ids = torch.tensor([0])
    priors = torch.tensor([0.25])

    above = torch.tensor([[[0.1, 0.45, 0.45], [0.1, 0.7, 0.2]]]).log().requires_grad_()
    above_loss = expected_entity_ratio_loss(
        above,
        positive_mask,
        unlabeled_mask,
        group_ids,
        priors,
        o_label_id=0,
        lower_width=0.1,
    )
    above_loss.loss.backward()
    assert above_loss.predicted_ratio.item() == pytest.approx(0.9)
    assert torch.all(above.grad[..., 0] < 0)
    assert torch.all(above.grad[..., 1:] > 0)

    below = torch.tensor([[[0.95, 0.03, 0.02], [0.95, 0.01, 0.04]]]).log().requires_grad_()
    below_loss = expected_entity_ratio_loss(
        below,
        positive_mask,
        unlabeled_mask,
        group_ids,
        priors,
        o_label_id=0,
        lower_width=0.1,
    )
    below_loss.loss.backward()
    assert below_loss.predicted_ratio.item() == pytest.approx(0.05)
    assert torch.all(below.grad[..., 0] > 0)
    assert torch.all(below.grad[..., 1:] < 0)


def test_expected_entity_ratio_loss_uses_logical_step_partial_token_mass() -> None:
    class RatioTrainer(pii_encoder_train.DualHeadLossMixin):
        partial_expected_entity_ratio_loss_weight = 10.0
        partial_expected_entity_ratio_lower_width = 0.1

    trainer = RatioTrainer()
    trainer.correctness_map = SimpleNamespace(new_outside_id=0)
    step_masses = pii_encoder_train.LogicalStepObjectiveMasses(
        totals={"partial_expected_entity_ratio": torch.tensor(6.0)},
        physical_batches=2,
        primary_group_weight_sum=torch.tensor(11.0),
        data_weight_sum=torch.tensor(1.0),
    )
    batches = (
        (
            torch.tensor([[[0.1, 0.9], [0.1, 0.9]]]).log().requires_grad_(),
            torch.tensor([[1, 0]]),
            torch.tensor([[0, 1]]),
            torch.tensor([0]),
            torch.tensor([0.25]),
        ),
        (
            torch.tensor(
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.5, 0.5], [0.5, 0.5]],
                ]
            )
            .log()
            .requires_grad_(),
            torch.tensor([[1, 0], [1, 0]]),
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([0, 1]),
            torch.tensor([0.25, 0.25]),
        ),
    )
    actual = torch.tensor(0.0)
    expected = torch.tensor(0.0)
    for logits, positive, unlabeled, groups, priors in batches:
        ratio = expected_entity_ratio_loss(
            logits,
            positive,
            unlabeled,
            groups,
            priors,
            o_label_id=0,
            lower_width=0.1,
        )
        telemetry = {}
        actual = actual + trainer.apply_presence_objectives(
            logits.sum() * 0.0,
            logits,
            None,
            None,
            None,
            unlabeled,
            positive,
            None,
            None,
            groups,
            priors,
            None,
            telemetry,
            step_masses,
        )
        expected = expected + 10.0 * ratio.loss * (ratio.tokens / 6.0) / 11.0
        assert telemetry["dual_partial_expected_entity_ratio_tokens"] == ratio.tokens
    torch.testing.assert_close(actual, expected)


def test_parent_presence_kl_only_updates_unlabeled_student_tokens() -> None:
    student = torch.tensor(
        [[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    parent = torch.tensor(
        [[[0.0, 2.0, 2.0], [0.0, 2.0, 2.0], [0.0, 2.0, 2.0]]],
        requires_grad=True,
    )

    loss = parent_presence_kl_loss(student, parent, torch.tensor([[0, 1, 0]]), o_label_id=0)
    loss.backward()

    assert loss.item() > 0.0
    assert torch.count_nonzero(student.grad[0, 1]) > 0
    assert torch.count_nonzero(student.grad[0, [0, 2]]) == 0
    assert parent.grad is None


def test_family_presence_loss_pools_family_rows_and_masks_unknowns() -> None:
    groups = [[0], [1, 2], [3]]
    labels = torch.tensor([[1, 2, -100]])
    good_logits = torch.tensor(
        [[[-8.0, 4.0, 4.0, -8.0], [-8.0, -8.0, -8.0, 8.0], [8.0, -8.0, -8.0, -8.0]]],
        requires_grad=True,
    )
    swapped_logits = torch.tensor(
        [[[-8.0, -8.0, -8.0, 8.0], [-8.0, 4.0, 4.0, -8.0], [8.0, -8.0, -8.0, -8.0]]]
    )

    good_loss = pii_encoder_train.family_presence_loss(good_logits, labels, groups)
    swapped_loss = pii_encoder_train.family_presence_loss(swapped_logits, labels, groups)
    good_loss.backward()

    assert good_loss < swapped_loss
    assert torch.count_nonzero(good_logits.grad[0, :2]) > 0
    assert torch.count_nonzero(good_logits.grad[0, 2]) == 0
    with pytest.raises(ValueError, match="nonempty label groups"):
        pii_encoder_train.family_presence_loss(good_logits.detach(), labels, [[0], []])


def test_build_family_presence_projection_masks_multi_family_old_labels() -> None:
    from scripts.pii_dual_head import CorrectnessMap
    from scripts.pii_ontology_v2 import load_ontology

    ontology = load_ontology()
    new_labels = tuple(ontology.bioes_labels())
    new_index = {label: index for index, label in enumerate(new_labels)}
    old_labels = ("O", "S-old_name", "S-old_broad")
    old_to_new = torch.zeros((len(old_labels), len(new_labels)), dtype=torch.bool)
    old_to_new[0, new_index["O"]] = True
    old_to_new[1, new_index["S-person_name"]] = True
    old_to_new[2, new_index["S-person_name"]] = True
    old_to_new[2, new_index["S-organization"]] = True

    groups, own_targets, secondary_targets = pii_encoder_train.build_family_presence_projection(
        CorrectnessMap(
            old_labels=old_labels,
            new_labels=new_labels,
            old_to_new=old_to_new,
            new_to_old=old_to_new.T.clone(),
            map_sha256="test",
            ontology_sha256="test",
            map_path="test",
            empirical_targets=0,
        )
    )

    assert groups[0] == [new_index["O"]]
    assert len(groups) == 1 + len(ontology.families)
    assert sorted(index for group in groups for index in group) == list(range(len(new_labels)))
    person_family = secondary_targets[new_index["S-person_name"]]
    assert own_targets[1] == person_family
    assert 2 not in own_targets
    assert 0 not in own_targets
    assert secondary_targets[new_index["B-person_name"]] == person_family
    assert new_index["O"] not in secondary_targets


def test_o_weighted_token_loss_downweights_o_and_ignores_masked_tokens() -> None:
    labels = torch.tensor([[0, 1, -100]])
    logits = torch.tensor([[[-2.0, 2.0], [-2.0, 2.0], [8.0, -8.0]]], requires_grad=True)

    stock = token_classification_loss(logits, labels)
    weighted = o_weighted_token_classification_loss(logits, labels, o_label_id=0, o_token_weight=0.25)
    weighted.backward()

    assert weighted < stock
    assert torch.count_nonzero(logits.grad[0, :2]) > 0
    assert torch.count_nonzero(logits.grad[0, 2]) == 0


def test_masked_predicate_bce_updates_only_known_token_channel_cells() -> None:
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    labels = torch.tensor([[[-100.0, -100.0], [1.0, 0.0], [-100.0, 1.0]]])

    loss = pii_encoder_train.masked_predicate_bce_loss(logits, labels)
    loss.backward()

    assert loss.item() == pytest.approx(math.log(2.0))
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1]) == 2
    assert torch.count_nonzero(logits.grad[0, 2]) == 1


def test_masked_predicate_bce_uses_cell_weights_without_unknown_dilution() -> None:
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    labels = torch.tensor([[[-100.0, -100.0], [1.0, 1.0]]])
    weights = torch.tensor([[[0.0, 0.0], [3.0, 1.0]]])

    loss = pii_encoder_train.masked_predicate_bce_loss(logits, labels, weights)
    expected = (
        3.0 * torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 1, 0], labels[0, 1, 0])
        + torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 1, 1], labels[0, 1, 1])
    ) / 4.0
    loss.backward()

    assert loss.item() == pytest.approx(expected.item())
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert abs(logits.grad[0, 1, 0]) > abs(logits.grad[0, 1, 1])

    bad_weights = weights.clone()
    bad_weights[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="unknown predicate targets"):
        pii_encoder_train.masked_predicate_bce_loss(logits.detach(), labels, bad_weights)

    invalid_zero_weight_target = torch.tensor([[[2.0]]])
    with pytest.raises(ValueError, match="known predicate targets must be binary"):
        pii_encoder_train.masked_predicate_bce_loss(
            torch.zeros_like(invalid_zero_weight_target),
            invalid_zero_weight_target,
            torch.zeros_like(invalid_zero_weight_target),
        )


def test_masked_predicate_bce_applies_conditioned_positive_learning_weights() -> None:
    logits = torch.tensor([[[-1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    weights = torch.ones_like(labels)
    condition_ids = torch.tensor([[0, 1]])
    positive_by_condition = torch.tensor([[3.0, 1.0], [1.0, 5.0]])
    positive_weights = pii_encoder_train.conditioned_predicate_positive_weights(
        labels,
        condition_ids,
        positive_by_condition,
    )

    loss = pii_encoder_train.masked_predicate_bce_loss(
        logits,
        labels,
        weights,
        positive_weights,
    )
    per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    expected = (
        3.0 * per_cell[0, 0, 0] + per_cell[0, 0, 1] + per_cell[0, 1, 0] + 5.0 * per_cell[0, 1, 1]
    ) / 10.0

    assert loss.item() == pytest.approx(expected.item())


def test_added_head_balance_spec_binds_reference_and_conditioned_predicate_inventory(
    tmp_path,
) -> None:
    predicate_spec = pii_encoder_train.load_predicate_spec(pii_encoder_train.DEFAULT_PREDICATE_SPEC_PATH)
    conditions = ("person_name", "person_reference")
    path = tmp_path / "balance.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pii-ont3-added-head-balance",
                "version": 1,
                "status": "frozen",
                "source_diagnostic": {},
                "derivation": {},
                "reference_positive_weights": {
                    "organization_reference": 2.0,
                    "person_reference": 3.0,
                },
                "predicate_positive_weights": {
                    condition: {channel: 4.0 for channel in predicate_spec.channels}
                    for condition in conditions
                },
            }
        ),
        encoding="utf-8",
    )

    result = pii_encoder_train.load_added_head_balance_spec(
        path,
        predicate_spec,
        conditions,
    )

    assert result.reference_positive_weights["person_reference"] == 3.0
    assert result.predicate_positive_weights["person_name"]["care_provider"] == 4.0


def test_conditioned_predicate_loss_updates_only_each_tokens_gold_type_block() -> None:
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    labels = torch.tensor([[[-100.0, -100.0], [1.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]])
    condition_ids = torch.tensor([[-1, 0, 1]])

    selected = pii_encoder_train.select_conditioned_predicate_logits(
        logits,
        condition_ids,
        weights,
    )
    loss = pii_encoder_train.masked_predicate_bce_loss(selected, labels, weights)
    loss.backward()

    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1, 0]) == 2
    assert torch.count_nonzero(logits.grad[0, 1, 1]) == 0
    assert torch.count_nonzero(logits.grad[0, 2, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 2, 1]) == 2


def test_subclass_loss_pools_whole_spans_and_averages_component_tokens() -> None:
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    loss = pii_encoder_train.masked_subclass_categorical_loss(
        logits,
        block_ids=torch.tensor([[0, 1]]),
        target_ids=torch.tensor([[2, 1]]),
        scope_ids=torch.tensor([[0, 1]]),
        token_masks=torch.tensor(
            [
                [
                    [False, True, True, False],
                    [False, False, True, True],
                ]
            ]
        ),
        objective_weights=torch.tensor([[2.0, 1.0]]),
        learning_weights=torch.tensor([[0.5, 2.0]]),
        blocks=[
            {"family": "whole", "primary_type": "place", "start": 0, "width": 3},
            {"family": "component", "primary_type": "name", "start": 3, "width": 2},
        ],
    )
    loss.backward()

    assert loss.item() == pytest.approx((math.log(3.0) + 2.0 * math.log(2.0)) / 3.0)
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1, :3]) == 3
    assert torch.count_nonzero(logits.grad[0, 1, 3:]) == 0
    assert torch.count_nonzero(logits.grad[0, 2]) == 5
    assert torch.count_nonzero(logits.grad[0, 3, :3]) == 0
    assert torch.count_nonzero(logits.grad[0, 3, 3:]) == 2


def test_span_dataset_projects_subclasses_with_inherited_and_component_weights() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Lewis Francis Stott"
            return {
                "input_ids": [0, 1, 2, 3, 4],
                "offset_mapping": [(0, 0), (0, 5), (6, 13), (14, 19), (0, 0)],
            }

    spec = pii_encoder_train.load_subclass_spec(pii_encoder_train.DEFAULT_SUBCLASS_SPEC_PATH)
    row = {
        "text": "Lewis Francis Stott",
        "spans": [[0, 19, "person_name"]],
        "primary_span_objective_weights": [0.8],
        "subclass_spans": [
            {
                "carrier_start": 0,
                "carrier_end": 19,
                "type": "person_name",
                "start": 0,
                "end": 5,
                "family": "name_component",
                "value": "given_name",
            },
            {
                "carrier_start": 0,
                "carrier_end": 19,
                "type": "person_name",
                "start": 6,
                "end": 13,
                "family": "name_component",
                "value": "middle_name",
                "learning_weight": 0.5,
            },
            {
                "carrier_start": 0,
                "carrier_end": 19,
                "type": "person_name",
                "start": 14,
                "end": 19,
                "family": "name_component",
                "value": "family_name",
                "objective_weight": 0.6,
            },
        ],
    }
    item = SpanDataset(
        [row],
        Tokenizer(),
        {
            "O": 0,
            "B-person_name": 1,
            "I-person_name": 2,
            "E-person_name": 3,
            "S-person_name": 4,
        },
        max_len=32,
        subclass_spec=spec,
    )[0]

    assert item["subclass_block_ids"] == [0, 0, 0]
    assert item["subclass_target_ids"] == [1, 2, 3]
    assert item["subclass_scope_ids"] == [1, 1, 1]
    assert item["subclass_token_masks"] == [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    assert item["subclass_objective_weights"] == pytest.approx([0.8, 0.8, 0.6])
    assert item["subclass_learning_weights"] == pytest.approx([1.0, 0.5, 1.0])


def test_partial_label_collator_pads_variable_subclass_components() -> None:
    collator = PartialLabelDataCollator.__new__(PartialLabelDataCollator)
    collator.tokenizer = SimpleNamespace(padding_side="right")
    collator.base = lambda features: {
        "input_ids": torch.tensor(
            [feature["input_ids"] + [0] * (3 - len(feature["input_ids"])) for feature in features]
        )
    }
    empty = {
        "subclass_block_ids": [],
        "subclass_target_ids": [],
        "subclass_scope_ids": [],
        "subclass_token_masks": [],
        "subclass_objective_weights": [],
        "subclass_learning_weights": [],
    }
    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "subclass_block_ids": [3],
                "subclass_target_ids": [2],
                "subclass_scope_ids": [0],
                "subclass_token_masks": [[0, 1]],
                "subclass_objective_weights": [0.8],
                "subclass_learning_weights": [0.5],
            },
            {"input_ids": [1, 2, 3], **empty},
        ]
    )

    assert batch["subclass_block_ids"].tolist() == [[3], [-1]]
    assert batch["subclass_target_ids"].tolist() == [[2], [-1]]
    assert batch["subclass_scope_ids"].tolist() == [[0], [-1]]
    assert batch["subclass_token_masks"].tolist() == [
        [[False, True, False]],
        [[False, False, False]],
    ]
    assert torch.allclose(
        batch["subclass_objective_weights"],
        torch.tensor([[0.8], [0.0]]),
    )
    assert torch.allclose(
        batch["subclass_learning_weights"],
        torch.tensor([[0.5], [0.0]]),
    )


def test_predicate_loss_weight_blends_with_primary_objective_only_when_labels_are_known() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = SimpleNamespace(predicate_logits=torch.zeros(1, 2, 1))
            loss = torch.tensor(2.0)
            return (loss, outputs) if return_outputs else loss

    class PredicateTrainer(pii_encoder_train.PredicateLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0

    trainer = PredicateTrainer()
    known = trainer.compute_loss(
        None,
        {
            "predicate_labels": torch.tensor([[[1.0], [-100.0]]]),
            "predicate_weights": torch.tensor([[[1.0], [0.0]]]),
        },
    )
    unknown = trainer.compute_loss(
        None,
        {
            "predicate_labels": torch.full((1, 2, 1), -100.0),
            "predicate_weights": torch.zeros(1, 2, 1),
        },
    )

    assert known.item() == pytest.approx((2.0 + math.log(2.0)) / 2.0)
    assert unknown.item() == pytest.approx(2.0)


def test_joint_predicate_subclass_loss_uses_one_symmetric_denominator() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = SimpleNamespace(
                predicate_logits=torch.zeros(1, 1, 1),
                subclass_logits=torch.zeros(1, 1, 2),
            )
            loss = torch.tensor(3.0)
            return (loss, outputs) if return_outputs else loss

    class JointTrainer(pii_encoder_train.PredicateSubclassLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        subclass_loss_weight = 2.0
        subclass_blocks = [{"start": 0, "width": 2}]

    loss = JointTrainer().compute_loss(
        SimpleNamespace(training=False),
        {
            "predicate_labels": torch.ones(1, 1, 1),
            "predicate_weights": torch.ones(1, 1, 1),
            "subclass_block_ids": torch.tensor([[0]]),
            "subclass_target_ids": torch.tensor([[1]]),
            "subclass_scope_ids": torch.tensor([[0]]),
            "subclass_token_masks": torch.tensor([[[True]]]),
            "subclass_objective_weights": torch.ones(1, 1),
            "subclass_learning_weights": torch.ones(1, 1),
        },
    )

    assert loss.item() == pytest.approx((3.0 + 3.0 * math.log(2.0)) / 4.0)


def test_logical_step_normalization_matches_concatenated_objective() -> None:
    from scripts.pii_dual_head import CorrectnessMap, head_loss_terms

    old_to_new = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
        ]
    )
    correctness_map = CorrectnessMap(
        old_labels=("O", "S-person_name"),
        new_labels=("O", "S-person_name", "S-person_reference"),
        old_to_new=old_to_new,
        new_to_old=old_to_new.T.clone(),
        map_sha256="test",
        ontology_sha256="test",
        map_path="test",
        empirical_targets=0,
        legacy_outside_unknown_primary_types=("person_reference",),
    )

    class Model:
        training = True

        def __call__(
            self,
            input_primary_logits,
            input_predicate_logits,
            input_subclass_logits,
        ):
            return SimpleNamespace(
                logits=input_primary_logits,
                predicate_logits=input_predicate_logits,
                subclass_logits=input_subclass_logits,
            )

    class JointTrainer(
        pii_encoder_train.PredicateSubclassLossMixin,
        pii_encoder_train.DualHeadLossMixin,
    ):
        logical_step_objective_normalization = True
        dual_head_retired = True
        complete_presence_loss_weight = 2.0
        predicate_loss_weight = 1.0
        subclass_loss_weight = 1.0
        subclass_blocks = [{"start": 0, "width": 2}]

    batches = [
        {
            "labels": torch.tensor([[-100, -100]]),
            "secondary_labels": torch.tensor([[2, 0]]),
            "primary_objective_weights": torch.ones(1, 2),
            "complete_presence_labels": torch.tensor([[1, 0]]),
            "partial_entity_positive_mask": torch.tensor([[1, 0]]),
            "consistency_mask": torch.tensor([[0, 1]]),
            "predicate_labels": torch.tensor([[[1.0], [-100.0]]]),
            "predicate_weights": torch.tensor([[[1.0], [0.0]]]),
            "subclass_block_ids": torch.tensor([[0]]),
            "subclass_target_ids": torch.tensor([[1]]),
            "subclass_scope_ids": torch.tensor([[0]]),
            "subclass_token_masks": torch.tensor([[[True, False]]]),
            "subclass_objective_weights": torch.tensor([[1.0]]),
            "subclass_learning_weights": torch.tensor([[1.0]]),
            "input_primary_logits": torch.tensor([[[0.1, 0.2, 1.1], [1.2, 0.0, -0.5]]]),
            "input_predicate_logits": torch.tensor([[[-0.7], [0.0]]]),
            "input_subclass_logits": torch.tensor([[[0.2, 0.8], [0.0, 0.0]]]),
        },
        {
            "labels": torch.tensor([[1, 0]]),
            "secondary_labels": torch.tensor([[-100, -100]]),
            "primary_objective_weights": torch.tensor([[0.5, 1.0]]),
            "complete_presence_labels": torch.tensor([[-100, -100]]),
            "partial_entity_positive_mask": torch.zeros(1, 2, dtype=torch.long),
            "consistency_mask": torch.zeros(1, 2, dtype=torch.long),
            "predicate_labels": torch.tensor([[[0.0], [1.0]]]),
            "predicate_weights": torch.tensor([[[3.0], [1.0]]]),
            "subclass_block_ids": torch.tensor([[0]]),
            "subclass_target_ids": torch.tensor([[0]]),
            "subclass_scope_ids": torch.tensor([[0]]),
            "subclass_token_masks": torch.tensor([[[False, True]]]),
            "subclass_objective_weights": torch.tensor([[3.0]]),
            "subclass_learning_weights": torch.tensor([[1.0]]),
            "input_primary_logits": torch.tensor([[[0.0, 0.9, -0.1], [0.8, 0.1, -0.2]]]),
            "input_predicate_logits": torch.tensor([[[1.0], [-0.5]]]),
            "input_subclass_logits": torch.tensor([[[0.0, 0.0], [0.6, -0.4]]]),
        },
    ]
    trainer = JointTrainer()
    trainer.correctness_map = correctness_map
    trainer.is_in_train = True
    masses = trainer._get_num_items_in_batch(batches, torch.device("cpu"))

    actual = sum(trainer.compute_loss(Model(), dict(batch), num_items_in_batch=masses) for batch in batches)

    primary_logits = torch.cat([batch["input_primary_logits"] for batch in batches])
    old_labels = torch.cat([batch["labels"] for batch in batches])
    new_labels = torch.cat([batch["secondary_labels"] for batch in batches])
    primary_weights = torch.cat([batch["primary_objective_weights"] for batch in batches])
    primary_terms = head_loss_terms(
        primary_logits,
        new_labels,
        old_labels,
        correctness_map.old_to_new,
        own_o_label_id=0,
        cross_o_label_id=0,
        token_objective_weights=primary_weights,
    )
    primary = primary_terms.total / primary_terms.normalizer
    presence = entity_presence_loss(
        batches[0]["input_primary_logits"],
        batches[0]["complete_presence_labels"],
        0,
    )
    predicate = pii_encoder_train.masked_predicate_bce_loss(
        torch.cat([batch["input_predicate_logits"] for batch in batches]),
        torch.cat([batch["predicate_labels"] for batch in batches]),
        torch.cat([batch["predicate_weights"] for batch in batches]),
    )
    subclass = pii_encoder_train.masked_subclass_categorical_loss(
        torch.cat([batch["input_subclass_logits"] for batch in batches]),
        torch.cat([batch["subclass_block_ids"] for batch in batches]),
        torch.cat([batch["subclass_target_ids"] for batch in batches]),
        torch.cat([batch["subclass_scope_ids"] for batch in batches]),
        torch.cat([batch["subclass_token_masks"] for batch in batches]),
        torch.cat([batch["subclass_objective_weights"] for batch in batches]),
        torch.cat([batch["subclass_learning_weights"] for batch in batches]),
        trainer.subclass_blocks,
    )
    primary_group = (primary + 2.0 * presence) / 3.0
    expected = (primary_group + predicate + subclass) / 3.0

    torch.testing.assert_close(actual, expected)
    assert masses.totals["primary"] == pytest.approx(3.5)
    assert masses.totals["complete_presence"] == pytest.approx(2.0)
    assert masses.totals["partial_expected_entity_ratio"] == pytest.approx(2.0)
    assert masses.totals["predicate"] == pytest.approx(5.0)
    assert masses.totals["subclass"] == pytest.approx(4.0)
    assert masses.primary_group_weight_sum == pytest.approx(3.0)
    assert masses.data_weight_sum == pytest.approx(3.0)


def test_logical_step_normalization_is_identical_for_equal_masses() -> None:
    masses = pii_encoder_train.LogicalStepObjectiveMasses(
        totals={
            "primary": torch.tensor(6.0),
            "complete_presence": torch.tensor(4.0),
            "predicate": torch.tensor(10.0),
            "subclass": torch.tensor(8.0),
        },
        physical_batches=2,
        primary_group_weight_sum=torch.tensor(3.0),
        data_weight_sum=torch.tensor(3.0),
    )
    primary = torch.tensor(2.0)
    presence = torch.tensor(0.5)
    predicate = torch.tensor(0.7)
    subclass = torch.tensor(1.2)

    corrected = torch.tensor(0.0)
    for _ in range(2):
        primary_group = (
            pii_encoder_train.logical_step_mean_contribution(
                primary,
                3.0,
                masses,
                "primary",
            )
            + 2.0
            * pii_encoder_train.logical_step_mean_contribution(
                presence,
                2.0,
                masses,
                "complete_presence",
            )
        ) / masses.primary_group_weight_sum
        corrected = (
            corrected
            + (
                primary_group
                + pii_encoder_train.logical_step_mean_contribution(
                    predicate,
                    5.0,
                    masses,
                    "predicate",
                )
                + pii_encoder_train.logical_step_mean_contribution(
                    subclass,
                    4.0,
                    masses,
                    "subclass",
                )
            )
            / masses.data_weight_sum
        )
    legacy = ((primary + 2.0 * presence) / 3.0 + predicate + subclass) / 3.0

    torch.testing.assert_close(corrected, legacy)


def test_logical_step_normalization_counts_parameter_priors_once_per_step(tmp_path) -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = torch.tensor(2.0)
            outputs = SimpleNamespace(logits=torch.zeros(1))
            return (loss, outputs) if return_outputs else loss

    class PriorTrainer(
        pii_encoder_train.EncoderParameterPriorTrainerMixin,
        pii_encoder_train.InheritedOutputParameterPriorTrainerMixin,
        BaseTrainer,
    ):
        pass

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Linear(1, 1, bias=False)
            self.classifier = nn.Linear(1, 2, bias=False)

    model = Model()
    with torch.no_grad():
        model.base_model.weight.fill_(2.0)
        model.classifier.weight[0].fill_(3.0)
    model.train()

    trainer = object.__new__(PriorTrainer)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(logging_steps=1, output_dir=str(tmp_path), pii_step_offset=0)
    trainer.encoder_prior_start_step = 0
    trainer.encoder_prior_weight = 0.3
    trainer.encoder_prior_anchors = {"weight": torch.tensor([[1.0]])}
    trainer.encoder_prior_logged_steps = {0}
    trainer.inherited_output_prior_start_step = 0
    trainer.inherited_output_prior_weight = 0.7
    trainer.inherited_output_prior_rows = 1
    trainer.inherited_output_prior_anchors = {"weight": torch.tensor([[1.0]])}
    trainer.inherited_output_prior_logged_steps = {0}
    masses = pii_encoder_train.LogicalStepObjectiveMasses(
        totals={},
        physical_batches=4,
        primary_group_weight_sum=torch.tensor(1.0),
        data_weight_sum=torch.tensor(1.0),
    )

    actual = trainer.compute_loss(model, {}, num_items_in_batch=masses)
    encoder_prior = 0.5 * (2.0 - 1.0) ** 2
    inherited_prior = 0.5 * (3.0 - 1.0) ** 2
    expected = 2.0 + (0.3 * encoder_prior + 0.7 * inherited_prior) / 4.0

    assert actual.item() == pytest.approx(expected)


def test_joint_reference_primary_positive_loss_uses_only_known_reference_tokens() -> None:
    from scripts.pii_dual_head import CorrectnessMap

    old_to_new = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    correctness_map = CorrectnessMap(
        old_labels=("O", "S-person_name"),
        new_labels=("O", "S-person_name", "S-person_reference"),
        old_to_new=old_to_new,
        new_to_old=old_to_new.T.clone(),
        map_sha256="test",
        ontology_sha256="test",
        map_path="test",
        empirical_targets=0,
        legacy_outside_unknown_primary_types=("person_reference",),
    )

    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del inputs, num_items_in_batch
            outputs = SimpleNamespace(
                logits=torch.zeros(1, 3, 3),
                predicate_logits=torch.zeros(1, 3, 1),
                subclass_logits=torch.zeros(1, 3, 2),
            )
            loss = torch.tensor(3.0)
            return (loss, outputs) if return_outputs else loss

    class JointTrainer(pii_encoder_train.PredicateSubclassLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        subclass_loss_weight = 1.0
        reference_primary_positive_loss_weight = 2.0
        subclass_blocks = [{"start": 0, "width": 2}]

    trainer = JointTrainer()
    trainer.correctness_map = correctness_map
    trainer.dual_head_telemetry = {}
    loss = trainer.compute_loss(
        SimpleNamespace(training=True),
        {
            "labels": torch.tensor([[-100, -100, 1]]),
            "secondary_labels": torch.tensor([[2, -100, -100]]),
            "primary_objective_weights": torch.tensor([[0.5, 0.0, 1.0]]),
            "predicate_labels": torch.tensor([[[1.0], [-100.0], [-100.0]]]),
            "predicate_weights": torch.tensor([[[1.0], [0.0], [0.0]]]),
            "subclass_block_ids": torch.tensor([[0]]),
            "subclass_target_ids": torch.tensor([[1]]),
            "subclass_scope_ids": torch.tensor([[0]]),
            "subclass_token_masks": torch.tensor([[[True, False, False]]]),
            "subclass_objective_weights": torch.ones(1, 1),
            "subclass_learning_weights": torch.ones(1, 1),
        },
    )

    assert loss.item() == pytest.approx((3.0 + math.log(2.0) + math.log(2.0) + 2.0 * math.log(3.0)) / 5.0)
    assert trainer.dual_head_telemetry["dual_reference_primary_positive_tokens"] == 1.0
    assert trainer.dual_head_telemetry["dual_reference_primary_positive_normalizer"] == 0.5


def test_joint_objective_validation_telemetry_separates_primary_and_added_heads() -> None:
    from scripts.pii_dual_head import CorrectnessMap

    old_to_new = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    correctness_map = CorrectnessMap(
        old_labels=("O", "S-person_name"),
        new_labels=("O", "S-person_name", "S-person_reference"),
        old_to_new=old_to_new,
        new_to_old=old_to_new.T.clone(),
        map_sha256="test",
        ontology_sha256="test",
        map_path="test",
        empirical_targets=0,
        legacy_outside_unknown_primary_types=("person_reference",),
    )

    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del inputs, num_items_in_batch
            outputs = SimpleNamespace(
                logits=torch.zeros(1, 3, 3),
                predicate_logits=torch.zeros(1, 3, 1),
                subclass_logits=torch.zeros(1, 3, 2),
            )
            loss = torch.tensor(9.0)
            return (loss, outputs) if return_outputs else loss

    class JointTrainer(pii_encoder_train.PredicateSubclassLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        subclass_loss_weight = 1.0
        subclass_blocks = [{"start": 0, "width": 2}]

    trainer = JointTrainer()
    trainer.correctness_map = correctness_map
    trainer._ont3_eval_objective_totals = {}
    trainer.compute_loss(
        SimpleNamespace(training=False),
        {
            "labels": torch.tensor([[-100, 1, -100]]),
            "secondary_labels": torch.tensor([[2, -100, 0]]),
            "primary_objective_weights": torch.ones(1, 3),
            "complete_presence_labels": torch.tensor([[1, 1, 0]]),
            "predicate_labels": torch.tensor([[[1.0], [-100.0], [-100.0]]]),
            "predicate_weights": torch.tensor([[[0.5], [0.0], [0.0]]]),
            "subclass_block_ids": torch.tensor([[0]]),
            "subclass_target_ids": torch.tensor([[1]]),
            "subclass_scope_ids": torch.tensor([[0]]),
            "subclass_token_masks": torch.tensor([[[True, False, False]]]),
            "subclass_objective_weights": torch.tensor([[0.25]]),
            "subclass_learning_weights": torch.tensor([[2.0]]),
        },
    )

    totals = trainer._ont3_eval_objective_totals
    assert set(totals) == {
        "complete_presence",
        "ont2_compatible_primary",
        "predicate",
        "primary",
        "reference_primary_positive",
        "subclass",
    }
    assert totals["primary"][1] == pytest.approx(3.0)
    assert totals["ont2_compatible_primary"][1] == pytest.approx(2.0)
    assert totals["reference_primary_positive"][1] == pytest.approx(1.0)
    assert totals["predicate"][1] == pytest.approx(0.5)
    assert totals["subclass"][1] == pytest.approx(0.5)
    assert totals["complete_presence"][1] == pytest.approx(3.0)


def test_joint_objective_validation_loop_publishes_support_weighted_metrics() -> None:
    class BaseTrainer:
        args = SimpleNamespace(device=torch.device("cpu"))
        accelerator = SimpleNamespace(reduce=lambda value, reduction: value)

        def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):
            del dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            self._record_eval_objective_loss("predicate", torch.tensor(2.0), 3.0)
            self._record_eval_objective_loss("predicate", torch.tensor(5.0), 1.0)
            return SimpleNamespace(metrics={})

    class JointTrainer(pii_encoder_train.PredicateSubclassLossMixin, BaseTrainer):
        pass

    output = JointTrainer().evaluation_loop([], "test", metric_key_prefix="heldout")

    assert output.metrics == {
        "heldout_ont3_predicate_loss": pytest.approx(2.75),
        "heldout_ont3_predicate_normalizer": pytest.approx(4.0),
        "heldout_ont3_predicate_active_batches": pytest.approx(2.0),
    }


def test_joint_subclass_loss_records_realized_categorical_targets() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = SimpleNamespace(
                predicate_logits=torch.zeros(1, 1, 1),
                subclass_logits=torch.zeros(1, 1, 2),
            )
            loss = torch.tensor(3.0)
            return (loss, outputs) if return_outputs else loss

    class JointTrainer(pii_encoder_train.PredicateSubclassLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        subclass_loss_weight = 2.0
        subclass_blocks = [{"start": 0, "width": 2}]
        subclass_exposure_blocks = (
            {
                "family": "kind",
                "primary_type": "carrier",
                "start": 0,
                "width": 2,
                "outcomes": ("Q", "specific"),
            },
        )

    trainer = JointTrainer()
    trainer.compute_loss(
        SimpleNamespace(training=True),
        {
            "predicate_labels": torch.ones(1, 1, 1),
            "predicate_weights": torch.ones(1, 1, 1),
            "subclass_block_ids": torch.tensor([[0, 0, -1]]),
            "subclass_target_ids": torch.tensor([[1, 0, -1]]),
            "subclass_scope_ids": torch.tensor([[0, 0, -1]]),
            "subclass_token_masks": torch.tensor([[[True], [True], [False]]]),
            "subclass_objective_weights": torch.tensor([[0.5, 2.0, 0.0]]),
            "subclass_learning_weights": torch.tensor([[2.0, 0.25, 0.0]]),
        },
    )

    receipt = trainer.subclass_training_exposure_receipt()
    assert receipt["physical_batches"] == 1
    assert receipt["windows"] == 1
    assert receipt["blocks"]["kind@carrier"]["targets"] == {
        "Q": {
            "components": 1,
            "objective_weight": pytest.approx(2.0),
            "effective_weight": pytest.approx(0.5),
        },
        "specific": {
            "components": 1,
            "objective_weight": pytest.approx(0.5),
            "effective_weight": pytest.approx(1.0),
        },
    }
    assert receipt["by_family_target"]["kind=specific"]["components"] == 1


def test_joint_reference_residual_uses_same_objective_denominator_and_records_exposure() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = SimpleNamespace(
                predicate_logits=torch.zeros(1, 1, 1),
                subclass_logits=torch.zeros(1, 1, 2),
                reference_type_logits=torch.zeros(1, 1, 2),
            )
            loss = torch.tensor(3.0)
            return (loss, outputs) if return_outputs else loss

    class JointTrainer(pii_encoder_train.PredicateSubclassLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        subclass_loss_weight = 2.0
        subclass_blocks = [{"start": 0, "width": 2}]
        reference_type_residual_loss_weight = 3.0
        reference_type_residual_types = (
            "organization_reference",
            "person_reference",
        )
        reference_type_residual_positive_weights = torch.tensor([5.0, 4.0])

    trainer = JointTrainer()
    loss = trainer.compute_loss(
        SimpleNamespace(training=True),
        {
            "predicate_labels": torch.ones(1, 1, 1),
            "predicate_weights": torch.ones(1, 1, 1),
            "reference_type_labels": torch.tensor([[[0.0, 1.0]]]),
            "reference_type_weights": torch.tensor([[[0.25, 0.75]]]),
            "subclass_block_ids": torch.tensor([[0]]),
            "subclass_target_ids": torch.tensor([[1]]),
            "subclass_scope_ids": torch.tensor([[0]]),
            "subclass_token_masks": torch.tensor([[[True]]]),
            "subclass_objective_weights": torch.ones(1, 1),
            "subclass_learning_weights": torch.ones(1, 1),
        },
    )

    assert loss.item() == pytest.approx((3.0 + 6.0 * math.log(2.0)) / 7.0)
    receipt = trainer.reference_type_residual_training_exposure_receipt()
    assert receipt["physical_batches"] == 1
    assert receipt["types"] == {
        "organization_reference": {
            "positive_tokens": 0,
            "known_negative_tokens": 1,
            "positive_objective_weight": 0.0,
            "known_negative_objective_weight": pytest.approx(0.25),
            "positive_learning_multiplier": 5.0,
        },
        "person_reference": {
            "positive_tokens": 1,
            "known_negative_tokens": 0,
            "positive_objective_weight": pytest.approx(0.75),
            "known_negative_objective_weight": 0.0,
            "positive_learning_multiplier": 4.0,
        },
    }


def test_predicate_loss_records_realized_token_cell_and_reference_exposure() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = SimpleNamespace(predicate_logits=torch.zeros(1, 3, 2))
            loss = torch.tensor(2.0)
            return (loss, outputs) if return_outputs else loss

    class PredicateTrainer(pii_encoder_train.PredicateLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        predicate_exposure_channels = ("alpha", "beta")
        primary_exposure_labels = ("O", "B-person_reference")
        predicate_exposure_reference_label_ids = {
            "organization_reference": (5, 6, 7, 8),
            "person_reference": (1, 2, 3, 4),
        }

    trainer = PredicateTrainer()
    trainer.compute_loss(
        SimpleNamespace(training=True),
        {
            "predicate_labels": torch.tensor([[[1.0, 0.0], [1.0, -100.0], [-100.0, 1.0]]]),
            "predicate_weights": torch.tensor([[[2.0, 0.5], [1.0, 0.0], [0.0, 3.0]]]),
            "primary_objective_weights": torch.tensor([[0.85, 1.0, 0.0]]),
            "secondary_labels": torch.tensor([[1, 0, -100]]),
        },
    )

    receipt = trainer.predicate_training_exposure_receipt()
    assert receipt["physical_batches"] == 1
    assert receipt["windows"] == 1
    assert receipt["predicate"]["positive_token_cells"] == {"alpha": 2, "beta": 1}
    assert receipt["predicate"]["known_negative_token_cells"] == {"alpha": 0, "beta": 1}
    assert receipt["predicate"]["positive_objective_weight"] == {
        "alpha": pytest.approx(3.0),
        "beta": pytest.approx(3.0),
    }
    assert receipt["primary_output"] == {
        "O": {"positive_tokens": 1, "positive_objective_weight": pytest.approx(1.0)},
        "B-person_reference": {
            "positive_tokens": 1,
            "positive_objective_weight": pytest.approx(0.85),
        },
    }
    assert receipt["reference"] == {
        "organization_reference": {
            "positive_tokens": 0,
            "known_negative_tokens": 2,
            "positive_objective_weight": 0.0,
            "known_negative_objective_weight": pytest.approx(1.85),
        },
        "person_reference": {
            "positive_tokens": 1,
            "known_negative_tokens": 1,
            "positive_objective_weight": pytest.approx(0.85),
            "known_negative_objective_weight": 1.0,
        },
    }


def test_predicate_exposure_separates_annotated_support_from_zero_objective_mass() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = SimpleNamespace(predicate_logits=torch.zeros(1, 2, 1, 1))
            loss = torch.tensor(2.0)
            return (loss, outputs) if return_outputs else loss

    class PredicateTrainer(pii_encoder_train.PredicateLossMixin, BaseTrainer):
        predicate_loss_weight = 1.0
        predicate_exposure_channels = ("hard_role",)
        predicate_exposure_condition_types = ("person_name",)
        primary_exposure_labels = ("O", "S-person_reference")
        predicate_exposure_reference_label_ids = {"person_reference": (1,)}

    trainer = PredicateTrainer()
    trainer.compute_loss(
        SimpleNamespace(training=True),
        {
            "predicate_labels": torch.tensor([[[1.0], [0.0]]]),
            "predicate_weights": torch.zeros(1, 2, 1),
            "predicate_condition_ids": torch.tensor([[0, 0]]),
            "primary_objective_weights": torch.zeros(1, 2),
            "secondary_labels": torch.tensor([[1, 0]]),
        },
    )

    receipt = trainer.predicate_training_exposure_receipt()
    assert receipt["predicate"]["positive_token_cells"] == {"hard_role": 1}
    assert receipt["predicate"]["known_negative_token_cells"] == {"hard_role": 1}
    assert receipt["predicate"]["positive_objective_weight"] == {"hard_role": 0.0}
    assert receipt["predicate"]["known_negative_objective_weight"] == {"hard_role": 0.0}
    assert receipt["predicate_by_primary_type"] == {
        "person_name": {
            "hard_role": {
                "positive_token_cells": 1,
                "known_negative_token_cells": 1,
                "positive_objective_weight": 0.0,
                "known_negative_objective_weight": 0.0,
            }
        }
    }
    assert receipt["primary_output"] == {
        "O": {"positive_tokens": 1, "positive_objective_weight": 0.0},
        "S-person_reference": {"positive_tokens": 1, "positive_objective_weight": 0.0},
    }
    assert receipt["reference"] == {
        "person_reference": {
            "positive_tokens": 1,
            "known_negative_tokens": 1,
            "positive_objective_weight": 0.0,
            "known_negative_objective_weight": 0.0,
        }
    }


def test_config_only_o_weighting_installs_and_applies_per_row_loss() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            outputs = SimpleNamespace(logits=inputs["logits"])
            loss = token_classification_loss(outputs.logits, inputs["labels"])
            return (loss, outputs) if return_outputs else loss

    trainer_class = o_token_weighted_trainer_class(
        BaseTrainer,
        o_weight_config="weights.yaml",
        o_token_loss_weight=1.0,
    )
    assert issubclass(trainer_class, OTokenLossWeightMixin)
    assert o_token_weighted_trainer_class(BaseTrainer) is BaseTrainer

    logits = torch.tensor(
        [
            [[-2.0, 2.0], [-2.0, 2.0]],
            [[-2.0, 2.0], [-2.0, 2.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 1], [0, 1]])
    row_weights = torch.tensor([0.1, 1.0])
    trainer = trainer_class()
    loss = trainer.compute_loss(
        SimpleNamespace(training=True),
        {"logits": logits, "labels": labels, "o_weight": row_weights},
    )

    expected = per_row_o_weighted_token_classification_loss(logits, labels, 0, row_weights)
    torch.testing.assert_close(loss, expected)


def test_native_primary_objective_preserves_historical_resume_and_pins_new_settings() -> None:
    resolve = pii_encoder_train.resolve_native_primary_objective
    requested = {"boundary": 2.0, "type": 1.0}
    current = resolve(requested)
    assert current == {"version": "weighted-softmax-margin-v1", **requested}
    assert resolve(requested, resume_config=SimpleNamespace()) == {"version": "legacy"}
    restored = SimpleNamespace(pii_native_primary_objective=json.loads(json.dumps(current)))
    assert resolve(requested, resume_config=restored) == current
    with pytest.raises(ValueError, match="cannot change during exact resume"):
        resolve({"boundary": 0.0, "type": 0.0}, resume_config=restored)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        resolve(requested, resume_config=SimpleNamespace(pii_native_primary_objective={"version": "bad"}))


@pytest.mark.parametrize("row_o_weights", [None, torch.tensor([0.25, 0.75])])
def test_native_margin_trainer_changes_weighted_loss_and_gradient(tmp_path, row_o_weights) -> None:
    from transformers import BertConfig, BertForTokenClassification, Trainer, TrainingArguments

    torch.manual_seed(173)
    model = BertForTokenClassification(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=5,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    trainer_class = o_token_weighted_trainer_class(Trainer, native_objective=True)
    trainer = trainer_class(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    trainer.native_objective = True
    trainer.o_token_loss_weight = 0.5
    trainer.bioes_structure_cost_matrix = pii_encoder_train.bioes_structure_cost_matrix(
        ["O", "B-person_name", "I-person_name", "E-person_name", "S-person_name"],
        boundary_cost=2.0,
        type_cost=1.0,
    )
    labels = torch.tensor([[0, 1, 3, -100], [4, 0, -100, -100]])
    weights = torch.tensor([[1.0, 0.5, 1.0, 0.0], [0.75, 1.0, 0.0, 0.0]])
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "labels": labels,
        "primary_objective_weights": weights,
    }
    if row_o_weights is not None:
        inputs["o_weight"] = row_o_weights
    loss, outputs = trainer.compute_loss(model, dict(inputs), return_outputs=True)
    valid = labels != -100
    selected = outputs.logits.float()[valid]
    targets = labels[valid]
    o_weights = (torch.full((2,), 0.5) if row_o_weights is None else row_o_weights)[:, None]
    applied = (weights * torch.where(labels == 0, o_weights, 1.0))[valid]
    expected = (
        torch.nn.functional.cross_entropy(
            selected + trainer.bioes_structure_cost_matrix[targets], targets, reduction="none"
        )
        * applied
    ).sum() / applied.sum()
    torch.testing.assert_close(loss, expected)
    margin_gradient = torch.autograd.grad(loss, model.classifier.weight)[0]
    trainer.bioes_structure_cost_matrix = None
    plain = trainer.compute_loss(model, dict(inputs))
    plain_gradient = torch.autograd.grad(plain, model.classifier.weight)[0]
    assert loss > plain
    assert not torch.allclose(margin_gradient, plain_gradient)
    model.eval()
    trainer.bioes_structure_cost_matrix = torch.ones(5, 5) * 10
    evaluation = trainer.compute_loss(model, dict(inputs))
    torch.testing.assert_close(evaluation, model(input_ids=inputs["input_ids"], labels=labels).loss)


@pytest.mark.parametrize("historical", [False, True])
def test_native_objective_checkpoint_resume_matches_uninterrupted_updates(tmp_path, historical) -> None:
    from transformers import BertConfig, BertForTokenClassification, Trainer, TrainingArguments

    torch.manual_seed(173)
    config = BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=5,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    requested = {"boundary": 2.0, "type": 1.0}
    if not historical:
        config.pii_native_primary_objective = pii_encoder_train.resolve_native_primary_objective(requested)
    model = BertForTokenClassification(config)
    rows = [{"input_ids": [1, 2, 3, 4], "labels": [0, 1, 3, -100]}] * 8

    def trainer_for(model, output, *, resume):
        objective = pii_encoder_train.resolve_native_primary_objective(
            requested, resume_config=model.config if resume or historical else None
        )
        enabled = objective["version"] != "legacy"
        trainer_class = o_token_weighted_trainer_class(Trainer, native_objective=enabled)
        trainer = trainer_class(
            model=model,
            train_dataset=rows,
            args=TrainingArguments(
                output_dir=str(output),
                use_cpu=True,
                report_to=[],
                max_steps=4,
                save_steps=2,
                per_device_train_batch_size=2,
                seed=173,
                learning_rate=0.001,
                disable_tqdm=True,
            ),
        )
        if enabled:
            trainer.bioes_structure_cost_matrix = pii_encoder_train.bioes_structure_cost_matrix(
                ["O", "B-person_name", "I-person_name", "E-person_name", "S-person_name"],
                boundary_cost=2.0,
                type_cost=1.0,
            )
        return trainer

    full = trainer_for(model, tmp_path / "full", resume=False)
    full.train()
    checkpoint = tmp_path / "full" / "checkpoint-2"
    restored = BertForTokenClassification.from_pretrained(checkpoint)
    resumed = trainer_for(restored, tmp_path / "resumed", resume=True)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    for name, tensor in full.model.state_dict().items():
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert resumed.state.global_step == full.state.global_step == 4


def test_symmetric_token_kl_loss_is_symmetric_and_ignores_masked_tokens() -> None:
    labels = torch.tensor([[0, 1, -100]])
    first = torch.tensor([[[2.0, -1.0], [0.0, 1.0], [8.0, -8.0]]], requires_grad=True)
    second = torch.tensor([[[-1.0, 2.0], [1.0, 0.0], [-8.0, 8.0]]], requires_grad=True)

    forward = symmetric_token_kl_loss(first, second, labels)
    reverse = symmetric_token_kl_loss(second, first, labels)
    forward.backward()

    assert torch.allclose(forward, reverse)
    assert forward > 0
    assert torch.count_nonzero(first.grad[0, :2]) > 0
    assert torch.count_nonzero(first.grad[0, 2]) == 0
    assert torch.count_nonzero(second.grad[0, :2]) > 0
    assert torch.count_nonzero(second.grad[0, 2]) == 0


def test_symmetric_token_kl_loss_is_zero_for_identical_posteriors() -> None:
    labels = torch.tensor([[0, 1]])
    logits = torch.tensor([[[2.0, -1.0], [0.0, 1.0]]])

    assert torch.isclose(symmetric_token_kl_loss(logits, logits, labels), torch.tensor(0.0), atol=1e-7)


def test_expand_output_labels_preserves_shared_rows_and_initializes_new_rows() -> None:
    model = nn.Module()
    model.classifier = nn.Linear(3, 3)
    model.num_labels = 3
    model.config = SimpleNamespace(
        label2id={"O": 0, "B-name": 1, "E-name": 2},
        initializer_range=0.02,
        pii_head_architecture=None,
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.arange(9).reshape(3, 3))
        model.classifier.bias.copy_(torch.arange(3))
    original_weight = model.classifier.weight.detach().clone()
    original_bias = model.classifier.bias.detach().clone()

    copied = expand_output_labels(model, ["O", "B-name", "I-name", "E-name"])

    assert copied == 3
    assert model.classifier.out_features == 4
    assert torch.equal(model.classifier.weight[0], original_weight[0])
    assert torch.equal(model.classifier.weight[1], original_weight[1])
    assert torch.equal(model.classifier.weight[3], original_weight[2])
    assert torch.equal(model.classifier.bias[[0, 1, 3]], original_bias)
    assert model.config.label2id == {"O": 0, "B-name": 1, "I-name": 2, "E-name": 3}


def test_expand_output_labels_can_reinitialize_every_output_row() -> None:
    model = nn.Module()
    model.classifier = nn.Linear(3, 3)
    model.num_labels = 3
    model.config = SimpleNamespace(
        label2id={"O": 0, "B-name": 1, "E-name": 2},
        initializer_range=0.02,
        pii_head_architecture=None,
    )
    with torch.no_grad():
        model.classifier.weight.fill_(7.0)
        model.classifier.bias.fill_(9.0)

    copied = expand_output_labels(
        model,
        ["O", "B-name", "I-name", "E-name"],
        copy_shared_rows=False,
    )

    assert copied == 0
    assert model.classifier.out_features == 4
    assert not torch.any(model.classifier.weight == 7.0)
    assert torch.count_nonzero(model.classifier.bias) == 0
    assert model.config.pii_output_head_reinitialized is True
    assert model.config.pii_warm_start_copied_label_rows == 0
    assert model.config.pii_warm_start_source_rows_per_label == {}


def test_expand_output_labels_supports_score_head_and_semantic_schema() -> None:
    model = nn.Module()
    model.score = nn.Linear(2, 5)
    model.num_labels = 5
    model.config = SimpleNamespace(
        label2id={"O": 0, "B-first_name": 1, "I-first_name": 2, "E-first_name": 3, "S-first_name": 4},
        initializer_range=0.02,
        pii_head_architecture=None,
    )
    with torch.no_grad():
        model.score.weight.copy_(torch.arange(10).reshape(5, 2))
        model.score.bias.copy_(torch.arange(5))
    original_weight = model.score.weight.detach().clone()
    original_bias = model.score.bias.detach().clone()

    copied = expand_output_labels(
        model,
        ["O", "B-given_name", "I-given_name", "E-given_name", "S-given_name", "B-email"],
        source_schema="openmed_nemotron_55",
    )

    assert copied == 5
    assert model.score.out_features == 6
    assert torch.equal(model.score.weight[:5], original_weight)
    assert torch.equal(model.score.bias[:5], original_bias)
    assert model.config.pii_warm_start_label_schema == "openmed_nemotron_55"
    assert model.config.pii_warm_start_copied_label_rows == 5


def test_projected_warm_start_labels_rejects_semantic_collisions() -> None:
    with pytest.raises(ValueError, match="maps multiple rows"):
        projected_warm_start_labels(
            {"B-CURRENCY": 0, "B-CURRENCYCODE": 1},
            {"B-currency_designator": 0},
            source_schema="openmed_54",
        )


def test_expand_output_labels_averages_rows_for_reporting_cut() -> None:
    model = nn.Module()
    model.classifier = nn.Linear(1, 4)
    model.num_labels = 4
    model.config = SimpleNamespace(
        label2id={"O": 0, "B-person_name": 1, "B-given_name": 2, "B-family_name": 3},
        initializer_range=0.02,
        pii_head_architecture=None,
    )
    with torch.no_grad():
        model.classifier.weight[:, 0].copy_(torch.tensor([0.0, 2.0, 4.0, 8.0]))
        model.classifier.bias.copy_(torch.tensor([0.0, 20.0, 40.0, 80.0]))

    copied = expand_output_labels(
        model,
        ["O", "B-person_name", "B-family_name"],
        source_cut="redaction_20_v1",
    )

    assert copied == 3
    assert torch.equal(model.classifier.weight[:, 0], torch.tensor([0.0, 3.0, 8.0]))
    assert torch.equal(model.classifier.bias, torch.tensor([0.0, 30.0, 80.0]))
    assert model.config.pii_warm_start_label_cut == "redaction_20_v1"
    assert model.config.pii_warm_start_source_rows_per_label == {
        "O": 1,
        "B-person_name": 2,
        "B-family_name": 1,
    }


def test_projected_warm_start_cut_groups_rejects_unknown_canonical_type() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        projected_warm_start_cut_groups(
            {"B-not_canonical": 0},
            {"B-person_name": 0},
            "redaction_20_v1",
        )


class _OffsetTokenizer:
    """Whitespace tokenizer with offsets, zero-width specials, and a truncation side."""

    truncation_side = "right"

    def __call__(self, text, truncation=False, max_length=None, return_offsets_mapping=False, **kwargs):
        offsets, position = [], 0
        for word in text.split():
            start = text.index(word, position)
            offsets.append((start, start + len(word)))
            position = start + len(word)
        ids = [0] + list(range(1, len(offsets) + 1)) + [0]
        offsets = [(0, 0)] + offsets + [(0, 0)]
        if truncation and max_length is not None and len(ids) > max_length:
            if self.truncation_side == "left":
                ids, offsets = ids[-max_length:], offsets[-max_length:]
            else:
                ids, offsets = ids[:max_length], offsets[:max_length]
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        if kwargs.get("return_special_tokens_mask"):
            encoded["special_tokens_mask"] = [int(a == b) for a, b in offsets]
        return encoded


def test_token_capacity_windows_preserve_tail_supervision(tmp_path):
    corpus = tmp_path / "tail.jsonl"
    text = "one two three four Anna"
    corpus.write_text(
        json.dumps(
            {
                "text": text,
                "spans": [[19, 23, "person_name"]],
                "supervision": ANNOTATED_SPANS_ONLY,
                "context": {"before": "Earlier", "after": "Later"},
            }
        )
        + "\n"
    )
    tokenizer = _OffsetTokenizer()
    labels = {"O": 0, "S-person_name": 1}
    legacy = window_records(corpus, 900, context_field="context")
    with pytest.raises(ValueError, match="no token-aligned span"):
        SpanDataset(legacy, tokenizer, labels, 4, context_field="context")[0]
    rows = window_records(corpus, 900, context_field="context", tokenizer=tokenizer, max_tokens=4)
    assert [row["text"][a:b] for row in rows for a, b, _ in row["spans"]] == ["Anna"]
    dataset = SpanDataset(rows, tokenizer, labels, 4, context_field="context")
    for index in range(len(rows)):
        item = dataset[index]
        assert len(item["input_ids"]) <= 4
        assert 1 in item["labels"]


def test_token_capacity_windows_keep_complete_text_and_protected_spans(tmp_path):
    corpus = tmp_path / "complete.jsonl"
    text = "one two three four five six"
    corpus.write_text(
        json.dumps(
            {
                "text": text,
                "spans": [[4, 13, "person_name"]],
                "ignored_spans": [[8, 18, "person_reference"]],
            }
        )
        + "\n"
    )
    tokenizer = _OffsetTokenizer()
    rows = window_records(corpus, 900, tokenizer=tokenizer, max_tokens=5, context_field="context")
    assert "".join(row["text"] for row in rows) == text
    for row in rows:
        provenance = row["training_window"]
        assert row["text"] == text[provenance["start"] : provenance["end"]]
        assert provenance["source_line_1based"] == 1
        assert row["context"]["before"] == text[: provenance["start"]]
        assert row["context"]["after"] == text[provenance["end"] :]
    assert [row["text"][a:b] for row in rows for a, b, _ in row["spans"]] == ["two three"]
    assert [row["text"][a:b] for row in rows for a, b, _ in row.get("ignored_spans", [])] == ["three four"]
    assert all(len(tokenizer(row["text"])["input_ids"]) <= 5 for row in rows)
    with pytest.raises(ValueError, match="indivisible span/character"):
        window_records(corpus, 900, tokenizer=tokenizer, max_tokens=4)


@pytest.mark.parametrize("context", [{"before": " Earlier. ", "after": " Later. "}, " Earlier. "])
def test_materialized_context_windows_survive_loader_round_trip(tmp_path, context):
    corpus = tmp_path / "source.jsonl"
    corpus.write_text(json.dumps({"text": "First. Second.", "spans": [], "context": context}) + "\n")
    tokenizer = _OffsetTokenizer()
    rows = window_records(corpus, 8, context_field="context", tokenizer=tokenizer, max_tokens=8)
    materialized = tmp_path / "windows.jsonl"
    materialized.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = window_records(materialized, 8, context_field="context", tokenizer=tokenizer, max_tokens=8)
    assert [(row["text"], row["context"]) for row in loaded] == [
        (row["text"], row["context"]) for row in rows
    ]


def _context_dataset(max_len):
    dataset = SpanDataset.__new__(SpanDataset)
    dataset.tok = _OffsetTokenizer()
    dataset.max_len = max_len
    dataset.context_field = "context_before"
    dataset.context_separator = "\n\n"
    dataset.context_side = "both"
    dataset.context_configurations = None
    dataset.document_start_marker = False
    return dataset


def test_context_tokens_are_visible_but_carry_no_target_offsets() -> None:
    row = {"id": "r", "text": "Anna met Bob", "context_before": "The hearing was postponed"}
    without = _context_dataset(64).encode_with_context({"id": "r", "text": row["text"]})[1]
    encoded, offsets = _context_dataset(64).encode_with_context(row)

    target = [(a, b) for a, b in offsets if b > a]
    # The target keeps its own character coordinates, so every span in the row still
    # lines up without being shifted.
    assert target == [(a, b) for a, b in without if b > a]
    assert row["text"][target[0][0] : target[-1][1]] == "Anna met Bob"
    # Context tokens are present as inputs but marked the way special tokens are.
    assert len(encoded["input_ids"]) == len(offsets)
    assert sum(1 for a, b in offsets if b <= a) == len(offsets) - len(target)
    assert len(encoded["input_ids"]) > len(without)


def test_context_is_truncated_before_the_target_sentence() -> None:
    row = {
        "id": "r",
        "text": "Anna met Bob",
        "context_before": "one two three four five six seven eight nine ten",
    }
    _, offsets = _context_dataset(8).encode_with_context(row)
    target = [(a, b) for a, b in offsets if b > a]
    assert len(offsets) == 8
    assert row["text"][target[0][0] : target[-1][1]] == "Anna met Bob"


def test_target_longer_than_the_budget_drops_context_entirely() -> None:
    row = {"id": "r", "text": "a b c d e f", "context_before": "ignored context"}
    _, offsets = _context_dataset(4).encode_with_context(row)
    assert len(offsets) == 4
    assert [(a, b) for a, b in offsets if b > a] == [(0, 1), (2, 3), (4, 5)]


def test_missing_or_empty_context_uses_the_plain_encoding() -> None:
    dataset = _context_dataset(64)
    plain = dataset.encode_with_context({"id": "r", "text": "Anna met Bob"})[1]
    empty = dataset.encode_with_context({"id": "r", "text": "Anna met Bob", "context_before": "  "})[1]
    assert plain == empty


@pytest.mark.parametrize("budget", [7, 8, 12, 64])
def test_two_sided_context_keeps_target_and_pretrained_specials(budget) -> None:
    row = {
        "id": "r",
        "text": "Anna met Bob",
        "context_before": {"before": "one two three four five", "after": "six seven eight nine ten"},
    }
    encoded, offsets = _context_dataset(budget).encode_with_context(row)
    assert len(offsets) == min(budget, 15)
    assert len(encoded["input_ids"]) == len(offsets)
    assert encoded["input_ids"][0] == encoded["input_ids"][-1] == 0
    assert [(a, b) for a, b in offsets if b > a] == [(0, 4), (5, 8), (9, 12)]
    assert encoded["attention_mask"] == [1] * len(offsets)


def test_two_sided_context_supervises_only_the_target(tmp_path) -> None:
    row = {
        "id": "r",
        "text": "Anna met Bob",
        "spans": [[0, 4, "person_name"], [9, 12, "person_name"]],
        "context": {"before": "Anna was waiting", "after": "Bob left later"},
    }
    labels = {"O": 0, **{f"{b}-person_name": i for i, b in enumerate("BIES", 1)}}
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps(row) + "\n")
    rows = window_records(corpus, 900, context_field="context")
    dataset = SpanDataset(rows, _OffsetTokenizer(), labels, 9, context_field="context")
    result = dataset[0]
    assert len(result["input_ids"]) == 9
    assert [v for v in result["labels"] if v != -100] == [labels["S-person_name"], 0, labels["S-person_name"]]
    assert sum(v == -100 for v in result["labels"]) == 6


@pytest.mark.parametrize("context", [{"before": "Earlier.", "after": "Later."}, "Earlier."])
def test_split_targets_keep_intervening_text_as_context(tmp_path, context) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "First. Second.", "spans": [], "context": context}) + "\n")
    rows = window_records(corpus, 8, context_field="context")
    assert [row["text"] for row in rows] == ["First.", " Second."]
    if isinstance(context, dict):
        assert rows[0]["context"] == {"before": "Earlier.", "after": " Second.\n\nLater."}
        assert rows[1]["context"] == {"before": "Earlier.\n\nFirst.", "after": "Later."}
    else:
        assert [row["context"] for row in rows] == ["Earlier.", "Earlier.\n\nFirst."]
    assert all("context" not in row for row in window_records(corpus, 8))


def test_previous_only_ignores_successors_added_by_target_splitting(tmp_path) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": "Anna. Bob.",
                "spans": [[0, 4, "person_name"], [6, 9, "person_name"]],
                "context": {"before": "Earlier.", "after": "Later."},
            }
        )
        + "\n"
    )
    rows = window_records(corpus, 6, context_field="context")
    assert len(rows) == 2
    assert "Bob." in rows[0]["context"]["after"]
    labels = {"O": 0, **{f"{b}-person_name": i for i, b in enumerate("BIES", 1)}}
    previous = SpanDataset(
        rows, _OffsetTokenizer(), labels, 64, context_field="context", context_side="previous"
    )
    both = SpanDataset(rows, _OffsetTokenizer(), labels, 64, context_field="context")
    for index, row in enumerate(rows):
        result = previous[index]
        assert len(result["input_ids"]) < len(both[index]["input_ids"])
        assert [v for v in result["labels"] if v != -100] == [labels["S-person_name"]]
        expected = previous.encode_with_context(row)
        row["context"]["after"] = "Arbitrary future text must not affect any token."
        assert previous.encode_with_context(row) == expected


def test_previous_only_with_no_predecessor_uses_plain_target() -> None:
    dataset = _context_dataset(64)
    dataset.context_side = "previous"
    row = {"text": "Anna", "context_before": {"before": "", "after": "Future."}}
    assert dataset.encode_with_context(row) == dataset.encode_with_context({"text": "Anna"})


def test_sampled_context_changes_each_draw_and_preserves_target_supervision():
    row = {
        "text": "Anna met Bob",
        "spans": [[0, 4, "person_name"], [9, 12, "person_name"]],
        "context": {"before": "Earlier words", "after": "Later words"},
    }
    labels = {"O": 0, **{f"{b}-person_name": i for i, b in enumerate("BIES", 1)}}
    dataset = SpanDataset(
        [row],
        _OffsetTokenizer(),
        labels,
        64,
        context_field="context",
        context_configurations=[[-1, 0, 1], [-1, 0], [0]],
    )
    state = pii_encoder_train.random.getstate()
    try:
        pii_encoder_train.random.seed(173)
        samples = [dataset[0] for _ in range(300)]
        pii_encoder_train.random.seed(173)
        assert samples == [dataset[0] for _ in range(300)]
    finally:
        pii_encoder_train.random.setstate(state)
    counts = Counter(len(sample["input_ids"]) for sample in samples)
    assert set(counts) == {5, 7, 9}
    assert all(70 < count < 130 for count in counts.values())
    for sample in samples:
        assert [v for v in sample["labels"] if v != -100] == [4, 0, 4]
    # Validation without the mixture remains full-context on every access.
    validation = SpanDataset([row], _OffsetTokenizer(), labels, 64, context_field="context")
    assert all(len(validation[0]["input_ids"]) == 9 for _ in range(10))


@pytest.mark.parametrize("offsets", [[0], [0, 1], [-1, 0], [-1, 0, 1]])
def test_context_mixture_at_document_boundary_matches_available_text(offsets):
    row = {"text": "Anna", "spans": [], "context": {"before": "", "after": "Next sentence"}}
    labels = {"O": 0}
    mixture = SpanDataset(
        [row], _OffsetTokenizer(), labels, 64, context_field="context", context_configurations=[offsets]
    )
    expected = dict(row, context={"before": "", "after": "Next sentence" if 1 in offsets else ""})
    plain = SpanDataset([expected], _OffsetTokenizer(), labels, 64, context_field="context")
    assert mixture[0] == plain[0]
    row["context"]["after"] = ""
    isolated = SpanDataset([row], _OffsetTokenizer(), labels, 64)
    assert mixture[0] == isolated[0]


def test_context_mixture_changes_encoder_training_updates(tmp_path):
    from transformers import BertConfig, BertForTokenClassification, Trainer, TrainingArguments

    row = {
        "text": "Anna met Bob",
        "spans": [[0, 4, "person_name"]],
        "context": {"before": "Earlier words", "after": "Later words"},
    }
    labels = {"O": 0, **{f"{b}-person_name": i for i, b in enumerate("BIES", 1)}}
    results = []
    for mixture in (None, [[-1, 0, 1]], [[0]]):
        torch.manual_seed(173)
        model = BertForTokenClassification(
            BertConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                num_labels=5,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        )
        trainer = o_token_weighted_trainer_class(Trainer)(
            model=model,
            args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
        )
        trainer.o_token_loss_weight = 0.75
        dataset = SpanDataset(
            [row], _OffsetTokenizer(), labels, 64, context_field="context", context_configurations=mixture
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(2):
            sample = dataset[0]
            batch = {k: torch.tensor([sample[k]]) for k in ("input_ids", "attention_mask", "labels")}
            optimizer.zero_grad()
            trainer.compute_loss(model, batch).backward()
            optimizer.step()
        results.append(model.bert.embeddings.position_embeddings.weight.detach().clone())
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])


def test_context_without_spare_capacity_uses_existing_target_truncation() -> None:
    dataset = _context_dataset(4)
    row = {"id": "r", "text": "one two three four", "context_before": {"before": "left", "after": "right"}}
    assert dataset.encode_with_context(row) == dataset.encode_with_context({"id": "r", "text": row["text"]})


@pytest.mark.parametrize("context", [{"before": "left"}, {"before": 1, "after": "right"}])
def test_invalid_document_context_is_rejected(context) -> None:
    with pytest.raises(ValueError, match="before and after"):
        _context_dataset(64).encode_with_context({"id": "r", "text": "Anna", "context_before": context})


def _domain_collator():
    collator = PartialLabelDataCollator.__new__(PartialLabelDataCollator)
    collator.tokenizer = SimpleNamespace(padding_side="right")
    # The real base collator pads every list it receives to the token length; this stands
    # in for it and would corrupt a posterior the same way if one reached it.
    collator.base = lambda features: {
        "input_ids": torch.tensor(
            [feature["input_ids"] + [0] * (3 - len(feature["input_ids"])) for feature in features]
        ),
        **{
            key: torch.tensor([feature[key] + [0] * (3 - len(feature[key])) for feature in features])
            for key in features[0]
            if key != "input_ids" and isinstance(features[0][key], list)
        },
    }
    return collator


def test_collator_keeps_a_domain_posterior_at_its_own_width() -> None:
    batch = _domain_collator()(
        [
            {"input_ids": [1, 2], "domain_segment_id": [0.0, 1.0, 0.0, 0.0]},
            {"input_ids": [1, 2, 3], "domain_segment_id": [1.0, 0.0, 0.0, 0.0]},
        ]
    )

    # Four domain classes, not padded out to the three-token sequence length.
    assert batch["domain_segment_id"].shape == (2, 4)
    assert batch["domain_segment_id"].dtype == torch.float
    assert batch["domain_segment_id"].tolist() == [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]


def test_collator_rejects_a_domain_posterior_missing_from_part_of_a_batch() -> None:
    with pytest.raises(ValueError, match="missing from part of a batch"):
        _domain_collator()(
            [
                {"input_ids": [1, 2], "domain_segment_id": [0.0, 1.0]},
                {"input_ids": [1, 2, 3]},
            ]
        )


def test_collator_rejects_domain_posteriors_of_disagreeing_width() -> None:
    with pytest.raises(ValueError, match="disagree in width"):
        _domain_collator()(
            [
                {"input_ids": [1, 2], "domain_segment_id": [0.0, 1.0]},
                {"input_ids": [1, 2, 3], "domain_segment_id": [0.0, 0.5, 0.5]},
            ]
        )


def test_type_only_supervision_is_indifferent_to_where_the_boundary_falls():
    """Foreign gold knows what an organization is, not where our annotators end it.

    Measured on the all-gold arm, admitting it cut outright organization misses
    by 15 and added 18 boundary errors, because a general-NER corpus carries its
    own convention about whether a leading article belongs inside the span. This
    loss keeps the first signal and discards the second, so every placement of
    the boundary within a type has to cost exactly the same.
    """
    import torch

    from scripts.pii_encoder_train import type_only_supervision_loss

    # 0 is O, 1 through 4 are the organization B, I, E and S tags, 5 is a person tag.
    organization = [[1, 2, 3, 4]]
    begins = torch.zeros(1, 1, 6)
    begins[0, 0, 1] = 8.0
    ends = torch.zeros(1, 1, 6)
    ends[0, 0, 3] = 8.0
    target = torch.tensor([[0]])

    assert type_only_supervision_loss(begins, target, organization) == pytest.approx(
        type_only_supervision_loss(ends, target, organization)
    )

    # Mass on a different type is penalised, and an unsupervised batch contributes
    # a zero the caller can add unconditionally.
    elsewhere = torch.zeros(1, 1, 6)
    elsewhere[0, 0, 5] = 8.0
    assert type_only_supervision_loss(elsewhere, target, organization) > type_only_supervision_loss(
        begins, target, organization
    )
    assert type_only_supervision_loss(begins, torch.full((1, 1), -100), organization) == 0.0


def test_differential_learning_rate_groups_give_residual_block_its_own_rate() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 4)
            self.head_residual_mlp = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 2)

        @property
        def base_model(self):
            return self.encoder

    model = TokenClassifier()
    groups = differential_learning_rate_groups(
        model, set(), head_lr=2e-5, encoder_lr=1e-5, weight_decay=0.0, block_lr=5e-4
    )
    rates = {group["group_name"]: group["lr"] for group in groups}
    assert rates == {"encoder-no-decay": 1e-5, "block-no-decay": 5e-4, "head-no-decay": 2e-5}
    block = next(group for group in groups if group["group_name"] == "block-no-decay")
    assert {id(p) for p in block["params"]} == {id(p) for p in model.head_residual_mlp.parameters()}
    with pytest.raises(ValueError, match="installed block"):
        del model.head_residual_mlp
        differential_learning_rate_groups(
            model, set(), head_lr=2e-5, encoder_lr=1e-5, weight_decay=0.0, block_lr=5e-4
        )
