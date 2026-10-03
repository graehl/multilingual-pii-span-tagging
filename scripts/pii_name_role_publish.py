#!/usr/bin/env python3
"""Fit and export the terminal all-data name-role character model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from scripts.pii_character_projection import (
    SCRIPT_V1_DATA_PATH,
    CharacterPairProjection,
    load_script_v1_projection,
)
from scripts.pii_name_role_char_pilot import (
    SCHEMA,
    NameRoleCharCNN,
    NameRoleDataset,
    cell_counts,
    encode_surface,
    evaluate_model,
    file_sha256,
    git_commit,
    load_unambiguous_examples,
    normalize_surface,
    partition_examples,
    sampling_plan,
    validate_output_directory,
)
from trainlib import WeightedLengthBatchSampler

ALL_DATA_SCHEMA = "pii-name-role-character-all-data-fit-v1"
EXPORT_SCHEMA = "pii-name-role-character-onnx-bundle-v1"
CHECKPOINT_SCHEMA = "pii-name-role-character-all-data-checkpoint-v1"
CONSUMED_DIAGNOSTIC_SCHEMA = "pii-name-role-character-consumed-diagnostic-v1"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def model_from_config(config: dict) -> NameRoleCharCNN:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"unsupported model config schema {config.get('schema')!r}")
    return NameRoleCharCNN(
        vocabulary_size=len(config["character_vocabulary"]),
        language_count=len(config["languages"]),
        embedding_dim=int(config["embedding_dim"]),
        convolution_channels=int(config["convolution_channels"]),
        language_dim=int(config["language_dim"]),
        language_dropout=float(config["language_dropout"]),
        language_noise=float(config["language_noise"]),
        shared_mixture_alpha=float(config["shared_mixture_alpha"]),
    )


def _relative_deployment_path(config_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{config_path}: {field} must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{config_path}: {field} must be relative to the deployment JSON")
    resolved = (config_path.parent / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{config_path}: missing {field} sidecar {resolved}")
    return resolved


@dataclass(frozen=True)
class NameKindOnnxDeployment:
    """Relocatable Python runtime for one published name-kind JSON contract."""

    config_path: Path
    config: dict
    model_path: Path
    script_projection: CharacterPairProjection
    session: Any

    @classmethod
    def load(
        cls,
        config_path: Path,
        *,
        session_factory: Callable[[str], Any] | None = None,
    ) -> NameKindOnnxDeployment:
        config_path = config_path.resolve()
        config = _load_json(config_path)
        if config.get("schema") != SCHEMA:
            raise ValueError(f"unsupported model config schema {config.get('schema')!r}")
        roles = config.get("roles")
        languages = config.get("languages")
        vocabulary = config.get("character_vocabulary")
        if not isinstance(roles, list) or not roles or not all(isinstance(item, str) for item in roles):
            raise ValueError(f"{config_path}: roles must be a nonempty string list")
        if not isinstance(languages, list) or len(set(languages)) != len(languages):
            raise ValueError(f"{config_path}: languages must be a unique list")
        if not isinstance(vocabulary, list) or len(set(vocabulary)) != len(vocabulary):
            raise ValueError(f"{config_path}: character_vocabulary must be a unique list")

        expected_inputs = {
            "char_ids": ["batch", int(config["max_characters"])],
            "language_probs": ["batch", len(languages)],
            "mixture_alpha": ["batch", 1],
        }
        expected_output = {
            "name": "log_probabilities",
            "shape": ["batch", len(roles)],
        }
        onnx_contract = config.get("onnx")
        if not isinstance(onnx_contract, dict):
            raise ValueError(f"{config_path}: missing onnx deployment contract")
        if onnx_contract.get("inputs") != expected_inputs:
            raise ValueError(f"{config_path}: onnx inputs disagree with model inventories")
        if onnx_contract.get("output") != expected_output:
            raise ValueError(f"{config_path}: onnx output disagrees with roles")
        model_path = _relative_deployment_path(config_path, onnx_contract.get("path"), "onnx.path")

        backoff = config.get("character_backoff")
        if not isinstance(backoff, dict) or backoff.get("unknown_character_backoff") != "script-block":
            raise ValueError(f"{config_path}: deployed name-kind model requires script-block backoff")
        projection_contract = backoff.get("script_projection")
        if not isinstance(projection_contract, dict):
            raise ValueError(f"{config_path}: missing character_backoff.script_projection")
        projection_path = _relative_deployment_path(
            config_path,
            projection_contract.get("path"),
            "character_backoff.script_projection.path",
        )
        projection_sha256 = file_sha256(projection_path)
        if projection_sha256 != projection_contract.get("sha256"):
            raise ValueError(f"{config_path}: SCRIPT projection hash mismatch: {projection_sha256}")
        script_projection = load_script_v1_projection(projection_path)

        if session_factory is None:
            import onnxruntime as ort

            session_factory = lambda path: ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"],
            )
        session = session_factory(str(model_path))
        observed_inputs = {item.name: [*item.shape] for item in session.get_inputs()}
        if observed_inputs != expected_inputs:
            raise ValueError(
                f"{config_path}: ONNX graph inputs {observed_inputs!r} do not match "
                f"deployment JSON {expected_inputs!r}"
            )
        outputs = session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != expected_output["name"]:
            raise ValueError(f"{config_path}: ONNX graph output does not match deployment JSON")
        if [*outputs[0].shape] != expected_output["shape"]:
            raise ValueError(f"{config_path}: ONNX graph output shape does not match deployment JSON")
        return cls(config_path, config, model_path, script_projection, session)

    def infer(
        self,
        surfaces: Sequence[str],
        languages: str | Sequence[str] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        if not surfaces:
            raise ValueError("at least one surface is required")
        if languages is None:
            requested_languages = [""] * len(surfaces)
        elif isinstance(languages, str):
            requested_languages = [languages] * len(surfaces)
        else:
            requested_languages = list(languages)
            if len(requested_languages) != len(surfaces):
                raise ValueError("languages must contain one value per surface")

        normalized = [normalize_surface(surface) for surface in surfaces]
        character_ids = {
            character: index for index, character in enumerate(self.config["character_vocabulary"])
        }
        characters = np.asarray(
            [
                encode_surface(
                    surface,
                    character_ids,
                    int(self.config["max_characters"]),
                    unknown_character_backoff="script-block",
                    script_projection=self.script_projection,
                )
                for surface in normalized
            ],
            dtype=np.int64,
        )
        language_ids = {language: index for index, language in enumerate(self.config["languages"])}
        language_probs = np.zeros((len(surfaces), len(language_ids)), dtype=np.float32)
        for row, language in enumerate(requested_languages):
            base = language.replace("_", "-").split("-", 1)[0].lower()
            if base in language_ids:
                language_probs[row, language_ids[base]] = 1.0
        mixture_alpha = np.full(
            (len(surfaces), 1),
            float(self.config["shared_mixture_alpha"]),
            dtype=np.float32,
        )
        output_name = self.config["onnx"]["output"]["name"]
        observed = self.session.run(
            [output_name],
            {
                "char_ids": characters,
                "language_probs": language_probs,
                "mixture_alpha": mixture_alpha,
            },
        )[0]
        expected_shape = (len(surfaces), len(self.config["roles"]))
        if observed.shape != expected_shape:
            raise ValueError(f"ONNX result has shape {observed.shape}; expected {expected_shape}")
        return normalized, observed


def load_model_output(output: Path) -> tuple[NameRoleCharCNN, dict, dict]:
    output = output.resolve()
    config_path = output / "config.json"
    model_path = output / "model.safetensors"
    config = _load_json(config_path)
    model = model_from_config(config)
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    receipt = {
        "output": str(output),
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "model": {"path": str(model_path), "sha256": file_sha256(model_path)},
    }
    return model, config, receipt


def _write_training_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_training_checkpoint(path: Path, signature: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported training checkpoint schema {state.get('schema')!r}")
    if state.get("signature") != signature:
        raise ValueError("training checkpoint does not match the requested fit")
    return state


def train_fixed_horizon(
    model: NameRoleCharCNN,
    dataset: NameRoleDataset,
    sampler_weights: Sequence[float],
    *,
    batch_size: int,
    samples_per_epoch: int,
    epochs: int,
    learning_rate: float,
    length_window_steps: int,
    seed: int,
    checkpoint_path: Path,
    checkpoint_signature: dict,
    resume: bool,
) -> tuple[list[dict], int, int]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    sampler = WeightedLengthBatchSampler(
        lengths=[min(len(example.surface) + 2, dataset.max_characters) for example in dataset.examples],
        weights=sampler_weights,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        epoch_examples=samples_per_epoch,
        length_window_steps=length_window_steps,
        seed=seed,
        # The recorded name-kind recipe predates the carried default.
        draw_policy="systematic",
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict] = []
    steps = 0
    completed_epoch = 0
    if resume:
        state = _load_training_checkpoint(checkpoint_path, checkpoint_signature)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        dataset.case_rng.setstate(state["case_rng_state"])
        dataset.backoff_rng.setstate(state["backoff_rng_state"])
        sampler._rng.setstate(state["sampler_rng_state"])
        sampler.epoch = int(state["sampler_epoch"])
        history = list(state["history"])
        steps = int(state["steps"])
        completed_epoch = int(state["completed_epoch"])
        if completed_epoch != len(history) or completed_epoch >= epochs:
            raise ValueError(
                f"checkpoint epoch {completed_epoch} is inconsistent with "
                f"history length {len(history)} or requested horizon {epochs}"
            )
    elapsed_before = float(history[-1]["elapsed_seconds"]) if history else 0.0
    run_started = time.perf_counter()
    for epoch in range(completed_epoch + 1, epochs + 1):
        model.train()
        loss_sum = 0.0
        rows = 0
        epoch_started = time.perf_counter()
        for characters, languages, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(characters, languages)
            loss = nn.functional.nll_loss(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            rows += len(targets)
            steps += 1
        scheduler.step()
        point = {
            "epoch": epoch,
            "steps": steps,
            "rows": rows,
            "train_loss": loss_sum / rows,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": elapsed_before + time.perf_counter() - run_started,
        }
        history.append(point)
        _write_training_checkpoint(
            checkpoint_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "signature": checkpoint_signature,
                "completed_epoch": epoch,
                "steps": steps,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "case_rng_state": dataset.case_rng.getstate(),
                "backoff_rng_state": dataset.backoff_rng.getstate(),
                "sampler_rng_state": sampler._rng.getstate(),
                "sampler_epoch": sampler.epoch,
            },
        )
        print(json.dumps({"phase": "all-data-train", **point}), flush=True)
    return history, steps, completed_epoch


def _former_partition_counts(parent_result: dict) -> dict:
    """The train/dev/test counts that were consumed, from either kind of parent.

    A partitioned fit records them under ``partition``; an all-data fit (a
    continuation of a continuation) only inherits them, so chase the chain.
    """
    if "partition" in parent_result:
        return parent_result["partition"]["counts"]
    return parent_result["composition"]["former_partition_counts"]


def _pre_lap_evidence(parent_result: dict, parent_result_path: Path) -> dict:
    """Last held-out scores before the evaluation partitions were consumed."""
    if "metrics" in parent_result:
        return {
            "result": {"path": str(parent_result_path), "sha256": file_sha256(parent_result_path)},
            "selected_epoch": parent_result["training"]["selected_epoch"],
            "completed_epochs": parent_result["training"]["completed_epochs"],
            "development_natural_language_macro_f1": parent_result["metrics"]["neural_development"][
                "natural_language"
            ]["macro_f1_observed_roles"],
            "diagnostic_test_natural_language_macro_f1": parent_result["metrics"]["neural_test"][
                "natural_language"
            ]["macro_f1_observed_roles"],
            "note": "Both former evaluation partitions entered this fit and cannot score it.",
        }
    inherited = dict(parent_result["pre_lap_evidence"])
    inherited["inherited_through"] = {
        "path": str(parent_result_path),
        "sha256": file_sha256(parent_result_path),
    }
    return inherited


def fit_all_data(args: argparse.Namespace) -> None:
    checkpoint_path = args.output / "training-checkpoint.pt"
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requires {checkpoint_path}")
        allowed = {checkpoint_path.name, checkpoint_path.name + ".tmp"}
        unexpected = [
            item
            for item in args.output.iterdir()
            if not item.name.endswith(".meta.md") and item.name not in allowed
        ]
        if unexpected:
            raise FileExistsError(f"resume output contains unexpected artifacts: {unexpected}")
    else:
        validate_output_directory(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.training_threads)

    model, config, parent = load_model_output(args.init_from_output)
    backoff = config["character_backoff"]
    if not backoff.get("hard_replacement") or backoff["rare_character_backoff_probability"] != 1:
        raise ValueError("all-data continuation currently requires a hard-backoff parent vocabulary")
    examples, loading = load_unambiguous_examples(args.context_jsonl)
    observed_languages = {example.language for example in examples}
    configured_languages = set(config["languages"])
    if observed_languages != configured_languages:
        raise ValueError(
            "all-data languages differ from the parent config: "
            f"missing={sorted(configured_languages - observed_languages)!r}, "
            f"new={sorted(observed_languages - configured_languages)!r}"
        )
    sampler_weights, language_mass, language_receipt = sampling_plan(
        examples,
        language_round=args.language_round,
        maximum_support_multiplier=args.maximum_support_multiplier,
        full_support_effective_cell_weight=args.full_support_effective_cell_weight,
    )
    character_ids = {character: index for index, character in enumerate(config["character_vocabulary"])}
    language_ids = {language: index for index, language in enumerate(config["languages"])}
    case_weights = (
        args.case_natural_weight,
        args.case_uppercase_weight,
        args.case_lowercase_weight,
    )
    dataset = NameRoleDataset(
        examples,
        character_ids=character_ids,
        language_ids=language_ids,
        max_characters=int(config["max_characters"]),
        case_weights=case_weights,
        unknown_character_backoff=backoff["unknown_character_backoff"],
        rare_characters=frozenset(),
        rare_character_backoff_probability=1.0,
        frequent_character_backoff_probability=float(backoff["frequent_character_backoff_probability"]),
        seed=args.seed,
    )
    checkpoint_signature = {
        "input_sha256": file_sha256(args.context_jsonl),
        "parent_model_sha256": parent["model"]["sha256"],
        "parent_config_sha256": parent["config"]["sha256"],
        "language_round_sha256": file_sha256(args.language_round),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "samples_per_epoch": args.samples_per_epoch,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "length_window_steps": args.length_window_steps,
        "maximum_support_multiplier": args.maximum_support_multiplier,
        "full_support_effective_cell_weight": args.full_support_effective_cell_weight,
        "case_weights": list(case_weights),
    }
    history, steps, resumed_from_epoch = train_fixed_horizon(
        model,
        dataset,
        sampler_weights,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        length_window_steps=args.length_window_steps,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        checkpoint_signature=checkpoint_signature,
        resume=args.resume,
    )

    model_path = args.output / "model.safetensors"
    config_path = args.output / "config.json"
    result_path = args.output / "result.json"
    save_file(model.state_dict(), model_path)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    parent_result_path = args.init_from_output.resolve() / "result.json"
    parent_result = _load_json(parent_result_path)
    result = {
        "schema": ALL_DATA_SCHEMA,
        "status": "completed_terminal_all_data_fit",
        "git_commit": git_commit(),
        "agentctl_run_id": os.environ.get("AGENTCTL_RUN_ID"),
        "seed": args.seed,
        "input": {
            "path": str(args.context_jsonl.resolve()),
            "sha256": file_sha256(args.context_jsonl),
            **loading,
        },
        "composition": {
            "policy": (
                "all globally unambiguous normalized-key/language examples; the former "
                "train, development, and test partitions are all eligible"
            ),
            "rows": len(examples),
            "cells": cell_counts(examples),
            "former_partition_counts": _former_partition_counts(parent_result),
            "fresh_evaluation_remaining": False,
        },
        "sampling": {
            "expected_language_mass": language_mass,
            "samples_per_epoch": args.samples_per_epoch,
            "receipt": language_receipt,
        },
        "training": {
            "initialization": parent,
            "terminal_weights_policy": "last epoch; no post-merge selector",
            "resumed_from_epoch": resumed_from_epoch,
            "epochs": args.epochs,
            "optimizer_steps": steps,
            "sampled_rows": args.epochs * args.samples_per_epoch,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": 1e-4,
                "schedule": "cosine annealing to zero over the fixed all-data horizon",
            },
            "case_intent_weights": {
                "natural_or_reconstructed": args.case_natural_weight,
                "all_uppercase": args.case_uppercase_weight,
                "all_lowercase": args.case_lowercase_weight,
            },
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
        },
        "pre_lap_evidence": _pre_lap_evidence(parent_result, parent_result_path),
        "artifacts": {
            "model": {
                "path": str(model_path.resolve()),
                "bytes": model_path.stat().st_size,
                "sha256": file_sha256(model_path),
            },
            "config": {
                "path": str(config_path.resolve()),
                "bytes": config_path.stat().st_size,
                "sha256": file_sha256(config_path),
            },
            "training_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": file_sha256(checkpoint_path),
                "published": False,
            },
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": "all-data-complete",
                "result": str(result_path.resolve()),
                "model_sha256": result["artifacts"]["model"]["sha256"],
                "epochs": args.epochs,
                "sampled_rows": result["training"]["sampled_rows"],
            }
        ),
        flush=True,
    )


def evaluate_consumed(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.inference_threads)
    model, config, source = load_model_output(args.fit_output)
    examples, loading = load_unambiguous_examples(args.context_jsonl)
    partitions = partition_examples(
        examples,
        train_cap_per_cell=args.train_cap_per_language_role,
        evaluation_cap_per_cell=args.evaluation_cap_per_language_role,
    )
    evaluated = partitions[args.partition]
    metrics = evaluate_model(
        model,
        evaluated,
        character_ids={character: index for index, character in enumerate(config["character_vocabulary"])},
        language_ids={language: index for index, language in enumerate(config["languages"])},
        max_characters=int(config["max_characters"]),
        unknown_character_backoff=config["character_backoff"]["unknown_character_backoff"],
        batch_size=args.batch_size,
    )
    natural = metrics["natural_language"]
    predicted = {role: values["predicted"] for role, values in natural["per_role"].items()}
    nondegenerate = (
        math.isfinite(natural["macro_f1_observed_roles"])
        and natural["macro_f1_observed_roles"] >= args.minimum_sanity_f1
        and all(count > 0 for count in predicted.values())
    )
    parent_result_path = args.pre_lap_output.resolve() / "result.json"
    parent_result = _load_json(parent_result_path)
    parent_metrics = parent_result["metrics"][f"neural_{args.partition}"]["natural_language"]
    result = {
        "schema": CONSUMED_DIAGNOSTIC_SCHEMA,
        "status": "passed_consumed_training_overlap_sanity" if nondegenerate else "failed_sanity",
        "git_commit": git_commit(),
        "agentctl_run_id": os.environ.get("AGENTCTL_RUN_ID"),
        "interpretation": {
            "trained_on_partition": True,
            "selection_use": False,
            "continuation_use": False,
            "model_card_use": False,
            "purpose": "confirm the terminal all-data trajectory is nondegenerate",
        },
        "input": {
            "path": str(args.context_jsonl.resolve()),
            "sha256": file_sha256(args.context_jsonl),
            **loading,
        },
        "partition": {
            "name": args.partition,
            "rows": len(evaluated),
            "train_cap_per_language_role": args.train_cap_per_language_role,
            "evaluation_cap_per_language_role": args.evaluation_cap_per_language_role,
        },
        "terminal_model": source,
        "pre_lap": {
            "result": {
                "path": str(parent_result_path),
                "sha256": file_sha256(parent_result_path),
            },
            "natural_language": parent_metrics,
        },
        "terminal_consumed_diagnostic": metrics,
        "nondegeneracy": {
            "passed": nondegenerate,
            "minimum_sanity_f1": args.minimum_sanity_f1,
            "both_roles_predicted": all(count > 0 for count in predicted.values()),
            "predicted_rows_by_role": predicted,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": "consumed-diagnostic-complete",
                "output": str(args.output.resolve()),
                "partition": args.partition,
                "rows": len(evaluated),
                "pre_lap_macro_f1": parent_metrics["macro_f1_observed_roles"],
                "terminal_macro_f1": natural["macro_f1_observed_roles"],
                "nondegenerate": nondegenerate,
            }
        ),
        flush=True,
    )
    if not nondegenerate:
        raise RuntimeError("terminal model failed the consumed-partition nondegeneracy sanity check")


def parity_batch(config: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    surfaces = ("Alice", "SMITH", "DeMarcus", "O'Connor", "李", "김")
    requested_languages = ("en", "en", "en", "en", "zh", "ko")
    character_ids = {character: index for index, character in enumerate(config["character_vocabulary"])}
    language_ids = {language: index for index, language in enumerate(config["languages"])}
    characters = torch.tensor(
        [
            encode_surface(
                surface,
                character_ids,
                int(config["max_characters"]),
                unknown_character_backoff=config["character_backoff"]["unknown_character_backoff"],
            )
            for surface in surfaces
        ],
        dtype=torch.int64,
    )
    language_probs = torch.zeros((len(surfaces), len(language_ids)), dtype=torch.float32)
    for row, language in enumerate(requested_languages):
        if language in language_ids:
            language_probs[row, language_ids[language]] = 1.0
    mixture_alpha = torch.full(
        (len(surfaces), 1),
        float(config["shared_mixture_alpha"]),
        dtype=torch.float32,
    )
    return characters, language_probs, mixture_alpha


def infer_onnx(args: argparse.Namespace) -> None:
    deployment = NameKindOnnxDeployment.load(args.config)
    normalized, log_probabilities = deployment.infer(args.surface, args.language)
    roles = deployment.config["roles"]
    rows = []
    for surface, normalized_surface, values in zip(
        args.surface,
        normalized,
        log_probabilities,
        strict=True,
    ):
        best = int(values.argmax())
        rows.append(
            {
                "surface": surface,
                "normalized_surface": normalized_surface,
                "predicted_role": roles[best],
                "log_probabilities": {role: float(values[index]) for index, role in enumerate(roles)},
            }
        )
    print(
        json.dumps(
            {
                "schema": "pii-name-kind-python-inference-v1",
                "config": str(deployment.config_path),
                "model": str(deployment.model_path),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def export_onnx(args: argparse.Namespace) -> None:
    validate_output_directory(args.output)
    model, config, source = load_model_output(args.fit_output)
    model.eval()
    characters, language_probs, mixture_alpha = parity_batch(config)
    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / f"{args.stem}.onnx"
    config_path = args.output / f"{args.stem}.config.json"
    script_encoding_path = args.output / f"{args.stem}-script-encoding.json"
    grammar_path = args.output / "name-postprocessor.json" if args.grammars else None
    readme_path = args.output / f"{args.stem}.README.md"
    manifest_path = args.output / f"{args.stem}.manifest.json"
    torch.onnx.export(
        model,
        (characters, language_probs, mixture_alpha),
        onnx_path,
        input_names=["char_ids", "language_probs", "mixture_alpha"],
        output_names=["log_probabilities"],
        dynamic_axes={
            "char_ids": {0: "batch"},
            "language_probs": {0: "batch"},
            "mixture_alpha": {0: "batch"},
            "log_probabilities": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    deployment_config = copy.deepcopy(config)
    deployment_config["onnx"] = {
        "path": onnx_path.name,
        "inputs": {
            "char_ids": ["batch", int(config["max_characters"])],
            "language_probs": ["batch", len(config["languages"])],
            "mixture_alpha": ["batch", 1],
        },
        "output": {
            "name": "log_probabilities",
            "shape": ["batch", len(config["roles"])],
        },
    }
    projection_contract = deployment_config["character_backoff"]["script_projection"]
    if file_sha256(args.script_encoding) != projection_contract["sha256"]:
        raise ValueError("SCRIPT encoding does not match the model config")
    projection_contract["path"] = script_encoding_path.name
    config_path.write_text(
        json.dumps(deployment_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(args.script_encoding, script_encoding_path)
    if grammar_path is not None:
        shutil.copyfile(args.grammars, grammar_path)
    shutil.copyfile(args.model_card, readme_path)

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with torch.inference_mode():
        expected = model(characters, language_probs, mixture_alpha=mixture_alpha).cpu().numpy()
    deployment = NameKindOnnxDeployment.load(config_path)
    _normalized, observed = deployment.infer(
        ("Alice", "SMITH", "DeMarcus", "O'Connor", "李", "김"),
        ("en", "en", "en", "en", "zh", "ko"),
    )
    absolute = np.abs(expected - observed)
    parity = {
        "rows": int(expected.shape[0]),
        "maximum_absolute_difference": float(absolute.max()),
        "mean_absolute_difference": float(absolute.mean()),
        "exact_argmax": bool(np.array_equal(expected.argmax(axis=1), observed.argmax(axis=1))),
        "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(expected, observed, rtol=1e-5, atol=1e-6)),
        "providers": ort.get_available_providers(),
    }
    if not parity["exact_argmax"] or not parity["allclose_rtol_1e-5_atol_1e-6"]:
        raise ValueError(f"ONNX parity failed: {parity}")
    fit_result_path = args.fit_output.resolve() / "result.json"
    bundled_paths = [onnx_path, config_path, script_encoding_path, readme_path]
    if grammar_path is not None:
        bundled_paths.append(grammar_path)
    files = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in bundled_paths}
    manifest = {
        "schema": EXPORT_SCHEMA,
        "status": "accepted_cpu_onnx_parity",
        "git_commit": git_commit(),
        "agentctl_run_id": os.environ.get("AGENTCTL_RUN_ID"),
        "stem": args.stem,
        "source": source,
        "fit_result": {"path": str(fit_result_path), "sha256": file_sha256(fit_result_path)},
        "contract": {
            "inputs": {
                "char_ids": ["batch", int(config["max_characters"])],
                "language_probs": ["batch", len(config["languages"])],
                "mixture_alpha": ["batch", 1],
            },
            "output": {"log_probabilities": ["batch", len(config["roles"])]},
            "roles": config["roles"],
            "normalization": "Unicode NFKC, Unicode whitespace runs to ASCII space, trim edges",
        },
        "parity": parity,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "onnx-export-complete",
                "manifest": str(manifest_path.resolve()),
                "onnx_sha256": files[onnx_path.name]["sha256"],
                "parity": parity,
            }
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-all-data")
    fit.add_argument("--context-jsonl", type=Path, required=True)
    fit.add_argument("--language-round", type=Path, required=True)
    fit.add_argument("--init-from-output", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=155)
    fit.add_argument("--batch-size", type=int, default=1024)
    fit.add_argument("--samples-per-epoch", type=int, default=120_000)
    fit.add_argument("--epochs", type=int, default=60)
    fit.add_argument("--learning-rate", type=float, default=0.002)
    fit.add_argument("--length-window-steps", type=int, default=8)
    fit.add_argument("--maximum-support-multiplier", type=float, default=2.0)
    fit.add_argument("--full-support-effective-cell-weight", type=float, default=1000.0)
    fit.add_argument("--case-natural-weight", type=float, default=0.7)
    fit.add_argument("--case-uppercase-weight", type=float, default=0.2)
    fit.add_argument("--case-lowercase-weight", type=float, default=0.1)
    fit.add_argument("--training-threads", type=int, default=4)
    fit.add_argument("--resume", action="store_true")
    fit.set_defaults(func=fit_all_data)

    diagnostic = commands.add_parser("evaluate-consumed")
    diagnostic.add_argument("--context-jsonl", type=Path, required=True)
    diagnostic.add_argument("--fit-output", type=Path, required=True)
    diagnostic.add_argument("--pre-lap-output", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--partition", choices=("development", "test"), default="development")
    diagnostic.add_argument("--train-cap-per-language-role", type=int, default=40_000)
    diagnostic.add_argument("--evaluation-cap-per-language-role", type=int, default=10_000)
    diagnostic.add_argument("--batch-size", type=int, default=2048)
    diagnostic.add_argument("--inference-threads", type=int, default=1)
    diagnostic.add_argument("--minimum-sanity-f1", type=float, default=0.5)
    diagnostic.set_defaults(func=evaluate_consumed)

    export = commands.add_parser("export-onnx")
    export.add_argument("--fit-output", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--model-card", type=Path, required=True)
    export.add_argument("--grammars", type=Path)
    export.add_argument("--script-encoding", type=Path, default=SCRIPT_V1_DATA_PATH)
    export.add_argument("--stem", default="name-kind")
    export.add_argument("--opset", type=int, default=17)
    export.set_defaults(func=export_onnx)

    infer = commands.add_parser("infer-onnx")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--surface", action="append", required=True)
    infer.add_argument("--language")
    infer.set_defaults(func=infer_onnx)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
