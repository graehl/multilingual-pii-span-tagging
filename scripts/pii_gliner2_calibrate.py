#!/usr/bin/env python
"""Train one frozen GLiNER2 P9 calibration trajectory and select its epoch."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pii_calibration_score import (  # noqa: E402
    aggregate_window_candidates,
    load_language_weights,
    read_jsonl,
    select_curve_point,
    threshold_curve,
)
from pii_gliner2_calibration_smoke import (  # noqa: E402
    LORA_TARGETS,
    install_batch_span_exclusions,
    install_c_weighted_loss,
    parameter_receipt,
    parse_entity_predictions,
)
from pii_gliner2_exact import (  # noqa: E402
    collate_exact_entity_rows_with_masks,
    configure_processor,
    configure_span_width,
    transform_exact_entity_row,
)
from pii_gliner2_projection import file_sha256  # noqa: E402
from pii_projector import Tagset  # noqa: E402

CALIBRATION_VERSION = "gliner2-p9-c-trajectory-v1"
DEFAULT_WIDTH_TIERS = (8, 16, 32, 48)
DEFAULT_THRESHOLDS = tuple(round(index / 100, 2) for index in range(1, 101))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def represented_width(record: Any) -> int:
    maximum = 1
    structure = record.structure_labels[0]
    for fields in structure[1]:
        for span_or_spans in fields:
            if span_or_spans is None or span_or_spans == (-1, -1):
                continue
            spans = span_or_spans if isinstance(span_or_spans, list) else [span_or_spans]
            for start, end in spans:
                maximum = max(maximum, int(end) - int(start) + 1)
    return maximum


def assign_width_tiers(
    processor: Any,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    tiers: Sequence[int],
    max_input_tokens: int,
) -> dict[int, list[int]]:
    if list(tiers) != sorted(set(tiers)) or min(tiers) <= 0:
        raise ValueError("width tiers must be unique, ascending, and positive")
    result: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        unrepresentable_receipt: list[dict[str, Any]] = []
        record = transform_exact_entity_row(
            processor,
            row,
            labels,
            max_input_tokens=max_input_tokens,
            max_span_width=max(tiers),
            unrepresentable_receipt=unrepresentable_receipt,
        )
        required = represented_width(record)
        try:
            tier = next(value for value in tiers if value >= required)
        except StopIteration as error:
            raise ValueError(f"row {index} needs span width {required}>{max(tiers)}") from error
        result[tier].append(index)
    return dict(sorted(result.items()))


def epoch_batches(
    tier_indices: dict[int, list[int]], *, batch_size: int, seed: int, epoch: int
) -> list[tuple[int, list[int]]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    generator = random.Random(seed + epoch * 1_000_003)
    batches = []
    for tier, indices in sorted(tier_indices.items()):
        shuffled = list(indices)
        generator.shuffle(shuffled)
        batches.extend(
            (tier, shuffled[start : start + batch_size]) for start in range(0, len(shuffled), batch_size)
        )
    generator.shuffle(batches)
    return batches


def optimizer_groups(
    model: Any, *, encoder_lr: float, task_lr: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    encoder = []
    task = []
    counts = Counter()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".encoder." in name:
            encoder.append(parameter)
            counts["encoder"] += parameter.numel()
        else:
            task.append(parameter)
            counts["task"] += parameter.numel()
    if not encoder or not task:
        raise ValueError("expected nonempty trainable encoder and task LoRA groups")
    return (
        [
            {"params": encoder, "lr": encoder_lr},
            {"params": task, "lr": task_lr},
        ],
        dict(counts),
    )


def cosine_schedule(optimizer: Any, *, total_steps: int, warmup_steps: int) -> Any:
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup steps must be in [0, total steps)")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def extract_window_candidates(
    model: Any,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    threshold: float,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    results = model.batch_extract_entities(
        [row["input"] for row in rows],
        list(labels),
        batch_size=batch_size,
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
    )
    if len(results) != len(rows):
        raise RuntimeError(f"GLiNER2 returned {len(results)}/{len(rows)} selection rows")
    return [parse_entity_predictions(result) for result in results]


def selection_key(point: dict[str, Any]) -> tuple[float, float, float, float]:
    weighted = point["importance_weighted_language_macro"]
    return (
        float(weighted["overlap_p1"]["F1"]),
        float(weighted["overlap_typed_p9"]["F1"]),
        float(weighted["character_p1"]["F1"]),
        -float(point["threshold"]),
    )


def latest_complete_epoch(output_dir: Path) -> int:
    epochs = []
    for path in output_dir.glob("checkpoint-epoch-*"):
        try:
            epoch = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if all(
            (path / name).is_file() for name in ("adapter_config.json", "optimizer.pt", "checkpoint.json")
        ):
            epochs.append(epoch)
    return max(epochs, default=0)


def contract(args: argparse.Namespace) -> dict[str, Any]:
    materialization = args.materialization.resolve()
    membership = args.membership.resolve()
    return {
        "version": CALIBRATION_VERSION,
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": file_sha256(args.model / "config.json"),
        },
        "materialization": {
            "path": str(materialization),
            "ledger_sha256": file_sha256(materialization / "ledger.json"),
            "train_sha256": file_sha256(materialization / "train.jsonl"),
            "selection_sha256": file_sha256(materialization / "selection.jsonl"),
        },
        "membership": {
            "path": str(membership),
            "manifest_sha256": file_sha256(membership / "manifest.json"),
            "selection_sha256": file_sha256(membership / "selection.jsonl"),
        },
        "configuration": {
            "positive_weight": args.positive_weight,
            "negative_keep": args.negative_keep,
            "mask_seed": args.mask_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr,
            "task_lr": args.task_lr,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": list(LORA_TARGETS),
            "width_tiers": list(args.width_tiers),
            "max_input_tokens": args.max_input_tokens,
            "candidate_floor": args.candidate_floor,
            "thresholds": list(DEFAULT_THRESHOLDS),
            "seed": args.seed,
            "development_limits": {
                "train_windows": args.train_limit,
                "selection_windows": args.selection_window_limit,
            },
        },
    }


def prepare_output(output_dir: Path, expected: dict[str, Any], resume: str) -> int:
    contract_path = output_dir / "contract.json"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        write_json_atomic(contract_path, expected)
        return 0
    if not contract_path.is_file():
        raise FileExistsError(f"existing output lacks contract: {output_dir}")
    observed = json.loads(contract_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError(f"existing trajectory contract differs: {output_dir}")
    if resume == "none":
        raise FileExistsError(f"trajectory output already exists and resume is disabled: {output_dir}")
    return latest_complete_epoch(output_dir)


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    *,
    epoch: int,
    checkpoint_record: dict[str, Any],
    selection_candidates: dict[str, list[dict[str, Any]]],
    selection_boundary_adjustments: Sequence[dict[str, Any]],
) -> Path:
    destination = output_dir / f"checkpoint-epoch-{epoch}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = output_dir / f".checkpoint-epoch-{epoch}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(str(temporary))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary / "optimizer.pt",
    )
    candidate_name = "selection-candidates.jsonl"
    candidate_rows = [
        {"source_id": source_id, "candidates": selection_candidates[source_id]}
        for source_id in sorted(selection_candidates)
    ]
    write_jsonl_atomic(temporary / candidate_name, candidate_rows)
    checkpoint_record["selection"]["candidates"] = {
        "file": candidate_name,
        "sha256": file_sha256(temporary / candidate_name),
        "documents": len(candidate_rows),
        "spans": sum(len(row["candidates"]) for row in candidate_rows),
    }
    adjustment_name = "selection-boundary-adjustments.jsonl"
    write_jsonl_atomic(temporary / adjustment_name, selection_boundary_adjustments)
    checkpoint_record["selection"]["boundary_adjustments"] = {
        "file": adjustment_name,
        "sha256": file_sha256(temporary / adjustment_name),
        "count": len(selection_boundary_adjustments),
    }
    checkpoint_record["adapter"] = {
        "path": str(destination),
        "weights_sha256": file_sha256(temporary / "adapter_model.safetensors"),
    }
    write_json_atomic(temporary / "checkpoint.json", checkpoint_record)
    temporary.replace(destination)
    return destination


def load_model(args: argparse.Namespace, completed_epoch: int) -> tuple[Any, Any]:
    from gliner2 import GLiNER2

    base = GLiNER2.from_pretrained(str(args.model), map_location=args.device)
    configure_processor(base.processor)
    configure_span_width(base, max(args.width_tiers))
    install_c_weighted_loss(
        base,
        positive_weight=args.positive_weight,
        negative_keep=args.negative_keep,
        mask_seed=args.mask_seed,
    )
    if completed_epoch:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            base,
            str(args.output_dir / f"checkpoint-epoch-{completed_epoch}"),
            is_trainable=True,
        )
    else:
        model = base.apply_lora(
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            targets=list(LORA_TARGETS),
        )
    return base, model


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    expected_contract = contract(args)
    completed_epoch = prepare_output(args.output_dir, expected_contract, args.resume)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_rows = read_jsonl(args.materialization / "train.jsonl")
    selection_windows = read_jsonl(args.materialization / "selection.jsonl")
    selection_sources = read_jsonl(args.membership / "selection.jsonl")
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.selection_window_limit:
        selection_windows = selection_windows[: args.selection_window_limit]
        selected_source_ids = {str(row["provenance"]["source_id"]) for row in selection_windows}
        selection_sources = [row for row in selection_sources if str(row["id"]) in selected_source_ids]
    labels = sorted(Tagset().cut_targets("redaction_9_v1"))
    language_weights = load_language_weights(args.language_round)
    base, model = load_model(args, completed_epoch)
    receipt = parameter_receipt(model)
    if receipt["total"] != 1_585_152:
        raise ValueError(f"unexpected GLiNER2 trainable parameter count: {receipt['total']}")
    tier_indices = assign_width_tiers(
        base.processor,
        train_rows,
        labels,
        tiers=args.width_tiers,
        max_input_tokens=args.max_input_tokens,
    )
    steps_per_epoch = sum(math.ceil(len(indices) / args.batch_size) for indices in tier_indices.values())
    total_steps = steps_per_epoch * args.epochs
    groups, optimizer_counts = optimizer_groups(model, encoder_lr=args.encoder_lr, task_lr=args.task_lr)
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=round(total_steps * args.warmup_ratio),
    )
    global_step = completed_epoch * steps_per_epoch
    if completed_epoch:
        state = torch.load(
            args.output_dir / f"checkpoint-epoch-{completed_epoch}" / "optimizer.pt",
            map_location=args.device,
            weights_only=True,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    history = []
    existing_records = []
    for epoch in range(1, completed_epoch + 1):
        existing_records.append(
            json.loads(
                (args.output_dir / f"checkpoint-epoch-{epoch}" / "checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        model.train()
        losses = []
        grad_norms = []
        for tier, indices in epoch_batches(
            tier_indices,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
        ):
            configure_span_width(base, tier)
            rows = [train_rows[index] for index in indices]
            batch, exclusion_masks = collate_exact_entity_rows_with_masks(
                base.processor,
                rows,
                labels,
                max_input_tokens=args.max_input_tokens,
                max_span_width=tier,
            )
            install_batch_span_exclusions(base, exclusion_masks, step=global_step)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if int(output["batch_size"]) != len(rows):
                raise RuntimeError(f"GLiNER2 accepted {output['batch_size']}/{len(rows)} rows")
            loss = output["total_loss"] / len(rows)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step + 1}: {loss.item()}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if not torch.isfinite(gradient) or gradient.item() == 0:
                raise RuntimeError(f"invalid gradient at step {global_step + 1}: {gradient.item()}")
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.item()))
            grad_norms.append(float(gradient.item()))
        configure_span_width(base, max(args.width_tiers))
        model.eval()
        selection_predictions = extract_window_candidates(
            model,
            selection_windows,
            labels,
            threshold=args.candidate_floor,
            batch_size=args.eval_batch_size,
        )
        boundary_adjustments: list[dict[str, Any]] = []
        candidates = aggregate_window_candidates(
            selection_sources,
            selection_windows,
            selection_predictions,
            boundary_receipt=boundary_adjustments,
        )
        curve = threshold_curve(
            selection_sources,
            candidates,
            thresholds=DEFAULT_THRESHOLDS,
            language_weights=language_weights,
        )
        selected = select_curve_point(curve)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {
                "mean_loss": sum(losses) / len(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "mean_preclip_gradient_norm": sum(grad_norms) / len(grad_norms),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            },
            "selection": {
                "criterion": (
                    "maximize selection-only 4:2:1 language-macro P1 symmetric-overlap F1; "
                    "tie-break typed-P9 overlap F1, character F1, then lower threshold"
                ),
                "selected_point": selected,
                "curve": curve,
            },
        }
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.output_dir,
            epoch=epoch,
            checkpoint_record=record,
            selection_candidates=candidates,
            selection_boundary_adjustments=boundary_adjustments,
        )
        existing_records.append(record)
        history.append(
            {
                "epoch": epoch,
                "mean_loss": record["train"]["mean_loss"],
                "selection_threshold": selected["threshold"],
                "selection_weighted_overlap_p1_f1": selected["importance_weighted_language_macro"][
                    "overlap_p1"
                ]["F1"],
            }
        )
        print(
            f"GLINER2_CALIBRATE epoch={epoch}/{args.epochs} step={global_step} "
            f"loss={record['train']['mean_loss']:.6f} "
            f"selection-F1={history[-1]['selection_weighted_overlap_p1_f1']:.6f}",
            flush=True,
        )
    selected_record = max(
        existing_records,
        key=lambda item: (selection_key(item["selection"]["selected_point"]), -item["epoch"]),
    )
    report = {
        "version": CALIBRATION_VERSION,
        "contract": expected_contract,
        "parameters": receipt,
        "optimizer_parameter_groups": optimizer_counts,
        "tier_documents": {str(tier): len(indices) for tier, indices in tier_indices.items()},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "checkpoints": [
            {
                "epoch": item["epoch"],
                "global_step": item["global_step"],
                "adapter": item["adapter"],
                "train": item["train"],
                "selection_point": item["selection"]["selected_point"],
            }
            for item in existing_records
        ],
        "selected": {
            "epoch": selected_record["epoch"],
            "adapter": selected_record["adapter"],
            "selection_point": selected_record["selection"]["selected_point"],
            "selection_threshold_is_not_postcal": True,
        },
        "elapsed_seconds_this_invocation": time.time() - started,
        "agentctl_run_id": str(__import__("os").environ.get("AGENTCTL_RUN_ID", "untracked")),
    }
    write_json_atomic(args.output_dir / "report.json", report)
    print(
        f"GLINER2_CALIBRATION_COMPLETE: C={args.positive_weight:g} "
        f"selected-epoch={selected_record['epoch']} -> {args.output_dir}",
        flush=True,
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--materialization", type=Path, required=True)
    result.add_argument("--membership", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--language-round", type=Path, default=HERE / "pii_language_round.yaml")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--positive-weight", type=float, required=True)
    result.add_argument("--negative-keep", type=float, default=0.5)
    result.add_argument("--mask-seed", type=int, default=20260812)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--eval-batch-size", type=int, default=8)
    result.add_argument("--encoder-lr", type=float, default=1e-5)
    result.add_argument("--task-lr", type=float, default=5e-4)
    result.add_argument("--warmup-ratio", type=float, default=0.05)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    result.add_argument("--lora-rank", type=int, default=8)
    result.add_argument("--lora-alpha", type=float, default=16.0)
    result.add_argument("--width-tiers", type=int, nargs="+", default=list(DEFAULT_WIDTH_TIERS))
    result.add_argument("--max-input-tokens", type=int, default=512)
    result.add_argument("--candidate-floor", type=float, default=0.01)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--selection-window-limit", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260812)
    result.add_argument("--device", default="cuda")
    result.add_argument("--resume", choices=("auto", "none"), default="auto")
    return result


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.positive_weight) or args.positive_weight <= 0:
        raise ValueError("positive weight must be finite and positive")
    if min(args.epochs, args.batch_size, args.eval_batch_size) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if min(args.train_limit, args.selection_window_limit) < 0:
        raise ValueError("development limits must be nonnegative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
