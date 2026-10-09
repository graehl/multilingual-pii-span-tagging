#!/usr/bin/env python
"""Run a tiny, disjoint GLiNER2 C-weighted entity-calibration smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pii_projector import Tagset  # noqa: E402

SMOKE_VERSION = "gliner2-c-weighted-smoke-v1"
LORA_TARGETS = ("encoder", "span_rep", "count_embed")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def row_identity(row: dict[str, Any]) -> str:
    provenance = row.get("provenance", {})
    return ":".join(
        (
            str(provenance.get("source_path", "")),
            str(provenance.get("source_line", "")),
            str(provenance.get("source_id", row.get("id", ""))),
        )
    )


def row_language(row: dict[str, Any]) -> str:
    return str(row.get("provenance", {}).get("lang") or row.get("lang") or "unknown")


def select_disjoint_rows(
    rows: Sequence[dict[str, Any]],
    *,
    train_docs: int,
    score_docs: int,
    seed: int,
    languages: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        row
        for row in rows
        if row.get("provenance", {}).get("projected_spans")
        and (languages is None or row_language(row) in languages)
    ]
    eligible.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row_identity(row)}".encode()).digest())
    required = train_docs + score_docs
    if len(eligible) < required:
        raise ValueError(f"need {required} positive projected rows, found {len(eligible)}")
    train = eligible[:train_docs]
    score = eligible[train_docs:required]
    overlap = {row_identity(row) for row in train} & {row_identity(row) for row in score}
    if overlap:
        raise AssertionError(f"train/score membership overlaps: {sorted(overlap)!r}")
    return train, score


def deterministic_loss_mask(
    labels: torch.Tensor,
    valid: torch.Tensor,
    *,
    negative_keep: float,
    seed: int,
) -> torch.Tensor:
    if not 0 < negative_keep <= 1:
        raise ValueError("negative_keep must be in (0, 1]")
    indices = torch.arange(labels.numel(), device=labels.device, dtype=torch.int64).reshape(labels.shape)
    mixed = (indices * 1_103_515_245 + seed * 12_345 + 12_345) & 0x7FFF_FFFF
    cutoff = round(negative_keep * 2_147_483_648)
    keep_negative = mixed < cutoff
    return valid.bool() & ((labels > 0.5) | keep_negative)


def normalized_c_weighted_bce(
    scores: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive")
    kept = loss_mask.to(dtype=scores.dtype)
    kept_count = kept.sum()
    if kept_count.item() == 0:
        raise ValueError("loss mask keeps no elements")
    weights = torch.where(labels > 0.5, positive_weight, 1.0).to(dtype=scores.dtype)
    weight_mass = (weights * kept).sum()
    element_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    return (element_loss * weights * kept).sum() * kept_count / weight_mass


def c_weighted_struct_loss(
    self: Any,
    span_rep: torch.Tensor,
    schema_emb: torch.Tensor,
    structure: list[Any],
    span_mask: torch.Tensor,
    masking_rate: float = 0.5,
) -> torch.Tensor:
    del masking_rate
    gold_count = min(structure[0], 19)
    struct_proj = self.count_embed(schema_emb[1:], gold_count)
    scores = torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)
    labels = torch.zeros_like(scores)
    for instance_index in range(gold_count):
        for field_index, span_or_spans in enumerate(structure[1][instance_index]):
            if span_or_spans is None or span_or_spans == (-1, -1):
                continue
            spans = span_or_spans if isinstance(span_or_spans, list) else [span_or_spans]
            for start, end in spans:
                width = end - start
                if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                    labels[instance_index, field_index, start, width] = 1

    valid_spans = (~span_mask[0]).reshape(1, 1, scores.shape[2], scores.shape[3])
    valid = valid_spans.expand_as(scores)
    exclusion = getattr(self, "_pii_active_span_exclusion", None)
    if exclusion is not None:
        exclusion = exclusion.to(device=scores.device)
        if tuple(exclusion.shape) != tuple(scores.shape[2:]):
            raise ValueError(f"span exclusion shape {tuple(exclusion.shape)} != {tuple(scores.shape[2:])}")
        valid = valid & ~exclusion.reshape(1, 1, *exclusion.shape)
    loss_mask = deterministic_loss_mask(
        labels,
        valid,
        negative_keep=self._pii_negative_keep,
        seed=(
            self._pii_mask_seed
            + getattr(self, "_pii_batch_mask_step", 0) * 1_000_003
            + getattr(self, "_pii_active_sample_index", 0) * 10_007
        ),
    )
    return normalized_c_weighted_bce(
        scores,
        labels,
        loss_mask,
        positive_weight=self._pii_positive_weight,
    )


def install_c_weighted_loss(
    model: Any, *, positive_weight: float, negative_keep: float, mask_seed: int
) -> None:
    model._pii_positive_weight = positive_weight
    model._pii_negative_keep = negative_keep
    model._pii_mask_seed = mask_seed
    model.compute_struct_loss = MethodType(c_weighted_struct_loss, model)

    original_sample_loss = model._compute_sample_loss

    def masked_sample_loss(self: Any, *args: Any, **kwargs: Any) -> Any:
        masks = getattr(self, "_pii_batch_span_exclusions", None)
        cursor = getattr(self, "_pii_batch_span_exclusion_cursor", 0)
        self._pii_active_span_exclusion = None if masks is None else masks[cursor]
        self._pii_active_sample_index = cursor
        self._pii_batch_span_exclusion_cursor = cursor + 1
        return original_sample_loss(*args, **kwargs)

    model._compute_sample_loss = MethodType(masked_sample_loss, model)


def install_batch_span_exclusions(model: Any, masks: Sequence[torch.Tensor], *, step: int = 0) -> None:
    model._pii_batch_span_exclusions = list(masks)
    model._pii_batch_span_exclusion_cursor = 0
    model._pii_batch_mask_step = step


def parse_entity_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    entities = result.get("entities", result)
    predictions: dict[tuple[int, int, str], dict[str, Any]] = {}
    for label, items in entities.items():
        for item in items or []:
            if not isinstance(item, dict) or "start" not in item or "end" not in item:
                continue
            prediction = {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "label": str(label),
                "confidence": float(item.get("confidence", 1.0)),
            }
            key = (prediction["start"], prediction["end"], prediction["label"])
            prior = predictions.get(key)
            if prior is None or prediction["confidence"] > prior["confidence"]:
                predictions[key] = prediction
    return sorted(predictions.values(), key=lambda item: (item["start"], item["end"], item["label"]))


def extract_candidates(
    model: Any,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    floor: float,
) -> list[list[dict[str, Any]]]:
    model.eval()
    model.processor.change_mode(False)
    candidates = []
    for row in rows:
        result = model.extract_entities(
            row["input"],
            list(labels),
            threshold=floor,
            include_confidence=True,
            include_spans=True,
        )
        candidates.append(parse_entity_predictions(result))
    return candidates


def symmetric_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    intersection = min(left["end"], right["end"]) - max(left["start"], right["start"])
    if intersection <= 0:
        return False
    left_length = left["end"] - left["start"]
    right_length = right["end"] - right["start"]
    return intersection * 5 >= left_length * 4 and intersection * 5 >= right_length * 4


def maximum_one_to_one_matches(
    gold: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> int:
    adjacency = [
        [index for index, prediction in enumerate(predictions) if predicate(item, prediction)]
        for item in gold
    ]
    matched_gold = [-1] * len(predictions)

    def augment(gold_index: int, seen: set[int]) -> bool:
        for prediction_index in adjacency[gold_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            prior = matched_gold[prediction_index]
            if prior == -1 or augment(prior, seen):
                matched_gold[prediction_index] = gold_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(gold)))


def prf(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "predicted": predicted, "gold": gold, "P": precision, "R": recall, "F1": f1}


def score_candidates(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[Sequence[dict[str, Any]]],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    totals = {"exact_typed": [0, 0, 0], "overlap_typed": [0, 0, 0], "overlap_p1": [0, 0, 0]}
    for row, row_candidates in zip(rows, candidates):
        gold = [
            {"start": int(start), "end": int(end), "label": str(label)}
            for start, end, label in row["provenance"]["projected_spans"]
        ]
        predictions = [item for item in row_candidates if item["confidence"] >= threshold]
        predicates = {
            "exact_typed": lambda left, right: (
                (left["start"], left["end"], left["label"]) == (right["start"], right["end"], right["label"])
            ),
            "overlap_typed": lambda left, right: (
                left["label"] == right["label"] and symmetric_overlap(left, right)
            ),
            "overlap_p1": symmetric_overlap,
        }
        for name, predicate in predicates.items():
            totals[name][0] += maximum_one_to_one_matches(gold, predictions, predicate)
            totals[name][1] += len(predictions)
            totals[name][2] += len(gold)
    return {name: prf(*counts) for name, counts in totals.items()}


def emitted_movement(
    before: Sequence[Sequence[dict[str, Any]]],
    after: Sequence[Sequence[dict[str, Any]]],
    threshold: float,
) -> dict[str, int]:
    added = removed = relabelled = changed_docs = 0
    for before_doc, after_doc in zip(before, after):
        before_set = {
            (item["start"], item["end"], item["label"])
            for item in before_doc
            if item["confidence"] >= threshold
        }
        after_set = {
            (item["start"], item["end"], item["label"])
            for item in after_doc
            if item["confidence"] >= threshold
        }
        if before_set != after_set:
            changed_docs += 1
        added += len(after_set - before_set)
        removed += len(before_set - after_set)
        before_geometry = {(start, end): label for start, end, label in before_set}
        after_geometry = {(start, end): label for start, end, label in after_set}
        relabelled += sum(
            before_geometry[key] != after_geometry[key]
            for key in before_geometry.keys() & after_geometry.keys()
        )
    return {"added": added, "removed": removed, "relabelled": relabelled, "changed_docs": changed_docs}


def parameter_receipt(model: Any) -> dict[str, Any]:
    groups: Counter[str] = Counter()
    tensors = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".encoder." in name:
            group = "encoder"
        elif ".span_rep." in name:
            group = "span_rep"
        elif ".count_embed." in name:
            group = "count_embed"
        else:
            group = "other"
        groups[group] += parameter.numel()
        tensors.append({"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()})
    return {"total": sum(groups.values()), "by_group": dict(sorted(groups.items())), "tensors": tensors}


def training_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"input": row["input"], "output": row["output"]} for row in rows]


def train_lora(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    encoder_lr: float,
    task_lr: float,
    seed: int,
) -> list[dict[str, float | int]]:
    records = training_records(rows)
    generator = random.Random(seed)
    order = list(range(len(records)))
    encoder_parameters = []
    task_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder_parameters if ".encoder." in name else task_parameters).append(parameter)
    if not encoder_parameters or not task_parameters:
        raise ValueError("expected trainable encoder and task LoRA parameter groups")
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": task_parameters, "lr": task_lr},
        ],
        weight_decay=0.0,
    )
    history = []
    cursor = 0
    model.train()
    model.processor.change_mode(True)
    for step in range(steps):
        if cursor == 0:
            generator.shuffle(order)
        indices = [order[(cursor + offset) % len(order)] for offset in range(batch_size)]
        cursor = (cursor + batch_size) % len(order)
        batch_rows = [(records[index]["input"], records[index]["output"]) for index in indices]
        batch = model.processor.collate_fn_train(batch_rows)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        if output["batch_size"] != len(batch_rows):
            raise RuntimeError(f"GLiNER2 accepted {output['batch_size']}/{len(batch_rows)} training rows")
        loss = output["total_loss"] / len(batch_rows)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm) or grad_norm.item() == 0:
            raise RuntimeError(f"invalid gradient norm at step {step + 1}: {grad_norm.item()}")
        optimizer.step()
        history.append({"step": step + 1, "loss": loss.item(), "grad_norm": grad_norm.item()})
    return history


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from gliner2 import GLiNER2

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = read_jsonl(args.input)
    languages = set(args.languages.split(",")) if args.languages else None
    train_rows, score_rows = select_disjoint_rows(
        rows,
        train_docs=args.train_docs,
        score_docs=args.score_docs,
        seed=args.seed,
        languages=languages,
    )
    labels = sorted(Tagset().cut_targets(args.output_schema))
    device = torch.device(args.device)
    started = time.time()
    base = GLiNER2.from_pretrained(str(args.model), map_location=str(device))
    baseline = extract_candidates(base, score_rows, labels, floor=args.candidate_floor)
    install_c_weighted_loss(
        base,
        positive_weight=args.positive_weight,
        negative_keep=args.negative_keep,
        mask_seed=args.mask_seed,
    )
    model = base.apply_lora(
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        targets=list(LORA_TARGETS),
    )
    receipt = parameter_receipt(model)
    history = train_lora(
        model,
        train_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        seed=args.seed,
    )
    adapted = extract_candidates(model, score_rows, labels, floor=args.candidate_floor)
    thresholds = [round(index / 100, 2) for index in range(1, 100)]
    curve = []
    for threshold in thresholds:
        before_metrics = score_candidates(score_rows, baseline, threshold)
        after_metrics = score_candidates(score_rows, adapted, threshold)
        movement = emitted_movement(baseline, adapted, threshold)
        curve.append(
            {
                "threshold": threshold,
                "before": before_metrics,
                "after": after_metrics,
                "movement": movement,
                "metric_changed": before_metrics != after_metrics,
            }
        )
    passing_points = [
        point for point in curve if point["metric_changed"] and point["movement"]["changed_docs"] > 0
    ]
    adapter_dir = args.output.parent / f"{args.output.stem}-adapter"
    model.save_pretrained(str(adapter_dir))
    report = {
        "version": SMOKE_VERSION,
        "smoke_pass": bool(passing_points),
        "input": {"path": str(args.input), "sha256": file_sha256(args.input)},
        "model": {"path": str(args.model)},
        "membership": {
            "train": [row_identity(row) for row in train_rows],
            "score": [row_identity(row) for row in score_rows],
            "train_languages": dict(sorted(Counter(row_language(row) for row in train_rows).items())),
            "score_languages": dict(sorted(Counter(row_language(row) for row in score_rows).items())),
        },
        "configuration": {
            "output_schema": args.output_schema,
            "labels": labels,
            "positive_weight": args.positive_weight,
            "negative_keep": args.negative_keep,
            "mask_seed": args.mask_seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr,
            "task_lr": args.task_lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": list(LORA_TARGETS),
            "candidate_floor": args.candidate_floor,
            "seed": args.seed,
            "device": str(device),
        },
        "parameters": receipt,
        "training": history,
        "curve": curve,
        "first_passing_point": passing_points[0] if passing_points else None,
        "adapter": {"path": str(adapter_dir)},
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(args.output, report)
    print(
        f"GLINER2_CALIBRATION_SMOKE: pass={report['smoke_pass']} "
        f"train={len(train_rows)} score={len(score_rows)} steps={args.steps} "
        f"trainable={receipt['total']} -> {args.output}"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True, help="accepted projected GLiNER2 JSONL")
    result.add_argument("--model", type=Path, required=True, help="revision-pinned GLiNER2 snapshot")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--output-schema", default="redaction_9_v1")
    result.add_argument("--languages", help="optional comma-separated smoke-language restriction")
    result.add_argument("--train-docs", type=int, default=8)
    result.add_argument("--score-docs", type=int, default=8)
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--positive-weight", type=float, default=3.0)
    result.add_argument("--negative-keep", type=float, default=0.5)
    result.add_argument("--mask-seed", type=int, default=20260812)
    result.add_argument("--encoder-lr", type=float, default=1e-5)
    result.add_argument("--task-lr", type=float, default=5e-4)
    result.add_argument("--lora-rank", type=int, default=8)
    result.add_argument("--lora-alpha", type=float, default=16.0)
    result.add_argument("--candidate-floor", type=float, default=0.01)
    result.add_argument("--seed", type=int, default=20260812)
    result.add_argument("--device", default="cuda")
    return result


def main() -> int:
    args = parser().parse_args()
    if min(args.train_docs, args.score_docs, args.steps, args.batch_size) <= 0:
        raise ValueError("train docs, score docs, steps, and batch size must be positive")
    report = run(args)
    return 0 if report["smoke_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
