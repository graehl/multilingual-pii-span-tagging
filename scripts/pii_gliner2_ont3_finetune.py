#!/usr/bin/env python3
# acli: 1 complete
"""Fine-tune GLiNER2 on Ont3 with audited word-span supervision.

Optional label transfer uses one-to-one model-facing names and initializes
new label tokens from preferred/allowed donor embeddings. Donor associations
do not implement many:many supervision or establish a fair architecture test.

The optional word-boundary mode snaps training spans outward using the
installed processor's splitter, including its appended punctuation, then
checks case-insensitive token-sequence occurrence identity. Changed extents
are recorded; evaluation character annotations remain unchanged. The library's
training format names entity surfaces and its processor marks every occurrence
of each, so a surface that is PII in one place and not another cannot be
expressed. Rather than accept that noise, every window is checked: expand its
labelled surfaces back to all their occurrences and keep the window only if
that reproduces the window's gold spans exactly. A window that fails is dropped
and counted, so the supervision that survives is exactly right under the
library's own semantics.

Windowing comes from `pii_gliner2_exact.py`, whose straddle handling and
provenance are label-set agnostic; only its projection into the nine-label
redaction cut was fixed, and that is now injectable, so ont3 spans pass through
an identity projector unchanged.

Two phases, either runnable alone. `materialize` writes training examples and a
rejection ledger; `train` fits the model on them. Materializing separately
matters because the rejection counts say what the representation cannot hold
before any GPU time is spent.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path.home() / "agents"))
import acli  # noqa: E402

SCHEMA = "pii-gliner2-ont3-finetune/v1"


def json_result(result: dict[str, Any] | None) -> dict[str, int | float | str]:
    """Keep trainer scalars while spelling non-finite floats as strings."""
    out: dict[str, int | float | str] = {}
    for key, value in (result or {}).items():
        if not isinstance(value, (int, float, str)):
            continue
        out[key] = str(value) if isinstance(value, float) and not math.isfinite(value) else value
    return out


def sample_current_mix(
    rows: list[dict[str, Any]],
    *,
    sampling_config: Path,
    language_weights_path: Path | None,
    draws: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw a fixed GLiNER epoch roster under the encoder's current mix."""
    from pii_annotation_sampling import share_annotation_sampling_mass

    spec = json.loads(sampling_config.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("sampling config schema_version must be 1")
    pool_matches = [(pool["name"], pool["match"]) for pool in spec["pools"]]
    pool_weights = {pool["name"]: float(pool["weight"]) for pool in spec["pools"]}
    pool_keys = []
    for index, row in enumerate(rows):
        matches = [
            name
            for name, required in pool_matches
            if all(row.get(field) == value for field, value in required.items())
        ]
        if len(matches) != 1:
            raise ValueError(f"training row {index} matches {len(matches)} sampling pools")
        pool_keys.append(matches[0])
    factor_field = spec.get("example_factor_field")
    factors = [1.0] * len(rows) if factor_field is None else [float(row[factor_field]) for row in rows]
    factor_totals: dict[str, float] = {}
    for pool, factor in zip(pool_keys, factors, strict=True):
        factor_totals[pool] = factor_totals.get(pool, 0.0) + factor
    weights = [
        pool_weights[pool] * factor / factor_totals[pool]
        for pool, factor in zip(pool_keys, factors, strict=True)
    ]
    weights, sharing = share_annotation_sampling_mass(rows, weights)
    if weights is None:
        weights = [1.0] * len(rows)
    language_weights = (
        json.loads(language_weights_path.read_text(encoding="utf-8"))
        if language_weights_path is not None
        else {}
    )
    weights = [
        weight * float(language_weights.get(str(row.get("lang") or "<unknown>"), 1.0))
        for row, weight in zip(rows, weights, strict=True)
    ]
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("current-mix sampling has no finite positive mass")
    weights = [weight / total for weight in weights]
    indices = random.Random(seed).choices(range(len(rows)), weights=weights, k=draws)
    sampled = [rows[index] for index in indices]
    return sampled, {
        "sampling_config": str(sampling_config),
        "language_weights": str(language_weights_path) if language_weights_path else None,
        "draws": draws,
        "seed": seed,
        "annotation_sharing": sharing,
        "expected_pool_mass": {
            pool: round(
                math.fsum(weight for key, weight in zip(pool_keys, weights, strict=True) if key == pool),
                8,
            )
            for pool in sorted(set(pool_keys))
        },
        "realized_pools": dict(sorted(Counter(row["sampling_pool"] for row in sampled).items())),
        "realized_languages": dict(
            sorted(Counter(str(row.get("lang") or "<unknown>") for row in sampled).items())
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_pooled_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    """Read named pool files, filling the encoder's file-owned pool label."""
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            row.setdefault("sampling_pool", path.stem)
            rows.append(row)
    return rows


def ont3_types(labels_path: Path) -> list[str]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    bioes = {label.partition("-")[2] for label in labels if label[:2] in ("B-", "I-", "E-", "S-")}
    return sorted(bioes) if bioes else sorted({label for label in labels if label != "O"})


def identity_spans(row: dict[str, Any]) -> list[tuple[int, int, str]]:
    """The row's own ont3 spans, already in the target inventory."""
    out = []
    for span in row.get("spans") or []:
        start, end, label = (
            (span[0], span[1], span[2])
            if isinstance(span, list)
            else (
                span["start"],
                span["end"],
                span.get("label") or span.get("type"),
            )
        )
        out.append((int(start), int(end), str(label)))
    return sorted(out)


def occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    """Every non-overlapping occurrence of one surface."""
    found = []
    start = text.find(surface)
    while start >= 0:
        found.append((start, start + len(surface)))
        start = text.find(surface, start + len(surface))
    return found


def surface_example(
    window: dict[str, Any],
    labels: list[str],
    negatives: int,
    seed: int,
    *,
    known_negative_labels: bool = True,
    word_splitter: Any = None,
) -> dict[str, Any] | None:
    """One training example, or None when surfaces cannot express its gold.

    The library reads a named surface as marking every occurrence. Expanding
    the window's labelled surfaces that way has to reproduce its gold spans
    exactly, otherwise training would be taught a span the annotator did not
    mark, or taught to ignore a repeat the annotator did.

    Only the labels present in the window are supervised, plus a deterministic
    sample of absent ones. Handing every example the whole 31-label inventory
    with almost all lists empty is what a first attempt did, and it collapsed:
    the loss went to zero and the trained model predicted nothing at all,
    because the overwhelming signal was absence.
    """
    text = window["input"]
    projected = (window.get("provenance") or {}).get("projected_spans") or []
    gold = {(int(a), int(b), str(c)) for a, b, c in projected}
    present: dict[str, list[str]] = {}
    for start, end, label in sorted(gold):
        if label not in labels:
            return None
        surface = text[start:end]
        present.setdefault(label, [])
        if surface not in present[label]:
            present[label].append(surface)
    if word_splitter is None:
        expanded = {
            (start, end, label)
            for label, surfaces in present.items()
            for surface in surfaces
            for start, end in occurrences(text, surface)
        }
    else:
        tokens = list(word_splitter(text))
        words = [token[0] for token in tokens]
        expanded = set()
        for label, surfaces in present.items():
            for surface in surfaces:
                needle = [token[0] for token in word_splitter(surface)]
                if not needle:
                    return None
                for index in range(len(words) - len(needle) + 1):
                    if words[index : index + len(needle)] == needle:
                        expanded.add((tokens[index][1], tokens[index + len(needle) - 1][2], label))
    if expanded != gold:
        return None
    absent = [label for label in labels if label not in present] if known_negative_labels else []
    chosen = random.Random(f"{seed}:{text}").sample(absent, min(negatives, len(absent)))
    entities = dict(present)
    for label in chosen:
        entities[label] = []
    return {"text": text, "entities": entities}


def snap_window(
    window: dict[str, Any], splitter: Any, max_width: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Snap to the actual processor's outward word bounds, including its period."""
    text = window["input"]
    if not text.endswith((".", "!", "?")):
        text += "."
    tokens = list(splitter(text))
    snapped = []
    changes = []
    for start, end, label in window["provenance"]["projected_spans"]:
        indices = [i for i, (_, a, b) in enumerate(tokens) if a < end and start < b]
        if not indices or len(indices) > max_width:
            raise ValueError("span has no representable word interval within the configured width")
        a, b = tokens[indices[0]][1], tokens[indices[-1]][2]
        snapped.append([a, b, label])
        if (a, b) != (start, end):
            changes.append({"original": [start, end, label], "snapped": [a, b, label], "surface": text[a:b]})
    result = {**window, "input": text, "provenance": {**window["provenance"], "projected_spans": snapped}}
    return result, changes


def materialize(args: Any) -> dict[str, Any]:
    from pii_gliner2_exact import window_projected_row

    labels = ont3_types(args.labels)
    rows = read_pooled_jsonl(args.input)
    projected_labels: Counter[str] = Counter()
    if args.label_map:
        map_spec = json.loads(args.label_map.read_text())
        labels = sorted(map_spec["ontology"]["primary_types"])
        mapping = map_spec["v1_to_v2"]
        for row in rows:
            spans = []
            mapped_targets = []
            for start, end, label in identity_spans(row):
                target = mapping[label]["fallback"]
                if args.acceptable_labels:
                    accepted = mapping[label]["accepted"]
                    if not accepted or not set(accepted) <= set(labels):
                        raise ValueError("acceptable entity targets must be nonempty Ont3 type sets")
                    mapped_targets.append([start, end, accepted])
                if target != "O":
                    if target not in labels:
                        raise ValueError(f"unrecognized mapped target: {label} -> {target}")
                    spans.append([start, end, target])
                if target != label:
                    projected_labels[f"{label}->{target}"] += 1
            row["spans"] = spans
            if args.acceptable_labels:
                row["_mapped_targets"] = mapped_targets
    if args.acceptable_labels and (not args.label_map or not args.snap_word_boundaries):
        raise ValueError("--acceptable-labels requires --label-map and --snap-word-boundaries")
    if args.limit:
        rows = rows[: args.limit]
    source_rows = len(rows)
    sampling = None
    splitter = None
    if args.snap_word_boundaries:
        from gliner2.processing.word_splitter import WhitespaceTokenSplitter

        splitter = WhitespaceTokenSplitter()
    if args.sample_records:
        if args.sampling_config is None:
            raise ValueError("--sample-records requires --sampling-config")
        rows, sampling = sample_current_mix(
            rows,
            sampling_config=args.sampling_config,
            language_weights_path=args.language_weights,
            draws=args.sample_records,
            seed=args.seed,
        )

    accepted = 0
    reasons: dict[str, int] = {}
    supervision_counts: dict[str, int] = {}
    accepted_languages: Counter[str] = Counter()
    language_examples: dict[str, list[dict[str, Any]]] = {}
    alignment = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            if not isinstance(row.get("text"), str):
                continue
            supervision = str(row.get("supervision") or "complete")
            supervision_counts[supervision] = supervision_counts.get(supervision, 0) + 1
            try:
                windows = window_projected_row(
                    row,
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                    drop_straddling_windows=True,
                    span_projector=identity_spans,
                )
            except (ValueError, KeyError, IndexError) as error:
                reasons[type(error).__name__] = reasons.get(type(error).__name__, 0) + 1
                continue
            for window in windows:
                if splitter is not None:
                    try:
                        window, changes = snap_window(window, splitter, args.max_span_width)
                    except ValueError:
                        reasons["word_span_unrepresentable"] = reasons.get("word_span_unrepresentable", 0) + 1
                        continue
                    alignment.extend(
                        {"source_id": row.get("id"), "sample_index": index, **change} for change in changes
                    )
                example = surface_example(
                    window,
                    labels,
                    args.negative_labels,
                    args.seed,
                    known_negative_labels=supervision == "complete" and not row.get("unknown_primary_types"),
                    word_splitter=splitter,
                )
                if example is None:
                    reasons["surface_ambiguous"] = reasons.get("surface_ambiguous", 0) + 1
                    continue
                if not example["entities"]:
                    reasons["empty_tasks"] = reasons.get("empty_tasks", 0) + 1
                    continue
                if args.acceptable_labels:
                    from pii_gliner2_mapped_loss import window_targets

                    example["acceptable_targets"] = window_targets(row, window, splitter)
                sink.write(json.dumps(example, ensure_ascii=False) + "\n")
                accepted += 1
                accepted_languages[str(row.get("lang") or "<unknown>")] += 1
                if args.minimum_language_share:
                    language_examples.setdefault(str(row.get("lang") or "<unknown>"), []).append(example)
            if (index + 1) % 2000 == 0:
                print(f"gliner2-materialize {index + 1}/{len(rows)}", flush=True)
        coverage_repeats: Counter[str] = Counter()
        floor = args.minimum_language_share
        if not 0 <= floor < 1 / max(1, len(accepted_languages)):
            raise ValueError("minimum language share must leave room for the full language inventory")
        rng = random.Random(args.seed)
        while floor:
            deficient = [
                lang for lang, count in sorted(accepted_languages.items()) if count / accepted < floor
            ]
            if not deficient:
                break
            for lang in deficient:
                count = math.ceil((floor * accepted - accepted_languages[lang]) / (1 - floor))
                for example in rng.choices(language_examples[lang], k=count):
                    sink.write(json.dumps(example, ensure_ascii=False) + "\n")
                accepted += count
                accepted_languages[lang] += count
                coverage_repeats[lang] += count
    if splitter is not None:
        args.output.with_suffix(".alignment.json").write_text(
            json.dumps(alignment, ensure_ascii=False) + "\n"
        )
    language_receipt = None
    if args.language_round:
        from pii_language_policy import validate_core_language_support

        language_receipt = validate_core_language_support(
            accepted_languages, language_round=args.language_round
        )
    return {
        "schema": SCHEMA,
        "phase": "materialize",
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "source_rows": source_rows,
        "sampled_rows": len(rows),
        "sampling": sampling,
        "windows_accepted": accepted,
        "windows_rejected": sum(reasons.values()),
        "rejection_reasons": dict(sorted(reasons.items())),
        "source_supervision": dict(sorted(supervision_counts.items())),
        "accepted_languages": dict(sorted(accepted_languages.items())),
        "coverage_repeats": dict(coverage_repeats),
        "minimum_language_share": args.minimum_language_share,
        "partial_negative_labels": 0,
        "labels": labels,
        "label_map": str(args.label_map) if args.label_map else None,
        "acceptable_labels": args.acceptable_labels,
        "projected_labels": dict(projected_labels),
        "language_support": language_receipt,
        "word_boundary_snapping": splitter is not None,
        "changed_span_extents": len(alignment),
        "max_span_width": args.max_span_width if splitter is not None else None,
    }


def verify_processor_supervision(
    processor: Any, records: list[dict[str, Any]], max_width: int
) -> dict[str, int]:
    """Exercise the installed processor and reject missing or truncated targets."""
    processor.change_mode(False)
    positive = 0
    for index, row in enumerate(records):
        text = row["text"]
        if not text.endswith((".", "!", "?")):
            text += "."
        transformed = processor.transform_and_format(text, {"entities": row["entities"]})
        if transformed.task_types != ["entities"]:
            raise ValueError(f"record {index}: unexpected processor task shape")
        fields = transformed.structure_labels[0][1][0]
        if len(fields) != len(row["entities"]):
            raise ValueError(f"record {index}: processor changed label count")
        for (label, surfaces), positions in zip(row["entities"].items(), fields, strict=True):
            if not surfaces:
                continue
            for start, end in positions:
                if start < 0 or end < start or end - start + 1 > max_width:
                    raise ValueError(f"record {index}, {label}: unreachable processor target {(start, end)}")
                positive += 1
    return {"records": len(records), "positive_word_spans": positive}


def train(args: Any) -> dict[str, Any]:
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    from trainlib import CheckpointMirrorCallback

    checkpoint_mirror = CheckpointMirrorCallback(args)

    class MirroredTrainer(GLiNER2Trainer):
        def _save_checkpoint(self, name):
            super()._save_checkpoint(name)
            if self.is_main_process and checkpoint_mirror.mirror:
                checkpoint_mirror.mirror.save(self.output_dir, force=name == "final")

    model = GLiNER2.from_pretrained(args.model, map_location=args.device)
    from pii_gliner2_label_transfer import (
        CONFIG_KEY,
        initialize_transfer,
        load_transfer,
        prompt_mapping,
        translate_records,
    )

    transfer_receipt = None
    if args.label_transfer:
        transfer_receipt = initialize_transfer(model, load_transfer(args.label_transfer))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "label-transfer-initialization.json").write_text(
            json.dumps(transfer_receipt, indent=2) + "\n"
        )
        # Retain the identical prompt/token initialization for step-zero scoring.
        model.save_pretrained(str(args.output_dir / "initial"))
    records = read_jsonl(args.train)
    evaluation = read_jsonl(args.eval) if args.eval else None
    inventory = sorted({label for row in records for label in row["entities"]})
    # A transferred checkpoint names its whole inventory; small data may exercise only part of it.
    transfer = getattr(model.config, CONFIG_KEY, None)
    mapping = prompt_mapping(model, sorted(transfer["labels"]) if transfer else inventory)
    if unknown := set(inventory) - set(mapping):
        raise ValueError(f"training labels outside the checkpoint's transfer inventory: {sorted(unknown)}")
    if args.acceptable_labels:
        from pii_gliner2_mapped_training import install_mapped_training, training_records

        install_mapped_training(model)
        records = training_records(records, mapping)
        if evaluation:
            evaluation = training_records(evaluation, mapping)
    else:
        records = translate_records(records, mapping)
        if evaluation:
            evaluation = translate_records(evaluation, mapping)
    if args.shuffle_labels:
        if args.acceptable_labels:
            raise ValueError("--shuffle-labels is incompatible with --acceptable-labels target order")
        # Records list present types before absent ones; shuffle each example's prompt order.
        model.processor.sampling_config.shuffle_entities = True
    supervision_check = None
    if args.verify_supervision or args.acceptable_labels:
        if args.acceptable_labels:
            for row in records + (evaluation or []):
                transformed = model.processor.collate_fn_train([(row["input"], row["output"])])
                for group in transformed.structure_labels[0][0][2]["groups"]:
                    if group["end"] - group["start"] + 1 > args.max_span_width:
                        raise ValueError("mapped target exceeds model span width")
            supervision_check = {"records": len(records), "acceptable_labels": True}
        else:
            supervision_check = verify_processor_supervision(model.processor, records, args.max_span_width)
            if evaluation:
                verify_processor_supervision(model.processor, evaluation, args.max_span_width)
        print("gliner2-supervision " + json.dumps(supervision_check), flush=True)
    config = TrainingConfig(
        output_dir=str(args.output_dir),
        experiment_name=args.experiment_name,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        scheduler_type="cosine",
        early_stopping=True,
        early_stopping_patience=args.patience,
        # The stock sanitizer reserializes InputExample and drops target
        # metadata. Mapped records have mandatory processor validation above.
        validate_data=not args.acceptable_labels,
        seed=args.seed,
        bf16=True,
        fp16=False,
    )
    started = time.perf_counter()
    trainer = MirroredTrainer(
        model=model.model if hasattr(model, "model") else model,
        config=config,
        processor=model.processor,
        train_data=records,
        eval_data=evaluation,
    )
    result = trainer.train()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir / "final"))
    checkpoint_mirror.finish(args.output_dir, is_world_process_zero=trainer.is_main_process)
    return {
        "schema": SCHEMA,
        "phase": "train",
        "train_records": len(records),
        "eval_records": len(evaluation) if evaluation else 0,
        "output_dir": str(args.output_dir),
        "elapsed_s": round(elapsed, 1),
        "result": json_result(result),
        "supervision_check": supervision_check,
        "label_transfer": transfer_receipt,
        "shuffle_labels": args.shuffle_labels,
    }


def main() -> None:
    parser = acli.argument_parser(description=__doc__, exit_codes={0: "done", 2: "invalid input"})
    sub = parser.add_subparsers(dest="phase", required=True)

    mat = sub.add_parser("materialize", help="write exact-offset GLiNER2 records")
    mat.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="ont3 rows with text and spans; repeat to concatenate admitted pools",
    )
    mat.add_argument("--output", type=Path, required=True, help="GLiNER2 records JSONL")
    mat.add_argument("--labels", type=Path, required=True, help="ont3 labels.json")
    mat.add_argument("--label-map", type=Path, help="use the declared single-label fallback for mapped gold")
    mat.add_argument(
        "--acceptable-labels",
        action="store_true",
        help="retain full accepted sets and coverage for mapped span loss",
    )
    mat.add_argument(
        "--language-round", type=Path, help="validate the post-admission uniform training roster"
    )
    mat.add_argument("--max-chars", type=int, default=900)
    mat.add_argument("--overlap-chars", type=int, default=160)
    mat.add_argument(
        "--minimum-language-share",
        type=float,
        default=0,
        help="restore this language share after word-span admission by explicitly counted resampling",
    )
    mat.add_argument(
        "--snap-word-boundaries",
        action="store_true",
        help="snap training spans outward using the installed GLiNER2 word splitter; audit changed extents",
    )
    mat.add_argument(
        "--max-span-width", type=int, default=8, help="model word-span width; reject wider training windows"
    )
    mat.add_argument(
        "--negative-labels",
        type=int,
        default=8,
        help="absent labels sampled per example as negatives; 0 supervises only what is present",
    )
    mat.add_argument("--seed", type=int, default=173, help="seed for the negative-label sample")
    mat.add_argument("--limit", type=int, default=0)
    mat.add_argument("--sampling-config", type=Path, help="current encoder pool-weight config")
    mat.add_argument("--language-weights", type=Path, help="current per-language sampling multipliers")
    mat.add_argument(
        "--sample-records",
        type=int,
        default=0,
        help="draw this many source rows with replacement before GLiNER windowing",
    )

    tr = sub.add_parser("train", help="fit GLiNER2 on materialized records")
    tr.add_argument("--train", type=Path, required=True)
    tr.add_argument("--eval", type=Path)
    tr.add_argument("--model", required=True)
    tr.add_argument("--label-transfer", type=Path, help="one-to-one Ont3 prompt names and embedding donors")
    tr.add_argument("--output-dir", type=Path, required=True)
    tr.add_argument("--experiment-name", default="gliner2-ont3")
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--max-steps", type=int, default=-1)
    tr.add_argument("--patience", type=int, default=6)
    tr.add_argument("--seed", type=int, default=173)
    tr.add_argument("--verify-supervision", action="store_true")
    tr.add_argument(
        "--acceptable-labels",
        action="store_true",
        help="train with accepted-label union loss and coverage masking",
    )
    tr.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="shuffle each training example's entity-type order (GLiNER's regularization); off as GL4 ran",
    )
    tr.add_argument("--max-span-width", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--encoder-lr", type=float, default=1e-5)
    tr.add_argument("--task-lr", type=float, default=5e-4)
    tr.add_argument("--eval-steps", type=int, default=500)
    tr.add_argument("--save-total-limit", type=int, default=3, help="number of periodic checkpoints retained")
    tr.add_argument("--warmup-steps", type=int, default=0, help="explicit warmup; 0 uses upstream ratio")
    tr.add_argument("--device", default="cuda")
    from trainlib import add_checkpoint_mirror_args

    add_checkpoint_mirror_args(tr)

    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        receipt = materialize(args) if args.phase == "materialize" else train(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        acli.die(str(error), 2)
    acli.emit(receipt, acli.resolve_format(args))


if __name__ == "__main__":
    main()
