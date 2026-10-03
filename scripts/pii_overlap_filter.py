#!/usr/bin/env python
"""Apply or sweep thresholds over cached dual-top-3 overlap neighbors.

An evaluation segment is overlap-flagged only when one training candidate is
present in both its lexical and semantic nearest-three lists and that same
candidate passes both thresholds. Retrieval is not recomputed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_text_output(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as output:
        yield output
    temporary.replace(path)


def read_jsonl_objects(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line_number, row in read_jsonl_objects(path):
        if "eval_id" not in row or "shared_top3" not in row:
            raise ValueError(f"{path}:{line_number}: not a joined overlap-neighbor row")
        yield row


def passing_candidates(
    row: dict[str, Any], lexical_threshold: float, semantic_threshold: float
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in row["shared_top3"]
        if candidate["chrf3_6_f1"] >= lexical_threshold and candidate["semantic_cosine"] >= semantic_threshold
    ]


def classify(
    input_path: Path,
    output_path: Path,
    clean_ids_path: Path,
    overlap_ids_path: Path,
    *,
    lexical_threshold: float,
    semantic_threshold: float,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    overlap_by_language: Counter[str] = Counter()
    with (
        atomic_text_output(output_path) as output,
        atomic_text_output(clean_ids_path) as clean_ids,
        atomic_text_output(overlap_ids_path) as overlap_ids,
    ):
        for row in read_jsonl(input_path):
            language = row.get("lang") or "unknown"
            matches = passing_candidates(row, lexical_threshold, semantic_threshold)
            is_overlap = bool(matches)
            result = {
                "eval_id": row["eval_id"],
                "eval_dataset": row.get("eval_dataset"),
                "lang": language,
                "is_overlap": is_overlap,
                "passing_shared_candidates": matches,
                "lexical_threshold": lexical_threshold,
                "semantic_threshold": semantic_threshold,
            }
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            (overlap_ids if is_overlap else clean_ids).write(row["eval_id"] + "\n")
            counts[language] += 1
            overlap_by_language[language] += is_overlap
    total = sum(counts.values())
    overlap = sum(overlap_by_language.values())
    return {
        "definition": (
            "same training candidate in lexical top-3 and semantic top-3, with both thresholds passing"
        ),
        "lexical_metric": "mean symmetric character 3-6-gram F1",
        "lexical_threshold": lexical_threshold,
        "semantic_metric": "multilingual-E5 cosine",
        "semantic_threshold": semantic_threshold,
        "segments": total,
        "overlap": overlap,
        "clean": total - overlap,
        "overlap_fraction": overlap / total if total else 0.0,
        "by_language": {
            language: {
                "segments": counts[language],
                "overlap": overlap_by_language[language],
                "clean": counts[language] - overlap_by_language[language],
            }
            for language in sorted(counts)
        },
        "classification": str(output_path),
        "clean_ids": str(clean_ids_path),
        "overlap_ids": str(overlap_ids_path),
    }


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from error
    if not thresholds or any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must all fall in [0,1]")
    return thresholds


def sweep(
    input_path: Path,
    lexical_thresholds: Sequence[float],
    semantic_thresholds: Sequence[float],
) -> dict[str, Any]:
    rows = list(read_jsonl(input_path))
    cells = []
    for lexical_threshold in lexical_thresholds:
        for semantic_threshold in semantic_thresholds:
            overlap = sum(
                bool(passing_candidates(row, lexical_threshold, semantic_threshold)) for row in rows
            )
            cells.append(
                {
                    "lexical_threshold": lexical_threshold,
                    "semantic_threshold": semantic_threshold,
                    "overlap": overlap,
                    "clean": len(rows) - overlap,
                    "overlap_fraction": overlap / len(rows) if rows else 0.0,
                }
            )
    return {
        "definition": (
            "same training candidate in lexical top-3 and semantic top-3, with both thresholds passing"
        ),
        "segments": len(rows),
        "cells": cells,
    }


def select_evaluation(
    input_path: Path,
    classification_path: Path,
    clean_output_path: Path,
    overlap_output_path: Path,
) -> dict[str, Any]:
    """Partition authored evaluation rows using a complete overlap classification."""

    classifications: dict[str, dict[str, Any]] = {}
    for line_number, row in read_jsonl_objects(classification_path):
        eval_id = row.get("eval_id")
        if not isinstance(eval_id, str) or not isinstance(row.get("is_overlap"), bool):
            raise ValueError(f"{classification_path}:{line_number}: invalid overlap classification")
        if eval_id in classifications:
            raise ValueError(f"{classification_path}:{line_number}: duplicate eval_id {eval_id!r}")
        classifications[eval_id] = row

    evaluation_rows: dict[str, dict[str, Any]] = {}
    ordered_ids = []
    for line_number, row in read_jsonl_objects(input_path):
        eval_id = row.get("id")
        if not isinstance(eval_id, str):
            raise ValueError(f"{input_path}:{line_number}: missing string id")
        if eval_id in evaluation_rows:
            raise ValueError(f"{input_path}:{line_number}: duplicate id {eval_id!r}")
        evaluation_rows[eval_id] = row
        ordered_ids.append(eval_id)

    missing_classification = sorted(evaluation_rows.keys() - classifications.keys())
    unknown_classification = sorted(classifications.keys() - evaluation_rows.keys())
    if missing_classification or unknown_classification:
        raise ValueError(
            "evaluation/classification IDs differ; "
            f"missing classification={missing_classification[:5]}, "
            f"unknown classification={unknown_classification[:5]}"
        )

    clean_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    with (
        atomic_text_output(clean_output_path) as clean_output,
        atomic_text_output(overlap_output_path) as overlap_output,
    ):
        for eval_id in ordered_ids:
            row = evaluation_rows[eval_id]
            language = row.get("lang") or "unknown"
            if classifications[eval_id]["is_overlap"]:
                overlap_output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                overlap_counts[language] += 1
            else:
                clean_output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                clean_counts[language] += 1
    languages = sorted(clean_counts.keys() | overlap_counts.keys())
    clean = sum(clean_counts.values())
    overlap = sum(overlap_counts.values())
    return {
        "input": str(input_path),
        "classification": str(classification_path),
        "clean_output": str(clean_output_path),
        "overlap_output": str(overlap_output_path),
        "segments": clean + overlap,
        "clean": clean,
        "overlap": overlap,
        "by_language": {
            language: {
                "segments": clean_counts[language] + overlap_counts[language],
                "clean": clean_counts[language],
                "overlap": overlap_counts[language],
            }
            for language in languages
        },
    }


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    commands = argument_parser.add_subparsers(dest="command", required=True)

    classify_parser = commands.add_parser("classify", help="materialize one frozen threshold cut")
    classify_parser.add_argument("--input", type=Path, required=True)
    classify_parser.add_argument("--output", type=Path, required=True)
    classify_parser.add_argument("--clean-ids", type=Path, required=True)
    classify_parser.add_argument("--overlap-ids", type=Path, required=True)
    classify_parser.add_argument("--manifest", type=Path, required=True)
    classify_parser.add_argument("--lexical-threshold", type=float, required=True)
    classify_parser.add_argument("--semantic-threshold", type=float, required=True)

    sweep_parser = commands.add_parser("sweep", help="count overlap under a threshold grid")
    sweep_parser.add_argument("--input", type=Path, required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument("--lexical-thresholds", type=parse_thresholds, required=True)
    sweep_parser.add_argument("--semantic-thresholds", type=parse_thresholds, required=True)

    select_parser = commands.add_parser(
        "select-eval", help="partition evaluation JSONL using a complete classification"
    )
    select_parser.add_argument("--input", type=Path, required=True)
    select_parser.add_argument("--classification", type=Path, required=True)
    select_parser.add_argument("--clean-output", type=Path, required=True)
    select_parser.add_argument("--overlap-output", type=Path, required=True)
    select_parser.add_argument("--manifest", type=Path, required=True)

    admission = commands.add_parser(
        "admit-annotation", help="materialize verified additional annotation intake"
    )
    admission.add_argument("--evidence-receipt", type=Path, required=True)
    admission.add_argument("--clean-output", type=Path, required=True)
    admission.add_argument("--manifest", type=Path, required=True)
    within = commands.add_parser(
        "prepare-within", help="remove self from top-four retrieval before joining nearest three"
    )
    within.add_argument("--input", type=Path, required=True)
    within.add_argument("--source-name", required=True)
    within.add_argument("--lexical", type=Path, required=True)
    within.add_argument("--semantic", type=Path, required=True)
    within.add_argument("--output-dir", type=Path, required=True)
    within.add_argument("--manifest", type=Path, required=True)
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "admit-annotation":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.pii_dedup_gate import materialize_annotation_intake

        payload = materialize_annotation_intake(args.evidence_receipt, args.clean_output, args.manifest)
        print(
            json.dumps({"retained": len(payload["retained_ids"]), "rejected": len(payload["rejected_ids"])})
        )
        return 0
    if args.command == "prepare-within":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.pii_dedup_gate import prepare_within_neighbors

        payload = prepare_within_neighbors(
            args.input, args.source_name, args.lexical, args.semantic, args.output_dir
        )
        output_path = args.manifest
    elif args.command == "classify":
        if not 0 <= args.lexical_threshold <= 1 or not 0 <= args.semantic_threshold <= 1:
            raise ValueError("thresholds must fall in [0,1]")
        payload = classify(
            args.input,
            args.output,
            args.clean_ids,
            args.overlap_ids,
            lexical_threshold=args.lexical_threshold,
            semantic_threshold=args.semantic_threshold,
        )
        output_path = args.manifest
    elif args.command == "sweep":
        payload = sweep(args.input, args.lexical_thresholds, args.semantic_thresholds)
        output_path = args.output
    elif args.command == "select-eval":
        payload = select_evaluation(
            args.input,
            args.classification,
            args.clean_output,
            args.overlap_output,
        )
        output_path = args.manifest
    else:  # pragma: no cover
        raise AssertionError(args.command)
    with atomic_text_output(output_path) as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
