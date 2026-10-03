"""Mine authentic entity surfaces from two positive span labelers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pii_annotation_cache import atomic_text_output
    from pii_projector import Tagset
except ModuleNotFoundError:  # Imported as scripts.pii_surface.agreement_mining in tests.
    from scripts.pii_annotation_cache import atomic_text_output
    from scripts.pii_projector import Tagset

SCHEMA = "pii-surface-positive-agreement-v1"
REPORT_SCHEMA = "pii-surface-positive-agreement-report-v1"
LEDGER_SCHEMA = "pii-surface-qualification-ledger-v1"
MINING_ACTION = "high_precision_positive_mining"
OVERLAP_NUMERATOR = 4
OVERLAP_DENOMINATOR = 5


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def ordered_ids(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    result = []
    seen = set()
    for line_number, row in enumerate(rows, 1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{path}:{line_number}: missing nonempty id")
        if row_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
        seen.add(row_id)
        result.append(row_id)
    return result


def require_aligned(
    reference_path: Path,
    reference_rows: list[dict[str, Any]],
    other_path: Path,
    other_rows: list[dict[str, Any]],
) -> None:
    reference_ids = ordered_ids(reference_path, reference_rows)
    other_ids = ordered_ids(other_path, other_rows)
    if reference_ids == other_ids:
        return
    limit = min(len(reference_ids), len(other_ids))
    mismatch = next((index for index in range(limit) if reference_ids[index] != other_ids[index]), limit)
    reference_id = reference_ids[mismatch] if mismatch < len(reference_ids) else "<end>"
    other_id = other_ids[mismatch] if mismatch < len(other_ids) else "<end>"
    raise ValueError(
        f"row identity/order mismatch at zero-based row {mismatch}: "
        f"{reference_path} has {reference_id!r}, {other_path} has {other_id!r}"
    )


def canonical_label(raw: Any, tagset: Tagset, where: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{where}: span label must be a string")
    label = raw.lower()
    if label not in tagset.nodes:
        raise ValueError(f"{where}: unknown canonical label {raw!r}")
    return label


def parse_spans(raw_spans: Any, text: str, tagset: Tagset, where: str) -> tuple[list[Span], int]:
    if not isinstance(raw_spans, list):
        raise ValueError(f"{where}: spans/preds must be an array")
    spans = []
    for index, raw in enumerate(raw_spans):
        if isinstance(raw, list) and len(raw) == 3:
            start, end, label = raw
        elif isinstance(raw, dict):
            start, end, label = raw.get("start"), raw.get("end"), raw.get("label")
        else:
            raise ValueError(f"{where}: invalid span {index}: {raw!r}")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"{where}: span {index} start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"{where}: span {index} end must be an integer")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{where}: span {index} [{start}, {end}) is invalid for text length {len(text)}")
        spans.append(Span(start, end, canonical_label(label, tagset, f"{where}: span {index}")))
    unique = sorted(set(spans))
    return unique, len(spans) - len(unique)


def symmetric_overlap(left: Span, right: Span) -> bool:
    intersection = min(left.end, right.end) - max(left.start, right.start)
    if intersection <= 0:
        return False
    left_length = left.end - left.start
    right_length = right.end - right.start
    return (
        intersection * OVERLAP_DENOMINATOR >= left_length * OVERLAP_NUMERATOR
        and intersection * OVERLAP_DENOMINATOR >= right_length * OVERLAP_NUMERATOR
    )


def positive_intersection(left: Span, right: Span) -> bool:
    return min(left.end, right.end) > max(left.start, right.start)


def agreement_pairs(
    teacher: list[Span],
    model: list[Span],
) -> tuple[list[tuple[str, Span, Span]], set[Span], set[Span]]:
    teacher_set = set(teacher)
    model_set = set(model)
    exact = sorted(teacher_set & model_set)
    matched_teacher = set(exact)
    matched_model = set(exact)
    pairs = [("exact_same_label", span, span) for span in exact]

    teacher_remaining = [span for span in teacher if span not in matched_teacher]
    model_remaining = [span for span in model if span not in matched_model]
    edges = [
        (teacher_span, model_span)
        for teacher_span in teacher_remaining
        for model_span in model_remaining
        if teacher_span.label == model_span.label and symmetric_overlap(teacher_span, model_span)
    ]
    teacher_degree = Counter(teacher_span for teacher_span, _ in edges)
    model_degree = Counter(model_span for _, model_span in edges)
    for teacher_span, model_span in sorted(edges):
        if teacher_degree[teacher_span] != 1 or model_degree[model_span] != 1:
            continue
        pairs.append(("unique_symmetric80_same_label", teacher_span, model_span))
        matched_teacher.add(teacher_span)
        matched_model.add(model_span)
    return pairs, matched_teacher, matched_model


def target_cells(ledger_path: Path, tagset: Tagset) -> set[tuple[str, str]]:
    cells = set()
    for line_number, row in enumerate(read_jsonl(ledger_path), 1):
        if row.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"{ledger_path}:{line_number}: expected schema {LEDGER_SCHEMA!r}")
        if MINING_ACTION not in row.get("triage_actions", []):
            continue
        language = row.get("language")
        tag = row.get("tag")
        if not isinstance(language, str) or not isinstance(tag, str):
            raise ValueError(f"{ledger_path}:{line_number}: missing language or tag")
        label = tag.lower()
        if label not in tagset.nodes:
            raise ValueError(f"{ledger_path}:{line_number}: unknown canonical tag {tag!r}")
        cells.add((language, label))
    if not cells:
        raise ValueError(f"{ledger_path}: no {MINING_ACTION!r} cells")
    return cells


def context_payload(text: str, start: int, end: int, radius: int) -> dict[str, Any]:
    context_start = max(0, start - radius)
    context_end = min(len(text), end + radius)
    return {
        "context_start": context_start,
        "context_end": context_end,
        "context": text[context_start:context_end],
    }


def candidate_id(row_id: str, tier: str, span: Span) -> str:
    identity = f"{row_id}\0{tier}\0{span.start}\0{span.end}\0{span.label}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def candidate_record(
    *,
    source_row: dict[str, Any],
    tier: str,
    status: str,
    model_span: Span,
    teacher_spans: list[Span],
    context_radius: int,
    adjudication_reason: str | None = None,
) -> dict[str, Any]:
    text = source_row["text"]
    record = {
        "schema": SCHEMA,
        "candidate_id": candidate_id(source_row["id"], tier, model_span),
        "status": status,
        "tier": tier,
        "id": source_row["id"],
        "document_id": source_row.get("document_id"),
        "language": source_row["lang"],
        "tag": model_span.label.upper(),
        "start": model_span.start,
        "end": model_span.end,
        "surface": text[model_span.start : model_span.end],
        "model_span": model_span.as_dict(),
        "teacher_spans": [span.as_dict() for span in teacher_spans],
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sampling_weight": source_row.get("sampling_weight"),
        **context_payload(text, model_span.start, model_span.end, context_radius),
    }
    if adjudication_reason is not None:
        record["adjudication_reason"] = adjudication_reason
    return record


def teacher_candidate_record(
    *,
    source_row: dict[str, Any],
    teacher_span: Span,
    model_spans: list[Span],
    context_radius: int,
    adjudication_reason: str,
) -> dict[str, Any]:
    text = source_row["text"]
    tier = "teacher_unmatched"
    return {
        "schema": SCHEMA,
        "candidate_id": candidate_id(source_row["id"], tier, teacher_span),
        "status": "targeted_adjudication_required",
        "tier": tier,
        "id": source_row["id"],
        "document_id": source_row.get("document_id"),
        "language": source_row["lang"],
        "tag": teacher_span.label.upper(),
        "start": teacher_span.start,
        "end": teacher_span.end,
        "surface": text[teacher_span.start : teacher_span.end],
        "teacher_span": teacher_span.as_dict(),
        "model_spans": [span.as_dict() for span in model_spans],
        "adjudication_reason": adjudication_reason,
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sampling_weight": source_row.get("sampling_weight"),
        **context_payload(text, teacher_span.start, teacher_span.end, context_radius),
    }


def model_support_for_teacher(
    teacher_span: Span,
    model_spans: list[Span],
    tagset: Tagset,
) -> tuple[str, list[Span]]:
    symmetric = [span for span in model_spans if symmetric_overlap(teacher_span, span)]
    intersecting = [span for span in model_spans if positive_intersection(teacher_span, span)]
    if any(span.label == teacher_span.label for span in symmetric):
        return "ambiguous_same_label_symmetric80", symmetric
    teacher_p20 = tagset.project_canonical_cut(teacher_span.label, "redaction_20_v1")
    if any(tagset.project_canonical_cut(span.label, "redaction_20_v1") == teacher_p20 for span in symmetric):
        return "v11_p20_symmetric80_support", symmetric
    teacher_p9 = tagset.project_canonical_cut(teacher_span.label, "redaction_9_v1")
    if any(tagset.project_canonical_cut(span.label, "redaction_9_v1") == teacher_p9 for span in symmetric):
        return "v11_p9_symmetric80_support", symmetric
    if symmetric:
        return "v11_type_disagreement", symmetric
    if intersecting:
        return "v11_geometry_or_type_disagreement", intersecting
    return "no_v11_overlap", []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with atomic_text_output(str(path)) as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def cell_summary(rows: list[dict[str, Any]], cells: set[tuple[str, str]]) -> list[dict[str, Any]]:
    counts = Counter((row["language"], row["tag"].lower(), row["tier"]) for row in rows)
    surfaces: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = row["language"], row["tag"].lower(), row["tier"]
        surfaces[key].add(unicodedata.normalize("NFC", row["surface"]).casefold())
    result = []
    for language, label in sorted(cells):
        result.append(
            {
                "language": language,
                "tag": label.upper(),
                "exact_agreement": counts[language, label, "exact_same_label"],
                "exact_distinct_nfc_casefold": len(surfaces[language, label, "exact_same_label"]),
                "overlap_agreement": counts[language, label, "unique_symmetric80_same_label"],
                "adjudication": counts[language, label, "v11_unmatched"],
                "teacher_adjudication": counts[language, label, "teacher_unmatched"],
            }
        )
    return result


def mine_positive_agreement(
    *,
    source_path: Path,
    teacher_path: Path,
    prediction_path: Path,
    ledger_path: Path,
    output_dir: Path,
    context_radius: int = 80,
) -> dict[str, Any]:
    if context_radius < 0:
        raise ValueError("context_radius must be nonnegative")
    tagset = Tagset()
    cells = target_cells(ledger_path, tagset)
    source_rows = read_jsonl(source_path)
    teacher_rows = read_jsonl(teacher_path)
    prediction_rows = read_jsonl(prediction_path)
    require_aligned(source_path, source_rows, teacher_path, teacher_rows)
    require_aligned(source_path, source_rows, prediction_path, prediction_rows)

    exact_rows = []
    overlap_rows = []
    adjudication_rows = []
    teacher_adjudication_rows = []
    duplicate_counts = Counter()
    targeted_teacher_spans = targeted_model_spans = 0
    for line_number, (source_row, teacher_row, prediction_row) in enumerate(
        zip(source_rows, teacher_rows, prediction_rows), 1
    ):
        text = source_row.get("text")
        language = source_row.get("lang")
        if not isinstance(text, str) or not isinstance(language, str):
            raise ValueError(f"{source_path}:{line_number}: missing text or language")
        if teacher_row.get("text") != text or teacher_row.get("lang") != language:
            raise ValueError(f"{teacher_path}:{line_number}: text/language differs from source")
        teacher_spans, teacher_duplicates = parse_spans(
            teacher_row.get("spans"), text, tagset, f"{teacher_path}:{line_number}"
        )
        model_spans, model_duplicates = parse_spans(
            prediction_row.get("preds"), text, tagset, f"{prediction_path}:{line_number}"
        )
        duplicate_counts["teacher"] += teacher_duplicates
        duplicate_counts["model"] += model_duplicates
        row_targets = {label for cell_language, label in cells if cell_language == language}
        teacher_targets = [span for span in teacher_spans if span.label in row_targets]
        model_targets = [span for span in model_spans if span.label in row_targets]
        targeted_teacher_spans += len(teacher_targets)
        targeted_model_spans += len(model_targets)
        pairs, matched_teacher, matched_model = agreement_pairs(teacher_targets, model_targets)
        for tier, teacher_span, model_span in pairs:
            status = (
                "agreement_candidate_quality_unreviewed"
                if tier == "exact_same_label"
                else "overlap_agreement_requires_review"
            )
            record = candidate_record(
                source_row=source_row,
                tier=tier,
                status=status,
                model_span=model_span,
                teacher_spans=[teacher_span],
                context_radius=context_radius,
            )
            (exact_rows if tier == "exact_same_label" else overlap_rows).append(record)

        for model_span in model_targets:
            if model_span in matched_model:
                continue
            symmetric = [span for span in teacher_spans if symmetric_overlap(span, model_span)]
            intersecting = [span for span in teacher_spans if positive_intersection(span, model_span)]
            same_label_symmetric = [span for span in symmetric if span.label == model_span.label]
            if same_label_symmetric:
                reason = "ambiguous_same_label_symmetric80"
            elif symmetric:
                reason = "teacher_type_disagreement"
            elif intersecting:
                reason = "teacher_geometry_or_type_disagreement"
            else:
                reason = "teacher_positive_only_omission_or_model_false_positive"
            adjudication_rows.append(
                candidate_record(
                    source_row=source_row,
                    tier="v11_unmatched",
                    status="targeted_adjudication_required",
                    model_span=model_span,
                    teacher_spans=symmetric or intersecting,
                    context_radius=context_radius,
                    adjudication_reason=reason,
                )
            )
        for teacher_span in teacher_targets:
            if teacher_span in matched_teacher:
                continue
            reason, support = model_support_for_teacher(teacher_span, model_spans, tagset)
            teacher_adjudication_rows.append(
                teacher_candidate_record(
                    source_row=source_row,
                    teacher_span=teacher_span,
                    model_spans=support,
                    context_radius=context_radius,
                    adjudication_reason=reason,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exact_agreement": output_dir / "exact-agreement.jsonl",
        "overlap_agreement": output_dir / "overlap-agreement.jsonl",
        "adjudication": output_dir / "adjudication.jsonl",
        "teacher_adjudication": output_dir / "teacher-adjudication.jsonl",
    }
    write_jsonl(outputs["exact_agreement"], exact_rows)
    write_jsonl(outputs["overlap_agreement"], overlap_rows)
    write_jsonl(outputs["adjudication"], adjudication_rows)
    write_jsonl(outputs["teacher_adjudication"], teacher_adjudication_rows)
    all_rows = exact_rows + overlap_rows + adjudication_rows + teacher_adjudication_rows
    report = {
        "schema": REPORT_SCHEMA,
        "status": "candidate_mining_complete_quality_unreviewed",
        "contract": {
            "teacher_supervision": "annotated_spans_only; omissions are unknown, never negative",
            "exact_tier": "identical half-open span and canonical fine label",
            "overlap_tier": (
                "same canonical fine label; intersection covers at least 80% of each span; "
                "both endpoints have exactly one eligible counterpart after exact matches"
            ),
            "adjudication": "unmatched v11 target spans require an independent focused judgment",
            "teacher_adjudication": (
                "unmatched 31B target spans retain the teacher fine label; v11 P20/P9 support is "
                "reported as evidence for focused judgment, never promoted to fine agreement"
            ),
            "admission": "no tier is admitted to a realizer or training corpus by this run",
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in {
                "source": source_path,
                "teacher": teacher_path,
                "predictions": prediction_path,
                "qualification_ledger": ledger_path,
            }.items()
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "rows": len(rows),
            }
            for name, path, rows in (
                ("exact_agreement", outputs["exact_agreement"], exact_rows),
                ("overlap_agreement", outputs["overlap_agreement"], overlap_rows),
                ("adjudication", outputs["adjudication"], adjudication_rows),
                (
                    "teacher_adjudication",
                    outputs["teacher_adjudication"],
                    teacher_adjudication_rows,
                ),
            )
        },
        "summary": {
            "source_rows": len(source_rows),
            "target_cells": len(cells),
            "targeted_teacher_spans": targeted_teacher_spans,
            "targeted_model_spans": targeted_model_spans,
            "exact_agreement": len(exact_rows),
            "overlap_agreement": len(overlap_rows),
            "adjudication": len(adjudication_rows),
            "teacher_adjudication": len(teacher_adjudication_rows),
            "duplicate_input_spans_removed": dict(sorted(duplicate_counts.items())),
        },
        "cells": cell_summary(all_rows, cells),
    }
    report_path = output_dir / "report.json"
    with atomic_text_output(str(report_path)) as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return report
