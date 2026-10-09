#!/usr/bin/env python
"""Project canonical character-span PII rows into GLiNER2 training rows.

GLiNER2 1.3.x accepts entity supervision as ``label -> surface strings``.
Its processor then finds every case-insensitive whitespace-token occurrence of
each surface.  That representation is lossy: an unlabelled duplicate surface,
the same surface carrying two labels, or a span inside an unsplit CJK token can
silently change the training targets.

This converter is deliberately fail-closed.  It runs the generated schema
through the installed GLiNER2 processor, maps the resulting word spans back to
character offsets, and emits a row only when the reconstructed multiset exactly
equals the canonical input spans.  Rejected rows retain the exact difference
and source provenance for later splitting or exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pii_annotation_cache import atomic_text_output, normalize_span, validate_spans  # noqa: E402
from pii_projector import Tagset  # noqa: E402

PROJECTION_VERSION = "gliner2-surface-roundtrip-v2"


@dataclass(frozen=True)
class SpanDifference:
    start: int
    end: int
    label: str
    surface: str


@dataclass(frozen=True)
class ProjectionDiagnostic:
    accepted: bool
    reasons: tuple[str, ...]
    missing: tuple[SpanDifference, ...]
    extra: tuple[SpanDifference, ...]
    unmapped: tuple[SpanDifference, ...]
    input_tokens: int
    schema_labels: tuple[str, ...]
    max_observed_span_width: int
    projected_span_count: int
    collapsed_projected_spans: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def labels_from_model_config(path: Path, tagset: Tagset) -> list[str]:
    """Return the canonical fine labels encoded by a BIOES model config."""
    config = json.loads(path.read_text(encoding="utf-8"))
    raw = config.get("id2label")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an id2label object")
    labels: set[str] = set()
    for encoded in raw.values():
        if encoded == "O":
            continue
        if not isinstance(encoded, str) or len(encoded) < 3 or encoded[1] != "-" or encoded[0] not in "BIES":
            raise ValueError(f"{path}: invalid BIOES label {encoded!r}")
        label = encoded[2:]
        if label not in tagset.nodes:
            raise ValueError(f"{path}: unknown canonical label {label!r}")
        labels.add(label)
    if not labels:
        raise ValueError(f"{path}: BIOES inventory is empty")
    return sorted(labels)


def inventory_for_output_schema(tagset: Tagset, output_schema: str) -> list[str]:
    if output_schema == "canonical":
        raise ValueError("canonical output needs an explicit label inventory")
    if tagset.is_cut_schema(output_schema):
        return sorted(tagset.cut_targets(output_schema))
    if output_schema in tagset.sources:
        return list(tagset.sources[output_schema])
    raise ValueError(f"unknown output schema {output_schema!r}")


def project_canonical_spans(
    tagset: Tagset,
    spans: Sequence[tuple[int, int, str]],
    inventory: Sequence[str],
    *,
    output_schema: str,
    alias_policy: str,
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]], int]:
    """Project canonical spans into the query labels used for GLiNER2 training."""
    if alias_policy not in {"all", "first"}:
        raise ValueError(f"unknown alias policy {alias_policy!r}")
    inventory_set = set(inventory)
    projected: list[tuple[int, int, str]] = []
    unmapped: list[tuple[int, int, str]] = []
    for start, end, canonical_label in spans:
        if output_schema == "canonical":
            labels = [canonical_label] if canonical_label in inventory_set else []
        elif tagset.is_cut_schema(output_schema):
            labels = [tagset.project_canonical_cut(canonical_label, output_schema)]
        else:
            compatible = set(tagset.back_project(canonical_label, output_schema))
            labels = [label for label in inventory if label in compatible]
        if not labels:
            unmapped.append((start, end, canonical_label))
            continue
        if alias_policy == "first":
            labels = labels[:1]
        projected.extend((start, end, label) for label in labels)

    if output_schema == "canonical":
        return sorted(projected), sorted(unmapped), 0
    unique = sorted(set(projected))
    return unique, sorted(unmapped), len(projected) - len(unique)


def _stable_negative_labels(
    inventory: Sequence[str],
    positive_labels: set[str],
    row_key: str,
    count: int,
    seed: int,
) -> list[str]:
    candidates = [label for label in inventory if label not in positive_labels]
    candidates.sort(key=lambda label: hashlib.sha256(f"{seed}\0{row_key}\0{label}".encode()).digest())
    return candidates[:count]


def build_entity_schema(
    text: str,
    spans: Sequence[tuple[int, int, str]],
    inventory: Sequence[str],
    *,
    row_key: str,
    negative_label_count: int,
    seed: int,
    surface_key: Callable[[str], Hashable] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Build a schema without repeating processor-equivalent mentions."""
    if surface_key is None:
        surface_key = lambda surface: surface
    positives: dict[str, list[str]] = {}
    seen_surfaces: dict[str, set[Hashable]] = {}
    for start, end, label in spans:
        mentions = positives.setdefault(label, [])
        surface = text[start:end]
        key = surface_key(surface)
        seen = seen_surfaces.setdefault(label, set())
        if key not in seen:
            mentions.append(surface)
            seen.add(key)

    positive_labels = set(positives)
    selected = sorted(positive_labels)
    selected.extend(
        _stable_negative_labels(
            inventory,
            positive_labels,
            row_key,
            negative_label_count,
            seed,
        )
    )
    selected = sorted(selected)
    if not selected:
        raise ValueError("GLiNER2 projection needs at least one positive or sampled negative label")
    return {label: positives.get(label, []) for label in selected}, selected


def _span_difference(span: tuple[int, int, str], text: str) -> SpanDifference:
    start, end, label = span
    surface = text[start:end] if 0 <= start <= end <= len(text) else ""
    return SpanDifference(start, end, label, surface)


def _counter_difference(
    left: Counter[tuple[int, int, str]], right: Counter[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    for span, count in (left - right).items():
        result.extend([span] * count)
    return sorted(result)


def roundtrip_schema(
    processor: Any,
    text: str,
    entities: dict[str, list[str]],
) -> tuple[list[tuple[int, int, str]], int, int]:
    """Recover character spans using the same processor semantics as training."""
    processor.change_mode(False)
    processed_text = text if text.endswith((".", "!", "?")) else text + "."
    transformed = processor.transform_and_format(processed_text, {"entities": entities})
    if transformed.task_types != ["entities"] or len(transformed.structure_labels) != 1:
        raise ValueError(f"processor did not produce one entity task: {transformed.task_types!r}")
    structure = transformed.structure_labels[0]
    if structure[0] != 1 or len(structure[1]) != 1:
        raise ValueError(f"unexpected GLiNER2 entity structure: {structure!r}")
    fields = structure[1][0]
    labels = list(entities)
    if len(fields) != len(labels):
        raise ValueError(f"processor returned {len(fields)} fields for {len(labels)} labels")

    recovered: list[tuple[int, int, str]] = []
    max_width = 0
    for label, positions in zip(labels, fields):
        for position in positions:
            if position is None or position == (-1, -1):
                continue
            start_word, end_word = position
            if not (0 <= start_word <= end_word < len(transformed.start_token_idx)):
                raise ValueError(f"invalid recovered word span {position!r} for {label!r}")
            max_width = max(max_width, end_word - start_word + 1)
            recovered.append(
                (
                    transformed.start_token_idx[start_word],
                    transformed.end_token_idx[end_word],
                    label,
                )
            )
    return recovered, len(transformed.input_ids), max_width


def project_row(
    processor: Any,
    row: dict[str, Any],
    inventory: Sequence[str],
    *,
    input_schema: str = "canonical",
    output_schema: str = "canonical",
    alias_policy: str = "all",
    negative_label_count: int = 16,
    seed: int = 20260804,
    max_input_tokens: int = 512,
    max_span_width: int = 8,
    row_key: str | None = None,
) -> tuple[dict[str, Any], ProjectionDiagnostic]:
    """Convert one row and return its GLiNER2 record plus exact diagnostic."""
    tagset = Tagset()
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("row text must be a non-empty string")
    raw_spans = row.get("spans")
    if not isinstance(raw_spans, list):
        raise ValueError("row spans must be a list")
    canonical_spans = sorted(
        (tuple(normalize_span(tagset, input_schema, span)) for span in raw_spans),
        key=lambda span: (span[0], span[1], span[2]),
    )
    validate_spans("row", 1, text, canonical_spans, allow_overlaps=True)
    spans, unmapped, collapsed_projected_spans = project_canonical_spans(
        tagset,
        canonical_spans,
        inventory,
        output_schema=output_schema,
        alias_policy=alias_policy,
    )
    if output_schema == "canonical" and unmapped:
        unknown = sorted({label for _, _, label in unmapped})
        raise ValueError(f"canonical labels absent from requested inventory: {', '.join(unknown)}")

    effective_key = row_key or str(row.get("id", hashlib.sha256(text.encode()).hexdigest()))
    entities, schema_labels = build_entity_schema(
        text,
        spans,
        inventory,
        row_key=effective_key,
        negative_label_count=negative_label_count,
        seed=seed,
        surface_key=lambda surface: tuple(
            token for token, _, _ in processor.word_splitter(surface, lower=True)
        ),
    )
    recovered, input_tokens, observed_width = roundtrip_schema(processor, text, entities)
    wanted_counter = Counter(spans)
    recovered_counter = Counter(recovered)
    missing = _counter_difference(wanted_counter, recovered_counter)
    extra = _counter_difference(recovered_counter, wanted_counter)

    reasons: list[str] = []
    if missing:
        reasons.append("roundtrip_missing")
    if extra:
        reasons.append("roundtrip_extra")
    if input_tokens > max_input_tokens:
        reasons.append("input_too_long")
    if observed_width > max_span_width:
        reasons.append("span_too_wide")
    accepted = not reasons

    output = {
        "input": text,
        "output": {"entities": entities},
        "provenance": {
            "projection": PROJECTION_VERSION,
            "source_id": row.get("id"),
            "lang": row.get("lang"),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "alias_policy": alias_policy,
            "canonical_spans": [list(span) for span in canonical_spans],
            "projected_spans": [list(span) for span in spans],
        },
    }
    diagnostic = ProjectionDiagnostic(
        accepted=accepted,
        reasons=tuple(reasons),
        missing=tuple(_span_difference(span, text) for span in missing),
        extra=tuple(_span_difference(span, text) for span in extra),
        unmapped=tuple(_span_difference(span, text) for span in unmapped),
        input_tokens=input_tokens,
        schema_labels=tuple(schema_labels),
        max_observed_span_width=observed_width,
        projected_span_count=len(spans),
        collapsed_projected_spans=collapsed_projected_spans,
    )
    return output, diagnostic


def project_files(args: argparse.Namespace) -> dict[str, Any]:
    from gliner2.processor import SchemaTransformer

    tagset = Tagset()
    if args.output_schema != "canonical":
        if args.label_vocab_config or args.all_canonical_nodes:
            raise ValueError("--label-vocab-config/--all-canonical-nodes apply only to canonical output")
        inventory = inventory_for_output_schema(tagset, args.output_schema)
        inventory_source = f"scripts/pii_tagset.yaml:{args.output_schema}"
    elif args.label_vocab_config:
        inventory = labels_from_model_config(args.label_vocab_config, tagset)
        inventory_source = str(args.label_vocab_config)
    elif args.all_canonical_nodes:
        inventory = sorted(tagset.nodes)
        inventory_source = "scripts/pii_tagset.yaml:nodes"
    else:
        raise ValueError("canonical output requires --label-vocab-config or --all-canonical-nodes")
    processor = SchemaTransformer(model_name=str(args.model))

    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    language_counts: dict[str, Counter[str]] = {}
    candidate_label_schema_exposures: Counter[str] = Counter()
    candidate_label_positive_exposures: Counter[str] = Counter()
    candidate_canonical_positive_exposures: Counter[str] = Counter()
    accepted_label_schema_exposures: Counter[str] = Counter()
    accepted_label_positive_exposures: Counter[str] = Counter()
    accepted_canonical_positive_exposures: Counter[str] = Counter()
    input_hashes = {str(path): file_sha256(path) for path in args.input}
    language_overrides: dict[str, str] = {}
    for spec in args.source_language:
        language, separator, raw_path = spec.partition("=")
        if not separator or not language or not raw_path:
            raise ValueError(f"invalid --source-language {spec!r}; expected LANG=PATH")
        path_key = str(Path(raw_path).resolve())
        if path_key in language_overrides:
            raise ValueError(f"duplicate --source-language path {path_key!r}")
        language_overrides[path_key] = language
    input_path_keys = {str(path.resolve()) for path in args.input}
    unknown_override_paths = sorted(set(language_overrides) - input_path_keys)
    if unknown_override_paths:
        raise ValueError("--source-language paths absent from --input: " + ", ".join(unknown_override_paths))

    with ExitStack() as stack:
        accepted_out = stack.enter_context(atomic_text_output(args.output))
        rejected_out = stack.enter_context(atomic_text_output(args.rejected))
        for input_path in args.input:
            for line_number, row in read_jsonl(input_path):
                override_language = language_overrides.get(str(input_path.resolve()))
                if override_language:
                    existing_language = row.get("lang")
                    if existing_language is not None and str(existing_language) != override_language:
                        raise ValueError(
                            f"{input_path}:{line_number}: row language {existing_language!r} "
                            f"conflicts with override {override_language!r}"
                        )
                    row = dict(row)
                    row["lang"] = override_language
                row_key = f"{input_path}:{line_number}:{row.get('id', '')}"
                output, diagnostic = project_row(
                    processor,
                    row,
                    inventory,
                    input_schema=args.input_schema,
                    output_schema=args.output_schema,
                    alias_policy=args.alias_policy,
                    negative_label_count=args.negative_label_count,
                    seed=args.seed,
                    max_input_tokens=args.max_input_tokens,
                    max_span_width=args.max_span_width,
                    row_key=row_key,
                )
                output["provenance"].update({"source_path": str(input_path), "source_line": line_number})
                counts["rows"] += 1
                language = str(row.get("lang", "unknown"))
                language_counts.setdefault(language, Counter())["rows"] += 1
                canonical_spans = output["provenance"]["canonical_spans"]
                projected_spans = output["provenance"]["projected_spans"]
                counts["canonical_spans"] += len(canonical_spans)
                counts["projected_spans"] += len(projected_spans)
                counts["unmapped_canonical_spans"] += len(diagnostic.unmapped)
                counts["collapsed_projected_spans"] += diagnostic.collapsed_projected_spans
                language_counts[language]["canonical_spans"] += len(canonical_spans)
                language_counts[language]["projected_spans"] += len(projected_spans)
                language_counts[language]["unmapped_canonical_spans"] += len(diagnostic.unmapped)
                for label in diagnostic.schema_labels:
                    candidate_label_schema_exposures[label] += 1
                for span in canonical_spans:
                    candidate_canonical_positive_exposures[span[2]] += 1
                for span in projected_spans:
                    candidate_label_positive_exposures[span[2]] += 1

                if diagnostic.accepted:
                    accepted_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
                    language_counts[language]["accepted"] += 1
                    accepted_label_schema_exposures.update(diagnostic.schema_labels)
                    accepted_canonical_positive_exposures.update(span[2] for span in canonical_spans)
                    accepted_label_positive_exposures.update(span[2] for span in projected_spans)
                else:
                    rejected = dict(row)
                    rejected["gliner2_projection"] = asdict(diagnostic)
                    rejected["gliner2_projection"]["source_path"] = str(input_path)
                    rejected["gliner2_projection"]["source_line"] = line_number
                    rejected_out.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                    counts["rejected"] += 1
                    language_counts[language]["rejected"] += 1
                    reason_counts.update(diagnostic.reasons)

    report = {
        "version": PROJECTION_VERSION,
        "inputs": input_hashes,
        "model": str(args.model),
        "inventory_source": inventory_source,
        "inventory_size": len(inventory),
        "inventory": inventory,
        "settings": {
            "input_schema": args.input_schema,
            "output_schema": args.output_schema,
            "alias_policy": args.alias_policy,
            "negative_label_count": args.negative_label_count,
            "seed": args.seed,
            "max_input_tokens": args.max_input_tokens,
            "max_span_width": args.max_span_width,
        },
        "counts": dict(counts),
        "reasons": dict(sorted(reason_counts.items())),
        "languages": {key: dict(value) for key, value in sorted(language_counts.items())},
        "candidate_label_schema_exposures": dict(sorted(candidate_label_schema_exposures.items())),
        "candidate_canonical_positive_exposures": dict(
            sorted(candidate_canonical_positive_exposures.items())
        ),
        "candidate_label_positive_exposures": dict(sorted(candidate_label_positive_exposures.items())),
        "accepted_label_schema_exposures": dict(sorted(accepted_label_schema_exposures.items())),
        "accepted_canonical_positive_exposures": dict(sorted(accepted_canonical_positive_exposures.items())),
        "accepted_label_positive_exposures": dict(sorted(accepted_label_positive_exposures.items())),
        "outputs": {
            "accepted": {"path": str(args.output), "sha256": file_sha256(args.output)},
            "rejected": {"path": str(args.rejected), "sha256": file_sha256(args.rejected)},
        },
    }
    with atomic_text_output(args.report) as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, action="append", required=True)
    result.add_argument(
        "--source-language",
        action="append",
        default=[],
        metavar="LANG=PATH",
        help="supply or verify a language for one input file (repeatable)",
    )
    result.add_argument("--output", type=Path, required=True, help="accepted GLiNER2 JSONL")
    result.add_argument(
        "--rejected", type=Path, required=True, help="rejected canonical rows plus diagnostics"
    )
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True, help="GLiNER2 model/tokenizer directory")
    inventory = result.add_mutually_exclusive_group()
    inventory.add_argument(
        "--label-vocab-config",
        type=Path,
        help="BIOES model config whose fine labels define the comparison inventory",
    )
    inventory.add_argument(
        "--all-canonical-nodes",
        action="store_true",
        help="use every node in pii_tagset.yaml, including abstract parents",
    )
    result.add_argument("--input-schema", default="canonical")
    result.add_argument(
        "--output-schema",
        default="canonical",
        help="GLiNER2 query-label schema: canonical, a source schema, or a reporting cut",
    )
    result.add_argument(
        "--alias-policy",
        choices=("all", "first"),
        default="all",
        help="for source schemas, supervise all compatible aliases or only the first",
    )
    result.add_argument("--negative-label-count", type=int, default=16)
    result.add_argument("--seed", type=int, default=20260804)
    result.add_argument("--max-input-tokens", type=int, default=512)
    result.add_argument("--max-span-width", type=int, default=8)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.negative_label_count < 0:
        raise ValueError("--negative-label-count must be nonnegative")
    report = project_files(args)
    counts = report["counts"]
    print(
        f"GLINER2_PROJECT: {counts.get('accepted', 0)}/{counts.get('rows', 0)} accepted; "
        f"{counts.get('rejected', 0)} rejected -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
