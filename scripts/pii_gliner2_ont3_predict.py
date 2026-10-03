#!/usr/bin/env python3
# acli: 1 complete
"""Run GLiNER2 zero-shot over ont3 type names and emit annotator-format rows.

GLiNER2 scores spans against label names supplied at inference, so the ont3
comparison does not need its shipped privacy vocabulary: the 31 ont3 primary
types are handed to it directly as labels. That is the arrangement worth
measuring, because it is how a span bi-encoder would be used against this
ontology without retraining.

It is also out of distribution for most of the ontology. The shipped
`fastino_42` vocabulary reaches 11 of the 31 ont3 types; the other 20,
including organization, both reference types and health_condition, have no
counterpart it was trained on. Expect the zero-shot numbers to understate what
a fine-tuned span bi-encoder would do, and read them as a floor.

Output is one row per input with character-offset `preds`, the format
`pii_ont3_standalone_payload.py` already converts for the ont3 scorer, so the
comparison runs through exactly the same scoring path as an encoder.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "agents"))
import acli  # noqa: E402

SCHEMA = "pii-gliner2-ont3-predict/v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ont3_labels(labels_path: Path) -> list[str]:
    """The ont3 primary type names, from either label form.

    A pool's `labels.json` holds bare type names; a checkpoint's holds BIOES
    labels. Both are accepted because the type names are what the span model
    needs and both files are in circulation.
    """
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    bioes = {label.partition("-")[2] for label in labels if label[:2] in ("B-", "I-", "O-", "E-", "S-")}
    types = sorted(bioes) if bioes else sorted({label for label in labels if label != "O"})
    if not types:
        raise ValueError(f"{labels_path}: no primary types to give the model")
    return types


def occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    """Every non-overlapping occurrence of one surface."""
    found = []
    start = text.find(surface)
    while start >= 0:
        found.append((start, start + len(surface)))
        start = text.find(surface, start + len(surface))
    return found


def spans_from(result: Any, text: str, label_names: list[str]) -> list[dict[str, Any]]:
    """Normalize one GLiNER2 result into character-offset spans.

    The library answers `{"entities": {label: [surface, ...]}}`: surfaces, not
    offsets. Its own training format has the same shape and its processor reads
    a named surface as marking *every* occurrence in the document, so that is
    the reading used here. Taking only the first occurrence instead would drop
    repeated mentions and understate recall against a scorer that counts each
    occurrence separately.

    The cost of this convention is the one the exact-offset work already
    documented: an identical surface that is PII in one place and not in
    another cannot be expressed, so some false positives here are the
    representation's rather than the model's.
    """
    if not isinstance(result, dict):
        raise ValueError(f"unexpected GLiNER2 result type {type(result).__name__}")
    entities = result.get("entities", result)
    if not isinstance(entities, dict):
        raise ValueError("GLiNER2 result has no entity mapping")
    allowed = set(label_names)
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    for label, entries in entities.items():
        if label not in allowed:
            continue
        for entry in entries or []:
            surface = entry.get("text") if isinstance(entry, dict) else entry
            if not isinstance(surface, str) or not surface.strip():
                continue
            for start, end in occurrences(text, surface):
                out[(start, end, label)] = {"start": start, "end": end, "label": label}
    return [out[key] for key in sorted(out)]


def run(args: Any) -> dict[str, Any]:
    from gliner2 import GLiNER2

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    labels = ont3_labels(args.labels)
    model = GLiNER2.from_pretrained(args.model, map_location=args.device)
    from pii_gliner2_label_transfer import prompt_mapping

    mapping = prompt_mapping(model, labels)
    reverse = {prompt: canonical for canonical, prompt in mapping.items()}
    prompt_labels = list(mapping.values())
    if args.device == "cpu":
        model = model.float()
    model = model.eval()

    started = time.perf_counter()
    written = 0
    spans = 0
    empty = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            text = row[args.text_key]
            found = spans_from(model.extract_entities(text, prompt_labels), text, prompt_labels)
            for span in found:
                span["label"] = reverse[span["label"]]
            spans += len(found)
            empty += 1 if not found else 0
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "preds": found,
                        "example_lang": row.get("bcp47") or row.get("lang") or "und",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
            if (index + 1) % 100 == 0:
                print(f"gliner2-ont3 {index + 1}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "model": str(args.model),
        "device": args.device,
        "labels": labels,
        "canonical_to_prompt": mapping,
        "label_count": len(labels),
        "input": str(args.input),
        "output": str(args.output),
        "rows": written,
        "spans": spans,
        "rows_without_a_span": empty,
        "elapsed_s": round(elapsed, 2),
        "rows_per_s": round(written / elapsed, 3) if elapsed else None,
    }


def main() -> None:
    parser = acli.argument_parser(description=__doc__, exit_codes={0: "predicted", 2: "invalid input"})
    parser.add_argument("--input", type=Path, required=True, help="rows with id and text")
    parser.add_argument("--output", type=Path, required=True, help="annotator-format predictions")
    parser.add_argument("--model", required=True, help="local GLiNER2 snapshot or model id")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="ont3 labels.json; its BIOES labels supply the type names given to the model",
    )
    parser.add_argument("--text-key", default="text", help="row key holding the text")
    parser.add_argument("--device", default="cpu", help="map_location for the model")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many rows")
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        acli.die(str(error), 2)
    acli.emit(receipt, acli.resolve_format(args))


if __name__ == "__main__":
    main()
