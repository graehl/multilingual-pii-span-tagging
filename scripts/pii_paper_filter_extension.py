#!/usr/bin/env python3
# acli: 1 complete
"""Predict a pinned paper privacy filter on an explicit evaluation population."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "agents"))
from pii_paper_pooled_eval import FILTER_BIASES, FILTERS, rows, sha, write

import acli


def main():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument("--model", choices=sorted(FILTERS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fine-grid",
        action="store_true",
        help="Add quarter-step O-logit biases over -4..4 to the saved bias grid",
    )
    parser.add_argument(
        "--score-cache",
        type=Path,
        help="Also write pii_eval's full-logit token score cache, so a further bias needs no encoder pass",
    )
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    biases = (
        sorted({float(b) for b in FILTER_BIASES} | {step / 4 for step in range(-16, 17)}, reverse=True)
        if args.fine_grid
        else FILTER_BIASES
    )
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    from huggingface_hub import snapshot_download
    from pii_eval import _predict_hf_tokcls_biases

    label, model_id, revision, schema, _ = FILTERS[args.model]
    checkpoint = snapshot_download(model_id, revision=revision)
    inputs = rows(args.input)
    if args.limit:
        inputs = inputs[: args.limit]
    if len({row["id"] for row in inputs}) != len(inputs):
        raise ValueError("duplicate input IDs")
    started = time.monotonic()
    predictions = _predict_hf_tokcls_biases(
        [row["text"] for row in inputs],
        lambda message: print(f"[predict] {message}", flush=True),
        model_id=checkpoint,
        o_logit_biases=biases,
        hf_windowing="character-conservative",
        score_cache_path=str(args.score_cache) if args.score_cache else None,
        score_cache_full_logits=bool(args.score_cache),
        record_ids=[row["id"] for row in inputs] if args.score_cache else None,
        record_datasets=[str(args.input)] * len(inputs) if args.score_cache else None,
    )
    points = {
        str(bias): [{"id": row["id"], "preds": spans} for row, (spans, _) in zip(inputs, values, strict=True)]
        for bias, values in predictions.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(
        args.output,
        {
            "schema": "pii-paper-filter-sweep/v1",
            "model": args.model,
            "label": label,
            "hf_id": model_id,
            "hf_revision": revision,
            "ontology_schema": schema,
            "decoder": "pii_eval._predict_hf_tokcls_biases; character-conservative; default unconstrained decoding",
            "input_sha256": sha(args.input),
            "smoke": bool(args.limit),
            "elapsed_seconds": time.monotonic() - started,
            "points": points,
        },
    )
    print(f"[complete] {args.model}: {len(inputs)} inputs, {len(points)} points", flush=True)


if __name__ == "__main__":
    main()
