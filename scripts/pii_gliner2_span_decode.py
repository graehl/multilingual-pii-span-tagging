#!/usr/bin/env python
"""Cache GLiNER2 raw span logits and decode legal non-overlapping span sets."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import types
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np

if __package__:
    from scripts.pii_eval import GLINER2_LABELS
else:
    from pii_eval import GLINER2_LABELS


class WordSplitter(Protocol):
    def __call__(self, text: str, lower: bool = True): ...


class SourceOffsetWordSplitter:
    """Preserve lowered tokens while mapping their boundaries to source text."""

    def __init__(self, splitter: WordSplitter):
        self.splitter = splitter

    def __call__(self, text: str, lower: bool = True):
        if not lower:
            yield from self.splitter(text, lower=False)
            return
        boundaries = [0]
        for char in text:
            boundaries.append(boundaries[-1] + len(char.lower()))
        if boundaries[-1] != len(text.lower()):
            raise ValueError("cannot align Unicode lowercase expansion to source text")
        for token, start, end in self.splitter(text, lower=True):
            source_start = bisect.bisect_right(boundaries, start) - 1
            source_end = bisect.bisect_left(boundaries, end)
            yield token, source_start, source_end


def cache(args: argparse.Namespace) -> None:
    import torch
    from gliner2 import GLiNER2

    rows = [json.loads(line) for line in Path(args.gold).read_text().splitlines()]
    model = GLiNER2.from_pretrained(args.model).to("cuda").eval()
    model.processor.word_splitter = SourceOffsetWordSplitter(model.processor.word_splitter)
    captured: list[dict] = []

    def capture_span_result(
        self,
        results,
        schema_name,
        task_type,
        embs,
        span_info,
        schema_tokens,
        text_tokens,
        text_len,
        original_text,
        start_mapping,
        end_mapping,
        threshold,
        metadata,
        cls_fields,
        include_confidence,
        include_spans,
    ):
        del task_type, text_tokens, threshold, metadata, cls_fields, include_confidence, include_spans
        if schema_name != "entities":
            raise ValueError(f"expected entity extraction, got {schema_name!r}")
        field_names = [
            schema_tokens[index + 1]
            for index in range(len(schema_tokens) - 1)
            if schema_tokens[index] in ("[E]", "[C]", "[R]")
        ]
        count_logits = self.count_pred(embs[0].unsqueeze(0))[0]
        predicted_count = int(count_logits.argmax().item())
        structure_projection = self.count_embed(embs[1:], max(predicted_count, 1))
        span_logits = torch.einsum("lkd,bpd->bplk", span_info["span_rep"], structure_projection)[
            0, :, -text_len:, :
        ]
        captured.append(
            {
                "labels": field_names,
                "count_logits": count_logits.float().cpu().numpy(),
                "predicted_count": predicted_count,
                "span_logits": span_logits.float().cpu().numpy(),
                "start_mapping": list(start_mapping),
                "end_mapping": list(end_mapping),
                "text": original_text,
            }
        )
        results[schema_name] = []

    model._extract_span_result = types.MethodType(capture_span_result, model)

    proposal_document = []
    proposal_start = []
    proposal_end = []
    proposal_label = []
    proposal_logit = []
    proposal_order = []
    count_logits = []
    predicted_counts = []
    for document_index, row in enumerate(rows):
        captured.clear()
        model.extract_entities(row["text"], GLINER2_LABELS)
        if len(captured) != 1:
            raise ValueError(f"{row['id']}: expected one entity score lattice, got {len(captured)}")
        item = captured[0]
        if item["labels"] != GLINER2_LABELS:
            raise ValueError(f"{row['id']}: GLiNER2 reordered the requested schema")
        count_logits.append(item["count_logits"])
        predicted_counts.append(item["predicted_count"])
        text_len = len(item["start_mapping"])
        order = 0
        for label_index in range(len(GLINER2_LABELS)):
            for token_start in range(text_len):
                for width in range(model.max_width):
                    token_end = token_start + width
                    if token_end >= text_len:
                        continue
                    char_start = int(item["start_mapping"][token_start])
                    char_end = int(item["end_mapping"][token_end])
                    if not row["text"][char_start:char_end].strip():
                        continue
                    proposal_document.append(document_index)
                    proposal_start.append(char_start)
                    proposal_end.append(char_end)
                    proposal_label.append(label_index)
                    proposal_logit.append(item["span_logits"][label_index, token_start, width])
                    proposal_order.append(order)
                    order += 1
        print(
            f"CACHE {document_index + 1}/{len(rows)} {row['id']}: "
            f"count={item['predicted_count']} proposals={order}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray([1], dtype=np.int32),
        calibration_family=np.asarray(["gliner2_raw_span_logit"]),
        labels=np.asarray(GLINER2_LABELS),
        document_ids=np.asarray([row["id"] for row in rows]),
        document_datasets=np.asarray([args.dataset] * len(rows)),
        count_logits=np.asarray(count_logits, dtype=np.float32),
        predicted_count=np.asarray(predicted_counts, dtype=np.int16),
        proposal_document=np.asarray(proposal_document, dtype=np.int32),
        proposal_start=np.asarray(proposal_start, dtype=np.int32),
        proposal_end=np.asarray(proposal_end, dtype=np.int32),
        proposal_label=np.asarray(proposal_label, dtype=np.uint16),
        proposal_logit=np.asarray(proposal_logit, dtype=np.float32),
        proposal_order=np.asarray(proposal_order, dtype=np.int32),
        max_width=np.asarray([model.max_width], dtype=np.int16),
    )
    print(f"WROTE {len(proposal_logit)} proposals for {len(rows)} documents -> {output}")


def overlaps(left: dict, right: dict) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def native_decode(proposals: list[dict]) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for proposal in proposals:
        if proposal["weight"] >= 0:
            by_label[proposal["label"]].append(proposal)
    selected = []
    for label_proposals in by_label.values():
        accepted = []
        for proposal in sorted(label_proposals, key=lambda item: (-item["logit"], item["order"])):
            if not any(overlaps(proposal, incumbent) for incumbent in accepted):
                accepted.append(proposal)
        selected.extend(accepted)
    return selected


def semimarkov_decode(proposals: list[dict]) -> list[dict]:
    candidates = [proposal for proposal in proposals if proposal["weight"] > 0]
    candidates.sort(key=lambda item: (item["end"], item["start"], item["label"], item["order"]))
    ends = [proposal["end"] for proposal in candidates]
    predecessors = [
        bisect.bisect_right(ends, proposal["start"], hi=index) - 1
        for index, proposal in enumerate(candidates)
    ]
    best = [0.0] * (len(candidates) + 1)
    take = [False] * len(candidates)
    for index, proposal in enumerate(candidates):
        with_proposal = proposal["weight"] + best[predecessors[index] + 1]
        without_proposal = best[index]
        if with_proposal > without_proposal:
            best[index + 1] = with_proposal
            take[index] = True
        else:
            best[index + 1] = without_proposal
    selected = []
    index = len(candidates) - 1
    while index >= 0:
        proposal = candidates[index]
        with_proposal = proposal["weight"] + best[predecessors[index] + 1]
        if take[index] and math.isclose(best[index + 1], with_proposal, rel_tol=0, abs_tol=1e-9):
            selected.append(proposal)
            index = predecessors[index]
        else:
            index -= 1
    return selected


def per_label_semimarkov_decode(proposals: list[dict]) -> list[dict]:
    """Replace GLiNER2's greedy suppression without changing overlap policy."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for proposal in proposals:
        by_label[proposal["label"]].append(proposal)
    selected = []
    for label_proposals in by_label.values():
        selected.extend(semimarkov_decode(label_proposals))
    return selected


def grouped_document_ranges(proposal_documents: np.ndarray, document_count: int) -> np.ndarray:
    if proposal_documents.ndim != 1:
        raise ValueError("proposal_document must be one-dimensional")
    if proposal_documents.size:
        if int(proposal_documents[0]) < 0 or int(proposal_documents[-1]) >= document_count:
            raise ValueError("proposal_document index is outside the document inventory")
        chunk_size = 1_000_000
        previous = int(proposal_documents[0])
        for start in range(0, proposal_documents.size, chunk_size):
            chunk = proposal_documents[start : start + chunk_size]
            if int(chunk[0]) < previous or np.any(chunk[1:] < chunk[:-1]):
                raise ValueError("proposal_document must be grouped in document order")
            previous = int(chunk[-1])
    return np.searchsorted(
        proposal_documents,
        np.arange(document_count + 1, dtype=proposal_documents.dtype),
        side="left",
    )


def decode_cache(data: dict[str, np.ndarray], decoder: str, threshold: float, count_gate: bool = True):
    if data["schema_version"].tolist() != [1]:
        raise ValueError("unsupported cache schema")
    labels = data["labels"].tolist()
    threshold_logit = math.log(threshold / (1 - threshold))
    proposal_documents = data["proposal_document"]
    document_ranges = grouped_document_ranges(proposal_documents, len(data["document_ids"]))
    rows = []
    for document_index, document_id in enumerate(data["document_ids"].tolist()):
        proposals = []
        if not count_gate or int(data["predicted_count"][document_index]) > 0:
            start = int(document_ranges[document_index])
            end = int(document_ranges[document_index + 1])
            logits = data["proposal_logit"][start:end]
            eligible = logits >= threshold_logit if decoder == "native" else logits > threshold_logit
            indices = np.flatnonzero(eligible) + start
            for proposal_index in indices.tolist():
                logit = float(data["proposal_logit"][proposal_index])
                proposals.append(
                    {
                        "start": int(data["proposal_start"][proposal_index]),
                        "end": int(data["proposal_end"][proposal_index]),
                        "label": labels[int(data["proposal_label"][proposal_index])],
                        "logit": logit,
                        "weight": logit - threshold_logit,
                        "order": int(data["proposal_order"][proposal_index]),
                    }
                )
        if decoder == "native":
            selected = native_decode(proposals)
        elif decoder == "exact-per-label":
            selected = per_label_semimarkov_decode(proposals)
        elif decoder in ("exact-global", "semimarkov"):
            selected = semimarkov_decode(proposals)
        else:
            raise ValueError(f"unknown decoder: {decoder}")
        predictions = [
            {"start": item["start"], "end": item["end"], "label": item["label"]}
            for item in sorted(selected, key=lambda item: (item["start"], item["end"], item["label"]))
        ]
        rows.append({"id": document_id, "preds": predictions})
    return rows


def decode(args: argparse.Namespace) -> None:
    with np.load(args.cache, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    rows = decode_cache(data, args.decoder, args.threshold, args.count_gate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(
        f"DECODE {args.decoder} threshold={args.threshold}: "
        f"{sum(len(row['preds']) for row in rows)} spans -> {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--model", required=True)
    cache_parser.add_argument("--gold", required=True)
    cache_parser.add_argument("--dataset", required=True)
    cache_parser.add_argument("--output", required=True)
    cache_parser.set_defaults(function=cache)
    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--cache", required=True)
    decode_parser.add_argument("--output", required=True)
    decode_parser.add_argument(
        "--decoder",
        choices=("native", "exact-per-label", "exact-global", "semimarkov"),
        required=True,
    )
    decode_parser.add_argument("--threshold", type=float, required=True)
    decode_parser.add_argument("--count-gate", action=argparse.BooleanOptionalAction, default=True)
    decode_parser.set_defaults(function=decode)
    args = parser.parse_args()
    if hasattr(args, "threshold") and not 0 < args.threshold < 1:
        parser.error("--threshold must be between zero and one")
    args.function(args)


if __name__ == "__main__":
    main()
