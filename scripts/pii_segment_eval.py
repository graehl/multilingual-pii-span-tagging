#!/usr/bin/env python
"""Segment long docs for LLM labeling, then merge segment preds back to
doc offsets (topics/pii-adaptation.md gate 3).

Entity/PII detection is a LOCAL task: it needs no whole-document
context. Feeding a whole 9k-token doc as one label request (as the
first gate-3 run did) is both a copying-scale failure and pointlessly
hard — the model under-labels dense text and returns unparseable JSON.
The fix is to label short segments (the regime where the out-of-box
labeler scored 0.897 on zh sentences and 0.956 on MEDDOCAN) and stitch
spans back by offset.

Offsets are preserved exactly: a segment's text is a verbatim slice
text[off:end] (never stripped), so a segment-relative span [a,b] maps
to the doc span [off+a, off+b]. The doc id and offset ride in the
segment id as "<docid>|<off>" so the labeler (which only preserves
`id`) carries them through untouched.

Usage:
  pii_segment_eval.py split <gold.jsonl> <out-segments.jsonl> [max_chars=600] [lang=en]
  pii_segment_eval.py merge <segment-preds.jsonl> <out-doc-preds.jsonl>
"""

import json
import sys


def segments(text, max_chars, protected_spans=(), *, tokenizer=None, max_tokens=None):
    """Yield (offset, seg_text) covering the whole doc, cutting at the
    last newline/sentence boundary before max_chars; verbatim slices so
    offsets stay exact. A boundary inside a protected span is extended to
    that span's end, even when the resulting segment exceeds max_chars.

    With a tokenizer and max_tokens, also fit every slice within the encoded
    budget (including special tokens). Indivisible over-budget spans fail;
    labels are never clipped to make a window fit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if (tokenizer is None) != (max_tokens is None) or (max_tokens is not None and max_tokens <= 0):
        raise ValueError("tokenizer and positive max_tokens must be supplied together")
    i, n = 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            cut = max(text.rfind("\n", i, end), text.rfind(". ", i, end) + 1 if ". " in text[i:end] else -1)
            if cut > i:
                end = cut
        while True:
            extended_end = max(
                (span_end for span_start, span_end, *_ in protected_spans if span_start < end < span_end),
                default=end,
            )
            if extended_end == end:
                break
            end = extended_end
        if tokenizer is not None:
            while True:
                encoded = tokenizer(
                    text[i:end],
                    truncation=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                if len(encoded["input_ids"]) <= max_tokens:
                    break
                budget = max_tokens - sum(encoded["special_tokens_mask"])
                offsets = [
                    pair
                    for pair, special in zip(
                        encoded["offset_mapping"], encoded["special_tokens_mask"], strict=True
                    )
                    if not special
                ]
                if budget <= 0:
                    raise ValueError("max_tokens leaves no room after special tokens")
                # Start of the first excluded token also handles multiple byte
                # pieces sharing one Unicode character. Retokenize each prefix:
                # cutting a string can change its last tokenization boundary.
                cut = min(end - 1, i + offsets[budget][0])
                while True:
                    safe_cut = min((a for a, b, *_ in protected_spans if a < cut < b), default=cut)
                    if safe_cut == cut:
                        break
                    cut = safe_cut
                if cut <= i:
                    # Check the smallest indivisible prefix before declaring
                    # failure; its isolated encoding can differ from the suffix.
                    cut = i + 1
                    while True:
                        extended = max((b for a, b, *_ in protected_spans if a < cut < b), default=cut)
                        if extended == cut:
                            break
                        cut = extended
                    prefix = tokenizer(text[i:cut], truncation=False)
                    if len(prefix["input_ids"]) > max_tokens:
                        raise ValueError(
                            f"indivisible span/character [{i}, {cut}) exceeds max_tokens={max_tokens}"
                        )
                end = cut
        seg = text[i:end]
        if seg.strip():
            yield i, seg
        i = end


def cmd_split(gold_path, out_path, max_chars=600, lang="en"):
    max_chars = int(max_chars)
    n_docs = n_seg = 0
    with open(out_path, "w") as out:
        for line in open(gold_path):
            r = json.loads(line)
            n_docs += 1
            for off, seg in segments(r["text"], max_chars, r.get("spans", ())):
                out.write(
                    json.dumps({"id": f"{r['id']}|{off}", "text": seg, "lang": lang}, ensure_ascii=False)
                    + "\n"
                )
                n_seg += 1
    print(f"split {n_docs} docs -> {n_seg} segments (max_chars={max_chars}) -> {out_path}")


def cmd_split_gold(gold_path, out_path, max_chars=600):
    """Segment a gold file into the unit the tagger is trained and served on.

    Evaluating a 3,500-character document scores an assembly of six or more
    windows and lets boundary effects accumulate, which is not how the model
    is trained. This projects gold spans into span-safe segments so the eval
    unit matches the training unit. Segments carrying no spans are KEPT: they
    are the false-positive surface, and dropping them would hide exactly the
    sparse regime that natural text is full of.
    """
    max_chars = int(max_chars)
    n_docs = n_seg = n_empty = n_spans = 0
    with open(out_path, "w") as out:
        for line in open(gold_path):
            r = json.loads(line)
            n_docs += 1
            spans = r.get("spans", [])
            protected = [(int(s["start"]), int(s["end"])) for s in spans]
            for off, seg in segments(r["text"], max_chars, protected):
                end = off + len(seg)
                kept = []
                for s in spans:
                    start, stop = int(s["start"]), int(s["end"])
                    if start >= off and stop <= end:
                        moved = dict(s)
                        moved["start"] = start - off
                        moved["end"] = stop - off
                        kept.append(moved)
                    elif start < end and stop > off:
                        raise AssertionError(
                            f"{r['id']}: span [{start}, {stop}) straddles segment [{off}, {end})"
                        )
                out.write(
                    json.dumps(
                        {
                            "id": f"{r['id']}|{off}",
                            "document_id": r.get("document_id", r["id"]),
                            "text": seg,
                            "spans": kept,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_seg += 1
                n_spans += len(kept)
                n_empty += not kept
    print(
        f"split-gold {n_docs} docs -> {n_seg} segments ({n_empty} span-free), "
        f"{n_spans} spans (max_chars={max_chars}) -> {out_path}"
    )


def cmd_merge(seg_pred_path, out_path):
    by_doc = {}
    for line in open(seg_pred_path):
        r = json.loads(line)
        docid, off = r["id"].rsplit("|", 1)
        off = int(off)
        dst = by_doc.setdefault(docid, [])
        for p in r["preds"]:
            q = dict(p)
            q["start"] = p["start"] + off
            q["end"] = p["end"] + off
            dst.append(q)
    with open(out_path, "w") as out:
        for docid, preds in by_doc.items():
            preds.sort(key=lambda p: (p["start"], p["end"]))
            out.write(json.dumps({"id": docid, "preds": preds}, ensure_ascii=False) + "\n")
    print(f"merged {sum(len(v) for v in by_doc.values())} spans over {len(by_doc)} docs -> {out_path}")


if __name__ == "__main__":
    {"split": cmd_split, "split-gold": cmd_split_gold, "merge": cmd_merge}[sys.argv[1]](*sys.argv[2:])
