"""Leave MAPA titles unsupervised in a training mixture (user-directed, 2026-09-27).

MAPA's coarse PERSON spans include job titles, offices and roles ("julkisasiamies
Sharpston"), while the approved Ont3 rule makes a title a separate
demographic_attribute. MAPA training rows therefore taught the model to put
titles inside person_name. This transform is the objective's counterpart of the
title-extents scoring policy: each ROLE or PROFESSION run that MAPA's fine layer
marks inside a PERSON span becomes an ignored span (no loss for any labeling),
and the PERSON span keeps only its name part, whose boundaries stay supervised.
Honorifics (fine TITLE) stay inside PERSON, as the rule says.

The fine layer is joined from data/pii-onboarded/mapa/train by (language, text).
Rows that do not join are unchanged and counted. Sampling weights are unchanged.

usage: mapa-train-titles.py MIXTURE_DIR OUTPUT_DIR
"""

import glob
import gzip
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

COPIED = ("val.jsonl", "evaluation.jsonl", "labels.json", "mapping.json")
TITLE_KINDS = {"ROLE", "PROFESSION"}
EDGE = " \t\n,;:"


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def fine_layer():
    fine = {}
    for path in sorted(glob.glob("data/pii-onboarded/mapa/train/*.jsonl.gz")):
        for line in gzip.open(path, "rt", encoding="utf-8"):
            row = json.loads(line)
            if " ".join(row["metadata"]["tokens"]) != row["text"]:
                raise SystemExit(f"{path}: {row['id']} is not a single-space token join")
            fine.setdefault(
                (row["lang"], row["text"]), (row["metadata"]["tokens"], row["metadata"]["fine_grained"])
            )
    return fine


def title_runs(tokens, tags, start, end):
    """Character extents of ROLE/PROFESSION token runs inside [start, end)."""
    runs, position, current = [], 0, None
    for token, tag in zip(tokens, tags, strict=True):
        a, b = position, position + len(token)
        position = b + 1
        if tag[2:] in TITLE_KINDS and start <= a and b <= end:
            if current and tag.startswith("I-"):
                current[1] = b
            else:
                current = [a, b]
                runs.append(current)
        else:
            current = None
    return runs


def trimmed(text, a, b):
    while a < b and text[a] in EDGE:
        a += 1
    while b > a and text[b - 1] in EDGE:
        b -= 1
    return (a, b) if a < b else None


def split_row(row, tokens, tags, counts):
    spans, ignored = [], list(row.get("ignored_spans") or [])
    for start, end, label in row["spans"]:
        runs = title_runs(tokens, tags, start, end) if label.endswith("__PERSON") else []
        if not runs:
            spans.append([start, end, label])
            continue
        counts["person_spans_split"] += 1
        cursor = start
        for a, b in runs:
            piece = trimmed(row["text"], cursor, a)
            if piece:
                spans.append([*piece, label])
            ignored.append([a, b, "mapa_title"])
            cursor = b
        piece = trimmed(row["text"], cursor, end)
        if piece:
            spans.append([*piece, label])
        else:
            counts["person_spans_title_only"] += 1
    row["spans"] = sorted(spans)
    # The trainer reads ignored_spans as a list; write the key only when non-empty.
    if ignored:
        row["ignored_spans"] = sorted(ignored)
    else:
        row.pop("ignored_spans", None)
    return row


def main():
    mixture, output = Path(sys.argv[1]), Path(sys.argv[2])
    fine = fine_layer()
    counts = Counter()
    output.mkdir(parents=True, exist_ok=True)
    if any(not p.name.endswith((".meta.md", ".meta.json")) for p in output.iterdir()):
        raise SystemExit(f"output already has content: {output}")
    with (
        (mixture / "train.jsonl").open(encoding="utf-8") as source,
        (output / "train.jsonl").open("x", encoding="utf-8") as sink,
    ):
        for line in source:
            row = json.loads(line)
            if row.get("src") == "mapa":
                counts["mapa_rows"] += 1
                key = (row["lang"], row["text"])
                if key in fine:
                    row = split_row(row, *fine[key], counts)
                else:
                    counts["mapa_rows_unjoined"] += 1
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for name in COPIED:
        shutil.copyfile(mixture / name, output / name)
    receipt = {
        "schema": "pii-mapa-title-ignore/v1",
        "policy": "MAPA fine ROLE/PROFESSION runs inside PERSON become ignored spans; PERSON keeps the name part",
        "input": identity(mixture / "train.jsonl"),
        "outputs": {name: identity(output / name) for name in ("train.jsonl", *COPIED)},
        "counts": dict(counts),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt["counts"]))


if __name__ == "__main__":
    main()
