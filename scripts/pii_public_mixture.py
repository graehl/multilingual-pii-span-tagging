#!/usr/bin/env python3
"""Compile an O4-style training directory from public corpora alone.

O4 samples a human-gold branch and a base branch. The human-gold branch here
is rebuilt exactly as O4's was: public publisher train splits keep their
native labels through the committed candidate map, every window is complete
supervision with the corpus's unrepresented primary types masked, alternate
nested positives are ignored, and MAPA titles inside PERSON are unsupervised.
The base branch is whatever public rows the reader assembled (for example
`pii-reproduce.py assemble`), sampled uniformly per distinct input. O4's
private Luna, Terra and internal pools have no public counterpart here.

Rows that overlap an evaluation row are removed. The default screen is the
paper's overlap detector (overlaplib.overlapping: lexical and E5 nearest
neighbors, both cuts passed by one evaluation row) against every
--evaluation file; --screen exact keeps only the exact-text screen.

Language caps bound each language's share of all training draws (default
from language-caps.yaml: 20% for any language, 8% for Hindi). A capped
language's base-branch rows are scaled down and the freed mass goes to the
other base rows in proportion, so the gold branch and the branch split are
unchanged; a cap the gold branch alone exceeds is reported, not enforced.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import acli

EVIDENCE = ROOT / "research/pii/frontier/evidence"
MAP = EVIDENCE / "four-corpus-v1/candidate-map.json"
NEGATIVE_COVERAGE = EVIDENCE / "four-corpus-v1/negative-coverage-v1.json"
GOLD_CORPORA = ("openner-commercial-core", "mapa", "aqmar-openner", "wojood-sample")


def load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def corpus_rows(onboarded: Path, corpus: str, split: str, limit: int | None):
    """Rows of one split in intake order: sorted language shards, 1-based ids."""
    shards = sorted((onboarded / corpus / split).glob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no {split} shards for {corpus} under {onboarded}; run fetch and prepare")
    number = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                number += 1
                if limit is not None and number > limit:
                    return
                yield f"{corpus}-{split}:{number}", row


def o4_gold_members(path: Path) -> set[str]:
    """Human-gold training ids O4 kept after its semantic overlap screen."""
    import csv

    with path.open(encoding="utf-8", newline="") as stream:
        members = {
            row["training_id"] for row in csv.DictReader(stream) if row["sampling_branch"] == "human_gold"
        }
    if not members:
        raise ValueError(f"{path} lists no human-gold training rows")
    return members


def gold_branch(args, mapping, unknown_by_corpus, tokenizer, screened, split):
    mixture = load_module("compile_mixture", EVIDENCE / "four-corpus-v1/compile-mixture.py")
    titles = load_module("mapa_titles", EVIDENCE / "title-extents-v1/mapa-train-titles.py")
    windows, audit = [], Counter()
    members = o4_gold_members(args.o4_membership) if args.o4_membership and split == "train" else None
    for corpus in args.gold_corpus:
        for key, original in corpus_rows(args.onboarded, corpus, split, args.max_records_per_corpus):
            if members is not None and key not in members:
                audit[f"{corpus}:not_in_o4_membership"] += 1
                continue
            if screen_key(original["text"]) in screened:
                audit[f"{corpus}:evaluation_text_removed"] += 1
                continue
            spans = []
            for span in original["spans"]:
                node = mapping["source_labels"][corpus][span["source_label"]]
                if mapping["v1_to_v2"][node]["accepted"] != ["O"]:
                    spans.append([span["start"], span["end"], node])
            row = {
                "id": key,
                "text": original["text"],
                "lang": original["lang"],
                "spans": spans,
                "label_space": "v1",
                "supervision": "annotated_spans_only",
                "src": corpus,
                "sampling_branch": "human_gold",
                "sampling_pool": corpus,
                "sampling_source_id": key,
                "sampling_source_sha256": hashlib.sha256(original["text"].encode()).hexdigest(),
                "sampling_weight": 1.0,
            }
            for window in mixture.gold_windows(row, tokenizer, "full-corpus", unknown_by_corpus[corpus]):
                if corpus == "mapa":
                    # Same transform O4 applied after compiling; offsets are window-relative.
                    metadata = original["metadata"]
                    if window["sampling_source_start"] == 0 and window["text"] == original["text"]:
                        window = titles.split_row(window, metadata["tokens"], metadata["fine_grained"], audit)
                    else:
                        audit["mapa_windowed_rows_titles_unchanged"] += 1
                windows.append(window)
            audit[f"{corpus}:rows"] += 1
    return windows, audit


def base_branch(args, tokenizer, screened):
    from pii_encoder_train import window_records

    path = args.base / "train.jsonl"
    rows = window_records(
        str(path), 900, context_field="document_context", tokenizer=tokenizer, max_tokens=512
    )
    kept = [row for row in rows if screen_key(row["text"]) not in screened]
    primary = set(json.loads(MAP.read_text())["ontology"]["primary_types"])
    dropped = Counter()
    for index, row in enumerate(kept):
        if row.get("supervision") != "annotated_spans_only":
            raise ValueError(
                f"base row {row['id']} is not annotated_spans_only; its negatives are unverified"
            )
        # Older unified-tagset labels have no Ont3 primary type. Removing a
        # positive from annotated-spans-only supervision creates no negative.
        dropped.update(span[2] for span in row["spans"] if span[2] not in primary)
        row["spans"] = [span for span in row["spans"] if span[2] in primary]
        row["sampling_branch"] = "base"
        row["sampling_weight"] = 1.0
        row["id"] = f"base-window-{index}"
    natural_removed = len(rows) - len(kept)
    annotated = Counter()
    for path in getattr(args, "annotated", None) or ():
        added = window_records(
            str(path), 900, context_field="document_context", tokenizer=tokenizer, max_tokens=512
        )
        for row in added:
            if screen_key(row["text"]) in screened:
                annotated["evaluation_text_removed"] += 1
                continue
            foreign = {span[2] for span in row["spans"] if span[2] not in primary}
            if foreign:
                # Teacher rows may be complete supervision; dropping a positive
                # would teach a false negative, so refuse instead.
                raise ValueError(f"{path}: annotated row carries non-Ont3 labels {sorted(foreign)}")
            row["sampling_branch"] = "base"
            row["sampling_weight"] = 1.0
            row["id"] = f"annotated-window-{len(kept)}"
            kept.append(row)
            annotated[f"{path.name}:{row.get('supervision')}"] += 1
    return kept, {
        "base_windows": len(rows),
        "base_evaluation_text_removed": natural_removed,
        "base_non_ont3_spans_dropped": dict(dropped),
        "annotated_windows": dict(annotated),
    }


def weighted(rows, share):
    from pii_annotation_sampling import share_annotation_sampling_mass

    weights, sharing = share_annotation_sampling_mass(rows, [1.0] * len(rows))
    total = math.fsum(weights)
    for row, weight in zip(rows, weights, strict=True):
        row["sampling_weight"] = weight / total * share
    return sharing


def screen_description(args) -> str:
    if not args.evaluation:
        return "none"
    if args.screen == "overlap":
        return "paper overlap detector against the evaluation files (exact matches included)"
    return "exact normalized text against the evaluation files"


def evaluation_keys(args) -> set[str]:
    """Exact normalized texts of every evaluation file, the screen that always applies."""
    return {
        screen_key(json.loads(line)["text"])
        for path in args.evaluation or ()
        for line in path.open(encoding="utf-8")
        if line.strip()
    }


def overlap_screen(args, groups: dict[str, list[dict]]) -> tuple[set[str], dict]:
    """Screen keys of training texts that overlap an evaluation row, and a per-group audit."""
    from overlaplib import DETECTOR_VERSION, overlapping, text_id

    texts, owners = {}, {}
    for group, rows in groups.items():
        for row in rows:
            identifier = text_id(row["text"])
            texts.setdefault(identifier, (row["text"], row["lang"]))
            owners.setdefault(identifier, set()).add(group)
    evaluation = [(f"evaluation-{index}", path) for index, path in enumerate(args.evaluation)]
    found = overlapping(
        texts, evaluation, args.out.parent / f".{args.out.name}-overlap", device=args.screen_device
    )
    removed = Counter(group for identifier in found for group in owners[identifier])
    audit = {
        "detector": "paper overlap rule: nearest-three lexical chrF F1 >= 0.30 and E5 cosine >= 0.875 "
        "on the same evaluation row",
        "detector_version": DETECTOR_VERSION,
        "evaluation": {name: sha256_file(path) for name, path in evaluation},
        "distinct_texts_screened": len(texts),
        "distinct_texts_removed": len(found),
        "removed_by_group": dict(removed),
        "examples": [
            {"text": texts[identifier][0][:160], **match} for identifier, match in list(found.items())[:20]
        ],
        "expected_collateral": "about 0.2% of ordinary text, about 5% of template-heavy text (calibration "
        "on disjoint validation splits, 2026-10-03)",
    }
    return {screen_key(texts[identifier][0]) for identifier in found}, audit


def load_language_caps(args) -> dict | None:
    """Per-language maximum share of training draws: file default and overrides, then CLI overrides."""
    import yaml

    if args.no_language_caps:
        return None
    config = yaml.safe_load(args.language_caps.read_text())
    caps = {"default": float(config["default"]), "languages": dict(config.get("languages") or {})}
    for value in args.language_cap or ():
        language, separator, share = value.partition("=")
        if not separator:
            raise ValueError(f"--language-cap needs LANG=SHARE: {value!r}")
        caps["languages"][language] = float(share)
    for share in (caps["default"], *caps["languages"].values()):
        if not 0 < share <= 1:
            raise ValueError(f"language cap must be in (0, 1]: {share}")
    return caps


def apply_language_caps(rows: list[dict], caps: dict) -> dict:
    """Scale capped languages' base rows down, giving the freed mass to the other base rows.

    Shares are of all training draws. Gold rows and the base branch's total
    mass are unchanged; a language whose gold rows alone exceed its cap keeps
    its base rows and is reported.
    """
    from trainlib_mix import apply_caps

    report = apply_caps(
        rows,
        {"default": caps["default"], "groups": caps["languages"]},
        group=lambda row: row["lang"],
        movable=lambda row: row["sampling_branch"] == "base",
    )
    return {
        "default": report["default"],
        "languages": report["groups"],
        "capped": report["capped"],
        "unattainable_from_gold": report["unattainable_from_fixed"],
    }


def write_language_round(path: Path, rows, floor: float) -> dict:
    """Declare this compiled mix's core languages for the trainer's coverage gate.

    O4's recorded round names the 35 languages of its private pools; public data
    covers fewer. The trainer checks only declared languages, so declare those
    whose realized sampling share reaches the floor.
    """
    share = Counter()
    for row in rows:
        share[row["lang"]] += row["sampling_weight"]
    total = math.fsum(share.values())
    core = sorted(
        (lang for lang, mass in share.items() if mass / total >= floor), key=lambda lang: -share[lang]
    )
    lines = [
        "schema_version: 1",
        "round_id: public_compiled_mix",
        "membership: exact",
        "languages:",
        *(f'  - code: "{lang}"' for lang in core),
        "training_mix:",
        f"  minimum_core_language_share: {floor}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return {
        "core": core,
        "floor": floor,
        "shares": {lang: mass / total for lang, mass in share.most_common()},
    }


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def base_validation(args, tokenizer, primary) -> list[dict]:
    """Assembler validation rows in the Ont3 label space.

    The O4 recipe selects checkpoints by span F1, which needs hard labels in
    the head's own ontology; native-node gold rows cannot serve.
    """
    from pii_encoder_train import window_records

    validation = window_records(
        str(args.base / "val.jsonl"),
        900,
        context_field="document_context",
        tokenizer=tokenizer,
        max_tokens=512,
    )[: args.max_validation_windows]
    for row in validation:
        row["spans"] = [span for span in row["spans"] if span[2] in primary]
        # Every remaining label is an Ont3 primary type: hard new-space targets.
        row["label_space"] = "v2"
    return validation


def compile_root(args) -> dict:
    """Base rows alone in the native Ont3 label space, for a fresh root stage."""
    from transformers import AutoTokenizer

    if args.base is None:
        raise ValueError("--stage root needs --base rows labeled with Ont3 primary types")
    mapping = json.loads(MAP.read_text())
    primary = mapping["ontology"]["primary_types"]
    screened = evaluation_keys(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows, audit = base_branch(args, tokenizer, screened)
    foreign = Counter(span[2] for row in rows for span in row["spans"] if span[2] not in primary)
    if foreign:
        raise ValueError(f"base rows carry labels outside the Ont3 primary types: {dict(foreign)}")
    validation = base_validation(args, tokenizer, primary)
    overlap = None
    if args.screen == "overlap":
        keys = getattr(args, "overlap_keys", None)
        if keys is None:
            keys, overlap = overlap_screen(args, {"base": rows, "validation": validation})
        rows, validation = (
            [row for row in part if screen_key(row["text"]) not in keys] for part in (rows, validation)
        )
    sharing = weighted(rows, 1.0)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "train.jsonl", rows)
    language_round = write_language_round(args.out / "language-round.yaml", rows, args.language_floor)
    write_jsonl(args.out / "val.jsonl", validation)
    (args.out / "labels.json").write_text(json.dumps({"labels": primary}, indent=2) + "\n")
    receipt = {
        "schema": "pii-public-root-v1",
        "rows": len(rows),
        "sources": dict(Counter(row.get("src") for row in rows)),
        "labels_used": dict(Counter(span[2] for row in rows for span in row["spans"])),
        "validation_windows": len(validation),
        "language_round": language_round,
        "validation": "assembler row split; not document-separated",
        "screen": screen_description(args),
        "overlap_screen": overlap,
        "audit": audit,
        "annotation_sharing": sharing,
        "inputs": {
            "base": sha256_file(args.base / "train.jsonl"),
            "evaluation": [sha256_file(path) for path in args.evaluation or ()],
        },
        "outputs": {
            name: sha256_file(args.out / name) for name in ("train.jsonl", "val.jsonl", "labels.json")
        },
        "tokenizer": args.tokenizer,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"ok": True, "out": str(args.out.resolve()), **receipt}


def compile_mixture(args) -> dict:
    from transformers import AutoTokenizer

    if not 0 < args.gold_share <= 1:
        raise ValueError("--gold-share must be in (0, 1]")
    if args.base is None and args.gold_share != 1:
        raise ValueError("a gold share below one needs --base rows for the remaining mass")
    mapping = json.loads(MAP.read_text())
    primary = set(mapping["ontology"]["primary_types"])
    negative = json.loads(NEGATIVE_COVERAGE.read_text())["negative_types_by_corpus"]
    unknown_by_corpus = {corpus: sorted(primary - set(negative[corpus])) for corpus in args.gold_corpus}
    if args.screen == "overlap" and not args.evaluation:
        raise ValueError("the overlap screen needs at least one --evaluation file")
    screened = evaluation_keys(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    gold, audit = gold_branch(args, mapping, unknown_by_corpus, tokenizer, screened, "train")
    base, base_audit = base_branch(args, tokenizer, screened) if args.base is not None else ([], None)
    if args.base is not None:
        validation = base_validation(args, tokenizer, mapping["ontology"]["primary_types"])
        validation_audit = {"source": "base assembler validation rows, Ont3 labels; not document-separated"}
    else:
        validation, validation_audit = gold_branch(
            args, mapping, unknown_by_corpus, tokenizer, screened, "validation"
        )
        validation = validation[: args.max_validation_windows]
        validation_audit["source"] = (
            "gold validation splits in native labels; unsuitable for span-F1 selection"
        )
    overlap = None
    if args.screen == "overlap":
        keys, overlap = overlap_screen(args, {"human_gold": gold, "base": base, "validation": validation})
        gold, base, validation = (
            [row for row in rows if screen_key(row["text"]) not in keys] for rows in (gold, base, validation)
        )
        args.overlap_keys = keys
    sharing = {"human_gold": weighted(gold, args.gold_share)}
    rows = list(gold)
    if args.base is not None:
        sharing["base"] = weighted(base, 1 - args.gold_share)
        rows.extend(base)
    caps = load_language_caps(args)
    language_caps = apply_language_caps(rows, caps) if caps and args.base is not None else None
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "train.jsonl", rows)
    language_round = write_language_round(args.out / "language-round.yaml", rows, args.language_floor)
    write_jsonl(args.out / "val.jsonl", validation)
    (args.out / "labels.json").write_text(
        json.dumps({"labels": sorted(mapping["v1_to_v2"])}, indent=2) + "\n"
    )
    (args.out / "mapping.json").write_bytes(MAP.read_bytes())
    branch = Counter()
    supported = set()
    for row in rows:
        branch[row["sampling_branch"]] += row["sampling_weight"]
        for span in row["spans"]:
            # A native node supervises every Ont3 type it may denote.
            supported.update(tag for tag in mapping["v1_to_v2"][span[2]]["accepted"] if tag != "O")
    receipt = {
        "schema": "pii-public-mixture-v1",
        "supported_types": sorted(supported),
        "gold_corpora": list(args.gold_corpus),
        "gold_share": args.gold_share,
        "branch_probability": dict(branch),
        "rows": dict(Counter(row["sampling_branch"] for row in rows)),
        "validation_windows": len(validation),
        "language_round": language_round,
        "max_records_per_corpus": args.max_records_per_corpus,
        "screen": screen_description(args),
        "overlap_screen": overlap,
        "language_caps": language_caps,
        "o4_membership": sha256_file(args.o4_membership) if args.o4_membership else None,
        "audit": dict(audit),
        "base_audit": base_audit,
        "validation_audit": dict(validation_audit),
        "annotation_sharing": sharing,
        "inputs": {
            "candidate_map": sha256_file(MAP),
            "negative_coverage": sha256_file(NEGATIVE_COVERAGE),
            "evaluation": [sha256_file(path) for path in args.evaluation or ()],
            "base": sha256_file(args.base / "train.jsonl") if args.base else None,
        },
        "outputs": {
            name: sha256_file(args.out / name)
            for name in ("train.jsonl", "val.jsonl", "labels.json", "mapping.json")
        },
        "tokenizer": args.tokenizer,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.base is not None:
        # A fresh O4-recipe fit first needs an Ont3 affine head; the trainer
        # creates one only from natively labeled rows, so ship them alongside.
        import argparse

        root = compile_root(argparse.Namespace(**{**vars(args), "out": args.out / "root"}))
        receipt["root"] = {"out": root["out"], "rows": root["rows"], "outputs": root["outputs"]}
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"ok": True, "out": str(args.out.resolve()), **receipt}


def build_parser():
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument("--onboarded", type=Path, help="Prepare output root for the gold corpora (o4 stage)")
    parser.add_argument(
        "--stage",
        choices=("o4", "root"),
        default="o4",
        help="o4: gold plus base in O4's mapped label space; root: base only, native Ont3 labels",
    )
    parser.add_argument(
        "--gold-corpus", action="append", help="Repeatable; default is the four paper corpora"
    )
    parser.add_argument(
        "--base", type=Path, help="Assembled corpus directory whose train.jsonl forms the base branch"
    )
    parser.add_argument(
        "--gold-share", type=float, default=0.5, help="Human-gold branch probability (O4: 0.5)"
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        action="append",
        help="Evaluation JSONL (id, text, lang) to keep out of training, e.g. the rebuilt human gold and "
        "the Ont3 inputs; repeatable",
    )
    parser.add_argument(
        "--screen",
        choices=("overlap", "exact"),
        default="overlap",
        help="overlap: the paper's overlap detector (default); exact: exact normalized text only",
    )
    parser.add_argument("--screen-device", default="cuda", help="Embedding device for the overlap screen")
    parser.add_argument(
        "--language-caps",
        type=Path,
        default=ROOT / "research/pii/frontier/software/language-caps.yaml",
        help="YAML: default maximum share of training draws for any language, plus per-language overrides",
    )
    parser.add_argument(
        "--language-cap",
        action="append",
        metavar="LANG=SHARE",
        help="Override one language's cap; repeatable",
    )
    parser.add_argument(
        "--no-language-caps", action="store_true", help="Sample languages by row weight alone"
    )
    parser.add_argument(
        "--o4-membership",
        type=Path,
        help="records/o4-training-membership.csv: keep exactly O4's screened human-gold rows",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        action="append",
        help="Teacher-annotated JSONL in Ont3 labels (e.g. annotate output) added to the base branch; repeatable",
    )
    parser.add_argument("--tokenizer", default="FacebookAI/xlm-roberta-large")
    parser.add_argument(
        "--max-records-per-corpus", type=int, help="First N rows of each corpus split (smoke)"
    )
    parser.add_argument("--max-validation-windows", type=int, default=2000)
    parser.add_argument(
        "--language-floor",
        type=float,
        default=0.005,
        help="Declare languages at or above this sampling share in language-round.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    acli.add_standard_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    acli.maybe_complete(parser)
    args = parser.parse_args()
    args.gold_corpus = tuple(args.gold_corpus or GOLD_CORPORA)
    if args.stage == "o4" and args.onboarded is None:
        parser.error("--stage o4 needs --onboarded")
    try:
        result = compile_root(args) if args.stage == "root" else compile_mixture(args)
    except (OSError, ValueError, KeyError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
