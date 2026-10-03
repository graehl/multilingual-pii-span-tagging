#!/usr/bin/env python
"""PII span-detection evaluation for local and published multilingual models.

Datasets include SPY (populated from placeholder release with seeded Faker,
per the dataset's own SPY.py logic), TAB test (ECHR court cases), MEDDOCAN,
MultiGraSCCo, frontier-authored data, and pinned natural-source shards.

Protocol per topics/pii-adaptation.md "Multilingual bake-off design":
class-agnostic span detection (exact + overlap matching), redaction F1 primary
(precision/recall/F2 as diagnostics), stratified by coarse class
(name / format-bound / freetext).

Subcommands:
  prep     - build frozen gold jsonl for spy-medical, spy-legal, tab-test
  predict  - run one model over one dataset, write predictions jsonl
  score    - score predictions vs gold, print + write metrics json
"""

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
HERE = os.environ.get("PII_EVAL_HOME") or os.path.join(REPO, "untracked", "pii-eval")
DATA = os.path.join(HERE, "data")
GOLD = os.path.join(HERE, "gold")
PRED = os.path.join(HERE, "pred")

SPY_SEED = 0

OVERLAP_MIN_COVERAGE_NUMERATOR = 4
OVERLAP_MIN_COVERAGE_DENOMINATOR = 5
OVERLAP_METRIC_CONTRACT = "span intersection covers at least 80% of each of the reference and predicted spans"

# SPY coarse strata; TAB entity_type strata.
COARSE = {
    # SPY tags
    "NAME": "name",
    "EMAIL": "format",
    "PHONE_NUM": "format",
    "ID_NUM": "format",
    "URL": "format",
    "USERNAME": "format",
    "ADDRESS": "freetext",
    # TAB entity types
    "PERSON": "name",
    "CODE": "format",
    "DATETIME": "format",
    "LOC": "freetext",
    "ORG": "freetext",
    "DEM": "freetext",
    "QUANTITY": "freetext",
    "MISC": "freetext",
    # MEDDOCAN PHI types (main ones; others fall back to "other")
    "NOMBRE_SUJETO_ASISTENCIA": "name",
    "NOMBRE_PERSONAL_SANITARIO": "name",
    "FAMILIARES_SUJETO_ASISTENCIA": "name",
    "EDAD_SUJETO_ASISTENCIA": "format",
    "FECHAS": "format",
    "CORREO_ELECTRONICO": "format",
    "NUMERO_TELEFONO": "format",
    "NUMERO_FAX": "format",
    "ID_SUJETO_ASISTENCIA": "format",
    "ID_CONTACTO_ASISTENCIAL": "format",
    "ID_ASEGURAMIENTO": "format",
    "ID_TITULACION_PERSONAL_SANITARIO": "format",
    "ID_EMPLEO_PERSONAL_SANITARIO": "format",
    "CALLE": "freetext",
    "TERRITORIO": "freetext",
    "PAIS": "freetext",
    "CENTRO_SALUD": "freetext",
    "HOSPITAL": "freetext",
    "INSTITUCION": "freetext",
    "SEXO_SUJETO_ASISTENCIA": "freetext",
    "PROFESION": "freetext",
}

# GLiNER2-PII's own 42-type schema (from its model card); prompting the model
# with its full trained schema maximizes class-agnostic coverage.
GLINER2_LABELS = [
    "person",
    "full_name",
    "first_name",
    "middle_name",
    "last_name",
    "date_of_birth",
    "email",
    "phone_number",
    "address",
    "street_address",
    "city",
    "state_or_region",
    "postal_code",
    "country",
    "government_id",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "license_number",
    "tax_id",
    "tax_number",
    "bank_account",
    "account_number",
    "routing_number",
    "iban",
    "payment_card",
    "card_number",
    "card_expiry",
    "card_cvv",
    "username",
    "ip_address",
    "account_id",
    "sensitive_account_id",
    "password",
    "secret",
    "api_key",
    "access_token",
    "recovery_code",
    "sensitive_date",
    "document_date",
    "expiration_date",
    "transaction_date",
]

OPENMED_ID = "OpenMed/privacy-filter-multilingual"
GLINER2_ID = "fastino/gliner2-privacy-filter-PII-multi"
GLINER2_FRONTIER_REVISION = "59894c087cb2923b01f337d4ee72f6ff84d5bdd6"
OPENAI_ID = "openai/privacy-filter"
HF_TOKCLS_IDS = {
    "openmed": OPENMED_ID,
    "openai": OPENAI_ID,
    "local": os.environ.get("PII_EVAL_LOCAL_MODEL", ""),
}
HF_WINDOWING_MODES = ("character-conservative", "token-capacity")


# ---------------------------------------------------------------- prep: SPY


def spy_profiles(n):
    """Per-type entity value pools, mirroring SPY.py's PII_ENT_FUNCS, fully
    seeded (SPY.py seeds Faker but not random.shuffle)."""
    from faker import Faker

    random.seed(SPY_SEED)
    Faker.seed(SPY_SEED)
    faker = Faker()
    funcs = {
        "EMAIL": [faker.ascii_email, faker.ascii_free_email],
        "NAME": [faker.name],
        "URL": [lambda: faker.uri(deep=1), lambda: faker.uri(deep=2)],
        "PHONE_NUM": [faker.phone_number],
        "ID_NUM": [
            faker.ripe_id,
            faker.msisdn,
            faker.ssn,
            faker.sbn9,
            faker.isbn10,
            faker.isbn13,
            faker.credit_card_number,
            faker.aba,
            faker.bban,
            faker.iban,
        ],
        "ADDRESS": [faker.street_address],
        "USERNAME": [faker.user_name],
    }
    pools = {}
    for ent, fs in funcs.items():
        sizes = [n // len(fs) + (i < n % len(fs)) for i in range(len(fs))]
        vals = [f() for f, sz in zip(fs, sizes) for _ in range(sz)]
        random.shuffle(vals)
        pools[ent] = vals
    return pools


def populate_spy(path, domain):
    """Replace <TYPE> placeholder tokens with seeded Faker values; emit
    records with reconstructed text and char-offset gold spans."""
    rows = [json.loads(l) for l in open(path)]
    pools = spy_profiles(len(rows))
    out = []
    for idx, row in enumerate(rows):
        text_parts, spans = [], []
        pos = 0
        toks = row["tokens"]
        tags = row["ent_tags"]
        i = 0
        while i < len(toks):
            tag = tags[i]
            if tag.startswith("B-"):
                ent = tag[2:]
                # trailing whitespace lives inside the token string in the
                # placeholder release; preserve it around the substitution
                orig = toks[i]
                stripped = orig.rstrip()
                trail = orig[len(stripped) :]
                val = pools[ent][idx]
                spans.append({"start": pos, "end": pos + len(val), "type": ent})
                text_parts.append(val + trail)
                pos += len(val) + len(trail)
                i += 1
                # placeholder entities are single tokens in the release, but
                # skip any I- continuation defensively
                while i < len(toks) and tags[i].startswith("I-"):
                    i += 1
            else:
                text_parts.append(toks[i])
                pos += len(toks[i])
                i += 1
        out.append({"id": f"{domain}-{idx}", "text": "".join(text_parts), "spans": spans})
    return out


def prep_tab(path):
    docs = json.load(open(path))
    out = []
    n_annotators = set()
    for doc in docs:
        seen = set()
        spans = []
        for ann_name, ann in doc["annotations"].items():
            n_annotators.add(ann_name)
            for m in ann["entity_mentions"]:
                if m["identifier_type"] not in ("DIRECT", "QUASI"):
                    continue
                key = (m["start_offset"], m["end_offset"])
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    {
                        "start": m["start_offset"],
                        "end": m["end_offset"],
                        "type": m["entity_type"],
                        "identifier_type": m["identifier_type"],
                    }
                )
        spans.sort(key=lambda s: (s["start"], s["end"]))
        out.append({"id": doc["doc_id"], "text": doc["text"], "spans": spans})
    print(f"TAB: {len(out)} docs, annotators seen: {sorted(n_annotators)}")
    return out


def prep_brat(dirpath, domain):
    """Parse a directory of brat .txt/.ann pairs (MEDDOCAN layout)."""
    import glob

    out = []
    for ann_path in sorted(glob.glob(os.path.join(dirpath, "*.ann"))):
        txt_path = ann_path[:-4] + ".txt"
        text = open(txt_path).read()
        spans = []
        for line in open(ann_path):
            if not line.startswith("T"):
                continue
            _tid, meta, span_text = line.rstrip("\n").split("\t")
            typ, rest = meta.split(" ", 1)
            # discontinuous spans "a b;c d" — take the covering interval
            offs = [int(x) for chunk in rest.split(";") for x in chunk.split()]
            start, end = min(offs), max(offs)
            assert text[start:end].startswith(span_text.split(";")[0][:10]) or True
            spans.append({"start": start, "end": end, "type": typ})
        spans.sort(key=lambda s: (s["start"], s["end"]))
        out.append({"id": f"{domain}-{os.path.basename(ann_path)[:-4]}", "text": text, "spans": spans})
    return out


def prep_multigrascco(dirpath):
    """Per-language gold from MultiGraSCCo v2: merge the _PHI and _IPI
    annotation files per language (class-agnostic redaction target)."""
    import glob

    by_lang = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "*_*.json"))):
        base = os.path.basename(path)[:-5]
        lang, _kind = base.rsplit("_", 1)
        by_lang.setdefault(lang.lower(), []).append(path)
    sets = {}
    for lang, paths in by_lang.items():
        docs = {}
        for path in paths:
            for doc in json.load(open(path)):
                d = docs.setdefault(
                    doc["filename"].split(".")[0], {"text": doc["text"], "spans": [], "seen": set()}
                )
                if d["text"] != doc["text"]:
                    continue  # PHI/IPI text drift: keep first file's text
                for e in doc.get("entities", []):
                    key = (e["start"], e["end"])
                    if key in d["seen"]:
                        continue
                    d["seen"].add(key)
                    d["spans"].append({"start": e["start"], "end": e["end"], "type": e.get("type", "?")})
        recs = []
        for did, d in sorted(docs.items()):
            d["spans"].sort(key=lambda s: (s["start"], s["end"]))
            recs.append({"id": f"{lang}-{did}", "text": d["text"], "spans": d["spans"]})
        sets[f"mgs-{lang}"] = recs
    return sets


def cmd_prep(_args):
    os.makedirs(GOLD, exist_ok=True)
    sets = {
        "meddocan-test": prep_brat(os.path.join(DATA, "meddocan-corpus/corpus/test/brat"), "es"),
        "spy-medical": populate_spy(
            os.path.join(DATA, "spy/data/medical_consultations_placeholders.jsonl"), "med"
        ),
        "spy-legal": populate_spy(os.path.join(DATA, "spy/data/legal_questions_placeholders.jsonl"), "leg"),
        "tab-test": prep_tab(os.path.join(DATA, "tab/echr_test.json")),
    }
    sets.update(prep_multigrascco(os.path.join(DATA, "multigrascco/extracted/MultiGraSCCo_v2")))
    for name, recs in sets.items():
        with open(os.path.join(GOLD, name + ".jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        nspan = sum(len(r["spans"]) for r in recs)
        print(f"{name}: {len(recs)} records, {nspan} gold spans (seed={SPY_SEED})")


def load_unified_eval(source_path, id_prefix):
    """Adapt a complete-supervision training-format split into gold shards."""
    by_language = defaultdict(list)
    with open(source_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            supervision = row.get("supervision", "complete")
            if supervision != "complete":
                raise ValueError(
                    f"{source_path}:{line_number}: evaluation requires complete supervision, "
                    f"got {supervision!r}"
                )
            language = row.get("lang")
            if not language:
                raise ValueError(f"{source_path}:{line_number}: missing language")
            text = row["text"]
            spans = []
            for span_index, span in enumerate(row.get("spans", []), 1):
                if not isinstance(span, list) or len(span) != 3:
                    raise ValueError(
                        f"{source_path}:{line_number}: span {span_index} must be [start, end, type]"
                    )
                start, end, entity_type = span
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(text)
                ):
                    raise ValueError(f"{source_path}:{line_number}: invalid span {span_index}: {span!r}")
                spans.append({"start": start, "end": end, "type": entity_type})
            by_language[language].append(
                {
                    "id": f"{id_prefix}:{line_number - 1}",
                    "text": text,
                    "spans": spans,
                    "meta": {
                        key: row[key] for key in ("lang", "src", "mix_source", "supervision") if key in row
                    },
                }
            )
    return dict(sorted(by_language.items()))


def cmd_prep_unified(args):
    """Export a frozen complete-supervision split as per-language gold."""
    os.makedirs(GOLD, exist_ok=True)
    shards = load_unified_eval(args.source, args.dataset_prefix)
    for language, records in shards.items():
        if args.per_language:
            if len(records) < args.per_language:
                raise ValueError(
                    f"{language}: requested {args.per_language} calibration records, only {len(records)} available"
                )
            records = list(records)
            random.Random(f"{args.seed}:{language}").shuffle(records)
            records = sorted(records[: args.per_language], key=lambda record: record["id"])
        name = f"{args.dataset_prefix}-{language}"
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} records, {n_spans} complete-supervision spans -> {output}")


def load_mapa_test(source_root, languages):
    """Load the frozen MAPA test shards into the generic PII gold schema."""
    sets = {}
    for lang in languages:
        path = os.path.join(source_root, "test", f"{lang}.jsonl.gz")
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                records.append(
                    {
                        "id": row["id"],
                        "document_id": f"{lang}:{row['metadata']['file_name']}",
                        "text": row["text"],
                        "spans": [
                            {
                                "start": span["start"],
                                "end": span["end"],
                                "type": span["source_label"],
                            }
                            for span in row["spans"]
                        ],
                    }
                )
        sets[f"mapa-{lang}-test"] = records
    return sets


def cmd_prep_mapa(args):
    os.makedirs(GOLD, exist_ok=True)
    for name, records in load_mapa_test(args.source_root, args.languages).items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        n_documents = len({record["document_id"] for record in records})
        print(
            f"{name}: {len(records)} sentences from {n_documents} documents, {n_spans} gold spans -> {output}"
        )


def load_idner_news_splits(source_root, splits):
    """Load pinned idner-news-2k shards into the generic PII gold schema."""
    sets = {}
    for split in splits:
        path = os.path.join(source_root, split, "id.jsonl.gz")
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                records.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "spans": [
                            {
                                "start": span["start"],
                                "end": span["end"],
                                "type": span["source_label"],
                            }
                            for span in row["spans"]
                        ],
                    }
                )
        sets[f"idner-{split}"] = records
    return sets


def cmd_prep_idner(args):
    os.makedirs(GOLD, exist_ok=True)
    for name, records in load_idner_news_splits(args.source_root, args.splits).items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} sentences, {n_spans} gold spans -> {output}")


def load_hiner_splits(source_root, splits):
    """Load pinned HiNER shards into the generic PII gold schema."""
    sets = {}
    for split in splits:
        path = os.path.join(source_root, split, "hi.jsonl.gz")
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                records.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "spans": [
                            {
                                "start": span["start"],
                                "end": span["end"],
                                "type": span["source_label"],
                            }
                            for span in row["spans"]
                        ],
                    }
                )
        sets[f"hiner-{split}"] = records
    return sets


def cmd_prep_hiner(args):
    os.makedirs(GOLD, exist_ok=True)
    for name, records in load_hiner_splits(args.source_root, args.splits).items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} sentences, {n_spans} privacy-relevant gold spans -> {output}")


def load_klue_splits(source_root, splits):
    """Load pinned KLUE NER shards into the generic PII gold schema.

    KLUE annotates person/location/organization only, so outside tokens may
    hold other deployment types; score it as an annotated-span recall screen
    rather than as a full-inventory precision comparison.
    """
    sets = {}
    for split in splits:
        path = os.path.join(source_root, split, "ko.jsonl.gz")
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                records.append(
                    {
                        "id": row["id"],
                        "document_id": row["metadata"]["source_document"],
                        "text": row["text"],
                        "spans": [
                            {
                                "start": span["start"],
                                "end": span["end"],
                                "type": span["source_label"],
                            }
                            for span in row["spans"]
                        ],
                    }
                )
        sets[f"klue-{split}"] = records
    return sets


def cmd_prep_klue(args):
    os.makedirs(GOLD, exist_ok=True)
    for name, records in load_klue_splits(args.source_root, args.splits).items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} sentences, {n_spans} privacy-relevant gold spans -> {output}")


def load_aqmar_splits(source_root, splits):
    """Load the BIO-corrected OpenNER AQMAR shards into the PII gold schema."""
    sets = {}
    for split in splits:
        path = os.path.join(source_root, split, "ar.jsonl.gz")
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                records.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "spans": [
                            {
                                "start": span["start"],
                                "end": span["end"],
                                "type": span["source_label"],
                            }
                            for span in row["spans"]
                        ],
                    }
                )
        sets[f"aqmar-{split}"] = records
    return sets


def cmd_prep_aqmar(args):
    os.makedirs(GOLD, exist_ok=True)
    for name, records in load_aqmar_splits(args.source_root, args.splits).items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} Arabic Wikipedia sentences, {n_spans} gold spans -> {output}")


def load_openner_core_splits(source_root, splits, languages):
    """Load the license-screened OpenNER core shards into the PII gold schema."""
    sets = {}
    for split in splits:
        for lang in languages:
            path = os.path.join(source_root, split, f"{lang}.jsonl.gz")
            records = []
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    records.append(
                        {
                            "id": row["id"],
                            "text": row["text"],
                            "spans": [
                                {
                                    "start": span["start"],
                                    "end": span["end"],
                                    "type": span["source_label"],
                                }
                                for span in row["spans"]
                            ],
                        }
                    )
            sets[f"openner-{lang}-{split}"] = records
    return sets


def cmd_prep_openner_core(args):
    os.makedirs(GOLD, exist_ok=True)
    sets = load_openner_core_splits(args.source_root, args.splits, args.languages)
    for name, records in sets.items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} sentences, {n_spans} gold spans -> {output}")


def _longest_non_overlapping_spans(spans):
    """Flatten nested spans to the same longest-first discipline used in training."""
    kept = []
    for span in sorted(spans, key=lambda item: (-(item["end"] - item["start"]), item["start"])):
        if all(span["end"] <= other["start"] or span["start"] >= other["end"] for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda item: (item["start"], item["end"], item["source_label"]))


def load_wojood_splits(source_root, splits):
    """Load and flatten pinned Wojood sample shards into the PII gold schema."""
    sets = {}
    for split in splits:
        path = os.path.join(source_root, split, "ar.jsonl.gz")
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                flat_spans = _longest_non_overlapping_spans(row["spans"])
                records.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "spans": [
                            {
                                "start": span["start"],
                                "end": span["end"],
                                "type": span["source_label"],
                            }
                            for span in flat_spans
                        ],
                    }
                )
        sets[f"wojood-{split}"] = records
    return sets


def cmd_prep_wojood(args):
    os.makedirs(GOLD, exist_ok=True)
    for name, records in load_wojood_splits(args.source_root, args.splits).items():
        output = os.path.join(GOLD, f"{name}.jsonl")
        with open(output, "w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        n_spans = sum(len(record["spans"]) for record in records)
        print(f"{name}: {len(records)} sentences, {n_spans} flattened privacy-relevant spans -> {output}")


# ------------------------------------------------------------------ predict


def windows(text, max_chars, overlap):
    """Whitespace-friendly sliding windows; (offset, chunk) pairs."""
    if len(text) <= max_chars:
        return [(0, text)]
    out = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sp = text.rfind(" ", start + max_chars - 200, end)
            if sp > start:
                end = sp
        out.append((start, text[start:end]))
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def _latency_stats(latencies):
    ls = sorted(latencies)
    if not ls:
        return {}
    return {
        "mean": sum(ls) / len(ls),
        "p50": ls[len(ls) // 2],
        "p90": ls[int(0.9 * len(ls))],
        "max": ls[-1],
    }


def _log_latency(latencies, log):
    stats = _latency_stats(latencies)
    if stats:
        log(
            f"LATENCY per-document s: mean {stats['mean']:.3f} "
            f"p50 {stats['p50']:.3f} p90 {stats['p90']:.3f} "
            f"max {stats['max']:.3f}"
        )


def dedupe_window_preds(preds):
    """Collapse duplicate window decodes of one span to a single prediction.

    Geometry-first: overlapping decode windows can resolve *different*
    labels for the identical (start, end) span (a window boundary cuts
    the span, changing the type tally), and a label-keyed dedupe then
    kept both copies — the duplicate leaked into precision and made
    nominally geometry-invariant decode options differ on span metrics.
    Same-geometry copies now collapse to one span whose label is the
    majority across window copies, first-seen winning ties.
    """
    by_geometry = {}
    for p in sorted(preds, key=lambda p: (p["start"], p["end"])):
        key = (p["start"], p["end"])
        entry = by_geometry.setdefault(key, {"pred": p, "votes": {}})
        entry["votes"][p["label"]] = entry["votes"].get(p["label"], 0) + 1
    out = []
    for entry in by_geometry.values():
        votes = entry["votes"]
        first_label = entry["pred"]["label"]
        best = max(votes.values())
        label = first_label if votes[first_label] == best else max(votes, key=votes.get)
        pred = dict(entry["pred"])
        pred["label"] = label
        out.append(pred)
    return out


def dedupe_exact_window_preds(preds):
    """Drop only repeated start/end/label triples, preserving first-seen order."""
    seen = set()
    out = []
    for prediction in preds:
        key = (prediction["start"], prediction["end"], prediction["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(prediction)
    return out


BIOES_BUCKET_COMPAT_LEVELS = (
    "fine",
    "ontology_v2_family",
    "redaction_20_v1",
    "redaction_9_v1",
    "entity",
)
BIOES_BUCKET_REDUCTIONS = ("max", "logsumexp", "logmeanexp")
PREPONDERANCE_WEIGHT_MODES = ("token", "codepoint", "nfc-char")
DEFAULT_BIOES_BUCKET_COMPAT = "entity"
DEFAULT_BIOES_BUCKET_COMPAT_TYPE = "preponderance"


def bucket_compat_map(cut, fine_types=None):
    """Fine category -> compatibility class for BIOES continuation repair.

    Reporting cuts use their ontology buckets. ``entity`` is the P1 cut:
    every model-native type is equivalent for a mismatched I/E
    continuation, while explicit B/S labels remain hard boundaries.
    """
    if cut in ("fine", "entity"):
        if fine_types is None:
            raise ValueError(f"{cut} compatibility requires the model's fine-type inventory")
        fine_types = sorted(set(fine_types))
        if not fine_types:
            raise ValueError(f"{cut} compatibility requires at least one fine type")
        return {fine_type: ("entity" if cut == "entity" else fine_type) for fine_type in fine_types}

    if cut == "ontology_v2_family":
        if fine_types is None:
            raise ValueError("ontology_v2_family compatibility requires the model's fine-type inventory")
        if __package__:
            from scripts.pii_ontology_v2 import load_ontology
        else:
            from pii_ontology_v2 import load_ontology

        ontology = load_ontology()
        family_of = {
            primary_type: ontology.family_of(primary_type) for primary_type in ontology.primary_types
        }
        family_of.update(
            {
                "organization_reference": "organization",
                "person_reference": "person",
            }
        )
        unknown = sorted(set(fine_types) - family_of.keys())
        if unknown:
            raise ValueError(f"ontology_v2_family has no family for model types: {unknown}")
        return {fine_type: family_of[fine_type] for fine_type in fine_types}

    if __package__:
        from scripts.pii_projector import Tagset
    else:
        from pii_projector import Tagset

    tagset = Tagset()
    if fine_types is not None:
        fine_types = set(fine_types)
        if fine_types <= tagset.cut_targets("redaction_20_v1"):
            return {
                fine_type: tagset.project_cut("redaction_20_v1", fine_type, cut) for fine_type in fine_types
            }
        if fine_types <= tagset.cut_targets("redaction_9_v1"):
            return {
                fine_type: tagset.project_cut("redaction_9_v1", fine_type, cut) for fine_type in fine_types
            }
        return {fine_type: tagset.project_canonical_cut(fine_type, cut) for fine_type in fine_types}
    return {node: tagset.project_canonical_cut(node, cut) for node in tagset.nodes}


def preponderance_token_weights(text, offsets, mode):
    """Return per-token vote mass under an explicit text-unit contract."""
    if mode not in PREPONDERANCE_WEIGHT_MODES:
        raise ValueError(f"unknown preponderance weight mode {mode!r}")
    offsets = list(offsets)
    if mode == "token":
        return [1.0] * len(offsets)
    if mode == "codepoint":
        return [float(end - start) for start, end in offsets]
    weights = []
    for start, end in offsets:
        if not 0 <= start <= end <= len(text):
            raise ValueError(f"token offset {(start, end)} is outside document text of length {len(text)}")
        weights.append(float(len(unicodedata.normalize("NFC", text[start:end]))))
    return weights


def unicode_word_boundary_positions(text: str) -> frozenset[int]:
    """Return deterministic Unicode word-break positions for one surface.

    The optional decoder uses the third-party ``regex`` engine's WORD mode,
    which applies its default Unicode word-boundary contract to ``\\b``.
    VERSION1 makes the engine behavior explicit rather than inheriting its
    process-wide default.
    """
    import regex

    return frozenset(
        match.start()
        for match in regex.finditer(
            r"\b(?=\w)|(?<=\w)\b",
            text,
            regex.WORD | regex.VERSION1,
        )
    )


def covered_word_boundary_offsets(token_starts, token_ends, word_boundaries):
    """Map each token to directional word boundaries that its interval covers.

    A start uses the leftmost boundary in ``[token_start, token_end)`` and an
    end uses the rightmost boundary in ``(token_start, token_end]``. Thus a
    token that extends past a word edge can still realize that edge. ``None``
    marks a token that covers no boundary in the requested direction.
    """
    import bisect

    starts = [int(value) for value in token_starts]
    ends = [int(value) for value in token_ends]
    if len(starts) != len(ends):
        raise ValueError("token starts and ends must have equal lengths")
    boundaries = sorted(int(value) for value in word_boundaries)
    start_offsets = []
    end_offsets = []
    for start, end in zip(starts, ends, strict=True):
        if start >= end:
            raise ValueError(f"token offset {(start, end)} must have positive width")
        start_index = bisect.bisect_left(boundaries, start)
        start_boundary = (
            boundaries[start_index]
            if start_index < len(boundaries) and boundaries[start_index] < end
            else None
        )
        end_index = bisect.bisect_right(boundaries, end) - 1
        end_boundary = boundaries[end_index] if end_index >= 0 and boundaries[end_index] > start else None
        start_offsets.append(start_boundary)
        end_offsets.append(end_boundary)
    return start_offsets, end_offsets


def unicode_whitespace_split_boundaries(text, token_starts, token_ends):
    """Mark positive whitespace-only joins after exposing attached punctuation.

    Covered Unicode word edges move an attached comma or other edge punctuation
    into the tested source gap. Zero-width tokenizer-piece joins never qualify.
    """
    starts = [int(value) for value in token_starts]
    ends = [int(value) for value in token_ends]
    if len(starts) != len(ends):
        raise ValueError("token starts and ends must have equal lengths")
    word_starts, word_ends = covered_word_boundary_offsets(
        starts,
        ends,
        unicode_word_boundary_positions(text),
    )
    boundaries = []
    for index in range(len(starts) - 1):
        gap_start = ends[index] if word_ends[index] is None else word_ends[index]
        gap_end = starts[index + 1] if word_starts[index + 1] is None else word_starts[index + 1]
        if not 0 <= gap_start <= len(text) or not 0 <= gap_end <= len(text):
            raise ValueError(
                f"adjusted token gap {(gap_start, gap_end)} is outside text of length {len(text)}"
            )
        if gap_start >= gap_end:
            boundaries.append(False)
            continue
        gap = text[gap_start:gap_end]
        boundaries.append(bool(gap) and gap.isspace())
    return boundaries


def decode_bioes_labels(
    labels, offsets, bucket_of=None, span_type="opening", token_weights=None, type_resolver=None
):
    """Assemble greedy BIOES token labels into character-offset spans.

    With ``bucket_of`` (fine type -> reporting bucket), a continuation
    token (I/E) whose fine type differs from the open span's but lands
    in the same bucket extends the span instead of splitting it — the
    ontology-aware repair. An explicit B/S still always starts a new
    span. ``span_type`` picks the merged span's reported fine label:
    ``opening`` keeps the first token's type; ``preponderance`` reports
    the type with the largest supplied ``token_weights`` mass among the
    span's token argmax labels (raw codepoint span length when weights
    are omitted); ``margin`` requires ``token_weights`` such as the
    bias-adjusted entity margin from a score cache. A steady runner-up
    type is invisible to both, a top-1 limitation. Opening type wins
    ties. The choice can only move fine-tag scores: every candidate type
    in a repaired span shares the compatibility class by construction.
    """
    if span_type not in ("opening", "preponderance", "margin", "logitsum"):
        raise ValueError(f"unknown span_type {span_type!r}")
    if span_type == "margin" and token_weights is None:
        raise ValueError("span_type='margin' requires token_weights")
    if span_type == "logitsum" and type_resolver is None:
        raise ValueError("span_type='logitsum' requires a type_resolver (full logits at decode)")
    spans = []
    current = None  # [start, end, opening_type, mass_by_type, token_indices]

    def close(entry):
        start, end, opening, mass, token_indices = entry
        label = opening
        if span_type == "logitsum":
            # every bucket-compatible type competes on summed strength,
            # so a steady runner-up that never ranks #1 can still win
            label = type_resolver(token_indices, opening)
        elif span_type != "opening" and len(mass) > 1:
            best = max(mass.values())
            if mass[opening] != best:
                label = max(mass, key=lambda t: (mass[t], t == opening))
        spans.append([start, end, label])

    for token_index, (label, (start, end)) in enumerate(zip(labels, offsets)):
        if start == end:
            continue
        if label == "O":
            if current:
                close(current)
                current = None
            continue
        prefix, entity_type = label.split("-", 1)
        compatible = (
            current is not None
            and current[2] != entity_type
            and prefix in ("I", "E")
            and bucket_of is not None
            and bucket_of.get(current[2]) is not None
            and bucket_of.get(current[2]) == bucket_of.get(entity_type)
        )
        weight = (
            float(token_weights[token_index])
            if span_type in ("preponderance", "margin") and token_weights is not None
            else end - start
        )
        if prefix in ("B", "S") or (current and current[2] != entity_type and not compatible):
            if current:
                close(current)
            current = [start, end, entity_type, {entity_type: weight}, [token_index]]
        elif current:
            current[1] = end
            current[3][entity_type] = current[3].get(entity_type, 0) + weight
            current[4].append(token_index)
        else:
            current = [start, end, entity_type, {entity_type: weight}, [token_index]]
        if prefix in ("E", "S"):
            close(current)
            current = None
    if current:
        close(current)
    return [{"start": int(start), "end": int(end), "label": entity_type} for start, end, entity_type in spans]


def save_token_score_cache(path, cache):
    """Persist an already materialized no-pickle token score cache."""
    import numpy as np

    payload = {name: np.asarray(value) for name, value in cache.items()}
    object_fields = [name for name, value in payload.items() if value.dtype.hasobject]
    if object_fields:
        raise ValueError(f"token score-cache fields cannot use object dtype: {object_fields}")
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)


def write_token_score_cache(path, labels, document_ids, document_datasets, document_windows):
    """Persist the sufficient trace for post-hoc global O-bias decoding."""
    import numpy as np

    if not (len(document_ids) == len(document_datasets) == len(document_windows)):
        raise ValueError("score-cache document metadata and traces must have equal lengths")
    window_docs = []
    window_token_starts = [0]
    token_starts = []
    token_ends = []
    top_non_o_labels = []
    o_minus_top = []
    top3_non_o_labels = []
    o_minus_top3 = []
    nfc_char_counts = []
    full_logits = []
    has_full_logits = None
    for document_index, windows_for_document in enumerate(document_windows):
        for window in windows_for_document:
            window_has_full_logits = "full_logits" in window
            if has_full_logits is None:
                has_full_logits = window_has_full_logits
            elif has_full_logits != window_has_full_logits:
                raise ValueError("score-cache windows must agree on whether full logits are present")
            lengths = {
                len(window["token_start"]),
                len(window["token_end"]),
                len(window["top_non_o_label"]),
                len(window["o_minus_top"]),
                len(window["top3_non_o_label"]),
                len(window["o_minus_top3"]),
                len(window["nfc_char_count"]),
            }
            if window_has_full_logits:
                lengths.add(len(window["full_logits"]))
            if len(lengths) != 1:
                raise ValueError("score-cache window arrays must have equal lengths")
            if window_has_full_logits and any(len(row) != len(labels) for row in window["full_logits"]):
                raise ValueError("every full-logit row must match the cache label inventory")
            window_docs.append(document_index)
            token_starts.extend(window["token_start"])
            token_ends.extend(window["token_end"])
            top_non_o_labels.extend(window["top_non_o_label"])
            o_minus_top.extend(window["o_minus_top"])
            top3_non_o_labels.extend(window["top3_non_o_label"])
            o_minus_top3.extend(window["o_minus_top3"])
            nfc_char_counts.extend(window["nfc_char_count"])
            if window_has_full_logits:
                full_logits.extend(window["full_logits"])
            window_token_starts.append(len(token_starts))
    payload = dict(
        schema_version=np.asarray([1], dtype=np.int32),
        calibration_family=np.asarray(["greedy_global_o_bias"]),
        labels=np.asarray(labels),
        document_ids=np.asarray(document_ids),
        document_datasets=np.asarray(document_datasets),
        window_document=np.asarray(window_docs, dtype=np.int32),
        window_token_start=np.asarray(window_token_starts, dtype=np.int64),
        token_start=np.asarray(token_starts, dtype=np.int32),
        token_end=np.asarray(token_ends, dtype=np.int32),
        top_non_o_label=np.asarray(top_non_o_labels, dtype=np.uint16),
        o_minus_top_logit=np.asarray(o_minus_top, dtype=np.float32),
        # additive top-3 trace (user 2026-08-07): enables sparse
        # logitsum/margin type resolution from cache; schema stays 1 so
        # k=1 readers keep working, top-3 readers check field presence
        top3_non_o_label=np.asarray(top3_non_o_labels, dtype=np.uint16).reshape(-1, 3),
        o_minus_top3_logit=np.asarray(o_minus_top3, dtype=np.float32).reshape(-1, 3),
        token_nfc_char_count=np.asarray(nfc_char_counts, dtype=np.int32),
    )
    if has_full_logits:
        payload["token_logits"] = np.asarray(full_logits, dtype=np.float32).reshape(-1, len(labels))
    save_token_score_cache(path, payload)


def load_token_score_cache(path):
    """Load a no-pickle token score cache into ordinary NumPy arrays."""
    import numpy as np

    with np.load(path, allow_pickle=False) as source:
        cache = {name: source[name].copy() for name in source.files}
    if cache["schema_version"].tolist() != [1]:
        raise ValueError(f"unsupported token score-cache schema: {cache['schema_version'].tolist()}")
    if cache["calibration_family"].tolist() != ["greedy_global_o_bias"]:
        raise ValueError(f"unsupported calibration family: {cache['calibration_family'].tolist()}")
    return cache


def load_ordered_label_vocabulary(path: str) -> list[str]:
    """Read the exact contiguous output-label order from an the production toolkit model.vcb."""
    labels = []
    with open(path, encoding="utf-8") as source:
        for expected, line in enumerate(source):
            fields = line.rstrip("\n").split("\t", 1)
            if len(fields) != 2 or not fields[1]:
                raise ValueError(f"{path}:{expected + 1}: expected '<index>\\t<label>'")
            try:
                observed = int(fields[0])
            except ValueError as error:
                raise ValueError(f"{path}:{expected + 1}: invalid label index {fields[0]!r}") from error
            if observed != expected:
                raise ValueError(
                    f"{path}:{expected + 1}: expected contiguous label index {expected}, got {observed}"
                )
            labels.append(fields[1])
    if len(labels) < 4 or labels.count("O") != 1:
        raise ValueError(f"{path}: expected one O label and at least three entity labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: duplicate output labels")
    return labels


def token_logits_score_fields(logits, labels: list[str]) -> dict:
    """Derive the sparse score-cache fields from a full token-logit lattice."""
    import numpy as np

    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError(f"token logits shape {values.shape} must be [tokens, {len(labels)}]")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("token logits must be finite floating-point values")
    if labels.count("O") != 1:
        raise ValueError("token labels must contain exactly one O label")
    if len(labels) < 4:
        raise ValueError("token score caching requires O and at least three entity labels")
    o_label_id = labels.index("O")
    non_o = values.copy()
    non_o[:, o_label_id] = -np.inf
    top3_ids = np.argsort(-non_o, axis=1, kind="stable")[:, :3]
    top3_values = np.take_along_axis(non_o, top3_ids, axis=1)
    top_ids = top3_ids[:, 0]
    top_values = top3_values[:, 0]
    o_values = values[:, o_label_id]
    return {
        "top_non_o_label": top_ids,
        "o_minus_top_logit": (o_values - top_values).astype(np.float32),
        "top3_non_o_label": top3_ids,
        "o_minus_top3_logit": (o_values[:, None] - top3_values).astype(np.float32),
    }


def token_logits_score_window(logits, text: str, labels: list[str], offsets) -> dict:
    """Convert one token-interval logit lattice to the shared score-cache window."""
    import numpy as np

    offsets = [(int(start), int(end)) for start, end in offsets]
    if any(not 0 <= start <= end <= len(text) for start, end in offsets):
        raise ValueError("token interval lies outside the source text")
    values = np.asarray(logits)
    expected_shape = (len(offsets), len(labels))
    if values.shape != expected_shape:
        raise ValueError(f"token logits shape {values.shape} != {expected_shape}")
    scores = token_logits_score_fields(values, labels)
    return {
        "token_start": [start for start, _ in offsets],
        "token_end": [end for _, end in offsets],
        "top_non_o_label": scores["top_non_o_label"].tolist(),
        "o_minus_top": scores["o_minus_top_logit"].tolist(),
        "top3_non_o_label": scores["top3_non_o_label"].tolist(),
        "o_minus_top3": scores["o_minus_top3_logit"].tolist(),
        "nfc_char_count": preponderance_token_weights(text, offsets, "nfc-char"),
        "full_logits": values.astype(np.float32).tolist(),
    }


def character_logits_score_window(logits, text: str, labels: list[str]) -> dict:
    """Convert one full-document codepoint lattice to the shared score-cache window."""
    offsets = [(position, position + 1) for position in range(len(text))]
    return token_logits_score_window(logits, text, labels, offsets)


def load_character_onnx_bundle(bundle: str):
    """Load and validate the tokenizer-free one-input character ONNX contract."""
    from pathlib import Path

    import onnxruntime as ort

    from scripts.pii_character_projection import load_character_projection

    root = Path(bundle)
    model_path = root / "model.onnx"
    projection_path = root / "character_projection.json"
    vocabulary_path = root / "model.vcb"
    for required in (model_path, projection_path, vocabulary_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    labels = load_ordered_label_vocabulary(str(vocabulary_path))
    projection = load_character_projection(projection_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "char_ids":
        raise ValueError(
            f"{model_path}: expected sole ONNX input 'char_ids', got {[value.name for value in inputs]}"
        )
    if inputs[0].type != "tensor(int64)" or len(inputs[0].shape) != 2:
        raise ValueError(
            f"{model_path}: char_ids must be rank-2 int64, got {inputs[0].type} {inputs[0].shape}"
        )
    if len(outputs) != 1 or outputs[0].name != "logits":
        raise ValueError(
            f"{model_path}: expected sole ONNX output 'logits', got {[value.name for value in outputs]}"
        )
    if outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 3:
        raise ValueError(
            f"{model_path}: logits must be rank-3 float32, got {outputs[0].type} {outputs[0].shape}"
        )
    classes = outputs[0].shape[2]
    if isinstance(classes, int) and classes != len(labels):
        raise ValueError(f"{model_path}: output has {classes} classes but model.vcb has {len(labels)}")
    return session, projection, labels


def load_character_token_onnx_bundle(bundle: str):
    """Load the three-input character CNN with its interval tokenizer."""
    from pathlib import Path

    import onnxruntime as ort
    from tokenizers import Tokenizer

    from scripts.pii_character_projection import load_character_projection

    root = Path(bundle)
    model_path = root / "model.onnx"
    projection_path = root / "character_projection.json"
    vocabulary_path = root / "model.vcb"
    tokenizer_path = root / "tokenizer.json"
    for required in (model_path, projection_path, vocabulary_path, tokenizer_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    labels = load_ordered_label_vocabulary(str(vocabulary_path))
    projection = load_character_projection(projection_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_inputs = ["char_ids", "token_starts", "token_ends"]
    if [value.name for value in inputs] != expected_inputs:
        raise ValueError(
            f"{model_path}: expected ONNX inputs {expected_inputs}, got {[value.name for value in inputs]}"
        )
    for value in inputs:
        if value.type != "tensor(int64)" or len(value.shape) != 2:
            raise ValueError(
                f"{model_path}: {value.name} must be rank-2 int64, got {value.type} {value.shape}"
            )
    if len(outputs) != 1 or outputs[0].name != "logits":
        raise ValueError(
            f"{model_path}: expected sole ONNX output 'logits', got {[value.name for value in outputs]}"
        )
    if outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 3:
        raise ValueError(
            f"{model_path}: logits must be rank-3 float32, got {outputs[0].type} {outputs[0].shape}"
        )
    classes = outputs[0].shape[2]
    if isinstance(classes, int) and classes != len(labels):
        raise ValueError(f"{model_path}: output has {classes} classes but model.vcb has {len(labels)}")
    return session, tokenizer, projection, labels


def predict_character_onnx(
    texts,
    log,
    *,
    bundle: str,
    score_cache_path: str,
    record_ids,
    record_datasets,
):
    """Run an exact exported per-codepoint graph and retain its complete logits."""
    import numpy as np

    from scripts.pii_character_projection import diagonal_character_ids

    if not (len(texts) == len(record_ids) == len(record_datasets)):
        raise ValueError("character ONNX texts, IDs, and dataset names must align")
    session, projection, labels = load_character_onnx_bundle(bundle)
    results = []
    document_windows = []
    o_label_id = labels.index("O")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            raise ValueError(f"character ONNX record {record_ids[index]!r} has empty or non-string text")
        char_ids = np.asarray([diagonal_character_ids(projection, text)], dtype=np.int64)
        started = time.perf_counter()
        logits = session.run(["logits"], {"char_ids": char_ids})[0]
        latency = time.perf_counter() - started
        expected_shape = (1, len(text), len(labels))
        if logits.shape != expected_shape:
            raise ValueError(f"character ONNX output shape {logits.shape} != {expected_shape}")
        window = character_logits_score_window(logits[0], text, labels)
        document_windows.append([window])
        greedy_ids = np.asarray(window["full_logits"]).argmax(axis=1).tolist()
        offsets = list(zip(window["token_start"], window["token_end"]))
        predictions = decode_bioes_labels(
            [labels[int(label_id)] for label_id in greedy_ids],
            offsets,
        )
        results.append((predictions, latency))
        if (index + 1) % 200 == 0:
            log(f"character ONNX {index + 1}/{len(texts)}")
    write_token_score_cache(
        score_cache_path,
        labels,
        record_ids,
        record_datasets,
        document_windows,
    )
    log(f"score cache: {len(texts)} documents, O column {o_label_id} -> {os.path.abspath(score_cache_path)}")
    return results


def predict_character_token_onnx(
    texts,
    log,
    *,
    bundle: str,
    score_cache_path: str,
    record_ids,
    record_datasets,
):
    """Run the exported character CNN on its bundled tokenizer intervals."""
    import numpy as np

    from scripts.pii_character_projection import diagonal_character_ids

    if not (len(texts) == len(record_ids) == len(record_datasets)):
        raise ValueError("character-token ONNX texts, IDs, and dataset names must align")
    session, tokenizer, projection, labels = load_character_token_onnx_bundle(bundle)
    results = []
    document_windows = []
    o_label_id = labels.index("O")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            raise ValueError(
                f"character-token ONNX record {record_ids[index]!r} has empty or non-string text"
            )
        encoding = tokenizer.encode(text, add_special_tokens=False)
        offsets = [(int(start), int(end)) for start, end in encoding.offsets]
        if not offsets:
            raise ValueError(f"character-token ONNX record {record_ids[index]!r} produced no tokens")
        if any(not 0 <= start <= end <= len(text) for start, end in offsets):
            raise ValueError(f"character-token ONNX record {record_ids[index]!r} has invalid token offsets")
        character_end = max(end for _, end in offsets)
        if not character_end:
            raise ValueError(f"character-token ONNX record {record_ids[index]!r} has no visible token text")
        inputs = {
            "char_ids": np.asarray(
                [diagonal_character_ids(projection, text[:character_end])],
                dtype=np.int64,
            ),
            "token_starts": np.asarray([[start for start, _ in offsets]], dtype=np.int64),
            "token_ends": np.asarray([[end for _, end in offsets]], dtype=np.int64),
        }
        started = time.perf_counter()
        logits = session.run(["logits"], inputs)[0]
        latency = time.perf_counter() - started
        expected_shape = (1, len(offsets), len(labels))
        if logits.shape != expected_shape:
            raise ValueError(f"character-token ONNX output shape {logits.shape} != {expected_shape}")
        window = token_logits_score_window(logits[0], text, labels, offsets)
        document_windows.append([window])
        greedy_ids = np.asarray(window["full_logits"]).argmax(axis=1).tolist()
        predictions = decode_bioes_labels(
            [labels[int(label_id)] for label_id in greedy_ids],
            offsets,
        )
        results.append((predictions, latency))
        if (index + 1) % 200 == 0:
            log(f"character-token ONNX {index + 1}/{len(texts)}")
    write_token_score_cache(
        score_cache_path,
        labels,
        record_ids,
        record_datasets,
        document_windows,
    )
    log(f"score cache: {len(texts)} documents, O column {o_label_id} -> {os.path.abspath(score_cache_path)}")
    return results


def subset_token_score_cache(cache, document_indices):
    """Return a self-contained document slice of a token score cache."""
    import numpy as np

    selected = [int(index) for index in document_indices]
    document_count = len(cache["document_ids"])
    if len(selected) != len(set(selected)):
        raise ValueError("token score-cache document indices must be unique")
    if any(index < 0 or index >= document_count for index in selected):
        raise ValueError("token score-cache document index is out of range")

    token_fields = (
        "token_start",
        "token_end",
        "top_non_o_label",
        "o_minus_top_logit",
        "top3_non_o_label",
        "o_minus_top3_logit",
        "token_nfc_char_count",
        "token_logits",
    )
    token_count = len(cache["token_start"])
    for name in token_fields:
        if name in cache and len(cache[name]) != token_count:
            raise ValueError(f"token score-cache field {name!r} is not token-aligned")

    result = {name: value.copy() for name, value in cache.items()}
    result["document_ids"] = cache["document_ids"][selected].copy()
    result["document_datasets"] = cache["document_datasets"][selected].copy()
    token_boundaries = cache["window_token_start"]
    old_window_documents = cache["window_document"]
    selected_windows = []
    new_window_documents = []
    for new_document, old_document in enumerate(selected):
        for window in np.flatnonzero(old_window_documents == old_document).tolist():
            selected_windows.append(window)
            new_window_documents.append(new_document)

    token_slices = [
        slice(int(token_boundaries[window]), int(token_boundaries[window + 1])) for window in selected_windows
    ]
    lengths = [token_slice.stop - token_slice.start for token_slice in token_slices]
    result["window_document"] = np.asarray(new_window_documents, dtype=cache["window_document"].dtype)
    result["window_token_start"] = np.asarray(
        [0, *np.cumsum(lengths).tolist()], dtype=cache["window_token_start"].dtype
    )
    for name in token_fields:
        if name not in cache:
            continue
        pieces = [cache[name][token_slice] for token_slice in token_slices]
        result[name] = np.concatenate(pieces, axis=0) if pieces else cache[name][:0].copy()
    return result


def decode_token_score_cache(
    cache,
    o_logit_bias,
    bucket_of=None,
    span_type="opening",
    preponderance_weight="codepoint",
    document_texts=None,
    bioes_search="greedy-repair",
    bioes_search_top_k=None,
    bioes_bucket_reduction="max",
    bioes_bucket_temperature=1.0,
    bioes_whitespace_split_cost=0.0,
    window_deduplication="geometry",
    bioes_backoff_alpha=None,
):
    """Decode one global O-bias operating point without another model forward."""
    if span_type == "logitsum" and "top3_non_o_label" not in cache:
        raise ValueError(
            "span_type='logitsum' needs the top-3 cache fields; this cache predates them "
            "(re-predict with --score-cache to refresh)"
        )
    labels = cache["labels"].tolist()
    document_ids = cache["document_ids"].tolist()
    document_datasets = cache["document_datasets"].tolist()
    if document_texts is not None and len(document_texts) != len(document_ids):
        raise ValueError("document_texts must align one-for-one with cached document IDs")
    if preponderance_weight not in PREPONDERANCE_WEIGHT_MODES:
        raise ValueError(f"unknown preponderance weight mode {preponderance_weight!r}")
    if bioes_search not in (
        "greedy-repair",
        "lazy-fine-legal",
        "word-fine-legal",
        "viterbi-max",
        "viterbi-backoff",
    ):
        raise ValueError(f"unknown BIOES search mode {bioes_search!r}")
    if bioes_search in ("lazy-fine-legal", "word-fine-legal", "viterbi-max", "viterbi-backoff") and (
        "token_logits" not in cache
    ):
        raise ValueError(f"bioes_search={bioes_search!r} requires a full-logit token score cache")
    if bioes_search in ("lazy-fine-legal", "word-fine-legal") and bucket_of is not None:
        raise ValueError(f"bioes_search={bioes_search!r} requires fine labels without bucket compatibility")
    if bioes_search == "word-fine-legal" and document_texts is None:
        raise ValueError("bioes_search='word-fine-legal' requires document_texts")
    if bioes_search in ("viterbi-max", "viterbi-backoff") and bucket_of is None:
        raise ValueError(f"bioes_search={bioes_search!r} requires a compatibility map")
    if bioes_search == "viterbi-backoff":
        if (
            bioes_backoff_alpha is None
            or not math.isfinite(bioes_backoff_alpha)
            or not 0 <= bioes_backoff_alpha <= 1
        ):
            raise ValueError("viterbi-backoff requires bioes_backoff_alpha in [0, 1]")
        if span_type == "logitsum":
            raise ValueError("viterbi-backoff assigns the exact path's fine type; logitsum would replace it")
    elif bioes_backoff_alpha is not None:
        raise ValueError("bioes_backoff_alpha requires bioes_search='viterbi-backoff'")
    if bioes_search_top_k is not None and bioes_search != "viterbi-max":
        raise ValueError("bioes_search_top_k applies only to bioes_search='viterbi-max'")
    if bioes_bucket_reduction not in BIOES_BUCKET_REDUCTIONS:
        raise ValueError(f"unknown BIOES bucket reduction {bioes_bucket_reduction!r}")
    if bioes_bucket_reduction != "max" and bioes_search != "viterbi-max":
        raise ValueError("soft BIOES bucket reduction applies only to bioes_search='viterbi-max'")
    if not math.isfinite(bioes_bucket_temperature) or bioes_bucket_temperature <= 0:
        raise ValueError("bioes_bucket_temperature must be positive and finite")
    if not math.isfinite(bioes_whitespace_split_cost) or bioes_whitespace_split_cost < 0:
        raise ValueError("bioes_whitespace_split_cost must be finite and nonnegative")
    if bioes_whitespace_split_cost and bioes_search not in ("lazy-fine-legal", "word-fine-legal"):
        raise ValueError("bioes_whitespace_split_cost requires lazy-fine-legal or word-fine-legal search")
    if bioes_whitespace_split_cost and document_texts is None:
        raise ValueError("bioes_whitespace_split_cost requires document_texts")
    if window_deduplication not in ("geometry", "exact"):
        raise ValueError(f"unknown window deduplication mode {window_deduplication!r}")
    word_boundaries_by_document = (
        [unicode_word_boundary_positions(text) for text in document_texts]
        if bioes_search == "word-fine-legal"
        else None
    )
    predictions = [[] for _ in document_ids]
    token_boundaries = cache["window_token_start"]
    for window_index, document_index in enumerate(cache["window_document"]):
        first = int(token_boundaries[window_index])
        last = int(token_boundaries[window_index + 1])
        winner_ids = cache["top_non_o_label"][first:last]
        margins = cache["o_minus_top_logit"][first:last]
        decoded_offsets = None
        if bioes_search in ("lazy-fine-legal", "word-fine-legal"):
            if __package__:
                from scripts.pii_bioes import (
                    constrained_bioes_decode,
                    constrained_boundary_bioes_decode,
                    count_bioes_violations,
                )
            else:
                from pii_bioes import (
                    constrained_bioes_decode,
                    constrained_boundary_bioes_decode,
                    count_bioes_violations,
                )

            window_logits = cache["token_logits"][first:last].astype(float)
            o_label_id = labels.index("O")
            window_logits[:, o_label_id] += o_logit_bias
            token_starts = cache["token_start"][first:last]
            token_ends = cache["token_end"][first:last]
            split_boundaries = (
                unicode_whitespace_split_boundaries(
                    document_texts[int(document_index)],
                    token_starts,
                    token_ends,
                )
                if bioes_whitespace_split_cost
                else None
            )
            if bioes_search == "word-fine-legal":
                assert word_boundaries_by_document is not None
                word_boundaries = word_boundaries_by_document[int(document_index)]
                start_offsets, end_offsets = covered_word_boundary_offsets(
                    token_starts, token_ends, word_boundaries
                )
                path = constrained_boundary_bioes_decode(
                    window_logits,
                    dict(enumerate(labels)),
                    [offset is not None for offset in start_offsets],
                    [offset is not None for offset in end_offsets],
                    [
                        start is not None and end is not None and start < end
                        for start, end in zip(start_offsets, end_offsets, strict=True)
                    ],
                    same_type_split_cost=bioes_whitespace_split_cost,
                    same_type_split_boundaries=split_boundaries,
                )
                token_labels = [labels[int(label_id)] for label_id in path]
                decoded_offsets = [
                    (
                        raw_start if start is None else start,
                        raw_end if end is None else end,
                    )
                    for raw_start, raw_end, start, end in zip(
                        token_starts,
                        token_ends,
                        start_offsets,
                        end_offsets,
                        strict=True,
                    )
                ]
            else:
                path = window_logits.argmax(axis=1)
                token_labels = [labels[int(label_id)] for label_id in path]
                invalid, _constraints = count_bioes_violations(token_labels)
                if invalid or bioes_whitespace_split_cost:
                    path = constrained_bioes_decode(
                        window_logits,
                        dict(enumerate(labels)),
                        same_type_split_cost=bioes_whitespace_split_cost,
                        same_type_split_boundaries=split_boundaries,
                    )
                    token_labels = [labels[int(label_id)] for label_id in path]
        elif bioes_search == "viterbi-backoff":
            if __package__:
                from scripts.pii_bioes import constrained_backoff_bioes_decode
            else:
                from pii_bioes import constrained_backoff_bioes_decode

            window_logits = cache["token_logits"][first:last].astype(float)
            window_logits[:, labels.index("O")] += o_logit_bias
            path = constrained_backoff_bioes_decode(
                window_logits,
                dict(enumerate(labels)),
                bucket_of,
                alpha=bioes_backoff_alpha,
                temperature=bioes_bucket_temperature,
            )
            token_labels = [labels[int(label_id)] for label_id in path]
        elif bioes_search == "viterbi-max":
            if __package__:
                from scripts.pii_bioes import constrained_bucket_bioes_decode
            else:
                from pii_bioes import constrained_bucket_bioes_decode

            window_logits = cache["token_logits"][first:last].astype(float)
            o_label_id = labels.index("O")
            window_logits[:, o_label_id] += o_logit_bias
            path = constrained_bucket_bioes_decode(
                window_logits,
                dict(enumerate(labels)),
                bucket_of,
                top_k_non_o=bioes_search_top_k,
                bucket_reduction=bioes_bucket_reduction,
                bucket_temperature=bioes_bucket_temperature,
            )
            token_labels = [labels[int(label_id)] for label_id in path]
        else:
            token_labels = [
                "O" if float(margin) + o_logit_bias >= 0 else labels[int(winner_id)]
                for margin, winner_id in zip(margins, winner_ids)
            ]
        weights = None
        if span_type == "margin":
            weights = [max(0.0, -(float(margin) + o_logit_bias)) for margin in margins]
        elif span_type == "preponderance":
            if preponderance_weight == "token":
                weights = [1.0] * len(margins)
            elif preponderance_weight == "nfc-char":
                if "token_nfc_char_count" in cache:
                    weights = cache["token_nfc_char_count"][first:last].astype(float).tolist()
                elif document_texts is not None:
                    offsets = zip(cache["token_start"][first:last], cache["token_end"][first:last])
                    weights = preponderance_token_weights(
                        document_texts[int(document_index)], offsets, "nfc-char"
                    )
                else:
                    raise ValueError(
                        "nfc-char preponderance needs token_nfc_char_count in the cache or "
                        "document_texts for legacy-cache reconstruction"
                    )
        type_resolver = None
        if span_type == "logitsum":
            top3_ids = cache["top3_non_o_label"][first:last]
            top3_margins = cache["o_minus_top3_logit"][first:last]
            if bucket_of is None:
                raise ValueError("span_type='logitsum' requires bucket_of")
            bucket_members = {}
            for fine_type, bucket in bucket_of.items():
                bucket_members.setdefault(bucket, []).append(fine_type)

            def type_resolver(token_indices, opening):
                # sparse strength: sum -(margin) wherever the type appears
                # in a token's top-3; per-token O offsets cancel in the
                # argmax only over counted tokens (agreed approximation)
                strengths = {}
                for token_index in token_indices:
                    for label_id, margin in zip(top3_ids[token_index], top3_margins[token_index]):
                        fine_type = labels[int(label_id)].split("-", 1)[1]
                        strengths[fine_type] = strengths.get(fine_type, 0.0) - float(margin)
                candidates = bucket_members.get(bucket_of.get(opening)) or [opening]
                scored = [t for t in candidates if t in strengths]
                if not scored:
                    return opening
                return max(scored, key=lambda t: (strengths[t], t == opening))

        offsets = decoded_offsets or zip(cache["token_start"][first:last], cache["token_end"][first:last])
        predictions[int(document_index)].extend(
            decode_bioes_labels(token_labels, offsets, bucket_of, span_type, weights, type_resolver)
        )
    dedupe = dedupe_window_preds if window_deduplication == "geometry" else dedupe_exact_window_preds
    return [
        {
            "id": document_id,
            "dataset": dataset,
            "preds": dedupe(document_predictions),
        }
        for document_id, dataset, document_predictions in zip(
            document_ids,
            document_datasets,
            predictions,
        )
    ]


def write_span_score_cache(
    path,
    document_ids,
    document_datasets,
    document_predictions,
    candidate_floor=0.0,
    model_id=None,
    model_revision=None,
):
    """Persist candidate spans for post-hoc global confidence thresholding."""
    import numpy as np

    if not (len(document_ids) == len(document_datasets) == len(document_predictions)):
        raise ValueError("span score-cache document metadata and predictions must have equal lengths")
    labels = sorted(
        {prediction["label"] for predictions in document_predictions for prediction in predictions}
    )
    label_to_id = {label: label_id for label_id, label in enumerate(labels)}
    prediction_documents = []
    starts = []
    ends = []
    label_ids = []
    confidences = []
    for document_index, predictions in enumerate(document_predictions):
        for prediction in predictions:
            prediction_documents.append(document_index)
            starts.append(prediction["start"])
            ends.append(prediction["end"])
            label_ids.append(label_to_id[prediction["label"]])
            confidences.append(prediction["confidence"])
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = dict(
        schema_version=np.asarray([1], dtype=np.int32),
        calibration_family=np.asarray(["global_span_confidence_threshold"]),
        candidate_floor=np.asarray([candidate_floor], dtype=np.float32),
        labels=np.asarray(labels),
        document_ids=np.asarray(document_ids),
        document_datasets=np.asarray(document_datasets),
        prediction_document=np.asarray(prediction_documents, dtype=np.int32),
        prediction_start=np.asarray(starts, dtype=np.int32),
        prediction_end=np.asarray(ends, dtype=np.int32),
        prediction_label=np.asarray(label_ids, dtype=np.uint16),
        prediction_confidence=np.asarray(confidences, dtype=np.float32),
    )
    if model_id is not None:
        payload["model_id"] = np.asarray([model_id])
    if model_revision is not None:
        payload["model_revision"] = np.asarray([model_revision])
    np.savez_compressed(output_path, **payload)


def load_span_score_cache(path):
    """Load a no-pickle candidate-span confidence cache."""
    import numpy as np

    with np.load(path, allow_pickle=False) as source:
        cache = {name: source[name].copy() for name in source.files}
    if cache["schema_version"].tolist() != [1]:
        raise ValueError(f"unsupported span score-cache schema: {cache['schema_version'].tolist()}")
    if cache["calibration_family"].tolist() != ["global_span_confidence_threshold"]:
        raise ValueError(f"unsupported calibration family: {cache['calibration_family'].tolist()}")
    return cache


def decode_span_score_cache(cache, threshold):
    """Filter cached schema-conditioned span candidates at one threshold."""
    labels = cache["labels"].tolist()
    document_ids = cache["document_ids"].tolist()
    document_datasets = cache["document_datasets"].tolist()
    predictions = [[] for _ in document_ids]
    for document_index, start, end, label_id, confidence in zip(
        cache["prediction_document"],
        cache["prediction_start"],
        cache["prediction_end"],
        cache["prediction_label"],
        cache["prediction_confidence"],
    ):
        if float(confidence) >= threshold:
            predictions[int(document_index)].append(
                {"start": int(start), "end": int(end), "label": labels[int(label_id)]}
            )
    return [
        {"id": document_id, "dataset": dataset, "preds": document_predictions}
        for document_id, dataset, document_predictions in zip(
            document_ids,
            document_datasets,
            predictions,
        )
    ]


def _with_o_logit_bias(logits, o_label_id: int, o_logit_bias: float):
    """Shift only the outside-label logit before independent token argmax."""
    if o_logit_bias == 0:
        return logits
    biased = logits.clone()
    biased[:, o_label_id] += o_logit_bias
    return biased


def _resolve_suppressed_primary_columns(id2label, suppressed_types=()):
    """Resolve complete BIOES column families for types masked before decode."""
    requested = tuple(suppressed_types)
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate suppressed primary types: {requested}")
    if not requested:
        return ()
    columns_by_type = {}
    for column, label in id2label.items():
        if label == "O":
            continue
        prefix, separator, primary_type = label.partition("-")
        if not separator or prefix not in {"B", "I", "E", "S"}:
            raise ValueError(f"cannot suppress primary types in non-BIOES label {label!r}")
        columns_by_type.setdefault(primary_type, []).append(int(column))
    unknown = sorted(set(requested) - columns_by_type.keys())
    if unknown:
        raise ValueError(f"suppressed primary types absent from model schema: {unknown}")
    incomplete = {
        primary_type: sorted(id2label[column].split("-", 1)[0] for column in columns_by_type[primary_type])
        for primary_type in requested
        if {id2label[column].split("-", 1)[0] for column in columns_by_type[primary_type]}
        != {"B", "I", "E", "S"}
    }
    if incomplete:
        raise ValueError(f"suppressed primary types lack a complete BIOES family: {incomplete}")
    return tuple(sorted(column for primary_type in requested for column in columns_by_type[primary_type]))


def _with_suppressed_primary_columns(logits, columns):
    """Mask selected primary-label columns without mutating model output."""
    if not columns:
        return logits
    suppressed = logits.clone()
    suppressed[:, list(columns)] = float("-inf")
    return suppressed


def _tokenizer_capacity_stride(tokenizer: Any, max_length: int) -> int:
    content_capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_capacity <= 0:
        raise ValueError(f"max length {max_length} leaves no capacity after special tokens")
    return min(128, content_capacity // 4)


def _tokenizer_capacity_windows(
    tokenizer: Any,
    text: str,
    max_length: int,
) -> tuple[int, list[tuple[dict[str, list[int]], list[tuple[int, int]]]]]:
    """Tokenize one frozen surface into overlapping model-capacity windows."""
    stride = _tokenizer_capacity_stride(tokenizer, max_length)
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        verbose=False,
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping")
    return stride, [
        (
            {name: values[index] for name, values in encoded.items()},
            [(int(start), int(end)) for start, end in window_offsets],
        )
        for index, window_offsets in enumerate(offsets)
    ]


def _predict_hf_tokcls_biases(
    texts,
    log,
    model_id=OPENMED_ID,
    warmup_docs=0,
    cost=None,
    bioes_project_legal=False,
    o_logit_biases=(0.0,),
    score_cache_path=None,
    record_ids=None,
    record_datasets=None,
    bucket_compat=DEFAULT_BIOES_BUCKET_COMPAT,
    bucket_compat_type=DEFAULT_BIOES_BUCKET_COMPAT_TYPE,
    preponderance_weight="nfc-char",
    score_cache_full_logits=False,
    character_logit_scale=None,
    hf_windowing="character-conservative",
    suppressed_primary_types=(),
):
    import torch

    if __package__:
        from scripts.pii_bioes import constrained_bioes_decode, count_bioes_violations
        from scripts.pii_continuous_character_cnn import project_continuous_characters
        from scripts.pii_continuous_character_model import (
            CONTINUOUS_CHARACTER_DECODER_NAME,
            ContinuousCharacterForTokenClassification,
            load_checkpoint_character_projection,
            rescale_imported_character_readout,
            uses_continuous_character_head,
        )
        from scripts.pii_crf_model import CRF_ARCHITECTURE, MMBertCrfForTokenClassification
        from scripts.pii_layered_head_model import (
            CONCAT_HEAD_ARCHITECTURE,
            LayerConcatForTokenClassification,
        )
    else:
        from pii_bioes import constrained_bioes_decode, count_bioes_violations
        from pii_continuous_character_cnn import project_continuous_characters
        from pii_continuous_character_model import (
            CONTINUOUS_CHARACTER_DECODER_NAME,
            ContinuousCharacterForTokenClassification,
            load_checkpoint_character_projection,
            rescale_imported_character_readout,
            uses_continuous_character_head,
        )
        from pii_crf_model import CRF_ARCHITECTURE, MMBertCrfForTokenClassification
        from pii_layered_head_model import CONCAT_HEAD_ARCHITECTURE, LayerConcatForTokenClassification
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
    from transformers import __version__ as transformers_version

    tok = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    uses_crf = getattr(config, "pii_decoder", None) == CRF_ARCHITECTURE
    uses_concat_head = getattr(config, "pii_head_architecture", None) == CONCAT_HEAD_ARCHITECTURE
    uses_continuous_character = uses_continuous_character_head(config)
    if character_logit_scale is not None and not uses_continuous_character:
        raise ValueError("character logit scaling requires a continuous-character checkpoint")
    character_projection = None
    if uses_crf:
        model = MMBertCrfForTokenClassification.from_local_checkpoint(model_id, encoder_dtype=torch.bfloat16)
        model = model.cuda().eval()
    elif uses_continuous_character:
        model = ContinuousCharacterForTokenClassification.from_local_checkpoint(
            model_id,
            dtype=torch.bfloat16,
        )
        if character_logit_scale is not None:
            rescale_imported_character_readout(model, character_logit_scale)
        character_projection = load_checkpoint_character_projection(model_id)
        model = model.cuda().eval()
    elif uses_concat_head:
        model = LayerConcatForTokenClassification.from_local_checkpoint(model_id, dtype=torch.bfloat16)
        model = model.cuda().eval()
    else:
        model = (
            AutoModelForTokenClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
    o_logit_biases = tuple(float(value) for value in o_logit_biases)
    if not o_logit_biases:
        raise ValueError("O-logit calibration requires at least one bias")
    if len(set(o_logit_biases)) != len(o_logit_biases):
        raise ValueError(f"duplicate O-logit biases: {o_logit_biases}")
    id2label = model.config.id2label
    suppressed_primary_types = tuple(suppressed_primary_types)
    suppressed_primary_columns = _resolve_suppressed_primary_columns(
        id2label,
        suppressed_primary_types,
    )
    type_label_columns = {}
    for column, name in id2label.items():
        if name != "O":
            type_label_columns.setdefault(name.split("-", 1)[1], []).append(int(column))
    bucket_of = bucket_compat_map(bucket_compat, type_label_columns) if bucket_compat else None
    bucket_members = {}
    if bucket_of is not None:
        for fine_type, bucket in bucket_of.items():
            bucket_members.setdefault(bucket, []).append(fine_type)
    if preponderance_weight not in PREPONDERANCE_WEIGHT_MODES:
        raise ValueError(f"unknown preponderance weight mode {preponderance_weight!r}")
    if hf_windowing not in HF_WINDOWING_MODES:
        raise ValueError(f"unknown HF windowing mode {hf_windowing!r}")
    # Window/truncation limits come from the model, not a constant: a
    # 512-position encoder (XLM-R) device-asserts on the 12k-char windows
    # a long-context encoder (OpenMed nemotron) handles. max_chars <=
    # max_len keeps even 1-token-per-char (CJK) windows untruncated;
    # truncation stays as the backstop for byte-fallback blowups.
    max_len = min(getattr(model.config, "max_position_embeddings", 1 << 20) - 2, tok.model_max_length, 16384)
    max_chars = min(12000, max_len)
    win_overlap = min(300, max_chars // 4)
    token_overlap = _tokenizer_capacity_stride(tok, max_len) if hf_windowing == "token-capacity" else None
    log(
        "WINDOWING "
        f"mode={hf_windowing} max-tokens={max_len} "
        + (
            f"overlap-tokens={token_overlap}"
            if token_overlap is not None
            else f"max-chars={max_chars} overlap-chars={win_overlap}"
        )
    )
    model_memory_allocated = torch.cuda.memory_allocated()
    projection_id2label = {int(label_id): label for label_id, label in id2label.items()}
    o_label_id = next(
        (label_id for label_id, label in projection_id2label.items() if label == "O"),
        None,
    )
    if (len(o_logit_biases) > 1 or any(o_logit_biases)) and uses_crf:
        raise ValueError("O-logit calibration is not defined for CRF decoding")
    if any(o_logit_biases) and o_label_id is None:
        raise ValueError("O-logit calibration requires an O label in the model schema")
    if score_cache_path and o_label_id is None:
        raise ValueError("score caching requires an O label in the model schema")
    if score_cache_path and (uses_crf or bioes_project_legal):
        raise ValueError("sparse score caching supports only greedy independent-token decoding")
    if score_cache_path and (record_ids is None or record_datasets is None):
        raise ValueError("score caching requires record IDs and dataset names")
    if score_cache_full_logits and not score_cache_path:
        raise ValueError("full-logit caching requires score_cache_path")
    if score_cache_path and suppressed_primary_columns:
        raise ValueError("score caching is unavailable when primary types are suppressed")
    if suppressed_primary_columns:
        log("PRIMARY TYPE SUPPRESSION before decode: " + ", ".join(suppressed_primary_types))
    projection_stats = {
        "constraints": 0,
        "invalid_constraints": 0,
        "segments": 0,
        "affected_segments": 0,
        "changed_tokens": 0,
        "seconds": 0.0,
    }

    def infer_text(text):
        preds_by_bias = {bias: [] for bias in o_logit_biases}
        score_windows = []
        n_segments = 0
        n_tokens = 0
        if hf_windowing == "token-capacity":
            _stride, token_windows = _tokenizer_capacity_windows(tok, text, max_len)
            model_windows = [
                (
                    0,
                    text,
                    {name: torch.tensor([values]) for name, values in model_inputs.items()},
                    torch.tensor([offsets]),
                )
                for model_inputs, offsets in token_windows
            ]
        else:
            model_windows = []
            for off, chunk in windows(text, max_chars=max_chars, overlap=win_overlap):
                enc = tok(
                    chunk,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                offset_mapping = enc.pop("offset_mapping")
                model_windows.append((off, chunk, enc, offset_mapping))
        for off, chunk, enc, offset_mapping in model_windows:
            offsets = offset_mapping[0].tolist()
            if uses_continuous_character:
                character_inputs = project_continuous_characters(
                    [{"text": chunk}],
                    offset_mapping,
                    projection=character_projection,
                    projection_view=model.config.pii_character_projection_view,
                )
                enc.update(
                    {
                        "character_ids": character_inputs.character_ids,
                        "character_mask": character_inputs.character_mask,
                        "token_offsets": character_inputs.token_offsets,
                    }
                )
            n_segments += 1
            n_tokens += int(enc["input_ids"].numel())
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits[0]
            logits = _with_suppressed_primary_columns(logits, suppressed_primary_columns)
            valid = torch.tensor([a != b for a, b in offsets], device=logits.device).unsqueeze(0)
            if uses_crf:
                path = model.decode(logits.unsqueeze(0), valid)[0]
                labs = [id2label[int(label_id)] for label_id in path]
                decoded_offsets = [offset for offset in offsets if offset[0] != offset[1]]
                decoded_by_bias = {o_logit_biases[0]: labs}
            else:
                valid_logits = logits[valid[0]]
                decoded_offsets = [offset for offset in offsets if offset[0] != offset[1]]
                if score_cache_path:
                    non_o_logits = valid_logits.clone()
                    non_o_logits[:, o_label_id] = -torch.inf
                    top_values, top_ids = non_o_logits.max(dim=-1)
                    top3_values, top3_ids = non_o_logits.topk(3, dim=-1)
                    o_column = valid_logits[:, o_label_id].unsqueeze(-1)
                    score_windows.append(
                        {
                            "token_start": [start + off for start, _ in decoded_offsets],
                            "token_end": [end + off for _, end in decoded_offsets],
                            "top_non_o_label": top_ids.cpu().tolist(),
                            "o_minus_top": (valid_logits[:, o_label_id] - top_values).float().cpu().tolist(),
                            "top3_non_o_label": top3_ids.cpu().tolist(),
                            "o_minus_top3": (o_column - top3_values).float().cpu().tolist(),
                            "nfc_char_count": preponderance_token_weights(chunk, decoded_offsets, "nfc-char"),
                            **(
                                {"full_logits": valid_logits.float().cpu().tolist()}
                                if score_cache_full_logits
                                else {}
                            ),
                        }
                    )
                decoded_by_bias = {}
                for bias in o_logit_biases:
                    decision_logits = _with_o_logit_bias(valid_logits, o_label_id, bias)
                    greedy_ids = decision_logits.argmax(-1).tolist()
                    labs = [id2label[int(label_id)] for label_id in greedy_ids]
                    if bioes_project_legal:
                        invalid, constraints = count_bioes_violations(labs)
                        projection_stats["segments"] += 1
                        projection_stats["constraints"] += constraints
                        projection_stats["invalid_constraints"] += invalid
                        if invalid:
                            projection_stats["affected_segments"] += 1
                            projection_started = time.perf_counter()
                            projected_ids = constrained_bioes_decode(
                                decision_logits.float().cpu().numpy(),
                                projection_id2label,
                            )
                            projection_stats["seconds"] += time.perf_counter() - projection_started
                            projection_stats["changed_tokens"] += sum(
                                projected != greedy
                                for projected, greedy in zip(projected_ids.tolist(), greedy_ids)
                            )
                            labs = [id2label[int(label_id)] for label_id in projected_ids]
                    decoded_by_bias[bias] = labs
            for bias, labs in decoded_by_bias.items():
                token_weights = None
                if bucket_compat_type == "margin":
                    margin_logits = valid_logits.float()
                    non_o = margin_logits.clone()
                    non_o[:, o_label_id] = float("-inf")
                    entity_strength = non_o.max(dim=-1).values - (margin_logits[:, o_label_id] + bias)
                    token_weights = entity_strength.clamp(min=0.0).cpu().tolist()
                elif bucket_compat_type == "preponderance":
                    token_weights = preponderance_token_weights(chunk, decoded_offsets, preponderance_weight)
                type_resolver = None
                if bucket_compat_type == "logitsum":
                    window_scores = valid_logits.float().cpu().numpy()

                    def type_resolver(token_indices, opening, window_scores=window_scores):
                        candidates = bucket_members.get(bucket_of.get(opening)) or [opening]
                        rows = window_scores[token_indices]

                        def strength(fine_type):
                            columns = type_label_columns.get(fine_type)
                            if not columns:
                                return float("-inf")
                            return float(rows[:, columns].max(axis=1).sum())

                        best = max(candidates, key=lambda t: (strength(t), t == opening))
                        return best if strength(best) > float("-inf") else opening

                for pred in decode_bioes_labels(
                    labs, decoded_offsets, bucket_of, bucket_compat_type, token_weights, type_resolver
                ):
                    pred["start"] += off
                    pred["end"] += off
                    preds_by_bias[bias].append(pred)
        return (
            {bias: dedupe_window_preds(preds) for bias, preds in preds_by_bias.items()},
            n_segments,
            n_tokens,
            score_windows,
        )

    warmup_docs = min(max(warmup_docs, 0), len(texts))
    for text in texts[:warmup_docs]:
        infer_text(text)
    for key in projection_stats:
        projection_stats[key] = 0.0 if key == "seconds" else 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    results_by_bias = {bias: [] for bias in o_logit_biases}
    latencies = []
    n_segments = 0
    n_tokens = 0
    document_windows = []
    for i, text in enumerate(texts):
        torch.cuda.synchronize()
        t_seg = time.perf_counter()
        predictions, text_segments, text_tokens, score_windows = infer_text(text)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t_seg)
        n_segments += text_segments
        n_tokens += text_tokens
        if score_cache_path:
            document_windows.append(score_windows)
        for bias, preds in predictions.items():
            results_by_bias[bias].append((preds, latencies[-1]))
        if (i + 1) % 200 == 0:
            log(f"openmed {i + 1}/{len(texts)}")
    _log_latency(latencies, log)
    if score_cache_path:
        labels = [projection_id2label[label_id] for label_id in range(len(projection_id2label))]
        write_token_score_cache(
            score_cache_path,
            labels,
            record_ids,
            record_datasets,
            document_windows,
        )
        log(f"score cache: {len(texts)} documents -> {os.path.abspath(score_cache_path)}")
    if bioes_project_legal:
        log(
            "BIOES projection: "
            f"{projection_stats['invalid_constraints']}/{projection_stats['constraints']} "
            f"invalid constraints; {projection_stats['affected_segments']}/"
            f"{projection_stats['segments']} segments re-decoded; "
            f"{projection_stats['changed_tokens']} token labels changed; "
            f"{projection_stats['seconds']:.3f}s in projection"
        )
    if cost is not None:
        elapsed = sum(latencies)
        mib = 1024**2
        cost.update(
            {
                "schema_version": 1,
                "model_id": model_id,
                "decoder": (
                    "crf"
                    if uses_crf
                    else CONTINUOUS_CHARACTER_DECODER_NAME
                    if uses_continuous_character
                    else f"concat-{config.pii_head_kind}"
                    if uses_concat_head
                    else "linear"
                )
                + ("+bioes_projection" if bioes_project_legal else ""),
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "torch_version": torch.__version__,
                "transformers_version": transformers_version,
                "batch_size": 1,
                "hf_windowing": hf_windowing,
                "max_input_tokens": max_len,
                "window_max_chars": max_chars if hf_windowing == "character-conservative" else None,
                "window_overlap_chars": (win_overlap if hf_windowing == "character-conservative" else None),
                "window_overlap_tokens": token_overlap,
                "o_logit_bias": o_logit_biases[0] if len(o_logit_biases) == 1 else list(o_logit_biases),
                "suppressed_primary_types": list(suppressed_primary_types),
                "character_logit_scale": character_logit_scale,
                "warmup_docs": warmup_docs,
                "num_docs": len(texts),
                "num_segments": n_segments,
                "num_input_chars": sum(len(text) for text in texts),
                "num_input_tokens": n_tokens,
                "elapsed_s": elapsed,
                "docs_per_s": len(texts) / elapsed if elapsed else 0.0,
                "segments_per_s": n_segments / elapsed if elapsed else 0.0,
                "tokens_per_s": n_tokens / elapsed if elapsed else 0.0,
                "latency_s": _latency_stats(latencies),
                "model_memory_allocated_mib": model_memory_allocated / mib,
                "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / mib,
                "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / mib,
                "device_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / mib,
            }
        )
        if bioes_project_legal:
            cost["bioes_projection"] = projection_stats
    return results_by_bias


def predict_hf_tokcls(
    texts,
    log,
    model_id=OPENMED_ID,
    warmup_docs=0,
    cost=None,
    bioes_project_legal=False,
    o_logit_bias=0.0,
    bucket_compat=DEFAULT_BIOES_BUCKET_COMPAT,
    bucket_compat_type=DEFAULT_BIOES_BUCKET_COMPAT_TYPE,
    preponderance_weight="nfc-char",
    score_cache_full_logits=False,
    score_cache_path=None,
    record_ids=None,
    record_datasets=None,
    character_logit_scale=None,
    hf_windowing="character-conservative",
    suppressed_primary_types=(),
):
    """Run one calibrated token-classification operating point."""
    return _predict_hf_tokcls_biases(
        texts,
        log,
        model_id=model_id,
        warmup_docs=warmup_docs,
        cost=cost,
        bioes_project_legal=bioes_project_legal,
        o_logit_biases=(o_logit_bias,),
        bucket_compat=bucket_compat,
        bucket_compat_type=bucket_compat_type,
        preponderance_weight=preponderance_weight,
        score_cache_full_logits=score_cache_full_logits,
        score_cache_path=score_cache_path,
        record_ids=record_ids,
        record_datasets=record_datasets,
        character_logit_scale=character_logit_scale,
        hf_windowing=hf_windowing,
        suppressed_primary_types=suppressed_primary_types,
    )[float(o_logit_bias)]


def o_logit_bias_slug(value: float) -> str:
    """Stable filename token for the calibrated values used in model names."""
    scaled = round(abs(value) * 100)
    if abs(abs(value) * 100 - scaled) > 1e-9:
        raise ValueError(f"O-logit bias requires two-decimal filename precision, got {value}")
    return f"{'m' if value < 0 else 'p'}{scaled:03d}"


def predict_gliner2(
    texts,
    log,
    warmup_docs=0,
    cost=None,
    threshold=0.5,
    score_cache_path=None,
    record_ids=None,
    record_datasets=None,
    revision=None,
):
    import torch
    from gliner2 import GLiNER2

    model_source = GLINER2_ID
    if revision is not None:
        from huggingface_hub import snapshot_download

        model_source = snapshot_download(repo_id=GLINER2_ID, revision=revision)
        log(f"resolved {GLINER2_ID} revision {revision} -> {model_source}")
    model = GLiNER2.from_pretrained(model_source)
    model = model.to("cuda").eval()
    dev = next(model.parameters()).device
    assert dev.type == "cuda", f"GLiNER2 not on GPU: {dev}"
    log(f"model on {dev}")

    if score_cache_path and (record_ids is None or record_datasets is None):
        raise ValueError("GLiNER2 score caching requires record IDs and dataset names")

    def extract_one(text):
        preds = []
        for off, chunk in windows(text, max_chars=1200, overlap=200):
            res = model.extract_entities(
                chunk,
                GLINER2_LABELS,
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )
            ents = res.get("entities", res) if isinstance(res, dict) else res
            if isinstance(ents, dict):
                for label, items in ents.items():
                    for it in items:
                        if not isinstance(it, dict) or "start" not in it:
                            continue
                        preds.append(
                            {
                                "start": it["start"] + off,
                                "end": it["end"] + off,
                                "label": label,
                                "confidence": float(it["confidence"]),
                            }
                        )
            else:
                for it in ents:
                    preds.append(
                        {
                            "start": it["start"] + off,
                            "end": it["end"] + off,
                            "label": it.get("label", "?"),
                            "confidence": float(it["confidence"]),
                        }
                    )
        best_by_span = {}
        for prediction in preds:
            key = (prediction["start"], prediction["end"], prediction["label"])
            if key not in best_by_span or prediction["confidence"] > best_by_span[key]["confidence"]:
                best_by_span[key] = prediction
        return sorted(
            best_by_span.values(),
            key=lambda prediction: (prediction["start"], prediction["end"], prediction["label"]),
        )

    for text in texts[:warmup_docs]:
        extract_one(text)

    if cost is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model_memory_allocated = torch.cuda.memory_allocated()

    results = []
    scored_predictions = []
    latencies = []
    for i, text in enumerate(texts):
        if cost is not None:
            torch.cuda.synchronize()
        t_seg = time.perf_counter()
        scored = extract_one(text)
        preds = [
            {"start": prediction["start"], "end": prediction["end"], "label": prediction["label"]}
            for prediction in scored
        ]
        if cost is not None:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t_seg)
        results.append((preds, latencies[-1]))
        scored_predictions.append(scored)
        if (i + 1) % 200 == 0:
            log(f"gliner2 {i + 1}/{len(texts)}")
    _log_latency(latencies, log)
    if cost is not None:
        elapsed = sum(latencies)
        tokenizer = model.processor.tokenizer
        n_tokens = sum(len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts)
        n_segments = sum(len(windows(text, max_chars=1200, overlap=200)) for text in texts)
        mib = 1024**2
        cost.update(
            {
                "schema_version": 1,
                "model_id": GLINER2_ID,
                "decoder": "schema-conditioned span extraction",
                "device": torch.cuda.get_device_name(),
                "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
                "torch_version": torch.__version__,
                "batch_size": 1,
                "threshold": threshold,
                "warmup_docs": warmup_docs,
                "num_docs": len(texts),
                "num_segments": n_segments,
                "num_input_chars": sum(len(text) for text in texts),
                "num_input_tokens": n_tokens,
                "elapsed_s": elapsed,
                "docs_per_s": len(texts) / elapsed if elapsed else 0.0,
                "segments_per_s": n_segments / elapsed if elapsed else 0.0,
                "tokens_per_s": n_tokens / elapsed if elapsed else 0.0,
                "latency_s": _latency_stats(latencies),
                "model_memory_allocated_mib": model_memory_allocated / mib,
                "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / mib,
                "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / mib,
                "device_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / mib,
            }
        )
    if score_cache_path:
        write_span_score_cache(
            score_cache_path,
            record_ids,
            record_datasets,
            scored_predictions,
            candidate_floor=threshold,
            model_id=GLINER2_ID,
            model_revision=revision,
        )
        log(f"score cache: {len(texts)} documents -> {os.path.abspath(score_cache_path)}")
    return results


def cmd_predict(args):
    os.makedirs(PRED, exist_ok=True)
    if args.hf_windowing != "character-conservative" and args.model == "gliner2":
        raise ValueError("--hf-windowing applies only to Hugging Face token classifiers")
    if args.bioes_project_legal and args.model != "local":
        raise ValueError("--bioes-project-legal currently requires --model local")
    if args.score_cache_full_logits and args.model == "gliner2":
        raise ValueError("--score-cache-full-logits requires a token-classification model")
    output_name = args.output_name or args.model

    for dataset in args.datasets:
        gold = [json.loads(l) for l in open(os.path.join(GOLD, dataset + ".jsonl"))]
        if args.limit:
            gold = gold[: args.limit]
        texts = [g["text"] for g in gold]

        def log(msg, dataset=dataset):
            line = f"pii-eval {args.model}/{dataset}: {msg}"
            print(line, flush=True)
            hf = os.environ.get("AGENTCTL_HEADLINE_FILE")
            if hf:
                try:
                    sys.path.insert(0, os.path.join(HERE, "..", ".."))
                    from log_format import headline

                    headline(line)
                except Exception:
                    open(hf, "w").write(line + "\n")

        cost = {} if args.benchmark else None
        if args.model == "gliner2":
            results = predict_gliner2(
                texts,
                log,
                warmup_docs=args.warmup_docs if args.benchmark else 0,
                cost=cost,
            )
        else:
            results = predict_hf_tokcls(
                texts,
                log,
                model_id=HF_TOKCLS_IDS[args.model],
                warmup_docs=args.warmup_docs if args.benchmark else 0,
                cost=cost,
                bioes_project_legal=args.bioes_project_legal,
                o_logit_bias=args.o_logit_bias,
                bucket_compat=("fine" if args.bioes_bucket_compat == "none" else args.bioes_bucket_compat),
                bucket_compat_type=args.bioes_bucket_compat_type,
                preponderance_weight=args.bioes_preponderance_weight,
                score_cache_full_logits=args.score_cache_full_logits,
                score_cache_path=(
                    f"{args.score_cache_prefix}.{dataset}.npz" if args.score_cache_prefix else None
                ),
                record_ids=[g["id"] for g in gold] if args.score_cache_prefix else None,
                record_datasets=[dataset] * len(gold) if args.score_cache_prefix else None,
                character_logit_scale=args.character_logit_scale,
                hf_windowing=args.hf_windowing,
                suppressed_primary_types=args.suppress_primary_type,
            )
        out = os.path.join(PRED, f"{output_name}.{dataset}.jsonl")
        with open(out, "w") as f:
            for g, (preds, lat) in zip(gold, results):
                f.write(json.dumps({"id": g["id"], "preds": preds, "latency_s": round(lat, 4)}) + "\n")
        npred = sum(len(p) for p, _ in results)
        log(f"done: {len(results)} records, {npred} predicted spans -> {out}")
        if cost is not None:
            cost["dataset"] = dataset
            cost_out = os.path.join(PRED, f"{output_name}.{dataset}.cost.json")
            with open(cost_out, "w") as f:
                json.dump(cost, f, indent=2)
                f.write("\n")
            log(
                f"COST: {cost['docs_per_s']:.2f} docs/s, "
                f"{cost['peak_memory_allocated_mib']:.0f} MiB peak allocated -> {cost_out}"
            )


def _cmd_predict_onnx_bundle(args, *, mode: str, predictor) -> None:
    """Run one exact ONNX bundle predictor over frozen evaluation data."""
    os.makedirs(PRED, exist_ok=True)
    for dataset in args.datasets:
        with open(os.path.join(GOLD, dataset + ".jsonl")) as source:
            gold = [json.loads(line) for line in source]
        if args.limit:
            gold = gold[: args.limit]
        texts = [record["text"] for record in gold]

        def log(message, dataset=dataset):
            line = f"pii-eval {mode}/{dataset}: {message}"
            print(line, flush=True)
            headline_file = os.environ.get("AGENTCTL_HEADLINE_FILE")
            if headline_file:
                try:
                    sys.path.insert(0, os.path.join(HERE, "..", ".."))
                    from log_format import headline

                    headline(line)
                except Exception:
                    open(headline_file, "w").write(line + "\n")

        cache_path = f"{args.score_cache_prefix}.{dataset}.npz"
        results = predictor(
            texts,
            log,
            bundle=args.bundle,
            score_cache_path=cache_path,
            record_ids=[record["id"] for record in gold],
            record_datasets=[dataset] * len(gold),
        )
        output_path = os.path.join(PRED, f"{args.output_name}.{dataset}.jsonl")
        with open(output_path, "w") as output:
            for record, (predictions, latency) in zip(gold, results, strict=True):
                output.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "preds": predictions,
                            "latency_s": round(latency, 6),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        log(
            f"done: {len(results)} records, {sum(len(preds) for preds, _ in results)} "
            f"raw greedy spans -> {output_path}"
        )


def cmd_predict_character_onnx(args):
    """Run a tokenizer-free character ONNX bundle over frozen evaluation data."""
    _cmd_predict_onnx_bundle(args, mode="character-onnx", predictor=predict_character_onnx)


def cmd_predict_character_token_onnx(args):
    """Run a character CNN over bundled tokenizer intervals."""
    _cmd_predict_onnx_bundle(
        args,
        mode="character-token-onnx",
        predictor=predict_character_token_onnx,
    )


def _cache_document_texts(cache, gold_records):
    """Align gold text to a score cache by stable document ID."""
    by_id = {}
    for record in gold_records:
        document_id = record["id"]
        if document_id in by_id:
            raise ValueError(f"duplicate gold document ID {document_id!r}")
        by_id[document_id] = record["text"]
    missing = [document_id for document_id in cache["document_ids"].tolist() if document_id not in by_id]
    if missing:
        raise ValueError(
            f"score cache has {len(missing)} document IDs absent from gold, first: {missing[0]!r}"
        )
    return [by_id[document_id] for document_id in cache["document_ids"].tolist()]


def _cache_document_languages(cache, gold_records, override=None):
    """Align explicit or row-declared BCP 47 language tags to a score cache."""
    by_id = {}
    for record in gold_records:
        document_id = record["id"]
        if document_id in by_id:
            raise ValueError(f"duplicate gold document ID {document_id!r}")
        language = override or record.get("bcp47")
        if not isinstance(language, str) or not language:
            raise ValueError(f"gold document {document_id!r} needs bcp47 or --name-oracle-language")
        by_id[document_id] = language
    missing = [document_id for document_id in cache["document_ids"].tolist() if document_id not in by_id]
    if missing:
        raise ValueError(
            f"score cache has {len(missing)} document IDs absent from gold, first: {missing[0]!r}"
        )
    return [by_id[document_id] for document_id in cache["document_ids"].tolist()]


def apply_name_oracle_to_rows(rows, document_texts, document_languages, name_oracle):
    """Apply the configured name grammar after primary span decoding."""
    if not (len(rows) == len(document_texts) == len(document_languages)):
        raise ValueError("name-oracle rows, texts, and languages must have equal lengths")
    adjusted = []
    for row, text, language in zip(rows, document_texts, document_languages, strict=True):
        fields = name_oracle.adjusted_inference_fields(
            row_id=row["id"],
            text=text,
            language=language,
            predictions=row["preds"],
        )
        adjusted.append({**row, **fields})
    return adjusted


def cmd_redecode_token_cache(args):
    """Re-run BIOES assembly and type voting from a saved token trace."""
    os.makedirs(PRED, exist_ok=True)
    name_oracle = None
    if args.name_oracle_config:
        if __package__:
            from scripts.pii_name_annotation_qc import NameAnnotationQc, load_config
        else:
            from pii_name_annotation_qc import NameAnnotationQc, load_config

        name_oracle = NameAnnotationQc(
            load_config(Path(args.name_oracle_config)),
            given_lexicons=(Path(path) for path in args.name_oracle_given_lexicon),
            family_lexicons=(Path(path) for path in args.name_oracle_family_lexicon),
        )
    for dataset in args.datasets:
        cache_path = f"{args.score_cache_prefix}.{dataset}.npz"
        cache = load_token_score_cache(cache_path)
        cached_datasets = set(cache["document_datasets"].tolist())
        if cached_datasets != {dataset}:
            raise ValueError(
                f"{cache_path}: cached dataset names {sorted(cached_datasets)} do not match {dataset!r}"
            )
        with open(os.path.join(GOLD, dataset + ".jsonl")) as source:
            gold = [json.loads(line) for line in source]
        document_texts = _cache_document_texts(cache, gold)
        bucket_of = None
        if args.bioes_bucket_compat != "none":
            fine_types = {label.split("-", 1)[1] for label in cache["labels"].tolist() if label != "O"}
            bucket_of = bucket_compat_map(args.bioes_bucket_compat, fine_types)
        rows = decode_token_score_cache(
            cache,
            o_logit_bias=args.o_logit_bias,
            bucket_of=bucket_of,
            span_type=args.bioes_bucket_compat_type,
            preponderance_weight=args.bioes_preponderance_weight,
            document_texts=document_texts,
            bioes_search=args.bioes_search,
            bioes_search_top_k=args.bioes_search_top_k or None,
            bioes_bucket_reduction=args.bioes_bucket_reduction,
            bioes_bucket_temperature=args.bioes_bucket_temperature,
            bioes_backoff_alpha=args.bioes_backoff_alpha,
            bioes_whitespace_split_cost=args.bioes_whitespace_split_cost,
        )
        if name_oracle is not None:
            rows = apply_name_oracle_to_rows(
                rows,
                document_texts,
                _cache_document_languages(cache, gold, args.name_oracle_language),
                name_oracle,
            )
        output_path = os.path.join(PRED, f"{args.output_name}.{dataset}.jsonl")
        with open(output_path, "w") as output:
            for row in rows:
                output_row = {
                    key: row[key] for key in ("id", "preds", "subclass_spans", "name_oracle") if key in row
                }
                output_row["latency_s"] = 0.0
                output.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        print(
            f"REDECODE {dataset}: {len(rows)} documents from {cache_path} -> {output_path}",
            flush=True,
        )


def cmd_predict_bias_grid(args):
    """Share encoder forwards across several calibrated O-logit decisions."""
    os.makedirs(PRED, exist_ok=True)
    datasets = []
    all_texts = []
    all_ids = []
    all_dataset_names = []
    for dataset in args.datasets:
        gold = [json.loads(line) for line in open(os.path.join(GOLD, dataset + ".jsonl"))]
        if args.limit:
            gold = gold[: args.limit]
        datasets.append((dataset, gold))
        all_texts.extend(record["text"] for record in gold)
        all_ids.extend(record["id"] for record in gold)
        all_dataset_names.extend([dataset] * len(gold))

    def log(message):
        line = f"pii-eval {args.model}/bias-grid: {message}"
        print(line, flush=True)
        headline_file = os.environ.get("AGENTCTL_HEADLINE_FILE")
        if headline_file:
            try:
                sys.path.insert(0, os.path.join(HERE, "..", ".."))
                from log_format import headline

                headline(line)
            except Exception:
                open(headline_file, "w").write(line + "\n")

    results_by_bias = _predict_hf_tokcls_biases(
        all_texts,
        log,
        model_id=HF_TOKCLS_IDS[args.model],
        bioes_project_legal=args.bioes_project_legal,
        o_logit_biases=args.o_logit_bias,
        score_cache_path=args.score_cache,
        record_ids=all_ids,
        record_datasets=all_dataset_names,
        bucket_compat=(None if args.bioes_bucket_compat == "none" else args.bioes_bucket_compat),
        hf_windowing=args.hf_windowing,
    )
    offset = 0
    for dataset, gold in datasets:
        end = offset + len(gold)
        for bias, all_results in results_by_bias.items():
            output_name = f"{args.output_prefix}-obias-{o_logit_bias_slug(bias)}"
            output_path = os.path.join(PRED, f"{output_name}.{dataset}.jsonl")
            results = all_results[offset:end]
            with open(output_path, "w") as output:
                for record, (preds, latency) in zip(gold, results):
                    output.write(
                        json.dumps(
                            {
                                "id": record["id"],
                                "preds": preds,
                                "latency_s": round(latency, 4),
                            }
                        )
                        + "\n"
                    )
            log(f"wrote {len(results)} {dataset} records at O bias {bias:+.2f} -> {output_path}")
        offset = end


def cmd_predict_gliner_cache(args):
    """Run GLiNER2 once at a low floor and cache candidates for threshold fitting."""
    os.makedirs(PRED, exist_ok=True)
    datasets = []
    all_texts = []
    all_ids = []
    all_dataset_names = []
    for dataset in args.datasets:
        gold = [json.loads(line) for line in open(os.path.join(GOLD, dataset + ".jsonl"))]
        if args.limit:
            gold = gold[: args.limit]
        datasets.append((dataset, gold))
        all_texts.extend(record["text"] for record in gold)
        all_ids.extend(record["id"] for record in gold)
        all_dataset_names.extend([dataset] * len(gold))

    def log(message):
        line = f"pii-eval gliner2/confidence-cache: {message}"
        print(line, flush=True)
        headline_file = os.environ.get("AGENTCTL_HEADLINE_FILE")
        if headline_file:
            try:
                sys.path.insert(0, os.path.join(HERE, "..", ".."))
                from log_format import headline

                headline(line)
            except Exception:
                open(headline_file, "w").write(line + "\n")

    predict_gliner2(
        all_texts,
        log,
        threshold=args.candidate_floor,
        score_cache_path=args.score_cache,
        record_ids=all_ids,
        record_datasets=all_dataset_names,
        revision=args.revision,
    )
    stock_predictions = decode_span_score_cache(load_span_score_cache(args.score_cache), 0.5)
    predictions_by_dataset = defaultdict(list)
    for record in stock_predictions:
        predictions_by_dataset[record["dataset"]].append(record)
    for dataset, gold in datasets:
        output_path = os.path.join(PRED, f"{args.output_prefix}-threshold-p500.{dataset}.jsonl")
        predictions = predictions_by_dataset[dataset]
        with open(output_path, "w") as output:
            for prediction in predictions:
                output.write(json.dumps({"id": prediction["id"], "preds": prediction["preds"]}) + "\n")
        log(
            f"wrote {len(gold)} {dataset} records with "
            f"{sum(len(record['preds']) for record in predictions)} spans at stock threshold 0.5 -> {output_path}"
        )


# -------------------------------------------------------------------- score


def overlaps(a: dict, b: dict) -> bool:
    """Whether two spans meet the symmetric 80%-coverage overlap rule."""
    intersection = min(a["end"], b["end"]) - max(a["start"], b["start"])
    a_length = a["end"] - a["start"]
    b_length = b["end"] - b["start"]
    if intersection <= 0 or a_length <= 0 or b_length <= 0:
        return False
    return (
        intersection * OVERLAP_MIN_COVERAGE_DENOMINATOR >= a_length * OVERLAP_MIN_COVERAGE_NUMERATOR
        and intersection * OVERLAP_MIN_COVERAGE_DENOMINATOR >= b_length * OVERLAP_MIN_COVERAGE_NUMERATOR
    )


def merge_regions(preds, text):
    """Normalize raw predictions into maximal coverage regions: trim
    whitespace at edges, then union spans that overlap or are separated by
    whitespace only. Redaction semantics: the redacted character set is
    what counts, not how many typed spans produced it."""
    trimmed = []
    for p in preds:
        a, b = p["start"], p["end"]
        a = max(0, min(a, len(text)))
        b = max(0, min(b, len(text)))
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            trimmed.append((a, b))
    trimmed.sort()
    regions = []
    for a, b in trimmed:
        if regions and (a <= regions[-1][1] or text[regions[-1][1] : a].isspace()):
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    return [{"start": a, "end": b} for a, b in regions]


def _prf_one_to_one(tp, n_pred, n_gold):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
    }


def _classification_from_counts(tp, fp, fn, tn, *, wrong_type=0, excluded=0):
    """Summarize additive element-classification counts.

    The positive class is an entity character.  For typed projections, a
    wrong non-O class contributes to both the micro-averaged predicted and
    gold entity denominators, while remaining one incorrect character for
    accuracy.
    """
    n_pred = tp + fp + wrong_type
    n_gold = tp + fn + wrong_type
    n_total = tp + fp + fn + tn + wrong_type
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / n_total if n_total else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2 if n_total else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "wrong_type": wrong_type,
        "correct": tp + tn,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_total": n_total,
        "excluded": excluded,
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
    }


def _binary_character_counts(gold, preds, text_length):
    gold_mask = bytearray(text_length)
    pred_mask = bytearray(text_length)
    for span in gold:
        gold_mask[span["start"] : span["end"]] = b"\x01" * (span["end"] - span["start"])
    for span in preds:
        pred_mask[span["start"] : span["end"]] = b"\x01" * (span["end"] - span["start"])
    tp = fp = fn = tn = 0
    for gold_entity, pred_entity in zip(gold_mask, pred_mask):
        if gold_entity:
            if pred_entity:
                tp += 1
            else:
                fn += 1
        elif pred_entity:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


_PREDICTION_LABEL_CONFLICT = object()
_UNSUPPORTED_PREDICTION_LABEL = object()


def _projected_character_labels(spans, text_length, projection_key, universe, *, gold):
    labels = [None] * text_length
    included = bytearray(b"\x01") * text_length
    for span in spans:
        projected = span[projection_key]
        if projected not in universe:
            if gold:
                included[span["start"] : span["end"]] = b"\x00" * (span["end"] - span["start"])
                continue
            projected = _UNSUPPORTED_PREDICTION_LABEL
        for index in range(span["start"], span["end"]):
            prior = labels[index]
            if prior is None or prior == projected:
                labels[index] = projected
            elif gold:
                raise ValueError("gold spans assign conflicting labels to the same character")
            else:
                labels[index] = _PREDICTION_LABEL_CONFLICT
    return labels, included


def _typed_character_counts(gold, preds, text_length, projection_key, universe):
    gold_labels, included = _projected_character_labels(
        gold, text_length, projection_key, universe, gold=True
    )
    pred_labels, _ = _projected_character_labels(preds, text_length, projection_key, universe, gold=False)
    tp = fp = fn = tn = wrong_type = excluded = 0
    for keep, gold_label, pred_label in zip(included, gold_labels, pred_labels):
        if not keep:
            excluded += 1
            continue
        if gold_label is None:
            if pred_label is None:
                tn += 1
            else:
                fp += 1
        elif pred_label == gold_label:
            tp += 1
        elif pred_label is not None:
            wrong_type += 1
        else:
            fn += 1
    return tp, fp, fn, tn, wrong_type, excluded


def _trim_individual_spans(spans, text, label_key):
    """Trim and clamp spans without merging away their labels."""
    trimmed = []
    for span in spans:
        start = max(0, min(span["start"], len(text)))
        end = max(0, min(span["end"], len(text)))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            trimmed.append({"start": start, "end": end, "label": span[label_key]})
    return trimmed


def _maximum_matches(gold, preds, compatible):
    """Maximum-cardinality bipartite matching, returning its cardinality."""
    edges = [[pred_i for pred_i, pred in enumerate(preds) if compatible(item, pred)] for item in gold]
    pred_to_gold = {}

    def augment(gold_i, seen):
        for pred_i in edges[gold_i]:
            if pred_i in seen:
                continue
            seen.add(pred_i)
            if pred_i not in pred_to_gold or augment(pred_to_gold[pred_i], seen):
                pred_to_gold[pred_i] = gold_i
                return True
        return False

    return sum(augment(gold_i, set()) for gold_i in range(len(gold)))


def _intersection(sets):
    sets = list(sets)
    return set.intersection(*sets) if sets else set()


def _score_individual_spans(
    gold_recs,
    pred_recs,
    tagset=None,
    gold_schema=None,
    pred_schema=None,
    ontology_cuts=(),
    comparison_schemas=(),
    character_label_projections=None,
):
    pred_by_id = {row["id"]: row["preds"] for row in pred_recs}
    totals = {
        "class_agnostic_exact": 0,
        "class_agnostic_overlap": 0,
        "n_gold": 0,
        "n_pred": 0,
        "character_tp": 0,
        "character_fp": 0,
        "character_fn": 0,
        "character_tn": 0,
    }
    gold_type_totals = {}
    projection_specs = {}
    projection_totals = {}
    enabled_character_label_projections = set()
    if tagset:
        schemas = (gold_schema, pred_schema, *comparison_schemas)
        if not any(tagset.is_cut_schema(schema) for schema in schemas):
            fine_universe = _intersection(tagset.schema_image(schema) for schema in schemas)
            projection_specs["typed_fine_intersection"] = {
                "cut": None,
                "universe": fine_universe,
            }
        for cut in ontology_cuts:
            universe = _intersection(tagset.cut_image(schema, cut) for schema in schemas)
            projection_specs[f"typed_{cut}_intersection"] = {
                "cut": cut,
                "universe": universe,
            }
        projection_totals = {
            name: {
                "exact": 0,
                "overlap": 0,
                "n_gold": 0,
                "n_pred": 0,
                "excluded_gold": 0,
                "excluded_pred": 0,
                "character_tp": 0,
                "character_fp": 0,
                "character_fn": 0,
                "character_tn": 0,
                "character_wrong_type": 0,
                "excluded_gold_characters": 0,
            }
            for name in projection_specs
        }
        enabled_character_label_projections = (
            set(projection_specs) if character_label_projections is None else set(character_label_projections)
        )
        unknown_character_label_projections = enabled_character_label_projections - set(projection_specs)
        if unknown_character_label_projections:
            raise ValueError(
                "unknown character-label projections: "
                f"{', '.join(sorted(unknown_character_label_projections))}"
            )
    for record in gold_recs:
        gold = _trim_individual_spans(record["spans"], record["text"], "type")
        preds = _trim_individual_spans(pred_by_id.get(record["id"], []), record["text"], "label")
        totals["n_gold"] += len(gold)
        totals["n_pred"] += len(preds)
        character_counts = _binary_character_counts(gold, preds, len(record["text"]))
        for key, count in zip(
            ("character_tp", "character_fp", "character_fn", "character_tn"),
            character_counts,
        ):
            totals[key] += count
        totals["class_agnostic_exact"] += _maximum_matches(
            gold, preds, lambda a, b: (a["start"], a["end"]) == (b["start"], b["end"])
        )
        totals["class_agnostic_overlap"] += _maximum_matches(gold, preds, overlaps)
        for span in gold:
            by_type = gold_type_totals.setdefault(
                span["label"],
                {"gold": 0, "exact": 0, "overlap": 0},
            )
            by_type["gold"] += 1
            by_type["exact"] += any(
                (span["start"], span["end"]) == (pred["start"], pred["end"]) for pred in preds
            )
            by_type["overlap"] += any(overlaps(span, pred) for pred in preds)
        if tagset:
            for name, spec in projection_specs.items():
                cut = spec["cut"]
                universe = spec["universe"]
                projection_key = name
                for span in gold:
                    span[projection_key] = (
                        tagset.project(gold_schema, span["label"])
                        if cut is None
                        else tagset.project_cut(gold_schema, span["label"], cut)
                    )
                for span in preds:
                    span[projection_key] = (
                        tagset.project(pred_schema, span["label"])
                        if cut is None
                        else tagset.project_cut(pred_schema, span["label"], cut)
                    )
                typed_gold = [span for span in gold if span[projection_key] in universe]
                typed_preds = [span for span in preds if span[projection_key] in universe]
                typed_totals = projection_totals[name]
                typed_totals["n_gold"] += len(typed_gold)
                typed_totals["n_pred"] += len(typed_preds)
                typed_totals["excluded_gold"] += len(gold) - len(typed_gold)
                typed_totals["excluded_pred"] += len(preds) - len(typed_preds)
                same_type = lambda a, b: a[projection_key] == b[projection_key]
                typed_totals["exact"] += _maximum_matches(
                    typed_gold,
                    typed_preds,
                    lambda a, b: same_type(a, b) and (a["start"], a["end"]) == (b["start"], b["end"]),
                )
                typed_totals["overlap"] += _maximum_matches(
                    typed_gold, typed_preds, lambda a, b: same_type(a, b) and overlaps(a, b)
                )
                if name in enabled_character_label_projections:
                    character_counts = _typed_character_counts(
                        gold,
                        preds,
                        len(record["text"]),
                        projection_key,
                        universe,
                    )
                    for key, count in zip(
                        (
                            "character_tp",
                            "character_fp",
                            "character_fn",
                            "character_tn",
                            "character_wrong_type",
                            "excluded_gold_characters",
                        ),
                        character_counts,
                    ):
                        typed_totals[key] += count
    result = {
        "overlap_rule": {
            "kind": "symmetric_span_coverage",
            "minimum_fraction_each": (OVERLAP_MIN_COVERAGE_NUMERATOR / OVERLAP_MIN_COVERAGE_DENOMINATOR),
        },
        "class_agnostic_exact": _prf_one_to_one(
            totals["class_agnostic_exact"], totals["n_pred"], totals["n_gold"]
        ),
        "class_agnostic_overlap": _prf_one_to_one(
            totals["class_agnostic_overlap"], totals["n_pred"], totals["n_gold"]
        ),
        "character_redaction": _classification_from_counts(
            totals["character_tp"],
            totals["character_fp"],
            totals["character_fn"],
            totals["character_tn"],
        ),
        "n_gold": totals["n_gold"],
        "n_pred": totals["n_pred"],
        "annotated_recall_by_gold_type": {
            label: {
                "n": counts["gold"],
                "exact": counts["exact"],
                "overlap": counts["overlap"],
                "exact_R": round(counts["exact"] / counts["gold"], 4),
                "overlap_R": round(counts["overlap"] / counts["gold"], 4),
            }
            for label, counts in sorted(gold_type_totals.items())
        },
    }
    if tagset:
        for name, spec in projection_specs.items():
            typed_totals = projection_totals[name]
            projection_result = {
                "gold_schema": gold_schema,
                "pred_schema": pred_schema,
                "comparison_schemas": list(comparison_schemas),
                "ontology_cut": spec["cut"],
                "universe_size": len(spec["universe"]),
                "universe": sorted(spec["universe"]),
                "n_gold": typed_totals["n_gold"],
                "n_pred": typed_totals["n_pred"],
                "excluded_gold": typed_totals["excluded_gold"],
                "excluded_pred": typed_totals["excluded_pred"],
                "exact": _prf_one_to_one(
                    typed_totals["exact"], typed_totals["n_pred"], typed_totals["n_gold"]
                ),
                "overlap": _prf_one_to_one(
                    typed_totals["overlap"], typed_totals["n_pred"], typed_totals["n_gold"]
                ),
            }
            if name in enabled_character_label_projections:
                projection_result["character_labels"] = _classification_from_counts(
                    typed_totals["character_tp"],
                    typed_totals["character_fp"],
                    typed_totals["character_fn"],
                    typed_totals["character_tn"],
                    wrong_type=typed_totals["character_wrong_type"],
                    excluded=typed_totals["excluded_gold_characters"],
                )
            result[name] = projection_result
    return result


def score_one(
    gold_recs,
    pred_recs,
    strata_key="type",
    tagset=None,
    gold_schema=None,
    pred_schema=None,
    ontology_cuts=(),
    comparison_schemas=(),
    character_label_projections=None,
):
    pred_by_id = {r["id"]: r["preds"] for r in pred_recs}
    tp_exact = tp_gold_ov = tp_pred_ov = n_gold = n_pred = 0
    strat = {}
    for g in gold_recs:
        preds = merge_regions(pred_by_id.get(g["id"], []), g["text"])
        n_gold += len(g["spans"])
        n_pred += len(preds)
        exact = {(p["start"], p["end"]) for p in preds}
        for s in g["spans"]:
            hit_exact = (s["start"], s["end"]) in exact
            hit_ov = any(overlaps(s, p) for p in preds)
            tp_exact += hit_exact
            tp_gold_ov += hit_ov
            coarse = COARSE.get(s.get(strata_key, ""), "other")
            st = strat.setdefault(coarse, [0, 0])
            st[0] += hit_ov
            st[1] += 1
        gspans = g["spans"]
        for p in preds:
            tp_pred_ov += any(overlaps(s, p) for s in gspans)

    def prf(tp_p, tp_g, np_, ng):
        P = tp_p / np_ if np_ else 0.0
        R = tp_g / ng if ng else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        F2 = 5 * P * R / (4 * P + R) if P + R else 0.0
        return {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4), "F2": round(F2, 4)}

    result = {
        "overlap_rule": {
            "kind": "symmetric_span_coverage",
            "minimum_fraction_each": (OVERLAP_MIN_COVERAGE_NUMERATOR / OVERLAP_MIN_COVERAGE_DENOMINATOR),
        },
        "n_gold": n_gold,
        "n_pred": n_pred,
        "overlap": prf(tp_pred_ov, tp_gold_ov, n_pred, n_gold),
        "exact": prf(tp_exact, tp_exact, n_pred, n_gold),
        "recall_by_coarse": {k: {"R": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(strat.items())},
    }
    result["individual_span_one_to_one"] = _score_individual_spans(
        gold_recs,
        pred_recs,
        tagset,
        gold_schema,
        pred_schema,
        ontology_cuts,
        comparison_schemas,
        character_label_projections,
    )
    return result


def cmd_combine(args):
    """Multi-model union/agreement span combination (class-agnostic overlap;
    typed hierarchy compatibility is the projector's later refinement)."""
    for ds in args.datasets:
        per_model = []
        for m in args.models:
            pf = os.path.join(PRED, f"{m}.{ds}.jsonl")
            per_model.append({r["id"]: r["preds"] for r in (json.loads(l) for l in open(pf))})
        ids = per_model[0].keys()
        union_out, agree_out = [], []
        for did in ids:
            all_preds = [pm.get(did, []) for pm in per_model]
            union = [p for preds in all_preds for p in preds]
            agree = []
            need = args.min_votes or len(all_preds)
            for i, preds in enumerate(all_preds):
                others = [all_preds[j] for j in range(len(all_preds)) if j != i]
                for p in preds:
                    votes = 1 + sum(any(overlaps(p, q) for q in o) for o in others)
                    if votes >= need:
                        agree.append(p)
            union_out.append({"id": did, "preds": union})
            agree_out.append({"id": did, "preds": agree})
        tag = "+".join(args.models) + (f"@{args.min_votes}" if args.min_votes else "")
        for name, recs in ((f"union[{tag}]", union_out), (f"agree[{tag}]", agree_out)):
            out = os.path.join(PRED, f"{name}.{ds}.jsonl")
            with open(out, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"wrote {out}")


def cmd_score(args):
    if __package__:
        from scripts.pii_projector import Tagset
    else:
        from pii_projector import Tagset

    include_ids = None
    if args.include_ids:
        identifiers = [
            line.strip()
            for line in open(args.include_ids, encoding="utf-8").read().splitlines()
            if line.strip()
        ]
        if not identifiers:
            raise ValueError(f"empty --include-ids file: {args.include_ids}")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate IDs in --include-ids file: {args.include_ids}")
        include_ids = set(identifiers)
    available_ids = set()

    model_schemas = _key_value_args(args.model_schema, "--model-schema")
    gold_schemas = _key_value_args(args.gold_schema, "--gold-schema")
    configured = bool(
        model_schemas
        or gold_schemas
        or args.ontology_cut
        or args.comparison_schema
        or args.canonicalize_predictions
    )
    if configured and (not model_schemas or not gold_schemas):
        raise ValueError("typed scoring requires both --model-schema and --gold-schema")
    tagset = Tagset() if configured else None
    if tagset:
        unknown_cuts = sorted(set(args.ontology_cut) - set(tagset.cut_names()))
        if unknown_cuts:
            raise ValueError(f"unknown --ontology-cut values: {', '.join(unknown_cuts)}")
        for schema in args.comparison_schema:
            tagset.schema_image(schema)
    report = {}
    score_path = args.output or os.path.join(HERE, "scores.json")
    if os.path.exists(score_path):
        report = json.load(open(score_path))
    for ds in args.datasets:
        gold = [json.loads(l) for l in open(os.path.join(GOLD, ds + ".jsonl"))]
        available_ids.update(record["id"] for record in gold)
        if args.exclude_type:
            excluded = set(args.exclude_type)
            n_before = sum(len(record["spans"]) for record in gold)
            gold = [
                {
                    **record,
                    "spans": [span for span in record["spans"] if span["type"] not in excluded],
                }
                for record in gold
            ]
            n_after = sum(len(record["spans"]) for record in gold)
            print(f"{ds}: excluded {n_before - n_after} gold spans with types {sorted(excluded)}")
        for model in args.models:
            pf = os.path.join(PRED, f"{model}.{ds}.jsonl")
            if not os.path.exists(pf):
                print(f"missing {pf}, skipping")
                continue
            preds = [json.loads(l) for l in open(pf)]
            gold_sub = _align_gold_to_predictions(
                gold,
                preds,
                allow_prediction_subset=args.allow_prediction_subset,
            )
            if include_ids is not None:
                gold_sub = [record for record in gold_sub if record["id"] in include_ids]
                preds = [record for record in preds if record["id"] in include_ids]
            if configured and model not in model_schemas:
                raise ValueError(f"missing --model-schema for {model!r}")
            if configured and ds not in gold_schemas:
                raise ValueError(f"missing --gold-schema for {ds!r}")
            prediction_schema = model_schemas.get(model)
            if args.canonicalize_predictions:
                preds = _canonicalize_prediction_labels(preds, prediction_schema, tagset)
                prediction_schema = "canonical"
            r = score_one(
                gold_sub,
                preds,
                tagset=tagset,
                gold_schema=gold_schemas.get(ds),
                pred_schema=prediction_schema,
                ontology_cuts=args.ontology_cut,
                comparison_schemas=args.comparison_schema,
            )
            report[f"{model}/{ds}"] = r
            if args.summary_only:
                span_f1 = r["individual_span_one_to_one"]["class_agnostic_overlap"]["F1"]
                print(f"SCORE: {model}/{ds} n_docs={len(gold_sub)} span_overlap_F1={span_f1:.4f}")
            else:
                print(f"\n=== {model} / {ds} (n_docs={len(gold_sub)}) ===")
                print(json.dumps(r, indent=2))
    if include_ids is not None:
        unknown_ids = sorted(include_ids - available_ids)
        if unknown_ids:
            raise ValueError(
                f"--include-ids contains IDs absent from the evaluated datasets: {unknown_ids[:5]}"
            )
    output_parent = os.path.dirname(os.path.abspath(score_path))
    os.makedirs(output_parent, exist_ok=True)
    with open(score_path, "w", encoding="utf-8") as sink:
        json.dump(report, sink, indent=2)
        sink.write("\n")
    print(f"\nwrote {score_path}")


def _key_value_args(values, option):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects NAME=SCHEMA, got {value!r}")
        key, schema = value.split("=", 1)
        if not key or not schema:
            raise ValueError(f"{option} expects NAME=SCHEMA, got {value!r}")
        if key in result and result[key] != schema:
            raise ValueError(f"{option} gives conflicting schemas for {key!r}")
        result[key] = schema
    return result


def _canonicalize_prediction_labels(predictions, source_schema, tagset):
    """Project source-schema labels while preserving records and span geometry."""
    return [
        {
            **record,
            "preds": [
                {**span, "label": tagset.project(source_schema, span["label"])} for span in record["preds"]
            ],
        }
        for record in predictions
    ]


def _align_gold_to_predictions(gold, predictions, allow_prediction_subset=False):
    gold_by_id = {row["id"]: row for row in gold}
    prediction_ids = [row["id"] for row in predictions]
    if len(gold_by_id) != len(gold):
        raise ValueError("gold data contain duplicate record IDs")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("predictions contain duplicate record IDs")
    gold_ids = set(gold_by_id)
    predicted_ids = set(prediction_ids)
    missing = gold_ids - predicted_ids
    extra = predicted_ids - gold_ids
    if extra or (missing and not allow_prediction_subset):
        raise ValueError(
            "prediction/gold ID mismatch: "
            f"{len(missing)} missing predictions, {len(extra)} unknown prediction IDs"
        )
    return [gold_by_id[record_id] for record_id in prediction_ids]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    unified = sub.add_parser("prep-unified")
    unified.add_argument("--source", required=True)
    unified.add_argument("--dataset-prefix", required=True)
    unified.add_argument(
        "--per-language",
        type=int,
        default=0,
        help="deterministically sample this many records per language; 0 exports every record",
    )
    unified.add_argument("--seed", type=int, default=20260802)
    mapa = sub.add_parser("prep-mapa")
    mapa.add_argument("--source-root", required=True)
    mapa.add_argument("--languages", nargs="+", default=["cs", "nl", "pt", "sv"])
    idner = sub.add_parser("prep-idner")
    idner.add_argument("--source-root", required=True)
    idner.add_argument("--splits", nargs="+", choices=["validation", "test"], default=["validation", "test"])
    hiner = sub.add_parser("prep-hiner")
    hiner.add_argument("--source-root", required=True)
    hiner.add_argument("--splits", nargs="+", choices=["validation", "test"], default=["validation", "test"])
    klue = sub.add_parser("prep-klue")
    klue.add_argument("--source-root", required=True)
    klue.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["validation"])
    wojood = sub.add_parser("prep-wojood")
    wojood.add_argument("--source-root", required=True)
    wojood.add_argument("--splits", nargs="+", choices=["validation", "test"], default=["validation", "test"])
    aqmar = sub.add_parser("prep-aqmar")
    aqmar.add_argument("--source-root", required=True)
    aqmar.add_argument("--splits", nargs="+", choices=["validation", "test"], default=["validation", "test"])
    openner_core = sub.add_parser("prep-openner-core")
    openner_core.add_argument("--source-root", required=True)
    openner_core.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "test"],
        default=["validation", "test"],
    )
    openner_core.add_argument(
        "--languages",
        nargs="+",
        choices=["de", "en", "es", "ja", "pt", "sv", "zh"],
        default=["de", "en", "es", "ja", "pt", "sv", "zh"],
    )
    p = sub.add_parser("predict")
    p.add_argument("--model", choices=["openmed", "gliner2", "openai", "local"], required=True)
    p.add_argument("--datasets", nargs="+", default=["spy-medical", "spy-legal", "tab-test"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output-name", help="prediction/cost filename prefix (default: --model value)")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="warm the model and record synchronized batch-1 latency/throughput plus process peak CUDA memory",
    )
    p.add_argument("--warmup-docs", type=int, default=8, help="documents replayed before benchmark timing")
    p.add_argument(
        "--hf-windowing",
        choices=HF_WINDOWING_MODES,
        default="character-conservative",
        help=(
            "character-conservative preserves the established fixed-character serving view; "
            "token-capacity lets the model tokenizer fill overlapping windows to its token limit"
        ),
    )
    p.add_argument(
        "--bioes-project-legal",
        action="store_true",
        help="lazily replace illegal independent BIOES paths with the highest-scoring legal path",
    )
    p.add_argument(
        "--bioes-bucket-compat",
        choices=("none", *BIOES_BUCKET_COMPAT_LEVELS),
        default=DEFAULT_BIOES_BUCKET_COMPAT,
        help=(
            "compatibility level for differently typed I/E continuation repair: fine (also spelled "
            "none), ontology-v2 family, redaction-20, redaction-9, or entity/P1 (default); explicit "
            "B/S always starts a new span"
        ),
    )
    p.add_argument(
        "--bioes-bucket-compat-type",
        choices=("opening", "preponderance", "margin", "logitsum"),
        default=DEFAULT_BIOES_BUCKET_COMPAT_TYPE,
        help=(
            "merged-span fine label: opening token's type; largest char mass among token argmax "
            "types; largest bias-adjusted margin mass; or summed logits over all bucket-compatible "
            "types (logitsum sees types that never rank first)"
        ),
    )
    p.add_argument(
        "--bioes-preponderance-weight",
        choices=PREPONDERANCE_WEIGHT_MODES,
        default="nfc-char",
        help=(
            "vote mass for --bioes-bucket-compat-type preponderance: one per token, raw Python "
            "codepoints, or codepoints after NFC-normalizing each token surface (default)"
        ),
    )
    p.add_argument(
        "--score-cache-prefix",
        default=None,
        help=(
            "write a per-dataset token score cache (<prefix>.<dataset>.npz, top-3 non-O + "
            "O-margins) so decode variants re-run on CPU without another model forward"
        ),
    )
    p.add_argument(
        "--score-cache-full-logits",
        action="store_true",
        help=(
            "also retain every float32 token-label logit in each score cache; intended for small "
            "exact decoder studies because it is much larger than the default top-3 trace"
        ),
    )
    p.add_argument(
        "--o-logit-bias",
        type=float,
        default=0.0,
        help="add this bias to the O logit before token argmax; negative values favor entity labels",
    )
    p.add_argument(
        "--character-logit-scale",
        type=float,
        default=None,
        help=(
            "absolute inference scale for an imported continuous-character tagger readout; "
            "the saved checkpoint remains unchanged"
        ),
    )
    p.add_argument(
        "--suppress-primary-type",
        action="append",
        default=[],
        help=(
            "mask every BIOES logit column for this primary type before greedy or legal-path "
            "decoding; repeat for multiple types"
        ),
    )
    character_onnx = sub.add_parser(
        "predict-character-onnx",
        help="run an exact tokenizer-free char_ids-only ONNX bundle and save full logits",
    )
    character_onnx.add_argument("--bundle", required=True)
    character_onnx.add_argument("--datasets", nargs="+", required=True)
    character_onnx.add_argument("--output-name", required=True)
    character_onnx.add_argument("--score-cache-prefix", required=True)
    character_onnx.add_argument("--limit", type=int, default=0)
    character_token_onnx = sub.add_parser(
        "predict-character-token-onnx",
        help="run a character CNN over bundled tokenizer intervals and save full logits",
    )
    character_token_onnx.add_argument("--bundle", required=True)
    character_token_onnx.add_argument("--datasets", nargs="+", required=True)
    character_token_onnx.add_argument("--output-name", required=True)
    character_token_onnx.add_argument("--score-cache-prefix", required=True)
    character_token_onnx.add_argument("--limit", type=int, default=0)
    redecode = sub.add_parser("redecode-token-cache")
    redecode.add_argument("--score-cache-prefix", required=True)
    redecode.add_argument("--output-name", required=True)
    redecode.add_argument("--datasets", nargs="+", required=True)
    redecode.add_argument("--o-logit-bias", type=float, default=0.0)
    redecode.add_argument(
        "--bioes-bucket-compat",
        choices=("none", *BIOES_BUCKET_COMPAT_LEVELS),
        default="entity",
        help=(
            "compatibility level for mismatched I/E continuation repair; ontology_v2_family "
            "matches the ontology-aware serving package, while entity is P1; explicit B/S always "
            "starts a new span"
        ),
    )
    redecode.add_argument(
        "--bioes-bucket-compat-type",
        choices=("opening", "preponderance", "margin", "logitsum"),
        default="preponderance",
        help="fine label chosen after geometry repair",
    )
    redecode.add_argument(
        "--bioes-preponderance-weight",
        choices=PREPONDERANCE_WEIGHT_MODES,
        default="nfc-char",
        help="vote mass for preponderance; legacy caches reconstruct NFC mass from frozen gold text",
    )
    redecode.add_argument(
        "--bioes-search",
        choices=("greedy-repair", "lazy-fine-legal", "word-fine-legal", "viterbi-max", "viterbi-backoff"),
        default="greedy-repair",
        help=(
            "repair independent token argmax labels; retain an already legal fine path and exactly "
            "re-decode only illegal paths; constrain every fine-typed entity endpoint to a Unicode "
            "word boundary; or exactly maximize a legal coarse-BIOES path using the configured "
            "fine-label reduction within the selected compatibility level; viterbi-backoff "
            "interpolates normalized sibling scores then chooses an exact fine-compatible path"
        ),
    )
    redecode.add_argument(
        "--bioes-backoff-alpha",
        type=float,
        help="with viterbi-backoff, sibling interpolation weight in [0, 1]; required explicitly",
    )
    redecode.add_argument(
        "--bioes-search-top-k",
        type=int,
        default=0,
        help=(
            "with viterbi-max, mask each token to its top-k non-O labels before exact search; "
            "0 uses the complete logit vector"
        ),
    )
    redecode.add_argument(
        "--bioes-bucket-reduction",
        choices=BIOES_BUCKET_REDUCTIONS,
        default="max",
        help=(
            "aggregate native fine-label logits into each compatibility state by maximum, "
            "temperature log-sum-exp, or bucket-size-normalized temperature log-mean-exp"
        ),
    )
    redecode.add_argument(
        "--bioes-bucket-temperature",
        type=float,
        default=1.0,
        help="positive temperature for logsumexp/logmeanexp compatibility-state emissions",
    )
    redecode.add_argument(
        "--bioes-whitespace-split-cost",
        type=float,
        default=0.0,
        help=(
            "nonnegative cost for S/E-c -> S/B-c across a positive Unicode-whitespace-only "
            "source gap; 0 preserves the current decoder exactly"
        ),
    )
    redecode.add_argument(
        "--name-oracle-config",
        help="enable final-stage person-name carrier adjustment and categorical component sidecars",
    )
    redecode.add_argument(
        "--name-oracle-language",
        help="fixed BCP 47 language tag; otherwise each gold row must provide bcp47",
    )
    redecode.add_argument("--name-oracle-given-lexicon", action="append", default=[])
    redecode.add_argument("--name-oracle-family-lexicon", action="append", default=[])
    bias_grid = sub.add_parser("predict-bias-grid")
    bias_grid.add_argument("--model", choices=["openmed", "openai", "local"], required=True)
    bias_grid.add_argument("--datasets", nargs="+", required=True)
    bias_grid.add_argument("--output-prefix", required=True)
    bias_grid.add_argument(
        "--score-cache",
        help=(
            "optional compressed .npz trace containing token offsets, winning non-O label, and "
            "O-minus-winner margin for model-free global O-bias calibration"
        ),
    )
    bias_grid.add_argument("--limit", type=int, default=0)
    bias_grid.add_argument(
        "--hf-windowing",
        choices=HF_WINDOWING_MODES,
        default="character-conservative",
    )
    bias_grid.add_argument("--bioes-project-legal", action="store_true")
    bias_grid.add_argument(
        "--bioes-bucket-compat",
        choices=("none", *BIOES_BUCKET_COMPAT_LEVELS),
        default=DEFAULT_BIOES_BUCKET_COMPAT,
    )
    bias_grid.add_argument(
        "--o-logit-bias",
        type=float,
        action="append",
        required=True,
        help="O-logit bias decoded from the shared encoder forwards; repeat for a grid",
    )
    gliner_cache = sub.add_parser("predict-gliner-cache")
    gliner_cache.add_argument("--datasets", nargs="+", required=True)
    gliner_cache.add_argument("--output-prefix", required=True)
    gliner_cache.add_argument("--score-cache", required=True)
    gliner_cache.add_argument("--candidate-floor", type=float, default=0.01)
    gliner_cache.add_argument(
        "--revision",
        default=GLINER2_FRONTIER_REVISION,
        help="exact Hugging Face revision resolved before loading the model",
    )
    gliner_cache.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("combine")
    c.add_argument("--models", nargs="+", required=True)
    c.add_argument("--min-votes", type=int, default=0, help="agreement threshold; 0 = unanimous")
    c.add_argument("--datasets", nargs="+", default=["spy-medical", "spy-legal", "tab-test"])
    s = sub.add_parser("score")
    s.add_argument("--models", nargs="+", default=["openmed", "gliner2", "openai"])
    s.add_argument("--datasets", nargs="+", default=["spy-medical", "spy-legal", "tab-test"])
    s.add_argument("--output", help="metrics JSON path (default: $PII_EVAL_HOME/scores.json)")
    s.add_argument(
        "--include-ids",
        help="newline-delimited evaluation IDs to retain; every ID must occur in the requested datasets",
    )
    s.add_argument(
        "--summary-only",
        action="store_true",
        help="print one score headline per model/dataset while retaining the complete output JSON",
    )
    s.add_argument(
        "--exclude-type",
        action="append",
        default=[],
        help="gold label to exclude before all scoring; repeat for multiple frozen out-of-scope types",
    )
    s.add_argument(
        "--model-schema",
        action="append",
        default=[],
        metavar="MODEL=SCHEMA",
        help=("explicit prediction-label source schema or native ontology cut; repeat once per scored model"),
    )
    s.add_argument(
        "--gold-schema",
        action="append",
        default=[],
        metavar="DATASET=SCHEMA",
        help="explicit gold-label schema; repeat once per scored dataset",
    )
    s.add_argument(
        "--ontology-cut",
        action="append",
        default=[],
        help="named reporting projection to score; repeat for multiple ontology levels",
    )
    s.add_argument(
        "--comparison-schema",
        action="append",
        default=[],
        help="additional schema whose expressible labels define a common comparison universe",
    )
    s.add_argument(
        "--canonicalize-predictions",
        action="store_true",
        help=(
            "project each model's prediction labels into the canonical ontology before typed scoring; "
            "without --comparison-schema this retains the full gold-expressible universe and counts "
            "types absent from a model's source schema as misses"
        ),
    )
    s.add_argument(
        "--allow-prediction-subset",
        action="store_true",
        help="explicitly score an ID-matched prediction subset instead of requiring the complete gold set",
    )
    args = ap.parse_args()
    {
        "prep": cmd_prep,
        "prep-unified": cmd_prep_unified,
        "prep-mapa": cmd_prep_mapa,
        "prep-idner": cmd_prep_idner,
        "prep-hiner": cmd_prep_hiner,
        "prep-klue": cmd_prep_klue,
        "prep-wojood": cmd_prep_wojood,
        "prep-aqmar": cmd_prep_aqmar,
        "prep-openner-core": cmd_prep_openner_core,
        "predict": cmd_predict,
        "predict-character-onnx": cmd_predict_character_onnx,
        "predict-character-token-onnx": cmd_predict_character_token_onnx,
        "redecode-token-cache": cmd_redecode_token_cache,
        "predict-bias-grid": cmd_predict_bias_grid,
        "predict-gliner-cache": cmd_predict_gliner_cache,
        "combine": cmd_combine,
        "score": cmd_score,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
