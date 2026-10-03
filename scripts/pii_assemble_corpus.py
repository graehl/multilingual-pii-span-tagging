#!/usr/bin/env python
"""Assemble the unified-tagset diverse PII training corpus
(topics/pii-robust-buildout.md, "Data assembly").

Every source is projected into the single target tagset via
scripts/pii_tagset.yaml (pii_projector.Tagset), selected by an explicit
released-source policy, and emitted as unified records:

  {"id", "text", "spans": [[start, end, node]], "src", "lang"}

Diversity is the point (buildout finding 2): multiple independent pipelines
(ai4privacy, Nemotron, real clinical/legal gold, SPY, transport-instantiated
languages), multiple scripts, and multiple domains. The reproducible legacy
policy retains the capped v1/v2 inputs. The new onboarded-full policy consumes
every committed Nemotron-PII and OpenPII 1M row; it keeps only the
OpenPII-1.5M language tail absent from OpenPII 1M so overlapping release
lineages are not counted twice merely because both were available.

Held out entirely (eval only, never emitted): mgs-* (10-lang clinical),
gold/*-fresh.jsonl (frontier-authored selection eval), meddocan-test,
tab-test. SPY goes to training per the buildout plan, which retires
gold/spy-*.jsonl as honest eval for models trained here.

Rare unified nodes are collapsed to their nearest ancestor whose corpus
count meets --min-node-count, so the trainer's label space stays dense.
Unmapped source labels are a hard error (fix the map, not the data).

Usage:
  pii_assemble_corpus.py build --out untracked/pii-eval/ft/unified-v4-full-sources \
      --released-source-set onboarded-full \
      [--cap-ai4p-lang 6000] [--cap-nemotron 12000] [--seed 0] \
      [--min-node-count 300] [--val-frac 0.02] \
      [--source meddocan --source tab]
"""

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from pii_projector import Tagset  # noqa: E402
from pii_rule_completion import (  # noqa: E402
    DEFAULT_COVERAGE_THRESHOLD,
    RULESET_VERSION,
    complete_records,
)

from log_format import headline  # noqa: E402

EVAL_HOME = os.environ.get("PII_EVAL_HOME") or os.path.join(REPO, "untracked/pii-eval")
DATA = os.path.join(EVAL_HOME, "data")
FT = os.path.join(EVAL_HOME, "ft")
ONBOARDED = os.environ.get("PII_ONBOARDED_HOME") or os.path.join(REPO, "data/pii-onboarded")
PII_ANNOTATIONS = os.path.join(REPO, "data/pii-annotations")
FINAL20_TRANSPORT = os.path.join(PII_ANNOTATIONS, "transport")
FINAL20_LANGUAGES = (
    "ar",
    "cs",
    "de",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "sv",
    "tr",
    "uk",
    "vi",
    "zh",
)
FINAL20_BROAD_LANGUAGES = ("ar", "cs", "hi", "id", "nl", "pt", "sv", "vi")
MAPA_NATURAL_SOURCE = "mapa-natural"
IDNER_NATURAL_SOURCE = "idner-news-natural"
HINER_NATURAL_SOURCE = "hiner-natural"
WOJOOD_NATURAL_SOURCE = "wojood-sample-natural"
AQMAR_NATURAL_SOURCE = "aqmar-natural"
OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE = "openner-commercial-core-natural"
ANNOTATED_SPANS_ONLY_SOURCES = frozenset(
    {
        MAPA_NATURAL_SOURCE,
        IDNER_NATURAL_SOURCE,
        HINER_NATURAL_SOURCE,
        WOJOOD_NATURAL_SOURCE,
        AQMAR_NATURAL_SOURCE,
        OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE,
    }
)


# ------------------------------------------------------------------ loaders
# Each yields (rec_id, text, [[start, end, source_label]], lang, locator).


def load_ai4p_1p5m(cap_per_lang, rng, exclude_langs=()):
    path = os.path.join(DATA, "ai4p-openpii-1.5m/data/train.jsonl")
    exclude_langs = set(exclude_langs)
    by_lang = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r["language"] in exclude_langs:
                continue
            by_lang[r["language"]].append(i)
    keep = set()
    for idxs in by_lang.values():
        if len(idxs) > cap_per_lang:
            idxs = rng.sample(idxs, cap_per_lang)
        keep.update(idxs)
    with open(path) as f:
        for i, line in enumerate(f):
            if i not in keep:
                continue
            r = json.loads(line)
            spans = [[s["start"], s["end"], s["label"]] for s in r["privacy_mask"]]
            yield (
                f"a4p15m-{r['uid']}",
                r["source_text"],
                spans,
                r["language"],
                {"path": path, "line_1based": i + 1},
            )


def onboarded_manifest(slug, root=ONBOARDED):
    path = Path(root) / slug / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_onboarded(slug, root=ONBOARDED, verify=True, splits=None):
    """Load every canonical shard while retaining the upstream label vocabulary."""
    root = Path(root)
    if verify:
        from pii_onboard_sources import SOURCES, check

        check(SOURCES[slug], root)
    manifest = onboarded_manifest(slug, root)
    splits = set(splits) if splits is not None else None
    for shard in sorted(manifest["shards"], key=lambda item: item["path"]):
        if splits is not None and Path(shard["path"]).parts[0] not in splits:
            continue
        path = root / slug / shard["path"]
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                spans = [[span["start"], span["end"], span["source_label"]] for span in row["spans"]]
                yield (
                    f"{slug}-{row['id']}",
                    row["text"],
                    spans,
                    row["lang"],
                    {"path": str(path), "line_1based": line_number},
                )


def load_ai4p_200k(cap_per_lang, rng):
    for path in sorted(glob.glob(os.path.join(DATA, "ai4p-200k/*_pii_*.jsonl"))):
        rows = list(enumerate(open(path), 1))
        if len(rows) > cap_per_lang:
            rows = rng.sample(rows, cap_per_lang)
        for line_number, line in rows:
            r = json.loads(line)
            spans = [[s["start"], s["end"], s["label"]] for s in r["privacy_mask"]]
            yield (
                f"a4p200k-{r['language']}-{r['id']}",
                r["source_text"],
                spans,
                r["language"],
                {"path": path, "line_1based": line_number},
            )


def load_nemotron(cap, rng):
    import ast

    import pyarrow.parquet as pq

    path = os.path.join(DATA, "nemotron-pii/data/train-00000-of-00001.parquet")
    t = pq.read_table(path)
    idxs = range(t.num_rows)
    if t.num_rows > cap:
        idxs = sorted(rng.sample(idxs, cap))
    texts, spans_col, uids = t.column("text"), t.column("spans"), t.column("uid")
    for i in idxs:
        spans = spans_col[i].as_py()
        if isinstance(spans, str):
            # The released parquet stores spans as a Python-repr string.
            spans = ast.literal_eval(spans)
        yield (
            f"nemotron-{uids[i].as_py()}",
            texts[i].as_py(),
            [[s["start"], s["end"], s["label"]] for s in spans],
            "en",
            {"path": path, "row_0based": i},
        )


def load_prepped(recs, prefix, lang):
    """Adapt pii_eval prep_* output ({id,text,spans:[{start,end,type}]})."""
    for r in recs:
        yield (
            f"{prefix}-{r['id']}",
            r["text"],
            [[s["start"], s["end"], s["type"]] for s in r["spans"]],
            lang,
            {"prepared_source": prefix, "id": r["id"]},
        )


def load_prepped_source(loader, loader_args, prefix, lang):
    """Defer source acquisition until its corpus iterator is consumed."""
    yield from load_prepped(loader(*loader_args), prefix, lang)


def load_ft_jsonl(name, lang):
    path = os.path.join(FT, name + ".jsonl")
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            yield f"{name}-{i}", r["text"], r["spans"], lang, {"path": path, "line_1based": i + 1}


def load_prod_jsonl(path, lang):
    for line_number, line in enumerate(open(path), 1):
        r = json.loads(line)
        yield (
            f"{lang}-tp-{r['id']}",
            r["text"],
            r["spans"],
            lang,
            {"path": path, "line_1based": line_number},
        )


def load_canonical_jsonl(path, source_name, default_lang=None):
    """Load committed rows whose spans already use unified target nodes."""
    with open(path) as source:
        for i, line in enumerate(source):
            row = json.loads(line)
            lang = row.get("lang", default_lang)
            if lang is None:
                raise ValueError(f"{path}:{i + 1}: canonical row has no language")
            yield (
                f"{source_name}-{row.get('id', i)}",
                row["text"],
                row["spans"],
                lang,
                {"path": str(path), "line_1based": i + 1},
            )


def final20_transport_sources():
    """Opt-in Final20 translation-stage increments, excluding raw carrier waves."""
    for code in FINAL20_BROAD_LANGUAGES:
        name = f"{code}-transport-final20"
        path = os.path.join(FINAL20_TRANSPORT, f"{code}-final20-v1.jsonl")
        yield name, "_identity", load_canonical_jsonl(path, name, code)

    name = "en-targeted-final20"
    path = os.path.join(PII_ANNOTATIONS, "targeted-coverage-v1.jsonl")
    yield name, "_identity", load_canonical_jsonl(path, name, "en")

    for code in FINAL20_LANGUAGES:
        name = f"{code}-targeted-final20"
        path = os.path.join(FINAL20_TRANSPORT, f"targeted-{code}-final20-v1.jsonl")
        yield name, "_identity", load_canonical_jsonl(path, name, code)


def released_sources(args, rng):
    """Released synthetic inputs selected by a reproducible source policy."""
    if args.released_source_set == "legacy-sampled":
        yield "ai4p-1.5m", "ai4privacy_new", load_ai4p_1p5m(args.cap_ai4p_lang, rng)
        yield "ai4p-200k", "ai4privacy_200k", load_ai4p_200k(args.cap_ai4p_lang, rng)
        yield "nemotron", "nemotron_pii", load_nemotron(args.cap_nemotron, rng)
        return

    openpii_manifest = onboarded_manifest("openpii-1m")
    openpii_languages = set(openpii_manifest["counts"]["languages"])
    yield "openpii-1m-full", "ai4privacy_new", load_onboarded("openpii-1m")
    yield "nemotron-full", "nemotron_pii", load_onboarded("nemotron-pii")
    yield (
        "ai4p-1.5m-extra-langs",
        "ai4privacy_new",
        load_ai4p_1p5m(args.cap_ai4p_lang, rng, exclude_langs=openpii_languages),
    )
    # The 200k line contributes the wider 52-type privacy vocabulary.
    yield "ai4p-200k", "ai4privacy_200k", load_ai4p_200k(args.cap_ai4p_lang, rng)


def expected_full_source_counts():
    return {
        "openpii-1m-full": onboarded_manifest("openpii-1m")["counts"]["records"],
        "nemotron-full": onboarded_manifest("nemotron-pii")["counts"]["records"],
    }


def sources(args, rng):
    """(source_name, schema, record iterator) triples for the selected mix."""
    import pii_eval

    yield from released_sources(args, rng)
    if args.source and MAPA_NATURAL_SOURCE in args.source:
        yield (
            MAPA_NATURAL_SOURCE,
            "mapa_coarse",
            load_onboarded("mapa", splits=("train", "validation")),
        )
    if args.source and IDNER_NATURAL_SOURCE in args.source:
        yield (
            IDNER_NATURAL_SOURCE,
            "idner_news_2k",
            load_onboarded("idner-news-2k", splits=("train",)),
        )
    if args.source and HINER_NATURAL_SOURCE in args.source:
        yield (
            HINER_NATURAL_SOURCE,
            "hiner_original",
            load_onboarded("hiner", splits=("train",)),
        )
    if args.source and WOJOOD_NATURAL_SOURCE in args.source:
        yield (
            WOJOOD_NATURAL_SOURCE,
            "wojood_nested",
            load_onboarded("wojood-sample", splits=("train",)),
        )
    if args.source and AQMAR_NATURAL_SOURCE in args.source:
        yield (
            AQMAR_NATURAL_SOURCE,
            "aqmar_core",
            load_onboarded("aqmar-openner", splits=("train",)),
        )
    if args.source and OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE in args.source:
        yield (
            OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE,
            "openner_core",
            load_onboarded("openner-commercial-core", splits=("train",)),
        )
    yield (
        "meddocan",
        "meddocan_phi",
        itertools.chain(
            load_prepped_source(
                pii_eval.prep_brat,
                (os.path.join(DATA, "meddocan-corpus/corpus/train/brat"), "es"),
                "meddocan-train",
                "es",
            ),
            load_prepped_source(
                pii_eval.prep_brat,
                (os.path.join(DATA, "meddocan-corpus/corpus/dev/brat"), "es"),
                "meddocan-dev",
                "es",
            ),
        ),
    )
    yield (
        "tab",
        "tab_8",
        itertools.chain(
            load_prepped_source(
                pii_eval.prep_tab, (os.path.join(DATA, "tab/echr_train.json"),), "tab-train", "en"
            ),
            load_prepped_source(
                pii_eval.prep_tab, (os.path.join(DATA, "tab/echr_dev.json"),), "tab-dev", "en"
            ),
        ),
    )
    yield (
        "spy",
        "spy_7",
        itertools.chain(
            load_prepped_source(
                pii_eval.populate_spy,
                (os.path.join(DATA, "spy/data/medical_consultations_placeholders.jsonl"), "med"),
                "spy-med",
                "en",
            ),
            load_prepped_source(
                pii_eval.populate_spy,
                (os.path.join(DATA, "spy/data/legal_questions_placeholders.jsonl"), "leg"),
                "spy-leg",
                "en",
            ),
        ),
    )
    yield "zh-transport", "openmed_nemotron_55", load_ft_jsonl("v4b-train-zh", "zh")
    # v2: instantiated TG-12b transport waves (pii_instantiate_transport.py);
    # spans carry nemotron src_labels, so the nemotron_pii map applies.
    for code in ("ja", "ko", "fr", "es", "de", "ru", "pl", "it", "tr", "uk", "fa"):
        path = os.path.join(DATA, f"../prod/{code}-instantiated-v1.jsonl")
        if os.path.exists(path):
            yield f"{code}-transport", "nemotron_pii", load_prod_jsonl(path, code)
    # v3: authored clinical training docs (build_clinical.py). Spans already
    # carry TARGET nodes (authored_v1 schema applied at merge time), so they
    # ride the identity schema — every node maps to itself.
    clinical = os.path.join(FT, "clinical-train-v1.jsonl")
    if os.path.exists(clinical):
        yield "clinical-authored", "_identity", load_ft_jsonl("clinical-train-v1", None)
    if args.final20_transport:
        yield from final20_transport_sources()


# ------------------------------------------------------------------ build


def resolve_overlaps(spans):
    """Greedy non-overlapping selection, longer spans first; BIO tagging
    cannot express overlaps. Returns (kept, n_dropped)."""
    kept = []
    for s in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(s[1] <= k[0] or s[0] >= k[1] for k in kept):
            kept.append(s)
    kept.sort(key=lambda s: s[0])
    return kept, len(spans) - len(kept)


def requires_upstream_rule_completion(source: str) -> bool:
    """Return whether a translated derivative must be repaired at its source.

    Applying regexes directly to translated text would preserve the old
    unprotected source carrier and teach contradictory outside labels. These
    sources are audited here but changed only by regenerating the affected
    source placeholders and translation rows.
    """
    return "transport" in source or (source.endswith("-targeted-final20") and source != "en-targeted-final20")


def rule_completion_audit_sample(reports, size):
    ranked = defaultdict(list)
    for source_id, report in reports.items():
        for row in report["proposed_affected"]:
            for addition in row["additions"]:
                item = {
                    "source_id": source_id,
                    "id": row["id"],
                    "lang": row["lang"],
                    "source": row["source"],
                    **addition,
                }
                rank = hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).digest()
                stratum = f"{source_id}/{addition['rule']}"
                ranked[stratum].append((rank, item))
    sample = []
    sample_counts = {}
    for stratum, rows in sorted(ranked.items()):
        selected = [item for _, item in sorted(rows, key=lambda pair: pair[0])[:size]]
        sample.extend(selected)
        sample_counts[stratum] = len(selected)
    encoded = json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()
    return sample, hashlib.sha256(encoded).hexdigest(), sample_counts


def cmd_build(args):
    rng = random.Random(args.seed)
    rule_completion_threshold = getattr(args, "rule_completion_threshold", DEFAULT_COVERAGE_THRESHOLD)
    rule_completion_mode = getattr(args, "rule_completion_mode", "apply")
    rule_completion_audit_size = getattr(args, "rule_completion_audit_size", 100)
    ts = Tagset()
    # Identity schema for sources whose spans already carry target nodes
    # (clinical-authored). Every tagset node maps to itself.
    ts.sources["_identity"] = {n: n for n in ts.nodes}
    records = []
    node_counts = Counter()
    stats = defaultdict(Counter)
    unmapped = Counter()
    rule_completion_reports = {}
    requested_sources = set(args.source) if args.source else None
    available_sources = set()
    for src, schema, it in sources(args, rng):
        available_sources.add(src)
        if requested_sources is not None and src not in requested_sources:
            continue
        smap = ts.sources[schema]
        source_records = []
        for source_record_number, loaded in enumerate(it, 1):
            if len(loaded) == 4:
                rec_id, text, raw_spans, lang = loaded
                locator = {
                    "source_stream": src,
                    "record_1based": source_record_number,
                    "id": rec_id,
                }
            else:
                rec_id, text, raw_spans, lang, locator = loaded
            spans = []
            for start, end, label in raw_spans:
                if not (0 <= start < end <= len(text)):
                    stats[src]["bad_offset"] += 1
                    continue
                node = smap.get(label)
                if node is None:
                    unmapped[f"{schema}.{label}"] += 1
                    continue
                spans.append([start, end, node])
            spans, dropped = resolve_overlaps(spans)
            stats[src]["overlap_dropped"] += dropped
            if not text.strip():
                continue
            record = {
                "id": rec_id,
                "text": text,
                "spans": spans,
                "src": src,
                "lang": lang,
                "_rule_source_locator": locator,
            }
            if src in ANNOTATED_SPANS_ONLY_SOURCES:
                record["supervision"] = "annotated_spans_only"
            source_records.append(record)

        upstream_required = requires_upstream_rule_completion(src)
        apply_completion = rule_completion_mode == "apply" and not upstream_required
        source_records, rule_report = complete_records(
            source_records,
            set(smap.values()),
            source_id=src,
            coverage_threshold=rule_completion_threshold,
            apply=apply_completion,
        )
        if upstream_required:
            rule_report["deferred_reason"] = (
                "translated derivative: complete and retranslate affected source carrier rows"
            )
        elif rule_completion_mode == "audit":
            rule_report["deferred_reason"] = "audit-only pass: no proposed span was materialized"
        rule_completion_reports[src] = rule_report
        stats[src]["rule_added"] = rule_report["added_spans"]
        stats[src]["rule_proposed"] = rule_report["proposed_spans"]
        stats[src]["rule_added_affected_docs"] = rule_report["affected_documents"]
        stats[src]["rule_proposed_affected_docs"] = rule_report["proposed_affected_documents"]
        for record in source_records:
            spans = record["spans"]
            for _, _, node in spans:
                node_counts[node] += 1
            records.append(record)
            stats[src]["docs"] += 1
            stats[src]["spans"] += len(spans)
            stats[f"{src}/lang"][record["lang"]] += 1
        message = f"ASSEMBLE: {src}: {stats[src]['docs']} docs, {stats[src]['spans']} spans"
        print(message)
        headline(message)
    if requested_sources is not None:
        unknown_sources = requested_sources - available_sources
        if unknown_sources:
            available = ", ".join(sorted(available_sources))
            unknown = ", ".join(sorted(unknown_sources))
            sys.exit(f"unknown --source value(s): {unknown}; available sources: {available}")
    if unmapped:
        for k, n in unmapped.most_common():
            print(f"ASSEMBLE: UNMAPPED {k}: {n}", file=sys.stderr)
        sys.exit(f"unmapped labels in {len(unmapped)} schema.label keys; extend pii_tagset.yaml")
    if args.released_source_set == "onboarded-full":
        for src, expected in expected_full_source_counts().items():
            if requested_sources is not None and src not in requested_sources:
                continue
            actual = stats[src]["docs"]
            if actual != expected:
                sys.exit(f"{src} was not fully consumed: expected {expected} records, observed {actual}")

    # Collapse rare nodes upward so the label space stays dense.
    collapse = {}
    for node, n in node_counts.items():
        target = node
        while node_counts_at(node_counts, collapse, target) < args.min_node_count:
            parent = (ts.nodes[target] or {}).get("parent")
            if parent is None:
                break
            target = parent
        if target != node:
            collapse[node] = target
    for r in records:
        for s in r["spans"]:
            s[2] = collapse.get(s[2], s[2])
        for addition in (r.get("rule_completion") or {}).get("additions", []):
            addition["label"] = collapse.get(addition["label"], addition["label"])
    final_counts = Counter(s[2] for r in records for s in r["spans"])
    retained_labels = set()
    retained_labels_path = None
    if args.retain_labels_from:
        retained_labels_path = Path(args.retain_labels_from)
        if retained_labels_path.is_dir():
            retained_labels_path /= "labels.json"
        retained = json.loads(retained_labels_path.read_text(encoding="utf-8"))
        retained_labels = set(retained["labels"])
        unknown_retained = retained_labels - set(ts.nodes)
        if unknown_retained:
            sys.exit(
                f"{retained_labels_path} contains labels absent from the canonical hierarchy: "
                f"{', '.join(sorted(unknown_retained))}"
            )
    output_labels = sorted(set(final_counts) | retained_labels)

    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    os.makedirs(args.out, exist_ok=True)
    for name, part in (("val", records[:n_val]), ("train", records[n_val:])):
        with open(os.path.join(args.out, name + ".jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(
        {
            "labels": output_labels,
            "counts": {label: final_counts[label] for label in output_labels},
            "collapsed": collapse,
            "retained_labels_from": str(retained_labels_path) if retained_labels_path else None,
        },
        open(os.path.join(args.out, "labels.json"), "w"),
        indent=2,
    )
    json.dump(
        {k: dict(v) for k, v in stats.items()},
        open(os.path.join(args.out, "stats.json"), "w"),
        indent=2,
    )
    rule_completion_path = os.path.join(args.out, "rule-completion.json")
    audit_sample, audit_sample_sha256, audit_sample_counts = rule_completion_audit_sample(
        rule_completion_reports, rule_completion_audit_size
    )
    json.dump(
        {
            "schema_version": 1,
            "ruleset": RULESET_VERSION,
            "mode": rule_completion_mode,
            "coverage_threshold": rule_completion_threshold,
            "precision_gate": {
                "minimum_safe_fraction": 0.98,
                "contract": (
                    "iterate rule precision and rerun the audit until the reviewed sample passes; "
                    "do not treat a failed attempt as grounds to abandon completion"
                ),
                "sampling_unit": "up to sample_size_per_source_rule for each source/recognizer stratum",
                "sample_size_per_source_rule": rule_completion_audit_size,
                "sample_size": len(audit_sample),
                "sample_counts": audit_sample_counts,
                "sample_sha256": audit_sample_sha256,
                "sample": audit_sample,
            },
            "sources": rule_completion_reports,
        },
        open(rule_completion_path, "w"),
        ensure_ascii=False,
        indent=2,
    )
    onboarded_sources = {}
    if args.released_source_set == "onboarded-full":
        for slug in ("nemotron-pii", "openpii-1m"):
            manifest = onboarded_manifest(slug)
            onboarded_sources[slug] = {
                "repo_id": manifest["upstream"]["repo_id"],
                "revision": manifest["upstream"]["revision"],
                "records": manifest["counts"]["records"],
                "spans": manifest["counts"]["spans"],
                "tagset_sha256": manifest["tagset_sha256"],
            }
    json.dump(
        {
            "schema_version": 1,
            "released_source_set": args.released_source_set,
            "final20_transport": args.final20_transport,
            "requested_sources": args.source,
            "source_supervision": {
                source: ("annotated_spans_only" if source in ANNOTATED_SPANS_ONLY_SOURCES else "complete")
                for source in sorted(key for key in stats if not key.endswith("/lang"))
            },
            "rule_completion": {
                "ruleset": RULESET_VERSION,
                "mode": rule_completion_mode,
                "coverage_threshold": rule_completion_threshold,
                "report": rule_completion_path,
                "translated_derivatives_require_upstream_retranslation": True,
            },
            "retain_labels_from": str(retained_labels_path) if retained_labels_path else None,
            "onboarded_sources": onboarded_sources,
            "seed": args.seed,
            "cap_ai4p_lang": args.cap_ai4p_lang,
            "cap_nemotron": args.cap_nemotron,
            "min_node_count": args.min_node_count,
            "val_frac": args.val_frac,
            "held_out_evaluation": [
                "gold/*-fresh.jsonl",
                "mgs-*",
                "meddocan-test",
                "tab-test",
                "mapa/test",
                "idner-news-2k/validation",
                "idner-news-2k/test",
                "hiner/validation",
                "hiner/test",
                "wojood-sample/validation",
                "wojood-sample/test",
                "aqmar-openner/validation",
                "aqmar-openner/test",
                "openner-commercial-core/validation",
                "openner-commercial-core/test",
            ],
        },
        open(os.path.join(args.out, "build.json"), "w"),
        indent=2,
    )
    message = (
        f"ASSEMBLE: wrote {len(records) - n_val} train / {n_val} val docs, "
        f"{sum(final_counts.values())} spans, {len(output_labels)} labels "
        f"({len(final_counts)} observed), "
        f"({len(collapse)} collapsed) -> {args.out}"
    )
    print(message)
    headline(message)


def node_counts_at(counts, collapse, node):
    """Corpus count a node would have after current collapses into it."""
    return counts[node] + sum(n for src, n in counts.items() if collapse.get(src) == node)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--cap-ai4p-lang", type=int, default=6000)
    b.add_argument("--cap-nemotron", type=int, default=12000)
    b.add_argument(
        "--released-source-set",
        choices=("legacy-sampled", "onboarded-full"),
        default="legacy-sampled",
        help=(
            "legacy-sampled reproduces v1/v2 caps; onboarded-full consumes every committed "
            "Nemotron-PII and OpenPII 1M record"
        ),
    )
    b.add_argument("--min-node-count", type=int, default=300)
    b.add_argument("--val-frac", type=float, default=0.02)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument(
        "--rule-completion-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "activate a high-precision completion rule when its same-type gold coverage is "
            "strictly below this fraction (default: 0.5)"
        ),
    )
    b.add_argument(
        "--rule-completion-mode",
        choices=("audit", "apply"),
        default="audit",
        help="audit proposals without changing spans, or apply after the >=98%% precision review passes",
    )
    b.add_argument(
        "--rule-completion-audit-size",
        type=int,
        default=100,
        help="deterministic proposals retained per source/recognizer for precision review (default: 100)",
    )
    b.add_argument(
        "--final20-transport",
        action="store_true",
        help=(
            "expose the committed Final20 broad and targeted canonical annotations as opt-in sources; "
            "combine with repeated --source to build the translation-stage increment alone"
        ),
    )
    b.add_argument(
        "--retain-labels-from",
        help=(
            "labels.json or a corpus directory whose canonical label inventory is retained even when "
            "some labels are absent from this stage increment; required for honestly labeled replay"
        ),
    )
    b.add_argument(
        "--source",
        action="append",
        help="include only this named corpus source; repeat for a controlled mixture (default: all)",
    )
    args = ap.parse_args()
    cmd_build(args)


if __name__ == "__main__":
    main()
