#!/usr/bin/env python
"""Package and verify reusable PII annotation caches.

The committed cache under data/pii-annotations avoids repeating expensive
frontier/LLM annotation.  Every cached span is normalized to the canonical
pii_tagset node vocabulary.  The manifest then defines two kinds of gates:

* coverage: every deployment tag has enough spans and distinct documents;
* balance: generated labels over same-domain text remain reasonably close to
  the hidden-gold type distribution after both are viewed through a common
  source schema.

Examples:
  python scripts/pii_annotation_cache.py pack \
      --input in.jsonl --output out.jsonl --schema authored_v1
  python scripts/pii_annotation_cache.py stats \
      --manifest data/pii-annotations/manifest.yaml \
      --output data/pii-annotations/stats.json
  python scripts/pii_annotation_cache.py check \
      --manifest data/pii-annotations/manifest.yaml \
      --stats data/pii-annotations/stats.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pii_projector import Tagset  # noqa: E402


def jsonl(path):
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def normalize_type(tagset, schema, label):
    """Return a canonical node, accepting canonical labels directly."""
    canonical = label.lower()
    if canonical in tagset.nodes:
        return canonical
    mapping = tagset.sources.get(schema)
    if mapping is None:
        raise ValueError(f"unknown source schema {schema!r}")
    node = mapping.get(label)
    if node is None:
        node = mapping.get(canonical)
    if node is None:
        node = mapping.get(label.upper())
    if node is None:
        raise ValueError(f"{schema}: unmapped label {label!r}")
    return node


def normalize_span(tagset, schema, span):
    if isinstance(span, list):
        if len(span) != 3:
            raise ValueError(f"span list must have three items: {span!r}")
        start, end, label = span
    else:
        start = span["start"]
        end = span["end"]
        label = span.get("type", span.get("label"))
        if label is None:
            raise ValueError(f"span has no type/label: {span!r}")
    return [int(start), int(end), normalize_type(tagset, schema, str(label))]


def load_texts(path):
    texts = {}
    for line_number, row in jsonl(path):
        doc_id = row.get("id")
        if doc_id is None:
            raise ValueError(f"{path}:{line_number}: text row has no id")
        if doc_id in texts:
            raise ValueError(f"{path}:{line_number}: duplicate id {doc_id!r}")
        texts[doc_id] = row["text"]
    return texts


@contextmanager
def atomic_text_output(path):
    """Replace path only after its complete new contents validate and flush."""
    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{os.path.basename(output_path)}.",
        delete=False,
    )
    try:
        yield temporary
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary.name, output_path)
    except BaseException:
        temporary.close()
        try:
            os.unlink(temporary.name)
        except FileNotFoundError:
            pass
        raise


def validate_spans(path, line_number, text, spans, allow_overlaps=False):
    previous_end = -1
    previous_key = None
    for start, end, label in spans:
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"{path}:{line_number}: invalid {label} span [{start}, {end}) for text length {len(text)}"
            )
        key = (start, end, label)
        if previous_key is not None and key < previous_key:
            raise ValueError(f"{path}:{line_number}: unsorted spans at {start}")
        if not allow_overlaps and start < previous_end:
            raise ValueError(f"{path}:{line_number}: overlapping spans at {start}")
        previous_key = key
        previous_end = max(previous_end, end)


def cmd_pack(args):
    tagset = Tagset()
    external_texts = load_texts(args.texts) if args.texts else {}
    seen_ids = set()
    written = span_count = 0
    with atomic_text_output(args.output) as output:
        for line_number, row in jsonl(args.input):
            doc_id = row.get("id", str(line_number - 1))
            if args.id_prefix:
                doc_id = f"{args.id_prefix}-{doc_id}"
            if doc_id in seen_ids:
                raise ValueError(f"{args.input}:{line_number}: duplicate output id {doc_id!r}")
            seen_ids.add(doc_id)
            raw_spans = row[args.spans_key]
            spans = sorted(
                (normalize_span(tagset, args.schema, span) for span in raw_spans),
                key=lambda span: (span[0], span[1], span[2]),
            )
            text = row.get("text")
            lookup_id = row.get("id", str(line_number - 1))
            if text is None:
                try:
                    text = external_texts[lookup_id]
                except KeyError as error:
                    raise ValueError(f"{args.input}:{line_number}: no text for id {lookup_id!r}") from error
            validate_spans(args.input, line_number, text, spans, args.allow_overlaps)
            packed = {"id": doc_id, "spans": spans}
            if not args.omit_text:
                packed["text"] = text
            for field in ("lang", "domain", "span_provenance", "materialization"):
                if field in row:
                    packed[field] = row[field]
            output.write(json.dumps(packed, ensure_ascii=False) + "\n")
            written += 1
            span_count += len(spans)
    print(f"PACK: {written} docs / {span_count} spans -> {args.output}")


def cmd_pack_texts(args):
    seen_ids = set()
    written = 0
    with atomic_text_output(args.output) as output:
        for line_number, row in jsonl(args.input):
            doc_id = row.get("id", str(line_number - 1))
            if args.id_prefix:
                doc_id = f"{args.id_prefix}-{doc_id}"
            if doc_id in seen_ids:
                raise ValueError(f"{args.input}:{line_number}: duplicate output id {doc_id!r}")
            seen_ids.add(doc_id)
            output.write(json.dumps({"id": doc_id, "text": row["text"]}, ensure_ascii=False) + "\n")
            written += 1
    print(f"PACK-TEXTS: {written} docs -> {args.output}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(base, path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def dataset_stats(base, dataset, excluded_types):
    path = resolve(base, dataset["path"])
    texts = load_texts(resolve(base, dataset["text_path"])) if dataset.get("text_path") else None
    counts = Counter()
    doc_counts = Counter()
    num_docs = 0
    num_chars = 0
    for line_number, row in jsonl(path):
        text = row.get("text")
        if text is None:
            if texts is None:
                raise ValueError(f"{path}:{line_number}: row has no text and dataset has no text_path")
            try:
                text = texts[row["id"]]
            except KeyError as error:
                raise ValueError(f"{path}:{line_number}: unknown text id {row['id']!r}") from error
        spans = row["spans"]
        validate_spans(path, line_number, text, spans, dataset.get("allow_overlaps", False))
        present = set()
        for _, _, label in spans:
            if label in excluded_types:
                raise ValueError(f"{path}:{line_number}: excluded type {label!r} is present")
            counts[label] += 1
            present.add(label)
        doc_counts.update(present)
        num_docs += 1
        num_chars += len(text)
    return {
        "path": dataset["path"],
        "sha256": sha256(path),
        "num_docs": num_docs,
        "num_chars": num_chars,
        "num_spans": sum(counts.values()),
        "num_types": len(counts),
        "spans_per_doc": round(sum(counts.values()) / num_docs, 6) if num_docs else 0.0,
        "counts": dict(sorted(counts.items())),
        "doc_counts": dict(sorted(doc_counts.items())),
    }


def coarsen_counts(tagset, counts, schema):
    mapping = tagset.sources[schema]
    coarse = Counter()
    for node, count in counts.items():
        source_labels = tagset.back_project(node, schema)
        target = mapping[source_labels[0]] if source_labels else "__other__"
        coarse[target] += count
    return coarse


def js_divergence_bits(left, right):
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        raise ValueError("Jensen-Shannon divergence needs two nonempty distributions")
    p = {key: left[key] / left_total for key in keys}
    q = {key: right[key] / right_total for key in keys}
    midpoint = {key: (p[key] + q[key]) / 2 for key in keys}

    def kl(distribution):
        return sum(
            probability * math.log2(probability / midpoint[key])
            for key, probability in distribution.items()
            if probability
        )

    return (kl(p) + kl(q)) / 2


def coverage_result(tagset, gate, all_stats):
    span_counts = Counter()
    document_counts = Counter()
    for dataset_id in gate["datasets"]:
        stats = all_stats[dataset_id]
        span_counts.update(stats["counts"])
        document_counts.update(stats["doc_counts"])
    if "required_schema" in gate:
        required_types = set(tagset.sources[gate["required_schema"]].values())
    else:
        required_types = set(gate["required_types"])
    required_types.update(gate.get("additional_required_types", []))
    required_types = sorted(required_types)
    deficits = {}
    for label in required_types:
        missing_spans = max(0, gate["min_spans_per_type"] - span_counts[label])
        missing_docs = max(0, gate["min_docs_per_type"] - document_counts[label])
        if missing_spans or missing_docs:
            deficits[label] = {
                "spans": span_counts[label],
                "docs": document_counts[label],
                "need_spans": missing_spans,
                "need_docs": missing_docs,
            }
    return {
        "passed": not deficits,
        "required": gate.get("required", True),
        "min_spans_per_type": gate["min_spans_per_type"],
        "min_docs_per_type": gate["min_docs_per_type"],
        "num_required_types": len(required_types),
        "num_covered_types": len(required_types) - len(deficits),
        "deficits": deficits,
    }


def balance_result(tagset, gate, all_stats):
    observed = Counter(all_stats[gate["dataset"]]["counts"])
    if gate.get("comparison_schema"):
        observed = coarsen_counts(tagset, observed, gate["comparison_schema"])
    reference = Counter(gate["reference_counts"])
    divergence = js_divergence_bits(observed, reference)
    return {
        "passed": divergence <= gate["max_js_divergence_bits"],
        "required": gate.get("required", True),
        "js_divergence_bits": round(divergence, 8),
        "max_js_divergence_bits": gate["max_js_divergence_bits"],
        "observed_counts": dict(sorted(observed.items())),
        "reference_counts": dict(sorted(reference.items())),
    }


def compute_stats(manifest_path):
    with open(manifest_path) as source:
        manifest = yaml.safe_load(source)
    base = os.path.dirname(os.path.abspath(manifest_path))
    tagset = Tagset(resolve(base, manifest["tagset"]))
    excluded_types = set(manifest.get("excluded_types", []))
    datasets = {
        dataset["id"]: dataset_stats(base, dataset, excluded_types) for dataset in manifest["datasets"]
    }
    coverage = {
        gate["id"]: coverage_result(tagset, gate, datasets) for gate in manifest.get("coverage_gates", [])
    }
    balance = {
        gate["id"]: balance_result(tagset, gate, datasets) for gate in manifest.get("balance_gates", [])
    }
    return {
        "schema_version": 1,
        "datasets": datasets,
        "coverage_gates": coverage,
        "balance_gates": balance,
    }


def cmd_stats(args):
    stats = compute_stats(args.manifest)
    with atomic_text_output(args.output) as output:
        json.dump(stats, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"STATS: {len(stats['datasets'])} datasets -> {args.output}")
    for family in ("coverage_gates", "balance_gates"):
        for gate_id, result in stats[family].items():
            if result["passed"]:
                status = "PASS"
            elif result["required"]:
                status = "FAIL"
            else:
                status = "REPORT-ONLY"
            print(f"STATS: {gate_id}: {status}")


def cmd_check(args):
    actual = compute_stats(args.manifest)
    with open(args.stats) as source:
        expected = json.load(source)
    errors = []
    if actual != expected:
        errors.append(f"cached statistics differ from {args.stats}; regenerate and review")
    for family in ("coverage_gates", "balance_gates"):
        for gate_id, result in actual[family].items():
            if result["required"] and not result["passed"]:
                errors.append(f"{family}.{gate_id} failed")
    if errors:
        raise SystemExit("CHECK FAILED:\n  " + "\n  ".join(errors))
    required_count = sum(
        result["required"]
        for family in ("coverage_gates", "balance_gates")
        for result in actual[family].values()
    )
    report_only_count = sum(
        not result["required"]
        for family in ("coverage_gates", "balance_gates")
        for result in actual[family].values()
    )
    print(
        f"CHECK: {len(actual['datasets'])} annotation datasets; "
        f"{required_count} required gates passed; "
        f"{report_only_count} report-only comparisons retained"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack", help="normalize one annotation JSONL")
    pack.add_argument("--input", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--schema", default="target")
    pack.add_argument("--spans-key", default="spans")
    pack.add_argument("--texts", help="optional JSONL carrying id/text for prediction-only input")
    pack.add_argument("--omit-text", action="store_true")
    pack.add_argument(
        "--allow-overlaps",
        action="store_true",
        help="retain overlapping teacher spans while still checking offsets and ordering",
    )
    pack.add_argument("--id-prefix", default="")
    pack.set_defaults(func=cmd_pack)

    texts = subparsers.add_parser("pack-texts", help="retain only id/text from a JSONL")
    texts.add_argument("--input", required=True)
    texts.add_argument("--output", required=True)
    texts.add_argument("--id-prefix", default="")
    texts.set_defaults(func=cmd_pack_texts)

    stats = subparsers.add_parser("stats", help="write manifest statistics")
    stats.add_argument("--manifest", required=True)
    stats.add_argument("--output", required=True)
    stats.set_defaults(func=cmd_stats)

    check = subparsers.add_parser("check", help="recompute and verify manifest statistics")
    check.add_argument("--manifest", required=True)
    check.add_argument("--stats", required=True)
    check.set_defaults(func=cmd_check)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
