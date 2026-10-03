"""Compile overlap-screened gold views with exact branch sampling probabilities."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from pii_annotation_sampling import share_annotation_sampling_mass  # noqa: E402
from pii_encoder_train import sampling_plan_for_rows, window_records  # noqa: E402
from pii_segment_eval import segments  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def read(path):
    return [json.loads(line) for line in path.open()]


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flat_views(spans):
    """Partition intervals without dropping nested positives or inventing O labels."""
    views = []
    for span in sorted(spans):
        for view in views:
            if view[-1][1] <= span[0]:
                view.append(span)
                break
        else:
            views.append([span])
    if Counter(map(tuple, spans)) != Counter(tuple(span) for view in views for span in view):
        raise AssertionError("annotation partition changed positive spans")
    return views


def gold_windows(row, tokenizer, outside_policy, unknown_types):
    """Window one gold row with its spans protected, one training row per flat view."""
    replay = []
    covered = []
    for offset, text in segments(row["text"], 900, row["spans"], tokenizer=tokenizer, max_tokens=512):
        end = offset + len(text)
        spans = [[a - offset, b - offset, label] for a, b, label in row["spans"] if offset <= a < b <= end]
        covered.extend([a + offset, b + offset, label] for a, b, label in spans)
        window_views = flat_views(spans)
        if not window_views and outside_policy == "full-corpus":
            window_views = [[]]
        for view in window_views:
            coverage = (
                {"supervision": "complete", "unknown_primary_types": unknown_types}
                if outside_policy == "full-corpus"
                or (outside_policy == "corpus-covered" and len(window_views) == 1)
                else {}
            )
            if outside_policy == "full-corpus":
                # Only the portions of alternate positives absent from this
                # flat view are unknown; genuine outside text stays negative.
                boundaries = sorted({point for a, b, _ in spans for point in (a, b)})
                ignored = []
                for a, b in zip(boundaries, boundaries[1:]):
                    if any(x <= a < b <= y for x, y, _ in spans) and not any(
                        x <= a < b <= y for x, y, _ in view
                    ):
                        if ignored and ignored[-1][1] == a:
                            ignored[-1][1] = b
                        else:
                            ignored.append([a, b, "alternate_gold_positive"])
                if ignored:
                    coverage["ignored_spans"] = ignored
            replay.append(
                {
                    **row,
                    **coverage,
                    "text": text,
                    "spans": view,
                    "sampling_source_start": offset,
                    "sampling_source_end": end,
                    "document_context": {"before": row["text"][:offset], "after": row["text"][end:]},
                }
            )
    if Counter(map(tuple, covered)) != Counter(map(tuple, row["spans"])):
        raise ValueError(f"windowing lost gold span: {row['id']}")
    return replay


def build(args):
    if not args.doses or any(not 0 < dose < 1 for dose in args.doses):
        raise ValueError("gold sampling probabilities must be between zero and one")
    control_receipt = json.loads((args.control / "receipt.json").read_text())
    for name in ("train.jsonl", "mapping.json", "labels.json", "val.jsonl", "evaluation.jsonl"):
        if identity(args.control / name)["sha256"] != control_receipt["outputs"][name]["sha256"]:
            raise ValueError(f"control input drift: {name}")
    screen_receipt = json.loads((args.screen / "receipt.json").read_text())
    ids_path = args.screen / "retained.ids"
    if identity(ids_path)["sha256"] != screen_receipt["outputs"]["retained.ids"]["sha256"]:
        raise ValueError("overlap membership drift")
    queries_path = args.intake / "incoming-queries.jsonl"
    expected_query = next(
        item for item in screen_receipt["inputs"] if item["path"].endswith("/incoming-queries.jsonl")
    )
    if identity(queries_path)["sha256"] != expected_query["sha256"]:
        raise ValueError("overlap query drift")
    retained = set(ids_path.read_text().splitlines())
    queries = {row["id"]: row for row in read(queries_path)}
    mapping = json.loads((args.control / "mapping.json").read_text())
    unknown_by_corpus = {
        corpus: sorted(
            set(mapping["ontology"]["primary_types"])
            - {tag for node in labels.values() for tag in mapping["v1_to_v2"][node]["accepted"]}
        )
        for corpus, labels in mapping["source_labels"].items()
    }
    if args.negative_coverage:
        if args.outside_policy != "full-corpus":
            raise ValueError("explicit negative coverage requires full-corpus supervision")
        coverage = json.loads(args.negative_coverage.read_text())["negative_types_by_corpus"]
        if set(coverage) != set(unknown_by_corpus):
            raise ValueError("negative coverage must name every source corpus exactly")
        primary = set(mapping["ontology"]["primary_types"])
        for corpus, negative_types in coverage.items():
            if (
                not isinstance(negative_types, list)
                or any(not isinstance(tag, str) for tag in negative_types)
                or len(set(negative_types)) != len(negative_types)
                or not set(negative_types) <= primary
            ):
                raise ValueError(f"invalid negative coverage: {corpus}")
            unknown_by_corpus[corpus] = sorted(primary - set(negative_types))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    base = read(args.control / "train.jsonl")
    base_weights, _, _ = sampling_plan_for_rows(base)
    base_weights, base_sharing = share_annotation_sampling_mass(base, base_weights)
    base_total = math.fsum(base_weights)
    base_weights = [weight / base_total for weight in base_weights]
    sources = {}
    inputs = [
        identity(args.control / "receipt.json"),
        identity(args.screen / "receipt.json"),
        identity(queries_path),
    ]
    if args.negative_coverage:
        inputs.append(identity(args.negative_coverage))
    rows = []
    audit = []
    seen = set()
    for key in sorted(retained):
        query = queries[key]
        if query["split"] != "train":
            raise ValueError(f"non-training member: {key}")
        corpus = query["corpus"]
        if corpus not in sources:
            path = args.intake / f"{corpus}-train.jsonl"
            inputs.append(identity(path))
            sources[corpus] = read(path)
        original = sources[corpus][query["source_locator"]["line"] - 1]
        if (
            original["id"] != query["source_id"]
            or original["text"] != query["text"]
            or original["lang"] != query["lang"]
        ):
            raise ValueError(f"source locator mismatch: {key}")
        spans = []
        for span in original["spans"]:
            node = mapping["source_labels"][corpus][span["source_label"]]
            accepted = mapping["v1_to_v2"][node]["accepted"]
            if accepted != ["O"]:
                spans.append([span["start"], span["end"], node])
        views = flat_views(spans)
        audit.append(
            {"id": key, "source_id": original["id"], "positive_spans": len(spans), "views": len(views)}
        )
        if views or args.outside_policy == "full-corpus":
            rows.append(
                {
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
            )
        seen.add(key)
    if seen != retained:
        raise AssertionError("membership mismatch")
    # Window with all source intervals protected, then partition annotations.
    replay = []
    for row in rows:
        replay.extend(gold_windows(row, tokenizer, args.outside_policy, unknown_by_corpus[row["src"]]))
    write(args.output / "annotation-views.jsonl", replay)
    replay_weights, replay_sharing = share_annotation_sampling_mass(replay, [1.0] * len(replay))
    total = math.fsum(replay_weights)
    replay_weights = [weight / total for weight in replay_weights]
    write(args.output / "annotation-audit.jsonl", audit)
    report = {
        "inputs": inputs,
        "retained_source_rows": len(retained),
        "source_rows_with_positives": sum(item["positive_spans"] > 0 for item in audit),
        "base_sharing": base_sharing,
        "replay_sharing": replay_sharing,
        "outside_policy": args.outside_policy,
        "negative_coverage": identity(args.negative_coverage) if args.negative_coverage else None,
        "unknown_types_by_corpus": unknown_by_corpus,
        "replay_supervision": dict(Counter(row["supervision"] for row in replay)),
        "doses": {},
        "sampling_policy": "Pre-shared within each branch; trainer annotation-variant-weighting=legacy consumes compiled probabilities without sharing again. Base distribution unchanged conditional on base; uniform distinct replay inputs, split among annotations. No language rebalance.",
    }
    for dose in args.doses:
        folder = args.output / f"p{round(dose * 100):02d}"
        folder.mkdir()
        mixture = [
            {**row, "sampling_weight": weight * (1 - dose), "sampling_branch": "base"}
            for row, weight in zip(base, base_weights, strict=True)
        ]
        mixture.extend(
            {**row, "sampling_weight": weight * dose}
            for row, weight in zip(replay, replay_weights, strict=True)
        )
        write(folder / "train.jsonl", mixture)
        for name in ("val.jsonl", "evaluation.jsonl", "mapping.json", "labels.json"):
            (folder / name).write_bytes((args.control / name).read_bytes())
        loaded = window_records(
            str(folder / "train.jsonl"),
            900,
            context_field="document_context",
            tokenizer=tokenizer,
            max_tokens=512,
        )
        actual, _, _ = sampling_plan_for_rows(loaded)
        if len(loaded) != len(mixture):
            raise ValueError("second loader changed membership")
        branch = defaultdict(float)
        languages = defaultdict(float)
        for expected, observed, weight in zip(mixture, loaded, actual, strict=True):
            for field in (
                "text",
                "spans",
                "lang",
                "label_space",
                "supervision",
                "unknown_primary_types",
                "document_context",
                "ignored_spans",
                "predicate_spans",
                "subclass_spans",
                "primary_span_objective_weights",
                "sampling_weight",
            ):
                if expected.get(field) != observed.get(field):
                    raise ValueError(f"loader changes {field}")
            branch[observed["sampling_branch"]] += weight
            languages[observed["lang"]] += weight
        if abs(branch["human_gold"] - dose) > 1e-10 or abs(math.fsum(actual) - 1) > 1e-10:
            raise ValueError("compiled branch probability differs from requested dose")
        dose_report = {
            "rows": len(mixture),
            "branch_probability": dict(branch),
            "language_probability": dict(languages),
            "outputs": {
                name: identity(folder / name)
                for name in ("train.jsonl", "val.jsonl", "evaluation.jsonl", "mapping.json", "labels.json")
            },
        }
        (folder / "receipt.json").write_text(json.dumps(dose_report, indent=2) + "\n")
        report["doses"][str(dose)] = dose_report
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "compile-mixture",
                "retained": len(retained),
                "replay_windows": len(replay),
                "doses": args.doses,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outside-policy",
        choices=("positive-only", "corpus-covered", "full-corpus"),
        default="positive-only",
    )
    parser.add_argument("--doses", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--negative-coverage", type=Path)
    for name in ("control", "screen", "intake", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())
