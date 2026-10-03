#!/usr/bin/env python
"""Build the many:many correctness map between the v1 and v2 PII ontologies.

A correctness map says, for one old-ontology category, which new-ontology
classes count as a correct label for the same span -- and, inverted, which old
categories a new class may legitimately have come from. Dual-head training
reads it so a row labelled in either ontology supervises both heads: the head
that owns the row's own vocabulary gets ordinary cross-entropy, and the other
head gets a marginal ("some member of this allowed set") term.

The map has two independent derivations, and this tool reconciles them:

* **Definitional** -- the committed v2 spec already answers the question for
  every canonical v1 node (``old_v1.<node>.acceptable``) and for every source
  schema label (``sources.<schema>.<label>.acceptable``). That is a reading of
  the two ontologies' definitions, made when the v2 tagset was written.
* **Empirical** -- wherever the same source segment exists under both
  ontologies, the pair of labels assigned to one span is direct evidence of an
  equivalence. The v2 gold views were relabelled from the qualified intake
  without moving span boundaries, so joining the two on (record id, start, end)
  yields observed (v1 label, v2 class) pairs with counts.

Agreement is the common case. The interesting output is disagreement: a pair
observed often but absent from the definitional acceptable set is either a
missing entry in the map or a wrong incumbent v1 tag, and the tool reports it
rather than silently widening the training objective. Only pairs that clear
both an absolute count and a per-node share threshold are admitted, and every
admitted target records which derivation put it there.

Evidence comes from training-disposition views only. An audit view (the frozen
evaluation stratum) can be scanned with ``--audit-view``; its observations are
reported for comparison and never admitted into the map the trainer reads.

    build   join the views, reconcile with the spec, write the map JSON
    check   revalidate an existing map against the current ontology spec
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pii_ontology_v2 import ONTOLOGY_VERSION, OntologyError, load_ontology  # noqa: E402

SCHEMA = "pii-ontology-v1-v2-correctness-map"
SCHEMA_VERSION = 1

# A v1 span whose v2 counterpart is absent is evidence for "this old category
# can correctly become nothing", which the spec spells as the O target.
OUTSIDE = "O"


class MapError(RuntimeError):
    """A malformed input or an unusable reconciliation result."""


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise MapError(f"{path}:{lineno}: {exc}") from None


def shard_files(root: Path, subdirectory: str) -> list[Path]:
    """Every record file under one view root, in a stable order."""
    base = root / subdirectory
    if not base.is_dir():
        raise MapError(f"{root}: expected a {subdirectory!r} directory")
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.name.endswith((".jsonl", ".jsonl.gz")))
    if not files:
        raise MapError(f"{base}: no .jsonl or .jsonl.gz record files")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v1_key(span: dict[str, Any], schema: str) -> tuple[str, str]:
    """The old-ontology identity of an intake span.

    Prefer the canonical v1 node the onboarding adapter recorded, because the
    spec's ``old_v1`` table is keyed by it and it is shared across source
    schemas. Fall back to the source schema's own label, which the spec covers
    separately under ``sources``.
    """
    metadata = span.get("source_metadata") or {}
    node = metadata.get("onboarded_v1_label")
    if isinstance(node, str) and node:
        return "canonical", node
    label = span.get("source_label")
    if not isinstance(label, str) or not label:
        raise MapError(f"intake span has neither onboarded_v1_label nor source_label: {span!r}")
    return "source", f"{schema}\t{label}"


def index_intake(root: Path) -> tuple[dict[str, dict[tuple[int, int], tuple[str, str]]], Counter]:
    """Map every intake record id to its spans' old-ontology identities."""
    index: dict[str, dict[tuple[int, int], tuple[str, str]]] = {}
    counts: Counter = Counter()
    for path in shard_files(root, "shards"):
        for record in read_jsonl(path):
            record_id = record.get("id")
            schema = record.get("source_schema")
            if not isinstance(record_id, str) or not isinstance(schema, str):
                raise MapError(f"{path}: record without an id/source_schema: {str(record)[:120]}")
            spans = record.get("spans") or []
            if not spans:
                continue
            offsets = index.setdefault(record_id, {})
            for span in spans:
                offsets[(int(span["start"]), int(span["end"]))] = v1_key(span, schema)
                counts["intake_spans"] += 1
            counts["intake_records_with_spans"] += 1
    counts["intake_records_indexed"] = len(index)
    return index, counts


def observe_view(
    view_root: Path,
    intake: dict[str, dict[tuple[int, int], tuple[str, str]]],
) -> tuple[dict[tuple[str, str], Counter], dict[tuple[str, str], set[str]], Counter]:
    """Count (v1 identity, v2 class) pairs over one relabelled gold view."""
    pairs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: Counter = Counter()
    seen: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for path in shard_files(view_root, "gold"):
        for record in read_jsonl(path):
            record_id = record.get("id")
            if not isinstance(record_id, str):
                raise MapError(f"{path}: record without an id")
            dataset = str(record.get("source_dataset") or "unknown")
            offsets = intake.get(record_id)
            if offsets is None:
                counts["gold_records_without_intake"] += 1
                continue
            counts["gold_records_joined"] += 1
            for span in record.get("spans") or []:
                key = (int(span["start"]), int(span["end"]))
                identity = offsets.get(key)
                if identity is None:
                    counts["gold_spans_without_intake_span"] += 1
                    continue
                seen[record_id].add(key)
                pairs[identity][str(span["type"])] += 1
                datasets[identity].add(dataset)
                counts["joined_spans"] += 1
            # An intake span the relabelling dropped is evidence for the O
            # target, but only for a record the view actually covers.
            for key, identity in offsets.items():
                if key not in seen[record_id]:
                    pairs[identity][OUTSIDE] += 1
                    datasets[identity].add(dataset)
                    counts["dropped_spans"] += 1
            seen.pop(record_id, None)
    return pairs, datasets, counts


def definitional_map(ontology) -> dict[tuple[str, str], tuple[str, tuple[str, ...]]]:
    """Every v1 identity the spec covers, with its fallback and acceptable set."""
    table: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    for node in ontology.old_v1:
        table[("canonical", node)] = (
            ontology.canonical_fallback(node),
            tuple(ontology.canonical_acceptable(node)),
        )
    for schema in ontology.source_schemas():
        # Legacy v1 labels and the schema's declared extensions both name spans
        # a corpus actually carries, so both need an acceptable set here.
        for label in (*ontology.source_labels(schema), *ontology.source_extension_labels(schema)):
            key = ("source", f"{schema}\t{label}")
            table[key] = (
                ontology.source_fallback(schema, label),
                tuple(ontology.source_acceptable(schema, label)),
            )
    return table


def identity_name(identity: tuple[str, str]) -> str:
    kind, value = identity
    return value if kind == "canonical" else value.replace("\t", ".")


def reconcile(
    ontology,
    pairs: dict[tuple[str, str], Counter],
    datasets: dict[tuple[str, str], set[str]],
    min_support: int,
    min_share: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge the two derivations into one map, keeping every target's origin."""
    spec = definitional_map(ontology)
    unknown = sorted(identity_name(k) for k in pairs if k not in spec)
    if unknown:
        raise MapError(
            f"{len(unknown)} observed v1 label(s) are absent from the ontology spec, so their"
            f" acceptable set is undefined: {', '.join(unknown[:6])}"
        )
    entries: dict[str, Any] = {}
    off_map: list[dict[str, Any]] = []
    for identity, (fallback, acceptable) in sorted(spec.items(), key=lambda item: identity_name(item[0])):
        observed = pairs.get(identity, Counter())
        total = sum(observed.values())
        origin = {target: "definitional" for target in acceptable}
        for target, count in observed.items():
            if target in origin:
                continue
            share = count / total if total else 0.0
            admitted = count >= min_support and share >= min_share
            off_map.append(
                {
                    "v1": identity_name(identity),
                    "v1_kind": identity[0],
                    "v2": target,
                    "count": count,
                    "share": round(share, 6),
                    "observations": total,
                    "admitted": admitted,
                    "datasets": sorted(datasets.get(identity, ())),
                }
            )
            if admitted:
                origin[target] = "empirical"
        entries[identity_name(identity)] = {
            "kind": identity[0],
            "fallback": fallback,
            "definitional": list(acceptable),
            "accepted": sorted(origin),
            "origin": dict(sorted(origin.items())),
            "observed": dict(sorted(observed.items(), key=lambda kv: (-kv[1], kv[0]))),
            "observations": total,
        }
    off_map.sort(key=lambda row: (-row["count"], row["v1"], row["v2"]))
    return entries, off_map


def invert(entries: dict[str, Any]) -> dict[str, list[str]]:
    """v2 class -> the v1 identities that may correctly carry it."""
    inverse: dict[str, set[str]] = defaultdict(set)
    for name, entry in entries.items():
        for target in entry["accepted"]:
            inverse[target].add(name)
    return {target: sorted(sources) for target, sources in sorted(inverse.items())}


def head_support(
    document: dict[str, Any],
    tagset,
    head_classes: Iterable[str],
    target_classes: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, tuple[str, ...]]]]:
    """Project canonical and source-specific map cells onto affine head rows."""
    entries = document.get("v1_to_v2")
    if not isinstance(entries, dict):
        raise MapError("correctness map has no v1_to_v2 table")
    sources = tuple(head_classes)
    target_order = (OUTSIDE, *target_classes)
    target_set = set(target_order)
    origins: dict[str, dict[str, set[str]]] = {source: defaultdict(set) for source in sources}

    def add(source: str, entry: dict[str, Any], origin: str) -> None:
        accepted = entry.get("accepted")
        if (
            not isinstance(accepted, list)
            or not accepted
            or not all(isinstance(target, str) for target in accepted)
        ):
            raise MapError(f"{origin}: expected a nonempty accepted target list")
        unknown = sorted(set(accepted) - target_set)
        if unknown:
            raise MapError(f"{origin}: accepted targets outside the v2 inventory: {', '.join(unknown)}")
        if len(accepted) != len(set(accepted)):
            raise MapError(f"{origin}: duplicate accepted targets")
        for target in accepted:
            origins[source][target].add(origin)

    for source in sources:
        entry = entries.get(source)
        if not isinstance(entry, dict) or entry.get("kind") != "canonical":
            raise MapError(f"no canonical correctness-map entry for v1 head class {source!r}")
        add(source, entry, "canonical")

    source_set = set(sources)
    for name, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("kind") != "source":
            continue
        schema, separator, label = name.partition(".")
        if not separator or schema not in tagset.sources:
            raise MapError(f"malformed source correctness-map key {name!r}")
        canonical = tagset.sources[schema].get(label)
        if canonical is None:
            continue  # Declared source extensions have no incumbent affine row.
        if canonical in source_set:
            add(canonical, entry, f"source:{name}")

    support = {
        source: tuple(target for target in target_order if target in origins[source]) for source in sources
    }
    frozen_origins = {
        source: {target: tuple(sorted(origins[source][target])) for target in support[source]}
        for source in sources
    }
    return support, frozen_origins


def build(args: argparse.Namespace) -> int:
    ontology = load_ontology(args.ontology)
    intake, intake_counts = index_intake(args.intake_root)

    pairs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    evidence_counts: Counter = Counter()
    for view in args.gold_view:
        view_pairs, view_datasets, counts = observe_view(view, intake)
        for identity, observed in view_pairs.items():
            pairs[identity].update(observed)
            datasets[identity].update(view_datasets[identity])
        evidence_counts.update(counts)

    audit_pairs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    audit_counts: Counter = Counter()
    for view in args.audit_view:
        view_pairs, _, counts = observe_view(view, intake)
        for identity, observed in view_pairs.items():
            audit_pairs[identity].update(observed)
        audit_counts.update(counts)

    entries, off_map = reconcile(ontology, pairs, datasets, args.min_support, args.min_share)

    audit_disagreement = []
    for identity, observed in sorted(audit_pairs.items(), key=lambda item: identity_name(item[0])):
        name = identity_name(identity)
        accepted = set(entries[name]["accepted"]) if name in entries else set()
        for target, count in observed.items():
            if target not in accepted:
                audit_disagreement.append({"v1": name, "v2": target, "count": count})
    audit_disagreement.sort(key=lambda row: (-row["count"], row["v1"], row["v2"]))

    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ontology": {
            "path": str(args.ontology),
            "sha256": sha256_file(args.ontology),
            "ontology_version": ONTOLOGY_VERSION,
            "projection_version": ontology.projection_version,
            "primary_types": list(ontology.primary_types),
        },
        "evidence": {
            "intake_root": str(args.intake_root),
            "gold_views": [str(view) for view in args.gold_view],
            "audit_views": [str(view) for view in args.audit_view],
            "counts": dict(sorted(intake_counts.items())) | dict(sorted(evidence_counts.items())),
            "audit_counts": dict(sorted(audit_counts.items())),
        },
        "evidence_caveat": (
            "The v2 gold views were relabelled under the same acceptable sets this map records, so"
            " agreement between the definitional and empirical derivations is partly by construction."
            " What the join independently establishes is which pairs actually occur and how often,"
            " which targets no corpus exercises, and that no relabelled span left its allowed set."
            " Evidence that a definitional pair is wrong has to come from outside the remap:"
            " the route-level judge verdicts, or a corpus labelled in v1 after the relabelling."
        ),
        "thresholds": {"min_support": args.min_support, "min_share": args.min_share},
        "v1_to_v2": entries,
        "v2_to_v1": invert(entries),
        "off_map_observations": off_map,
        "audit_disagreement": audit_disagreement,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )

    admitted = sum(1 for row in off_map if row["admitted"])
    covered = sum(1 for entry in entries.values() if entry["observations"])
    print(
        f"PII_V1_V2_MAP phase=build v1_labels={len(entries)} with_evidence={covered}"
        f" joined_spans={evidence_counts['joined_spans']} dropped_spans={evidence_counts['dropped_spans']}"
        f" off_map={len(off_map)} admitted={admitted} audit_disagreements={len(audit_disagreement)}"
        f" out={args.out}",
        flush=True,
    )
    for row in off_map[:15]:
        mark = "ADMIT" if row["admitted"] else "flag "
        print(f"  {mark} {row['v1']} -> {row['v2']} n={row['count']} share={row['share']:.4f}", flush=True)
    return 0


def check(args: argparse.Namespace) -> int:
    document = json.loads(args.map.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise MapError(f"{args.map}: schema {document.get('schema')!r} is not {SCHEMA!r}")
    ontology = load_ontology(args.ontology)
    spec = {identity_name(key): value for key, value in definitional_map(ontology).items()}
    entries = document["v1_to_v2"]
    missing = sorted(set(spec) - set(entries))
    extra = sorted(set(entries) - set(spec))
    if missing or extra:
        raise MapError(
            f"{args.map}: v1 label coverage disagrees with the spec: {len(missing)} missing, {len(extra)} unknown"
        )
    primaries = set(ontology.primary_types) | {OUTSIDE}
    for name, entry in entries.items():
        fallback, acceptable = spec[name]
        if entry["fallback"] != fallback or list(entry["definitional"]) != list(acceptable):
            raise MapError(f"{args.map}: {name} definitional entry drifted from the spec")
        if not set(entry["accepted"]) >= set(acceptable):
            raise MapError(f"{args.map}: {name} dropped a definitional target from its accepted set")
        unknown = sorted(set(entry["accepted"]) - primaries)
        if unknown:
            raise MapError(f"{args.map}: {name} accepts non-ontology target(s) {', '.join(unknown)}")
    inverse = document["v2_to_v1"]
    rebuilt = invert(entries)
    if inverse != rebuilt:
        raise MapError(f"{args.map}: v2_to_v1 is not the inverse of v1_to_v2")
    print(
        f"PII_V1_V2_MAP phase=check v1_labels={len(entries)} v2_classes={len(inverse)}"
        f" empirical_targets={sum(1 for e in entries.values() for o in e['origin'].values() if o == 'empirical')}"
        f" map={args.map}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build", help="join the views and write the correctness map")
    builder.add_argument("--ontology", type=Path, default=Path(__file__).with_name("pii_tagset_v2.yaml"))
    builder.add_argument(
        "--intake-root", type=Path, required=True, help="qualified intake root (v1 source labels)"
    )
    builder.add_argument(
        "--gold-view",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="relabelled v2 gold view whose observations may enter the map (training dispositions only)",
    )
    builder.add_argument(
        "--audit-view",
        type=Path,
        action="append",
        default=[],
        help="held-out view reported for comparison and never admitted into the map",
    )
    builder.add_argument("--min-support", type=int, default=20, help="minimum count for an off-map pair")
    builder.add_argument(
        "--min-share", type=float, default=0.02, help="minimum share of the v1 label's observations"
    )
    builder.add_argument("--out", type=Path, required=True)
    builder.set_defaults(func=build)

    checker = sub.add_parser("check", help="revalidate a map against the current ontology spec")
    checker.add_argument("--map", type=Path, required=True)
    checker.add_argument("--ontology", type=Path, default=Path(__file__).with_name("pii_tagset_v2.yaml"))
    checker.set_defaults(func=check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MapError, OntologyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
