#!/usr/bin/env python
"""Repair family-first order in CJK rows from audited defective sources.

Chinese, Japanese, and Korean write the family name first, and the
frontier-authored evaluation gold follows that convention (87.8% family-first
for Korean, 92.2% for Japanese). Affected materialized training sources do the
opposite: 3.7% family-first for Korean, 17.6% for Japanese, 2.8% for Chinese.
A tagger trained there learns that the leading component of a CJK name is the
given name, which is why 65 of 79 gold Korean surnames are predicted
`given_name`.

The generated *values* are already drawn from the right inventories — Korean
`family_name` spans hold real surnames (김, 이, 박, 최) and `given_name` spans
hold real given names (성진, 영환, 정수). So the defect is word order alone,
and relabeling would be actively wrong: it would attach `family_name` to a
value that is a given name, contradicting the inventory signal the model can
otherwise learn. This instead swaps the two surface strings and moves the two
spans with them, leaving every label attached to its own value.

The rewrite is length-preserving over the pair's region, so spans outside that
region keep their offsets exactly. Eligibility is provenance-gated by
``pii_named_entity_materializer.yaml``; target language alone is insufficient.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

CJK = ("ko", "ja", "zh")
GIVEN, FAMILY = "given_name", "family_name"
MATERIALIZER_POLICY_PATH = Path(__file__).resolve().parent / "pii_named_entity_materializer.yaml"


def load_repair_policy(path: Path = MATERIALIZER_POLICY_PATH) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported named-entity materializer schema: {config.get('schema_version')}")
    try:
        policy = config["source_repairs"]["cjk_family_first"]
    except (KeyError, TypeError) as exc:
        raise ValueError("materializer policy lacks source_repairs.cjk_family_first") from exc
    required = {"languages", "source_language_codes", "legacy_sources"}
    missing = required - set(policy)
    if missing:
        raise ValueError(f"CJK repair policy lacks keys: {sorted(missing)}")
    return policy


def requires_family_first_repair(row: dict, policy: dict) -> bool:
    """Return whether target language and recorded source authorize repair."""
    if row.get("lang") not in policy["languages"]:
        return False
    materialization = row.get("materialization") or {}
    source_language_code = (
        materialization.get("source_language_code")
        or row.get("source_language_code")
        or row.get("source_lang_code")
    )
    if source_language_code in policy["source_language_codes"]:
        return True
    return row.get("src") in policy["legacy_sources"]


def reorder_row(row: dict, max_gap: int) -> int:
    """Rewrite adjacent given->family pairs as family->given. Returns pairs fixed."""
    fixed = 0
    while True:
        spans = row.get("spans", [])
        names = sorted((s for s in spans if s[2] in (GIVEN, FAMILY)), key=lambda s: s[0])
        target = None
        for first, second in zip(names, names[1:]):
            if first[2] == GIVEN and second[2] == FAMILY and 0 <= second[0] - first[1] <= max_gap:
                target = (first, second)
                break
        if target is None:
            return fixed
        given, family = target
        text = row["text"]
        separator = text[given[1] : family[0]]
        given_text = text[given[0] : given[1]]
        family_text = text[family[0] : family[1]]
        region_start, region_end = given[0], family[1]
        row["text"] = text[:region_start] + family_text + separator + given_text + text[region_end:]
        family[0], family[1] = region_start, region_start + len(family_text)
        given[0] = family[1] + len(separator)
        given[1] = given[0] + len(given_text)
        # The pair now reads family-then-given, so it cannot rematch above.
        fixed += 1
        if fixed > 64:
            raise AssertionError("runaway name reordering")


def verify_row(row: dict, original_text: str) -> None:
    if len(row["text"]) != len(original_text):
        raise AssertionError("name reordering changed text length")
    for span in row.get("spans", []):
        if not 0 <= span[0] < span[1] <= len(row["text"]):
            raise AssertionError(f"span {span[:2]} escapes text after reordering")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--languages", nargs="+", default=list(CJK))
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--materializer-config", type=Path, default=MATERIALIZER_POLICY_PATH)
    args = parser.parse_args()

    source, output = Path(args.data), Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} already exists; remove it explicitly first")
    languages = set(args.languages)
    repair_policy = load_repair_policy(args.materializer_config)
    output.mkdir(parents=True)

    stats = Counter()
    for name in ("train.jsonl", "val.jsonl"):
        if not (source / name).is_file():
            continue
        with (
            (source / name).open(encoding="utf-8") as reader,
            (output / name).open("w", encoding="utf-8") as writer,
        ):
            for line in reader:
                row = json.loads(line)
                if row.get("lang") in languages and requires_family_first_repair(row, repair_policy):
                    original = row["text"]
                    fixed = reorder_row(row, args.max_gap)
                    row.pop("_moved_ids", None)
                    verify_row(row, original)
                    stats[f"{name}:pairs_reordered"] += fixed
                    stats[f"{name}:rows_touched"] += fixed > 0
                writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stats[f"{name}:rows"] += 1

    for name in ("labels.json", "sampling.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)

    manifest = {
        "schema_version": 1,
        "kind": "cjk_name_order_reorder",
        "source": str(source.resolve()),
        "languages": sorted(languages),
        "max_gap": args.max_gap,
        "materializer_config": str(args.materializer_config),
        "materializer_config_sha256": hashlib.sha256(args.materializer_config.read_bytes()).hexdigest(),
        "counts": dict(stats),
        "note": "surface strings swapped so family precedes given; labels stay with their own values; text length preserved",
    }
    (output / "cjk_name_order.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
