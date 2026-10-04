import gzip
import json
from pathlib import Path

from scripts.pii_surface_audit import (
    audit,
    mask_carrier,
    normalize_value,
    origin_family,
    resolve_input,
    row_language,
    row_source,
    script_profile,
    shape_signature,
)


def write_jsonl(path: Path, rows: list[dict], *, compressed: bool = False) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_surface_audit_preserves_duplicate_values_and_generator_provenance(tmp_path: Path):
    path = tmp_path / "ar.jsonl"
    rows = [
        {
            "id": f"row-{index}",
            "lang": "ar",
            "text": "المريض Jason رقمه A-123",
            "spans": [[7, 12, "given_name"], [18, 23, "medical_record_number"]],
            "span_provenance": [
                {"start": 7, "end": 12, "generator": "name:latin-kept"},
                {"start": 18, "end": 23, "generator": "bespoke:org-person-id-v1"},
            ],
        }
        for index in range(2)
    ]
    write_jsonl(path, rows)

    result = audit([("transport", [path])], top_k=2)
    names = next(group for group in result["surface_groups"] if group["label"] == "given_name")

    assert names["origin"] == "name:latin-kept"
    assert names["count"] == 2
    assert names["distinct_normalized"] == 1
    assert names["duplicate_occurrences"] == 1
    assert names["scripts"] == {"Latin": 2}
    assert result["diagnostics"]["unused_provenance_records"] == 0
    assert "never rejected" in result["semantics"]["duplicate_values"]


def test_surface_audit_compacts_origins_and_compares_masked_carriers(tmp_path: Path):
    faker_path = tmp_path / "faker.jsonl"
    natural_path = tmp_path / "natural.jsonl"
    common = {
        "lang": "de",
        "spans": [[6, 11, "given_name"]],
    }
    write_jsonl(
        faker_path,
        [
            {
                **common,
                "text": "Hallo Jason!",
                "span_provenance": [{"start": 6, "end": 11, "generator": "name:native:faker:de"}],
            }
        ],
    )
    write_jsonl(
        natural_path,
        [
            {
                **common,
                "text": "Hallo Erika!",
                "span_provenance": [
                    {
                        "start": 6,
                        "end": 11,
                        "generator": "name:native:natural-pool:v1:entry-7",
                    }
                ],
            }
        ],
    )

    result = audit([("faker", [faker_path]), ("natural", [natural_path])])

    assert {
        (group["dataset"], group["origin_family"], group["count"])
        for group in result["surface_aggregate_groups"]
    } == {("faker", "name-native-faker", 1), ("natural", "natural-pool", 1)}
    assert result["carrier_pair_groups"] == [
        {
            "language": "de",
            "left_dataset": "faker",
            "right_dataset": "natural",
            "left_count": 1,
            "right_count": 1,
            "left_distinct": 1,
            "right_distinct": 1,
            "distinct_intersection": 1,
            "distinct_union": 1,
            "distinct_jaccard": 1.0,
            "multiset_intersection": 1,
            "multiset_union": 1,
            "exact_multiset_equal": True,
        }
    ]


def test_origin_family_retains_source_and_generator_contracts():
    assert origin_family("generator", "natural-pool:v1:id") == "natural-pool"
    assert origin_family("generator", "name:native:natural-pool:v1:id") == "natural-pool"
    assert origin_family("generator", "name:native:faker:de") == "name-native-faker"
    assert origin_family("generator", "faker:de") == "faker"
    assert origin_family("row_source", "mapa") == "row-source:mapa"


def test_surface_audit_reads_dict_spans_and_gzip(tmp_path: Path):
    path = tmp_path / "natural.jsonl.gz"
    write_jsonl(
        path,
        [
            {
                "lang": "hi",
                "text": "नाम आशा",
                "spans": [{"start": 4, "end": 7, "label": "person_name"}],
                "metadata": {"source": "hiner"},
            }
        ],
        compressed=True,
    )

    result = audit([("natural", [path])])

    assert result["totals"] == {"rows": 1, "spans": 1, "datasets": 1}
    assert result["surface_groups"][0]["origin"] == "hiner"
    assert result["surface_groups"][0]["scripts"] == {"Devanagari": 1}


def test_surface_audit_accepts_eval_meta_language_and_type_spans(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "eval-1",
                "meta": {"lang": "zh", "src": "zh-targeted-final20"},
                "text": "姓名王小明",
                "spans": [{"start": 2, "end": 5, "type": "person_name"}],
            }
        ],
    )

    result = audit([("eval", [path])])

    assert result["totals"] == {"rows": 1, "spans": 1, "datasets": 1}
    assert result["surface_groups"][0]["language"] == "zh"
    assert result["surface_groups"][0]["label"] == "person_name"
    assert result["surface_groups"][0]["origin"] == "zh-targeted-final20"
    row = {"meta": {"lang": "zh", "mix_source": "primary"}}
    assert row_language(row) == "zh"
    assert row_source(row, "fallback") == "primary"


def test_carrier_masking_ignores_entity_value_equality():
    carrier, overlaps = mask_carrier(
        "Patient Jason has ID A-123.",
        [
            type("SpanLike", (), {"start": 8, "end": 13, "label": "given_name"})(),
            type("SpanLike", (), {"start": 21, "end": 26, "label": "medical_record_number"})(),
        ],
    )

    assert carrier == "patient <given_name> has id <medical_record_number>."
    assert overlaps == 0


def test_unicode_profiles_and_shapes_are_report_only():
    assert normalize_value("  JASON\u3000Smith ") == "jason smith"
    assert script_profile("West أفراح") == "Arabic+Latin"
    assert script_profile("1985-01-02") == "NoLetters"
    assert script_profile("山田太郎") == "Han"
    assert shape_signature("A-123") == "L-D{3}"


def test_resolve_input_accepts_directories_and_quoted_globs(tmp_path: Path):
    write_jsonl(tmp_path / "a.jsonl", [])
    write_jsonl(tmp_path / "b.jsonl.gz", [], compressed=True)
    (tmp_path / "ignored.txt").write_text("x")

    name, directory_paths = resolve_input(f"all={tmp_path}")
    _, glob_paths = resolve_input(f"plain={tmp_path}/*.jsonl")

    assert name == "all"
    assert [path.name for path in directory_paths] == ["a.jsonl", "b.jsonl.gz"]
    assert [path.name for path in glob_paths] == ["a.jsonl"]
