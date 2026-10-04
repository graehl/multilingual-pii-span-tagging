import json
import random
from pathlib import Path

import pytest
import yaml

from scripts.pii_instantiate_transport import Filler
from scripts.pii_surface.mix_policy import SurfaceMixPolicy
from scripts.pii_surface_pool import (
    SurfacePool,
    build_pool,
    load_surface_adjudications,
    name_components,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_natural_pool_partitions_source_text_before_frequency_aggregation(tmp_path: Path):
    source = tmp_path / "natural.jsonl"
    rows = []
    for index, name in enumerate(["Ada Lovelace", "Ada Lovelace", "Grace Hopper", "山田太郎"]):
        rows.append(
            {
                "id": f"row-{index}",
                "lang": "zh" if index == 3 else "en",
                "text": name,
                "spans": [{"start": 0, "end": len(name), "label": "person_name"}],
            }
        )
    write_jsonl(source, rows)

    entries, report = build_pool([("gold", [source])], holdout_modulus=10, holdout_fold=0)

    assert sum(report["rows_by_split"].values()) == 4
    assert report["coverage"]["train_languages"]
    assert report["train_audit_value_overlap"]
    assert {entry["split"] for entry in entries} <= {"train", "audit"}
    ada = [entry for entry in entries if entry["tag"] == "PERSON_NAME" and entry["value"] == "Ada Lovelace"]
    assert len(ada) == 1
    assert sum(entry["count"] for entry in ada) == 2
    assert all(entry["source_counts"] == {"gold": entry["count"]} for entry in ada)
    assert all(entry["source_record_hashes"] for entry in ada)


def test_pool_tag_scope_is_explicit_and_rejects_unknown_tags(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    text = "Ada Lovelace joined Acme."
    rows = [
        {
            "id": f"row-{index}",
            "lang": "en",
            "text": text,
            "spans": [
                [0, len("Ada Lovelace"), "person_name"],
                [text.index("Acme"), text.index("Acme") + len("Acme"), "organization"],
            ],
        }
        for index in range(2)
    ]
    write_jsonl(source, rows)

    entries, report = build_pool(
        [("teacher", [source])],
        tags={"PERSON_NAME"},
        carrier_role="train-only",
        holdout_modulus=10,
        holdout_fold=0,
    )

    assert {entry["tag"] for entry in entries} == {"PERSON_NAME"}
    assert len({entry["split"] for entry in entries}) == 1
    assert report["configuration"]["tags"] == ["PERSON_NAME"]
    assert report["configuration"]["carrier_role"] == "train-only"
    assert report["rejected"]["spans_outside_tag_scope"] == 2
    with pytest.raises(ValueError, match="unsupported pool tags: NOT_A_TAG"):
        build_pool([("teacher", [source])], tags={"NOT_A_TAG"})
    with pytest.raises(ValueError, match="unsupported carrier role: neither"):
        build_pool([("teacher", [source])], carrier_role="neither")


def test_pool_version_is_explicit_and_binds_entry_identity(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "row-1",
                "lang": "en",
                "text": "Ada Lovelace",
                "spans": [[0, 12, "person_name"]],
            }
        ],
    )

    first, first_report = build_pool([("teacher", [source])], pool_version="teacher-v1")
    second, _ = build_pool([("teacher", [source])], pool_version="teacher-v2")

    assert first_report["pool_version"] == "teacher-v1"
    assert {row["pool_version"] for row in first} == {"teacher-v1"}
    assert {row["entry_id"] for row in first}.isdisjoint(row["entry_id"] for row in second)


def test_pool_accepts_an_explicit_source_manifest(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    manifest = tmp_path / "teacher-manifest.json"
    write_jsonl(
        source,
        [
            {
                "id": "row-1",
                "lang": "en",
                "text": "Ada Lovelace",
                "spans": [[0, 12, "person_name"]],
            }
        ],
    )
    manifest.write_text(
        json.dumps({"dataset": "teacher", "upstream": {"revision": "abc123"}}),
        encoding="utf-8",
    )

    _, report = build_pool(
        [("teacher", [source])],
        source_manifests_by_dataset={"teacher": manifest},
        require_source_manifests=True,
    )

    assert report["missing_source_manifests"] == []
    assert report["source_manifests"]["teacher"]["upstream"] == {"revision": "abc123"}


def test_pool_can_preserve_the_exact_observed_surface(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    text = "前：ＡＣＭＥ\t会社。後"
    write_jsonl(
        source,
        [
            {
                "id": "row-1",
                "lang": "ja",
                "text": text,
                "spans": [[2, 9, "organization"]],
            }
        ],
    )

    entries, report = build_pool(
        [("teacher", [source])],
        surface_normalization="preserve",
    )

    assert entries[0]["value"] == "ＡＣＭＥ\t会社"
    assert report["configuration"]["surface_normalization"] == "preserve"


def test_name_components_back_off_when_boundary_is_not_reliable():
    assert name_components("en", "Ada Lovelace") == ("Ada", "Lovelace")
    assert name_components("zh", "欧阳娜娜") == ("娜娜", "欧阳")
    assert name_components("vi", "Nguyễn Văn An") == ("An", "Nguyễn")
    assert name_components("vi", "Jose Mourinho") is None
    assert name_components("ar", "أحمد بن علي") is None
    assert name_components("ja", "山田太郎") is None
    assert name_components("en", "J. Smith") is None


def test_pool_can_retain_only_direct_name_component_labels(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    text = "Ada Lovelace met Grace"
    write_jsonl(
        source,
        [
            {
                "id": "row-1",
                "lang": "en",
                "text": text,
                "spans": [
                    [0, len("Ada Lovelace"), "person_name"],
                    [text.index("Grace"), len(text), "given_name"],
                ],
            }
        ],
    )

    entries, report = build_pool(
        [("teacher", [source])],
        name_component_mode="direct-only",
    )

    assert {(entry["tag"], entry["value"]) for entry in entries} == {
        ("GIVEN_NAME", "Grace"),
        ("PERSON_NAME", "Ada Lovelace"),
    }
    assert report["configuration"]["name_component_mode"] == "direct-only"
    assert report["semantics"]["name_components"] == (
        "Full person names and directly labeled given/family components are retained. Generic person-name "
        "spans never produce derived components."
    )

    with pytest.raises(ValueError, match="unsupported name-component mode: inferred"):
        build_pool([("teacher", [source])], name_component_mode="inferred")


def test_pool_excludes_exact_adjudicated_source_span(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    adjudication_path = tmp_path / "adjudication.jsonl"
    text = "Google Analytics is made by Google"
    product_end = len("Google Analytics")
    company_start = text.rindex("Google")
    write_jsonl(
        source,
        [
            {
                "id": "row-1",
                "lang": "en",
                "text": text,
                "spans": [
                    [0, product_end, "organization"],
                    [company_start, len(text), "organization"],
                ],
            }
        ],
    )
    write_jsonl(
        adjudication_path,
        [
            {
                "dataset": "teacher",
                "id": "row-1",
                "start": 0,
                "end": product_end,
                "label": "organization",
                "value": "Google Analytics",
                "action": "exclude",
                "reason": "product_as_organization",
            }
        ],
    )
    adjudications, receipt = load_surface_adjudications(adjudication_path)

    entries, report = build_pool(
        [("teacher", [source])],
        surface_adjudications=adjudications,
        surface_adjudication_receipt=receipt,
    )

    assert {(entry["tag"], entry["value"]) for entry in entries} == {
        ("ORGANIZATION", "Google"),
    }
    assert report["rejected"]["adjudicated_surface_spans"] == 1
    assert report["surface_adjudication"] == {
        "configured": 1,
        "matched": 1,
        "excluded_by_reason": {"product_as_organization": 1},
    }
    assert report["configuration"]["surface_adjudication"] == receipt


def test_pool_rejects_unmatched_or_duplicate_surface_adjudication(tmp_path: Path):
    source = tmp_path / "teacher.jsonl"
    adjudication_path = tmp_path / "adjudication.jsonl"
    write_jsonl(
        source,
        [{"id": "row-1", "lang": "en", "text": "Google", "spans": [[0, 6, "organization"]]}],
    )
    decision = {
        "dataset": "teacher",
        "id": "row-1",
        "start": 0,
        "end": 6,
        "label": "organization",
        "value": "Alphabet",
        "action": "exclude",
        "reason": "wrong_surface",
    }
    write_jsonl(adjudication_path, [decision])
    adjudications, _ = load_surface_adjudications(adjudication_path)

    with pytest.raises(ValueError, match="surface adjudications did not match"):
        build_pool([("teacher", [source])], surface_adjudications=adjudications)

    write_jsonl(adjudication_path, [decision, {**decision, "reason": "second_reason"}])
    with pytest.raises(ValueError, match="duplicate adjudicated span"):
        load_surface_adjudications(adjudication_path)


def test_surface_pool_sampling_preserves_weights_and_provenance(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "test-v1",
                "entry_id": "ada",
                "split": "train",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": "Ada",
                "count": 100,
            },
            {
                "pool_version": "test-v1",
                "entry_id": "grace",
                "split": "train",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": "Grace",
                "count": 1,
            },
            {
                "pool_version": "test-v1",
                "entry_id": "audit-only",
                "split": "audit",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": "Heldout",
                "count": 1000,
            },
        ],
    )
    pool = SurfacePool.load(pool_path)
    values = [pool.draw("en", "GIVEN_NAME", random.Random(seed)) for seed in range(40)]

    assert all(value is not None and value[0] != "Heldout" for value in values)
    assert sum(value[0] == "Ada" for value in values if value is not None) >= 35
    assert all(value[1].startswith("natural-pool:test-v1:") for value in values if value is not None)
    assert pool.distinct_count("en", "GIVEN_NAME") == 2
    assert pool.distinct_count("de", "GIVEN_NAME") == 0

    tempered_values = [
        pool.draw("en", "GIVEN_NAME", random.Random(seed), count_temperature=0.0) for seed in range(400)
    ]
    tempered_ada = sum(value[0] == "Ada" for value in tempered_values if value is not None)
    assert 160 < tempered_ada < 240


def test_surface_pool_exposes_reweighted_count_receipt(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    receipt = {
        "basis": "exact_occurrences",
        "authentic_count": 9,
        "unseen_pseudocount": 1,
        "sampling_count": 10,
    }
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "frequency-v1",
                "entry_id": "ada",
                "split": "train",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": "Ada",
                "count": 10,
                "source_observed_count": 3,
                "sampling_count_receipt": receipt,
            }
        ],
    )
    pool = SurfacePool.load(pool_path)
    draw = pool.draw("en", "GIVEN_NAME", random.Random(1))
    assert draw is not None
    _, provenance = draw

    assert pool.provenance_metadata(provenance) == [
        {
            "pool_version": "frequency-v1",
            "entry_id": "ada",
            "value_normalized": "",
            "observed_count": 3,
            "sampling_count": 10,
            "sampling_count_receipt": receipt,
            "source_counts": {},
            "source_record_hashes": [],
        }
    ]


def test_surface_pool_rejects_malformed_reweighting_receipts(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "frequency-v1",
                "entry_id": "ada",
                "split": "train",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": "Ada",
                "count": 10,
                "source_observed_count": "3",
            }
        ],
    )

    try:
        SurfacePool.load(pool_path)
    except ValueError as error:
        assert "source_observed_count" in str(error)
    else:
        raise AssertionError("string source count should fail")


def test_filler_uses_natural_pool_only_when_explicitly_enabled(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "test-v1",
                "entry_id": "given",
                "split": "train",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": "NaturalGiven",
                "count": 1,
            },
            {
                "pool_version": "test-v1",
                "entry_id": "family",
                "split": "train",
                "lang": "en",
                "tag": "FAMILY_NAME",
                "value": "NaturalFamily",
                "count": 1,
            },
        ],
    )
    pool = SurfacePool.load(pool_path)
    filler = Filler("en", 7, pool, natural_surface_rate=1.0, natural_min_distinct_full_rate=1)

    values = {filler.fill("GIVEN_NAME", "old")[0] for _ in range(20)}
    families = {filler.fill("FAMILY_NAME", "old")[0] for _ in range(20)}

    assert "NaturalGiven" in values
    assert "NaturalFamily" in families
    assert Filler("en", 7).fill("GIVEN_NAME", "old")[1].startswith("name:native:faker:")


def test_filler_can_force_fresh_faker_or_exact_empirical_branch(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "test-v1",
                "entry_id": "given",
                "split": "train",
                "lang": "vi",
                "tag": "GIVEN_NAME",
                "value": "Ngọc",
                "count": 1,
            }
        ],
    )
    filler = Filler("vi", 7, SurfacePool.load(pool_path))

    empirical = filler.fill("GIVEN_NAME", "old", surface_route="exact-empirical")
    fresh_faker = filler.fill("GIVEN_NAME", "old", surface_route="fresh-faker")

    assert empirical == ("Ngọc", "selective-exact-empirical:natural-pool:test-v1:given")
    assert fresh_faker[0] in {"John", "Jane"}
    assert fresh_faker[1].startswith("name:native:faker:")

    with pytest.raises(ValueError, match="fresh Faker route is unavailable"):
        filler.fill("AGE", "31", surface_route="fresh-faker")

    filler.begin_document(
        {
            "[ADDRESS_1]": {"tag": "ADDRESS"},
            "[CITY_2]": {"tag": "CITY"},
        },
        seed=11,
    )
    fresh_address = filler.fill("ADDRESS", "old address", surface_route="fresh-faker")
    assert fresh_address[0] != "old address"
    assert fresh_address[1].startswith("faker:")


def test_filler_preserves_component_provenance_for_composed_person_name(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "test-v1",
                "entry_id": entry_id,
                "split": "train",
                "lang": "en",
                "tag": tag,
                "value": value,
                "count": 1,
            }
            for entry_id, tag, value in (
                ("given", "GIVEN_NAME", "NaturalGiven"),
                ("family", "FAMILY_NAME", "NaturalFamily"),
            )
        ],
    )
    filler = Filler(
        "en",
        1,
        SurfacePool.load(pool_path),
        natural_surface_rate=1.0,
        natural_min_distinct_full_rate=1,
    )

    value, provenance = filler.fill("PERSON_NAME", "old")

    assert value == "NaturalGiven NaturalFamily"
    assert "natural-pool:test-v1:given" in provenance
    assert "natural-pool:test-v1:family" in provenance


def test_filler_scales_natural_rate_for_thin_groups(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "test-v1",
                "entry_id": f"given-{index}",
                "split": "train",
                "lang": "en",
                "tag": "GIVEN_NAME",
                "value": f"Natural{index}",
                "count": 1,
            }
            for index in range(10)
        ],
    )
    filler = Filler("en", 9, SurfacePool.load(pool_path), natural_surface_rate=0.5)

    provenances = [filler.fill("GIVEN_NAME", "old")[1] for _ in range(2000)]
    natural_fraction = sum("natural-pool:" in value for value in provenances) / len(provenances)

    # 10 distinct values / 100-value full-rate threshold scales the 0.5 cap to 0.05.
    assert 0.03 < natural_fraction < 0.07


def test_source_fallback_draws_empirical_values_without_faker_backoff(tmp_path: Path):
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(
        pool_path,
        [
            {
                "pool_version": "test-v1",
                "entry_id": "organization",
                "split": "train",
                "lang": "vi",
                "tag": "ORGANIZATION",
                "value": "Đại học Quốc gia Hà Nội",
                "count": 1,
            }
        ],
    )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "recipe_version": "empirical-first-v1",
                "defaults": {
                    "natural_surface_rate": 1.0,
                    "natural_min_distinct_full_rate": 1,
                    "non_empirical_fallback": "source",
                },
            }
        ),
        encoding="utf-8",
    )
    filler = Filler(
        "vi",
        11,
        surface_pool=SurfacePool.load(pool_path),
        surface_policy=SurfaceMixPolicy.load(recipe_path),
    )

    assert filler.fill("ORGANIZATION", "Old Org")[0] == "Đại học Quốc gia Hà Nội"
    assert filler.fill("STREET_ADDRESS", "12 Đường Huế") == (
        "12 Đường Huế",
        "orig:surface-policy-source-v1",
    )


def test_pool_binds_onboarding_manifest_and_exposes_draw_lineage(tmp_path: Path):
    source_root = tmp_path / "onboarded"
    train = source_root / "train"
    train.mkdir(parents=True)
    source = train / "en.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "row-1",
                "lang": "en",
                "text": "Ada Lovelace",
                "spans": [[0, 12, "person_name"]],
            }
        ],
    )
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "example",
                "upstream": {
                    "revision": "abc123",
                    "license": "CC0-1.0",
                    "url": "https://example.invalid/data",
                },
            }
        ),
        encoding="utf-8",
    )

    entries, report = build_pool(
        [("natural-example", [source])],
        holdout_modulus=2,
        holdout_fold=1,
        require_source_manifests=True,
    )

    assert report["missing_source_manifests"] == []
    assert report["source_manifests"]["natural-example"]["upstream"]["license"] == "CC0-1.0"
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(pool_path, entries)
    pool = SurfacePool.load(pool_path, split=entries[0]["split"])
    draw = pool.draw("en", "GIVEN_NAME", random.Random(1))
    assert draw is not None
    value, provenance = draw
    metadata = pool.provenance_metadata(provenance)
    assert value == "Ada"
    assert metadata[0]["source_counts"] == {"natural-example": 1}
    assert len(metadata[0]["source_record_hashes"]) == 1
