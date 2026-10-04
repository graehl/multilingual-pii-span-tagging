import gzip
import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from scripts.pii_onboard_sources import (
    OPENNER_COMMERCIAL_CORE_COMPONENTS,
    SOURCES,
    ShardWriter,
    SourceArtifact,
    SourceComponent,
    aqmar_rows,
    bio_spans,
    disambiguate_record_id,
    hiner_rows,
    idner_rows,
    klue_rows,
    normalized_spans,
    openner_core_rows,
    openpii_rows,
    source_schema_sha256,
    validate_source_artifacts,
    wojood_rows,
)
from scripts.pii_projector import Tagset

HEALTH_PHI_SAMPLE = "ai4privacy-health-phi-400k-sample-1k"
HEALTH_PHI_LABELS = {
    "BLOODTYPE": "blood_type",
    "DIAGNOSES": "medical_condition",
    "DISABILITYSTATUS": "health_attribute",
    "GENETICINFO": "health_attribute",
    "HEALTHINSURANCENUM": "insurance_id",
    "IMMUNIZATIONSTATUS": "health_attribute",
    "MEDICALRECORDNUM": "medical_record_number",
    "MENTALHEALTHINFO": "health_attribute",
    "PREGNANCYSTATUS": "health_attribute",
    "PRESCRIPTIONINFO": "health_attribute",
    "TESTRESULTS": "health_attribute",
    "TREATMENTINFO": "health_attribute",
}


def test_pinned_sources_have_expected_canonical_maps() -> None:
    tagset = Tagset()
    assert tagset.sources[SOURCES["nemotron-pii"].source_schema]["medical_record_number"] == (
        "medical_record_number"
    )
    assert tagset.sources[SOURCES["openpii-1m"].source_schema]["SURNAME"] == "family_name"
    assert tagset.sources[SOURCES["mapa"].source_schema]["PERSON"] == "person_name"
    assert tagset.sources[SOURCES["mapa"].source_schema]["ADDRESS"] == "location"
    assert tagset.sources[SOURCES["mapa"].source_schema]["AMOUNT"] == "quantity"
    assert tagset.sources[SOURCES["idner-news-2k"].source_schema] == {
        "LOC": "location",
        "ORG": "organization",
        "PER": "person_name",
    }
    assert tagset.sources[SOURCES["hiner"].source_schema] == {
        "LANGUAGE": "language_spoken",
        "LOCATION": "location",
        "ORGANIZATION": "organization",
        "PERSON": "person_name",
        "RELIGION": "religious_belief",
    }
    assert SOURCES["hiner"].ignored_labels == (
        "FESTIVAL",
        "GAME",
        "LITERATURE",
        "MISC",
        "NUMEX",
        "TIMEX",
    )
    assert tagset.sources[SOURCES["wojood-sample"].source_schema] == {
        "PERS": "person_name",
        "ORG": "organization",
        "GPE": "location",
        "LOC": "location",
        "FAC": "location",
        "OCC": "occupation",
        "LANGUAGE": "language_spoken",
        "WEBSITE": "url",
        "DATE": "date",
        "TIME": "time",
        "MONEY": "monetary_amount",
    }
    assert SOURCES["wojood-sample"].ignored_labels == (
        "CARDINAL",
        "CURR",
        "EVENT",
        "LAW",
        "NORP",
        "ORDINAL",
        "PERCENT",
        "PRODUCT",
        "QUANTITY",
        "UNIT",
    )
    assert tagset.sources[SOURCES["aqmar-openner"].source_schema] == {
        "LOC": "location",
        "ORG": "organization",
        "PER": "person_name",
    }
    assert SOURCES["aqmar-openner"].required_citations == (
        "Mohit et al. (2012), Recall-Oriented Learning of Named Entities in Arabic Wikipedia",
    )
    assert tagset.sources[SOURCES["openner-commercial-core"].source_schema] == {
        "LOC": "location",
        "ORG": "organization",
        "PER": "person_name",
    }
    assert {component.lang for component in OPENNER_COMMERCIAL_CORE_COMPONENTS} == {
        "de",
        "en",
        "es",
        "ja",
        "pt",
        "sv",
        "zh",
    }


def test_health_phi_projection_extends_openpii_without_changing_it() -> None:
    tagset = Tagset()
    openpii_map = tagset.sources[SOURCES["openpii-1m"].source_schema]
    health_map = tagset.sources[SOURCES[HEALTH_PHI_SAMPLE].source_schema]
    assert {label: health_map[label] for label in openpii_map} == openpii_map
    assert source_schema_sha256("ai4privacy_new") == (
        "10842f18f137fecf0db9fdeb4eee86e9764a360ee123273235093e73c55d4ee9"
    )
    assert set(openpii_map).isdisjoint(HEALTH_PHI_LABELS)
    assert {label: health_map[label] for label in HEALTH_PHI_LABELS} == HEALTH_PHI_LABELS


def test_health_phi_sample_pins_exact_local_input() -> None:
    spec = SOURCES[HEALTH_PHI_SAMPLE]
    assert spec.fetch_backend == "local"
    assert spec.source_artifacts == (
        SourceArtifact(
            path="data/train.jsonl",
            bytes=3_863_416,
            sha256="fa851ac72f1dfb67218d5b5bd5958663526969b66b0427fddf7aba420f3ad9ce",
        ),
    )


def test_source_artifact_validation_accepts_exact_file_and_rejects_drift(tmp_path) -> None:
    payload = b'{"split":"train"}\n'
    data = tmp_path / "data"
    data.mkdir()
    path = data / "train.jsonl"
    path.write_bytes(payload)
    spec = replace(
        SOURCES[HEALTH_PHI_SAMPLE],
        source_artifacts=(
            SourceArtifact(
                path="data/train.jsonl",
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
        ),
    )

    assert validate_source_artifacts(spec, tmp_path) == [
        {
            "path": "data/train.jsonl",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]

    path.write_bytes(b"drift\n")
    with pytest.raises(ValueError, match="byte size"):
        validate_source_artifacts(spec, tmp_path)

    path.write_bytes(b'{"split":"traiN"}\n')
    with pytest.raises(ValueError, match="SHA-256"):
        validate_source_artifacts(spec, tmp_path)


def test_source_artifact_validation_rejects_path_escape(tmp_path) -> None:
    spec = replace(
        SOURCES[HEALTH_PHI_SAMPLE],
        source_artifacts=(SourceArtifact(path="../train.jsonl", bytes=0, sha256="0" * 64),),
    )
    with pytest.raises(ValueError, match="relative to the source root"):
        validate_source_artifacts(spec, tmp_path)


def test_health_phi_rows_preserve_source_stratum_and_count_literal_placeholders(tmp_path) -> None:
    text = "A [GIVENNAMEPLACEHOLDER_1]"
    raw = {
        "source_text": text,
        "masked_text": "[MEDICALRECORDNUM_1] [GIVENNAMEPLACEHOLDER_1]",
        "privacy_mask": [
            {
                "start": 0,
                "end": 1,
                "label": "MEDICALRECORDNUM",
                "label_index": 1,
                "value": "A",
            }
        ],
        "split": "train",
        "uid": 7,
        "language": "en",
        "region": "US",
        "script": "Latn",
        "mbert_tokens": [],
        "mbert_token_classes": [],
        "source_dataset": "base",
    }
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    anomalies = Counter()

    rows = list(
        openpii_rows(
            tmp_path,
            Tagset().sources[SOURCES[HEALTH_PHI_SAMPLE].source_schema],
            anomalies,
            metadata_fields=("region", "script", "source_dataset"),
            count_source_placeholders=True,
        )
    )

    assert rows == [
        (
            "train",
            {
                "id": "7",
                "text": text,
                "spans": [
                    {
                        "start": 0,
                        "end": 1,
                        "label": "medical_record_number",
                        "source_label": "MEDICALRECORDNUM",
                    }
                ],
                "lang": "en",
                "metadata": {"region": "US", "script": "Latn", "source_dataset": "base"},
            },
        )
    ]
    assert anomalies == {"source_literal_placeholder": 1}


def test_normalized_spans_retain_source_and_canonical_labels() -> None:
    text = "Patient MRN-2048"
    spans = normalized_spans(
        text,
        json.dumps([{"start": 8, "end": 16, "text": "MRN-2048", "label": "medical_record_number"}]),
        {"medical_record_number": "medical_record_number"},
    )
    assert spans == [
        {
            "start": 8,
            "end": 16,
            "label": "medical_record_number",
            "source_label": "medical_record_number",
        }
    ]


def test_normalized_spans_reject_offset_text_mismatch() -> None:
    with pytest.raises(ValueError, match="span text mismatch"):
        normalized_spans(
            "John",
            [{"start": 0, "end": 4, "value": "Jane", "label": "GIVENNAME"}],
            {"GIVENNAME": "given_name"},
        )


def test_normalized_spans_accept_casefolded_source_value_and_count_it() -> None:
    anomalies = Counter()
    spans = normalized_spans(
        "Black",
        [{"start": 0, "end": 5, "text": "black", "label": "eye_color"}],
        {"eye_color": "eye_color"},
        anomalies,
    )
    assert spans[0]["label"] == "eye_color"
    assert anomalies == {"casefold_source_value": 1}


def test_duplicate_upstream_ids_are_retained_and_reported() -> None:
    occurrences = Counter()
    output_ids = set()
    anomalies = Counter()
    first = {"id": "same", "metadata": {}}
    second = {"id": "same", "metadata": {}}
    disambiguate_record_id(first, occurrences, output_ids, anomalies)
    disambiguate_record_id(second, occurrences, output_ids, anomalies)
    assert first == {"id": "same", "metadata": {}}
    assert second == {"id": "same~2", "metadata": {"upstream_id": "same"}}
    assert anomalies == {"duplicate_upstream_id": 1}


def test_shard_writer_hashes_the_fully_closed_file(tmp_path) -> None:
    writer = ShardWriter(tmp_path, "train", "en")
    writer.write({"id": "1", "text": "x", "spans": [], "lang": "en", "metadata": {}})
    metadata = writer.close()
    path = tmp_path / metadata["path"]
    assert metadata["bytes"] == path.stat().st_size
    with gzip.open(path, "rt", encoding="utf-8") as source:
        assert json.loads(source.readline())["text"] == "x"


def test_mapa_bio_spans_use_reconstructed_character_offsets() -> None:
    text, spans = bio_spans(
        ["Dr.", "Jane", "Doe", ",", "Prague"],
        ["O", "B-PERSON", "I-PERSON", "O", "B-ADDRESS"],
        {"PERSON": "person_name", "ADDRESS": "address"},
    )
    assert text == "Dr. Jane Doe , Prague"
    assert spans == [
        {
            "start": 4,
            "end": 12,
            "label": "person_name",
            "source_label": "PERSON",
        },
        {
            "start": 15,
            "end": 21,
            "label": "address",
            "source_label": "ADDRESS",
        },
    ]


def test_mapa_bio_spans_reject_invalid_continuation() -> None:
    with pytest.raises(ValueError, match="invalid MAPA continuation"):
        bio_spans(["Jane"], ["I-PERSON"], {"PERSON": "person_name"})


def test_idner_rows_preserve_upstream_splits_and_reconstruct_offsets(tmp_path) -> None:
    fixture = (
        "Presiden PROPN O\r\n"
        "Joko PROPN B-PER\r\n"
        "Widodo PROPN I-PER\r\n"
        "di ADP O\r\n"
        "Jakarta PROPN B-LOC\r\n"
        "\r\n"
    )
    for name in ("train.txt", "dev.txt", "test.txt"):
        (tmp_path / name).write_bytes(fixture.encode())

    rows = list(
        idner_rows(
            tmp_path,
            {"PER": "person_name", "LOC": "location", "ORG": "organization"},
            Counter(),
        )
    )

    assert [split for split, _ in rows] == ["train", "validation", "test"]
    assert rows[0][1] == {
        "id": "id:train:1",
        "text": "Presiden Joko Widodo di Jakarta",
        "spans": [
            {
                "start": 9,
                "end": 20,
                "label": "person_name",
                "source_label": "PER",
            },
            {
                "start": 24,
                "end": 31,
                "label": "location",
                "source_label": "LOC",
            },
        ],
        "lang": "id",
        "metadata": {
            "sentence_number": 1,
            "tokens": ["Presiden", "Joko", "Widodo", "di", "Jakarta"],
            "pos_tags": ["PROPN", "PROPN", "PROPN", "ADP", "PROPN"],
            "text_reconstruction": "single_space_join",
        },
    }


def test_hiner_rows_preserve_splits_and_ignore_nonprivacy_entities(tmp_path) -> None:
    fixture = "भारत\tB-LOCATION\nमें\tO\nदीवाली\tB-FESTIVAL\nमनाई\tO\nगई\tO\n\n"
    original = tmp_path / "data" / "original"
    original.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        (original / f"{split}.conll").write_text(fixture, encoding="utf-8")

    rows = list(
        hiner_rows(
            tmp_path,
            {"LOCATION": "location"},
            Counter(),
            ("FESTIVAL",),
        )
    )

    assert [split for split, _ in rows] == ["train", "validation", "test"]
    assert rows[0][1] == {
        "id": "hi:train:1",
        "text": "भारत में दीवाली मनाई गई",
        "spans": [
            {
                "start": 0,
                "end": 4,
                "label": "location",
                "source_label": "LOCATION",
            }
        ],
        "lang": "hi",
        "metadata": {
            "sentence_number": 1,
            "tokens": ["भारत", "में", "दीवाली", "मनाई", "गई"],
            "bio_tags": ["B-LOCATION", "O", "B-FESTIVAL", "O", "O"],
            "text_reconstruction": "single_space_join",
        },
    }


KLUE_TRAIN_FIXTURE = (
    "## 주석 : 무시되는 파일 머리말\n"
    "## klue-ner-v1_train_00000_wikitree\t<김철수:PS>는 <서울:LC>에서 <어제:DT> 만났다.\n"
    "김\tB-PS\n철\tI-PS\n수\tI-PS\n는\tO\n \tO\n"
    "서\tB-LC\n울\tI-LC\n에\tO\n서\tO\n \tO\n"
    "어\tB-DT\n제\tI-DT\n \tO\n만\tO\n났\tO\n다\tO\n.\tO\n\n"
)
KLUE_DEV_FIXTURE = (
    "## klue-ner-v1_dev_00000_wikinews\t<한국은행:OG>이 발표했다.\n"
    "한\tB-OG\n국\tI-OG\n은\tI-OG\n행\tI-OG\n이\tO\n \tO\n"
    "발\tO\n표\tO\n했\tO\n다\tO\n.\tO\n\n"
)


def _write_klue_fixture(tmp_path, train: str, dev: str) -> None:
    data_root = tmp_path / "klue_benchmark" / "klue-ner-v1.1"
    data_root.mkdir(parents=True)
    (data_root / "klue-ner-v1.1_train.tsv").write_text(train, encoding="utf-8")
    (data_root / "klue-ner-v1.1_dev.tsv").write_text(dev, encoding="utf-8")


def test_klue_pinned_projection() -> None:
    tagset = Tagset()
    assert tagset.sources[SOURCES["klue-ner"].source_schema] == {
        "PS": "person_name",
        "LC": "location",
        "OG": "organization",
    }
    assert SOURCES["klue-ner"].ignored_labels == ("DT", "TI", "QT")
    assert SOURCES["klue-ner"].required_citations


def test_klue_rows_reconstruct_exact_text_and_ignore_relative_time(tmp_path) -> None:
    _write_klue_fixture(tmp_path, KLUE_TRAIN_FIXTURE, KLUE_DEV_FIXTURE)
    source_map = {"PS": "person_name", "LC": "location", "OG": "organization"}

    rows = list(klue_rows(tmp_path, source_map, Counter(), ("DT",)))

    assert [split for split, _ in rows] == ["train", "validation"]
    assert rows[0][1] == {
        "id": "klue-ner-v1_train_00000_wikitree",
        "text": "김철수는 서울에서 어제 만났다.",
        "spans": [
            {"start": 0, "end": 3, "label": "person_name", "source_label": "PS"},
            {"start": 5, "end": 7, "label": "location", "source_label": "LC"},
        ],
        "lang": "ko",
        "metadata": {
            "sentence_number": 1,
            "source_document": "wikitree",
            "text_reconstruction": "character_concatenation",
        },
    }
    assert rows[1][1]["spans"] == [{"start": 0, "end": 4, "label": "organization", "source_label": "OG"}]


def test_klue_rows_keep_char_track_and_count_header_markup_disagreement(tmp_path) -> None:
    broken_train = KLUE_TRAIN_FIXTURE.replace("울\tI-LC", "울\tO")
    _write_klue_fixture(tmp_path, broken_train, KLUE_DEV_FIXTURE)
    source_map = {"PS": "person_name", "LC": "location", "OG": "organization"}
    anomalies = Counter()

    rows = list(klue_rows(tmp_path, source_map, anomalies, ("DT",)))

    assert anomalies == {"header_markup_mismatch": 1}
    train_row = rows[0][1]
    assert train_row["metadata"]["header_markup_mismatch"] is True
    assert {"start": 5, "end": 6, "label": "location", "source_label": "LC"} in train_row["spans"]


def test_klue_rows_keep_char_track_and_count_header_text_disagreement(tmp_path) -> None:
    broken_train = KLUE_TRAIN_FIXTURE.replace("났\tO", "닜\tO")
    _write_klue_fixture(tmp_path, broken_train, KLUE_DEV_FIXTURE)
    source_map = {"PS": "person_name", "LC": "location", "OG": "organization"}
    anomalies = Counter()

    rows = list(klue_rows(tmp_path, source_map, anomalies, ("DT",)))

    assert anomalies == {"header_text_mismatch": 1}
    train_row = rows[0][1]
    assert train_row["metadata"]["header_text_mismatch"] is True
    assert "닜" in train_row["text"]


def test_aqmar_rows_require_corrected_bio_and_reconstruct_offsets(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "ner_tags": {
                            "feature": {"names": labels, "_type": "ClassLabel"},
                            "_type": "List",
                        }
                    }
                }
            }
        ).encode()
    }
    table = pa.table(
        {
            "id": [7],
            "tokens": [["زار", "أحمد", "جامعة", "القاهرة"]],
            "ner_tags": [[0, 5, 3, 4]],
        }
    ).replace_schema_metadata(metadata)
    root = tmp_path / "AQMAR" / "ara"
    root.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        pq.write_table(table, root / f"{split}-00000-of-00001.parquet")

    rows = list(
        aqmar_rows(
            tmp_path,
            {"PER": "person_name", "ORG": "organization", "LOC": "location"},
            Counter(),
        )
    )

    assert [split for split, _ in rows] == ["train", "validation", "test"]
    assert rows[0][1]["text"] == "زار أحمد جامعة القاهرة"
    assert rows[0][1]["spans"] == [
        {
            "start": 4,
            "end": 8,
            "label": "person_name",
            "source_label": "PER",
        },
        {
            "start": 9,
            "end": 22,
            "label": "organization",
            "source_label": "ORG",
        },
    ]
    assert rows[0][1]["metadata"]["bio_provenance"] == ("Liu_et_al_2019_corrected_via_OpenNER_1.0")


def test_openner_core_rows_preserve_component_and_strict_bio(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "ner_tags": {
                            "feature": {"names": labels, "_type": "ClassLabel"},
                            "_type": "List",
                        }
                    }
                }
            }
        ).encode()
    }
    table = pa.table(
        {
            "id": [11],
            "tokens": [["Ada", "visits", "Berlin"]],
            "ner_tags": [[5, 0, 1]],
        }
    ).replace_schema_metadata(metadata)
    component = SourceComponent(
        name="fixture",
        relative_root="Fixture/eng",
        lang="en",
        license="CC-BY-4.0",
        url="https://example.invalid/fixture",
        attribution="Fixture authors",
    )
    root = tmp_path / component.relative_root
    root.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        pq.write_table(table, root / f"{split}-00000-of-00001.parquet")

    rows = list(
        openner_core_rows(
            tmp_path,
            {"PER": "person_name", "ORG": "organization", "LOC": "location"},
            Counter(),
            (component,),
        )
    )

    assert [split for split, _ in rows] == ["train", "validation", "test"]
    assert rows[0][1]["id"] == "en:openner:fixture-eng:train:11"
    assert rows[0][1]["text"] == "Ada visits Berlin"
    assert rows[0][1]["metadata"]["source_component"] == "fixture"
    assert rows[0][1]["spans"] == [
        {"start": 0, "end": 3, "label": "person_name", "source_label": "PER"},
        {"start": 11, "end": 17, "label": "location", "source_label": "LOC"},
    ]


def test_wojood_rows_preserve_nested_spans_and_official_same_type_rule(tmp_path) -> None:
    fixture = "جامعة B-ORG\nبيرزيت I-ORG B-ORG B-GPE\nالحدث B-EVENT\nالمدرس I-OCC\n\n"
    data = tmp_path / "data"
    data.mkdir()
    for split_name in ("train", "val", "test"):
        (data / f"{split_name}.txt").write_text(fixture, encoding="utf-8")

    anomalies = Counter()
    rows = list(
        wojood_rows(
            tmp_path,
            {"ORG": "organization", "GPE": "location", "OCC": "occupation"},
            anomalies,
            ("EVENT",),
        )
    )

    assert [split for split, _ in rows] == ["train", "validation", "test"]
    assert anomalies == {
        "same_type_nested_tag_dropped": 3,
        "orphan_i_repaired": 3,
    }
    assert rows[0][1]["text"] == "جامعة بيرزيت الحدث المدرس"
    assert rows[0][1]["spans"] == [
        {
            "start": 0,
            "end": 12,
            "label": "organization",
            "source_label": "ORG",
        },
        {
            "start": 6,
            "end": 12,
            "label": "location",
            "source_label": "GPE",
        },
        {
            "start": 19,
            "end": 25,
            "label": "occupation",
            "source_label": "OCC",
        },
    ]
    assert rows[0][1]["metadata"]["same_type_projection"] == "first_tag_per_official_nested_loader"
