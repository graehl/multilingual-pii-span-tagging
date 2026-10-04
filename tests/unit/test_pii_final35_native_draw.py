"""Role assignment for the Final35 native draw, including the training role."""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pii_final35_native_draw import (  # noqa: E402
    ROLE_BUCKETS,
    TRAINING_ROLE,
    DrawError,
    draw_role_for_document,
    parse_quotas,
    passes_quality,
    role_for_document,
)


@pytest.mark.parametrize(
    "text",
    [
        "ناقش الطبيب نتائج الفحص مع المريض وشرح له خيارات العلاج المتاحة.",
        "ناقش الطبيب توصيات WHO مع المريض وشرح له خيارات العلاج المتاحة.",
        "הרופא הסביר למטופל את תוצאות הבדיקה ואת אפשרויות הטיפול הקיימות.",
        "डॉक्टर ने मरीज को जांच के परिणाम और उपचार के विकल्प समझाए।",
        "医生向病人解释了检查结果和可以选择的治疗方案。病人认真听取建议，并在家人的陪同下预约了下一次检查。",
    ],
)
def test_uncased_running_text_passes_quality(text: str) -> None:
    assert passes_quality(text, 40, 600) is None


def test_uppercase_banner_is_rejected_but_ordinary_text_is_not() -> None:
    text = "THE DOCTOR EXPLAINED THE TEST RESULTS AND TREATMENT OPTIONS TO THE PATIENT."
    assert passes_quality(text, 40, 600) == "all_caps"
    assert passes_quality(text.capitalize(), 40, 600) is None


def test_whitespace_word_and_sentence_length_guards() -> None:
    assert passes_quality("a" * 61, 40, 600, language="en") == "overlong_token"
    assert passes_quality("医" * 601, 40, 600, language="zh") == "too_long"


@pytest.mark.parametrize(
    "language,text",
    [
        (
            "zh",
            "医生详细解释了检查结果，并建议病人在接下来的几周内按时服药和记录身体状况，"
            "如果出现新的症状或者感到不适，应当及时联系医院并预约下一次检查。",
        ),
        (
            "ja",
            "医師は検査の結果を詳しく説明し、今後数週間は決められた時間に薬を飲んで体調を記録し、"
            "新しい症状が現れたり気分が悪くなったりした場合には病院に連絡するよう患者に勧めました。",
        ),
        (
            "th",
            "แพทย์อธิบายผลการตรวจอย่างละเอียดและแนะนำให้ผู้ป่วยรับประทานยาตามเวลา"
            "พร้อมบันทึกอาการทุกวันและติดต่อโรงพยาบาลหากมีอาการผิดปกติ",
        ),
    ],
)
def test_draw_keeps_unspaced_running_sentences(monkeypatch, language: str, text: str) -> None:
    import pii_final35_native_draw as draw

    # The draw assigns roles and offsets after NFKC normalization.
    text = unicodedata.normalize("NFKC", text)
    assert 60 < len(text) < 600 and not any(char.isspace() for char in text)
    role = role_for_document(hashlib.sha256(text.encode()).hexdigest())[0]
    monkeypatch.setattr(draw, "fineweb_stream", lambda *a, **kw: iter([{"id": "native", "text": text}]))
    monkeypatch.setattr(draw, "sentence_spans", lambda *a: [(0, len(text))])
    monkeypatch.setattr(draw, "segmenter_provenance", lambda *a: {"fixture": True})
    rows, receipt = draw.draw_language(
        language,
        quotas={(role, "ordinary"): 1},
        cues=[],
        splitter=None,
        seed=1,
        shuffle_buffer=1,
        max_documents=1,
        max_sentences_per_document=1,
        minimum_chars=40,
        maximum_chars=600,
        context_chars=1500,
        reference_minimum=3,
        excluded_hashes=set(),
    )
    assert receipt["satisfied"] and len(rows) == 1
    assert rows[0]["text"] == text
    assert rows[0]["source_document_start"] == 0
    assert rows[0]["source_document_end"] == len(text)


def _hash_in_bucket(low: int, high: int) -> str:
    n = 0
    while True:
        digest = hashlib.sha256(str(n).encode()).hexdigest()
        if low <= int(digest[:8], 16) % 100 < high:
            return digest
        n += 1


def test_every_bucket_has_exactly_one_role() -> None:
    for bucket in range(100):
        roles = [role for role, (low, high) in ROLE_BUCKETS.items() if low <= bucket < high]
        assert len(roles) == 1


def test_training_run_maps_only_rotation_documents_to_training() -> None:
    quotas = parse_quotas(["training/ordinary=5"])
    for role, (low, high) in ROLE_BUCKETS.items():
        digest = _hash_in_bucket(low, high)
        draw_role, bucket_role, bucket = draw_role_for_document(digest, quotas)
        assert bucket_role == role == role_for_document(digest)[0]
        assert low <= bucket < high
        if role in ("rotation-1", "rotation-2"):
            assert draw_role == TRAINING_ROLE
        else:
            assert draw_role == role


def test_evaluation_run_keeps_bucket_roles() -> None:
    quotas = parse_quotas(["development/identifier=1", "final/ordinary=2"])
    digest = _hash_in_bucket(*ROLE_BUCKETS["rotation-1"])
    assert draw_role_for_document(digest, quotas)[0] == "rotation-1"


def test_training_quotas_cannot_mix_with_evaluation_roles() -> None:
    with pytest.raises(DrawError):
        parse_quotas(["training/ordinary=5", "rotation-1/ordinary=5"])
    with pytest.raises(DrawError):
        parse_quotas(["audit/ordinary=5"])


def test_draw_keeps_ordinal_before_filtering_and_sampling(monkeypatch):
    import pii_final35_native_draw as draw

    # The first sentence fails the length filter; the second survives after
    # Unicode whitespace trimming. Its ordinal remains 2 rather than 1.
    text = "X.  Médicos atienden a los pacientes.  "
    source = {"id": "original", "text": text}
    role = role_for_document(hashlib.sha256(text.encode()).hexdigest())[0]
    monkeypatch.setattr(draw, "fineweb_stream", lambda *a, **kw: iter([source]))
    monkeypatch.setattr(draw, "sentence_spans", lambda *a: [(0, 2), (2, len(text))])
    monkeypatch.setattr(draw, "segmenter_provenance", lambda *a: {"fixture": True})
    rows, receipt = draw.draw_language(
        "es",
        quotas={(role, "ordinary"): 1},
        cues=[],
        splitter=None,
        seed=1,
        shuffle_buffer=1,
        max_documents=1,
        max_sentences_per_document=1,
        minimum_chars=5,
        maximum_chars=100,
        context_chars=100,
        reference_minimum=3,
        excluded_hashes=set(),
    )
    assert receipt["satisfied"] and len(rows) == 1
    row = rows[0]
    assert row["sentence_ordinal"] == 2
    assert row["source_document_start"] == 4
    assert text[row["source_document_start"] : row["source_document_end"]] == row["text"]
    assert row["text"] == "Médicos atienden a los pacientes."


def test_cli_preserves_full_source_documents_without_changing_draw(monkeypatch, tmp_path) -> None:
    import pii_final35_native_draw as draw

    text = "  Médicos atienden a los pacientes.\n\n  La clínica publica sus horarios.  "
    role = role_for_document(hashlib.sha256(text.encode()).hexdigest())[0]
    source = {"id": "source-one", "text": text}

    class Splitter:
        segmenter_model = "fixture"
        segmenter_model_revision = "fixture"
        segmenter_name = "fixture"
        segmenter_version = "fixture"

        def split(self, texts):
            for value in texts:
                boundary = value.index("\n\n") + 2
                yield [value[:boundary], value[boundary:]]

    monkeypatch.setattr(draw, "fineweb_stream", lambda *a, **kw: iter([source]))
    monkeypatch.setattr(draw, "SaTCharacterSpanSplitter", lambda *a, **kw: Splitter())
    lexicon = tmp_path / "lexicon"
    lexicon.mkdir()
    (lexicon / "es.json").write_text(json.dumps({"lexicon": {key: [] for key in draw.LEXICON_FIELDS}}))
    documents = tmp_path / "documents.jsonl"
    observed = []
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        receipt_path = tmp_path / f"{enabled}.json"
        argv = [
            "--languages",
            "es",
            "--output-dir",
            str(output),
            "--receipt",
            str(receipt_path),
            "--cue-lexicon-dir",
            str(lexicon),
            "--quota",
            f"{role}/ordinary=2",
            "--minimum-chars",
            "5",
            "--context-chars",
            "10",
            "--shuffle-buffer",
            "1",
        ]
        if enabled:
            argv += ["--source-documents", str(documents)]
        assert draw.main(argv) == 0
        observed.append((output / "es.jsonl").read_bytes())
        receipt = json.loads(receipt_path.read_text())
        assert ("source_documents" in receipt) == enabled
        if enabled:
            assert receipt["source_documents"]["rows"] == 1
            assert receipt["source_documents"]["sha256"] == hashlib.sha256(documents.read_bytes()).hexdigest()
    assert observed[0] == observed[1]
    saved = [json.loads(line) for line in documents.read_text().splitlines()]
    assert len(saved) == 1 and saved[0]["text"] == text
    assert saved[0]["id"] == source["id"]
    for row in map(json.loads, observed[1].splitlines()):
        assert saved[0]["text_sha256"] == row["source_document_sha256"]
        assert text[row["source_document_start"] : row["source_document_end"]] == row["text"]
        assert row["annotation_context"] != text
    with pytest.raises(SystemExit) as error:
        draw.main(argv)
    assert error.value.code == 2
    assert documents.read_text() == json.dumps(saved[0], ensure_ascii=False, sort_keys=True) + "\n"


@pytest.mark.parametrize("exclusion_kind", ["source_document_id", "source_id", "raw_hash", "unhashed_text"])
def test_cli_excludes_source_across_normalization(monkeypatch, tmp_path, exclusion_kind) -> None:
    import pii_final35_native_draw as draw

    raw = "Ａ doctor explains the results to the patient."
    normalized = unicodedata.normalize("NFKC", raw)
    role = role_for_document(hashlib.sha256(normalized.encode()).hexdigest())[0]
    prior = {
        "source_document_id": {"source_document_id": "already-evaluated", "text": "A different segment."},
        "source_id": {"source": {"id": "already-evaluated"}, "text": "A different segment."},
        "raw_hash": {"source_document_sha256": hashlib.sha256(raw.encode()).hexdigest()},
        "unhashed_text": {"text": raw},
    }[exclusion_kind]

    class Splitter:
        segmenter_model = segmenter_model_revision = segmenter_name = segmenter_version = "fixture"

        def split(self, texts):
            for value in texts:
                yield [value]

    monkeypatch.setattr(
        draw, "fineweb_stream", lambda *a, **kw: iter([{"id": "already-evaluated", "text": raw}])
    )
    monkeypatch.setattr(draw, "SaTCharacterSpanSplitter", lambda *a, **kw: Splitter())
    lexicon = tmp_path / "lexicon"
    lexicon.mkdir()
    (lexicon / "en.json").write_text(json.dumps({"lexicon": {key: [] for key in draw.LEXICON_FIELDS}}))
    exclusion = tmp_path / "prior.jsonl"
    exclusion.write_text(json.dumps(prior) + "\n")
    for enabled in (False, True):
        output, receipt = tmp_path / str(enabled), tmp_path / f"{enabled}.json"
        argv = [
            "--languages",
            "en",
            "--output-dir",
            str(output),
            "--receipt",
            str(receipt),
            "--cue-lexicon-dir",
            str(lexicon),
            "--quota",
            f"{role}/ordinary=1",
            "--shuffle-buffer",
            "1",
        ]
        if enabled:
            argv += ["--exclude-jsonl", str(exclusion)]
        assert draw.main(argv) == 0
        result = json.loads(receipt.read_text())
        assert result["languages"][0]["rows"] == (0 if enabled else 1)
        assert result["languages"][0]["satisfied"] == (not enabled)
