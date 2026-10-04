"""Needle validators, paragraph selection and cursor resumption."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import pii_needle_select as select  # noqa: E402
from scripts.pii_final35_native_draw import (  # noqa: E402
    ROLE_BUCKETS,
    TRAINING_SOURCE_ROLES,
    role_for_document,
)
from scripts.pii_needle_validators import VALIDATORS  # noqa: E402
from scripts.pii_ont3_surface_retrieval import sha256_text  # noqa: E402

NEEDLES = ROOT / "data/pii-needles/rare-type-needles-v1.json"


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        ("iban", "GB82 WEST 1234 5698 7654 32"),
        ("payment_card", "4111 1111 1111 1111"),
        ("imei", "490154203237518"),
        ("pesel", "44051401359"),
        ("tckn", "10000000146"),
        ("chinese_resident_id", "11010519491231002X"),
        ("israeli_id", "123456782"),
        ("snils", "112-233-445 95"),
        ("ipv4", "192.168.1.20"),
        ("ipv6", "2001:db8::1:0:0:1"),
        ("latitude_longitude", "50.0875, 14.4213"),
    ],
)
def test_published_valid_examples_pass(validator, value):
    assert VALIDATORS[validator](value)


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        ("iban", "GB82 WEST 1234 5698 7654 33"),
        ("payment_card", "4111 1111 1111 1112"),
        ("pesel", "44051401358"),
        ("tckn", "10000000147"),
        ("ipv4", "300.1.1.1"),
        ("latitude_longitude", "95.1234, 14.4213"),
    ],
)
def test_corrupted_values_fail(validator, value):
    assert not VALIDATORS[validator](value)


@pytest.mark.parametrize(
    ("validator", "prefix"),
    [("aadhaar", "23412341234"), ("korean_rrn", "800101123456"), ("rodne_cislo", "780101123")],
)
def test_exactly_one_check_digit_completes_a_number(validator, prefix):
    assert sum(VALIDATORS[validator](prefix + str(digit)) for digit in range(10)) == 1


def test_paragraph_selection_threshold_cap_and_cues():
    needles = select.load_needles(NEEDLES)["needles"]
    strong = "Platbu pošlete na účet CZ65 0800 0000 1920 0014 5399, variabilní symbol 2024."
    weak = "Pište na info@example.cz nebo volejte +420 777 123 456."
    assert select.score(select.needle_hits(strong, "cs", needles), 2) >= 3
    assert select.score(select.needle_hits(weak, "cs", needles), 2) < 3
    many_ips = " ".join(f"10.0.0.{n}" for n in range(1, 9))
    assert select.score(select.needle_hits(many_ips, "en", needles), 2) == 4
    # A cue-gated needle needs its cue: a bare 9-digit number is not an Israeli ID.
    assert not select.needle_hits("המספר 123456782 בלבד", "he", needles)
    assert select.needle_hits("ת.ז 123456782", "he", needles)[0]["needle"] == "israeli_id"


def training_document(text):
    """Return text whose NFKC hash falls in a training rotation bucket."""
    for n in range(200):
        candidate = f"{text}\nPoznámka {n}."
        if role_for_document(sha256_text(candidate))[0] in TRAINING_SOURCE_ROLES:
            return candidate
    raise AssertionError("no training-bucket variant")


def test_cli_scans_a_range_and_resumes_from_the_cursor(tmp_path):
    iban_line = "Číslo účtu: CZ65 0800 0000 1920 0014 5399, majitel Jan Novák."
    documents = [
        {"id": f"d{n}", "text": training_document(iban_line if n % 2 == 0 else "Obyčejný text bez čísel.")}
        for n in range(4)
    ]
    local = tmp_path / "cs.jsonl"
    local.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents), encoding="utf-8")
    cursor = tmp_path / "cursor.json"
    common = [
        "--languages",
        "cs",
        "--needles",
        str(NEEDLES),
        "--output-dir",
        str(tmp_path / "out"),
        "--scan",
        "2",
        "--cursor",
        str(cursor),
        "--local-jsonl",
        f"cs={local}",
        "--acli-quiet",
    ]
    ledger = tmp_path / "used.jsonl"
    common += ["--used-ledger", str(ledger)]
    assert select.main([*common, "--receipt", str(tmp_path / "r1.json")]) == 0
    first = json.loads((tmp_path / "r1.json").read_text())["languages"][0]
    assert first["range"] == {"first": {"line": 0}, "next": {"line": 2}}
    assert first["paragraphs_selected"] == 1
    assert json.loads(cursor.read_text())["languages"]["cs"]["next"] == {"line": 2}
    assert select.main([*common, "--receipt", str(tmp_path / "r2.json")]) == 0
    second = json.loads((tmp_path / "r2.json").read_text())["languages"][0]
    assert second["range"] == {"first": {"line": 2}, "next": None}
    # The same IBAN paragraph text recurs in document d2, so the ledger skips it.
    assert second["paragraphs_selected"] == 0
    assert second["counts"]["previously_used_paragraph"] == 1
    [entry] = [json.loads(line) for line in ledger.open()]
    assert entry["source_locator"]["line"] == 0 and entry["source_document_id"] == "d0"
    rows = [json.loads(line) for line in Path(first["paragraphs"]["path"]).open()]
    assert rows[0]["source_locator"]["kind"] == "local_jsonl" and rows[0]["draw_role"] == "training"
    assert rows[0]["needle_hits"][0]["type"] == "bank_account_number"
    # A new run without the ledger may select it again; with it, the whole range is spent.
    with pytest.raises(SystemExit):
        select.main([*common, "--receipt", str(tmp_path / "r3.json")])


def test_numeric_ids_need_letter_free_boundaries_but_allow_cjk_neighbours():
    needles = select.load_needles(NEEDLES)["needles"]
    assert not select.needle_hits("fond ISIN CZ0008011806 zanikl", "cs", needles)
    assert (
        select.needle_hits("身份证号11010519491231002X，电话", "zh", needles)[0]["needle"]
        == "chinese_resident_id"
    )


def test_preceding_context_window_and_seek_offsets():
    needles = select.load_needles(NEEDLES)["needles"]
    text = "První věta.\nDruhá věta jiného odstavce.\nÚčet CZ65 0800 0000 1920 0014 5399 patří Janovi."
    rows, _ = select.select_language(
        "cs",
        [({"line": 0}, None, {"id": "x", "text": text})],
        identity={"kind": "local_jsonl", "path": "x.jsonl", "sha256": "0"},
        scan=1,
        needles=needles,
        threshold=3,
        max_hits_per_needle=2,
        paragraph_mode="line",
        min_chars=10,
        max_chars=500,
        max_paragraphs_per_document=2,
        roles=ROLE_BUCKETS,
        context_chars=30,
    )
    [row] = rows
    assert row["preceding_context"] == text[row["preceding_context_start"] : row["source_document_start"]]
    assert row["source_document_start"] - row["preceding_context_start"] == 30
    assert text[row["first_hit_document_offset"] :].startswith("CZ65")


def test_imported_draw_rows_mark_their_spans_used():
    needles = select.load_needles(NEEDLES)["needles"]
    text = "Úvod.\nÚčet CZ65 0800 0000 1920 0014 5399 patří Janovi."
    start = text.index("Účet")
    used = select.UsedRegions()
    used.add_row(
        {"source_document_id": "x", "source_document_start": start, "source_document_end": start + 10}
    )
    rows, summary = select.select_language(
        "cs",
        [({"line": 0}, None, {"id": "x", "text": text})],
        identity={"kind": "local_jsonl", "path": "x.jsonl", "sha256": "0"},
        used=used,
        scan=1,
        needles=needles,
        threshold=3,
        max_hits_per_needle=2,
        paragraph_mode="line",
        min_chars=10,
        max_chars=500,
        max_paragraphs_per_document=2,
        roles=ROLE_BUCKETS,
    )
    assert rows == [] and summary["counts"]["previously_used_paragraph"] == 1


def role_document(text, role):
    """Return text whose NFKC hash falls in a bucket of the given role."""
    for n in range(500):
        candidate = f"{text}\nPoznámka {n}."
        if role_for_document(sha256_text(candidate))[0] == role:
            return candidate
    raise AssertionError(f"no {role} variant")


def test_roles_restrict_selection_to_held_out_buckets():
    needles = select.load_needles(NEEDLES)["needles"]
    line = "Účet CZ65 0800 0000 1920 0014 5399 patří Janovi."
    documents = [
        ({"line": n}, None, {"id": role, "text": role_document(line + f" {n}", role)})
        for n, role in enumerate(("rotation-1", "development", "final"))
    ]
    rows, summary = select.select_language(
        "cs",
        documents,
        identity={"kind": "local_jsonl", "path": "x.jsonl", "sha256": "0"},
        scan=3,
        needles=needles,
        threshold=3,
        max_hits_per_needle=2,
        paragraph_mode="line",
        min_chars=10,
        max_chars=500,
        max_paragraphs_per_document=2,
        roles=("development",),
    )
    assert [row["draw_role"] for row in rows] == ["development"]
    assert summary["counts"]["unselected_bucket_role"] == 2


class LineSplitter:
    """Stub splitter: one sentence per '. '-terminated piece."""

    segmenter_model = segmenter_model_revision = segmenter_name = segmenter_version = "stub"

    def split(self, texts):
        for text in texts:
            pieces = [piece + ". " for piece in text.split(". ")]
            pieces[-1] = pieces[-1][:-2]
            yield [piece for piece in pieces if piece]


def test_previous_sentence_outside_the_paragraph_is_located_and_marked_used():
    text = "Úvodní věta. Kontext před odstavcem.\nÚčet CZ65 0800 0000 1920 0014 5399. Druhá věta."
    start = text.index("Účet")
    paragraph = {
        "id": "p",
        "lang": "cs",
        "text": text[start:],
        "needle_hits": [],
        "preceding_context": text[:start],
        "preceding_context_start": 0,
        "source_document_start": start,
        **{
            key: None
            for key in (
                "source_locator",
                "source_document_id",
                "source_document_sha256",
                "draw_role",
                "bucket_role",
                "partition_bucket",
                "draw_stratum",
                "source",
                "supervision_scope",
            )
        },
    }
    first, second = select.sentence_rows([paragraph], LineSplitter(), {})
    a, b = first["previous_sentence_document_span"]
    assert text[a:b] == first["previous_sentence"] == "Kontext před odstavcem."
    a, b = second["previous_sentence_document_span"]
    assert text[a:b] == second["previous_sentence"] == first["text"]
    used = select.UsedRegions()
    used.add_row({"source_document_id": "x", "context_spans": [first["previous_sentence_document_span"]]})
    assert used.overlaps("x", 0, start - 1) and not used.overlaps("x", 0, 5)


def test_locator_reloads_the_verified_document(tmp_path):
    text = training_document("Účet CZ65 0800 0000 1920 0014 5399 patří Janovi.")
    local = tmp_path / "cs.jsonl"
    local.write_text(
        json.dumps({"id": "a", "text": "x"}) + "\n" + json.dumps({"id": "b", "text": text}) + "\n"
    )
    identity = select.stream_identity("cs", "train", local)
    [(locator, _, _), (second, _, _)] = list(select.read_documents(identity, None))
    record = {**identity, **second, "document_sha256_nfkc": sha256_text(text)}
    assert select.load_document(record) == text
    with pytest.raises(select.SelectError):
        select.load_document({**record, "document_sha256_nfkc": "0" * 64})


def test_types_option_restricts_scoring_needles(tmp_path):
    text = training_document("Kontakt: info@example.cz, účet CZ65 0800 0000 1920 0014 5399.")
    local = tmp_path / "cs.jsonl"
    local.write_text(json.dumps({"id": "a", "text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    base = [
        "--languages",
        "cs",
        "--needles",
        str(NEEDLES),
        "--scan",
        "1",
        "--local-jsonl",
        f"cs={local}",
        "--threshold",
        "1",
        "--acli-quiet",
    ]
    assert (
        select.main(
            [
                *base,
                "--output-dir",
                str(tmp_path / "a"),
                "--receipt",
                str(tmp_path / "a.json"),
                "--types",
                "email",
            ]
        )
        == 0
    )
    receipt = json.loads((tmp_path / "a.json").read_text())
    assert receipt["types"] == ["email"]
    assert receipt["languages"][0]["needle_hits_in_selected"] == {"email": 1}
    with pytest.raises(SystemExit):
        select.main(
            [
                *base,
                "--output-dir",
                str(tmp_path / "b"),
                "--receipt",
                str(tmp_path / "b.json"),
                "--types",
                "no_such_type",
            ]
        )
