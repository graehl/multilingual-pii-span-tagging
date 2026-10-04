import pytest

from scripts.pii_cjk_name_order import (
    load_repair_policy,
    reorder_row,
    requires_family_first_repair,
    verify_row,
)


def test_korean_pair_becomes_family_first_with_labels_on_their_values() -> None:
    row = {
        "lang": "ko",
        "text": "환자 성진 김 님께",
        "spans": [[3, 5, "given_name"], [6, 7, "family_name"]],
    }
    original = row["text"]
    assert reorder_row(row, 2) == 1
    verify_row(row, original)
    assert row["text"] == "환자 김 성진 님께"
    family = next(s for s in row["spans"] if s[2] == "family_name")
    given = next(s for s in row["spans"] if s[2] == "given_name")
    assert row["text"][family[0] : family[1]] == "김"
    assert row["text"][given[0] : given[1]] == "성진"
    assert family[0] < given[0]


def test_unrelated_spans_keep_their_offsets() -> None:
    row = {
        "lang": "ko",
        "text": "성진 김 010-1234",
        "spans": [[0, 2, "given_name"], [3, 4, "family_name"], [5, 13, "phone_number"]],
    }
    reorder_row(row, 2)
    phone = next(s for s in row["spans"] if s[2] == "phone_number")
    assert row["text"][phone[0] : phone[1]] == "010-1234"


def test_already_family_first_is_untouched() -> None:
    row = {
        "lang": "ko",
        "text": "김 성진",
        "spans": [[0, 1, "family_name"], [2, 4, "given_name"]],
    }
    before = row["text"]
    assert reorder_row(row, 2) == 0
    assert row["text"] == before


def test_distant_pair_is_not_reordered() -> None:
    row = {
        "lang": "ko",
        "text": "성진 이 사람의 성은 김 입니다",
        "spans": [[0, 2, "given_name"], [12, 13, "family_name"]],
    }
    assert reorder_row(row, 2) == 0


def test_verify_rejects_length_change() -> None:
    row = {"text": "ab", "spans": [[0, 1, "given_name"]]}
    with pytest.raises(AssertionError, match="changed text length"):
        verify_row(row, "abc")


def test_repair_requires_defective_source_provenance() -> None:
    policy = load_repair_policy()
    assert requires_family_first_repair(
        {"lang": "ko", "materialization": {"source_language_code": "en"}}, policy
    )
    assert requires_family_first_repair({"lang": "zh", "src": "ai4p-1.5m-extra-langs"}, policy)
    assert not requires_family_first_repair({"lang": "ko", "src": "native-korean-gold"}, policy)
    assert not requires_family_first_repair(
        {"lang": "ja", "materialization": {"source_language_code": "ja"}}, policy
    )
    assert not requires_family_first_repair(
        {"lang": "de", "materialization": {"source_language_code": "en"}}, policy
    )
