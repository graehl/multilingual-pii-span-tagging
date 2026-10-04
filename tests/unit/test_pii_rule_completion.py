from scripts.pii_rule_completion import (
    DEFAULT_COVERAGE_THRESHOLD,
    PHONE_FAX_RULES,
    complete_records,
    detect_emails,
    detect_fax_numbers,
    detect_luhn_accounts,
    detect_phone_numbers,
    recognized_card_shape,
)


def record(text, spans, row_id="row"):
    return {
        "id": row_id,
        "lang": "en",
        "text": text,
        "spans": spans,
        "_rule_source_locator": {"path": "source.jsonl", "line_1based": 7},
    }


def test_email_inventory_gap_adds_uncovered_span_with_locator():
    rows, report = complete_records(
        [record("Write to ada@example.org.", [])],
        {"person_name"},
        source_id="partial",
    )

    assert rows[0]["spans"] == [[9, 24, "email"]]
    assert rows[0]["rule_completion"]["source"] == {"path": "source.jsonl", "line_1based": 7}
    assert report["rules"]["email"]["activation_reasons"] == [
        "declared_inventory_gap",
        "same_type_coverage_below_threshold",
    ]
    assert report["affected"][0]["additions"][0]["surface"] == "ada@example.org"


def test_email_rule_activates_only_below_half_same_type_coverage():
    text = "a@b.co c@d.co"
    exactly_half = [record(text, [[0, 6, "email"]])]
    unchanged, report = complete_records(exactly_half, {"email"}, source_id="half")
    assert report["coverage_threshold"] == DEFAULT_COVERAGE_THRESHOLD
    assert report["rules"]["email"]["same_type_coverage"] == 0.5
    assert report["rules"]["email"]["active"] is False
    assert unchanged[0]["spans"] == [[0, 6, "email"]]

    below_half = [record("a@b.co c@d.co e@f.co", [[0, 6, "email"]])]
    completed, report = complete_records(below_half, {"email"}, source_id="below")
    assert report["rules"]["email"]["same_type_coverage"] == 1 / 3
    assert report["rules"]["email"]["active"] is True
    assert completed[0]["spans"] == [
        [0, 6, "email"],
        [7, 13, "email"],
        [14, 20, "email"],
    ]


def test_existing_conflicting_span_wins_over_rule():
    text = "Account 4111 1111 1111 1111"
    completed, report = complete_records(
        [record(text, [[8, len(text), "misc_identifier"]])],
        set(),
        source_id="conflict",
    )

    assert completed[0]["spans"] == [[8, len(text), "misc_identifier"]]
    assert report["rules"]["luhn_account"]["conflicting_overlaps"] == 1
    assert report["rules"]["luhn_account"]["added"] == 0


def test_luhn_rule_accepts_card_shapes_but_labels_broad_account_number():
    surfaces = [
        "4222222222222",
        "4111 1111 1111 1111",
        "5555-5555-5555-4444",
        "3782 822463 10005",
    ]
    for surface in surfaces:
        detections = list(detect_luhn_accounts(f"Account {surface}."))
        assert len(detections) == 1
        assert detections[0].label == "account_number"
        assert detections[0].start == 8

    assert recognized_card_shape("4111111111111111111")
    assert not list(detect_luhn_accounts("Bad 4111 1111 1111 1112."))
    assert not list(detect_luhn_accounts("Other issuer 6011111111111117."))
    assert not list(detect_luhn_accounts("Long 4111 1111 1111 1111 9999."))
    assert not list(detect_luhn_accounts("Serial Number: 4111 1111 1111 1111."))
    assert not list(detect_luhn_accounts("Contact Olivia at 4486700083279."))


def test_email_detector_excludes_adjacent_sentence_punctuation():
    detection = list(detect_emails("Contact first.last+tag@example.co.uk, please."))[0]
    assert detection.start == 8
    assert detection.end == 36


def test_audit_only_reports_proposal_without_changing_spans():
    rows, report = complete_records(
        [record("Write to ada@example.org.", [])],
        set(),
        source_id="audit",
        apply=False,
    )

    assert rows[0]["spans"] == []
    assert "rule_completion" not in rows[0]
    assert report["proposed_spans"] == 1
    assert report["proposed_affected_documents"] == 1
    assert report["added_spans"] == 0


def test_contextual_phone_and_fax_candidates_preserve_complete_surfaces():
    text = "Phone: +1 (212) 555-0123 ext. 9; Fax 212-555-0199."

    phone = list(detect_phone_numbers(text))
    fax = list(detect_fax_numbers(text))

    assert [text[item.start : item.end] for item in phone] == ["+1 (212) 555-0123 ext. 9"]
    assert [item.label for item in phone] == ["phone_number"]
    assert [text[item.start : item.end] for item in fax] == ["212-555-0199"]
    assert [item.label for item in fax] == ["fax_number"]


def test_contextual_phone_candidates_cover_non_latin_cues_and_suffix_roles():
    cases = (
        ("電話番号：03-1234-5678", "03-1234-5678"),
        ("팩스: 02-123-4567", None),
        ("212-555-0123 (phone)", "212-555-0123"),
        ("هاتف: +971 4 123 4567", "+971 4 123 4567"),
    )
    for text, expected in cases:
        surfaces = [text[item.start : item.end] for item in detect_phone_numbers(text)]
        assert surfaces == ([] if expected is None else [expected])
    assert ["02-123-4567"] == [
        "팩스: 02-123-4567"[item.start : item.end] for item in detect_fax_numbers("팩스: 02-123-4567")
    ]


def test_suffix_role_does_not_capture_number_from_preceding_field():
    text = "Phone: 217-788-4532 Fax: 217-788-4533"

    assert [text[item.start : item.end] for item in detect_phone_numbers(text)] == ["217-788-4532"]
    assert [text[item.start : item.end] for item in detect_fax_numbers(text)] == ["217-788-4533"]


def test_contextual_number_candidates_reject_dates_ips_long_ids_and_unanchored_context():
    rejected = (
        "phone records dated 2024-01-01",
        "phone: 192.168.1.1",
        "phone: 1234567890123456",
        "The phone is nearby; reference 212-555-0123",
        "Call 212-555-0123",
    )
    for text in rejected:
        assert not list(detect_phone_numbers(text))
        assert not list(detect_fax_numbers(text))


def test_phone_fax_rules_can_be_audited_without_application():
    text = "Phone: 212-555-0123; Fax: 212-555-0199"
    rows, report = complete_records(
        [record(text, [])],
        set(),
        source_id="candidate-audit",
        apply=False,
        rules=PHONE_FAX_RULES,
    )

    assert rows[0]["spans"] == []
    assert report["rules_evaluated"] == ["phone", "fax"]
    assert report["proposed_spans"] == 2
    assert {addition["label"] for addition in report["proposed_affected"][0]["additions"]} == {
        "phone_number",
        "fax_number",
    }


def test_phone_fax_rules_preserve_existing_gold_conflicts():
    text = "Fax: 212-555-0199"
    rows, report = complete_records(
        [record(text, [[5, len(text), "phone_number"]])],
        set(),
        source_id="gold-wins",
        apply=False,
        rules=PHONE_FAX_RULES,
    )

    assert rows[0]["spans"] == [[5, len(text), "phone_number"]]
    assert report["rules"]["fax"]["conflicting_overlaps"] == 1
    assert report["proposed_spans"] == 0
