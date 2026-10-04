import json

from scripts.pii_transport_semantic_guard import (
    analyze_guard,
    colon_field_key_tag_literals,
    currency_amounts_inside_spans,
    currency_amounts_outside_spans,
    literal_tag_values_outside_spans,
    semantic_guard_reasons,
)


def test_canonical_tag_literal_outside_spans_is_rejected():
    text = "The employee identifier is employee_id."

    assert literal_tag_values_outside_spans(text, []) == ["employee_id"]
    assert semantic_guard_reasons(text, []) == ["literal_tag_outside_span:employee_id"]


def test_canonical_tag_literal_inside_span_is_allowed():
    text = "The literal employee_id appeared in a quoted value."
    start = text.index("employee_id")

    assert (
        literal_tag_values_outside_spans(text, [[start, start + len("employee_id"), "misc_identifier"]]) == []
    )


def test_canonical_tag_literal_used_as_a_field_key_is_allowed():
    text = '{"employee_id": "AB123", device_id: "XY456"}'

    assert literal_tag_values_outside_spans(text, []) == []
    assert colon_field_key_tag_literals(text) == ["employee_id", "device_id"]


def test_noncanonical_tag_like_token_is_outside_guard_scope():
    text = "The response object contains auth_token."

    assert literal_tag_values_outside_spans(text, []) == []
    assert semantic_guard_reasons(text, []) == []


def test_currency_amount_outside_spans_is_rejected():
    text = "The sale price is $450,000 and the fee is 1250 EUR."

    assert currency_amounts_outside_spans(text, []) == ["$450,000", "1250 EUR"]


def test_currency_amount_number_inside_span_is_allowed_with_separate_designator():
    text = "The sale price is $450,000."
    amount_start = text.index("450,000")

    assert (
        currency_amounts_outside_spans(
            text,
            [
                [amount_start - 1, amount_start, "currency_designator"],
                [amount_start, amount_start + len("450,000"), "monetary_amount"],
            ],
        )
        == []
    )
    assert currency_amounts_inside_spans(
        text,
        [
            [amount_start - 1, amount_start, "currency_designator"],
            [amount_start, amount_start + len("450,000"), "monetary_amount"],
        ],
    ) == ["$450,000"]


def test_coverage_accepts_legacy_spans_with_extra_fields():
    text = "The amount is $450,000."
    amount_start = text.index("450,000")

    assert (
        currency_amounts_outside_spans(
            text,
            [[amount_start, amount_start + len("450,000"), "monetary_amount", "legacy-extra"]],
        )
        == []
    )
    assert (
        currency_amounts_outside_spans(
            text,
            [{"start": amount_start, "end": amount_start + len("450,000"), "type": "monetary_amount"}],
        )
        == []
    )


def test_ordinary_numbers_are_not_treated_as_currency_amounts():
    assert currency_amounts_outside_spans("The building has 12 floors and opened in 2023.", []) == []


def test_analysis_distinguishes_supported_and_known_unsupported_rejections(tmp_path):
    input_path = tmp_path / "input.jsonl"
    adjudication_path = tmp_path / "adjudication.jsonl"
    rows = [
        {"audit_id": 1, "id": "a", "lang": "en", "text": "ID employee_id", "spans": []},
        {"audit_id": 2, "id": "b", "lang": "en", "text": "Price $50", "spans": []},
        {"audit_id": 3, "id": "c", "lang": "en", "text": "No finding", "spans": []},
    ]
    adjudications = [
        {"audit_id": 1, "adjudication": "supported"},
        {"audit_id": 2, "adjudication": "unsupported"},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    adjudication_path.write_text(
        "".join(json.dumps(row) + "\n" for row in adjudications),
        encoding="utf-8",
    )

    report = analyze_guard(input_path, adjudication_path)

    assert report["retained_documents"] == 1
    assert report["rejected_documents"] == 2
    assert report["adjudication"]["supported_rejected"] == [1]
    assert report["adjudication"]["known_unsupported_rejected"] == [2]
