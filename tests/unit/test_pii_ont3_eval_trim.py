"""The encoder route's edge-punctuation output rule."""

from scripts.pii_ont3_eval import trim_edge_punctuation


def _span(start: int, end: int, label: str) -> dict:
    return {"start": start, "end": end, "label": label}


def test_absorbed_brackets_quotes_and_clause_punctuation_are_dropped() -> None:
    text = "(06512, le 1er janvier 2017. «ЕнергоМашбанку», Box Houlgate»"
    spans = [
        _span(0, 6, "postal_code"),  # "(06512"
        _span(11, 28, "date"),  # "1er janvier 2017."
        _span(29, 46, "organization"),  # "«ЕнергоМашбанку»,"
        _span(47, 60, "organization"),  # "Box Houlgate»"
    ]
    trimmed, changed = trim_edge_punctuation(spans, text)
    assert changed == 4
    assert [text[s["start"] : s["end"]] for s in trimmed] == [
        "06512",
        "1er janvier 2017",
        "ЕнергоМашбанку",
        "Box Houlgate",
    ]


def test_names_keep_a_final_period_and_symbol_types_are_untouched() -> None:
    text = "Acme Inc. wrote to @chefincamicia, see https://x.y/z),"
    spans = [
        _span(0, 9, "organization"),  # "Acme Inc." keeps its period
        _span(19, 34, "username"),  # "@chefincamicia," untouched by the rule
        _span(39, 54, "url"),  # untouched
    ]
    trimmed, changed = trim_edge_punctuation(spans, text)
    assert changed == 0
    assert [text[s["start"] : s["end"]] for s in trimmed] == [
        "Acme Inc.",
        "@chefincamicia,",
        "https://x.y/z),",
    ]


def test_a_span_that_would_vanish_is_kept() -> None:
    text = "see (), then 12."
    spans = [_span(4, 6, "date"), _span(13, 16, "age")]
    trimmed, changed = trim_edge_punctuation(spans, text)
    assert changed == 1
    assert (trimmed[0]["start"], trimmed[0]["end"]) == (4, 6)
    assert text[trimmed[1]["start"] : trimmed[1]["end"]] == "12"
