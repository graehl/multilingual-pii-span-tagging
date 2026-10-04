"""The encoder route's maximal-span output contract for names and organizations."""

from scripts.pii_ont3_eval import merge_adjacent_same_type

TYPES = frozenset({"person_name", "organization"})


def _span(start: int, end: int, label: str) -> dict:
    return {"start": start, "end": end, "label": label}


def test_adjacent_name_components_merge_into_one_span() -> None:
    text = "Robert McQuinn met Jessie Willcox Smith."
    spans = [
        _span(0, 6, "person_name"),
        _span(7, 14, "person_name"),
        _span(19, 25, "person_name"),
        _span(26, 39, "person_name"),
    ]
    merged, merges = merge_adjacent_same_type(spans, text, TYPES)
    assert merges == 2
    assert [(s["start"], s["end"], s["label"]) for s in merged] == [
        (0, 14, "person_name"),
        (19, 39, "person_name"),
    ]
    assert text[0:14] == "Robert McQuinn" and text[19:39] == "Jessie Willcox Smith"


def test_only_whitespace_gaps_and_same_type_merge() -> None:
    text = "Smith, Jones and Acme Corp Ltd"
    spans = [
        _span(0, 5, "person_name"),
        _span(7, 12, "person_name"),  # comma between: stays separate
        _span(17, 21, "organization"),
        _span(22, 26, "organization"),  # whitespace: merges
        _span(27, 30, "person_name"),  # different type from the previous: stays separate
    ]
    merged, merges = merge_adjacent_same_type(spans, text, TYPES)
    assert merges == 1
    assert [(s["start"], s["end"], s["label"]) for s in merged] == [
        (0, 5, "person_name"),
        (7, 12, "person_name"),
        (17, 26, "organization"),
        (27, 30, "person_name"),
    ]


def test_types_outside_the_contract_and_overlaps_are_left_alone() -> None:
    text = "12 May 2020"
    spans = [_span(0, 2, "date"), _span(3, 6, "date"), _span(3, 11, "date")]
    merged, merges = merge_adjacent_same_type(spans, text, TYPES)
    assert merges == 0
    assert len(merged) == 3
    merged_dates, merges_dates = merge_adjacent_same_type(spans, text, frozenset({"date"}))
    # the overlapping third span does not start after the previous end, so it is kept as is
    assert merges_dates == 1
    assert [(s["start"], s["end"]) for s in merged_dates] == [(0, 6), (3, 11)]
