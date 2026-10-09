from collections import Counter

import pytest

from scripts.pii_gliner2_projection import project_row
from scripts.pii_projector import Tagset


class MinimalTokenizer:
    """One-subword tokenizer sufficient to exercise the installed processor."""

    def __init__(self):
        self.ids = {}

    def add_special_tokens(self, spec):
        for token in spec["additional_special_tokens"]:
            self.convert_tokens_to_ids(token)

    def tokenize(self, token):
        return [token]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        if tokens not in self.ids:
            self.ids[tokens] = len(self.ids) + 1
        return self.ids[tokens]


@pytest.fixture
def processor():
    from gliner2.processor import SchemaTransformer

    return SchemaTransformer(tokenizer=MinimalTokenizer())


def project(processor, text, spans, inventory=("city", "given_name", "person_name"), **kwargs):
    row = {"id": "test", "lang": "en", "text": text, "spans": spans}
    return project_row(
        processor,
        row,
        inventory,
        negative_label_count=kwargs.pop("negative_label_count", 0),
        **kwargs,
    )


def test_repeated_same_label_mentions_roundtrip(processor):
    output, diagnostic = project(
        processor,
        "Ada met Ada.",
        [[0, 3, "given_name"], [8, 11, "given_name"]],
    )
    assert diagnostic.accepted
    assert output["output"]["entities"]["given_name"] == ["Ada"]


@pytest.mark.parametrize(
    ("spans", "reasons"),
    [
        ([[0, 6, "given_name"]], {"roundtrip_extra"}),
        (
            [[0, 6, "given_name"], [11, 17, "city"]],
            {"roundtrip_extra"},
        ),
    ],
)
def test_ambiguous_duplicate_surface_is_rejected(processor, spans, reasons):
    _, diagnostic = project(processor, "Jordan met Jordan.", spans)
    assert not diagnostic.accepted
    assert set(diagnostic.reasons) == reasons


def test_token_substring_is_not_invented(processor):
    _, accepted = project(processor, "Ann met Anna.", [[0, 3, "given_name"]])
    assert accepted.accepted

    _, rejected = project(processor, "John arrived.", [[0, 2, "given_name"]])
    assert not rejected.accepted
    assert rejected.reasons == ("roundtrip_missing",)


def test_punctuation_email_and_unicode_case_roundtrip(processor):
    text = "Éva met ÉVA; email x@example.test."
    email_start = text.index("x@example.test")
    spans = [
        [0, 3, "given_name"],
        [8, 11, "given_name"],
        [email_start, email_start + len("x@example.test"), "email"],
    ]
    _, diagnostic = project(
        processor,
        text,
        spans,
        inventory=("email", "given_name"),
    )
    assert diagnostic.accepted


def test_nested_distinct_surfaces_roundtrip(processor):
    text = "Ada Lovelace"
    spans = [[0, 12, "person_name"], [4, 12, "family_name"]]
    _, diagnostic = project(
        processor,
        text,
        spans,
        inventory=("family_name", "person_name"),
    )
    assert diagnostic.accepted


def test_no_entity_row_has_explicit_negative_schema(processor):
    output, diagnostic = project(
        processor,
        "Nothing private here.",
        [],
        negative_label_count=2,
    )
    assert diagnostic.accepted
    assert len(output["output"]["entities"]) == 2
    assert all(value == [] for value in output["output"]["entities"].values())


def test_span_wider_than_model_capacity_is_rejected(processor):
    text = "one two three four five six seven eight nine"
    _, diagnostic = project(
        processor,
        text,
        [[0, len(text), "person_name"]],
        max_span_width=8,
    )
    assert not diagnostic.accepted
    assert diagnostic.reasons == ("span_too_wide",)
    assert diagnostic.max_observed_span_width == 9


def test_every_canonical_label_can_be_represented(processor):
    inventory = sorted(Tagset().nodes)
    observed = Counter()
    for label in inventory:
        output, diagnostic = project(
            processor,
            "value",
            [[0, 5, label]],
            inventory=inventory,
        )
        assert diagnostic.accepted, (label, diagnostic)
        observed.update(output["output"]["entities"].keys())
    assert set(observed) == set(inventory)


def test_redaction_9_projection_covers_native_unmapped_fine_labels(processor):
    tagset = Tagset()
    output, diagnostic = project(
        processor,
        "Ada joined ACME.",
        [[0, 3, "person_name"], [11, 15, "organization"]],
        inventory=sorted(tagset.cut_targets("redaction_9_v1")),
        output_schema="redaction_9_v1",
    )
    assert diagnostic.accepted
    assert not diagnostic.unmapped
    assert output["output"]["entities"] == {"name": ["Ada"], "organization": ["ACME"]}
    assert output["provenance"]["projected_spans"] == [
        [0, 3, "name"],
        [11, 15, "organization"],
    ]


def test_fastino_projection_uses_all_aliases_and_reports_unmapped(processor):
    tagset = Tagset()
    output, diagnostic = project(
        processor,
        "Ada joined ACME.",
        [[0, 3, "person_name"], [11, 15, "organization"]],
        inventory=list(tagset.sources["fastino_42"]),
        output_schema="fastino_42",
    )
    assert diagnostic.accepted
    assert output["output"]["entities"] == {"full_name": ["Ada"], "person": ["Ada"]}
    assert [(span.label, span.surface) for span in diagnostic.unmapped] == [("organization", "ACME")]


def test_fastino_first_alias_uses_model_card_order(processor):
    tagset = Tagset()
    output, diagnostic = project(
        processor,
        "Ada arrived.",
        [[0, 3, "person_name"]],
        inventory=list(tagset.sources["fastino_42"]),
        output_schema="fastino_42",
        alias_policy="first",
    )
    assert diagnostic.accepted
    assert output["output"]["entities"] == {"person": ["Ada"]}


def test_many_to_one_projection_collapses_identical_spans(processor):
    tagset = Tagset()
    output, diagnostic = project(
        processor,
        "contact",
        [[0, 7, "email"], [0, 7, "phone_number"]],
        inventory=sorted(tagset.cut_targets("redaction_9_v1")),
        output_schema="redaction_9_v1",
    )
    assert diagnostic.accepted
    assert diagnostic.collapsed_projected_spans == 1
    assert output["provenance"]["projected_spans"] == [[0, 7, "location_or_contact"]]
