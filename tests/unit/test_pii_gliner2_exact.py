from types import SimpleNamespace

import pytest
import torch

from scripts.pii_gliner2_exact import (
    CjkAwareTokenSplitter,
    RepresentationError,
    collate_exact_entity_rows,
    collate_exact_entity_rows_with_masks,
    configure_span_width,
    representation_ledger,
    text_windows,
    token_budgeted_projected_rows,
    transform_exact_entity_row,
    window_projected_row,
)


class MinimalTokenizer:
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


def processor():
    from gliner2.processor import SchemaTransformer

    return SchemaTransformer(tokenizer=MinimalTokenizer())


def test_cjk_splitter_preserves_ascii_tokens_and_offsets():
    text = "카드ABC123 与张三 x@example.test"
    observed = list(CjkAwareTokenSplitter()(text))
    assert [token for token, _, _ in observed] == [
        "카",
        "드",
        "abc",
        "123",
        "与",
        "张",
        "三",
        "x",
        "@example",
        ".",
        "test",
    ]
    assert all(text[start:end].lower() == token for token, start, end in observed)


def test_cjk_splitter_stops_url_before_unspaced_cjk_prose():
    text = "网址https://example.test/path。继续"
    observed = list(CjkAwareTokenSplitter()(text))
    assert ("https://example.test/path", 2, 27) in observed
    assert observed[-2:] == [("继", 28, 29), ("续", 29, 30)]


def test_windows_are_gold_independent_and_cover_text():
    first = text_windows("x" * 31, max_chars=12, overlap_chars=5)
    second = text_windows("x" * 31, max_chars=12, overlap_chars=5)
    assert [(item.start, item.end) for item in first] == [(item.start, item.end) for item in second]
    assert first[0].start == 0
    assert first[-1].end == 31
    assert all(left.end >= right.start for left, right in zip(first, first[1:]))


def test_over_budget_window_is_refined_with_required_overlap():
    row = {
        "id": "doc",
        "lang": "ko",
        "membership_split": "train",
        "text": "가" * 36,
        "spans": [],
    }
    windows, split_count = token_budgeted_projected_rows(
        processor(),
        row,
        ["name"],
        max_chars=24,
        overlap_chars=10,
        max_input_tokens=22,
        max_span_width=8,
    )
    assert split_count > 0
    assert windows[0]["provenance"]["window_start"] == 0
    assert windows[-1]["provenance"]["window_end"] == 36
    assert all(
        left["provenance"]["window_end"] - right["provenance"]["window_start"] >= 10
        for left, right in zip(windows, windows[1:])
    )


def test_exact_offsets_do_not_invent_duplicate_surface_targets():
    labels = ["name", "organization"]
    row = {
        "input": "Jordan met Jordan.",
        "provenance": {"projected_spans": [[0, 6, "name"]]},
    }
    batch = collate_exact_entity_rows(
        processor(),
        [row],
        labels,
        max_input_tokens=64,
        max_span_width=8,
    )
    structure = batch.structure_labels[0][0]
    assert structure == [1, [[[(0, 0)], (-1, -1)]]]


def test_exact_offsets_represent_cjk_inside_stock_whitespace_token():
    labels = ["unique_identifier"]
    row = {
        "input": "카드번호1234입니다.",
        "provenance": {"projected_spans": [[4, 8, "unique_identifier"]]},
    }
    batch = collate_exact_entity_rows(
        processor(),
        [row],
        labels,
        max_input_tokens=64,
        max_span_width=8,
    )
    assert batch.structure_labels[0][0] == [1, [[[(4, 4)]]]]


def test_metric_equivalent_boundary_adjustment_is_receipted():
    labels = ["location_or_contact"]
    row = {
        "input": "Visit https://example.test/path.",
        "provenance": {"projected_spans": [[6, 31, "location_or_contact"]]},
    }
    receipt = []
    transform_exact_entity_row(
        processor(),
        row,
        labels,
        max_input_tokens=64,
        max_span_width=8,
        alignment_receipt=receipt,
    )
    assert receipt == [
        {
            "gold": [6, 31],
            "represented": [6, 32],
            "label": "location_or_contact",
            "match_rule": "symmetric-80-percent-character-coverage",
        }
    ]


def test_boundary_adjustment_below_metric_overlap_is_rejected():
    row = {
        "input": "Sudanese",
        "provenance": {"projected_spans": [[0, 5, "location_or_contact"]]},
    }
    with pytest.raises(RepresentationError, match="below symmetric 80%"):
        transform_exact_entity_row(
            processor(),
            row,
            ["location_or_contact"],
            max_input_tokens=64,
            max_span_width=8,
        )


def test_unrepresentable_span_can_be_receipted_without_dropping_row():
    row = {
        "input": "Sudanese account 73592046",
        "provenance": {
            "projected_spans": [
                [0, 5, "location_or_contact"],
                [17, 25, "financial"],
            ]
        },
    }
    receipt = []
    transformed = transform_exact_entity_row(
        processor(),
        row,
        ["financial", "location_or_contact"],
        max_input_tokens=64,
        max_span_width=8,
        unrepresentable_receipt=receipt,
    )
    assert receipt[0]["gold"] == [0, 5]
    assert transformed.structure_labels == [[1, [[[(2, 2)], (-1, -1)]]]]


def test_boundary_fragments_and_unreachable_spans_are_excluded_from_loss():
    row = {
        "input": "Alpha Beta Gamma Delta",
        "provenance": {
            "window_start": 10,
            "window_end": 32,
            "projected_spans": [[6, 10, "name"], [19, 22, "name"]],
            "straddling_projected_spans": [[8, 15, "name"]],
            "unrepresentable_complete_spans": [
                {"gold": [19, 22], "label": "name", "reason": "unaligned_span"}
            ],
        },
    }
    _batch, masks = collate_exact_entity_rows_with_masks(
        processor(),
        [row],
        ["name"],
        max_input_tokens=64,
        max_span_width=4,
    )
    mask = masks[0]
    assert mask[0, 0]
    assert not mask[1, 0]
    assert mask[3, 0]


def test_span_width_extension_adds_no_parameters():
    layer = SimpleNamespace(max_width=8)
    model = SimpleNamespace(
        max_width=8,
        config=SimpleNamespace(max_width=8),
        span_rep=SimpleNamespace(span_rep_layer=layer),
        parameters=lambda: [torch.nn.Parameter(torch.zeros(3, 4))],
    )
    before = sum(parameter.numel() for parameter in model.parameters())
    assert configure_span_width(model, 24) == {"original": 8, "configured": 24}
    after = sum(parameter.numel() for parameter in model.parameters())
    assert before == after == 12
    assert (model.max_width, model.config.max_width, layer.max_width) == (24, 24, 24)


def test_straddling_training_window_is_retained_and_span_is_covered():
    text = "가" * 80
    row = {
        "id": "doc",
        "lang": "en",
        "membership_split": "train",
        "text": text,
        "spans": [[27, 38, "person_name"]],
    }
    windows = window_projected_row(
        row,
        max_chars=30,
        overlap_chars=15,
        drop_straddling_windows=True,
    )
    assert all(not item["provenance"]["straddling_projected_spans"] for item in windows)
    accepted, ledger = representation_ledger(
        processor(),
        [row],
        ["name"],
        max_chars=30,
        overlap_chars=15,
        max_input_tokens=64,
        max_span_width=24,
    )
    assert any(item["provenance"]["straddling_projected_spans"] for item in accepted)
    assert ledger["settings"]["straddling_gold_policy"] == (
        "retain-window-and-mask-boundary-fragments-in-loss"
    )
    assert ledger["counts"]["uncovered_spans"] == 0
