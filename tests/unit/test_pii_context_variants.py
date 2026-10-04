from types import SimpleNamespace

import pytest

from scripts import pii_encoder_train
from scripts.pii_document_context import resolve_context_weights
from scripts.pii_encoder_train import expand_context_variants

CONFIGURATIONS = [[-1, 0], [0]]


def test_each_context_variant_becomes_an_entry_with_its_weight_share():
    rows = [
        {"id": "a", "text": "Anna left.", "context": {"before": "She was here.", "after": "Then rain."}},
        {"id": "b", "text": "Bo stayed.", "context": {"before": "", "after": "Later."}},
        {"id": "c", "text": "Cy spoke."},
    ]
    entries, weights, keys = expand_context_variants(
        rows, [1.0, 2.0, 3.0], ["p", "q", "r"], "context", CONFIGURATIONS
    )

    # Row a has a previous sentence: two entries, half its weight each.
    a = [(e["context"], w) for e, w in zip(entries, weights) if e["id"] == "a"]
    assert sorted(a, key=lambda x: x[0]["before"]) == [
        ({"before": "", "after": ""}, 0.5),
        ({"before": "She was here.", "after": ""}, 0.5),
    ]
    # Row b has no previous sentence: both configurations select the same context, one entry.
    assert [(e["context"], w) for e, w in zip(entries, weights) if e["id"] == "b"] == [
        ({"before": "", "after": ""}, 2.0)
    ]
    # Row c has no context field: unchanged.
    assert [(e, w) for e, w in zip(entries, weights) if e["id"] == "c"] == [(rows[2], 3.0)]
    # Every row keeps its total weight, and pool keys follow their rows.
    for row, weight in zip(rows, [1.0, 2.0, 3.0]):
        assert sum(w for e, w in zip(entries, weights) if e["id"] == row["id"]) == pytest.approx(weight)
    assert keys == ["p", "p", "q", "r"]


def test_configuration_weights_split_a_row_by_weight():
    rows = [
        {"id": "a", "text": "Anna left.", "context": {"before": "She was here.", "after": ""}},
        {"id": "b", "text": "Bo stayed.", "context": {"before": "", "after": ""}},
    ]
    weights = resolve_context_weights([4, 1], [[0], [-1, 0]])
    assert weights == pytest.approx([0.8, 0.2])
    entries, entry_weights, _ = expand_context_variants(
        rows, [1.0, 2.0], None, "context", [[0], [-1, 0]], weights
    )
    a = sorted((e["context"]["before"], w) for e, w in zip(entries, entry_weights) if e["id"] == "a")
    assert a == [("", pytest.approx(0.8)), ("She was here.", pytest.approx(0.2))]
    # Without a previous sentence both configurations merge and the row keeps its weight.
    assert [w for e, w in zip(entries, entry_weights) if e["id"] == "b"] == [pytest.approx(2.0)]


@pytest.mark.parametrize(
    ("weights", "configurations"),
    [([1], [[0], [-1, 0]]), ([1, 0], [[0], [-1, 0]]), ([1, True], [[0], [-1, 0]]), ([1, 1], None)],
)
def test_configuration_weights_are_validated(weights, configurations):
    with pytest.raises(ValueError):
        resolve_context_weights(weights, configurations)


def test_exact_resume_keeps_saved_configuration_weights():
    saved = resolve_context_weights([4, 1], [[0], [-1, 0]])
    assert resolve_context_weights(None, [[0], [-1, 0]], saved, exact_resume=True) == saved
    with pytest.raises(ValueError):
        resolve_context_weights([1, 1], [[0], [-1, 0]], saved, exact_resume=True)


def test_length_text_counts_a_fixed_previous_sentence():
    dataset = SimpleNamespace(
        context_field="context",
        context_configurations=None,
        context_side="previous",
        context_separator="\n\n",
        document_start_marker=False,
    )
    row = {"text": "Anna left.", "context": {"before": " She was here. ", "after": "Then rain."}}
    assert pii_encoder_train.SpanDataset.length_text(dataset, row) == "She was here.\n\nAnna left."
    # Under per-draw configurations the variant is unknown, so only the target counts.
    dataset.context_configurations = CONFIGURATIONS
    assert pii_encoder_train.SpanDataset.length_text(dataset, row) == "Anna left."
