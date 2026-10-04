import hashlib

import pytest

from scripts.pii_document_context import (
    adjacent_document_context,
    check_context_mixture,
    resolve_context_configurations,
    resolve_context_side,
    select_document_context,
)


def test_context_mixture_resume_preserves_order_and_rejects_changes():
    mixture = [[-1, 0, 1], [-1, 0], [0]]
    assert resolve_context_configurations(mixture) == mixture
    assert resolve_context_configurations(None, mixture, exact_resume=True) == mixture
    assert resolve_context_configurations(None, None, exact_resume=True) is None
    with pytest.raises(ValueError, match="exact resume"):
        resolve_context_configurations([[0]], mixture, exact_resume=True)
    with pytest.raises(ValueError, match="exact resume"):
        resolve_context_configurations(mixture, None, exact_resume=True)


def test_previous_only_mixture_may_not_select_the_next_neighbor():
    check_context_mixture("context", "previous", [[-1, 0], [0]])
    check_context_mixture("context", "both", [[-1, 0, 1], [0]])
    check_context_mixture(None, "both", None)
    with pytest.raises(ValueError, match="next neighbor"):
        check_context_mixture("context", "previous", [[-1, 0, 1], [0]])
    with pytest.raises(ValueError, match="context-field"):
        check_context_mixture(None, "both", [[0]])


@pytest.mark.parametrize("value", [[], [0], [[1]], [[-2, 0]], [[0, 0]], [[1, 0]], [[False]], [[0], [0]]])
def test_context_mixture_rejects_invalid_offsets(value):
    with pytest.raises(ValueError, match="context"):
        resolve_context_configurations(value)


def test_context_selection_does_not_mutate_or_invent_neighbors():
    context = {"before": "", "after": "Next sentence."}
    assert select_document_context(context, [-1, 0, 1]) == select_document_context(context, [0, 1])
    assert select_document_context(context, [-1, 0]) == {"before": "", "after": ""}
    assert context == {"before": "", "after": "Next sentence."}
    assert select_document_context("Previous sentence.", [0]) == ""


def test_context_side_inheritance_and_exact_resume():
    assert resolve_context_side(None) == "both"
    assert resolve_context_side(None, "previous", exact_resume=True) == "previous"
    assert resolve_context_side("both", "previous") == "both"
    with pytest.raises(ValueError, match="exact resume"):
        resolve_context_side("both", "previous", exact_resume=True)
    with pytest.raises(ValueError, match="both or previous"):
        resolve_context_side(None, "unknown")


@pytest.mark.parametrize(
    "origin,paragraph,expected", [(0, False, True), (100, False, False), (100, True, True)]
)
def test_neighbors_require_complete_source_boundaries(origin, paragraph, expected):
    excerpt = "Before. Target. After."
    source = {
        "text": "Target.",
        "annotation_context": excerpt,
        "annotation_context_start": origin,
        "annotation_context_kind": "blank_line_paragraph" if paragraph else "bounded_paragraph_window",
        "source_document_start": origin + 8,
        "source_document_end": origin + 15,
        "source_document_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
    }
    context, intervals = adjacent_document_context(source, [(0, 8), (8, 16), (16, 22)])
    assert context == ({"before": "Before.", "after": "After."} if expected else {"before": "", "after": ""})
    assert intervals == (
        {"before": [origin, origin + 7], "after": [origin + 16, origin + 22]} if expected else {}
    )
    source["source_document_end"] += 1
    with pytest.raises(ValueError, match="character interval"):
        adjacent_document_context(source, [(0, 8), (8, 16), (16, 22)])
