import numpy as np
import pytest

from scripts.pii_gliner2_span_decode import (
    SourceOffsetWordSplitter,
    grouped_document_ranges,
    native_decode,
    per_label_semimarkov_decode,
    semimarkov_decode,
)


def proposal(start, end, label, weight, order):
    return {
        "start": start,
        "end": end,
        "label": label,
        "logit": weight,
        "weight": weight,
        "order": order,
    }


def geometries(items):
    return {(item["start"], item["end"], item["label"]) for item in items}


def test_semimarkov_prefers_two_compatible_spans_over_greedy_wide_span():
    proposals = [
        proposal(0, 4, "name", 3.0, 0),
        proposal(0, 2, "name", 2.0, 1),
        proposal(2, 4, "name", 2.0, 2),
    ]

    assert geometries(native_decode(proposals)) == {(0, 4, "name")}
    assert geometries(semimarkov_decode(proposals)) == {
        (0, 2, "name"),
        (2, 4, "name"),
    }


def test_semimarkov_resolves_cross_label_overlap_globally():
    proposals = [
        proposal(0, 3, "name", 2.0, 0),
        proposal(1, 4, "address", 3.0, 1),
    ]

    assert len(native_decode(proposals)) == 2
    assert len(per_label_semimarkov_decode(proposals)) == 2
    assert geometries(semimarkov_decode(proposals)) == {(1, 4, "address")}


def test_per_label_semimarkov_improves_greedy_without_removing_cross_label_overlap():
    proposals = [
        proposal(0, 4, "name", 3.0, 0),
        proposal(0, 2, "name", 2.0, 1),
        proposal(2, 4, "name", 2.0, 2),
        proposal(1, 3, "address", 5.0, 3),
    ]

    assert geometries(per_label_semimarkov_decode(proposals)) == {
        (0, 2, "name"),
        (2, 4, "name"),
        (1, 3, "address"),
    }


def test_nonpositive_proposals_are_not_selected():
    proposals = [proposal(0, 1, "name", 0.0, 0), proposal(1, 2, "name", -1.0, 1)]

    assert geometries(native_decode(proposals)) == {(0, 1, "name")}
    assert semimarkov_decode(proposals) == []


def test_grouped_document_ranges_include_empty_documents_and_reject_disorder():
    documents = np.asarray([0, 0, 2, 2], dtype=np.int32)

    assert grouped_document_ranges(documents, 3).tolist() == [0, 2, 2, 4]
    with pytest.raises(ValueError, match="grouped in document order"):
        grouped_document_ranges(np.asarray([0, 1, 0], dtype=np.int32), 2)


def test_source_offset_word_splitter_lowercases_without_moving_offsets():
    class ExpandingLowercaseSplitter:
        def __call__(self, text, lower=True):
            if lower:
                yield "i", 0, 1
                yield "̇", 1, 2
                yield "laç", 2, 5
            else:
                yield text, 0, len(text)

    text = "İlaç"

    splitter = SourceOffsetWordSplitter(ExpandingLowercaseSplitter())
    assert list(splitter(text)) == [("i", 0, 1), ("̇", 0, 1), ("laç", 1, 4)]
    assert list(splitter(text, lower=False)) == [(text, 0, 4)]
