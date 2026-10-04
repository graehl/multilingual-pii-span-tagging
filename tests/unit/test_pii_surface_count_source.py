"""Whole-document counting can select the large split without changing retrieval."""

import sys
from types import SimpleNamespace

import pytest

from scripts.pii_ont3_surface_retrieval import fineweb_stream, source_descriptor


@pytest.mark.parametrize(
    ("language", "override", "expected"),
    [("en", None, "train"), ("hr", None, "test"), ("hr", "train", "train")],
)
def test_stream_split_matches_saved_source(monkeypatch, language, override, expected):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(shuffle=lambda **kwargs: [{"id": "source", "text": "Alice"}])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    rows = list(fineweb_stream(language, seed=11, shuffle_buffer=20, upstream_split=override))
    source = source_descriptor(rows[0], language, upstream_split=override)
    assert calls[0][1]["split"] == source["upstream_split"] == expected
    assert calls[0][1]["revision"] == source["dataset_revision"]
    assert calls[0][1]["streaming"] is True
