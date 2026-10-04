"""Run the name-kind conformance corpus through the shipped configuration.

The corpus holds real carriers whose components the shipped configuration and
the annotator already agree on, so a failure means the postprocessor's reading
moved. It is also the sync corpus for the C++ implementation in the production toolkit: both read
the same configuration and the same ONNX graph, so both should reproduce every
case.

The character scorer is not committed, so the test skips where the exported
bundle is absent. Point NAME_KIND_BUNDLE at an export directory to use another.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from scripts.pii_name_annotation_qc import NameAnnotationQc, load_config  # noqa: E402
from scripts.pii_name_role_publish import NameKindOnnxDeployment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "data" / "name-kind-conformance-v1.json"
BUNDLE = Path(os.environ.get("NAME_KIND_BUNDLE") or (ROOT / "untracked" / "name-kind-v4"))


def load_corpus() -> dict:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def oracle() -> NameAnnotationQc:
    model_config = BUNDLE / "name-kind.config.json"
    if not model_config.is_file():
        pytest.skip(f"no name-kind bundle at {BUNDLE}; set NAME_KIND_BUNDLE")
    corpus = load_corpus()
    return NameAnnotationQc(
        load_config(ROOT / corpus["config"]),
        name_kind_deployment=NameKindOnnxDeployment.load(model_config),
        override_name_kinds=True,
    )


def assigned(analysis: dict) -> dict:
    """Name components per carrier, Q excluded, in a comparable shape."""
    out = collections.defaultdict(list)
    for span in analysis["candidate"]["subclass_spans"]:
        if span.get("family") == "name_component" and span["value"] != "Q":
            out[(span["carrier_start"], span["carrier_end"])].append(
                [span["start"], span["end"], span["value"]]
            )
    return {carrier: sorted(spans) for carrier, spans in out.items()}


def test_the_corpus_covers_every_language_and_shape():
    corpus = load_corpus()
    cases = corpus["cases"]
    assert len(cases) >= 100
    languages = collections.Counter(case["language"] for case in cases)
    assert set(languages) == {"de", "en", "es", "fr", "ja", "ko", "ru", "zh"}
    assert min(languages.values()) >= 8
    shapes = {case["assignment_shape"] for case in cases}
    assert {"family_last", "family_first", "mononym"} <= shapes
    for case in cases:
        # Every case states agreed behaviour, so the two readings cannot differ.
        assert case["expected_components"] == case["annotator_components"], case["id"]
        start, end = case["carrier"]
        assert case["surface"] == case["text"][start:end], case["id"]


def test_shipped_configuration_reproduces_every_conformance_case(oracle):
    corpus = load_corpus()
    disagreed = []
    for case in corpus["cases"]:
        analysis = oracle.analyze(
            row_id=case["id"],
            text=case["text"],
            language=case["language"],
            annotations=case["annotations"],
        )
        got = assigned(analysis).get(tuple(case["carrier"]), [])
        if got != case["expected_components"]:
            disagreed.append((case["id"], case["surface"], case["expected_components"], got))
    assert not disagreed, "\n".join(
        f"{case_id} {surface!r}: expected {want}, got {got}" for case_id, surface, want, got in disagreed
    )
