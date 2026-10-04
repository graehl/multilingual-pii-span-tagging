"""Repeated annotations share exposure through the actual window/sampler path."""

import json
from collections import Counter

import pytest

from scripts.pii_annotation_sampling import share_annotation_sampling_mass
from scripts.pii_encoder_train import sampling_plan_for_rows, window_records
from trainlib import WeightedLengthBatchSampler


def test_reannotation_keeps_one_input_mass_after_pool_compilation(tmp_path):
    rows = [
        {"lang": "en", "text": "Ada arrived.", "spans": [[0, 3, "person_name"]]},
        {"lang": "en", "text": "Ada arrived.", "spans": []},
        {"lang": "en", "text": "Bob departed.", "spans": [[0, 3, "person_name"]]},
    ]
    for row in rows:
        row["sampling_pool"] = "native"
    corpus = tmp_path / "train.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = tmp_path / "sampling.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pools": [
                    {"name": "native", "match": {"sampling_pool": "native"}, "weight": 1.0},
                ],
            }
        )
    )
    windows = window_records(corpus, 900)
    original, _, _ = sampling_plan_for_rows(windows, config)
    assert original == pytest.approx([1 / 3] * 3)
    shared, receipt = share_annotation_sampling_mass(windows, original)
    assert shared == pytest.approx([0.25, 0.25, 0.5])
    assert receipt["multiply_annotated_inputs"] == 1
    assert receipt["rows"] == 3
    assert receipt["distinct_inputs"] == 2
    sampler = WeightedLengthBatchSampler(
        lengths=[len(row["text"]) for row in windows],
        weights=shared,
        batch_size=4,
        gradient_accumulation_steps=1,
        seed=173,
        epoch_examples=400,
        length_window_steps=100,
    )
    drawn = Counter(index for batch in sampler for index in batch)
    assert drawn == {0: 100, 1: 100, 2: 200}


def test_all_annotations_share_equally_even_across_weighted_pools():
    rows = [
        {"lang": "en", "text": "Same sentence"},
        {"lang": "en", "text": "Same sentence"},
        {"lang": "en", "text": "Different sentence"},
    ]
    weights, _ = share_annotation_sampling_mass(rows, [0.2, 0.6, 0.4])
    assert weights == pytest.approx([0.25, 0.25, 0.5])


def test_unique_inputs_preserve_existing_weights_and_uniform_mode():
    rows = [{"lang": "en", "text": "A"}, {"lang": "de", "text": "A"}]
    weights, receipt = share_annotation_sampling_mass(rows, None)
    assert weights is None
    assert receipt["multiply_annotated_inputs"] == 0
    weights, _ = share_annotation_sampling_mass(rows, [0.2, 0.8])
    assert weights == [0.2, 0.8]
