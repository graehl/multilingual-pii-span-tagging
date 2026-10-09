import json

from scripts.pii_gliner2_ont3_finetune import (
    json_result,
    read_pooled_jsonl,
    sample_current_mix,
    snap_window,
    surface_example,
)


def test_trainer_receipt_spells_non_finite_metric_as_string():
    assert json_result({"best_metric": float("inf"), "steps": 8}) == {
        "best_metric": "inf",
        "steps": 8,
    }


def test_current_mix_sampling_applies_pool_and_language_weights(tmp_path):
    rows = [
        {"text": "one", "lang": "en", "sampling_pool": "a"},
        {"text": "two", "lang": "fr", "sampling_pool": "b"},
    ]
    sampling = tmp_path / "sampling.json"
    sampling.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pools": [
                    {"name": "a", "match": {"sampling_pool": "a"}, "weight": 0.5},
                    {"name": "b", "match": {"sampling_pool": "b"}, "weight": 0.5},
                ],
            }
        )
    )
    languages = tmp_path / "languages.json"
    languages.write_text(json.dumps({"en": 0.0, "fr": 1.0}))

    sampled, receipt = sample_current_mix(
        rows,
        sampling_config=sampling,
        language_weights_path=languages,
        draws=4,
        seed=173,
    )

    assert sampled == [rows[1]] * 4
    assert receipt["realized_pools"] == {"b": 4}


def test_pool_file_supplies_missing_inline_pool(tmp_path):
    path = tmp_path / "final35-training-v1.jsonl"
    path.write_text(json.dumps({"text": "one", "lang": "en"}) + "\n")

    assert read_pooled_jsonl([path])[0]["sampling_pool"] == "final35-training-v1"


def test_partial_supervision_does_not_invent_negative_labels():
    window = {
        "input": "Alice met Bob",
        "provenance": {"projected_spans": [[0, 5, "person_name"]]},
    }

    example = surface_example(
        window,
        ["organization", "person_name"],
        negatives=1,
        seed=173,
        known_negative_labels=False,
    )

    assert example == {"text": "Alice met Bob", "entities": {"person_name": ["Alice"]}}


def test_complete_supervision_keeps_requested_negative_labels():
    window = {
        "input": "Alice met Bob",
        "provenance": {"projected_spans": [[0, 5, "person_name"]]},
    }

    example = surface_example(
        window,
        ["organization", "person_name"],
        negatives=1,
        seed=173,
        known_negative_labels=True,
    )

    assert example == {
        "text": "Alice met Bob",
        "entities": {"person_name": ["Alice"], "organization": []},
    }


def test_snapping_accounts_for_processor_appended_url_punctuation():
    text = "www.example.com"

    def splitter(value):
        return [(value.lower(), 0, len(value))]

    window, changes = snap_window(
        {"input": text, "provenance": {"projected_spans": [[0, len(text), "url"]]}}, splitter, 8
    )
    assert window["input"] == text + "."
    assert window["provenance"]["projected_spans"] == [[0, len(text) + 1, "url"]]
    assert changes[0]["original"] == [0, len(text), "url"]
    assert surface_example(window, ["url"], 0, 173, word_splitter=splitter)["entities"] == {
        "url": [text + "."]
    }


def test_token_matching_rejects_unannotated_case_variant():
    def splitter(value):
        if value == "Alice alice":
            return [("alice", 0, 5), ("alice", 6, 11)]
        return [(value.lower(), 0, len(value))]

    window = {"input": "Alice alice", "provenance": {"projected_spans": [[0, 5, "person_name"]]}}
    assert surface_example(window, ["person_name"], 0, 173, word_splitter=splitter) is None
