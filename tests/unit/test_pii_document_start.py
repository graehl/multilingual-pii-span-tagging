import pytest

from scripts.pii_document_context import encode_document_context, select_document_context
from scripts.pii_encoder_train import expand_context_variants

START = {"before": "", "after": "", "document_start": True}


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "FacebookAI/xlm-roberta-large", local_files_only=True
        )
    except OSError:
        pytest.skip("xlm-roberta-large tokenizer is not cached")


def test_the_flag_travels_only_with_the_previous_neighbor():
    assert select_document_context(START, [-1, 0]) == START
    assert select_document_context(START, [0]) == {"before": "", "after": ""}


def test_a_document_start_cannot_have_a_previous_sentence():
    with pytest.raises(ValueError, match="first sentence"):
        select_document_context({"before": "Earlier.", "after": "", "document_start": True}, [-1, 0])


def test_marker_is_the_empty_first_segment_pair_template(tokenizer):
    text = "Anna Berg lives in Oslo."
    bare = tokenizer(text, return_offsets_mapping=True)
    encoded, offsets = encode_document_context(
        tokenizer, text, START, 512, side="previous", document_start_marker=True
    )

    # <s></s></s> target </s>: exactly the tokenizer's own pair encoding with an empty first segment.
    assert encoded["input_ids"] == tokenizer("", text)["input_ids"]
    assert encoded["input_ids"] == [
        bare["input_ids"][0],
        tokenizer.sep_token_id,
        tokenizer.sep_token_id,
        *bare["input_ids"][1:],
    ]
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])
    # Marker positions are zero-width like every special token; target offsets are unchanged.
    assert offsets[:3] == [(0, 0)] * 3
    assert [o for o in offsets if o[1] > o[0]] == [tuple(o) for o in bare["offset_mapping"] if o[1] > o[0]]


def test_models_without_the_marker_see_the_bare_input(tokenizer):
    # None sends the caller down its ordinary no-context path, exactly as before the flag existed.
    assert encode_document_context(tokenizer, "Anna Berg.", START, 512, side="previous") is None


def test_known_start_rows_split_into_marker_and_bare_variants_unknown_rows_do_not():
    rows = [
        {"id": "start", "text": "Anna Berg.", "context": dict(START)},
        {"id": "unknown", "text": "Bo Lind.", "context": {"before": "", "after": ""}},
    ]
    entries, weights, _ = expand_context_variants(rows, [1.0, 1.0], None, "context", [[-1, 0], [0]])
    variants = sorted((e["id"], "document_start" in e["context"], w) for e, w in zip(entries, weights))
    assert variants == [("start", False, 0.5), ("start", True, 0.5), ("unknown", False, 1.0)]
