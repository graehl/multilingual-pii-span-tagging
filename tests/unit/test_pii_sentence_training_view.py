import json

from scripts.pii_sentence_training_view import (
    CAPACITY_EXCEPTION,
    capacity_split_intervals,
    materialize_file,
    sentence_rows_for_batch,
    tokenizer_provenance,
)


class FixedSplitter:
    segmenter_model = "test/fixed"
    segmenter_model_revision = "test-revision"
    segmenter_name = "test.FixedSplitter"
    segmenter_version = "1"

    def split(self, texts):
        assert texts == ["Ada called. Bob replied."]
        return iter([["Ada called. ", "Bob replied."]])


class FixedTokenizer:
    init_kwargs = {"_commit_hash": "tokenizer-revision"}
    name_or_path = "test/tokenizer"
    model_max_length = 8

    def __call__(self, texts, **_kwargs):
        single = isinstance(texts, str)
        rows = [texts] if single else texts
        encoded = [[0, *range(len(text.split())), 2] for text in rows]
        return {"input_ids": encoded[0] if single else encoded}


def test_tokenizer_provenance_records_immutable_revision() -> None:
    assert tokenizer_provenance(FixedTokenizer(), "test/requested")["revision"] == ("tokenizer-revision")


def test_sentence_training_rows_reconstruct_text_spans_and_intake_provenance() -> None:
    rows, stats = sentence_rows_for_batch(
        [
            {
                "id": "doc-1",
                "text": "Ada called. Bob replied.",
                "spans": [[0, 3, "person_name"], [12, 15, "person_name"]],
                "lang": "en",
                "supervision": "complete",
            }
        ],
        [7],
        source_name="train.jsonl",
        source_sha256="f" * 64,
        splitter=FixedSplitter(),
        tokenizer=FixedTokenizer(),
        max_tokens=8,
    )

    assert [row["text"] for row in rows] == ["Ada called. ", "Bob replied."]
    assert [row["spans"] for row in rows] == [
        [[0, 3, "person_name"]],
        [[0, 3, "person_name"]],
    ]
    assert [row["intake_row_id"] for row in rows] == [
        "train.jsonl:7",
        "train.jsonl:7",
    ]
    assert [(row["source_start"], row["source_end"]) for row in rows] == [
        (0, 12),
        (12, 24),
    ]
    assert all("sampling_intake_factor" not in row for row in rows)
    assert stats["output_rows"] == 2
    assert stats["spans"] == 2


def test_overcapacity_sentence_uses_explicit_span_safe_pieces() -> None:
    class WholeSentenceSplitter(FixedSplitter):
        def split(self, texts):
            return iter([[text] for text in texts])

    text = "Ada; Bob; Cara; Dora."
    rows, stats = sentence_rows_for_batch(
        [{"text": text, "spans": [[5, 8, "person_name"]], "lang": "en"}],
        [11],
        source_name="val.jsonl",
        source_sha256="e" * 64,
        splitter=WholeSentenceSplitter(),
        tokenizer=FixedTokenizer(),
        max_tokens=4,
    )

    assert "".join(row["text"] for row in rows) == text
    assert [row["sentence_ordinal"] for row in rows] == [1, 1]
    assert [row["sentence_piece_ordinal"] for row in rows] == [1, 2]
    assert all(row["sentence_piece_count"] == 2 for row in rows)
    assert all(row["overcapacity_exception"] == CAPACITY_EXCEPTION for row in rows)
    assert all(row["view_token_count"] <= 4 for row in rows)
    assert stats["max_original_sentence_tokens"] == 6
    assert stats["max_sentence_tokens"] == 4
    assert stats["overcapacity_sentences"] == 1


def test_sentence_training_rows_reject_over_capacity_sentence() -> None:
    class LongTokenizer(FixedTokenizer):
        def __call__(self, texts, **_kwargs):
            single = isinstance(texts, str)
            rows = [texts] if single else texts
            encoded = [list(range(len(text) + 2)) for text in rows]
            return {"input_ids": encoded[0] if single else encoded}

    try:
        sentence_rows_for_batch(
            [
                {
                    "text": "ABCDEFGHI",
                    "spans": [],
                    "lang": "en",
                    "supervision": "complete",
                }
            ],
            [1],
            source_name="train.jsonl",
            source_sha256="f" * 64,
            splitter=FixedSplitterOneSentence(),
            tokenizer=LongTokenizer(),
            max_tokens=8,
        )
    except ValueError as error:
        assert "no lexical boundary" in str(error)
    else:
        raise AssertionError("over-capacity sentence was accepted")


def test_capacity_split_can_use_declared_no_space_source_boundaries() -> None:
    class CharacterTokenizer(FixedTokenizer):
        def __call__(self, texts, **_kwargs):
            single = isinstance(texts, str)
            rows = [texts] if single else texts
            encoded = [list(range(len(text) + 2)) for text in rows]
            return {"input_ids": encoded[0] if single else encoded}

    text = "東京都千代田区"
    pieces = capacity_split_intervals(
        text,
        [[0, 2, "location"]],
        CharacterTokenizer(),
        5,
        fallback_boundaries=tuple(range(1, len(text))),
    )

    assert "".join(text[start:end] for start, end, _tokens in pieces) == text
    assert all(tokens <= 5 for _start, _end, tokens in pieces)
    assert 1 not in [end for _start, end, _tokens in pieces[:-1]]


def test_capacity_split_does_not_make_edge_only_punctuation_piece() -> None:
    class CharacterTokenizer(FixedTokenizer):
        def __call__(self, texts, **_kwargs):
            single = isinstance(texts, str)
            rows = [texts] if single else texts
            encoded = [list(range(len(text) + 2)) for text in rows]
            return {"input_ids": encoded[0] if single else encoded}

    text = ":東京都千代田区"
    pieces = capacity_split_intervals(
        text,
        [],
        CharacterTokenizer(),
        6,
        fallback_boundaries=tuple(range(1, len(text))),
    )

    assert pieces[0][1] != 1


def test_materialize_file_filters_exact_languages_and_sources(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "selected.jsonl"
    rows = [
        {"text": "Ada called.", "spans": [], "lang": "en", "src": "tab"},
        {"text": "Ana llamó.", "spans": [], "lang": "es", "src": "meddocan"},
        {"text": "Bob called.", "spans": [], "lang": "en", "src": "synthetic"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = materialize_file(
        source,
        destination,
        source_name="train.jsonl",
        splitter=FixedSplitterOneSentence(),
        tokenizer=FixedTokenizer(),
        max_tokens=8,
        batch_size=2,
        include_languages=frozenset({"en"}),
        include_sources=frozenset({"tab"}),
    )

    output = json.loads(destination.read_text(encoding="utf-8"))
    assert output["text"] == "Ada called."
    assert output["intake_line_number"] == 1
    assert result["source_rows"] == 3
    assert result["input_rows"] == 1
    assert result["excluded_rows"] == 2


class FixedSplitterOneSentence:
    segmenter_model = "test/fixed"
    segmenter_model_revision = "test-revision"
    segmenter_name = "test.FixedSplitterOneSentence"
    segmenter_version = "1"

    def split(self, texts):
        return iter([[text] for text in texts])
