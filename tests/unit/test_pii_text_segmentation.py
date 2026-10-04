import re

from scripts.pii_text_segmentation import (
    expand_training_views,
    placeholder_alignment_metrics,
    placeholder_partition,
    sentence_spans,
    split_at_line_boundaries,
)


class FixedSplitter:
    segmenter_model = "test/fixed"
    segmenter_model_revision = "test-revision"
    segmenter_name = "test.FixedSplitter"
    segmenter_version = "1"

    def __init__(self, pieces):
        self.pieces = pieces

    def split(self, texts):
        assert len(texts) == 1
        return iter([self.pieces])


def test_line_boundary_split_preserves_every_delimiter() -> None:
    text = "first line\n\nsecond line\r\nlast"

    pieces = split_at_line_boundaries(text)

    assert pieces == ["first line\n\n", "second line\r\n", "last"]
    assert "".join(pieces) == text


def test_line_boundary_split_attaches_leading_blank_lines_to_content() -> None:
    assert split_at_line_boundaries("\n\nfield: value\n") == ["\n\nfield: value\n"]


def test_sentence_spans_merge_candidate_boundary_inside_labeled_span() -> None:
    text = "Contact Ada Lovelace. Next sentence."
    splitter = FixedSplitter(["Contact Ada ", "Lovelace. ", "Next sentence."])

    spans = sentence_spans(text, [[8, 20, "person_name"]], splitter)

    assert spans == [(0, 22), (22, len(text))]
    assert "".join(text[start:end] for start, end in spans) == text


def test_placeholder_partition_ignores_segment_order_but_preserves_cooccurrence() -> None:
    pattern = re.compile(r"\[([A-Z_]+_\d+)\]")
    source = "Call [PHONE_1]. Mail [EMAIL_2]."
    reordered = "Écrivez à [EMAIL_2]. Appelez [PHONE_1]."
    merged = "Appelez [PHONE_1] ou écrivez à [EMAIL_2]."

    source_partition = placeholder_partition(source, [(0, 16), (16, len(source))], pattern)
    reordered_partition = placeholder_partition(
        reordered,
        [(0, 22), (22, len(reordered))],
        pattern,
    )
    merged_partition = placeholder_partition(merged, [(0, len(merged))], pattern)

    assert reordered_partition == source_partition
    assert merged_partition != source_partition


def test_placeholder_partition_preserves_duplicate_placeholder_multiplicity() -> None:
    pattern = re.compile(r"\[([A-Z_]+_\d+)\]")
    separate = "[NAME_1] called. [NAME_1] replied."
    together = "[NAME_1] called and [NAME_1] replied."

    assert placeholder_partition(separate, [(0, 17), (17, len(separate))], pattern) != (
        placeholder_partition(together, [(0, len(together))], pattern)
    )


def test_placeholder_alignment_scores_monotone_and_far_reordered_sequences() -> None:
    pattern = re.compile(r"\[([A-Z_]+_\d+)\]")
    monotone = "[A_1] [B_2] [C_3] [D_4]"
    reordered = "[D_4] [B_2] [C_3] [A_1]"

    aligned = placeholder_alignment_metrics(monotone, monotone, pattern)
    displaced = placeholder_alignment_metrics(monotone, reordered, pattern)

    assert aligned["mean_normalized_displacement"] == 0.0
    assert aligned["inversion_rate"] == 0.0
    assert displaced["mean_normalized_displacement"] == 0.5
    assert displaced["max_normalized_displacement"] == 1.0
    assert displaced["inversion_rate"] == 5 / 6


def test_placeholder_alignment_handles_repeated_identities_by_occurrence() -> None:
    pattern = re.compile(r"\[([A-Z_]+_\d+)\]")
    source = "[NAME_1] [PHONE_2] [NAME_1]"
    target = "[NAME_1] [NAME_1] [PHONE_2]"

    metrics = placeholder_alignment_metrics(source, target, pattern)

    assert metrics["same_occurrences"] is True
    assert metrics["inversion_rate"] == 1 / 3


def test_expand_training_views_keeps_equal_segment_base_weight_and_rebases_spans() -> None:
    row = {
        "id": "doc-7",
        "text": "Ada called. Bob replied.",
        "spans": [[0, 3, "name"], [12, 15, "name"]],
        "lang": "en",
    }
    splitter = FixedSplitter(["Ada called. ", "Bob replied."])

    views = expand_training_views(row, splitter, long_view_alpha=0.4)

    assert [view["text_view"] for view in views] == ["paragraph", "segment", "segment"]
    assert [view["sampling_weight"] for view in views] == [0.8, 0.6, 0.6]
    assert sum(view["sampling_weight"] for view in views) == 2.0
    assert views[1]["spans"] == [[0, 3, "name"]]
    assert views[2]["spans"] == [[0, 3, "name"]]
    assert len({view["view_group_id"] for view in views}) == 1
    assert [view["intake_row_id"] for view in views[1:]] == ["doc-7", "doc-7"]
    assert [view["sentence_ordinal"] for view in views[1:]] == [1, 2]
    assert [(view["source_start"], view["source_end"]) for view in views[1:]] == [
        (0, 12),
        (12, 24),
    ]
    assert {view["segmenter_model_revision"] for view in views[1:]} == {"test-revision"}
    assert len({view["source_text_sha256"] for view in views[1:]}) == 1


def test_expand_training_views_deduplicates_one_segment_paragraph() -> None:
    row = {
        "text": "One sentence with Ada.",
        "spans": [[18, 21, "name"]],
        "lang": "en",
        "sampling_weight": 0.25,
    }
    splitter = FixedSplitter([row["text"]])

    views = expand_training_views(row, splitter, long_view_alpha=0.4)

    assert len(views) == 1
    assert views[0]["text_view"] == "paragraph+segment"
    assert views[0]["sampling_weight"] == 0.25


def test_long_document_mass_scales_with_segments_and_splits_evenly_between_views() -> None:
    long_row = {
        "id": "long",
        "text": "First sentence. Second sentence.",
        "spans": [],
        "lang": "en",
    }
    short_row = {
        "id": "short",
        "text": "Only sentence.",
        "spans": [],
        "lang": "en",
    }
    long_views = expand_training_views(
        long_row,
        FixedSplitter(["First sentence. ", "Second sentence."]),
        long_view_alpha=0.5,
    )
    short_views = expand_training_views(
        short_row,
        FixedSplitter(["Only sentence."]),
        long_view_alpha=0.5,
    )

    assert [row["sampling_weight"] for row in long_views] == [1.0, 0.5, 0.5]
    assert sum(row["sampling_weight"] for row in long_views) == 2.0
    assert len(short_views) == 1
    assert short_views[0]["sampling_weight"] == 1.0


def test_segment_matched_default_omits_long_view_without_diluting_segments() -> None:
    row = {"text": "First. Second.", "spans": [], "lang": "en"}

    views = expand_training_views(row, FixedSplitter(["First. ", "Second."]))

    assert [view["text_view"] for view in views] == ["segment", "segment"]
    assert [view["sampling_weight"] for view in views] == [1.0, 1.0]
