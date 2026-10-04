"""The paper's two-view overlap rule and stable text keys, on hand-made neighbor records."""

from overlaplib import PAPER_LEXICAL_THRESHOLD, PAPER_SEMANTIC_THRESHOLD, passing_candidates, text_id


def record(*shared):
    return {
        "shared_top3": [
            {"train_id": f"eval:{i}", "chrf3_6_f1": f1, "semantic_cosine": cos}
            for i, (f1, cos) in enumerate(shared)
        ]
    }


def test_one_neighbor_must_pass_both_cuts():
    cuts = (PAPER_LEXICAL_THRESHOLD, PAPER_SEMANTIC_THRESHOLD)
    assert [c["train_id"] for c in passing_candidates(record((0.31, 0.88)), *cuts)] == ["eval:0"]
    # Semantic similarity alone, or lexical alone, never counts.
    assert passing_candidates(record((0.10, 0.99), (0.95, 0.50)), *cuts) == []


def test_text_ids_are_stable_and_distinct():
    assert text_id("a") == text_id("a") != text_id("b")
    assert len(text_id("a")) == 24
