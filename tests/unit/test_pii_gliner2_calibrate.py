import json

from scripts.pii_gliner2_calibrate import (
    epoch_batches,
    represented_width,
    selection_key,
    write_jsonl_atomic,
)


def test_write_jsonl_atomic_preserves_empty_candidate_rows(tmp_path):
    path = tmp_path / "candidates.jsonl"
    rows = [
        {"source_id": "a", "candidates": []},
        {
            "source_id": "b",
            "candidates": [{"start": 1, "end": 2, "label": "name", "confidence": 0.75}],
        },
    ]
    write_jsonl_atomic(path, rows)
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == rows


def test_epoch_batches_cover_every_row_once_within_width_tier():
    tiers = {8: [0, 1, 2], 16: [3, 4]}
    first = epoch_batches(tiers, batch_size=2, seed=17, epoch=1)
    second = epoch_batches(tiers, batch_size=2, seed=17, epoch=1)
    assert first == second
    assert sorted(index for _tier, batch in first for index in batch) == [0, 1, 2, 3, 4]
    assert all(all(index in tiers[tier] for index in batch) for tier, batch in first)


def test_represented_width_reads_all_entity_fields():
    class Record:
        structure_labels = [[1, [[[(2, 6)], (-1, -1), [(1, 2), (3, 9)]]]]]

    assert represented_width(Record()) == 7


def test_selection_key_prefers_p1_then_typed_then_character():
    point = {
        "threshold": 0.4,
        "importance_weighted_language_macro": {
            "overlap_p1": {"F1": 0.8},
            "overlap_typed_p9": {"F1": 0.7},
            "character_p1": {"F1": 0.9},
        },
    }
    assert selection_key(point) == (0.8, 0.7, 0.9, -0.4)
