from types import SimpleNamespace

import pytest

from scripts.pii_encoder_train import SpanDataset

TYPES = ("email", "locality", "admin_area", "person_name")
# Old "LOC" may become either place type; old "PER" is exactly person_name.
ACCEPTS = {"LOC": frozenset({"locality", "admin_area"}), "PER": frozenset({"person_name"})}


def _dataset(legacy=()):
    return SimpleNamespace(
        status_types=TYPES, status_accepts=ACCEPTS, legacy_outside_unknown_primary_types=frozenset(legacy)
    )


def _status(row, supervision="complete", label_space="v2", unknown=(), legacy=()):
    # Tokens: [<s>, 0-5, 6-10, 11-20, </s>]; spans must reach a non-special token.
    offsets = [(0, 0), (0, 5), (6, 10), (11, 20), (0, 0)]
    special = [a == b for a, b in offsets]
    ids = SpanDataset.tag_status_ids(
        _dataset(legacy), row, supervision, label_space, list(unknown), offsets, special
    )
    return {name: ids[f"status_{name}_id"] for name in TYPES}


def test_successor_rows_mark_present_and_absent_types():
    row = {"spans": [[0, 5, "email"], [11, 20, "person_name"]]}
    assert _status(row) == {"email": 2, "locality": 1, "admin_area": 1, "person_name": 2}


def test_an_unannotated_type_is_unknown_not_absent():
    row = {"spans": [[0, 5, "email"]]}
    assert _status(row, unknown=("locality",))["locality"] == 0


def test_an_ambiguous_old_span_hides_both_candidates():
    row = {"spans": [[6, 10, "LOC"], [11, 20, "PER"]]}
    status = _status(row, label_space="v1", unknown=("email",))
    # LOC could be either place type: neither is present or absent.
    assert status == {"email": 0, "locality": 0, "admin_area": 0, "person_name": 2}


def test_partial_rows_can_show_presence_but_never_absence():
    row = {"spans": [[11, 20, "person_name"]]}
    assert _status(row, supervision="annotated_spans_only") == {
        "email": 0,
        "locality": 0,
        "admin_area": 0,
        "person_name": 2,
    }


def test_old_rows_without_an_unknown_list_inherit_the_legacy_unknown_types():
    row = {"spans": [[11, 20, "PER"]]}
    assert _status(row, label_space="v1", legacy=("email",))["email"] == 0


def test_spans_that_reach_no_target_token_do_not_count():
    row = {"spans": [[25, 30, "email"]]}
    assert _status(row)["email"] == 1


def test_an_old_type_without_accepted_successors_is_rejected():
    with pytest.raises(ValueError, match="no accepted successor types"):
        _status({"spans": [[0, 5, "SSN"]]}, label_space="v1")
