"""Mechanical affix alignment repairs only unambiguous rewritten surfaces."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pii_llm_label import (  # noqa: E402
    mechanical_alignment,
    propose_affix_alignment,
    validated_annotation_sequence,
)

TAGS = {"organization", "person_name", "organization_reference", "location"}


def items(*surfaces):
    return [{"i": n, "t": t, "type": k} for n, (t, k) in enumerate(surfaces, 1)]


def test_tamil_dictionary_form_becomes_the_inflected_word():
    text = "அவர் நிலாவொளி ஆய்வகத்தில் பணிபுரிகிறார்."
    [repair] = propose_affix_alignment(items(("நிலாவொளி ஆய்வகம்", "organization")), text)
    assert repair["source_surface"] == "நிலாவொளி ஆய்வகத்தில்"
    assert repair["reason"] == "mechanical-affix-alignment-v1:stem-rewrite"


def test_hebrew_restored_article_is_dropped_after_absorbing_prefix():
    text = "הוא עובד במועצה המקומית."
    [repair] = propose_affix_alignment(items(("המועצה המקומית", "organization_reference")), text)
    assert repair["source_surface"] == "מועצה המקומית"


def test_literal_surfaces_and_ambiguous_candidates_are_left_alone():
    text = "முருகனின் நண்பர் முருகனுக்கு எழுதினார்."
    assert propose_affix_alignment(items(("முருகனின்", "person_name")), text) == []
    assert propose_affix_alignment(items(("முருகன்", "person_name")), text) == []


def test_short_stems_and_long_extensions_are_not_repaired():
    assert propose_affix_alignment(items(("Ana", "person_name")), "Anna came.") == []
    assert propose_affix_alignment(items(("Kasim", "person_name")), "Kasimirowiczowie came.") == []


def test_alignment_record_validates_through_the_reviewed_sequence():
    text = "அவர் நிலாவொளி ஆய்வகத்தில் பணிபுரிகிறார்."
    raw = json.dumps(items(("நிலாவொளி ஆய்வகம்", "organization")), ensure_ascii=False)
    alignment = mechanical_alignment(raw, text)
    preds, details = validated_annotation_sequence(raw, text, TAGS, reviewed_alignment=alignment)
    assert [(p["start"], p["end"]) for p in preds] == [(5, 5 + len("நிலாவொளி ஆய்வகத்தில்"))]
    assert details["alignment_repairs"][0]["kind"] == "deinflection"


def test_arabic_restored_alif_after_lam_keeps_the_written_lam():
    text = "قدم بلاغا للنائب العام أمس."
    [repair] = propose_affix_alignment(items(("النائب العام", "person_reference")), text)
    assert repair["source_surface"] == "لنائب العام"
    assert repair["reason"].endswith("absorbed-alif")


def test_unique_surfaces_out_of_order_are_put_in_source_order():
    text = "Ana met Bob at Acme."
    raw = json.dumps(items(("Acme", "organization"), ("Ana", "person_name")), ensure_ascii=False)
    alignment = mechanical_alignment(raw, text)
    assert alignment["order"] == "unique-source-position" and alignment["repairs"] == []
    preds, details = validated_annotation_sequence(raw, text, TAGS, reviewed_alignment=alignment)
    assert [p["label"] for p in preds] == ["person_name", "organization"]
    assert details["alignment_order"] == "unique-source-position"


def test_repeated_surfaces_are_not_reordered():
    text = "Acme hired Ana; Acme grew."
    raw = json.dumps(items(("Ana", "person_name"), ("Acme", "organization")), ensure_ascii=False)
    assert mechanical_alignment(raw, text) is None
