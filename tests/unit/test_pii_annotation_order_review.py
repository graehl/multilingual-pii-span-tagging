import hashlib
import json

import pytest

from scripts.pii_llm_label import validated_annotation_sequence


def test_reviewed_nested_occurrence_repairs_order_without_moving_other_spans():
    text = "leaders of Israel Rabbi Israel Lau"
    items = [
        {"i": 1, "t": "leaders of Israel", "type": "person_reference"},
        {"i": 2, "t": "Israel", "type": "admin_area"},
        {"i": 3, "t": "Rabbi Israel Lau", "type": "person_name"},
    ]
    raw = json.dumps(items)
    tags = {item["type"] for item in items}
    with pytest.raises(ValueError):
        validated_annotation_sequence(raw, text, tags)
    start = text.index("Israel")
    review = {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "repairs": [
            {
                "kind": "occurrence",
                "i": 2,
                "emitted": "Israel",
                "source_surface": "Israel",
                "start": start,
                "end": start + 6,
                "reason": "Country inside the leadership reference, not the person's given name.",
            }
        ],
    }
    spans, _ = validated_annotation_sequence(raw, text, tags, reviewed_alignment=review)
    assert spans == [
        {"start": 0, "end": 17, "label": "person_reference"},
        {"start": 11, "end": 17, "label": "admin_area"},
        {"start": 18, "end": len(text), "label": "person_name"},
    ]
    review["repairs"][0].update(start=text.rindex("Israel"), end=text.rindex("Israel") + 6)
    with pytest.raises(ValueError, match="violates source order"):
        validated_annotation_sequence(raw, text, tags, reviewed_alignment=review)
