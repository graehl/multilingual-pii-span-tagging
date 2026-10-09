import json
from pathlib import Path

import pytest

from scripts.pii_gliner2_label_transfer import load_transfer, translate_records

SPEC = (
    Path(__file__).resolve().parents[2] / "research/pii/frontier/evidence/gliner2-ont3-label-transfer-v1.json"
)


def test_donors_follow_existing_cross_ontology_mapping():
    spec = load_transfer(SPEC)
    assert len(spec["labels"]) == 31
    mapping = {label: entry["prompt"] for label, entry in spec["labels"].items()}
    rows = [{"text": "Jane", "entities": {"person_name": ["Jane"], "email": []}}]
    translated = translate_records(rows, mapping)
    assert translated == [{"text": "Jane", "entities": {"full_name": ["Jane"], "email": []}}]
    reverse = {v: k for k, v in mapping.items()}
    assert translate_records(translated, reverse) == rows


@pytest.mark.parametrize(
    "target,donor", [("device_identifier", "account_id"), ("record_identifier", "license_number")]
)
def test_invalid_donor_analogy_is_rejected(tmp_path, target, donor):
    spec = json.loads(SPEC.read_text())
    spec["labels"][target]["donors"] = [{"name": donor, "role": "allowed", "weight": 1}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="not an allowed source mapping"):
        load_transfer(path)
