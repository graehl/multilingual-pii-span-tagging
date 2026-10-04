import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pii_public_mixture import apply_language_caps  # noqa: E402


def mixture():
    rows = [
        {"lang": "de", "sampling_branch": "human_gold", "sampling_weight": 0.30},
        {"lang": "en", "sampling_branch": "human_gold", "sampling_weight": 0.20},
        {"lang": "hi", "sampling_branch": "base", "sampling_weight": 0.20},
        {"lang": "hi", "sampling_branch": "base", "sampling_weight": 0.10},
        {"lang": "en", "sampling_branch": "base", "sampling_weight": 0.15},
        {"lang": "es", "sampling_branch": "base", "sampling_weight": 0.05},
    ]
    return rows


def share(rows, language):
    return math.fsum(r["sampling_weight"] for r in rows if r["lang"] == language)


def test_capped_language_moves_its_base_mass_to_other_base_rows():
    rows = mixture()
    report = apply_language_caps(rows, {"default": 1.0, "languages": {"hi": 0.08}})
    assert share(rows, "hi") == pytest.approx(0.08)
    assert math.fsum(r["sampling_weight"] for r in rows if r["sampling_branch"] == "base") == pytest.approx(0.5)
    assert share(rows, "de") == pytest.approx(0.30)  # gold untouched
    # The freed 0.22 goes to en and es base rows in proportion to their mass (3:1).
    assert share(rows, "es") == pytest.approx(0.05 * (1 + 0.22 / 0.20))
    assert report["capped"]["hi"]["after"] == pytest.approx(0.08)


def test_a_cap_exceeded_by_gold_alone_is_reported_and_takes_no_freed_mass():
    rows = mixture()
    report = apply_language_caps(rows, {"default": 1.0, "languages": {"hi": 0.08, "de": 0.20}})
    assert report["unattainable_from_gold"] == {"de": pytest.approx(0.30)}
    assert share(rows, "de") == pytest.approx(0.30)


def test_redistribution_can_push_another_language_to_its_cap():
    rows = mixture()
    apply_language_caps(rows, {"default": 0.40, "languages": {"hi": 0.08}})
    for language in ("de", "en", "es", "hi"):
        assert share(rows, language) <= {"hi": 0.08}.get(language, 0.40) + 1e-9
    assert math.fsum(r["sampling_weight"] for r in rows) == pytest.approx(1.0)
