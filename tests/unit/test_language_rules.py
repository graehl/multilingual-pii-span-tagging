"""Per-language annotation instructions render only for their own language."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pii_llm_label import (  # noqa: E402
    build_prompt,
    build_prompt_contract,
    load_language_rules,
    resolve_language_rules,
)

EXAMPLES = {
    "en": {"text": "Ann met Bob.", "labels": [{"t": "Ann", "type": "person_name", "n": 1}]},
    "ta": {"text": "முருகனின் வீடு.", "labels": [{"t": "முருகனின்", "type": "person_name", "n": 1}]},
}
TEMPLATE = "Rules:\n- general\n{language_rules}\n{example}\nInput:\n{text}\n"
RULES = {"ta": ["Copy the inflected word exactly."]}


def test_rules_render_only_for_their_language():
    _, tamil = build_prompt(
        TEMPLATE, EXAMPLES, "ta-IN", ["person_name"], "x", "json-seq", language_rules=RULES
    )
    _, english = build_prompt(
        TEMPLATE, EXAMPLES, "en", ["person_name"], "x", "json-seq", language_rules=RULES
    )
    assert "Additional rules for this input's language (ta):\n- Copy the inflected word exactly." in tamil
    assert "Additional rules" not in english and "{language_rules}" not in english


def test_resolution_never_falls_back_to_another_language():
    assert resolve_language_rules(RULES, "ta-LK") == ("ta", RULES["ta"])
    assert resolve_language_rules({"en": ["x"]}, "ta") == (None, [])


def test_placeholder_required_when_rules_given():
    with pytest.raises(ValueError, match="language_rules"):
        build_prompt(
            "{example}\n{text}", EXAMPLES, "ta", ["person_name"], "x", "json-seq", language_rules=RULES
        )


def test_template_placeholder_is_empty_without_rules():
    _, prompt = build_prompt(TEMPLATE, EXAMPLES, "ta", ["person_name"], "x", "json-seq")
    assert "{language_rules}" not in prompt and "Additional rules" not in prompt


def test_contract_records_rules_only_when_used():
    docs = [{"id": "a", "lang": "ta", "text": "x"}, {"id": "b", "lang": "en", "text": "y"}]
    with_rules = build_prompt_contract(
        TEMPLATE,
        EXAMPLES,
        docs,
        ["person_name"],
        "json-seq",
        "en",
        "m",
        language_rules=RULES,
        language_rules_sha256="abc",
    )
    assert with_rules["version"] == 8
    assert with_rules["language_rules"] == {"sha256": "abc", "languages": ["ta"]}
    assert [s["language_rules_language"] for s in with_rules["samples"]] == ["ta", None]
    plain = build_prompt_contract(TEMPLATE, EXAMPLES, docs, ["person_name"], "json-seq", "en", "m")
    assert "language_rules" not in plain and all("language_rules_language" not in s for s in plain["samples"])


def test_loader_rejects_empty_rule_lists(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"ta": []}))
    with pytest.raises(ValueError, match="nonempty list"):
        load_language_rules(path)
    path.write_text(json.dumps(RULES))
    assert load_language_rules(path) == RULES
