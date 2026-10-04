import itertools
import json
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.pii_name_annotation_qc import (
    NameAnnotationQc,
    audit_native_arrays,
    audit_records,
    load_config,
)
from scripts.pii_subclass import load_subclass_spec

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts" / "pii_name_annotation_qc_profiles_v2.json"
GRAMMAR_PROPOSAL = ROOT / "scripts" / "pii_name_component_grammars_v1.json"
LEGACY_CONFIG = ROOT / "scripts" / "pii_name_annotation_qc_profiles_v1.json"
SUBCLASS_SPEC = ROOT / "scripts" / "pii_subclass_families_v4.json"


def test_infer_person_names_respects_name_list_grammar():
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name", max_name_gap_chars=5)
    result = qc.adjusted_inference_fields(
        row_id="list",
        text="Tim Robinson, Sally Williams",
        language="en",
        predictions=[
            {"start": 0, "end": 3, "label": "given_name"},
            {"start": 4, "end": 12, "label": "family_name"},
            {"start": 14, "end": 19, "label": "given_name"},
            {"start": 20, "end": 28, "label": "family_name"},
        ],
    )
    assert [(s["start"], s["end"], s["label"]) for s in result["preds"]] == [
        (0, 12, "person_name"),
        (14, 28, "person_name"),
    ]


@pytest.mark.parametrize("budget,expected", [(5, [(0, 5), (8, 9)]), (6, [(0, 9)])])
def test_infer_person_name_gap_is_total_not_per_link(budget, expected):
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name", max_name_gap_chars=budget)
    result = analyze("A   B   C", [(0, 1, "given_name"), (4, 5, "family_name"), (8, 9, "family_name")], qc=qc)
    assert [(s["start"], s["end"]) for s in result["candidate"]["primary_spans"]] == expected


def test_infer_extends_existing_name_and_preserves_component_tokens():
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name")
    result = analyze("Tim A Robinson", [(0, 5, "person_name"), (6, 14, "family_name")], qc=qc)
    assert [(s["start"], s["end"]) for s in result["candidate"]["primary_spans"]] == [(0, 14)]
    assert result["candidate"]["subclass_spans"] == [
        {
            "start": 6,
            "end": 14,
            "family": "name_component",
            "value": "family_name",
            "carrier_start": 0,
            "carrier_end": 14,
            "type": "person_name",
        }
    ]


def test_infer_complete_given_middle_family_sequence():
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name")
    result = analyze(
        "Tim A Robinson", [(0, 3, "given_name"), (4, 5, "middle_name"), (6, 14, "family_name")], qc=qc
    )
    assert result["candidate"]["primary_spans"] == [{"start": 0, "end": 14, "label": "person_name"}]


def test_infer_keeps_existing_refinements_and_other_families():
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name")
    parts = [
        {"start": 0, "end": 3, "family": "name_component", "value": "given_name"},
        {"start": 4, "end": 12, "family": "name_component", "value": "family_name"},
        {"start": 0, "end": 12, "family": "test_other_family", "value": "unchanged"},
    ]
    result = qc.adjusted_inference_fields(
        row_id="nested", text="Tim Robinson", language="en", predictions=[], subclass_annotations=parts
    )
    assert result["preds"] == [{"start": 0, "end": 12, "label": "person_name"}]
    assert result["subclass_spans"] == [
        {**parts[0], "carrier_start": 0, "carrier_end": 12, "type": "person_name"},
        {**parts[1], "carrier_start": 0, "carrier_end": 12, "type": "person_name"},
        parts[2],
    ]


@pytest.mark.parametrize("grammar,expected", [(True, [(0, 12), (14, 28)]), (False, [(0, 28)])])
def test_infer_grammar_switch(grammar, expected):
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name", apply_name_grammar=grammar)
    result = analyze(
        "Tim Robinson, Sally Williams",
        [(0, 3, "given_name"), (4, 12, "family_name"), (14, 19, "given_name"), (20, 28, "family_name")],
        qc=qc,
    )
    assert [(s["start"], s["end"]) for s in result["candidate"]["primary_spans"]] == expected


def test_infer_cannot_bridge_other_predicted_entity():
    qc = NameAnnotationQc(load_config(CONFIG), mode="infer-person-name")
    result = analyze(
        "Tim X Bell", [(0, 3, "given_name"), (4, 5, "organization"), (6, 10, "family_name")], qc=qc
    )
    assert result["candidate"]["primary_spans"] == [
        {"start": 0, "end": 3, "label": "person_name"},
        {"start": 4, "end": 5, "label": "organization"},
        {"start": 6, "end": 10, "label": "person_name"},
    ]


def test_infer_records_production_cli_does_not_use_reference(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "one",
                "bcp47": "en",
                "text": "Tim Robinson",
                "preds": [
                    {"start": 0, "end": 3, "label": "given_name"},
                    {"start": 4, "end": 12, "label": "family_name"},
                ],
                "spans": [[0, 12, "organization"]],
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pii_name_annotation_qc.py"),
            "records",
            "--input",
            str(source),
            "--output",
            str(output),
            "--mode",
            "infer-person-name",
            "--annotations-field",
            "preds",
            "--output-view",
            "predictions",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads(output.read_text())
    assert row["preds"] == [{"start": 0, "end": 12, "label": "person_name"}]
    assert len(row["subclass_spans"]) == 2
    assert row["name_oracle"]["mode"] == "infer-person-name"


@pytest.mark.parametrize("defect", [None, "surface", "length", "conflict", "parent_type", "ignored_q"])
def test_raw_name_components_survive_unrelated_parser_quarantine(tmp_path, defect):
    source, raw, output = (tmp_path / name for name in ("input.jsonl", "raw.jsonl", "output.jsonl"))
    source.write_text(
        json.dumps(
            {"id": "one", "bcp47": "en", "text": "Tim Robinson", "preds": [], "annotation_banned": True}
        )
        + "\n"
    )
    parts = [
        {
            "start": 0,
            "end": 3,
            "t": "Tim",
            "family": "name_component",
            "value": "given_name",
            "carrier_start": 0,
            "carrier_end": 3,
            "carrier_type": "person_name",
        },
        {
            "start": 4,
            "end": 12,
            "t": "Robinson",
            "family": "name_component",
            "value": "family_name",
            "carrier_start": 4,
            "carrier_end": 12,
            "carrier_type": "person_name",
        },
        {"family": "invalid_unrelated_family"},
    ]
    if defect == "surface":
        parts[0]["t"] = "Jim"
    if defect == "conflict":
        parts.append({**parts[0], "value": "family_name"})
    if defect == "parent_type":
        parts[0]["carrier_type"] = "person_reference"
        parts[1]["carrier_type"] = "person_reference"
    if defect == "ignored_q":
        parts.append({"family": "name_component", "value": "Q", "t": "unscored"})
    raw.write_text(
        json.dumps(
            {
                "id": "one",
                "response": {
                    "choices": [
                        {
                            "finish_reason": "length" if defect == "length" else "stop",
                            "message": {"content": json.dumps({"subclass_spans": parts})},
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    command = [
        sys.executable,
        "-m",
        "scripts.pii_name_annotation_qc",
        "records",
        "--input",
        str(source),
        "--output",
        str(output),
        "--mode",
        "infer-person-name",
        "--annotations-field",
        "preds",
        "--output-view",
        "predictions",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    assert json.loads(output.read_text())["preds"] == []
    repaired_output = tmp_path / "repaired.jsonl"
    command[command.index("--output") + 1] = str(repaired_output)
    subprocess.run(
        [*command, "--raw-name-input", str(raw)], check=True, capture_output=True, text=True, cwd=ROOT
    )
    row = json.loads(repaired_output.read_text())
    if defect in {"surface", "length", "conflict"}:
        assert row["preds"] == []
        assert row["subclass_spans"] == []
        assert row["raw_name_recovery"]["rejected"] is not None
        return
    assert row["preds"] == [{"start": 0, "end": 12, "label": "person_name"}]
    assert [
        (p["start"], p["end"], p["value"], p["carrier_start"], p["carrier_end"])
        for p in row["subclass_spans"]
    ] == [(0, 3, "given_name", 0, 12), (4, 12, "family_name", 0, 12)]
    assert row["raw_name_recovery"]["components"] == 2


def test_override_name_kinds_does_not_use_supplied_roles():
    qc = NameAnnotationQc(load_config(CONFIG), override_name_kinds=True)
    arguments = dict(
        row_id="override",
        text="Tim Robinson",
        language="en",
        predictions=[{"start": 0, "end": 12, "label": "person_name"}],
    )
    original = [
        {"start": 0, "end": 3, "family": "name_component", "value": "family_name"},
        {"start": 4, "end": 12, "family": "name_component", "value": "given_name"},
    ]
    assert qc.adjusted_inference_fields(
        **arguments, subclass_annotations=original
    ) == qc.adjusted_inference_fields(**arguments)


def explicit_v4_payload():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    proposal = json.loads(GRAMMAR_PROPOSAL.read_text(encoding="utf-8"))
    payload["version"] = 4
    payload["name_kind_model"] = {
        "config_path": "name-kind.config.json",
        "candidate_scope": "component subspans permitted by the selected language grammar",
        "role_components": {"given": "given_name", "family": "family_name"},
        "score_transform": "centered_log_odds",
        "score_weight": 1.0,
    }
    for key in (
        "symbols",
        "ignored_values",
        "name_atom_joiners",
        "sequence_pattern",
        "sequence_pattern_semantics",
    ):
        payload[key] = proposal[key]
    payload["grammars"] = {
        name: {key: value for key, value in grammar.items() if key != "legacy_component_order"}
        for name, grammar in proposal["grammars"].items()
    }
    return payload


class FakeNameKindDeployment:
    config = {"roles": ["given", "family"]}

    def __init__(self, scores=None):
        self.scores = scores or {}

    def infer(self, surfaces, language):
        del language
        return surfaces, [self.scores.get(surface, [0.0, 0.0]) for surface in surfaces]


def engine(*, given=(), family=()) -> NameAnnotationQc:
    return NameAnnotationQc(
        load_config(CONFIG),
        given_lexicons=given,
        family_lexicons=family,
    )


def analyze(
    text,
    annotations,
    *,
    language="en",
    grammar_language=None,
    subclasses=None,
    qc=None,
):
    return (qc or engine()).analyze(
        row_id="row",
        text=text,
        language=language,
        grammar_language=grammar_language,
        annotations=annotations,
        subclass_annotations=subclasses,
    )


def components(result):
    return [
        (item["span"], item["value"], item["text"], item["status"])
        for item in result["proposals"][0]["components"]
    ]


def test_v2_config_exposes_grammar_and_matched_language_rule():
    result = analyze(
        "William Tambellini",
        [{"span": [0, 18], "class": "person_name"}],
        language="en-US",
    )

    assert result["profile"] == "given_first"
    assert result["grammar"] == "given_first"
    assert result["language_rule"] == "en"

    default = engine().profile("und")
    assert default.name == "und"
    assert default.component_order == "any"

    direct = engine().profile("family_first")
    assert direct.name == "family_first"
    assert direct.grammar_script is None


def test_explicit_postprocessor_regex_matches_current_component_sequences():
    proposal = json.loads(GRAMMAR_PROPOSAL.read_text(encoding="utf-8"))
    pattern = re.compile(proposal["sequence_pattern"])
    accepted = {
        "G",
        "M",
        "F",
        "GF",
        "GMF",
        "FG",
        "FGM",
    }

    assert {
        "".join(sequence)
        for length in range(5)
        for sequence in itertools.product("GMF", repeat=length)
        if pattern.fullmatch("".join(sequence))
    } == accepted
    assert set(proposal["grammars"]) == {
        "und",
        "given_first",
        "family_first",
        "ko_hangul",
        "zh_han",
    }
    assert proposal["grammars"]["und"]["order_preference"] is None


def test_v4_config_loads_explicit_independent_grammar_fields(tmp_path: Path):
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(explicit_v4_payload()), encoding="utf-8")

    config = load_config(path)
    profile = NameAnnotationQc(
        config,
        name_kind_deployment=FakeNameKindDeployment(),
    ).profile("und")

    assert config.sequence_pattern == "(?:G|M|F|GM?F|FGM?)"
    assert config.component_symbols == {
        "given_name": "G",
        "middle_name": "M",
        "family_name": "F",
    }
    assert config.name_kind_model["config_path"] == "name-kind.config.json"
    assert config.name_atom_joiner_ranges == ((39, 39), (45, 45), (700, 700), (8217, 8217))
    assert profile.order_preference is None
    assert profile.partition_policy == "all_family_prefix_and_suffix_partitions"


def test_v4_sequence_pattern_filters_generated_candidates(tmp_path: Path):
    payload = explicit_v4_payload()
    payload["sequence_pattern"] = "F"
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    qc = NameAnnotationQc(
        load_config(path),
        name_kind_deployment=FakeNameKindDeployment(),
    )

    result = analyze(
        "William Tambellini",
        [{"span": [0, 18], "class": "person_name"}],
        qc=qc,
    )

    assert result["proposals"][0]["assignment"] is None
    assert result["proposals"][0]["components"] == []
    assert result["findings"][-1]["reason"] == "no_legal_component_sequence"


@pytest.mark.parametrize("override", [False, True])
def test_neural_standalone_name_needs_no_dictionary(tmp_path: Path, override: bool):
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(explicit_v4_payload()), encoding="utf-8")
    qc = NameAnnotationQc(
        load_config(path),
        name_kind_deployment=FakeNameKindDeployment({"Ismayilova": [-4.0, 0.0]}),
        override_name_kinds=override,
    )
    result = qc.adjusted_inference_fields(
        row_id="standalone",
        text="Ismayilova",
        language="en",
        predictions=[{"start": 0, "end": 10, "label": "person_name"}],
    )
    assert [(s["start"], s["end"], s["value"]) for s in result["subclass_spans"]] == [(0, 10, "family_name")]
    assert result["preds"] == [{"start": 0, "end": 10, "label": "person_name"}]


def test_v4_name_kind_scores_share_the_explicit_candidate_grammar(tmp_path: Path):
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(explicit_v4_payload()), encoding="utf-8")
    qc = NameAnnotationQc(
        load_config(path),
        name_kind_deployment=FakeNameKindDeployment(
            {
                "Smith": [-4.0, 0.0],
                "John": [0.0, -4.0],
            }
        ),
    )

    result = analyze(
        "Smith John",
        [{"span": [0, 10], "class": "person_name"}],
        qc=qc,
    )

    assert result["proposals"][0]["assignment"]["order"] == "family_first"
    assert [item[1] for item in components(result)] == ["family_name", "given_name"]
    assert any(
        item.startswith("name_kind:family_name:") for item in result["proposals"][0]["assignment"]["evidence"]
    )


def test_expected_components_assign_contiguous_subspans_without_redeciding_roles():
    assignment = engine().assign_expected_components(
        text="John Alexander Smith",
        language="en",
        values=("given_name", "middle_name", "family_name"),
    )

    assert assignment["status"] == "assigned"
    assert [
        (item["start"], item["end"], item["value"], item["text"]) for item in assignment["components"]
    ] == [
        (0, 4, "given_name", "John"),
        (5, 14, "middle_name", "Alexander"),
        (15, 20, "family_name", "Smith"),
    ]


def test_expected_components_honor_supplied_family_first_order():
    assignment = engine().assign_expected_components(
        text="Smith John",
        language="en",
        values=("family_name", "given_name"),
    )

    assert assignment["status"] == "assigned"
    assert [item["value"] for item in assignment["components"]] == [
        "family_name",
        "given_name",
    ]


def test_expected_components_assign_compact_han_subspans():
    assignment = engine().assign_expected_components(
        text="王小明",
        language="zh",
        values=("family_name", "given_name"),
    )

    assert assignment["status"] == "assigned"
    assert [
        (item["start"], item["end"], item["value"], item["text"]) for item in assignment["components"]
    ] == [
        (0, 1, "family_name", "王"),
        (1, 3, "given_name", "小明"),
    ]


def test_expected_components_can_use_the_annotation_sequence_grammar():
    grammar = load_subclass_spec(SUBCLASS_SPEC).family_by_name["name_component"].sequence_grammar
    assert grammar is not None
    assignment = engine().assign_expected_components(
        text="Lászlóné BALÁZS",
        language="en",
        values=("middle_name", "family_name"),
        sequence_pattern=grammar.pattern,
    )

    assert assignment["status"] == "assigned"
    assert [item["value"] for item in assignment["components"]] == [
        "middle_name",
        "family_name",
    ]


def test_expected_q_only_component_skips_empty_name_kind_batch(tmp_path: Path):
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(explicit_v4_payload()), encoding="utf-8")
    grammar = load_subclass_spec(SUBCLASS_SPEC).family_by_name["name_component"].sequence_grammar
    assert grammar is not None
    assignment = NameAnnotationQc(
        load_config(path),
        name_kind_deployment=FakeNameKindDeployment(),
    ).assign_expected_components(
        text="Dr.",
        language="en",
        values=("Q",),
        sequence_pattern=grammar.pattern,
    )

    assert assignment["status"] == "assigned"
    assert [item["value"] for item in assignment["components"]] == ["Q"]


def test_language_section_can_select_grammar_and_override_each_settings_section(
    tmp_path: Path,
):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["languages"]["en-GB"] = {
        "grammar": "family_first",
        "limits": {"max_name_atoms": 4},
        "scoring": {"minimum_assignment_margin": 0.25},
        "lexicons": {"honorifics": ["lord"]},
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = NameAnnotationQc(load_config(path)).profile("en_GB")

    assert profile.name == "family_first"
    assert profile.language_rule == "en-gb"
    assert profile.max_name_atoms == 4
    assert profile.minimum_assignment_margin == 0.25
    assert profile.honorifics == {"lord"}
    assert profile.max_join_gap_chars == 4


def test_grammar_script_config_rejects_unknown_range_name(tmp_path: Path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["languages"]["en"]["grammar_script"] = "Unknown"
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown grammar_script"):
        load_config(path)


def test_grammar_script_fallback_matches_native_for_complete_adjusted_carrier():
    matched = analyze(
        "Nagy Anna Maria",
        [{"span": [0, 15], "class": "person_name"}],
        language="und",
        grammar_language="hu",
    )
    assert matched["profile"] == "family_first"
    assert matched["grammar_script"] == "Latin"
    assert matched["proposals"][0]["grammar"] == "family_first"
    assert [item[1] for item in components(matched)] == [
        "family_name",
        "given_name",
        "middle_name",
    ]

    mismatched = analyze(
        "王 小明",
        [{"span": [0, 4], "class": "person_name"}],
        language="und",
        grammar_language="hu",
    )
    assert mismatched["profile"] == "family_first"
    assert mismatched["proposals"][0]["grammar"] == "und"
    assert [item[1] for item in components(mismatched)] == ["given_name", "family_name"]

    excluded = analyze(
        "Nagy 2 Anna",
        [{"span": [0, 11], "class": "person_name"}],
        language="und",
        grammar_language="hu",
    )
    assert excluded["proposals"][0]["grammar"] == "und"


def test_grammar_script_gate_reads_only_the_boundary_adjusted_name_span():
    text = "王Nagy Anna"
    adjusted = analyze(
        text,
        [{"span": [1, len(text)], "class": "person_name"}],
        language="und",
        grammar_language="hu",
    )
    assert adjusted["proposals"][0]["grammar"] == "family_first"

    includes_foreign_codepoint = analyze(
        text,
        [{"span": [0, len(text)], "class": "person_name"}],
        language="und",
        grammar_language="hu",
    )
    assert includes_foreign_codepoint["proposals"][0]["grammar"] == "und"


def test_hash_bound_v1_config_remains_readable():
    profile = NameAnnotationQc(load_config(LEGACY_CONFIG)).profile("ko")

    assert profile.name == "ko_hangul"
    assert profile.language_rule == "ko"


def test_native_name_carrier_keeps_score_and_gets_components():
    text = "My name is William Tambellini."
    result = analyze(
        text,
        [
            {
                "span": [11, 29],
                "class": "person_name",
                "text": "William Tambellini",
                "score": 0.9208,
            },
        ],
    )

    assert result["proposals"][0]["carrier"] == {
        "start": 11,
        "end": 29,
        "label": "person_name",
        "text": "William Tambellini",
        "score": 0.9208,
    }
    assert components(result) == [
        ([11, 18], "given_name", "William", "missing"),
        ([19, 29], "family_name", "Tambellini", "missing"),
    ]
    assert result["summary"]["finding_counts"] == {"missing_name_component_detail": 2}
    assert result["candidate"]["primary_spans"] == [result["proposals"][0]["carrier"]]
    assert result["candidate"]["status"] == "proposed"


def test_existing_fine_native_spans_are_checked_then_coalesced():
    text = "My name is William Tambellini."
    result = analyze(
        text,
        [
            {"span": [11, 18], "class": "given_name", "text": "William", "score": 0.9},
            {
                "span": [19, 29],
                "class": "family_name",
                "text": "Tambellini",
                "score": 0.8,
            },
        ],
    )

    assert [item[-1] for item in components(result)] == ["agree", "agree"]
    assert result["summary"]["finding_counts"] == {"oversplit_person_name": 1}
    assert result["proposals"][0]["assignment"]["evidence"] == [
        "language_order:given_first",
        "source_agrees:given_name",
        "source_agrees:family_name",
    ]


def test_full_name_has_one_consecutive_middle_component():
    text = "My name is Lewis Francis William Stott."
    start = text.index("Lewis")
    end = text.index(".")
    result = analyze(text, [{"span": [start, end], "class": "person_name"}])

    assert components(result) == [
        ([start, start + 5], "given_name", "Lewis", "missing"),
        (
            [text.index("Francis"), text.index("William") + len("William")],
            "middle_name",
            "Francis William",
            "missing",
        ),
        ([text.index("Stott"), end], "family_name", "Stott", "missing"),
    ]


def test_comma_inversion_overrides_given_first_language_prior():
    text = "Stott, Lewis Francis William testified."
    result = analyze(text, [{"span": [0, 28], "class": "person_name"}])

    assert result["proposals"][0]["assignment"]["kind"] == "comma_inversion"
    assert components(result) == [
        ([0, 5], "family_name", "Stott", "missing"),
        ([7, 12], "given_name", "Lewis", "missing"),
        ([13, 28], "middle_name", "Francis William", "missing"),
    ]


def test_comma_inversion_still_coalesces_component_spans():
    text = "Stott, Lewis testified."
    result = analyze(
        text,
        [
            {"span": [0, 5], "class": "family_name"},
            {"span": [7, 12], "class": "given_name"},
        ],
    )

    assert [item["carrier"]["text"] for item in result["proposals"]] == ["Stott, Lewis"]
    assert result["proposals"][0]["assignment"]["kind"] == "comma_inversion"


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        # A one-word name before a comma once absorbed the next name as an
        # inverted "Last, First", which merged comma-separated lists of names.
        ("Stott, Lewis Francis William testified.", [(0, 5), (7, 28)]),
        # A split name is a tagging error for the decoder's adjacent-span
        # merge to repair, not for name-kind postprocessing.
        ("My name is William Tambellini.", [(11, 18), (19, 29)]),
    ],
)
def test_whole_names_are_never_joined_or_extended(text, spans):
    result = analyze(text, [{"span": list(span), "class": "person_name"} for span in spans])

    assert [(s["start"], s["end"]) for s in result["candidate"]["primary_spans"]] == spans


def test_compact_hangul_name_uses_language_profile():
    result = analyze("김민수", [{"span": [0, 3], "class": "person_name"}], language="ko")

    assert result["profile"] == "ko_hangul"
    assert components(result) == [
        ([0, 1], "family_name", "김", "missing"),
        ([1, 3], "given_name", "민수", "missing"),
    ]
    assert not result["proposals"][0]["review_required"]


def test_family_lexicon_selects_compound_han_family_name(tmp_path: Path):
    family = tmp_path / "family.txt"
    family.write_text("欧阳\n", encoding="utf-8")
    qc = engine(family=[family])

    result = analyze(
        "欧阳娜娜",
        [{"span": [0, 4], "class": "person_name"}],
        language="zh",
        qc=qc,
    )

    assert components(result) == [
        ([0, 2], "family_name", "欧阳", "missing"),
        ([2, 4], "given_name", "娜娜", "missing"),
    ]
    assert "family_lexicon:欧阳" in result["proposals"][0]["assignment"]["evidence"]


def test_lexicons_can_override_language_order_without_rewriting_labels(tmp_path: Path):
    given = tmp_path / "given.txt"
    family = tmp_path / "family.txt"
    given.write_text("Anastasiya\n", encoding="utf-8")
    family.write_text("Tyunyayeva\n", encoding="utf-8")
    qc = engine(given=[given], family=[family])
    text = "Tyunyayeva Anastasiya Ivanovna"

    result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}], qc=qc)

    assert result["proposals"][0]["assignment"]["kind"].startswith("family_first:")
    assert components(result) == [
        ([0, 10], "family_name", "Tyunyayeva", "missing"),
        ([11, 21], "given_name", "Anastasiya", "missing"),
        ([22, 30], "middle_name", "Ivanovna", "missing"),
    ]


def test_annotation_disagreement_requires_review_and_is_not_self_confirming():
    text = "William Tambellini"
    subclasses = [
        {"start": 0, "end": 7, "family": "name_component", "value": "family_name"},
        {"start": 8, "end": 18, "family": "name_component", "value": "given_name"},
    ]

    result = analyze(
        text,
        [{"span": [0, 18], "class": "person_name"}],
        subclasses=subclasses,
    )

    assert [item["value"] for item in result["proposals"][0]["components"]] == [
        "given_name",
        "family_name",
    ]
    assert result["summary"]["finding_counts"] == {"name_component_disagreement": 2}
    assert result["candidate"]["status"] == "review_required"


def test_two_family_lexicon_hits_form_one_family_subspan(tmp_path: Path):
    given = tmp_path / "given.txt"
    family = tmp_path / "family.txt"
    given.write_text("Canòlic\n", encoding="utf-8")
    family.write_text("Mingorance\nCairat\n", encoding="utf-8")
    qc = engine(given=[given], family=[family])
    text = "Canòlic Mingorance Cairat"

    result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}], qc=qc)

    assert components(result) == [
        ([0, 7], "given_name", "Canòlic", "missing"),
        ([8, 25], "family_name", "Mingorance Cairat", "missing"),
    ]


def test_blocking_punctuation_prevents_coalescing():
    result = analyze(
        "Alice / Bob",
        [
            {"span": [0, 5], "class": "person_name"},
            {"span": [8, 11], "class": "person_name"},
        ],
    )

    assert [item["carrier"]["text"] for item in result["proposals"]] == ["Alice", "Bob"]
    assert "oversplit_person_name" not in result["summary"]["finding_counts"]
    assert result["summary"]["review_required"] == 2


def test_comma_between_two_complete_names_does_not_coalesce():
    text = "Lətif Hüseynov, Vasilka Sancin"
    result = analyze(
        text,
        [
            {"span": [0, 14], "class": "person_name"},
            {"span": [16, 30], "class": "person_name"},
        ],
    )

    assert [item["carrier"]["text"] for item in result["proposals"]] == [
        "Lətif Hüseynov",
        "Vasilka Sancin",
    ]


def test_exact_person_reference_relabel_suppresses_legacy_name_carrier():
    text = "The court heard the applicant."
    start = text.index("the applicant")
    end = start + len("the applicant")
    result = analyze(
        text,
        [
            {"start": start, "end": end, "type": "person_name"},
            {"start": start, "end": end, "label": "person_reference"},
        ],
    )

    assert result["proposals"] == []
    assert result["candidate"]["primary_spans"] == [{"start": start, "end": end, "label": "person_reference"}]


def test_unresolved_edge_initial_is_not_guessed():
    text = "Mr H. de Suremain"
    result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}])

    assert components(result) == [
        ([0, 2], "Q", "Mr", "missing"),
        ([6, 17], "family_name", "de Suremain", "missing"),
    ]
    assert result["proposals"][0]["review_required"]
    assert result["findings"][-1]["reason"] == "unresolved_edge_initial"


@pytest.mark.parametrize("retain", [False, True])
def test_given_initial_retention_requires_middle_and_spelled_family(tmp_path: Path, retain: bool):
    payload = explicit_v4_payload()
    payload["defaults"]["scoring"]["retain_given_initial_with_middle"] = retain
    payload["defaults"]["scoring"]["weights"]["middle_presence"] = 2
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    qc = NameAnnotationQc(
        load_config(path),
        name_kind_deployment=FakeNameKindDeployment({"Suremain": [-4.0, 4.0]}),
    )
    text = "H. A. Suremain"
    result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}], qc=qc)
    actual = [(part[1], part[2]) for part in components(result)]
    expected = [("middle_name", "A"), ("family_name", "Suremain")]
    if retain:
        expected.insert(0, ("given_name", "H"))
    assert actual == expected

    text = "Mr H. de Suremain"
    result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}], qc=qc)
    assert not any(part[1] == "given_name" and part[2] == "H" for part in components(result))


@pytest.mark.parametrize("value", [1, "true", None])
def test_given_initial_retention_rejects_non_boolean_config(tmp_path: Path, value):
    payload = explicit_v4_payload()
    payload["defaults"]["scoring"]["retain_given_initial_with_middle"] = value
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="retain_given_initial_with_middle must be a boolean"):
        load_config(path)


def test_given_initial_retention_is_language_scoped_and_preserves_review_flag(tmp_path: Path):
    payload = explicit_v4_payload()
    payload["languages"]["ru"]["scoring"] = {
        "retain_given_initial_with_middle": True,
        "weights": {"middle_presence": 2},
    }
    path = tmp_path / "profiles-v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)
    qc = NameAnnotationQc(config, name_kind_deployment=FakeNameKindDeployment())
    assert qc.profile("ru-RU").retain_given_initial_with_middle
    assert not qc.profile("en").retain_given_initial_with_middle
    assert not qc.profile("und").retain_given_initial_with_middle

    for text in ("Н. Н. Г.", "Н. Н. de Г."):
        result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}], language="ru", qc=qc)
        assert not any(part[1] == "given_name" and part[2] == "Н" for part in components(result))

    config.language_overrides["ru"]["minimum_assignment_margin"] = 1e6
    qc = NameAnnotationQc(config, name_kind_deployment=FakeNameKindDeployment())
    text = "Н. Н. Грунский"
    result = analyze(text, [{"span": [0, len(text)], "class": "person_name"}], language="ru", qc=qc)
    assert result["proposals"][0]["review_required"]
    assert any(part[1] == "given_name" and part[2] == "Н" for part in components(result))


def test_surface_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        analyze(
            "William Tambellini",
            [{"span": [0, 7], "class": "person_name", "text": "Willian"}],
        )


def test_native_cli_preserves_line_alignment(tmp_path: Path):
    texts = tmp_path / "input.txt"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "qc.jsonl"
    texts.write_text("My name is William Tambellini.\n", encoding="utf-8")
    predictions.write_text(
        json.dumps(
            [
                {"span": [11, 29], "class": "person_name", "text": "William Tambellini"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit_native_arrays(
        Namespace(
            config=str(CONFIG),
            family_lexicon=[],
            given_lexicon=[],
            language="en",
            output=str(output),
            predictions=str(predictions),
            texts=str(texts),
        )
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["id"] == "0"
    assert row["proposals"][0]["carrier"]["text"] == "William Tambellini"


def test_native_cli_does_not_leave_partial_output_after_late_invalid_row(tmp_path: Path):
    texts = tmp_path / "input.txt"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "qc.jsonl"
    texts.write_text("Ada Lovelace\nGrace Hopper\n", encoding="utf-8")
    predictions.write_text(
        json.dumps([{"span": [0, 12], "class": "person_name", "text": "Ada Lovelace"}])
        + "\n"
        + json.dumps([{"span": [0, 12], "class": "person_name", "text": "Grace Hoppe"}])
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        audit_native_arrays(
            Namespace(
                config=str(CONFIG),
                family_lexicon=[],
                given_lexicon=[],
                language="en",
                output=str(output),
                predictions=str(predictions),
                texts=str(texts),
            )
        )

    assert not output.exists()


def test_records_cli_joins_luna_predictions_to_blind_source_packet(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "qc.jsonl"
    text = "Lewis Francis William Stott"
    source.write_text(
        json.dumps(
            {
                "id": "one",
                "bcp47": "en",
                "text": text,
                "base_spans": [{"start": 0, "end": len(text), "type": "person_name"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "id": "one",
                "preds": [],
                "subclass_spans": [
                    {
                        "carrier_start": 0,
                        "carrier_end": len(text),
                        "type": "person_name",
                        "start": 0,
                        "end": 5,
                        "family": "name_component",
                        "value": "given_name",
                    },
                    {
                        "carrier_start": 0,
                        "carrier_end": len(text),
                        "type": "person_name",
                        "start": 6,
                        "end": 21,
                        "family": "name_component",
                        "value": "middle_name",
                    },
                    {
                        "carrier_start": 0,
                        "carrier_end": len(text),
                        "type": "person_name",
                        "start": 22,
                        "end": 27,
                        "family": "name_component",
                        "value": "family_name",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit_records(
        Namespace(
            annotations_field="auto",
            config=str(CONFIG),
            family_lexicon=[],
            given_lexicon=[],
            id_field="id",
            input=str(predictions),
            language=None,
            language_field="bcp47",
            output=str(output),
            source_annotations_field="base_spans",
            source_input=str(source),
            subclass_field="subclass_spans",
            text_field="text",
        )
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert [item["status"] for item in row["proposals"][0]["components"]] == [
        "agree",
        "agree",
        "agree",
    ]
    assert row["summary"]["finding_counts"] == {}
