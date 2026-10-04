import json

import pytest

from scripts.pii_language_policy import (
    allocate_importance_language_counts,
    language_shares,
    load_core_language_policy,
    validate_core_language_support,
)


def write_round(tmp_path, languages=("en", "es", "vi"), minimum=0.01):
    path = tmp_path / "round.yaml"
    path.write_text(
        "round_id: test-round\n"
        "languages:\n"
        + "".join(f"  - code: {language}\n" for language in languages)
        + "training_mix:\n"
        + f"  minimum_core_language_share: {minimum}\n",
        encoding="utf-8",
    )
    return path


def test_joint_model_requires_every_core_language_and_allows_extras(tmp_path):
    language_round = write_round(tmp_path)

    receipt = validate_core_language_support(
        {"en": 0.4, "es": 0.3, "vi": 0.2, "tl": 0.1},
        language_round=language_round,
    )

    assert receipt["mode"] == "joint"
    assert receipt["required_core_languages_for_model"] == ["en", "es", "vi"]
    assert receipt["extra_languages"] == ["tl"]
    with pytest.raises(ValueError, match=r"missing=\['vi'\]"):
        validate_core_language_support(
            {"en": 0.5, "es": 0.5},
            language_round=language_round,
        )
    with pytest.raises(ValueError, match="below=.*vi"):
        validate_core_language_support(
            {"en": 0.5, "es": 0.495, "vi": 0.005},
            language_round=language_round,
        )


def test_routed_components_cover_core_in_aggregate_and_enforce_assigned_floor(tmp_path):
    language_round = write_round(tmp_path)
    manifest = tmp_path / "routes.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "test-routes-v1",
                "routes": {
                    "latin": {
                        "languages": ["en", "es"],
                        "joint_sampling_mass": 0.8,
                    },
                    "rider": {
                        "languages": ["vi", "tl"],
                        "joint_sampling_mass": 0.2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    receipt = validate_core_language_support(
        {"en": 0.7, "es": 0.3},
        language_round=language_round,
        split_manifest=manifest,
        split_component="latin",
    )

    assert receipt["mode"] == "routed_component"
    assert receipt["required_core_languages_for_model"] == ["en", "es"]
    assert receipt["route"]["required_components"] == ["latin", "rider"]
    assert receipt["aggregate_equivalent_expected_language_shares"]["en"] == pytest.approx(0.56)
    with pytest.raises(ValueError, match="undeclared core languages.*vi"):
        validate_core_language_support(
            {"en": 0.7, "es": 0.2, "vi": 0.1},
            language_round=language_round,
            split_manifest=manifest,
            split_component="latin",
        )
    with pytest.raises(ValueError, match="below=.*vi"):
        validate_core_language_support(
            {"vi": 0.04, "tl": 0.96},
            language_round=language_round,
            split_manifest=manifest,
            split_component="rider",
        )


def test_routed_manifest_must_cover_the_whole_core(tmp_path):
    language_round = write_round(tmp_path)
    manifest = tmp_path / "routes.json"
    manifest.write_text(
        json.dumps(
            {
                "routes": {
                    "latin": {
                        "languages": ["en", "es"],
                        "joint_sampling_mass": 1.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="omits core languages.*vi"):
        validate_core_language_support(
            {"en": 0.5, "es": 0.5},
            language_round=language_round,
            split_manifest=manifest,
            split_component="latin",
        )


def test_default_final20_policy_and_weighted_share_helper():
    policy = load_core_language_policy()

    assert len(policy["core_languages"]) == 20
    assert policy["minimum_expected_share"] == 0.006
    assert language_shares(["en", "en", "es"], [1, 2, 3]) == {"en": 0.5, "es": 0.5}


def test_importance_counts_preserve_exact_budget_and_declared_ratios(tmp_path):
    language_round = write_round(tmp_path)
    language_round.write_text(
        language_round.read_text(encoding="utf-8")
        + "language_importance:\n"
        + "  weights:\n"
        + "    en: 4\n"
        + "    es: 2\n"
        + "    vi: 1\n",
        encoding="utf-8",
    )

    counts = allocate_importance_language_counts(70, language_round=language_round)

    assert counts == {"en": 40, "es": 20, "vi": 10}
