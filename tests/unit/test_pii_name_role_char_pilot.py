import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.pii_language_policy import load_core_language_policy
from scripts.pii_name_role_char_pilot import (
    DEFAULT_LANGUAGE_ROUND,
    Example,
    NameRoleCharCNN,
    build_character_vocabulary,
    case_variants,
    character_backoff_plan,
    encode_surface,
    load_initial_model,
    load_unambiguous_examples,
    natural_case,
    sampling_plan,
    selected_epoch,
    unknown_character_token,
)


def test_publisher_uppercase_is_reconstructed_or_titlecased(tmp_path) -> None:
    path = tmp_path / "names.jsonl"
    rows = [
        {
            "surface": "DEMARCUS",
            "role": "family",
            "source": "us-census-2010",
            "context_language": "en-US",
        },
        {
            "surface": "DeMarcus",
            "role": "family",
            "source": "other-mixed-case",
            "context_language": "en-US",
        },
        {
            "surface": "O'NEIL",
            "role": "family",
            "source": "us-census-2010",
            "context_language": "en-US",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    examples, receipt = load_unambiguous_examples(path)

    assert {example.surface for example in examples} == {"DeMarcus", "O'Neil"}
    assert "DEMARCUS" not in {example.surface for example in examples}
    assert receipt["publisher_case_matched_from_other_sources"] == 1
    assert receipt["publisher_case_title_fallback"] == 1


def test_case_intent_preserves_internal_mixed_case() -> None:
    assert natural_case("DeMarcus") == "DeMarcus"
    assert case_variants(
        "DeMarcus",
        natural_weight=0.7,
        uppercase_weight=0.2,
        lowercase_weight=0.1,
    ) == (("DeMarcus", 0.7), ("DEMARCUS", 0.2), ("demarcus", 0.1))
    with pytest.raises(ValueError, match="sum to one"):
        case_variants(
            "DeMarcus",
            natural_weight=0.8,
            uppercase_weight=0.2,
            lowercase_weight=0.1,
        )


def test_full_observed_vocabulary_and_typed_unknowns() -> None:
    examples = [Example(surface="DeMarcus", key="demarcus", role="given", language="en")]
    vocabulary, ids = build_character_vocabulary(
        examples,
        rare_character_maximum_count=-1,
        rare_character_backoff_probability=0.0,
        maximum_characters=0,
        case_weights=(0.7, 0.2, 0.1),
        unknown_character_backoff="typed",
    )

    assert {"D", "e", "M", "a", "r", "c", "u", "s"} <= set(vocabulary)
    assert unknown_character_token("Á") == "<UNK_LATIN_UPPER>"
    encoded = encode_surface("Áda", ids, max_characters=8)
    assert encoded[1] == ids["<UNK_LATIN_UPPER>"]


def test_script_block_backoff_hard_and_sampled_controls_are_separate() -> None:
    examples = [Example(surface="AaA", key="aaa", role="given", language="en")]
    hard_vocabulary, hard_ids = build_character_vocabulary(
        examples,
        rare_character_maximum_count=1,
        rare_character_backoff_probability=1.0,
        maximum_characters=0,
        case_weights=(0.7, 0.2, 0.1),
        unknown_character_backoff="script-block",
    )
    soft_vocabulary, soft_ids = build_character_vocabulary(
        examples,
        rare_character_maximum_count=1,
        rare_character_backoff_probability=0.5,
        maximum_characters=0,
        case_weights=(0.7, 0.2, 0.1),
        unknown_character_backoff="script-block",
    )
    rare, receipt = character_backoff_plan(
        examples,
        rare_character_maximum_count=1,
        rare_character_backoff_probability=0.5,
        frequent_character_backoff_probability=0.0,
        unknown_character_backoff="script-block",
        vocabulary=soft_vocabulary,
    )

    assert rare == frozenset({"a"})
    assert "A" in hard_ids
    assert "a" not in hard_ids
    assert "a" in soft_ids
    assert len(hard_vocabulary) < len(soft_vocabulary)
    assert receipt["rare_distinct_codepoints"] == 1

    hard = encode_surface(
        "Aa",
        hard_ids,
        max_characters=6,
        unknown_character_backoff="script-block",
    )
    rare_replaced = encode_surface(
        "Aa",
        soft_ids,
        max_characters=6,
        unknown_character_backoff="script-block",
        rare_characters=rare,
        rare_character_backoff_probability=1.0,
    )
    frequent_replaced = encode_surface(
        "Aa",
        soft_ids,
        max_characters=6,
        unknown_character_backoff="script-block",
        rare_characters=rare,
        frequent_character_backoff_probability=1.0,
    )
    assert hard[1] == hard_ids["A"]
    assert hard[2] != hard_ids["A"]
    assert rare_replaced[1] == soft_ids["A"]
    assert rare_replaced[2] != soft_ids["a"]
    assert frequent_replaced[1] != soft_ids["A"]
    assert frequent_replaced[2] == soft_ids["a"]


def test_probability_mixture_endpoints_are_explicit() -> None:
    model = NameRoleCharCNN(
        vocabulary_size=32,
        language_count=2,
        embedding_dim=4,
        convolution_channels=3,
        language_dim=2,
        language_dropout=0.0,
        language_noise=0.0,
        shared_mixture_alpha=0.5,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.shared_classifier.bias.copy_(torch.tensor([2.0, 0.0]))
        model.conditioned_classifier.bias.copy_(torch.tensor([0.0, 2.0]))
    characters = torch.zeros((1, 6), dtype=torch.int64)
    language = torch.tensor([[1.0, 0.0]])

    shared = model(characters, language, mixture_alpha=1.0).exp()
    conditioned = model(characters, language, mixture_alpha=0.0).exp()
    mixed = model(characters, language, mixture_alpha=0.5).exp()
    per_row_mixed = model(
        characters,
        language,
        mixture_alpha=torch.tensor([[0.5]], dtype=torch.float32),
    ).exp()

    assert shared.argmax(dim=1).item() == 0
    assert conditioned.argmax(dim=1).item() == 1
    assert mixed.sum().item() == pytest.approx(1.0)
    assert mixed[0, 0].item() == pytest.approx(0.5)
    assert torch.equal(mixed, per_row_mixed)


def test_continuation_requires_matching_model_semantics(tmp_path) -> None:
    config = {
        "schema": "pii-name-role-character-pilot-v1",
        "roles": ["given", "family"],
        "character_vocabulary": ["<PAD>", "<BOS>", "<EOS>", "<TRUNC>", "A"],
        "languages": ["en", "ja"],
        "max_characters": 8,
        "embedding_dim": 4,
        "convolution_channels": 3,
        "language_dim": 2,
        "shared_mixture_alpha": 0.2,
        "character_backoff": {
            "rare_character_maximum_count": 1,
            "rare_character_backoff_probability": 1.0,
            "frequent_character_backoff_probability": 0.0,
            "unknown_character_backoff": "script-block",
            "script_projection": {"sha256": "projection-hash"},
        },
    }
    source = NameRoleCharCNN(
        vocabulary_size=5,
        language_count=2,
        embedding_dim=4,
        convolution_channels=3,
        language_dim=2,
        language_dropout=0.0,
        language_noise=0.0,
        shared_mixture_alpha=0.2,
    )
    output = tmp_path / "source"
    output.mkdir()
    (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(source.state_dict(), output / "model.safetensors")

    target = NameRoleCharCNN(
        vocabulary_size=5,
        language_count=2,
        embedding_dim=4,
        convolution_channels=3,
        language_dim=2,
        language_dropout=0.0,
        language_noise=0.0,
        shared_mixture_alpha=0.2,
    )
    receipt = load_initial_model(target, output, config)

    assert receipt["model"]["sha256"]
    assert all(torch.equal(source.state_dict()[key], target.state_dict()[key]) for key in source.state_dict())

    incompatible = dict(config)
    incompatible["languages"] = ["ja", "en"]
    with pytest.raises(ValueError, match="languages"):
        load_initial_model(target, output, incompatible)


def test_selected_epoch_can_preserve_inherited_checkpoint() -> None:
    history = [
        {"epoch": 1, "development_natural_language_macro_f1": 0.80},
        {"epoch": 2, "development_natural_language_macro_f1": 0.85},
    ]
    inherited = {"natural_language": {"macro_f1_observed_roles": 0.90}}

    assert selected_epoch(history, inherited) == 0
    assert selected_epoch(history, None) == 2


def test_final35_name_round_loads_all_literal_language_codes() -> None:
    policy = load_core_language_policy(DEFAULT_LANGUAGE_ROUND)

    assert len(policy["core_languages"]) == 35
    assert "no" in policy["core_languages"]


def test_low_trust_pool_keeps_language_mass_and_trusted_rows_displace(tmp_path) -> None:
    language_round = tmp_path / "languages.yaml"
    language_round.write_text(
        "schema_version: 1\n"
        "languages:\n"
        "  - code: en\n"
        "  - code: ms\n"
        "  - code: te\n"
        "training_mix:\n"
        "  minimum_core_language_share: 0.01\n"
        "language_importance:\n"
        "  weights:\n"
        "    en: 4.0\n"
        "    ms: 1.0\n"
        "    te: 1.0\n",
        encoding="utf-8",
    )
    examples = [
        Example("Alice", "alice", "given", "en"),
        Example("Smith", "smith", "family", "en"),
        Example("Aminah", "aminah", "given", "ms", supervision_weight=0.1),
        Example("Bakar", "bakar", "family", "ms", supervision_weight=0.1),
        Example("Ismail", "ismail", "family", "ms", supervision_weight=1.0),
    ]

    weights, language_mass, receipt = sampling_plan(
        examples,
        language_round=language_round,
        maximum_support_multiplier=2.0,
        full_support_effective_cell_weight=1.0,
    )

    assert language_mass["en"] > language_mass["ms"] > 0
    assert sum(
        weight for example, weight in zip(examples, weights, strict=True) if example.language == "ms"
    ) == pytest.approx(language_mass["ms"])
    assert weights[4] == pytest.approx(10 * weights[3])
    assert receipt["support_multiplier"]["per_language"]["en"] == 2.0
    assert receipt["support_multiplier"]["per_language"]["ms"] == pytest.approx(1.1)
    assert receipt["void_languages"] == ["te"]
    assert receipt["status"] == "verified_represented_sampler_with_voids"
    assert receipt["weighting_order"][-1] == ("relative supervision weight within language-role cell")
