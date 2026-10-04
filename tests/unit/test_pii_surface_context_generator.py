import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from scripts.pii_seed_tree import SeedTree
from scripts.pii_surface.agreement_mining import SCHEMA as AGREEMENT_SCHEMA
from scripts.pii_surface.context_generator import (
    SCHEMA,
    TAG_CONDITIONING_EXACT,
    TAG_CONDITIONING_P20_P9,
    CharacterVocabulary,
    ContextCharacterGenerator,
    ContextExample,
    LoadedContextSurfaceGenerator,
    ModelDimensions,
    build_tag_conditioning,
    collate_examples,
    load_carrier_examples,
    load_context_inputs,
    load_examples,
    load_natural_examples,
)
from scripts.pii_surface.exact_agreement_pool import split_for_source
from scripts.pii_surface.language_conditioning_bundle import (
    INTRINSIC_WINNER,
    LoadedLanguageConditioningSurfaceGenerator,
)
from scripts.pii_surface.language_conditioning_bundle import (
    SCHEMA as LANGUAGE_BUNDLE_SCHEMA,
)
from scripts.pii_surface.rerealize import placeholderize_row, realize_placeholder_row
from scripts.pii_surface_pool import record_hash


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_context_checkpoint(path: Path, *, mode: str, version: str) -> dict[str, str]:
    examples = [ContextExample("a", "0" * 64, "train", "en", "PERSON_NAME", "", "", "Ada")]
    vocabulary = CharacterVocabulary.build(examples)
    tags = ("<unk>", "PERSON_NAME")
    dimensions = ModelDimensions(8, 4, 6, 10, 0)
    conditioning = build_tag_conditioning(tags, mode)
    model = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=2,
        tag_count=len(tags),
        dimensions=dimensions,
        tag_conditioning=conditioning,
    )
    path.mkdir(parents=True)
    weights = path / "model.pt"
    torch.save(model.state_dict(), weights)
    config = {
        "schema": SCHEMA,
        "model_version": version,
        "context_chars": 20,
        "max_target_chars": 8,
        "vocabulary": list(vocabulary.tokens),
        "languages": ["<unk>", "en"],
        "tags": list(tags),
        "tag_conditioning": conditioning.to_config(),
        "dimensions": {
            "character_embedding": 8,
            "condition_embedding": 4,
            "context_hidden": 6,
            "decoder_hidden": 10,
            "dropout": 0,
        },
        "decoding": {"temperature": 0.8, "top_k": 4},
        "weights_sha256": sha256(weights),
    }
    config_path = path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return {"config_sha256": sha256(config_path), "weights_sha256": sha256(weights)}


def source_hash_for_split(split: str) -> str:
    for value in range(1000):
        candidate = hashlib.sha256(str(value).encode()).hexdigest()
        if split_for_source(candidate, 10, 0) == split:
            return candidate
    raise AssertionError(f"failed to find source hash for {split}")


def text_for_split(split: str) -> str:
    for value in range(1000):
        text = f"Before Name{value} after"
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        if split_for_source(source_hash, 10, 0) == split:
            return text
    raise AssertionError(f"failed to find source text for {split}")


def agreement_row(
    candidate_id: str,
    source_hash: str,
    surface: str = "Ada",
    tag: str = "PERSON_NAME",
) -> dict:
    context = f"Before {surface} after"
    start = len("Before ")
    return {
        "schema": AGREEMENT_SCHEMA,
        "candidate_id": candidate_id,
        "status": "agreement_candidate_quality_unreviewed",
        "tier": "exact_same_label",
        "language": "en",
        "tag": tag,
        "surface": surface,
        "source_text_sha256": source_hash,
        "start": start,
        "end": start + len(surface),
        "context_start": 0,
        "context": context,
    }


def test_load_examples_uses_source_partition_and_bounded_context(tmp_path: Path) -> None:
    agreement = tmp_path / "agreement.jsonl"
    rows = [
        agreement_row("train", source_hash_for_split("train")),
        agreement_row("audit", source_hash_for_split("audit"), surface="Grace"),
    ]
    agreement.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    examples, rejected = load_examples(
        agreement,
        context_chars=3,
        max_target_chars=8,
    )

    assert rejected == {}
    assert [example.split for example in examples] == ["train", "audit"]
    assert examples[0].left == "re "
    assert examples[0].right == " af"
    assert examples[1].target == "Grace"


def test_load_natural_examples_recovers_literal_carrier_holes(tmp_path: Path) -> None:
    text = "Dr Ada Lovelace joined Acme with Grace at 02139."
    row = {
        "id": "en:test:1",
        "lang": "en",
        "text": text,
        "spans": [
            {
                "start": text.index("Ada"),
                "end": text.index("Ada") + len("Ada Lovelace"),
                "label": "person_name",
            },
            {
                "start": text.index("Acme"),
                "end": text.index("Acme") + len("Acme"),
                "label": "organization",
            },
            {
                "start": text.index("Grace"),
                "end": text.index("Grace") + len("Grace"),
                "label": "given_name",
            },
            {
                "start": text.index("02139"),
                "end": text.index("02139") + len("02139"),
                "label": "postal_code",
            },
        ],
    }
    source = tmp_path / "en.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as output:
        output.write(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"upstream": {"license": "test"}}), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configuration": {
                    "languages": ["en"],
                    "holdout_modulus": 10,
                    "holdout_fold": 0,
                },
                "input_files": {"natural-test": [str(source)]},
                "source_manifests": {"natural-test": {"path": str(manifest), "sha256": sha256(manifest)}},
            }
        ),
        encoding="utf-8",
    )

    examples, rejected, source_files = load_natural_examples(
        report,
        context_chars=4,
        max_target_chars=32,
    )

    assert [(example.tag, example.target) for example in examples] == [
        ("PERSON_NAME", "Ada Lovelace"),
        ("ORGANIZATION", "Acme"),
        ("GIVEN_NAME", "Grace"),
    ]
    assert (examples[0].left, examples[0].right) == ("Dr ", " joi")
    assert rejected == {"closed_or_structured_tag": 1}
    assert source_files == [{"dataset": "natural-test", "path": str(source), "sha256": sha256(source)}]

    carrier_examples, carrier_rejected, carrier_files = load_carrier_examples(
        report,
        context_chars=4,
        max_target_chars=32,
    )
    assert carrier_examples == examples
    assert carrier_rejected == rejected
    assert carrier_files == source_files

    scoped_report = tmp_path / "scoped-report.json"
    scoped = json.loads(report.read_text(encoding="utf-8"))
    scoped["configuration"]["tags"] = ["PERSON_NAME"]
    scoped_report.write_text(json.dumps(scoped), encoding="utf-8")
    scoped_examples, scoped_rejected, _scoped_files = load_carrier_examples(
        scoped_report,
        context_chars=4,
        max_target_chars=32,
    )
    assert [(example.tag, example.target) for example in scoped_examples] == [("PERSON_NAME", "Ada Lovelace")]
    assert scoped_rejected == {"tag_outside_scope": 3}


def test_carrier_examples_keep_repeated_source_text_in_one_split(tmp_path: Path) -> None:
    text = "Ada Lovelace joined Acme."
    span = {
        "start": 0,
        "end": len("Ada Lovelace"),
        "label": "person_name",
    }
    expected_split = split_for_source(hashlib.sha256(text.encode()).hexdigest(), 10, 0)
    record_split_ids: dict[str, str] = {}
    for index in range(1000):
        row_id = f"duplicate:{index}"
        row = {"id": row_id, "lang": "en", "text": text, "spans": [span]}
        record_split = split_for_source(record_hash("duplicate-source", row_id, row), 10, 0)
        record_split_ids.setdefault(record_split, row_id)
        if len(record_split_ids) == 2:
            break
    assert set(record_split_ids) == {"train", "audit"}

    rows = [
        {
            "id": row_id,
            "lang": "en",
            "text": text,
            "spans": [span],
        }
        for row_id in record_split_ids.values()
    ]
    source = tmp_path / "duplicates.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"upstream": {"license": "test"}}), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configuration": {
                    "languages": ["en"],
                    "holdout_modulus": 10,
                    "holdout_fold": 0,
                },
                "input_files": {"duplicate-source": [str(source)]},
                "source_manifests": {
                    "duplicate-source": {
                        "path": str(manifest),
                        "sha256": sha256(manifest),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    examples, rejected, _source_files = load_carrier_examples(
        report,
        context_chars=4,
        max_target_chars=32,
    )

    assert rejected == {}
    assert len(examples) == 2
    assert {example.source_text_sha256 for example in examples} == {hashlib.sha256(text.encode()).hexdigest()}
    assert {example.split for example in examples} == {expected_split}


def test_context_inputs_withhold_audit_examples_from_train_only_carrier(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "audit"):
        text = text_for_split(split)
        start = text.index("Name")
        rows.append(
            {
                "id": split,
                "lang": "en",
                "text": text,
                "spans": [[start, text.index(" after"), "person_name"]],
            }
        )
    source = tmp_path / "carrier.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"upstream": {"license": "test"}}), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pool_version": "train-only-v1",
                "configuration": {
                    "languages": ["en"],
                    "tags": ["PERSON_NAME"],
                    "carrier_role": "train-only",
                    "holdout_modulus": 10,
                    "holdout_fold": 0,
                },
                "input_files": {"carrier": [str(source)]},
                "source_manifests": {"carrier": {"path": str(manifest), "sha256": sha256(manifest)}},
            }
        ),
        encoding="utf-8",
    )

    inputs = load_context_inputs(
        agreement_path=None,
        agreement_report_path=None,
        carrier_pool_report_paths=[report],
        context_chars=4,
        max_target_chars=32,
    )

    assert len(inputs.examples) == 1
    assert inputs.examples[0].split == "train"
    assert inputs.examples_by_source == {"exact_agreement": 0, "train-only-v1": 1}
    assert inputs.input_receipt["carrier_pools"][0]["carrier_role"] == "train-only"
    assert inputs.input_receipt["carrier_pools"][0]["withheld_audit_examples"] == 1


def test_context_character_generator_shapes_teacher_forced_logits() -> None:
    examples = [
        ContextExample("a", "0" * 64, "train", "en", "PERSON_NAME", "hi ", "!", "Ada"),
        ContextExample("b", "1" * 64, "train", "en", "PERSON_NAME", "dear ", ".", "Lin"),
    ]
    vocabulary = CharacterVocabulary.build(examples)
    rows = []
    for example in examples:
        target = vocabulary.encode_target(example.target)
        rows.append(
            {
                "context_ids": vocabulary.encode_context(example.left, example.right),
                "decoder_input_ids": target[:-1],
                "target_ids": target[1:],
                "language_id": 1,
                "tag_id": 1,
                "example": example,
            }
        )
    batch = collate_examples(rows)
    model = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=2,
        tag_count=2,
        dimensions=ModelDimensions(
            character_embedding=8,
            condition_embedding=4,
            context_hidden=6,
            decoder_hidden=10,
            dropout=0,
        ),
    )

    logits = model(
        batch["context_ids"],
        batch["context_lengths"],
        batch["language_ids"],
        batch["tag_ids"],
        batch["decoder_input_ids"],
    )

    assert logits.shape == (
        2,
        batch["decoder_input_ids"].shape[1],
        len(vocabulary.tokens),
    )


def test_hierarchical_tag_conditioning_adds_total_p20_and_p9_lookups() -> None:
    tags = ("<unk>", "GIVEN_NAME", "PERSON_NAME", "ORGANIZATION", "CITY")
    conditioning = build_tag_conditioning(tags, TAG_CONDITIONING_P20_P9)

    assert conditioning.p20_labels[conditioning.tag_to_p20_ids[1]] == "person_name"
    assert conditioning.p20_labels[conditioning.tag_to_p20_ids[2]] == "person_name"
    assert conditioning.p9_labels[conditioning.tag_to_p9_ids[1]] == "name"
    assert conditioning.p9_labels[conditioning.tag_to_p9_ids[3]] == "organization"
    assert conditioning.p9_labels[conditioning.tag_to_p9_ids[4]] == "location_or_contact"

    model = ContextCharacterGenerator(
        vocabulary_size=16,
        language_count=2,
        tag_count=len(tags),
        dimensions=ModelDimensions(8, 4, 6, 10, 0),
        tag_conditioning=conditioning,
    )
    encoded = model.encode_tag(torch.tensor([1, 2, 3, 4]))
    assert encoded.shape == (4, 4)
    assert "p20_embedding.weight" in model.state_dict()
    assert "p9_embedding.weight" in model.state_dict()


def test_loaded_generator_checks_weights_and_backs_off_unknown_cells(tmp_path: Path) -> None:
    examples = [ContextExample("a", "0" * 64, "train", "en", "PERSON_NAME", "", "", "Ada")]
    vocabulary = CharacterVocabulary.build(examples)
    dimensions = ModelDimensions(
        character_embedding=8,
        condition_embedding=4,
        context_hidden=6,
        decoder_hidden=10,
        dropout=0,
    )
    model = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=2,
        tag_count=2,
        dimensions=dimensions,
    )
    weights = tmp_path / "model.pt"
    torch.save(model.state_dict(), weights)
    config = {
        "schema": SCHEMA,
        "model_version": "unit-v1",
        "context_chars": 20,
        "max_target_chars": 8,
        "vocabulary": list(vocabulary.tokens),
        "languages": ["<unk>", "en"],
        "tags": ["<unk>", "PERSON_NAME"],
        "dimensions": {
            "character_embedding": 8,
            "condition_embedding": 4,
            "context_hidden": 6,
            "decoder_hidden": 10,
            "dropout": 0,
        },
        "decoding": {"temperature": 0.8, "top_k": 4},
        "weights_sha256": sha256(weights),
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded = LoadedContextSurfaceGenerator(tmp_path)

    assert loaded.generate("pl", "PERSON_NAME", "", "", seed=1) is None
    assert loaded.generate("en", "ORGANIZATION", "", "", seed=1) is None
    assert loaded.receipt()["model_version"] == "unit-v1"


def test_loaded_generator_restores_hierarchical_tag_conditioning(tmp_path: Path) -> None:
    examples = [ContextExample("a", "0" * 64, "train", "en", "PERSON_NAME", "", "", "Ada")]
    vocabulary = CharacterVocabulary.build(examples)
    tags = ("<unk>", "PERSON_NAME")
    dimensions = ModelDimensions(8, 4, 6, 10, 0)
    conditioning = build_tag_conditioning(tags, TAG_CONDITIONING_P20_P9)
    model = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=2,
        tag_count=len(tags),
        dimensions=dimensions,
        tag_conditioning=conditioning,
    )
    weights = tmp_path / "model.pt"
    torch.save(model.state_dict(), weights)
    config = {
        "schema": SCHEMA,
        "model_version": "unit-hierarchy-v1",
        "context_chars": 20,
        "max_target_chars": 8,
        "vocabulary": list(vocabulary.tokens),
        "languages": ["<unk>", "en"],
        "tags": list(tags),
        "tag_conditioning": conditioning.to_config(),
        "dimensions": {
            "character_embedding": 8,
            "condition_embedding": 4,
            "context_hidden": 6,
            "decoder_hidden": 10,
            "dropout": 0,
        },
        "decoding": {"temperature": 0.8, "top_k": 4},
        "weights_sha256": sha256(weights),
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded = LoadedContextSurfaceGenerator(tmp_path)

    assert loaded.tag_conditioning == conditioning
    assert loaded.receipt()["tag_conditioning"]["mode"] == TAG_CONDITIONING_P20_P9


def test_language_conditioning_bundle_routes_declared_mode(tmp_path: Path) -> None:
    exact_path = tmp_path / "models" / "exact" / "en"
    hierarchical_path = tmp_path / "models" / "hierarchical" / "en"
    exact_hashes = write_context_checkpoint(
        exact_path,
        mode=TAG_CONDITIONING_EXACT,
        version="unit-exact-v1",
    )
    hierarchical_hashes = write_context_checkpoint(
        hierarchical_path,
        mode=TAG_CONDITIONING_P20_P9,
        version="unit-hierarchical-v1",
    )
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "schema": LANGUAGE_BUNDLE_SCHEMA,
                "bundle_version": "unit-bundle-v1",
                "configuration": {"context_chars_per_side": 20, "max_target_chars": 8},
                "language_comparisons": [
                    {
                        "language": "en",
                        "intrinsic_winner": TAG_CONDITIONING_P20_P9,
                        "exact": {
                            "relative_path": "models/exact/en",
                            **exact_hashes,
                        },
                        "hierarchical": {
                            "relative_path": "models/hierarchical/en",
                            **hierarchical_hashes,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    routed_exact = LoadedLanguageConditioningSurfaceGenerator(
        tmp_path,
        route_mode=TAG_CONDITIONING_EXACT,
    )
    routed_winner = LoadedLanguageConditioningSurfaceGenerator(
        tmp_path,
        route_mode=INTRINSIC_WINNER,
    )

    assert routed_exact.routes["en"].config["model_version"] == "unit-exact-v1"
    assert routed_winner.routes["en"].config["model_version"] == "unit-hierarchical-v1"
    assert routed_winner.generate("fr", "PERSON_NAME", "", "", seed=1) is None
    assert routed_winner.receipt()["routes"] == [
        {
            "language": "en",
            "tag_conditioning": TAG_CONDITIONING_P20_P9,
            "relative_path": "models/hierarchical/en",
            "model_version": "unit-hierarchical-v1",
            "config_sha256": hierarchical_hashes["config_sha256"],
            "weights_sha256": hierarchical_hashes["weights_sha256"],
        }
    ]


def test_realizer_prefers_context_generator_and_links_repeats() -> None:
    class Filler:
        surface_pool = None
        surface_policy = None
        document_rejection_reasons = []

        def begin_document(self, _ph_map, *, seed=None):
            assert seed is not None

        def fill(self, *_args, **_kwargs):
            raise AssertionError("conditioned generation should precede filler fallback")

        def surface_metadata(self, _tag, _generator):
            return {"policy": None, "pool_entries": []}

    class Generator:
        context_chars = 20

        def __init__(self):
            self.calls = []

        def generate(self, language, tag, left, right, *, seed):
            self.calls.append((language, tag, left, right, seed))
            return "Grace", "context-char-gru:unit-v1:greedy"

        def receipt(self):
            return {"model_version": "unit-v1"}

    source = {
        "id": "row",
        "lang": "en",
        "text": "Ada met Ada.",
        "spans": [[0, 3, "person_name"], [8, 11, "person_name"]],
    }
    placeholder = placeholderize_row(source)
    generator = Generator()

    realized = realize_placeholder_row(
        source,
        placeholder,
        filler=cast(Any, Filler()),
        seed_tree=SeedTree(7),
        realization_index=3,
        materialization_version="unit-v1",
        partition_role="training",
        surface_generator=generator,
    )

    assert realized["text"] == "Grace met Grace."
    assert realized["spans"] == [[0, 5, "person_name"], [10, 15, "person_name"]]
    assert len(generator.calls) == 1
    assert realized["materialization"]["context_surface_generator"] == {
        "model_version": "unit-v1",
        "configured_attempt_rate": 1.0,
        "gate": "deterministic-per-linked-entity-v1",
    }


def test_realizer_can_deterministically_mix_generator_and_fallback() -> None:
    class Filler:
        surface_pool = None
        surface_policy = None
        document_rejection_reasons = []

        def begin_document(self, _ph_map, *, seed=None):
            assert seed is not None

        def fill(self, *_args, **_kwargs):
            return "Observed", "natural-pool:unit-v1:observed"

        def surface_metadata(self, _tag, _generator):
            return {"policy": None, "pool_entries": [{"entry_id": "observed"}]}

    class Generator:
        def __init__(self):
            self.calls = 0

        def generate(self, *_args, **_kwargs):
            self.calls += 1
            return "Learned", "context-char-gru:unit-v1:sample"

        def receipt(self):
            return {"model_version": "unit-v1"}

    source = {
        "id": "row",
        "lang": "en",
        "text": "Ada arrived.",
        "spans": [[0, 3, "person_name"]],
    }
    placeholder = placeholderize_row(source)
    generator = Generator()
    outputs = []
    for realization_index in range(64):
        result = realize_placeholder_row(
            source,
            placeholder,
            filler=cast(Any, Filler()),
            seed_tree=SeedTree(7),
            realization_index=realization_index,
            materialization_version="unit-v1",
            partition_role="training",
            surface_generator=generator,
            surface_generator_rate=0.5,
        )
        outputs.append(result["text"])

    assert set(outputs) == {"Learned arrived.", "Observed arrived."}
    assert 16 <= generator.calls <= 48
    repeated = realize_placeholder_row(
        source,
        placeholder,
        filler=cast(Any, Filler()),
        seed_tree=SeedTree(7),
        realization_index=11,
        materialization_version="unit-v1",
        partition_role="training",
        surface_generator=generator,
        surface_generator_rate=0.5,
    )
    assert repeated["text"] == outputs[11]


def test_realizer_applies_exact_cell_hierarchical_alpha_beta_mix() -> None:
    class Cell:
        native_name_rate = 1.0

        def __init__(self, beta: float, alpha: float):
            self.fresh_faker_beta = beta
            self.replacement_char_alpha = alpha

    class Policy:
        def for_tag(self, _language, tag):
            return {
                "PERSON_NAME": Cell(1.0, 0.5),
                "ORGANIZATION": Cell(0.0, 1.0),
                "OCCUPATION": Cell(0.0, 0.0),
            }[tag]

        def receipt(self):
            return {"recipe_version": "unit-selective-v1"}

    class Filler:
        class Pool:
            version = "unit-v1"

            def distinct_count(self, _language, _tag):
                return 1

        surface_pool = Pool()
        surface_policy = Policy()
        document_rejection_reasons = []

        def begin_document(self, _ph_map, *, seed=None):
            assert seed is not None

        def fill(self, tag, *_args, surface_route="policy", **_kwargs):
            assert surface_route in {"fresh-faker", "exact-empirical"}
            return {
                "PERSON_NAME": ("Fresh Faker", "faker:unit"),
                "OCCUPATION": ("Exact Empirical", "natural-pool:unit-v1:exact"),
            }[tag]

        def surface_metadata(self, _tag, _generator):
            return {"policy": None, "pool_entries": []}

    class Generator:
        def __init__(self):
            self.tags = []

        def generate(self, _language, tag, _left, _right, *, seed):
            assert seed is not None
            self.tags.append(tag)
            return "Learned", "context-char-gru:unit-v1:sample"

        def receipt(self):
            return {"model_version": "unit-v1"}

    source = {
        "id": "row",
        "lang": "vi",
        "text": "John joined Acme as a diver.",
        "spans": [
            [0, 4, "person_name"],
            [12, 16, "organization"],
            [22, 27, "occupation"],
        ],
    }
    generator = Generator()

    result = realize_placeholder_row(
        source,
        placeholderize_row(source),
        filler=cast(Any, Filler()),
        seed_tree=SeedTree(7),
        realization_index=3,
        materialization_version="unit-v1",
        partition_role="training",
        surface_generator=generator,
    )

    assert result["text"] == "Fresh Faker joined Learned as a Exact Empirical."
    assert generator.tags == ["ORGANIZATION"]
    assert [span["surface_mix"]["selected_branch"] for span in result["span_provenance"]] == [
        "fresh_faker",
        "character_generator",
        "exact_empirical",
    ]
    assert result["span_provenance"][0]["surface_mix"]["configured_branch_probabilities"] == {
        "fresh_faker": 1.0,
        "character_generator": 0.0,
        "exact_empirical": 0.0,
    }


def test_slot_conditioning_is_opt_in_and_changes_the_condition():
    """Slot case is an extra generator input, off unless the data carries it.

    The generator formerly saw only language, tag and raw context. Conditioning
    on the case the carrier slot demands lets one entity's realizations across
    cases share a stem instead of being memorized as unrelated strings
    (gaps/pii-transport-placeholder-morphology.md).
    """
    import torch

    from scripts.pii_surface.context_generator import (
        CharacterVocabulary,
        ContextCharacterGenerator,
        ContextExample,
        EncodedExamples,
        ModelDimensions,
        collate_examples,
        indexed_vocabulary,
    )

    rows = [
        ("pl", "location", "w ", "", "Austrii", "locative"),
        ("pl", "location", "do ", "", "Austrii", "genitive"),
        ("pl", "location", "", "", "Austria", "none"),
    ] * 4
    examples = [
        ContextExample(
            candidate_id=f"c{index}",
            source_text_sha256=f"{index:064x}",
            split="train",
            language=language,
            tag=tag,
            left=left,
            right=right,
            target=target,
            slot=slot,
        )
        for index, (language, tag, left, right, target, slot) in enumerate(rows)
    ]
    vocabulary = CharacterVocabulary.build(examples)
    languages, language_ids = indexed_vocabulary(e.language for e in examples)
    tags, tag_ids = indexed_vocabulary(e.tag for e in examples)
    slots, slot_ids = indexed_vocabulary(e.slot for e in examples)
    assert set(slots) >= {"genitive", "locative", "none"}

    dataset = EncodedExamples(
        examples,
        vocabulary=vocabulary,
        language_ids=language_ids,
        tag_ids=tag_ids,
        slot_ids=slot_ids,
    )
    batch = collate_examples([dataset[i] for i in range(len(examples))])
    dimensions = ModelDimensions()

    conditioned = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=len(languages),
        tag_count=len(tags),
        dimensions=dimensions,
        slot_count=len(slots),
    )
    assert conditioned.slot_embedding is not None
    varied = conditioned(
        batch["context_ids"],
        batch["context_lengths"],
        batch["language_ids"],
        batch["tag_ids"],
        batch["decoder_input_ids"],
        slot_ids=batch["slot_ids"],
    )
    flattened = conditioned(
        batch["context_ids"],
        batch["context_lengths"],
        batch["language_ids"],
        batch["tag_ids"],
        batch["decoder_input_ids"],
        slot_ids=torch.zeros_like(batch["slot_ids"]),
    )
    assert not torch.allclose(varied, flattened), "slot ids must reach the condition"

    # a model built with slot conditioning refuses to run without the ids
    # rather than quietly substituting a default
    with pytest.raises(ValueError):
        conditioned(
            batch["context_ids"],
            batch["context_lengths"],
            batch["language_ids"],
            batch["tag_ids"],
            batch["decoder_input_ids"],
        )

    # slot_count 0 reproduces the pre-existing architecture exactly
    legacy = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=len(languages),
        tag_count=len(tags),
        dimensions=dimensions,
        slot_count=0,
    )
    assert legacy.slot_embedding is None
    legacy_logits = legacy(
        batch["context_ids"],
        batch["context_lengths"],
        batch["language_ids"],
        batch["tag_ids"],
        batch["decoder_input_ids"],
    )
    assert legacy_logits.shape == varied.shape
