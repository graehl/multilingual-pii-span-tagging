import json
from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import pytest

from scripts.pii_name_resource_inventory import (
    language_role_support,
    provider_role_surfaces,
    read_supplemental,
    read_wiktionary,
    wiktionary_specifications,
)


def test_provider_role_surfaces_accepts_mapping_and_split_collections() -> None:
    class DirectProvider:
        first_names = OrderedDict(((" Ada ", 1), ("Zoë", 2)))
        first_names_female = ("Amina",)

    class SplitProvider:
        last_names_female = ("Kowalska",)
        last_names_male = ("Kowalski",)

    assert provider_role_surfaces(DirectProvider, "first_names") == {"Ada", "Amina", "Zoë"}
    assert provider_role_surfaces(SplitProvider, "last_names") == {"Kowalska", "Kowalski"}


def test_supplemental_context_and_weight_flow_to_resources(tmp_path) -> None:
    path = tmp_path / "supplemental.jsonl"
    records = [
        {
            "kind": "metadata",
            "source_contexts": {
                "faker-ms": {
                    "context_language": "ms",
                    "context_nations": ["MY"],
                    "context_basis": "test fixture",
                    "supervision_tier": "synthetic_suspect",
                    "supervision_weight": 0.1,
                }
            },
        },
        {"kind": "surface", "source": "faker-ms", "role": "given", "surface": " Aminah "},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    resources = defaultdict(lambda: defaultdict(set))
    contexts = {}

    metadata = read_supplemental(path, resources, contexts)

    assert metadata["kind"] == "metadata"
    assert resources["faker-ms"]["given"] == {"Aminah"}
    assert contexts["faker-ms"]["supervision_weight"] == 0.1


def test_supplemental_rejects_nonpositive_weight(tmp_path) -> None:
    path = tmp_path / "supplemental.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "metadata",
                "source_contexts": {
                    "faker-ms": {
                        "context_language": "ms",
                        "context_nations": [],
                        "context_basis": "test fixture",
                        "supervision_tier": "synthetic_suspect",
                        "supervision_weight": 0,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid supervision weight"):
        read_supplemental(path, defaultdict(lambda: defaultdict(set)), {})


def test_wiktionary_adds_language_context_dynamically(tmp_path) -> None:
    path = tmp_path / "wiktionary.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "surface",
                "group": "te",
                "role": "family",
                "surface": "రెడ్డి",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    resources = defaultdict(lambda: defaultdict(set))
    contexts = {}

    read_wiktionary(path, resources, contexts)

    assert resources["enwiktionary-te"]["family"] == {"రెడ్డి"}
    assert contexts["enwiktionary-te"]["context_language"] == "te"
    assert contexts["enwiktionary-te"]["supervision_weight"] == 0.5


def test_wiktionary_manifest_expands_to_cli_specifications(tmp_path) -> None:
    path = tmp_path / "categories.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pii-name-wiktionary-category-manifest-v1",
                "categories": [
                    {"role": "given", "group": "ms", "category": "Malay given names"},
                    {"role": "family", "group": "ms", "category": "Malay surnames"},
                ],
            }
        ),
        encoding="utf-8",
    )

    specifications = wiktionary_specifications(
        SimpleNamespace(category=["given:te:Telugu given names"], category_manifest=[str(path)])
    )

    assert specifications == [
        "given:te:Telugu given names",
        "given:ms:Malay given names",
        "family:ms:Malay surnames",
    ]


def test_language_support_distinguishes_rows_from_effective_weight() -> None:
    resources = {
        "trusted": {"given": {"Ada", "Amina"}},
        "weak": {"given": {"Zara"}, "family": {"Smith"}},
    }
    contexts = {
        "trusted": {"context_language": "en-US", "supervision_weight": 1.0},
        "weak": {"context_language": "en", "supervision_weight": 0.1},
    }

    support = language_role_support(resources, contexts)

    assert support["en"]["given"] == {"source_rows": 3, "effective_weight": 2.1}
    assert support["en"]["family"] == {"source_rows": 1, "effective_weight": 0.1}
