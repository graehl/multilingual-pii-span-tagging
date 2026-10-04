"""Gold replay retains empty inputs and negatives outside nested annotations."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from scripts.pii_encoder_train import SpanDataset


@pytest.mark.parametrize("separate_coverage", [False, True])
def test_full_gold_compiler_preserves_negative_and_nested_evidence(tmp_path, separate_coverage):
    path = Path(__file__).parents[2] / "research/pii/frontier/evidence/four-corpus-v1/compile-mixture.py"
    spec = importlib.util.spec_from_file_location("gold_compiler", path)
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "Ford": 2, "Foundation": 3, "waits": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tok.save_pretrained(tmp_path / "tokenizer")
    control, screen, intake = [tmp_path / name for name in ("control", "screen", "intake")]
    for folder in (control, screen, intake):
        folder.mkdir()
    mapping = {
        "ontology": {"primary_types": ["person_name", "organization", "credential"]},
        "source_labels": {"toy": {"PER": "source_person", "ORG": "source_org"}},
        "v1_to_v2": {
            "source_person": {"accepted": ["person_name"]},
            "source_org": {"accepted": ["organization"]},
        },
    }
    old = {"O": 0}
    for kind in ("source_person", "source_org"):
        for prefix in "BIES":
            old[f"{prefix}-{kind}"] = len(old)
    new = {"O": 0}
    for kind in mapping["ontology"]["primary_types"]:
        for prefix in "BIES":
            new[f"{prefix}-{kind}"] = len(new)
    base = {
        "text": "Ford waits",
        "spans": [[0, 4, "source_person"]],
        "lang": "en",
        "sampling_weight": 1.0,
        "supervision": "complete",
        "label_space": "v1",
    }
    for name in ("train", "val", "evaluation"):
        compiler.write(control / f"{name}.jsonl", [base])
    (control / "mapping.json").write_text(json.dumps(mapping))
    (control / "labels.json").write_text(json.dumps(old))
    (control / "receipt.json").write_text(
        json.dumps({"outputs": {p.name: compiler.identity(p) for p in control.iterdir()}})
    )
    source = [
        {"id": "empty", "lang": "en", "text": "waits", "spans": []},
        {
            "id": "nested",
            "lang": "en",
            "text": "Ford Foundation waits",
            "spans": [
                {"start": 0, "end": 4, "source_label": "PER"},
                {"start": 0, "end": 15, "source_label": "ORG"},
            ],
        },
    ]
    compiler.write(intake / "toy-train.jsonl", source)
    queries = [
        {**row, "source_id": row["id"], "split": "train", "corpus": "toy", "source_locator": {"line": i + 1}}
        for i, row in enumerate(source)
    ]
    compiler.write(intake / "incoming-queries.jsonl", queries)
    (screen / "retained.ids").write_text("empty\nnested\n")
    (screen / "receipt.json").write_text(
        json.dumps(
            {
                "outputs": {"retained.ids": compiler.identity(screen / "retained.ids")},
                "inputs": [compiler.identity(intake / "incoming-queries.jsonl")],
            }
        )
    )
    coverage = tmp_path / "negative-coverage.json"
    coverage.write_text(json.dumps({"negative_types_by_corpus": {"toy": ["person_name"]}}))
    compiler.build(
        SimpleNamespace(
            control=control,
            screen=screen,
            intake=intake,
            tokenizer=tmp_path / "tokenizer",
            output=tmp_path / "compiled",
            outside_policy="full-corpus",
            negative_coverage=coverage if separate_coverage else None,
            doses=[0.4],
        )
    )
    rows = compiler.read(tmp_path / "compiled/annotation-views.jsonl")
    assert len(rows) == 3
    data = SpanDataset(rows, tok, old, 32, secondary_label2id=new, mapped_outside_by_row=True)
    for row, encoded in zip(rows, data):
        assert row["supervision"] == "complete"
        assert row["unknown_primary_types"] == (
            ["credential", "organization"] if separate_coverage else ["credential"]
        )
        outside_allowed = encoded["mapped_outside_allowed"]
        assert outside_allowed[new["S-organization"]] == separate_coverage
        assert not outside_allowed[new["S-person_name"]]
        assert encoded["labels"][-1] == old["O"]  # waits remains a real negative in every view
        if not row["spans"]:
            assert encoded["labels"] == [old["O"]]
        elif row["spans"][0][2] == "source_person":
            assert encoded["labels"][:2] == [old["S-source_person"], -100]
        else:
            assert encoded["labels"][:2] == [old["B-source_org"], old["E-source_org"]]
    receipt = json.loads((tmp_path / "compiled/receipt.json").read_text())
    assert receipt["doses"]["0.4"]["branch_probability"]["human_gold"] == 0.4
