import json
from pathlib import Path

import pytest
import torch

from scripts import pii_reproduction, pii_software_workflow
from scripts.pii_domain_select import load_examples, parse_weighted, weighted_neighbor_mean


def test_weighted_specs_parse_paths_and_weights():
    assert parse_weighted("a/b.jsonl") == (Path("a/b.jsonl"), 1.0)
    assert parse_weighted("a/b.jsonl:2") == (Path("a/b.jsonl"), 2.0)
    assert parse_weighted("c:/x.txt") == (Path("c:/x.txt"), 1.0)
    with pytest.raises(ValueError):
        parse_weighted("a.txt:0")


def test_integer_weight_equals_repeating_examples():
    torch.manual_seed(0)
    similarity = torch.rand(7, 4)
    weights = torch.tensor([2.0, 1.0, 3.0, 1.0])
    repeated = torch.cat([similarity[:, [i] * int(w)] for i, w in enumerate(weights.tolist())], dim=1)
    expected = repeated.topk(5, dim=1).values.mean(dim=1)
    assert torch.allclose(weighted_neighbor_mean(similarity, weights, 5), expected)


def test_fractional_weight_fills_part_of_a_slot():
    similarity = torch.tensor([[0.9, 0.5]])
    # Half a copy of the 0.9 example, then the 0.5 example fills the remaining 1.5 of 2 slots.
    score = weighted_neighbor_mean(similarity, torch.tensor([0.5, 3.0]), 2)
    assert score.item() == pytest.approx((0.5 * 0.9 + 1.5 * 0.5) / 2)


def test_examples_read_jsonl_or_plain_lines_with_file_weights(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"text": "long enough example text", "lang": "el"}) + "\n")
    (tmp_path / "b.txt").write_text("another long example line\nshort\n")
    examples = load_examples([f"{tmp_path / 'a.jsonl'}:2", str(tmp_path / "b.txt")], min_chars=10)
    assert [(e["lang"], e["weight"]) for e in examples] == [("el", 2.0), (None, 1.0)]


def test_needle_sets_concatenate_with_scaled_weights(tmp_path):
    for name in ("one", "two"):
        needles = [{"name": "n", "type": "email", "weight": 3, "pattern": "x"}]
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"schema": "pii-needle-set/v1", "needles": needles})
        )
    specs = pii_software_workflow.weighted_specs([f"{tmp_path / 'one.json'}:2,{tmp_path / 'two.json'}"])
    merged = json.loads(pii_software_workflow.merged_needle_set(specs, tmp_path / "m.json").read_text())
    assert [(n["name"], n["weight"]) for n in merged["needles"]] == [("one:n", 6.0), ("two:n", 3.0)]


def test_select_defaults_to_the_source_only_draw_and_optional_selectors_take_defaults():
    parser = pii_reproduction.build_parser()
    plain = parser.parse_args(["select", "--language", "el"])
    assert plain.needles is None and plain.domain is None
    targeted = parser.parse_args(["select", "--language", "el", "--needles", "--domain"])
    assert targeted.needles == [pii_software_workflow.NEEDLE_SET]
    assert targeted.domain == [pii_software_workflow.HUMAN_GOLD_DOMAIN]
