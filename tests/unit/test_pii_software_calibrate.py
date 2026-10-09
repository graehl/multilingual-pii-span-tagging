"""calibrate applied to a paper system's own receipt curves must reproduce the paper's operating point."""

import gzip
import json
from pathlib import Path

import pytest

from scripts.pii_software_calibrate import calibrate
from scripts.pii_software_receipts import OPERATING_POINTS, expand, operating_point_curves, read_json

RECEIPTS = Path(__file__).resolve().parents[2] / "research/pii/frontier/software/records/receipts"


@pytest.fixture(scope="module")
def curves():
    manifest = json.loads((RECEIPTS / "receipts.json").read_text())["receipts"]
    names = (OPERATING_POINTS["comparison"], *OPERATING_POINTS["grids"])
    reports = {name: expand(read_json(RECEIPTS / manifest[name]["receipt"])) for name in names}
    return operating_point_curves(reports)


def evaluation_directory(path: Path, curves: dict, model: str) -> Path:
    """The files evaluate writes, holding one paper system's curves."""
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"control": "o4"}))
    for name, populations in (
        ("scores-human.json.gz", ("human",)),
        ("scores-ont3.json.gz", ("ont3", "heldout")),
    ):
        scores = {
            "populations": {population: curves["populations"][population] for population in populations},
            "systems": {
                model: {population: curves["systems"][model][population] for population in populations}
            },
        }
        with gzip.open(path / name, "wt", encoding="utf-8") as stream:
            json.dump(scores, stream)
    return path


@pytest.mark.parametrize("model", ["gliner2-o4", "o3"])
def test_reproduces_paper_operating_point(tmp_path, curves, model):
    paper = json.loads((RECEIPTS / "operating-points-trust-region.json").read_text())["systems"][model]
    result = calibrate(evaluation_directory(tmp_path / "evaluation", curves, model), "silver-dev")
    assert result["selected_threshold"] == paper["selected_threshold"]
    assert result["trust_region"]["bounds"] == paper["trust_region"]["bounds"]
    for name, key in (("silver-dev", "silver_dev"), ("gold-7", "gold7"), ("silver-test", "silver_test")):
        stated = paper["evaluations"][key]["80"]["selected"]
        assert result["evaluations"][name]["selected"]["regions_80"]["F1"] == round(100 * stated["F1"], 2)
    assert result["evaluations"]["silver-dev"]["role"] == "development"
    assert result["evaluations"]["gold-7"]["role"] == "held out"
    paired = result["evaluations"]["gold-7"]["paired_versus_o4"]
    assert paired["thresholds"] == {model: paper["selected_threshold"], "o4": 0}


def test_subset_population_is_reported_unpaired(tmp_path, curves):
    """A declared subset (the demo's human gold) is scored but cannot pair with O4's full receipt."""
    from scripts.pii_software_calibrate import scored_view

    human = curves["systems"]["o3"]["human"]
    subset = {
        "populations": curves["populations"],
        "systems": {
            "o3": {**curves["systems"]["o3"], "human": {"points": scored_view(human["points"], {"en"})}}
        },
    }
    result = calibrate(evaluation_directory(tmp_path / "evaluation", subset, "o3"), "silver-dev")
    assert result["evaluations"]["gold-7"]["paired_versus_o4"] is None
    assert result["evaluations"]["gold-7"]["rows"] == curves["populations"]["human"]["languages"]["en"]
    assert result["evaluations"]["silver-test"]["paired_versus_o4"] is not None


def test_missing_development_population_is_refused(tmp_path, curves):
    evaluation = evaluation_directory(tmp_path / "evaluation", curves, "o3")
    (evaluation / "scores-ont3.json.gz").unlink()
    with pytest.raises(ValueError, match="no silver-dev scores"):
        calibrate(evaluation, "silver-dev")
