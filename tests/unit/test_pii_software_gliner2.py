"""train-gliner2 reads the paper's GL4 recipe from its shipped run records."""

import pytest

from scripts import pii_reproduction as driver
from scripts import pii_software_workflow as workflow


@pytest.mark.parametrize("objective", sorted(workflow.GL4_RECORDS))
def test_recorded_gl4_phases(objective):
    materialize, selector, train = (
        driver.recorded_argv(path, workflow.GLINER2_SCRIPT) for path in workflow.GL4_RECORDS[objective]
    )
    # Every option the driver binds was recorded, so binding replaces rather than adds.
    for argv, bound in ((materialize, workflow.GL4_MATERIALIZE_BOUND), (train, workflow.GL4_TRAIN_BOUND)):
        assert all(name in argv for name in bound)
    kept = workflow.recorded_phase(materialize, workflow.GL4_MATERIALIZE_BOUND, "materialize")
    assert (
        kept[kept.index("--sampling-config") + 1]
        == "research/pii/frontier/evidence/gliner2-o4-v1/sampling.json"
    )
    assert kept[kept.index("--minimum-language-share") + 1] == "0.0075"
    assert "--snap-word-boundaries" in kept
    assert not set(kept) & set(workflow.GL4_MATERIALIZE_BOUND)
    workflow.recorded_phase(selector, workflow.GL4_MATERIALIZE_BOUND, "materialize")
    options = workflow.recorded_phase(train, workflow.GL4_TRAIN_BOUND, "train")
    recorded = dict(zip(options, options[1:]))
    assert (
        recorded["--label-transfer"] == "research/pii/frontier/evidence/gliner2-ont3-label-transfer-v1.json"
    )
    assert (recorded["--encoder-lr"], recorded["--task-lr"]) == ("1e-5", "2e-5")
    assert (recorded["--batch-size"], recorded["--grad-accum"]) == ("8", "4")
    assert driver.recorded_value(train, "--max-steps") == "2000"
    assert driver.recorded_value(train, "--warmup-steps") == "1000"
    accepted = "--acceptable-labels" in options
    assert accepted == (objective == "accepted")
    assert accepted == ("--acceptable-labels" in kept)
    # GL4 shuffles type order; the accepted-label variant was only run before shuffling.
    assert ("--shuffle-labels" in options) == (objective == "fallback")


def test_first_gl4_run_is_the_same_recipe_unshuffled():
    shuffled = driver.recorded_argv(workflow.GL4_RECORDS["fallback"][2], workflow.GLINER2_SCRIPT)
    first = driver.recorded_argv(workflow.GL4_UNSHUFFLED_TRAIN, workflow.GLINER2_SCRIPT)
    kept = [workflow.recorded_phase(argv, workflow.GL4_TRAIN_BOUND, "train") for argv in (shuffled, first)]
    assert kept[0] == [*kept[1], "--shuffle-labels"]


def test_wrong_phase_is_refused():
    train = driver.recorded_argv(workflow.GL4_RECORDS["fallback"][2], workflow.GLINER2_SCRIPT)
    with pytest.raises(ValueError, match="not a materialize run"):
        workflow.recorded_phase(train, workflow.GL4_TRAIN_BOUND, "materialize")
