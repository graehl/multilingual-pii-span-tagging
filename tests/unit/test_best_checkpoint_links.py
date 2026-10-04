"""`best` as a symlink, and the retention policy around the selected checkpoint."""

import json

import pytest

from trainlib import (
    checkpoint_steps,
    detach_best_directory,
    link_best_to_checkpoint,
    prune_checkpoints_before_selected,
)


def _checkpoint(root, step, payload="weights"):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "model.safetensors").write_text(payload, encoding="utf-8")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    return path


def test_best_links_to_the_checkpoint_relatively(tmp_path):
    checkpoint = _checkpoint(tmp_path, 2500)
    best = tmp_path / "best"
    link_best_to_checkpoint(best, checkpoint)
    assert best.is_symlink()
    # Relative, so moving or rsyncing the whole run directory does not break it.
    assert not best.readlink().is_absolute()
    assert (best / "model.safetensors").read_text(encoding="utf-8") == "weights"


def test_relinking_replaces_the_link_without_touching_the_old_target(tmp_path):
    first = _checkpoint(tmp_path, 2500, "first")
    second = _checkpoint(tmp_path, 5000, "second")
    best = tmp_path / "best"
    link_best_to_checkpoint(best, first)
    link_best_to_checkpoint(best, second)
    assert (best / "model.safetensors").read_text(encoding="utf-8") == "second"
    assert (first / "model.safetensors").read_text(encoding="utf-8") == "first"


def test_detach_unlinks_without_deleting_the_checkpoint(tmp_path):
    checkpoint = _checkpoint(tmp_path, 2500)
    best = tmp_path / "best"
    link_best_to_checkpoint(best, checkpoint)
    detach_best_directory(best)
    # The hazard this exists for: a later save must not write through the link into the
    # numbered checkpoint, which is both the resume point and a link target.
    assert not best.exists()
    assert (checkpoint / "model.safetensors").read_text(encoding="utf-8") == "weights"


def test_detach_removes_a_real_directory_too(tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    (best / "model.safetensors").write_text("stale", encoding="utf-8")
    detach_best_directory(best)
    assert not best.exists()


def test_pruning_keeps_the_selection_its_predecessor_and_everything_after(tmp_path):
    for step in (500, 1000, 1500, 2000, 2500, 3000, 3500):
        _checkpoint(tmp_path, step)
    dropped, freed = prune_checkpoints_before_selected(tmp_path, 2500)
    assert dropped == [500, 1000, 1500]
    assert freed > 0
    assert sorted(checkpoint_steps(tmp_path)) == [2000, 2500, 3000, 3500]


def test_pruning_never_removes_the_link_target(tmp_path):
    for step in (500, 1000, 2500):
        _checkpoint(tmp_path, step)
    best = tmp_path / "best"
    link_best_to_checkpoint(best, tmp_path / "checkpoint-2500")
    prune_checkpoints_before_selected(tmp_path, 2500)
    assert best.is_symlink()
    assert (best / "model.safetensors").read_text(encoding="utf-8") == "weights"


def test_pruning_is_a_no_op_when_the_selection_is_the_earliest(tmp_path):
    for step in (2500, 3000):
        _checkpoint(tmp_path, step)
    dropped, freed = prune_checkpoints_before_selected(tmp_path, 2500)
    assert dropped == []
    assert freed == 0


@pytest.mark.parametrize("selected", [2500, 3000])
def test_pruning_is_idempotent(tmp_path, selected):
    for step in (500, 1000, 2500, 3000):
        _checkpoint(tmp_path, step)
    prune_checkpoints_before_selected(tmp_path, selected)
    before = sorted(checkpoint_steps(tmp_path))
    prune_checkpoints_before_selected(tmp_path, selected)
    assert sorted(checkpoint_steps(tmp_path)) == before


def test_recorded_path_is_repository_relative_inside_the_repo():
    from scripts.pii_language_policy import REPO_ROOT, recorded_path

    inside = REPO_ROOT / "scripts" / "pii_language_round.yaml"
    # An absolute record says where one machine kept its checkout; the workers hold the
    # same tree under a different home and could not resolve it.
    assert recorded_path(inside) == "scripts/pii_language_round.yaml"


def test_recorded_path_stays_absolute_outside_the_repo(tmp_path):
    from scripts.pii_language_policy import recorded_path

    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("x", encoding="utf-8")
    assert recorded_path(outside) == str(outside.resolve())


def test_pruning_keeps_any_checkpoint_something_links_to(tmp_path):
    from trainlib import referenced_checkpoints

    for step in (500, 1000, 1500, 2000, 2500):
        _checkpoint(tmp_path, step)
    # An archive that links to selected steps rather than copying them is asserting it
    # still needs them; retention honours that instead of requiring the archive to teach
    # the pruner about itself.
    archive = tmp_path / "major-checkpoints"
    archive.mkdir()
    (archive / "step-000500").symlink_to("../checkpoint-500", target_is_directory=True)

    assert referenced_checkpoints(tmp_path) == {500}
    dropped, _freed = prune_checkpoints_before_selected(tmp_path, 2500)
    assert 500 not in dropped
    assert dropped == [1000, 1500]
    assert (archive / "step-000500" / "model.safetensors").is_file()


def test_referenced_checkpoints_ignores_links_out_of_the_run(tmp_path):
    from trainlib import referenced_checkpoints

    _checkpoint(tmp_path, 500)
    elsewhere = tmp_path.parent / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "stray").symlink_to(elsewhere, target_is_directory=True)
    assert referenced_checkpoints(tmp_path) == set()


def _major(root, step, payload="adapter"):
    path = root / "major-checkpoints" / f"step-{step:06d}-oa-0.8000-ia-0.5000"
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_text(payload, encoding="utf-8")
    return path


def test_save_links_declare_the_newest_majors(tmp_path):
    from trainlib import link_recent_majors

    for step in (10, 20, 60, 90, 110, 180, 450):
        _major(tmp_path, step)
    linked = link_recent_majors(tmp_path, keep=5)
    assert len(linked) == 5
    assert all(link.is_symlink() and not link.readlink().is_absolute() for link in linked)
    kept = sorted(int(link.name.split("-")[1]) for link in linked)
    assert kept == [90, 110, 180, 450, 60][-5:] or kept == [60, 90, 110, 180, 450]


def test_pruning_majors_keeps_exactly_what_is_linked(tmp_path):
    from trainlib import link_recent_majors, major_checkpoints, prune_unreferenced_majors

    for step in (10, 20, 60, 90, 110, 180, 450):
        _major(tmp_path, step)
    link_recent_majors(tmp_path, keep=5)
    dropped, freed = prune_unreferenced_majors(tmp_path)
    assert dropped == [10, 20]
    assert freed > 0
    assert sorted(major_checkpoints(tmp_path)) == [60, 90, 110, 180, 450]


def test_a_hand_added_link_pins_an_older_major(tmp_path):
    from trainlib import link_recent_majors, major_checkpoints, prune_unreferenced_majors

    for step in (10, 20, 60, 90, 110, 180, 450):
        _major(tmp_path, step)
    link_recent_majors(tmp_path, keep=5)
    # The declaration is separable from the deletion so a human can override it.
    pinned = major_checkpoints(tmp_path)[10]
    (tmp_path / "save" / "pinned-10").symlink_to(
        f"../major-checkpoints/{pinned.name}", target_is_directory=True
    )
    dropped, _freed = prune_unreferenced_majors(tmp_path)
    assert dropped == [20]
    assert 10 in major_checkpoints(tmp_path)


def test_pruning_majors_is_idempotent_and_safe_without_an_archive(tmp_path):
    from trainlib import prune_unreferenced_majors

    assert prune_unreferenced_majors(tmp_path) == ([], 0)


def _resumable(root, step):
    path = _checkpoint(root, step)
    for name in ("optimizer.pt", "scheduler.pt"):
        (path / name).write_text("state", encoding="utf-8")
    return path


def test_a_resumable_checkpoint_always_survives(tmp_path):
    from trainlib import is_valid_trainer_checkpoint

    _resumable(tmp_path, 500)
    # Selection lands on a later checkpoint saved without optimizer state, so neither the
    # selection rule nor a link would leave anything to resume from.
    _checkpoint(tmp_path, 1000)
    _checkpoint(tmp_path, 1500)
    dropped, _freed = prune_checkpoints_before_selected(tmp_path, 1500)
    assert 500 not in dropped
    assert is_valid_trainer_checkpoint(tmp_path / "checkpoint-500")


def test_the_resumable_guarantee_does_not_add_a_second_one(tmp_path):
    for step in (500, 1000, 1500, 2000):
        _resumable(tmp_path, step)
    dropped, _freed = prune_checkpoints_before_selected(tmp_path, 2000)
    # 1500 is kept as the predecessor and is resumable, so 500 and 1000 still go.
    assert dropped == [500, 1000]
