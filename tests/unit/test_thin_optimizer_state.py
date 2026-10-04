from trainlib import thin_optimizer_state


def test_only_the_terminal_checkpoint_keeps_optimizer_state(tmp_path):
    for step in (2000, 4000, 6000):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "optimizer.pt").write_bytes(b"x" * 10)
        (checkpoint / "model.safetensors").write_bytes(b"w")
        (checkpoint / "scheduler.pt").write_bytes(b"s")
    (tmp_path / "best").symlink_to(tmp_path / "checkpoint-4000")

    thinned, freed = thin_optimizer_state(tmp_path)

    assert thinned == [2000, 4000] and freed == 20
    assert (tmp_path / "checkpoint-6000/optimizer.pt").is_file()
    for step in (2000, 4000):
        assert not (tmp_path / f"checkpoint-{step}/optimizer.pt").exists()
        assert (tmp_path / f"checkpoint-{step}/model.safetensors").is_file()
        assert (tmp_path / f"checkpoint-{step}/scheduler.pt").is_file()


def test_an_empty_run_directory_is_a_no_op(tmp_path):
    assert thin_optimizer_state(tmp_path) == ([], 0)
