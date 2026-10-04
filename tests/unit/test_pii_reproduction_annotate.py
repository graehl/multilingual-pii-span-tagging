import pytest

from scripts import pii_reproduction


def parse(*argv):
    return pii_reproduction.build_parser().parse_args(
        ["annotate", "--input", "rows.jsonl", "--out", "out", *argv]
    )


def test_verification_replay_is_a_complete_admission_and_defaults_to_the_paper_prompt():
    args = parse("--verification-replay")
    assert args.verification_replay
    assert args.prompt_revision == "paper-prompted"


def test_verification_replay_cannot_be_combined_with_a_receipt():
    with pytest.raises(SystemExit):
        parse("--verification-replay", "--dedup-receipt", "receipt.json")


def test_annotate_still_requires_some_admission():
    with pytest.raises(SystemExit):
        parse()


def test_luna_is_shorthand_for_a_codex_model():
    assert parse("--luna", "--verification-replay").codex_model == "gpt-6-luna"
    assert parse("--codex-model", "gpt-x", "--verification-replay").codex_model == "gpt-x"
    with pytest.raises(SystemExit):
        parse("--luna", "--codex-model", "gpt-x", "--verification-replay")


def login_home(tmp_path):
    login = tmp_path / "login"
    login.mkdir()
    (login / "auth.json").write_text('{"token": "first"}')
    return login


def test_codex_home_is_built_isolated_and_refreshed(tmp_path):
    login = login_home(tmp_path)
    home = pii_reproduction.prepare_codex_home(tmp_path / "home", login)
    assert (home / "config.toml").read_bytes() == pii_reproduction.CODEX_HOME_CONFIG.read_bytes()
    assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
    (home / "skills").mkdir()
    (home / "AGENTS.md").write_text("instructions")
    (login / "auth.json").write_text('{"token": "second"}')
    pii_reproduction.prepare_codex_home(home, login)
    assert not (home / "skills").exists() and not (home / "AGENTS.md").exists()
    assert (home / "auth.json").read_text() == '{"token": "second"}'


def test_codex_home_refuses_a_directory_it_did_not_create(tmp_path):
    (tmp_path / "home").mkdir()
    with pytest.raises(ValueError, match="not created by"):
        pii_reproduction.prepare_codex_home(tmp_path / "home", login_home(tmp_path))


def test_codex_home_needs_a_login(tmp_path):
    with pytest.raises(ValueError, match="codex login"):
        pii_reproduction.prepare_codex_home(tmp_path / "home", tmp_path)


def test_codex_route_takes_the_web_receipt_and_runs_concurrently(tmp_path, monkeypatch):
    monkeypatch.setattr(pii_reproduction, "DEFAULT_CODEX_HOME", tmp_path / ".codex-pii-annotate")
    monkeypatch.setenv("CODEX_HOME", str(login_home(tmp_path)))
    captured = {}

    def run_logged(command, directory, *, settings):
        captured.update(command=command, settings=settings)
        return {}

    monkeypatch.setattr(pii_reproduction, "run_logged", run_logged)
    monkeypatch.setattr(pii_reproduction, "prompts_by_language", lambda output: {})
    monkeypatch.setattr(pii_reproduction, "training_rows", lambda *args: {})
    args = parse("--luna", "--prompt-revision", "paper-teacher", "--web-receipt", "receipt.json")
    pii_reproduction.annotate_command(args)
    command = captured["command"]

    def value(option):
        return command[command.index(option) + 1]

    assert value("--backend") == "codex" and value("--model") == "gpt-6-luna"
    assert value("--concurrency") == "6" and value("--effort") == "low"
    assert value("--web-receipt").endswith("receipt.json")
    assert value("--codex-home") == str(tmp_path / ".codex-pii-annotate")
    assert captured["settings"]["backend"] == "codex-exec"


def test_codex_route_rejects_hugging_face_options():
    with pytest.raises(ValueError, match="Hugging Face"):
        pii_reproduction.annotate_command(parse("--luna", "--model", "x", "--web-receipt", "r.json"))


def test_prompt_revisions_name_existing_prompt_files():
    root = pii_reproduction.ROOT / "prompts/pii-label"
    for revision in pii_reproduction.PROMPT_REVISIONS.values():
        for key in ("task", "catalog", "examples"):
            assert (root / revision[key]).is_file()
