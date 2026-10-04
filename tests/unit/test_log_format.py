from log_format import headline


def test_headline_writes_structured_fields(monkeypatch, tmp_path):
    path = tmp_path / "headline.txt"
    monkeypatch.setenv("AGENTCTL_HEADLINE_FILE", str(path))

    message = headline("train", step=3, loss=0.5)

    assert message == "train | step=3 loss=0.5"
    assert path.read_text(encoding="utf-8") == message + "\n"
