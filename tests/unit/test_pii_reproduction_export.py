import json

import pytest

from scripts.pii_reproduction_export import stage_files

SENTENCE = "\n\nThe paper is at https://arxiv.org/html/1.2.\n"
PROFILE = [
    {"match": SENTENCE, "replacement": "\n"},
    {"match": "arxiv.org/html/1.2", "forbid": True},
    {"match": "Author Name", "replacement": "Anonymous Author"},
]


def stage(tmp_path, guide, readme="readme"):
    (tmp_path / "guide.md").write_text(guide)
    (tmp_path / "profile.json").write_text(json.dumps(PROFILE))
    return stage_files(
        [(tmp_path / "guide.md", "guide.md")],
        tmp_path / "out",
        readme=readme,
        redactions=[tmp_path / "profile.json"],
        anonymous=True,
    )


def test_anonymous_stage_removes_the_paper_sentence(tmp_path):
    stage(tmp_path, f"# Title\n\nIntro by Author Name.{SENTENCE}\nRest.\n")
    assert (tmp_path / "out/guide.md").read_text() == "# Title\n\nIntro by Anonymous Author.\n\nRest.\n"


def test_forbidden_text_that_survives_redaction_fails_the_stage(tmp_path):
    # A rewrapped sentence no longer matches its removal entry; the forbid entry catches the link.
    with pytest.raises(ValueError, match="identity"):
        stage(tmp_path, "Intro.\n\nThe paper is at\nhttps://arxiv.org/html/1.2.\n")


def test_omit_entries_leave_a_file_out_of_the_build(tmp_path):
    (tmp_path / "guide.md").write_text("Intro.\n")
    (tmp_path / "CITATION.cff").write_text("title: x\n")
    (tmp_path / "profile.json").write_text(json.dumps([{"omit": "CITATION.cff"}, *PROFILE]))
    files = [(tmp_path / "guide.md", "guide.md"), (tmp_path / "CITATION.cff", "CITATION.cff")]
    stage_files(files, tmp_path / "out", readme="r", redactions=[tmp_path / "profile.json"], anonymous=True)
    assert (tmp_path / "out/guide.md").exists() and not (tmp_path / "out/CITATION.cff").exists()


def stage_with_data(tmp_path, data):
    (tmp_path / "data.jsonl").write_text(data)
    (tmp_path / "guide.md").write_text("By Author Name.\n")
    (tmp_path / "profile.json").write_text(json.dumps([{"verbatim": "data/"}, *PROFILE]))
    files = [(tmp_path / "data.jsonl", "data/eval.jsonl"), (tmp_path / "guide.md", "guide.md")]
    stage_files(files, tmp_path / "out", readme="r", redactions=[tmp_path / "profile.json"], anonymous=True)


def test_verbatim_entries_ship_data_byte_for_byte(tmp_path):
    stage_with_data(tmp_path, '{"text": "Ein  Satz\\u00e9"}\n')
    assert (tmp_path / "out/data/eval.jsonl").read_bytes() == (tmp_path / "data.jsonl").read_bytes()
    assert (tmp_path / "out/guide.md").read_text() == "By Anonymous Author.\n"


def test_identity_in_verbatim_data_fails_the_stage(tmp_path):
    # Released data is never rewritten, so an identity in it must be removed at its source.
    with pytest.raises(ValueError, match="identity"):
        stage_with_data(tmp_path, '{"provenance": "/home/Author Name/eval.jsonl"}\n')


def test_public_stage_keeps_the_paper_sentence(tmp_path):
    (tmp_path / "guide.md").write_text(f"Intro.{SENTENCE}")
    stage_files(
        [(tmp_path / "guide.md", "guide.md")], tmp_path / "out", readme="r", redactions=[], anonymous=False
    )
    assert "arxiv.org/html/1.2" in (tmp_path / "out/guide.md").read_text()
