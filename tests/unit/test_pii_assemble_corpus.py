import gzip
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.pii_assemble_corpus as assemble


def test_script_help_resolves_repository_imports(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(assemble.__file__)), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def write_snapshot(root, slug="openpii-1m", splits=("train",)):
    dataset = root / slug
    row = {
        "id": "example-1",
        "text": "Ada Lovelace",
        "spans": [
            {
                "start": 4,
                "end": 12,
                "label": "family_name",
                "source_label": "SURNAME",
            }
        ],
        "lang": "en",
        "metadata": {},
    }
    shards = []
    for split in splits:
        shard = dataset / split / "en.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(shard, "wt", encoding="utf-8") as output:
            output.write(json.dumps({**row, "id": f"{split}-example-1"}) + "\n")
        shards.append({"path": f"{split}/en.jsonl.gz"})
    manifest = {
        "counts": {"records": len(splits), "spans": len(splits), "languages": {"en": len(splits)}},
        "shards": shards,
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_onboarded_can_exclude_frozen_test_split(tmp_path):
    write_snapshot(tmp_path, slug="mapa", splits=("train", "validation", "test"))

    rows = list(
        assemble.load_onboarded(
            "mapa",
            root=tmp_path,
            verify=False,
            splits=("train", "validation"),
        )
    )

    assert [row[0] for row in rows] == [
        "mapa-train-example-1",
        "mapa-validation-example-1",
    ]
    assert all("test" not in row[0] for row in rows)


def test_mapa_source_is_opt_in_and_marks_partial_supervision(tmp_path, monkeypatch):
    retained = tmp_path / "prior"
    retained.mkdir()
    (retained / "labels.json").write_text(
        json.dumps({"labels": ["person_name"]}),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        assemble,
        "sources",
        lambda args, rng: [
            (
                assemble.MAPA_NATURAL_SOURCE,
                "mapa_coarse",
                (
                    (
                        "mapa-one",
                        "Ada",
                        [[0, 3, "PERSON"]],
                        "en",
                    ),
                ),
            )
        ],
    )
    args = SimpleNamespace(
        seed=0,
        source=[assemble.MAPA_NATURAL_SOURCE],
        min_node_count=1,
        val_frac=0.5,
        out=str(out),
        released_source_set="legacy-sampled",
        final20_transport=False,
        retain_labels_from=str(retained),
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    assemble.cmd_build(args)

    rows = [
        json.loads(line)
        for split in ("train", "val")
        for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    build = json.loads((out / "build.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["supervision"] == "annotated_spans_only"
    assert build["source_supervision"] == {assemble.MAPA_NATURAL_SOURCE: "annotated_spans_only"}


def test_idner_source_uses_only_train_and_marks_partial_supervision(monkeypatch):
    calls = []

    def fake_load(slug, root=assemble.ONBOARDED, verify=True, splits=None):
        calls.append((slug, splits))
        return iter(())

    monkeypatch.setattr(assemble, "load_onboarded", fake_load)
    args = SimpleNamespace(
        source=[assemble.IDNER_NATURAL_SOURCE],
        released_source_set="legacy-sampled",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    selected = []
    for name, schema, rows in assemble.sources(args, random.Random(0)):
        if name == assemble.IDNER_NATURAL_SOURCE:
            selected.append((name, schema, list(rows)))
            break

    assert selected == [
        (assemble.IDNER_NATURAL_SOURCE, "idner_news_2k", []),
    ]
    assert calls == [("idner-news-2k", ("train",))]
    assert assemble.IDNER_NATURAL_SOURCE in assemble.ANNOTATED_SPANS_ONLY_SOURCES


def test_hiner_source_uses_only_train_and_marks_partial_supervision(monkeypatch):
    calls = []

    def fake_load(slug, root=assemble.ONBOARDED, verify=True, splits=None):
        calls.append((slug, splits))
        return iter(())

    monkeypatch.setattr(assemble, "load_onboarded", fake_load)
    args = SimpleNamespace(
        source=[assemble.HINER_NATURAL_SOURCE],
        released_source_set="legacy-sampled",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    selected = []
    for name, schema, rows in assemble.sources(args, random.Random(0)):
        if name == assemble.HINER_NATURAL_SOURCE:
            selected.append((name, schema, list(rows)))
            break

    assert selected == [(assemble.HINER_NATURAL_SOURCE, "hiner_original", [])]
    assert calls == [("hiner", ("train",))]
    assert assemble.HINER_NATURAL_SOURCE in assemble.ANNOTATED_SPANS_ONLY_SOURCES


def test_wojood_source_uses_only_train_and_marks_partial_supervision(monkeypatch):
    calls = []

    def fake_load(slug, root=assemble.ONBOARDED, verify=True, splits=None):
        calls.append((slug, splits))
        return iter(())

    monkeypatch.setattr(assemble, "load_onboarded", fake_load)
    args = SimpleNamespace(
        source=[assemble.WOJOOD_NATURAL_SOURCE],
        released_source_set="legacy-sampled",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    selected = []
    for name, schema, rows in assemble.sources(args, random.Random(0)):
        if name == assemble.WOJOOD_NATURAL_SOURCE:
            selected.append((name, schema, list(rows)))
            break

    assert selected == [(assemble.WOJOOD_NATURAL_SOURCE, "wojood_nested", [])]
    assert calls == [("wojood-sample", ("train",))]
    assert assemble.WOJOOD_NATURAL_SOURCE in assemble.ANNOTATED_SPANS_ONLY_SOURCES


def test_aqmar_source_uses_only_corrected_train_and_marks_partial_supervision(monkeypatch):
    calls = []

    def fake_load(slug, root=assemble.ONBOARDED, verify=True, splits=None):
        calls.append((slug, splits))
        return iter(())

    monkeypatch.setattr(assemble, "load_onboarded", fake_load)
    args = SimpleNamespace(
        source=[assemble.AQMAR_NATURAL_SOURCE],
        released_source_set="legacy-sampled",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    selected = []
    for name, schema, rows in assemble.sources(args, random.Random(0)):
        if name == assemble.AQMAR_NATURAL_SOURCE:
            selected.append((name, schema, list(rows)))
            break

    assert selected == [(assemble.AQMAR_NATURAL_SOURCE, "aqmar_core", [])]
    assert calls == [("aqmar-openner", ("train",))]
    assert assemble.AQMAR_NATURAL_SOURCE in assemble.ANNOTATED_SPANS_ONLY_SOURCES


def test_openner_core_source_uses_only_train_and_marks_partial_supervision(monkeypatch):
    calls = []

    def fake_load(slug, root=assemble.ONBOARDED, verify=True, splits=None):
        calls.append((slug, splits))
        return iter(())

    monkeypatch.setattr(assemble, "load_onboarded", fake_load)
    args = SimpleNamespace(
        source=[assemble.OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE],
        released_source_set="legacy-sampled",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    selected = []
    for name, schema, rows in assemble.sources(args, random.Random(0)):
        if name == assemble.OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE:
            selected.append((name, schema, list(rows)))
            break

    assert selected == [
        (assemble.OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE, "openner_core", []),
    ]
    assert calls == [("openner-commercial-core", ("train",))]
    assert assemble.OPENNER_COMMERCIAL_CORE_NATURAL_SOURCE in assemble.ANNOTATED_SPANS_ONLY_SOURCES


def test_load_onboarded_uses_every_shard_and_retains_source_label(tmp_path):
    write_snapshot(tmp_path)
    rows = list(assemble.load_onboarded("openpii-1m", root=tmp_path, verify=False))
    assert rows == [
        (
            "openpii-1m-train-example-1",
            "Ada Lovelace",
            [[4, 12, "SURNAME"]],
            "en",
            {
                "path": str(tmp_path / "openpii-1m/train/en.jsonl.gz"),
                "line_1based": 1,
            },
        )
    ]


def test_onboarded_full_policy_replaces_overlapping_release_inputs(monkeypatch):
    manifests = {
        "openpii-1m": {"counts": {"languages": {"en": 1, "fr": 1}}},
        "nemotron-pii": {"counts": {"languages": {"en": 1}}},
    }
    monkeypatch.setattr(assemble, "onboarded_manifest", lambda slug, root=assemble.ONBOARDED: manifests[slug])
    args = SimpleNamespace(
        released_source_set="onboarded-full",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )
    names = [name for name, _, _ in assemble.released_sources(args, random.Random(0))]
    assert names == [
        "openpii-1m-full",
        "nemotron-full",
        "ai4p-1.5m-extra-langs",
        "ai4p-200k",
    ]
    assert "ai4p-1.5m" not in names
    assert "nemotron" not in names


def test_legacy_policy_remains_reproducible():
    args = SimpleNamespace(
        released_source_set="legacy-sampled",
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )
    names = [name for name, _, _ in assemble.released_sources(args, random.Random(0))]
    assert names == ["ai4p-1.5m", "ai4p-200k", "nemotron"]


def test_load_canonical_jsonl_retains_target_nodes_and_language(tmp_path):
    path = tmp_path / "canonical.jsonl"
    path.write_text(
        json.dumps({"id": "one", "text": "Ada", "spans": [[0, 3, "given_name"]], "lang": "en"}) + "\n",
        encoding="utf-8",
    )

    assert list(assemble.load_canonical_jsonl(path, "targeted", "fr")) == [
        (
            "targeted-one",
            "Ada",
            [[0, 3, "given_name"]],
            "en",
            {"path": str(path), "line_1based": 1},
        )
    ]


def test_final20_transport_sources_are_explicit_complete_increment(monkeypatch, tmp_path):
    monkeypatch.setattr(assemble, "PII_ANNOTATIONS", str(tmp_path))
    monkeypatch.setattr(assemble, "FINAL20_TRANSPORT", str(tmp_path / "transport"))

    sources = list(assemble.final20_transport_sources())
    names = [name for name, _, _ in sources]

    assert len(names) == len(assemble.FINAL20_BROAD_LANGUAGES) + 1 + len(assemble.FINAL20_LANGUAGES)
    assert names[:2] == ["ar-transport-final20", "cs-transport-final20"]
    assert "en-targeted-final20" in names
    assert names[-1] == "zh-targeted-final20"
    assert all(schema == "_identity" for _, schema, _ in sources)


def test_retained_label_inventory_is_canonical_and_auditable(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "one", "text": "Ada", "spans": [[0, 3, "given_name"]], "lang": "en"}) + "\n",
        encoding="utf-8",
    )
    retained = tmp_path / "prior"
    retained.mkdir()
    (retained / "labels.json").write_text(
        json.dumps({"labels": ["email", "given_name"]}),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        assemble,
        "sources",
        lambda args, rng: [("increment", "_identity", assemble.load_canonical_jsonl(source, "increment"))],
    )
    args = SimpleNamespace(
        seed=0,
        source=["increment"],
        min_node_count=1,
        val_frac=0.5,
        out=str(out),
        released_source_set="legacy-sampled",
        final20_transport=True,
        retain_labels_from=str(retained),
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
    )

    assemble.cmd_build(args)

    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    build = json.loads((out / "build.json").read_text(encoding="utf-8"))
    assert labels["labels"] == ["email", "given_name"]
    assert labels["counts"] == {"email": 0, "given_name": 1}
    assert labels["retained_labels_from"] == str(retained / "labels.json")
    assert build["retain_labels_from"] == str(retained / "labels.json")


def test_build_completes_below_half_rule_coverage_and_records_source_line(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    text = "a@b.co c@d.co e@f.co"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "one",
                        "text": text,
                        "spans": [[0, 6, "email"]],
                        "lang": "en",
                    }
                ),
                json.dumps({"id": "two", "text": "sin datos", "spans": [], "lang": "es"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        assemble,
        "sources",
        lambda args, rng: [("increment", "_identity", assemble.load_canonical_jsonl(source, "increment"))],
    )
    args = SimpleNamespace(
        seed=0,
        source=["increment"],
        min_node_count=1,
        val_frac=0.5,
        out=str(out),
        released_source_set="legacy-sampled",
        final20_transport=False,
        retain_labels_from=None,
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
        rule_completion_mode="apply",
        rule_completion_threshold=0.5,
        rule_completion_audit_size=100,
    )

    assemble.cmd_build(args)

    rows = [
        json.loads(line)
        for split in ("train", "val")
        for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads((out / "rule-completion.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    row = next(row for row in rows if row["id"] == "increment-one")
    assert row["spans"] == [[0, 6, "email"], [7, 13, "email"], [14, 20, "email"]]
    assert row["rule_completion"]["source"] == {"path": str(source), "line_1based": 1}
    assert stats["increment/lang"] == {"en": 1, "es": 1}
    assert report["sources"]["increment"]["rules"]["email"]["same_type_coverage"] == 1 / 3
    assert report["sources"]["increment"]["added_spans"] == 2


def test_translated_derivative_is_audited_but_requires_upstream_retranslation(tmp_path, monkeypatch):
    source = tmp_path / "translated.jsonl"
    source.write_text(
        json.dumps({"id": "one", "text": "a@b.co", "spans": [], "lang": "ar"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        assemble,
        "sources",
        lambda args, rng: [
            ("ar-transport-final20", "_identity", assemble.load_canonical_jsonl(source, "translated"))
        ],
    )
    args = SimpleNamespace(
        seed=0,
        source=["ar-transport-final20"],
        min_node_count=1,
        val_frac=0.5,
        out=str(out),
        released_source_set="legacy-sampled",
        final20_transport=True,
        retain_labels_from=None,
        cap_ai4p_lang=6000,
        cap_nemotron=12000,
        rule_completion_mode="apply",
        rule_completion_threshold=0.5,
        rule_completion_audit_size=100,
    )

    assemble.cmd_build(args)

    rows = [
        json.loads(line)
        for split in ("train", "val")
        for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads((out / "rule-completion.json").read_text(encoding="utf-8"))
    assert rows[0]["spans"] == []
    source_report = report["sources"]["ar-transport-final20"]
    assert source_report["proposed_spans"] == 1
    assert source_report["added_spans"] == 0
    assert "retranslate" in source_report["deferred_reason"]


def test_promoted_phone_fax_rule_audits_and_applies_explicitly(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "one", "text": "Phone: 212-555-0123", "spans": [], "lang": "en"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        assemble,
        "sources",
        lambda args, rng: [("increment", "_identity", assemble.load_canonical_jsonl(source, "increment"))],
    )
    common = {
        "seed": 0,
        "source": ["increment"],
        "min_node_count": 1,
        "val_frac": 0.5,
        "released_source_set": "legacy-sampled",
        "final20_transport": False,
        "retain_labels_from": None,
        "cap_ai4p_lang": 6000,
        "cap_nemotron": 12000,
        "rule_completion_threshold": 0.5,
        "rule_completion_audit_size": 100,
    }

    audit_out = tmp_path / "audit"
    assemble.cmd_build(SimpleNamespace(**common, out=str(audit_out), rule_completion_mode="audit"))
    report = json.loads((audit_out / "rule-completion.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for split in ("train", "val")
        for line in (audit_out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["sources"]["increment"]["rules"]["phone"]["uncovered"] == 1
    assert report["sources"]["increment"]["proposed_spans"] == 1
    assert rows[0]["spans"] == []

    apply_out = tmp_path / "apply"
    assemble.cmd_build(SimpleNamespace(**common, out=str(apply_out), rule_completion_mode="apply"))
    applied_rows = [
        json.loads(line)
        for split in ("train", "val")
        for line in (apply_out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert applied_rows[0]["spans"] == [[7, 19, "phone_number"]]
