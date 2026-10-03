#!/usr/bin/env python
"""Fetch, canonicalize, and verify durable PII source snapshots.

The committed artifact is intentionally not a copy of each Hugging Face
transport format. It retains the text, exact character spans, both source and
canonical labels, language, and selection-relevant metadata as deterministic
gzip-compressed JSONL shards:

    {"id", "text", "spans", "lang", "metadata"}

Each span is {"start", "end", "label", "source_label"}. The split and source
are carried by the shard path and manifest rather than repeated in every row.
"""

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from pii_projector import TAGSET_PATH, Tagset  # noqa: E402

from log_format import headline  # noqa: E402


@dataclass(frozen=True)
class SourceArtifact:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceComponent:
    name: str
    relative_root: str
    lang: str
    license: str
    url: str
    attribution: str


@dataclass(frozen=True)
class SourceSpec:
    slug: str
    repo_id: str
    revision: str
    url: str
    source_schema: str
    license: str
    fetch_backend: str
    allow_patterns: tuple[str, ...]
    ignored_labels: tuple[str, ...] = ()
    required_citations: tuple[str, ...] = ()
    components: tuple[SourceComponent, ...] = ()
    source_artifacts: tuple[SourceArtifact, ...] = ()


OPENNER_COMMERCIAL_CORE_COMPONENTS = (
    SourceComponent(
        name="AnCora Spanish",
        relative_root="AnCora/spa",
        lang="es",
        license="CC-BY-4.0",
        url="https://github.com/UniversalDependencies/UD_Spanish-AnCora",
        attribution="Taulé et al. (2008), AnCora: Multilevel Annotated Corpora for Catalan and Spanish",
    ),
    SourceComponent(
        name="GermEval 2014",
        relative_root="GermEval/deu",
        lang="de",
        license="CC-BY-4.0",
        url="https://sites.google.com/site/germeval2014ner/data",
        attribution="Benikova et al. (2014), GermEval 2014 Named Entity Recognition Shared Task",
    ),
    SourceComponent(
        name="Japanese GSD NER",
        relative_root="Japanese_GSD/jap",
        lang="ja",
        license="CC-BY-SA-4.0",
        url="https://github.com/megagonlabs/UD_Japanese-GSD",
        attribution="Asahara et al. (2018), Universal Dependencies Version 2 for Japanese",
    ),
    SourceComponent(
        name="UNER English EWT",
        relative_root="UNER_English_EWT/eng",
        lang="en",
        license="CC-BY-SA-4.0",
        url="https://github.com/UniversalNER/UNER_English-EWT",
        attribution="Mayhew et al. (2024), Universal NER",
    ),
    SourceComponent(
        name="UNER Portuguese Bosque",
        relative_root="UNER_Portuguese-Bosque/por",
        lang="pt",
        license="CC-BY-SA-4.0",
        url="https://github.com/UniversalNER/UNER_Portuguese-Bosque",
        attribution="Mayhew et al. (2024), Universal NER",
    ),
    SourceComponent(
        name="UNER Swedish Talbanken",
        relative_root="UNER_Swedish_Talkbanken/swe",
        lang="sv",
        license="CC-BY-SA-4.0",
        url="https://github.com/UniversalNER/UNER_Swedish-Talbanken",
        attribution="Mayhew et al. (2024), Universal NER",
    ),
    SourceComponent(
        name="UNER Simplified Chinese GSD",
        relative_root="UNER_Chinese_GSDSIMP/cmn",
        lang="zh",
        license="CC-BY-SA-4.0",
        url="https://github.com/UniversalNER/UNER_Chinese-GSDSIMP",
        attribution="Mayhew et al. (2024), Universal NER",
    ),
)


SOURCES = {
    "nemotron-pii": SourceSpec(
        slug="nemotron-pii",
        repo_id="nvidia/Nemotron-PII",
        revision="b70ffaf5ff39e079776134c5bf4381f00a9fd1ed",
        url="https://huggingface.co/datasets/nvidia/Nemotron-PII",
        source_schema="nemotron_pii",
        license="CC-BY-4.0",
        fetch_backend="huggingface",
        allow_patterns=("README.md", "data/*.parquet"),
    ),
    "openpii-1m": SourceSpec(
        slug="openpii-1m",
        repo_id="ai4privacy/pii-masking-openpii-1m",
        revision="ecfdc547f4a0955600cfe6ab98ba2a162207fcc0",
        url="https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m",
        source_schema="ai4privacy_new",
        license="CC-BY-4.0",
        fetch_backend="huggingface",
        allow_patterns=("README.md", "distribution.json", "data/*.jsonl"),
    ),
    "ai4privacy-health-phi-400k-sample-1k": SourceSpec(
        slug="ai4privacy-health-phi-400k-sample-1k",
        repo_id="ai4privacy/pii-masking-health-phi-400k",
        revision="f1c06d3062dfb3dbc42f38eafd739895ebc769de",
        url="https://huggingface.co/datasets/ai4privacy/pii-masking-health-phi-400k",
        source_schema="ai4privacy_health_phi",
        license="ai4privacy-commercial",
        fetch_backend="local",
        allow_patterns=(),
        source_artifacts=(
            SourceArtifact(
                path="data/train.jsonl",
                bytes=3_863_416,
                sha256="fa851ac72f1dfb67218d5b5bd5958663526969b66b0427fddf7aba420f3ad9ce",
            ),
        ),
    ),
    "mapa": SourceSpec(
        slug="mapa",
        repo_id="joelniklaus/mapa",
        revision="bbb2a0157b760465002fd12a61af81b475cd387a",
        url="https://huggingface.co/datasets/joelniklaus/mapa",
        source_schema="mapa_coarse",
        license="CC-BY-4.0",
        fetch_backend="huggingface",
        allow_patterns=("README.md", "train.jsonl", "validation.jsonl", "test.jsonl"),
    ),
    "idner-news-2k": SourceSpec(
        slug="idner-news-2k",
        repo_id="khairunnisaor/idner-news-2k",
        revision="625ff40f85b03c615c036ef11bc23cb70c79987a",
        url="https://github.com/khairunnisaor/idner-news-2k",
        source_schema="idner_news_2k",
        license="MIT",
        fetch_backend="git",
        allow_patterns=(),
    ),
    "hiner": SourceSpec(
        slug="hiner",
        repo_id="cfiltnlp/HiNER",
        revision="fdec0c85a6c39a47220932c476f969bb4e2446df",
        url="https://github.com/cfiltnlp/HiNER",
        source_schema="hiner_original",
        license="CC-BY-SA-4.0",
        fetch_backend="git",
        allow_patterns=(),
        ignored_labels=("FESTIVAL", "GAME", "LITERATURE", "MISC", "NUMEX", "TIMEX"),
    ),
    "klue-ner": SourceSpec(
        slug="klue-ner",
        repo_id="KLUE-benchmark/KLUE",
        revision="3efd98708a40ff49251fddde35453f8fbb11f536",
        url="https://github.com/KLUE-benchmark/KLUE",
        source_schema="klue_ner",
        license="CC-BY-SA-4.0",
        fetch_backend="git",
        allow_patterns=(),
        ignored_labels=("DT", "TI", "QT"),
        required_citations=("Park et al. (2021), KLUE: Korean Language Understanding Evaluation",),
    ),
    "wojood-sample": SourceSpec(
        slug="wojood-sample",
        repo_id="SinaLab/ArabicNER",
        revision="ef3a7f4e806a2d909b9107d757c5f65545ae83dc",
        url="https://github.com/SinaLab/ArabicNER",
        source_schema="wojood_nested",
        license="MIT",
        fetch_backend="git",
        allow_patterns=(),
        ignored_labels=(
            "CARDINAL",
            "CURR",
            "EVENT",
            "LAW",
            "NORP",
            "ORDINAL",
            "PERCENT",
            "PRODUCT",
            "QUANTITY",
            "UNIT",
        ),
    ),
    "aqmar-openner": SourceSpec(
        slug="aqmar-openner",
        repo_id="bltlab/open-ner-core-types",
        revision="59ce4c55fc54c241e45d2d2c6f1a42f5eaf69719",
        url="https://huggingface.co/datasets/bltlab/open-ner-core-types",
        source_schema="aqmar_core",
        license="citation-required",
        fetch_backend="huggingface",
        allow_patterns=("README.md", "AQMAR/ara/*.parquet"),
        required_citations=(
            "Mohit et al. (2012), Recall-Oriented Learning of Named Entities in Arabic Wikipedia",
        ),
    ),
    "openner-commercial-core": SourceSpec(
        slug="openner-commercial-core",
        repo_id="bltlab/open-ner-core-types",
        revision="59ce4c55fc54c241e45d2d2c6f1a42f5eaf69719",
        url="https://huggingface.co/datasets/bltlab/open-ner-core-types",
        source_schema="openner_core",
        license="CC-BY-4.0 collection; CC-BY-4.0 and CC-BY-SA-4.0 components",
        fetch_backend="huggingface",
        allow_patterns=(
            "README.md",
            "AnCora/spa/*.parquet",
            "GermEval/deu/*.parquet",
            "Japanese_GSD/jap/*.parquet",
            "UNER_English_EWT/eng/*.parquet",
            "UNER_Portuguese-Bosque/por/*.parquet",
            "UNER_Swedish_Talkbanken/swe/*.parquet",
            "UNER_Chinese_GSDSIMP/cmn/*.parquet",
        ),
        required_citations=("Palen-Michel et al. (2025), OpenNER 1.0",),
        components=OPENNER_COMMERCIAL_CORE_COMPONENTS,
    ),
}

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
AI4PRIVACY_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")


def parse_spans(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_artifact_manifest(spec: SourceSpec) -> list[dict]:
    return [artifact.__dict__ for artifact in spec.source_artifacts]


def validate_source_artifacts(spec: SourceSpec, source_root: Path) -> list[dict]:
    expected = source_artifact_manifest(spec)
    source_root_resolved = source_root.resolve()
    seen = set()
    for artifact in spec.source_artifacts:
        relative_path = Path(artifact.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"source artifact path must stay relative to the source root: {artifact.path!r}")
        if artifact.path in seen:
            raise ValueError(f"duplicate source artifact path {artifact.path!r}")
        seen.add(artifact.path)
        path = source_root / relative_path
        try:
            path.resolve(strict=True).relative_to(source_root_resolved)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"source artifact is missing or escapes its source root: {artifact.path!r}"
            ) from error
        if not path.is_file():
            raise ValueError(f"source artifact is not a file: {artifact.path!r}")
        observed_bytes = path.stat().st_size
        if observed_bytes != artifact.bytes:
            raise ValueError(
                f"source artifact {artifact.path!r} byte size {observed_bytes} != pinned {artifact.bytes}"
            )
        observed_sha256 = sha256(path)
        if observed_sha256 != artifact.sha256:
            raise ValueError(f"source artifact {artifact.path!r} SHA-256 differs from the pinned input")
    return expected


def tagset_sha256() -> str:
    return sha256(Path(TAGSET_PATH))


def source_schema_sha256(source_schema: str) -> str:
    source_map = Tagset().sources[source_schema]
    encoded = json.dumps(
        source_map,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_projection_sha256(spec: SourceSpec) -> str:
    projection = {
        "mapping": Tagset().sources[spec.source_schema],
        "ignored_labels": sorted(spec.ignored_labels),
    }
    if spec.components:
        projection["components"] = [component.__dict__ for component in spec.components]
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_spans(
    text: str,
    raw_spans,
    source_map: dict[str, str],
    anomalies: Counter | None = None,
) -> list[dict]:
    spans = []
    for raw in parse_spans(raw_spans) or []:
        start = int(raw["start"])
        end = int(raw["end"])
        source_label = str(raw["label"])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"invalid span [{start}, {end}) for text length {len(text)}")
        expected = raw.get("text", raw.get("value"))
        if expected is not None and text[start:end] != str(expected):
            if text[start:end].casefold() == str(expected).casefold():
                if anomalies is not None:
                    anomalies["casefold_source_value"] += 1
            else:
                raise ValueError(
                    f"span text mismatch at [{start}, {end}): {text[start:end]!r} != {str(expected)!r}"
                )
        try:
            label = source_map[source_label]
        except KeyError as error:
            raise ValueError(f"unmapped {source_label!r}") from error
        spans.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "source_label": source_label,
            }
        )
    spans.sort(key=lambda span: (span["start"], span["end"], span["source_label"]))
    return spans


def nemotron_rows(source_root: Path, source_map: dict[str, str], anomalies: Counter):
    import pyarrow.parquet as pq

    files = sorted((source_root / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Nemotron parquet files under {source_root / 'data'}")
    for path in files:
        split = path.name.split("-", 1)[0]
        if split not in {"train", "test"}:
            raise ValueError(f"cannot infer Nemotron split from {path.name}")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048):
            for raw in batch.to_pylist():
                text = raw["text"]
                yield (
                    split,
                    {
                        "id": str(raw["uid"]),
                        "text": text,
                        "spans": normalized_spans(text, raw["spans"], source_map, anomalies),
                        "lang": "en",
                        "metadata": {
                            "domain": raw["domain"],
                            "document_type": raw["document_type"],
                            "document_description": raw["document_description"],
                            "document_format": raw["document_format"],
                            "locale": raw["locale"],
                        },
                    },
                )


def openpii_rows(
    source_root: Path,
    source_map: dict[str, str],
    anomalies: Counter,
    metadata_fields: tuple[str, ...] = ("region", "script"),
    count_source_placeholders: bool = False,
):
    files = sorted((source_root / "data").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no AI4Privacy JSONL files under {source_root / 'data'}")
    for path in files:
        expected_split = path.stem
        if expected_split not in {"train", "validation"}:
            raise ValueError(f"cannot infer AI4Privacy split from {path.name}")
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                raw = json.loads(line)
                split = raw["split"]
                if split != expected_split:
                    raise ValueError(f"{path}:{line_number}: row split {split!r} != {expected_split!r}")
                text = raw["source_text"]
                if count_source_placeholders:
                    anomalies["source_literal_placeholder"] += len(AI4PRIVACY_PLACEHOLDER.findall(text))
                lang = raw["language"]
                yield (
                    split,
                    {
                        "id": str(raw["uid"]),
                        "text": text,
                        "spans": normalized_spans(text, raw["privacy_mask"], source_map, anomalies),
                        "lang": lang,
                        "metadata": {field: raw[field] for field in metadata_fields},
                    },
                )


def bio_spans(
    tokens: list[str],
    labels: list[str],
    source_map: dict[str, str],
    source_name: str = "MAPA",
    ignored_labels: frozenset[str] = frozenset(),
) -> tuple[str, list[dict]]:
    if len(tokens) != len(labels):
        raise ValueError(f"{source_name} token/label length mismatch: {len(tokens)} != {len(labels)}")
    text = " ".join(tokens)
    token_starts = []
    offset = 0
    for token in tokens:
        token_starts.append(offset)
        offset += len(token) + 1

    spans = []
    active_label = None
    active_start = None
    active_end = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is not None:
            spans.append(
                {
                    "start": active_start,
                    "end": active_end,
                    "label": source_map[active_label],
                    "source_label": active_label,
                }
            )
        active_label = active_start = active_end = None

    for token_index, (token, tag) in enumerate(zip(tokens, labels)):
        if tag == "O":
            close_active()
            continue
        try:
            boundary, source_label = tag.split("-", 1)
        except ValueError as error:
            raise ValueError(f"invalid {source_name} BIO tag {tag!r}") from error
        if source_label in ignored_labels:
            close_active()
            continue
        if source_label not in source_map:
            raise ValueError(f"unmapped {source_name} label {source_label!r}")
        start = token_starts[token_index]
        end = start + len(token)
        if boundary == "B":
            close_active()
            active_label = source_label
            active_start = start
            active_end = end
        elif boundary == "I":
            if active_label != source_label:
                raise ValueError(
                    f"invalid {source_name} continuation {tag!r} after "
                    f"{active_label!r} at token {token_index}"
                )
            active_end = end
        else:
            raise ValueError(f"invalid {source_name} BIO boundary {boundary!r}")
    close_active()
    return text, spans


def mapa_rows(source_root: Path, source_map: dict[str, str], anomalies: Counter):
    del anomalies
    files = [source_root / f"{split}.jsonl" for split in ("train", "validation", "test")]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MAPA JSONL files: {', '.join(missing)}")
    for path in files:
        split = path.stem
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                raw = json.loads(line)
                tokens = [str(token) for token in raw["tokens"]]
                coarse = [str(label) for label in raw["coarse_grained"]]
                fine = [str(label) for label in raw["fine_grained"]]
                if len(tokens) != len(fine):
                    raise ValueError(
                        f"{path}:{line_number}: MAPA token/fine-label length mismatch: "
                        f"{len(tokens)} != {len(fine)}"
                    )
                text, spans = bio_spans(tokens, coarse, source_map)
                lang = str(raw["language"])
                file_name = str(raw["file_name"])
                sentence_number = int(raw["sentence_number"])
                yield (
                    split,
                    {
                        "id": f"{lang}:{file_name}:{sentence_number}",
                        "text": text,
                        "spans": spans,
                        "lang": lang,
                        "metadata": {
                            "document_type": raw["type"],
                            "file_name": file_name,
                            "sentence_number": sentence_number,
                            "tokens": tokens,
                            "fine_grained": fine,
                            "text_reconstruction": "single_space_join",
                            "source_line": line_number,
                        },
                    },
                )


def idner_rows(source_root: Path, source_map: dict[str, str], anomalies: Counter):
    del anomalies
    files = {
        "train": source_root / "train.txt",
        "validation": source_root / "dev.txt",
        "test": source_root / "test.txt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing idner-news-2k CoNLL files: {', '.join(missing)}")

    for split, path in files.items():
        tokens: list[str] = []
        pos_tags: list[str] = []
        labels: list[str] = []
        sentence_number = 0

        def finish_sentence():
            nonlocal sentence_number
            if not tokens:
                return None
            sentence_number += 1
            text, spans = bio_spans(
                tokens,
                labels,
                source_map,
                source_name="idner-news-2k",
            )
            row = {
                "id": f"id:{split}:{sentence_number}",
                "text": text,
                "spans": spans,
                "lang": "id",
                "metadata": {
                    "sentence_number": sentence_number,
                    "tokens": list(tokens),
                    "pos_tags": list(pos_tags),
                    "text_reconstruction": "single_space_join",
                },
            }
            tokens.clear()
            pos_tags.clear()
            labels.clear()
            return split, row

        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped:
                    completed = finish_sentence()
                    if completed is not None:
                        yield completed
                    continue
                fields = stripped.split()
                if len(fields) != 3:
                    raise ValueError(
                        f"{path}:{line_number}: expected token, POS, and BIO tag; found {len(fields)} fields"
                    )
                token, pos_tag, label = fields
                tokens.append(token)
                pos_tags.append(pos_tag)
                labels.append(label)
        completed = finish_sentence()
        if completed is not None:
            yield completed


def hiner_rows(
    source_root: Path,
    source_map: dict[str, str],
    anomalies: Counter,
    ignored_labels: tuple[str, ...],
):
    del anomalies
    files = {
        split: source_root / "data" / "original" / f"{split}.conll"
        for split in ("train", "validation", "test")
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HiNER CoNLL files: {', '.join(missing)}")

    ignored = frozenset(ignored_labels)
    observed_labels = set()
    for split, path in files.items():
        tokens: list[str] = []
        labels: list[str] = []
        sentence_number = 0

        def finish_sentence():
            nonlocal sentence_number
            if not tokens:
                return None
            sentence_number += 1
            text, spans = bio_spans(
                tokens,
                labels,
                source_map,
                source_name="HiNER",
                ignored_labels=ignored,
            )
            row = {
                "id": f"hi:{split}:{sentence_number}",
                "text": text,
                "spans": spans,
                "lang": "hi",
                "metadata": {
                    "sentence_number": sentence_number,
                    "tokens": list(tokens),
                    "bio_tags": list(labels),
                    "text_reconstruction": "single_space_join",
                },
            }
            tokens.clear()
            labels.clear()
            return split, row

        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.rstrip("\r\n")
                if not stripped:
                    completed = finish_sentence()
                    if completed is not None:
                        yield completed
                    continue
                fields = stripped.split("\t")
                if len(fields) != 2:
                    raise ValueError(
                        f"{path}:{line_number}: expected token and BIO tag; found {len(fields)} fields"
                    )
                token, label = fields
                if label != "O":
                    try:
                        _, source_label = label.split("-", 1)
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: invalid HiNER BIO tag {label!r}") from error
                    observed_labels.add(source_label)
                tokens.append(token)
                labels.append(label)
        completed = finish_sentence()
        if completed is not None:
            yield completed

    expected_labels = set(source_map) | ignored
    if observed_labels != expected_labels:
        missing_labels = sorted(expected_labels - observed_labels)
        unknown_labels = sorted(observed_labels - expected_labels)
        raise ValueError(
            "HiNER source-label inventory differs from the pinned projection: "
            f"missing={missing_labels}, unknown={unknown_labels}"
        )


KLUE_SENTENCE_HEADER = re.compile(r"^## (?P<record_id>klue-ner\S*)\t(?P<tagged>.*)$")
KLUE_INLINE_TAG = re.compile(r"<([^<>]*?):([A-Z]{2})>")


def klue_char_runs(characters: list[str], labels: list[str]) -> tuple[str, list[tuple[str, int, int]]]:
    """Convert KLUE's per-character BIO track into (label, start, end) runs.

    KLUE annotates every character, so concatenation reproduces the exact
    original text and run indices are already exact character offsets.
    """
    if len(characters) != len(labels):
        raise ValueError(f"KLUE character/label length mismatch: {len(characters)} != {len(labels)}")
    text = "".join(characters)
    runs: list[tuple[str, int, int]] = []
    active_label = None
    active_start = None

    def close_active(end: int) -> None:
        nonlocal active_label, active_start
        if active_label is not None:
            runs.append((active_label, active_start, end))
        active_label = active_start = None

    for index, tag in enumerate(labels):
        if tag == "O":
            close_active(index)
            continue
        boundary, _, source_label = tag.partition("-")
        if not source_label:
            raise ValueError(f"invalid KLUE BIO tag {tag!r}")
        if boundary == "B":
            close_active(index)
            active_label = source_label
            active_start = index
        elif boundary == "I":
            if active_label != source_label:
                raise ValueError(
                    f"invalid KLUE continuation {tag!r} after {active_label!r} at character {index}"
                )
        else:
            raise ValueError(f"invalid KLUE BIO boundary {boundary!r}")
    close_active(len(labels))
    return text, runs


def klue_rows(
    source_root: Path,
    source_map: dict[str, str],
    anomalies: Counter,
    ignored_labels: tuple[str, ...],
):
    data_root = source_root / "klue_benchmark" / "klue-ner-v1.1"
    files = {
        "train": data_root / "klue-ner-v1.1_train.tsv",
        "validation": data_root / "klue-ner-v1.1_dev.tsv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing KLUE NER TSV files: {', '.join(missing)}")

    ignored = frozenset(ignored_labels)
    observed_labels = set()
    for split, path in files.items():
        record_id = None
        tagged_header = None
        characters: list[str] = []
        labels: list[str] = []
        sentence_number = 0

        def finish_sentence():
            nonlocal record_id, tagged_header, sentence_number
            if record_id is None:
                if characters:
                    raise ValueError(f"{path}: KLUE character lines without a sentence header")
                return None
            if not characters:
                raise ValueError(f"{path}: KLUE header {record_id!r} has no character lines")
            sentence_number += 1
            text, runs = klue_char_runs(characters, labels)
            # The per-character track is the annotation of record; the inline
            # header markup is display-only and disagrees when literal angle
            # brackets in the source text collide with the markup syntax
            # (movie titles like `<세라핀>`, emoticons like `><` in NSMC
            # reviews). Keep the char-track text/spans and count both kinds
            # of disagreement instead of rejecting.
            text_mismatch = KLUE_INLINE_TAG.sub(r"\1", tagged_header) != text
            if text_mismatch:
                anomalies["header_text_mismatch"] += 1
            markup = [(match.group(1), match.group(2)) for match in KLUE_INLINE_TAG.finditer(tagged_header)]
            markup_mismatch = [(text[start:end], label) for label, start, end in runs] != markup
            if markup_mismatch and not text_mismatch:
                anomalies["header_markup_mismatch"] += 1
            spans = []
            for source_label, start, end in runs:
                observed_labels.add(source_label)
                if source_label in ignored:
                    continue
                if source_label not in source_map:
                    raise ValueError(f"unmapped KLUE label {source_label!r}")
                spans.append(
                    {
                        "start": start,
                        "end": end,
                        "label": source_map[source_label],
                        "source_label": source_label,
                    }
                )
            metadata = {
                "sentence_number": sentence_number,
                "source_document": record_id.rsplit("_", 1)[-1],
                "text_reconstruction": "character_concatenation",
            }
            if text_mismatch:
                metadata["header_text_mismatch"] = True
            elif markup_mismatch:
                metadata["header_markup_mismatch"] = True
            row = {
                "id": record_id,
                "text": text,
                "spans": spans,
                "lang": "ko",
                "metadata": metadata,
            }
            result = (split, row)
            record_id = None
            tagged_header = None
            characters.clear()
            labels.clear()
            return result

        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                line = line.rstrip("\n")
                if not line:
                    completed = finish_sentence()
                    if completed is not None:
                        yield completed
                    continue
                if line.startswith("## "):
                    header = KLUE_SENTENCE_HEADER.match(line)
                    if header is None:
                        continue
                    if record_id is not None or characters:
                        raise ValueError(f"{path}:{line_number}: KLUE header inside an open sentence")
                    record_id = header.group("record_id")
                    tagged_header = header.group("tagged")
                    continue
                character, delimiter, tag = line.rpartition("\t")
                if not delimiter or len(character) != 1:
                    raise ValueError(f"{path}:{line_number}: expected single character and BIO tag")
                if record_id is None:
                    raise ValueError(f"{path}:{line_number}: KLUE character line before any header")
                characters.append(character)
                labels.append(tag)
        completed = finish_sentence()
        if completed is not None:
            yield completed

    expected_labels = set(source_map) | ignored
    if observed_labels != expected_labels:
        missing_labels = sorted(expected_labels - observed_labels)
        unknown_labels = sorted(observed_labels - expected_labels)
        raise ValueError(
            "KLUE NER source-label inventory differs from the pinned projection: "
            f"missing={missing_labels}, unknown={unknown_labels}"
        )


def aqmar_rows(source_root: Path, source_map: dict[str, str], anomalies: Counter):
    """Read Liu et al.'s BIO-corrected AQMAR as packaged by OpenNER 1.0."""
    del anomalies
    import pyarrow.parquet as pq

    label_names = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
    split_files = {
        "train": source_root / "AQMAR" / "ara" / "train-00000-of-00001.parquet",
        "validation": source_root / "AQMAR" / "ara" / "dev-00000-of-00001.parquet",
        "test": source_root / "AQMAR" / "ara" / "test-00000-of-00001.parquet",
    }
    missing = [str(path) for path in split_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing OpenNER AQMAR parquet files: {', '.join(missing)}")

    for split, path in split_files.items():
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        try:
            features = json.loads(metadata[b"huggingface"])["info"]["features"]
            observed_label_names = features["ner_tags"]["feature"]["names"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: missing OpenNER Hugging Face label metadata") from error
        if observed_label_names != label_names:
            raise ValueError(
                f"{path}: OpenNER AQMAR labels changed: {observed_label_names!r} != {label_names!r}"
            )

        for batch in parquet.iter_batches(batch_size=2048):
            for raw in batch.to_pylist():
                tokens = [str(token) for token in raw["tokens"]]
                try:
                    labels = [label_names[int(tag)] for tag in raw["ner_tags"]]
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}: invalid OpenNER AQMAR tag index") from error
                text, spans = bio_spans(
                    tokens,
                    labels,
                    source_map,
                    source_name="OpenNER AQMAR",
                )
                yield (
                    split,
                    {
                        "id": f"ar:aqmar:{split}:{raw['id']}",
                        "text": text,
                        "spans": spans,
                        "lang": "ar",
                        "metadata": {
                            "source_record_id": int(raw["id"]),
                            "tokens": tokens,
                            "bio_tags": labels,
                            "text_reconstruction": "single_space_join",
                            "bio_provenance": "Liu_et_al_2019_corrected_via_OpenNER_1.0",
                        },
                    },
                )


def openner_core_rows(
    source_root: Path,
    source_map: dict[str, str],
    anomalies: Counter,
    components: tuple[SourceComponent, ...],
):
    """Read a license-screened multilingual subset of OpenNER's core release."""
    del anomalies
    import pyarrow.parquet as pq

    label_names = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
    for component in components:
        component_root = source_root / component.relative_root
        split_files = {
            "train": component_root / "train-00000-of-00001.parquet",
            "validation": component_root / "dev-00000-of-00001.parquet",
            "test": component_root / "test-00000-of-00001.parquet",
        }
        missing = [str(path) for path in split_files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OpenNER {component.name} parquet files: {', '.join(missing)}")

        component_id = component.relative_root.replace("/", "-").lower()
        for split, path in split_files.items():
            parquet = pq.ParquetFile(path)
            metadata = parquet.schema_arrow.metadata or {}
            try:
                features = json.loads(metadata[b"huggingface"])["info"]["features"]
                observed_label_names = features["ner_tags"]["feature"]["names"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}: missing OpenNER Hugging Face label metadata") from error
            if observed_label_names != label_names:
                raise ValueError(
                    f"{path}: OpenNER core labels changed: {observed_label_names!r} != {label_names!r}"
                )

            for batch in parquet.iter_batches(batch_size=2048):
                for raw in batch.to_pylist():
                    tokens = [str(token) for token in raw["tokens"]]
                    try:
                        labels = [label_names[int(tag)] for tag in raw["ner_tags"]]
                    except (IndexError, TypeError, ValueError) as error:
                        raise ValueError(f"{path}: invalid OpenNER core tag index") from error
                    text, spans = bio_spans(
                        tokens,
                        labels,
                        source_map,
                        source_name=f"OpenNER {component.name}",
                    )
                    yield (
                        split,
                        {
                            "id": f"{component.lang}:openner:{component_id}:{split}:{raw['id']}",
                            "text": text,
                            "spans": spans,
                            "lang": component.lang,
                            "metadata": {
                                "source_component": component.name,
                                "source_record_id": int(raw["id"]),
                                "tokens": tokens,
                                "bio_tags": labels,
                                "text_reconstruction": "single_space_join",
                                "bio_provenance": "OpenNER_1.0_standardized_and_repaired",
                            },
                        },
                    )


def wojood_spans(
    tokens: list[str],
    token_tags: list[list[str]],
    source_map: dict[str, str],
    anomalies: Counter,
) -> tuple[str, list[dict]]:
    """Project Wojood's per-type nested BIO tracks into overlapping spans."""
    text = " ".join(tokens)
    spans = []
    for source_label in source_map:
        labels = []
        for tags in token_tags:
            matches = [tag for tag in tags if tag != "O" and tag.split("-", 1)[-1] == source_label]
            if len(matches) > 1:
                anomalies["same_type_nested_tag_dropped"] += len(matches) - 1
            labels.append(matches[0] if matches else "O")
        active = False
        for index, tag in enumerate(labels):
            if tag.startswith("I-") and not active:
                labels[index] = f"B-{source_label}"
                anomalies["orphan_i_repaired"] += 1
            active = labels[index] != "O"
        reconstructed, type_spans = bio_spans(
            tokens,
            labels,
            source_map,
            source_name="Wojood",
        )
        if reconstructed != text:
            raise AssertionError("Wojood type projection changed reconstructed text")
        spans.extend(type_spans)
    spans.sort(key=lambda span: (span["start"], span["end"], span["source_label"]))
    return text, spans


def wojood_rows(
    source_root: Path,
    source_map: dict[str, str],
    anomalies: Counter,
    ignored_labels: tuple[str, ...],
):
    split_files = {
        "train": source_root / "data" / "train.txt",
        "validation": source_root / "data" / "val.txt",
        "test": source_root / "data" / "test.txt",
    }
    missing = [str(path) for path in split_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Wojood sample files: {', '.join(missing)}")

    ignored = frozenset(ignored_labels)
    observed_labels = set()
    for split, path in split_files.items():
        tokens: list[str] = []
        token_tags: list[list[str]] = []
        sentence_number = 0

        def finish_sentence():
            nonlocal sentence_number
            if not tokens:
                return None
            sentence_number += 1
            text, spans = wojood_spans(tokens, token_tags, source_map, anomalies)
            row = {
                "id": f"ar:{split}:{sentence_number}",
                "text": text,
                "spans": spans,
                "lang": "ar",
                "metadata": {
                    "sentence_number": sentence_number,
                    "tokens": list(tokens),
                    "nested_bio_tags": [list(tags) for tags in token_tags],
                    "text_reconstruction": "single_space_join",
                    "same_type_projection": "first_tag_per_official_nested_loader",
                },
            }
            tokens.clear()
            token_tags.clear()
            return split, row

        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped:
                    completed = finish_sentence()
                    if completed is not None:
                        yield completed
                    continue
                fields = stripped.split()
                if len(fields) < 2:
                    raise ValueError(f"{path}:{line_number}: expected token and one or more BIO tags")
                token, tags = fields[0], fields[1:]
                for tag in tags:
                    if tag == "O":
                        continue
                    try:
                        boundary, source_label = tag.split("-", 1)
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: invalid Wojood BIO tag {tag!r}") from error
                    if boundary not in {"B", "I"}:
                        raise ValueError(f"{path}:{line_number}: invalid Wojood BIO boundary {boundary!r}")
                    observed_labels.add(source_label)
                    if source_label not in source_map and source_label not in ignored:
                        raise ValueError(f"{path}:{line_number}: unmapped Wojood label {source_label!r}")
                tokens.append(token)
                token_tags.append(tags)
        completed = finish_sentence()
        if completed is not None:
            yield completed

    expected_labels = set(source_map) | ignored
    if observed_labels != expected_labels:
        missing_labels = sorted(expected_labels - observed_labels)
        unknown_labels = sorted(observed_labels - expected_labels)
        raise ValueError(
            "Wojood source-label inventory differs from the pinned projection: "
            f"missing={missing_labels}, unknown={unknown_labels}"
        )


class ShardWriter:
    def __init__(self, root: Path, split: str, lang: str):
        for component in (split, lang):
            if not SAFE_COMPONENT.fullmatch(component):
                raise ValueError(f"unsafe shard component {component!r}")
        self.relative_path = Path(split) / f"{lang}.jsonl.gz"
        self.path = root / self.relative_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = self.path.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.output = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        self.records = 0
        self.spans = 0

    def write(self, row: dict) -> None:
        self.output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        self.output.write("\n")
        self.records += 1
        self.spans += len(row["spans"])

    def close(self) -> dict:
        self.output.close()
        self.raw.close()
        return {
            "path": self.relative_path.as_posix(),
            "bytes": self.path.stat().st_size,
            "sha256": sha256(self.path),
            "records": self.records,
            "spans": self.spans,
        }


def source_rows(
    spec: SourceSpec,
    source_root: Path,
    source_map: dict[str, str],
    anomalies: Counter,
):
    if spec.slug == "nemotron-pii":
        return nemotron_rows(source_root, source_map, anomalies)
    if spec.slug == "openpii-1m":
        return openpii_rows(source_root, source_map, anomalies)
    if spec.slug == "ai4privacy-health-phi-400k-sample-1k":
        return openpii_rows(
            source_root,
            source_map,
            anomalies,
            metadata_fields=("region", "script", "source_dataset"),
            count_source_placeholders=True,
        )
    if spec.slug == "mapa":
        return mapa_rows(source_root, source_map, anomalies)
    if spec.slug == "idner-news-2k":
        return idner_rows(source_root, source_map, anomalies)
    if spec.slug == "hiner":
        return hiner_rows(source_root, source_map, anomalies, spec.ignored_labels)
    if spec.slug == "klue-ner":
        return klue_rows(source_root, source_map, anomalies, spec.ignored_labels)
    if spec.slug == "wojood-sample":
        return wojood_rows(source_root, source_map, anomalies, spec.ignored_labels)
    if spec.slug == "aqmar-openner":
        return aqmar_rows(source_root, source_map, anomalies)
    if spec.slug == "openner-commercial-core":
        return openner_core_rows(source_root, source_map, anomalies, spec.components)
    raise AssertionError(spec.slug)


def disambiguate_record_id(
    row: dict,
    occurrences: Counter,
    output_ids: set[str],
    anomalies: Counter,
) -> None:
    upstream_id = row["id"]
    occurrences[upstream_id] += 1
    occurrence = occurrences[upstream_id]
    if occurrence > 1:
        row["metadata"]["upstream_id"] = upstream_id
        row["id"] = f"{upstream_id}~{occurrence}"
        anomalies["duplicate_upstream_id"] += 1
    if row["id"] in output_ids:
        raise ValueError(f"record id collision after disambiguation: {row['id']!r}")
    output_ids.add(row["id"])


def build(spec: SourceSpec, source_root: Path, output_root: Path, *, max_records: int = 0) -> Path:
    if max_records < 0:
        raise ValueError("max_records must be nonnegative")
    destination = output_root / spec.slug
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; verify it with `check` or remove it explicitly")
    source_artifacts = validate_source_artifacts(spec, source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spec.slug}.", dir=output_root))
    tagset = Tagset()
    source_map = tagset.sources[spec.source_schema]
    writers = {}
    id_occurrences = Counter()
    output_ids = set()
    counts = Counter()
    canonical_labels = Counter()
    source_labels = Counter()
    anomalies = Counter()
    try:
        rows = source_rows(spec, source_root, source_map, anomalies)
        for split, row in islice(rows, max_records or None):
            disambiguate_record_id(row, id_occurrences, output_ids, anomalies)
            key = (split, row["lang"])
            writer = writers.get(key)
            if writer is None:
                writer = writers[key] = ShardWriter(temporary, *key)
            writer.write(row)
            counts["records"] += 1
            counts["spans"] += len(row["spans"])
            counts[f"split:{split}"] += 1
            counts[f"lang:{row['lang']}"] += 1
            for span in row["spans"]:
                canonical_labels[span["label"]] += 1
                source_labels[span["source_label"]] += 1
            if counts["records"] % 25_000 == 0:
                message = f"PII onboard {spec.slug}: {counts['records']:,} records, {counts['spans']:,} spans"
                print(message, flush=True)
                headline(message)
        shards = [writers[key].close() for key in sorted(writers)]
        writers.clear()
        manifest = {
            "schema_version": 1,
            "dataset": spec.slug,
            "selection": {"max_records": max_records, "order": "source iterator", "smoke": bool(max_records)},
            "upstream": {
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "url": spec.url,
                "license": spec.license,
                "required_citations": list(spec.required_citations),
            },
            "source_artifacts": source_artifacts,
            "source_schema": spec.source_schema,
            "tagset_sha256": tagset_sha256(),
            "source_schema_sha256": source_schema_sha256(spec.source_schema),
            "source_projection_sha256": source_projection_sha256(spec),
            "ignored_source_labels": list(spec.ignored_labels),
            "record_schema": {
                "fields": ["id", "text", "spans", "lang", "metadata"],
                "span_fields": ["start", "end", "label", "source_label"],
                "compression": "gzip",
                "encoding": "UTF-8",
            },
            "counts": {
                "records": counts["records"],
                "spans": counts["spans"],
                "splits": {
                    key.removeprefix("split:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("split:")
                },
                "languages": {
                    key.removeprefix("lang:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("lang:")
                },
                "canonical_labels": dict(canonical_labels.most_common()),
                "source_labels": dict(source_labels.most_common()),
                "source_anomalies": dict(anomalies.most_common()),
            },
            "shards": shards,
        }
        if spec.components:
            manifest["upstream"]["components"] = [component.__dict__ for component in spec.components]
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        for writer in writers.values():
            writer.output.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    message = (
        f"PII onboard {spec.slug} complete: {counts['records']:,} records, "
        f"{counts['spans']:,} spans, {len(shards)} shards -> {destination}"
    )
    print(message)
    headline(message)
    return destination


def validate_row(row: dict, source_map: dict[str, str], path: Path, line_number: int) -> None:
    if set(row) != {"id", "text", "spans", "lang", "metadata"}:
        raise ValueError(f"{path}:{line_number}: unexpected record fields {sorted(row)}")
    text = row["text"]
    for span in row["spans"]:
        if set(span) != {"start", "end", "label", "source_label"}:
            raise ValueError(f"{path}:{line_number}: unexpected span fields {sorted(span)}")
        start, end = span["start"], span["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{path}:{line_number}: invalid span [{start}, {end})")
        expected_label = source_map.get(span["source_label"])
        if span["label"] != expected_label:
            raise ValueError(
                f"{path}:{line_number}: {span['source_label']!r} maps to "
                f"{expected_label!r}, not {span['label']!r}"
            )


def check(spec: SourceSpec, output_root: Path) -> dict:
    root = output_root / spec.slug
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["upstream"]["repo_id"] != spec.repo_id:
        raise ValueError("manifest repository differs from the pinned source repository")
    if manifest["upstream"]["revision"] != spec.revision:
        raise ValueError("manifest upstream revision differs from the pinned source revision")
    if manifest["upstream"]["license"] != spec.license:
        raise ValueError("manifest license differs from the pinned source license")
    if manifest["upstream"].get("required_citations", []) != list(spec.required_citations):
        raise ValueError("manifest citation obligations differ from the pinned source")
    if manifest["upstream"].get("components", []) != [component.__dict__ for component in spec.components]:
        raise ValueError("manifest source components differ from the pinned source")
    if manifest.get("source_artifacts", []) != source_artifact_manifest(spec):
        raise ValueError("manifest input artifacts differ from the pinned source artifacts")
    if manifest["source_schema"] != spec.source_schema:
        raise ValueError("manifest source schema differs from the pinned source schema")
    if manifest.get("ignored_source_labels", []) != list(spec.ignored_labels):
        raise ValueError("manifest ignored-label inventory differs from the pinned source projection")
    expected_projection_hash = manifest.get("source_projection_sha256")
    if expected_projection_hash is not None and expected_projection_hash != source_projection_sha256(spec):
        raise ValueError("source projection changed since onboarding; rebuild or review the projection")
    expected_source_hash = manifest.get("source_schema_sha256")
    if expected_source_hash is not None:
        if expected_source_hash != source_schema_sha256(spec.source_schema):
            raise ValueError("source schema changed since onboarding; rebuild or review the projection")
    elif manifest["tagset_sha256"] != tagset_sha256():
        raise ValueError("tagset changed since onboarding; rebuild or review the projection")
    source_map = Tagset().sources[spec.source_schema]
    ids = set()
    records = 0
    spans = 0
    splits = Counter()
    languages = Counter()
    canonical_labels = Counter()
    source_labels = Counter()
    shard_paths = set()
    for expected in manifest["shards"]:
        path = root / expected["path"]
        shard_paths.add(path.resolve())
        split = path.parent.name
        shard_lang = path.name.removesuffix(".jsonl.gz")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{path}: byte size differs from manifest")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from manifest")
        shard_records = 0
        shard_spans = 0
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                validate_row(row, source_map, path, line_number)
                if row["id"] in ids:
                    raise ValueError(f"{path}:{line_number}: duplicate id {row['id']!r}")
                if row["lang"] != shard_lang:
                    raise ValueError(
                        f"{path}:{line_number}: row language {row['lang']!r} "
                        f"differs from shard language {shard_lang!r}"
                    )
                ids.add(row["id"])
                shard_records += 1
                shard_spans += len(row["spans"])
                splits[split] += 1
                languages[row["lang"]] += 1
                for span in row["spans"]:
                    canonical_labels[span["label"]] += 1
                    source_labels[span["source_label"]] += 1
        if (shard_records, shard_spans) != (expected["records"], expected["spans"]):
            raise ValueError(f"{path}: record/span counts differ from manifest")
        records += shard_records
        spans += shard_spans
    actual_paths = {path.resolve() for path in root.glob("*/*.jsonl.gz")}
    if actual_paths != shard_paths:
        raise ValueError("manifest shard set differs from files on disk")
    if records != manifest["counts"]["records"] or spans != manifest["counts"]["spans"]:
        raise ValueError("dataset totals differ from manifest")
    recomputed = {
        "splits": dict(sorted(splits.items())),
        "languages": dict(sorted(languages.items())),
        "canonical_labels": dict(canonical_labels.most_common()),
        "source_labels": dict(source_labels.most_common()),
    }
    for name, actual in recomputed.items():
        if actual != manifest["counts"][name]:
            raise ValueError(f"{name} distribution differs from manifest")
    message = f"PII onboard check {spec.slug}: {records:,} records, {spans:,} spans verified"
    print(message)
    headline(message)
    return manifest


def refresh_integrity(spec: SourceSpec, output_root: Path) -> dict:
    root = output_root / spec.slug
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        path = root / shard["path"]
        shard["bytes"] = path.stat().st_size
        shard["sha256"] = sha256(path)
    manifest["source_schema_sha256"] = source_schema_sha256(spec.source_schema)
    manifest["source_projection_sha256"] = source_projection_sha256(spec)
    manifest["ignored_source_labels"] = list(spec.ignored_labels)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    message = f"PII onboard integrity refreshed: {spec.slug}, {len(manifest['shards'])} shards"
    print(message)
    headline(message)
    return manifest


def fetch(spec: SourceSpec, source_root: Path, cache_dir: Path | None) -> Path:
    if spec.fetch_backend == "local":
        if cache_dir is not None:
            raise ValueError("--cache-dir is not supported for local source artifacts")
        validate_source_artifacts(spec, source_root)
        message = f"PII local source {spec.slug} verified -> {source_root}"
        print(message)
        headline(message)
        return source_root
    if spec.fetch_backend == "git":
        if cache_dir is not None:
            raise ValueError("--cache-dir is only supported for Hugging Face sources")
        if source_root.exists() and any(source_root.iterdir()):
            raise FileExistsError(f"{source_root} is not empty")
        source_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(source_root)], check=True)
        subprocess.run(["git", "-C", str(source_root), "remote", "add", "origin", spec.url], check=True)
        subprocess.run(
            ["git", "-C", str(source_root), "fetch", "--depth=1", "origin", spec.revision],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source_root), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != spec.revision:
            raise ValueError(f"fetched revision {revision} != pinned {spec.revision}")
        message = f"PII fetch {spec.slug} complete -> {source_root}"
        print(message)
        headline(message)
        return source_root
    if spec.fetch_backend != "huggingface":
        raise ValueError(f"unsupported fetch backend {spec.fetch_backend!r}")

    from huggingface_hub import snapshot_download

    source_root.mkdir(parents=True, exist_ok=True)
    message = f"PII fetch {spec.repo_id}@{spec.revision[:12]}"
    print(message, flush=True)
    headline(message)
    downloaded = snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=source_root,
        cache_dir=cache_dir,
        allow_patterns=list(spec.allow_patterns),
    )
    message = f"PII fetch {spec.slug} complete -> {downloaded}"
    print(message)
    headline(message)
    return Path(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "build", "refresh-integrity", "check"))
    parser.add_argument("source", choices=tuple(SOURCES))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO / "data/pii-onboarded")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="retain the first N source records for a smoke; 0 retains all",
    )
    args = parser.parse_args()
    if args.max_records < 0 or (args.max_records and args.command != "build"):
        parser.error("--max-records must be nonnegative and applies only to build")
    spec = SOURCES[args.source]
    if args.command in {"fetch", "build"} and args.source_root is None:
        parser.error("--source-root is required for fetch/build")
    if args.command == "fetch":
        fetch(spec, args.source_root, args.cache_dir)
    elif args.command == "build":
        build(spec, args.source_root, args.output_root, max_records=args.max_records)
    elif args.command == "refresh-integrity":
        refresh_integrity(spec, args.output_root)
    else:
        check(spec, args.output_root)


if __name__ == "__main__":
    main()
