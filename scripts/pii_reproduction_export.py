"""Select repository-relative Python dependencies for the software export."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import re
import tempfile
import tokenize
from pathlib import Path

BINARY_SUFFIXES = {".safetensors", ".onnx"}
# Submission limit for software.tgz (user, 2026-09-30).
MAXIMUM_ARCHIVE_BYTES = 200_000_000

ENTRY_POINTS = (
    "pii-reproduce.py",
    "scripts/pii_onboard_sources.py",
    "scripts/pii_assemble_corpus.py",
    "scripts/pii_api_label.py",
    "scripts/pii_llm_label.py",
    "scripts/pii_encoder_train.py",
    "scripts/pii_eval.py",
    "scripts/pii_segment_eval.py",
    "scripts/pii_sentence_training_view.py",
    "scripts/pii_software_workflow.py",
    "scripts/pii_public_gold.py",
    "scripts/pii_public_mixture.py",
    "scripts/pii_redact.py",
    # evaluate's Ont3 view, and its default serving stages on a prediction sweep.
    "scripts/pii_software_ont3.py",
    "scripts/pii_serving_chain.py",
    "scripts/pii_character_boundary_refiner.py",
    "scripts/pii_software_receipts.py",
    "scripts/pii_fetch_web.py",
    "scripts/pii_software_screen.py",
    # The evaluation-overlap receipts the dedup gate verifies: retrieval, then thresholds.
    "scripts/pii_overlap_neighbors.py",
    "scripts/pii_overlap_filter.py",
    # select: the paper's training draw, needle selection and domain-near selection.
    "scripts/pii_final35_native_draw.py",
    "scripts/pii_needle_select.py",
    "scripts/pii_domain_select.py",
    "scripts/pii_software_sources.py",
    "scripts/pii_software_names.py",
    "scripts/pii_name_resource_inventory.py",
    "scripts/pii_name_role_char_pilot.py",
    "scripts/pii_name_role_publish.py",
    "scripts/pii_paper_o4_eval.py",
    "scripts/pii_paper_o4_figures.py",
    # Privacy Filter predictions on the quarter-step bias grid behind its operating point.
    "scripts/pii_paper_filter_extension.py",
    # Paper scripts the workflow loads by path rather than by import.
    "research/pii/frontier/evidence/human-gold-v1/score.py",
    "research/pii/frontier/evidence/priority9-shared-v1/render-comparison.py",
    "research/pii/frontier/evidence/priority9-shared-v1/predict-sweeps.py",
    "research/pii/frontier/evidence/four-corpus-v1/compile-mixture.py",
    "research/pii/frontier/evidence/title-extents-v1/mapa-train-titles.py",
)


def python_dependencies(root: Path) -> list[Path]:
    """Find local imports, retaining optional paths without rewriting modules."""
    selected: set[Path] = set()
    pending = [root / path for path in ENTRY_POINTS]
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected.add(path)
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
                bases = (root, root / "scripts")
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                names = [prefix, *(f"{prefix}.{item.name}".strip(".") for item in node.names)]
                bases = (path.parents[node.level - 1],) if node.level else (root, root / "scripts")
            else:
                continue
            for name in names:
                if "*" in name:
                    continue
                for base in bases:
                    stem = base.joinpath(*name.split("."))
                    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                        if candidate.is_file() and candidate.is_relative_to(root):
                            pending.append(candidate)
    return sorted(selected)


def stage_sources(root: Path, output: Path, *, readme: str, redactions: list[Path], anonymous: bool) -> dict:
    """Stage the source subset with the given redaction profiles applied."""
    return stage_files(export_files(root), output, readme=readme, redactions=redactions, anonymous=anonymous)


def load_profile(redactions: Path) -> list[dict]:
    profile = json.loads(redactions.read_text(encoding="utf-8"))
    if not isinstance(profile, list) or not profile:
        raise ValueError("redaction profile must be a nonempty list of {match, replacement} objects")
    return profile


def load_omissions(redactions: Path) -> set[str]:
    """Package paths an `omit` entry leaves out of the build entirely."""
    omitted = set()
    for item in load_profile(redactions):
        if "omit" in item:
            if set(item) != {"omit"} or not isinstance(item["omit"], str) or not item["omit"]:
                raise ValueError("an omit entry names one package path and nothing else")
            omitted.add(item["omit"])
    return omitted


def load_verbatim(redactions: Path) -> set[str]:
    """Package path prefixes a `verbatim` entry ships byte for byte: never redacted, still scanned.

    For released data, such as evaluation text whose bytes predictions are
    hashed against; a replacement there would corrupt the data, not anonymize
    it, so any profile match in it fails the build instead.
    """
    prefixes = set()
    for item in load_profile(redactions):
        if "verbatim" in item:
            if set(item) != {"verbatim"} or not isinstance(item["verbatim"], str) or not item["verbatim"]:
                raise ValueError("a verbatim entry names one package path prefix and nothing else")
            prefixes.add(item["verbatim"])
    return prefixes


def load_redactions(redactions: Path) -> list[tuple[re.Pattern[str], str | None, bool]]:
    """Literal replacements, plus `forbid` entries: text that must not survive redaction at all."""
    replacements = []
    for item in load_profile(redactions):
        if "omit" in item or "verbatim" in item:
            continue
        forbid = item.get("forbid") is True
        if (
            "match" not in item
            or ("replacement" in item) == forbid
            or set(item) - {"match", "replacement", "allow_code_literal", "forbid"}
            or not isinstance(item["match"], str)
            or not item["match"]
        ):
            raise ValueError(
                "each redaction needs a nonempty literal match and either a replacement or forbid"
            )
        if not forbid and not isinstance(item["replacement"], str):
            raise ValueError("redaction replacement must be text")
        replacements.append(
            (
                re.compile(re.escape(item["match"]), re.IGNORECASE),
                None if forbid else item["replacement"],
                bool(item.get("allow_code_literal", False)),
            )
        )
    return replacements


def tracked_files(repo: Path) -> tuple[list[tuple[Path, str]], str]:
    """A published repository's tracked files at a clean HEAD, and its README text."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout

    if git("status", "--porcelain"):
        raise ValueError(f"{repo} has uncommitted changes; package from a committed state")
    names = [name for name in git("ls-files", "-z").split("\0") if name]
    files = [(repo / name, name) for name in names if name not in ("README.md", "stage-manifest.json")]
    return files, (repo / "README.md").read_text(encoding="utf-8")


def export_files(root: Path) -> list[tuple[Path, str]]:
    """Repository files the software ships, each with its path inside the package."""
    sources = python_dependencies(root)
    sources.append(root / "research/pii/frontier/software/guide.md")
    sources.extend(sorted((root / "research/pii/frontier/software/docs").glob("*.md")))
    sources.extend(
        sorted(path for path in (root / "research/pii/frontier/software/models").rglob("*") if path.is_file())
    )
    license_source = root / "research/pii/frontier/software/LICENSE.md"
    sources.append(license_source)
    sources.append(root / "research/pii/frontier/software/pixi.toml")
    sources.append(root / "research/pii/frontier/software/smoke-language-round.yaml")
    sources.append(root / "research/pii/frontier/software/language-caps.yaml")
    # Unit tests of shipped code, run by the package's verify.toml.
    sources.append(root / "tests/conftest.py")
    listed = (root / "research/pii/frontier/software/shipped-tests.txt").read_text(encoding="utf-8")
    sources.extend(
        root / "tests/unit" / name
        for name in (line.strip() for line in listed.splitlines())
        if name and not name.startswith("#")
    )
    sources.extend(
        sorted(
            path for path in (root / "research/pii/frontier/software/records").rglob("*") if path.is_file()
        )
    )
    sources.extend(
        root / name
        for name in (
            "scripts/pii_tagset.yaml",
            "scripts/pii_language_round.yaml",
            "scripts/pii_language_round_final35.yaml",
            "scripts/pii_reproducibility.yaml",
            "scripts/pii_register_languages_v1.json",
            "scripts/pii_predicate_channels_v1.json",
            "scripts/pii_ont3_bernoulli_predicate_channels_v1.json",
            "scripts/pii_subclass_families_v2.json",
            "scripts/pii_subclass_families_v3.json",
            "scripts/pii_source_classes_v1.json",
            "scripts/pii_o_weight.yaml",
            # Loaded by shipped modules from their own directory.
            "scripts/pii_locale_profiles.yaml",
            "scripts/pii_named_entity_materializer.yaml",
            # Fixtures of shipped unit tests (shipped-tests.txt).
            "scripts/pii_subclass_families_v4.json",
            "scripts/pii_name_component_grammars_v1.json",
            "scripts/pii_name_annotation_qc_profiles_v1.json",
            "tests/data/name-kind-conformance-v1.json",
            "prompts/pii-label/catalog-ontology-v3-primary-v3.md",
            "prompts/pii-label/task-ont3-primary-annotate-context-v9.txt",
            "prompts/pii-label/examples-reviewed-final35-recall-ont3-v2.json",
            # annotate --prompt-revision paper-prompted (the default).
            "prompts/pii-label/paper-eval/task.txt",
            "prompts/pii-label/paper-eval/catalog.md",
            "prompts/pii-label/paper-eval/examples.json",
            "prompts/pii-label/paper-eval/README.md",
            # The dedup gate verifies pre-overlaplib receipts against these committed detector revisions.
            "scripts/pii_dedup_legacy_detector_code.json",
            # annotate --codex-model builds its isolated Codex home from this.
            "scripts/codex-annotation-home.config.toml",
            # select --needles default set.
            "data/pii-needles/rare-type-needles-v1.json",
            "research/pii/frontier/evidence/local-llm-label-quarantine-v1.json",
            "scripts/pii_tagset_v2.yaml",
            "scripts/pii_gold_v2.schema.json",
            "research/pii/frontier/evidence/four-corpus-v1/candidate-map.json",
            "research/pii/frontier/evidence/four-corpus-v1/negative-coverage-v1.json",
            "research/pii/frontier/evidence/four-corpus-v1/language-round.yaml",
            "research/pii/frontier/evidence/human-gold-v1/selected-ids.json",
            # Counts-only corroboration the ontology loader validates.
            "research/pii/frontier/evidence/ontology-v2-source-extension-evidence-v1.json",
            # Serving stages: regex supplementation and name-component postprocessing.
            "scripts/pii_regex_tags_v1.json",
            "scripts/pii_regex_policy_ont3_v1.json",
            "research/pii/frontier/models/name-components/name-postprocessor-name-kind.json",
            "scripts/pii_name_annotation_qc_profiles_v2.json",
            # Name-kind build: language round, Wiktionary categories, model card, script encoding.
            "scripts/pii_name_role_language_round_v2.yaml",
            "scripts/pii_name_wiktionary_final35_gap_v1.json",
            "research/pii/frontier/models/name-components/README.md",
            "scripts/vendor/script_bpe_v1/script_encoding_v1.json",
            "scripts/vendor/script_bpe_v1/LICENSE",
            "scripts/vendor/script_bpe_v1/VENDORED.md",
        )
    )
    # Text-free onboarding manifests: pinned upstream, license, shard hashes and
    # counts. The ontology loader validates some; readers can compare their own
    # prepared shards against all of them.
    sources.extend(sorted((root / "data/pii-onboarded").glob("*/manifest.json")))
    # The paper's Ont3 evaluation populations, released verbatim (see the public redaction profile).
    sources.extend(sorted(path for path in (root / "data/ont3-evaluation").iterdir() if path.is_file()))
    # select's default draw classifies sentences with these per-language reference-cue lexicons.
    sources.extend(
        path
        for path in sorted((root / "data/pii-annotations/final35/reference-cue-lexicon-v1").glob("*.json"))
        if not path.name.endswith(".raw.json")
    )
    lock = root / "research/pii/frontier/software/pixi.lock"
    if lock.is_file():
        sources.append(lock)
    # Separate pinned vLLM environment for the batching annotation teacher.
    sources.append(root / "research/pii/frontier/software/serve/pixi.toml")
    sources.append(root / "research/pii/frontier/software/serve/pixi.lock")
    software = root / "research/pii/frontier/software"
    renamed = {
        license_source: "LICENSE.md",
        software / "gitignore": ".gitignore",
        software / "CITATION.cff": "CITATION.cff",
        software / "verify.toml": "verify.toml",
    }
    # verify.toml declares the package's checks; scripts/verify (vendored from ~/agents) runs them.
    sources += [
        software / "gitignore",
        software / "CITATION.cff",
        software / "verify.toml",
        root / "scripts/verify",
    ]
    return [(source, renamed.get(source, source.relative_to(root).as_posix())) for source in sources]


def stage_files(
    files: list[tuple[Path, str]],
    output: Path,
    *,
    readme: str,
    redactions: list[Path],
    anonymous: bool,
) -> dict:
    """Copy files to their package paths, apply the redaction profiles in order, and scan.

    The public release applies the public profile (internal infrastructure);
    the anonymous review build adds the review profile (identities). Every
    profile's matches must be gone from the result.
    """
    if output.exists():
        raise FileExistsError(f"stage destination already exists: {output}")
    replacements = [entry for profile in redactions for entry in load_redactions(profile)]
    omitted = {path for profile in redactions for path in load_omissions(profile)}
    if missing := omitted - {packaged for _source, packaged in files}:
        raise ValueError(f"omit entries name files the build does not have: {sorted(missing)}")
    files = [(source, packaged) for source, packaged in files if packaged not in omitted]
    verbatim = tuple(sorted({prefix for profile in redactions for prefix in load_verbatim(profile)}))
    if unused := [prefix for prefix in verbatim if not any(p.startswith(prefix) for _s, p in files)]:
        raise ValueError(f"verbatim entries match no file the build has: {unused}")

    def redact(text: str) -> str:
        for pattern, replacement, _allow_code in replacements:
            if replacement is not None:
                text = pattern.sub(lambda _match: replacement, text)
        return text

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pii-stage-", dir=output.parent) as temporary:
        stage = Path(temporary) / "software"
        stage.mkdir()
        records = []
        for source, packaged in files:
            exact = packaged.startswith(verbatim) if verbatim else False
            relative = packaged if exact else redact(packaged)
            destination = stage / relative
            if not destination.resolve().is_relative_to(stage.resolve()):
                raise ValueError("redaction produced a path outside the stage directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix in BINARY_SUFFIXES or exact:
                # Weights carry no prose and released data must keep its bytes: copied as is.
                destination.write_bytes(source.read_bytes())
                records.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "redacted": False,
                    }
                )
                continue
            compressed = source.suffix == ".gz"
            original = (
                gzip.decompress(source.read_bytes()).decode("utf-8")
                if compressed
                else source.read_text(encoding="utf-8")
            )
            text = redact_python_prose(original, redact) if source.suffix == ".py" else redact(original)
            if compressed:
                with destination.open("xb") as stream:
                    stream.write(
                        source.read_bytes() if text == original else gzip.compress(text.encode(), mtime=0)
                    )
            else:
                with destination.open("x", encoding="utf-8") as stream:
                    stream.write(text)
            if destination.suffix == ".py":
                ast.parse(text, filename=relative)
            records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "redacted": original != text,
                }
            )
        (stage / "README.md").write_text(redact(readme), encoding="utf-8")
        manifest = {
            "schema": "pii-software-stage-v1",
            "anonymous": anonymous,
            "status": "staged source subset; see README.md for verified workflows and "
            "research/pii/frontier/software/records/README.md for evidence and limits",
            "files": records,
        }
        (stage / "stage-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Verbatim files are scanned too: released data cannot be redacted, so a match must be
        # removed at its source.
        for path in stage.rglob("*"):
            if path.is_file() and path.suffix not in BINARY_SUFFIXES:
                content = (
                    gzip.decompress(path.read_bytes()).decode("utf-8")
                    if path.suffix == ".gz"
                    else path.read_text(encoding="utf-8")
                )
                code_literals = python_code_literals(content) if path.suffix == ".py" else []
                for pattern, _replacement, allow_code in replacements:
                    if pattern.search(path.relative_to(stage).as_posix()):
                        raise ValueError("anonymous stage still has an identifying filename")
                    for match in pattern.finditer(content):
                        if not allow_code or not any(
                            start <= match.start() and match.end() <= end for start, end in code_literals
                        ):
                            raise ValueError(
                                f"anonymous stage still contains an identity in {path.relative_to(stage)}"
                            )
        stage.rename(output)
    return {
        "ok": True,
        "stage": str(output.resolve()),
        "anonymous": anonymous,
        "files": len(records),
        "redacted_files": sum(record["redacted"] for record in records),
        "redaction_profiles": [profile.name for profile in redactions],
        "redaction_scan": "passed" if redactions else "not requested",
        "status": manifest["status"],
    }


def python_text_regions(text: str) -> list[tuple[int, int, bool]]:
    """Identify Python prose and executable string literals without reformatting."""
    tree = ast.parse(text)
    docstrings = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node, clean=False) is not None
        ):
            value = node.body[0].value
            docstrings.add((value.lineno, value.col_offset))
    line_offsets = [0]
    for line in text.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    regions = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        start = line_offsets[token.start[0] - 1] + token.start[1]
        end = line_offsets[token.end[0] - 1] + token.end[1]
        prose = token.type == tokenize.COMMENT or token.start in docstrings
        regions.append((start, end, prose))
    return regions


def python_code_literals(text: str) -> list[tuple[int, int]]:
    return [(start, end) for start, end, prose in python_text_regions(text) if not prose]


def redact_python_prose(text: str, redact) -> str:
    for start, end, prose in reversed(python_text_regions(text)):
        if prose:
            text = text[:start] + redact(text[start:end]) + text[end:]
    return text


def build_archive(stage: Path, archive: Path) -> dict:
    """Deterministic, anonymously owned tarball of a staged tree, then a fresh-extraction check."""
    import os
    import subprocess
    import sys
    import tarfile

    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")

    def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if info.isdir() or info.mode & 0o111 else 0o644
        return info

    # A zero gzip timestamp keeps identical stages byte-identical as archives.
    with (
        archive.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle,
    ):
        bundle.add(stage, arcname="software", recursive=False, filter=normalized)
        for path in sorted(stage.rglob("*")):
            bundle.add(
                path,
                arcname=f"software/{path.relative_to(stage).as_posix()}",
                recursive=False,
                filter=normalized,
            )
    if archive.stat().st_size > MAXIMUM_ARCHIVE_BYTES:
        raise ValueError(
            f"archive is {archive.stat().st_size:,} bytes; the submission limit is {MAXIMUM_ARCHIVE_BYTES:,}"
        )
    with tempfile.TemporaryDirectory(prefix=".pii-archive-check-", dir=archive.parent) as temporary:
        check = Path(temporary)
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            if any(not (m.name == "software" or m.name.startswith("software/")) for m in members):
                raise ValueError("archive member outside software/")
            if any(m.issym() or m.islnk() or m.uid or m.gid or m.uname or m.gname for m in members):
                raise ValueError("archive has links or non-anonymous ownership")
            bundle.extractall(check, filter="data")
        root = check / "software"
        manifest = json.loads((root / "stage-manifest.json").read_text())
        for row in manifest["files"]:
            path = root / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"extracted file differs from manifest: {row['path']}")
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=row["path"])
        expected = {row["path"] for row in manifest["files"]} | {"README.md", "stage-manifest.json"}
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError(f"extracted files differ from manifest: {sorted(actual ^ expected)[:5]}")
        home = check / "empty-home"
        home.mkdir()
        environment = dict(os.environ, HOME=str(home), PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
        shown = subprocess.run(
            [sys.executable, "pii-reproduce.py", "-h"],
            cwd=root,
            env=environment,
            capture_output=True,
            check=True,
        )
        if shown.stdout != (root / "README.md").read_bytes():
            raise ValueError("help output in a fresh extraction differs from README.md")
    return {
        "archive": str(archive.resolve()),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "bytes": archive.stat().st_size,
        "files": len(manifest["files"]),
        "checks": "safe paths, anonymous ownership, manifest hashes, Python syntax, help equals README in an empty home",
    }
