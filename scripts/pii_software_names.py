#!/usr/bin/env python3
"""Build the name-kind model from public name lexicons.

The paper's name-component postprocessor asks a small character model whether
a name component is a given or a family name. This command rebuilds that
model from the same public lexicons: U.S. Census surnames, INSEE given names,
the SSA baby-name mirror, EDRDG JMnedict, English Wiktionary name categories
and Faker person providers, each fetched only after its license terms are
approved (`pii-reproduce.py licenses`). It then builds the role inventory,
fits the development-selected recipe (64 convolution channels, learning rate
0.003), continues on all data for 120 epochs and exports the ONNX bundle that
`redact --serve` loads.

Differences from the paper's deployed model: Wiktionary categories and
JMnedict change over time, so the lexicons will not be byte-identical, and
the deployed model was additionally continued on private annotated text.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acli

INVENTORY = "scripts/pii_name_resource_inventory.py"
LANGUAGE_ROUND = ROOT / "scripts/pii_name_role_language_round_v2.yaml"
# Pinned where the upstream allows it; hashes are those of the paper's inputs.
FILES = {
    "census.zip": (
        "us-census-2010-surnames",
        "https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
        "117c41cb4668727b7627b2845b6df3f83eb2a22a1813f42c0ff4bdcab86de135",
    ),
    "insee.zip": (
        "insee-prenoms-2024",
        "https://www.insee.fr/fr/statistiques/fichier/8894961/prenoms-2024-liste_csv.zip",
        "9f40ff9476287c3d7e9fe6f3854bc438f02645e621479d34b81bd8f0f425b676",
    ),
    "JMnedict.xml.gz": (
        "edrdg-jmnedict",
        "http://ftp.edrdg.org/pub/Nihongo/JMnedict.xml.gz",
        "e7a16e70b378bd9c774ea2d628d8ac6b331c987ea9e871220e6dc2ba804e559c",
    ),
}
SSA = ("https://github.com/dxdc/babynames", "6c7ff5c762c7eff5a11ad9065e47828154dfb392")
FAKER = ("https://github.com/joke2k/faker", "v40.31.0")
FAKER_LANGUAGES = "de es it pt nl pl cs sv uk tr id vi bn te hr da fi el ro no th ms fil fa ta ur he".split()
WIKTIONARY = {
    "major-scripts": [
        "given:ar:Arabic male given names",
        "given:ar:Arabic female given names",
        "family:ar:Arabic surnames",
        "given:zh:Chinese given names",
        "family:zh:Chinese surnames",
        "given:ko:Korean given names",
        "family:ko:Korean surnames",
        "given:ru:Russian male given names",
        "given:ru:Russian female given names",
        "family:ru:Russian surnames",
        "given:hi:Hindi male given names",
        "given:hi:Hindi female given names",
        "family:hi:Hindi surnames",
    ],
    "final35-gap": "scripts/pii_name_wiktionary_final35_gap_v1.json",
    "fr-family": ["family:fr:French surnames"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.with_suffix(".out").open("w") as stdout, log.with_suffix(".err").open("w") as stderr:
        subprocess.run([str(part) for part in command], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def git_checkout(url: str, revision: str, target: Path, log: Path) -> None:
    if (target / ".git").is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "-q", target], log.with_name(log.name + "-init"))
    # A tag is fetched as a local tag too: consumers verify it resolves.
    ref = revision if len(revision) == 40 else f"refs/tags/{revision}:refs/tags/{revision}"
    run(["git", "-C", target, "fetch", "-q", "--depth=1", url, ref], log.with_name(log.name + "-fetch"))
    run(
        ["git", "-C", target, "checkout", "-q", "--detach", "FETCH_HEAD"],
        log.with_name(log.name + "-checkout"),
    )


def fetch(work: Path) -> dict:
    from scripts.pii_software_sources import NAME_SOURCES, require

    require(list(NAME_SOURCES))
    raw, logs = work / "raw", work / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (source, url, pinned) in FILES.items():
        path = raw / name
        placed = path.is_file()
        if not placed:
            request = urllib.request.Request(url, headers={"User-Agent": "pii-software-names/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response:
                path.write_bytes(response.read())
        # Some publishers answer blocked clients with an HTML page and status 200.
        magic = path.read_bytes()[:2]
        if magic != (b"PK" if name.endswith(".zip") else b"\x1f\x8b"):
            path.unlink()
            raise ValueError(
                f"{url} did not return the expected archive (the server may block this network). "
                f"Download it in a browser, save it as {path} and rerun; a manually placed file is "
                f"checked against the paper's input SHA-256 {pinned}"
            )
        observed = sha256_file(path)
        files[source] = {
            "url": url,
            "sha256": observed,
            "matches_paper_input": observed == pinned,
            # Placed by hand or kept from an earlier run; either way hash-checked above.
            "already_present": placed,
        }
    git_checkout(*SSA, raw / "ssa", logs / "ssa")
    git_checkout(*FAKER, raw / "faker", logs / "faker")
    for key, spec in WIKTIONARY.items():
        output = raw / f"wiktionary-{key}.jsonl"
        if output.is_file():
            continue
        command = [sys.executable, INVENTORY, "fetch-wiktionary", "--output", output]
        if isinstance(spec, str):
            command.extend(("--category-manifest", spec))
        else:
            for category in spec:
                command.extend(("--category", category))
        run(command, logs / f"wiktionary-{key}")
    faker = raw / "faker-final35.jsonl"
    if not faker.is_file():
        command = [
            sys.executable,
            INVENTORY,
            "extract-faker",
            "--checkout",
            raw / "faker",
            "--revision",
            FAKER[1],
        ]
        command += ["--supervision-weight", "0.1", "--output", faker]
        for language in FAKER_LANGUAGES:
            command.extend(("--language", language))
        run(command, logs / "faker-extract")
    return {"files": files, "ssa_revision": SSA[1], "faker_revision": FAKER[1]}


def build(work: Path) -> Path:
    raw = work / "raw"
    output = work / "inventory"
    context = output / "name-surfaces-with-context.jsonl"
    if context.is_file():
        return context
    command = [
        sys.executable,
        INVENTORY,
        "build",
        "--census",
        raw / "census.zip",
        "--insee",
        raw / "insee.zip",
        "--ssa-dir",
        raw / "ssa/raw",
        "--ssa-revision",
        SSA[1],
        "--jmnedict",
        raw / "JMnedict.xml.gz",
        "--supplemental",
        raw / "faker-final35.jsonl",
        "--output-dir",
        output,
        "--inventory",
        output / "inventory.json",
    ]
    for key in WIKTIONARY:
        command.extend(("--wiktionary", raw / f"wiktionary-{key}.jsonl"))
    run(command, work / "logs/build")
    if not context.is_file():
        raise RuntimeError(f"inventory build did not write {context}")
    return context


def fit(work: Path, context: Path, threads: int) -> Path:
    pilot, final, bundle = work / "pilot", work / "all-data", work / "bundle"
    if not (pilot / "model.safetensors").is_file():
        command = [sys.executable, "-m", "scripts.pii_name_role_char_pilot", "fit-evaluate"]
        command += ["--context-jsonl", context, "--language-round", LANGUAGE_ROUND, "--output", pilot]
        command += ["--seed", "155", "--batch-size", "1024", "--samples-per-epoch", "120000"]
        command += ["--maximum-epochs", "300", "--patience", "10", "--learning-rate", "0.003"]
        command += ["--convolution-channels", "64", "--training-threads", threads, "--inference-threads", "1"]
        command += ["--benchmark-repetitions", "1"]
        run(command, work / "logs/pilot")
    if not (final / "model.safetensors").is_file():
        command = [sys.executable, "-m", "scripts.pii_name_role_publish", "fit-all-data"]
        command += ["--context-jsonl", context, "--language-round", LANGUAGE_ROUND]
        command += ["--init-from-output", pilot, "--output", final, "--seed", "155"]
        command += ["--batch-size", "1024", "--samples-per-epoch", "120000", "--epochs", "120"]
        command += [
            "--learning-rate",
            "0.002",
            "--length-window-steps",
            "8",
            "--maximum-support-multiplier",
            "2",
        ]
        command += ["--full-support-effective-cell-weight", "1000", "--training-threads", threads]
        run(command, work / "logs/all-data")
    if not (bundle / "name-kind.config.json").is_file():
        command = [
            sys.executable,
            "-m",
            "scripts.pii_name_role_publish",
            "export-onnx",
            "--fit-output",
            final,
        ]
        command += ["--output", bundle, "--stem", "name-kind"]
        command += ["--model-card", ROOT / "research/pii/frontier/models/name-components/README.md"]
        command += ["--grammars", ROOT / "scripts/pii_name_annotation_qc_profiles_v2.json"]
        run(command, work / "logs/export")
    return bundle / "name-kind.config.json"


def names(args) -> dict:
    work = args.out.resolve()
    fetched = fetch(work)
    context = build(work)
    config = fit(work, context, args.threads)
    receipt = {
        "schema": "pii-software-name-kind-build-v1",
        "fetched": fetched,
        "inventory": {"path": str(context), "sha256": sha256_file(context)},
        "bundle": str(config),
        "note": "Public-lexicon recipe; the paper's deployed model was also continued on private text.",
    }
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"ok": True, **receipt}


def main() -> None:
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument(
        "--out", type=Path, required=True, help="Work directory; reruns resume finished stages"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU training threads")
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        result = names(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
