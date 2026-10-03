#!/usr/bin/env python
"""Fetch and inventory aggregate name lexicons for name-component QC."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import io
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import KeysView
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"
USER_AGENT = "pii-name-resource-inventory/1.0"
GIVEN_TYPES = {
    "given name or forename, gender not specified",
    "female given name or forename",
    "male given name or forename",
}
FAMILY_TYPES = {"family or surname"}
SOURCE_CONTEXTS = {
    "us-census-2010": {
        "context_language": "en-US",
        "context_nations": ["US"],
        "context_basis": "national corpus locale; not asserted intrinsic name origin",
        "supervision_tier": "aggregate_role_labeled",
        "supervision_weight": 1.0,
    },
    "ssa-national-mirror-2024": {
        "context_language": "en-US",
        "context_nations": ["US"],
        "context_basis": "national corpus locale; not asserted intrinsic name origin",
        "supervision_tier": "aggregate_role_labeled",
        "supervision_weight": 1.0,
    },
    "insee-prenoms-2024": {
        "context_language": "fr-FR",
        "context_nations": ["FR"],
        "context_basis": "national corpus locale; not asserted intrinsic name origin",
        "supervision_tier": "aggregate_role_labeled",
        "supervision_weight": 1.0,
    },
    "edrdg-jmnedict": {
        "context_language": "ja-JP",
        "context_nations": ["JP"],
        "context_basis": "lexical-resource target language and locale",
        "supervision_tier": "lexical_role_labeled",
        "supervision_weight": 1.0,
    },
}


def normalize_surface(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def validate_source_context(source: str, context: dict) -> dict:
    required = {"context_language", "context_nations", "context_basis", "supervision_tier"}
    if not isinstance(context, dict) or not required <= set(context):
        raise ValueError(f"source {source!r} has an invalid context record")
    weight = float(context.get("supervision_weight", float("nan")))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"source {source!r} has an invalid supervision weight")
    return {**context, "supervision_weight": weight}


def wiktionary_context(language: str) -> dict:
    if re.fullmatch(r"[a-z]{2,3}", language) is None:
        raise ValueError(f"invalid Wiktionary language group {language!r}")
    return {
        "context_language": language,
        "context_nations": [],
        "context_basis": "linguistic category; no nation inferred",
        "supervision_tier": "category_role_weak",
        "supervision_weight": 0.5,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str, attempts: int = 5) -> tuple[dict, bool]:
    throttled = False
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response), throttled
        except HTTPError as error:
            if error.code not in {429, 503} or attempt + 1 == attempts:
                raise
            throttled = True
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after is not None else min(60.0, 5.0 * (2**attempt))
            time.sleep(delay)
        except (OSError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(30.0, 2.0**attempt))
    raise AssertionError("unreachable")


def wiktionary_specifications(args: argparse.Namespace) -> list[str]:
    specifications = list(args.category)
    for manifest_name in args.category_manifest:
        manifest_path = Path(manifest_name)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "pii-name-wiktionary-category-manifest-v1":
            raise ValueError(f"{manifest_path}: unsupported category manifest schema")
        entries = manifest.get("categories")
        if not isinstance(entries, list):
            raise ValueError(f"{manifest_path}: categories must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"role", "group", "category"}:
                raise ValueError(f"{manifest_path}: invalid category entry")
            specifications.append(f"{entry['role']}:{entry['group']}:{entry['category']}")
    if not specifications:
        raise ValueError("at least one --category or --category-manifest is required")
    if len(specifications) != len(set(specifications)):
        raise ValueError("duplicate Wiktionary category specifications")
    return specifications


def provider_role_surfaces(provider: type, marker: str) -> set[str]:
    collections = [getattr(provider, name) for name in dir(provider) if marker in name]
    surfaces = set()
    for collection in collections:
        if isinstance(collection, dict):
            collection = collection.keys()
        if isinstance(collection, (tuple, list, set, KeysView)):
            surfaces.update(value for value in collection if isinstance(value, str))
    return {surface for value in surfaces if (surface := normalize_surface(value))}


def cmd_extract_faker(args: argparse.Namespace) -> None:
    if not math.isfinite(args.supervision_weight) or args.supervision_weight <= 0:
        raise ValueError("--supervision-weight must be positive and finite")
    checkout = Path(args.checkout).resolve()
    observed_revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", f"{args.revision}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_revision != expected_revision:
        raise ValueError(
            f"Faker checkout is {observed_revision}, expected {args.revision} ({expected_revision})"
        )
    sys.path.insert(0, str(checkout))
    faker = importlib.import_module("faker")
    if not Path(faker.__file__).resolve().is_relative_to(checkout):
        raise ValueError(f"imported Faker outside pinned checkout: {faker.__file__}")
    person_root = checkout / "faker" / "providers" / "person"
    requested_languages = sorted(set(args.language))
    source_contexts = {}
    records = []
    missing_languages = []
    for language in requested_languages:
        if re.fullmatch(r"[a-z]{2,3}", language) is None:
            raise ValueError(f"invalid Faker language {language!r}")
        locales = sorted(
            path.name
            for path in person_root.iterdir()
            if path.is_dir() and (path.name == language or path.name.startswith(f"{language}_"))
        )
        if not locales:
            missing_languages.append(language)
            continue
        nations = sorted({locale.split("_", 1)[1] for locale in locales if "_" in locale})
        source_id = f"faker-{language}"
        source_contexts[source_id] = {
            "context_language": language,
            "context_nations": nations,
            "context_basis": "synthetic Faker person-provider locale; not asserted intrinsic name origin",
            "supervision_tier": "synthetic_suspect",
            "supervision_weight": args.supervision_weight,
        }
        role_locales: dict[str, dict[str, set[str]]] = {
            "given": defaultdict(set),
            "family": defaultdict(set),
        }
        for locale in locales:
            provider = importlib.import_module(f"faker.providers.person.{locale}").Provider
            for surface in provider_role_surfaces(provider, "first_names"):
                role_locales["given"][surface].add(locale)
            for surface in provider_role_surfaces(provider, "last_names"):
                role_locales["family"][surface].add(locale)
        for role, surface_locales in role_locales.items():
            for surface, observed_locales in surface_locales.items():
                records.append(
                    {
                        "kind": "surface",
                        "source": source_id,
                        "role": role,
                        "surface": surface,
                        "locales": sorted(observed_locales),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        destination.write(
            json.dumps(
                {
                    "kind": "metadata",
                    "schema": "pii-name-supplemental-surfaces-v1",
                    "source_family": "faker-person-providers",
                    "source_url": "https://github.com/joke2k/faker",
                    "license": "MIT",
                    "license_url": "https://github.com/joke2k/faker/blob/master/LICENSE.txt",
                    "revision": observed_revision,
                    "revision_name": args.revision,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "requested_languages": requested_languages,
                    "missing_languages": missing_languages,
                    "source_contexts": source_contexts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for record in sorted(records, key=lambda item: (item["source"], item["role"], item["surface"])):
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "rows": len(records),
                "languages": len(source_contexts),
                "missing_languages": missing_languages,
            },
            ensure_ascii=False,
        )
    )


def cmd_fetch_wiktionary(args: argparse.Namespace) -> None:
    rate_values = (
        args.request_delay,
        args.minimum_request_delay,
        args.acceleration,
        args.backoff_factor,
    )
    if not all(math.isfinite(value) and value > 0 for value in rate_values):
        raise ValueError("request delays and rate factors must be positive and finite")
    if args.minimum_request_delay > args.request_delay:
        raise ValueError("--minimum-request-delay cannot exceed --request-delay")
    if args.acceleration >= 1:
        raise ValueError("--acceleration must be less than 1")
    if args.backoff_factor <= 1:
        raise ValueError("--backoff-factor must be greater than 1")
    records = []
    category_counts = {}
    current_delay = args.request_delay
    for specification in wiktionary_specifications(args):
        fields = specification.split(":", 2)
        if len(fields) != 3 or fields[0] not in {"given", "family"}:
            raise ValueError("--category must be ROLE:GROUP:CATEGORY with ROLE given or family")
        role, group, category = fields
        continuation = None
        while True:
            parameters = {
                "action": "query",
                "format": "json",
                "maxlag": "5",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmnamespace": "0",
                "cmtype": "page",
                "cmlimit": "500",
            }
            if continuation is not None:
                parameters["cmcontinue"] = continuation
            payload, throttled = fetch_json(f"{WIKTIONARY_API}?{urllib.parse.urlencode(parameters)}")
            for item in payload["query"]["categorymembers"]:
                surface = normalize_surface(item["title"])
                if surface:
                    records.append(
                        {
                            "source": "enwiktionary-category-api",
                            "role": role,
                            "group": group,
                            "category": category,
                            "pageid": item["pageid"],
                            "surface": surface,
                        }
                    )
            continuation = payload.get("continue", {}).get("cmcontinue")
            current_delay = (
                current_delay * args.backoff_factor
                if throttled
                else max(args.minimum_request_delay, current_delay * args.acceleration)
            )
            time.sleep(current_delay)
            if continuation is None:
                break
        category_counts[specification] = sum(
            record["role"] == role and record["group"] == group and record["category"] == category
            for record in records
        )
    unique = {
        (record["role"], record["group"], record["category"], record["pageid"]): record for record in records
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        destination.write(
            json.dumps(
                {
                    "kind": "metadata",
                    "source": "enwiktionary-category-api",
                    "api": WIKTIONARY_API,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "license": "CC BY-SA 4.0 / GFDL",
                    "license_url": "https://en.wiktionary.org/wiki/Wiktionary:Copyrights",
                    "rate_policy": {
                        "initial_delay_seconds": args.request_delay,
                        "minimum_delay_seconds": args.minimum_request_delay,
                        "success_multiplier": args.acceleration,
                        "throttle_multiplier": args.backoff_factor,
                    },
                    "category_counts": category_counts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for record in sorted(
            unique.values(),
            key=lambda item: (item["group"], item["role"], item["surface"], item["pageid"]),
        ):
            destination.write(json.dumps({"kind": "surface", **record}, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"output": str(output.resolve()), "rows": len(unique), "category_counts": category_counts},
            ensure_ascii=False,
        )
    )


def add_surface(
    resources: dict[str, dict[str, set[str]]],
    source: str,
    role: str,
    value: str,
) -> None:
    surface = normalize_surface(value)
    if surface and not any(unicodedata.category(character) == "Cc" for character in surface):
        resources[source][role].add(surface)


def read_census(path: Path, resources: dict[str, dict[str, set[str]]]) -> None:
    with zipfile.ZipFile(path) as archive:
        with archive.open("Names_2010Census.csv") as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
                add_surface(resources, "us-census-2010", "family", row["name"])


def read_insee(path: Path, resources: dict[str, dict[str, set[str]]]) -> None:
    with zipfile.ZipFile(path) as archive:
        with archive.open("prenoms-2024-liste.csv") as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter=";"):
                add_surface(resources, "insee-prenoms-2024", "given", row["prenom"])


def read_ssa_directory(path: Path, resources: dict[str, dict[str, set[str]]]) -> None:
    files = sorted(path.glob("yob[0-9][0-9][0-9][0-9].txt"))
    if not files:
        raise ValueError(f"no SSA yobYYYY.txt files under {path}")
    for input_path in files:
        with input_path.open(encoding="utf-8") as source:
            for row in csv.reader(source):
                if len(row) != 3:
                    raise ValueError(f"{input_path}: expected name, sex, count rows")
                add_surface(resources, "ssa-national-mirror-2024", "given", row[0])


def read_jmnedict(path: Path, resources: dict[str, dict[str, set[str]]]) -> None:
    with gzip.open(path, "rb") as raw:
        for _event, element in ET.iterparse(raw, events=("end",)):
            if element.tag != "entry":
                continue
            types = {item.text for item in element.findall("./trans/name_type")}
            roles = []
            if types & GIVEN_TYPES:
                roles.append("given")
            if types & FAMILY_TYPES:
                roles.append("family")
            if roles:
                surfaces = [item.text for item in element.findall("./k_ele/keb")]
                surfaces.extend(item.text for item in element.findall("./r_ele/reb"))
                for role in roles:
                    for surface in surfaces:
                        if surface is not None:
                            add_surface(resources, "edrdg-jmnedict", role, surface)
            element.clear()


def read_wiktionary(
    path: Path,
    resources: dict[str, dict[str, set[str]]],
    source_contexts: dict[str, dict],
) -> None:
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("kind") != "surface":
                continue
            source_id = f"enwiktionary-{record['group']}"
            source_contexts.setdefault(source_id, wiktionary_context(record["group"]))
            add_surface(
                resources,
                source_id,
                record["role"],
                record["surface"],
            )


def read_supplemental(
    path: Path,
    resources: dict[str, dict[str, set[str]]],
    source_contexts: dict[str, dict],
) -> dict:
    metadata = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") == "metadata":
                if metadata is not None:
                    raise ValueError(f"{path}:{line_number}: repeated metadata record")
                metadata = record
                contexts = record.get("source_contexts")
                if not isinstance(contexts, dict):
                    raise ValueError(f"{path}:{line_number}: metadata lacks source_contexts")
                for source_id, context in contexts.items():
                    validated = validate_source_context(source_id, context)
                    if source_id in source_contexts and source_contexts[source_id] != validated:
                        raise ValueError(f"{path}:{line_number}: conflicting context for {source_id}")
                    source_contexts[source_id] = validated
                continue
            if record.get("kind") != "surface":
                raise ValueError(f"{path}:{line_number}: unsupported supplemental record")
            source_id = record.get("source")
            if source_id not in source_contexts:
                raise ValueError(f"{path}:{line_number}: unknown supplemental source {source_id!r}")
            role = record.get("role")
            if role not in {"given", "family"}:
                raise ValueError(f"{path}:{line_number}: unsupported role {role!r}")
            add_surface(resources, source_id, role, str(record.get("surface", "")))
    if metadata is None:
        raise ValueError(f"{path}: missing metadata record")
    return metadata


def script_bucket(surface: str) -> str:
    names = {unicodedata.name(character, "") for character in surface if character.isalpha()}
    if any("HIRAGANA" in name or "KATAKANA" in name for name in names):
        return "japanese"
    if any("HANGUL" in name for name in names):
        return "korean"
    if any("CJK UNIFIED" in name or "IDEOGRAPH" in name for name in names):
        return "han"
    if any("ARABIC" in name for name in names):
        return "arabic"
    if any("CYRILLIC" in name for name in names):
        return "cyrillic"
    if any("DEVANAGARI" in name for name in names):
        return "devanagari"
    if names and all("LATIN" in name for name in names):
        return "latin"
    return "mixed-or-other"


def input_record(path: Path, source_url: str, license_name: str, license_url: str) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_url": source_url,
        "license": license_name,
        "license_url": license_url,
    }


def ssa_directory_record(path: Path, revision: str) -> dict:
    files = sorted(path.glob("yob[0-9][0-9][0-9][0-9].txt"))
    digest = hashlib.sha256()
    for input_path in files:
        digest.update(input_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(input_path)))
    return {
        "path": str(path.resolve()),
        "files": len(files),
        "first_year": min(int(input_path.stem[3:]) for input_path in files),
        "last_year": max(int(input_path.stem[3:]) for input_path in files),
        "tree_sha256": digest.hexdigest(),
        "mirror_repo": "https://github.com/dxdc/babynames",
        "mirror_revision": revision,
        "authoritative_source": "https://www.ssa.gov/oact/babynames/names.zip",
        "catalog_record": "https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data",
        "license": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    }


def language_role_support(
    resources: dict[str, dict[str, set[str]]],
    source_contexts: dict[str, dict],
) -> dict:
    support: dict[str, dict[str, dict[str, float | int]]] = defaultdict(
        lambda: defaultdict(lambda: {"source_rows": 0, "effective_weight": 0.0})
    )
    for source, roles in resources.items():
        context = source_contexts[source]
        language = str(context["context_language"]).split("-", 1)[0].lower()
        weight = float(context["supervision_weight"])
        for role, surfaces in roles.items():
            support[language][role]["source_rows"] += len(surfaces)
            support[language][role]["effective_weight"] += len(surfaces) * weight
    return {
        language: {role: values for role, values in sorted(roles.items())}
        for language, roles in sorted(support.items())
    }


def cmd_build(args: argparse.Namespace) -> None:
    resources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    source_contexts = {
        source: validate_source_context(source, context) for source, context in SOURCE_CONTEXTS.items()
    }
    census = Path(args.census)
    insee = Path(args.insee)
    ssa_directory = Path(args.ssa_dir)
    jmnedict = Path(args.jmnedict)
    wiktionary = [Path(path) for path in args.wiktionary]
    read_census(census, resources)
    read_insee(insee, resources)
    read_ssa_directory(ssa_directory, resources)
    read_jmnedict(jmnedict, resources)
    for path in wiktionary:
        read_wiktionary(path, resources, source_contexts)
    supplemental_metadata = {
        str(Path(path).resolve()): read_supplemental(Path(path), resources, source_contexts)
        for path in args.supplemental
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    pooled: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for source, roles in resources.items():
        for role, surfaces in roles.items():
            path = output_dir / f"{source}.{role}.txt"
            path.write_text("\n".join(sorted(surfaces)) + "\n", encoding="utf-8")
            outputs[path.name] = {
                "path": str(path.resolve()),
                "role": role,
                "source": source,
                "unique_surfaces": len(surfaces),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for surface in surfaces:
                pooled[script_bucket(surface)][role].add(surface)

    contextual_path = output_dir / "name-surfaces-with-context.jsonl"
    with contextual_path.open("w", encoding="utf-8") as destination:
        for source, roles in sorted(resources.items()):
            if source not in source_contexts:
                raise ValueError(f"missing corpus context for source {source!r}")
            context = source_contexts[source]
            for role, surfaces in sorted(roles.items()):
                for surface in sorted(surfaces):
                    destination.write(
                        json.dumps(
                            {
                                "surface": surface,
                                "role": role,
                                "source": source,
                                "script_bucket": script_bucket(surface),
                                **context,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    outputs[contextual_path.name] = {
        "path": str(contextual_path.resolve()),
        "kind": "per-surface provenance and corpus-level language/nation context",
        "rows": sum(len(surfaces) for roles in resources.values() for surfaces in roles.values()),
        "bytes": contextual_path.stat().st_size,
        "sha256": sha256(contextual_path),
    }

    pooled_counts = {}
    for bucket, roles in pooled.items():
        for role, surfaces in roles.items():
            path = output_dir / f"pooled.{bucket}.{role}.txt"
            path.write_text("\n".join(sorted(surfaces)) + "\n", encoding="utf-8")
            outputs[path.name] = {
                "path": str(path.resolve()),
                "role": role,
                "script_bucket": bucket,
                "unique_surfaces": len(surfaces),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        given = roles.get("given", set())
        family = roles.get("family", set())
        pooled_counts[bucket] = {
            "given": len(given),
            "family": len(family),
            "given_family_overlap": len(given & family),
        }

    manifest = {
        "schema": "pii-name-resource-inventory-v2",
        "built_at": datetime.now(UTC).isoformat(),
        "normalization": "Unicode NFKC, then all Unicode whitespace runs fold to one ASCII space and edges trim; casing is preserved",
        "inputs": {
            "us-census-2010": input_record(
                census,
                "https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
                "U.S. federal government aggregate data",
                "https://www.census.gov/about/policies/quality/data-protection-and-privacy-policy.html",
            ),
            "insee-prenoms-2024": input_record(
                insee,
                "https://www.insee.fr/fr/statistiques/fichier/8894961/prenoms-2024-liste_csv.zip",
                "Licence Ouverte / Open Licence 2.0",
                "https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
            ),
            "ssa-national-mirror-2024": ssa_directory_record(ssa_directory, args.ssa_revision),
            "edrdg-jmnedict": input_record(
                jmnedict,
                "http://ftp.edrdg.org/pub/Nihongo/JMnedict.xml.gz",
                "CC BY-SA 4.0",
                "https://www.edrdg.org/edrdg/licence.html",
            ),
            "enwiktionary-category-api": [
                input_record(
                    path,
                    "https://en.wiktionary.org/w/api.php",
                    "CC BY-SA 4.0 / GFDL",
                    "https://en.wiktionary.org/wiki/Wiktionary:Copyrights",
                )
                for path in wiktionary
            ],
            "supplemental": {
                str(Path(path).resolve()): {
                    "path": str(Path(path).resolve()),
                    "bytes": Path(path).stat().st_size,
                    "sha256": sha256(Path(path)),
                    "metadata": supplemental_metadata[str(Path(path).resolve())],
                }
                for path in args.supplemental
            },
        },
        "source_counts": {
            source: {
                "roles": {role: len(surfaces) for role, surfaces in sorted(roles.items())},
                **source_contexts[source],
            }
            for source, roles in sorted(resources.items())
        },
        "language_role_support": language_role_support(resources, source_contexts),
        "pooled_script_counts": pooled_counts,
        "outputs": outputs,
        "known_limitations": [
            "Census and INSEE publish display forms in uppercase; the inventory preserves that observed casing.",
            "Wiktionary category membership is useful weak supervision, not an exhaustive or adjudicated registry.",
            "JMnedict readings and written forms are both retained; one surface can legitimately be both given and family.",
            "The current SSA archive returned HTTP 403 from this host on two attempts; a pinned mirror supplies raw 1880-2024 files, so the newly published 2025 file is absent.",
            "Language and nation fields describe the page or corpus context and are not claims about a surface's etymology or an individual bearer.",
            "Faker rows are synthetic/suspect weak support at weight 0.1; their raw counts must not be read as equivalent to aggregate or lexical role labels.",
        ],
    }
    manifest_path = Path(args.inventory)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"inventory": str(manifest_path.resolve()), "pooled": pooled_counts}, ensure_ascii=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-wiktionary")
    fetch.add_argument("--category", action="append", default=[])
    fetch.add_argument("--category-manifest", action="append", default=[])
    fetch.add_argument("--request-delay", type=float, default=0.5)
    fetch.add_argument("--minimum-request-delay", type=float, default=0.1)
    fetch.add_argument("--acceleration", type=float, default=0.8)
    fetch.add_argument("--backoff-factor", type=float, default=2.0)
    fetch.add_argument("--output", required=True)
    fetch.set_defaults(func=cmd_fetch_wiktionary)
    faker = subparsers.add_parser("extract-faker")
    faker.add_argument("--checkout", required=True)
    faker.add_argument("--revision", required=True)
    faker.add_argument("--language", action="append", required=True)
    faker.add_argument("--supervision-weight", type=float, default=0.1)
    faker.add_argument("--output", required=True)
    faker.set_defaults(func=cmd_extract_faker)
    build = subparsers.add_parser("build")
    build.add_argument("--census", required=True)
    build.add_argument("--insee", required=True)
    build.add_argument("--ssa-dir", required=True)
    build.add_argument("--ssa-revision", required=True)
    build.add_argument("--jmnedict", required=True)
    build.add_argument("--wiktionary", action="append", required=True)
    build.add_argument("--supplemental", action="append", default=[])
    build.add_argument("--output-dir", required=True)
    build.add_argument("--inventory", required=True)
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
