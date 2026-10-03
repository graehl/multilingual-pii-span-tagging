#!/usr/bin/env python3
"""Data sources the software fetches, their license terms, and your approvals.

Every command that downloads data (`data`, `fetch-web`, `names`) first checks
that each source it will fetch has an approval recorded in the approvals file
(default `license-approvals.json` in the package root). An approval stores
the license and terms URL you accepted; if a source's recorded terms change,
it no longer counts and must be approved again. Approving records your
decision to use the source under its terms; it does not relicense anything.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import acli

APPROVALS = ROOT / "license-approvals.json"

# Name lexicons behind the name-kind model (names command).
NAME_SOURCES = {
    "us-census-2010-surnames": {
        "license": "U.S. federal government aggregate data (public domain)",
        "terms_url": "https://www.census.gov/about/policies/quality/data-protection-and-privacy-policy.html",
        "url": "https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
        "used_by": "names",
    },
    "insee-prenoms-2024": {
        "license": "Licence Ouverte / Open Licence 2.0",
        "terms_url": "https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
        "url": "https://www.insee.fr/fr/statistiques/fichier/8894961/prenoms-2024-liste_csv.zip",
        "used_by": "names",
    },
    "ssa-babynames-mirror": {
        "license": "CC0 1.0 (U.S. Social Security Administration data via mirror)",
        "terms_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "url": "https://github.com/dxdc/babynames",
        "used_by": "names",
    },
    "edrdg-jmnedict": {
        "license": "CC BY-SA 4.0 (EDRDG licence)",
        "terms_url": "https://www.edrdg.org/edrdg/licence.html",
        "url": "http://ftp.edrdg.org/pub/Nihongo/JMnedict.xml.gz",
        "used_by": "names",
    },
    "wiktionary-name-categories": {
        "license": "CC BY-SA 4.0 / GFDL",
        "terms_url": "https://en.wiktionary.org/wiki/Wiktionary:Copyrights",
        "url": "https://en.wiktionary.org/w/api.php",
        "used_by": "names",
    },
    "faker-person-providers": {
        "license": "MIT",
        "terms_url": "https://github.com/joke2k/faker/blob/master/LICENSE.txt",
        "url": "https://github.com/joke2k/faker",
        "used_by": "names",
    },
}
WEB_SOURCES = {
    "fineweb": {
        "license": "ODC-By 1.0; subject to Common Crawl terms of use",
        "terms_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        "used_by": "fetch-web",
    },
    "fineweb-2": {
        "license": "ODC-By 1.0; subject to Common Crawl terms of use",
        "terms_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        "used_by": "fetch-web",
    },
}


def registry() -> dict[str, dict]:
    """Every fetchable source with its recorded license terms."""
    from pii_onboard_sources import SOURCES

    sources = {}
    for slug, spec in SOURCES.items():
        sources[slug] = {
            "license": spec.license,
            "terms_url": spec.url,
            "url": spec.url,
            "revision": spec.revision,
            "required_citations": list(spec.required_citations),
            "components": [{"name": c.name, "license": c.license, "url": c.url} for c in spec.components],
            "used_by": "data",
        }
    return {**sources, **WEB_SOURCES, **NAME_SOURCES}


def load_approvals(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"approved": {}}


def unapproved(sources: list[str], path: Path) -> list[str]:
    known = registry()
    approvals = load_approvals(path)["approved"]
    missing = []
    for source in sources:
        if source not in known:
            raise ValueError(f"unknown data source {source!r}")
        approval = approvals.get(source)
        if not approval or approval.get("license") != known[source]["license"]:
            missing.append(source)
    return missing


def require(sources: list[str], path: Path = APPROVALS) -> None:
    missing = unapproved(sources, path)
    if missing:
        raise ValueError(
            "review and approve these sources' license terms first: "
            + ", ".join(missing)
            + " (pii-reproduce.py licenses lists their terms; approve with pii-reproduce.py licenses "
            + " ".join(f"--approve {source}" for source in missing)
            + f"; approvals file {path})"
        )


def licenses(args) -> dict:
    known = registry()
    approvals = load_approvals(args.approvals)
    # Commercial terms are never approved in bulk; name such a source explicitly.
    bulk = [name for name, item in known.items() if "commercial" not in item["license"].lower()]
    targets = sorted(set(bulk) | set(args.approve or [])) if args.approve_all else (args.approve or [])
    for source in targets:
        if source not in known:
            raise ValueError(f"unknown data source {source!r}")
        approvals["approved"][source] = {
            "license": known[source]["license"],
            "terms_url": known[source]["terms_url"],
            "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    if targets:
        args.approvals.write_text(
            json.dumps(approvals, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    missing = set(unapproved(sorted(known), args.approvals))
    return {
        "ok": True,
        "approvals": str(args.approvals.resolve()),
        "sources": [
            {
                "source": name,
                "used_by": item["used_by"],
                "license": item["license"],
                "terms_url": item["terms_url"],
                "approved": name not in missing,
                **(
                    {"required_citations": item["required_citations"]}
                    if item.get("required_citations")
                    else {}
                ),
            }
            for name, item in sorted(known.items())
        ],
    }


def check(args) -> dict:
    require(args.source, args.approvals)
    return {"ok": True, "approved": args.source}


def main() -> None:
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument("--approvals", type=Path, default=APPROVALS)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("licenses", help="List sources and license terms; optionally approve.")
    listing.add_argument(
        "--approve", action="append", help="Approve this source's recorded terms; repeatable"
    )
    listing.add_argument(
        "--approve-all",
        action="store_true",
        help="Approve every listed source's terms except commercial ones, which must be named with --approve",
    )
    listing.set_defaults(action=licenses)
    checking = commands.add_parser("check", help="Fail unless every named source is approved.")
    checking.add_argument("--source", action="append", required=True)
    checking.set_defaults(action=check)
    acli.add_standard_args(parser)
    for command in (listing, checking):
        acli.add_standard_args(command)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        result = args.action(args)
    except (OSError, ValueError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
