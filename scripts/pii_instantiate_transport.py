#!/usr/bin/env python
"""Instantiate transported placeholder waves into labeled training data
(topics/pii-robust-buildout.md v2 data increment; generalizes
pii_instantiate_zh.py beyond zh).

A wave record is {id, src, hyp, survived, ...}: src holds the source-language
[TAG_i] placeholder doc, hyp its translation, where TAG may itself be
translated (es/de/ru/pl do this; the mined tag tables prove indices
survive). Resolution is INDEX-KEYED: src defines i -> TAG, and any
hyp token [anything_i] is filled for that TAG — no tag-name mapping
needed, so translated tags cost nothing. A doc is usable iff hyp's
placeholder-index multiset equals src's, including multiplicity for a
linked index that occurs more than once; index-broken docs (3-18% by
language) are skipped and counted.

Emitted spans carry the placeholder's ORIGINAL nemotron label
(ph_map src_label), so pii_assemble_corpus.py ingests these under the
existing nemotron_pii schema map — no new tagset rows.

Name-mix policy v1: 80% native-locale Faker names, 20% Latin-kept
Western (en_US) — the zh script's 10% transliterated slot is zh-only
lexicon work and is deferred for other languages (ja katakana
foreigners noted as a refinement in the board). Format-bound classes
use locale providers where Faker has them, ASCII where global;
unresolvable freetext falls back to the original English value
(provenance "orig").

Usage:
  pii_instantiate_transport.py --lang ru \\
      [--wave prod/tg-ru-7k.jsonl] [--ph prod/nemotron-ph-7000.jsonl] \\
      [--out prod/ru-instantiated-v1.jsonl] [--seed 20260721]
"""

import argparse
import hashlib
import json
import os
import random
import re
import string
from collections import Counter
from datetime import date
from pathlib import Path

from faker import Faker

try:
    from pii_slot_morphology import SlotMorphology
    from pii_surface.mix_policy import SurfaceMixPolicy
    from pii_surface_pool import SurfacePool
    from pii_transport_semantic_guard import RULESET as SEMANTIC_GUARD_RULESET
    from pii_transport_semantic_guard import semantic_guard_reasons
except ModuleNotFoundError:  # Imported as scripts.pii_instantiate_transport in tests.
    from scripts.pii_slot_morphology import SlotMorphology
    from scripts.pii_surface.mix_policy import SurfaceMixPolicy
    from scripts.pii_surface_pool import SurfacePool
    from scripts.pii_transport_semantic_guard import RULESET as SEMANTIC_GUARD_RULESET
    from scripts.pii_transport_semantic_guard import semantic_guard_reasons

try:
    from pii_cjk_name_order import MATERIALIZER_POLICY_PATH, load_repair_policy, requires_family_first_repair
    from pii_cjk_name_order import reorder_row as reorder_cjk_names
    from pii_seed_tree import DEFAULT_CONFIG_PATH as REPRODUCIBILITY_CONFIG_PATH
    from pii_seed_tree import SeedTree, load_seed_tree
except ModuleNotFoundError:  # Imported as scripts.pii_instantiate_transport in tests.
    from scripts.pii_cjk_name_order import (
        MATERIALIZER_POLICY_PATH,
        load_repair_policy,
        requires_family_first_repair,
    )
    from scripts.pii_cjk_name_order import reorder_row as reorder_cjk_names
    from scripts.pii_seed_tree import DEFAULT_CONFIG_PATH as REPRODUCIBILITY_CONFIG_PATH
    from scripts.pii_seed_tree import SeedTree, load_seed_tree

HERE = os.environ.get("PII_EVAL_HOME") or os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCALE_PROFILE = Path(__file__).resolve().parent / "pii_locale_profiles.yaml"
PROD = os.path.join(HERE, "prod")

SRC_TOK = re.compile(r"\[([A-Z0-9_]+)_(\d+)\]")
HYP_TOK = re.compile(r"\[([^\[\]]+?)_(\d+)\]")

FAKER_LOCALE = {
    "en": "en_US",
    "de": "de_DE",
    "fr": "fr_FR",
    "es": "es_ES",
    "it": "it_IT",
    "pt": "pt_PT",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "cs": "cs_CZ",
    "sv": "sv_SE",
    "ru": "ru_RU",
    "uk": "uk_UA",
    "zh": "zh_CN",
    "ja": "ja_JP",
    "ko": "ko_KR",
    "hi": "hi_IN",
    "ar": "ar_AA",
    "tr": "tr_TR",
    "id": "id_ID",
    "vi": "vi_VN",
    "fa": "fa_IR",
    # Added 2026-09-13 for the ont3 35-language round. The map previously held
    # only the Final20 languages, so any other language raised KeyError as soon
    # as a Filler was built, even for a recipe that generates nothing.
    "bn": "bn_BD",
    "da": "da_DK",
    "el": "el_GR",
    "fi": "fi_FI",
    "fil": "fil_PH",
    "he": "he_IL",
    "hr": "hr_HR",
    "no": "no_NO",
    "ro": "ro_RO",
    "ta": "ta_IN",
    "th": "th_TH",
    # Faker ships no locale for these three. The entries are script and region
    # proxies so a Filler can be constructed; they are wrong for generating
    # names and must not be used for Faker-backed realization without review.
    # Malay takes Indonesian, which is the closest available and Latin script.
    # Telugu takes Tamil, South Indian but a different script. Urdu takes
    # Persian, which at least shares the Perso-Arabic script.
    "ms": "id_ID",
    "te": "ta_IN",
    "ur": "fa_IR",
}
COUNTRY = {
    "en": "United States",
    "de": "Deutschland",
    "fr": "France",
    "es": "España",
    "it": "Italia",
    "pt": "Portugal",
    "nl": "Nederland",
    "pl": "Polska",
    "cs": "Česko",
    "sv": "Sverige",
    "ru": "Россия",
    "uk": "Україна",
    "zh": "中国",
    "ja": "日本",
    "ko": "대한민국",
    "hi": "भारत",
    "ar": "مصر",
    "tr": "Türkiye",
    "id": "Indonesia",
    "vi": "Việt Nam",
    "fa": "ایران",
}
MATERIALIZATION_VERSION = "final20-v1"
SURFACE_FEEDBACK_VERSION = "final20-surface-feedback-v3"
SEMANTIC_FILTER_VERSION = "final20-transport-semantic-policy-v2"
TEMPORAL_TAGS = {"DATE", "DATE_TIME", "DATE_OF_BIRTH", "TIME"}
NAME_COMPONENT_TAGS = {"GIVEN_NAME", "FAMILY_NAME", "MIDDLE_NAME", "PERSON_NAME"}
LOCATION_TAGS = {
    "ADDRESS",
    "BUILDING_NUMBER",
    "CITY",
    "COORDINATE",
    "COUNTRY",
    "COUNTY",
    "GPS_COORDINATES",
    "LOCATION",
    "POSTAL_CODE",
    "REGION",
    "STATE",
    "STREET_ADDRESS",
}
UNLOCALIZED_CATEGORICAL_TAGS = {
    "EDUCATION_LEVEL",
    "EMPLOYMENT_STATUS",
    "GENDER",
    "LANGUAGE",
    "POLITICAL_VIEW",
    "RACE_ETHNICITY",
    "RELIGIOUS_BELIEF",
    "SEXUALITY",
}

FAKER_SURFACE_TAGS = {
    "ADDRESS",
    "CARD_NUMBER",
    "CITY",
    "EMAIL",
    "FAMILY_NAME",
    "FAX_NUMBER",
    "FINANCIAL_ORG",
    "GIVEN_NAME",
    "HEALTHCARE_ORG",
    "IBAN",
    "IP_ADDRESS",
    "IPV4",
    "IPV6",
    "JOB_TITLE",
    "MAC_ADDRESS",
    "MIDDLE_NAME",
    "NATIONAL_ID",
    "OCCUPATION",
    "ORGANIZATION",
    "PERSON_NAME",
    "PHONE_NUMBER",
    "POSTAL_CODE",
    "REGION",
    "SSN",
    "STATE",
    "STREET_ADDRESS",
    "URL",
    "USERNAME",
}


def carrier_document_id(row_id):
    """Return the placeholder-carrier identity shared by split fragments."""
    return row_id.split("#", 1)[0]


def entity_realization_seed(
    seed_tree,
    lang,
    carrier_id,
    realization_index,
    tag,
    source_value,
    *,
    entity_key=None,
):
    """Fork one stable draw for a carrier entity, independent of fragment order."""
    return seed_tree.fork(
        "named-entity-materializer",
        lang,
        carrier_id,
        realization_index,
        "entity",
        tag,
        source_value if entity_key is None else entity_key,
    )


def name_mode_groups(ph_map):
    """Map consecutive name-component placeholders to one stable person-name group."""
    ordered = []
    for placeholder, info in ph_map.items():
        match = SRC_TOK.fullmatch(placeholder)
        if match:
            slot_index = info.get("slot_index", int(match.group(2)))
            tag = info.get("tag", match.group(1))
            ordered.append((int(slot_index), tag, placeholder))
    ordered.sort()

    groups = {}
    current = []

    def finish_group():
        if current:
            key = tuple(item[2] for item in current)
            groups.update((item[2], key) for item in current)
            current.clear()

    for index, tag, placeholder in ordered:
        if tag not in NAME_COMPONENT_TAGS:
            finish_group()
            continue
        if current and index != current[-1][0] + 1:
            finish_group()
        current.append((index, tag, placeholder))
    finish_group()
    return groups


class Filler:
    def __init__(
        self,
        lang,
        seed,
        surface_pool=None,
        natural_surface_rate=0.0,
        natural_min_distinct_full_rate=100,
        natural_count_temperature=0.5,
        reject_unlocalized_categorical=False,
        locale_renderer=None,
        surface_policy=None,
    ):
        if not 0 <= natural_surface_rate <= 1:
            raise ValueError("natural_surface_rate must be in [0, 1]")
        if natural_surface_rate and surface_pool is None:
            raise ValueError("natural_surface_rate requires a surface_pool")
        if natural_min_distinct_full_rate <= 0:
            raise ValueError("natural_min_distinct_full_rate must be positive")
        if not 0 <= natural_count_temperature <= 1:
            raise ValueError("natural_count_temperature must be in [0, 1]")
        if surface_policy is not None and surface_pool is None:
            raise ValueError("surface_policy requires a surface_pool")
        self.rng = random.Random(seed)
        self.loc = Faker(FAKER_LOCALE[lang])
        self.en = Faker("en_US")
        self.loc.seed_instance(seed)
        self.en.seed_instance(seed + 1)
        self.lang = lang
        self.surface_pool = surface_pool
        self.natural_surface_rate = natural_surface_rate
        self.natural_min_distinct_full_rate = natural_min_distinct_full_rate
        self.natural_count_temperature = natural_count_temperature
        self.reject_unlocalized_categorical = reject_unlocalized_categorical
        self.locale_renderer = locale_renderer
        self.surface_policy = surface_policy
        self._document_cache = None
        self._preserve_linked_locations = False
        self.document_rejection_reasons = []

    def reseed_document(self, seed):
        """Reset every stochastic provider from one document-level fork."""
        tree = SeedTree(seed)
        self.rng.seed(tree.fork("python"))
        self.loc.seed_instance(tree.fork("faker", "target"))
        self.en.seed_instance(tree.fork("faker", "english"))

    def begin_document(self, ph_map, *, seed=None):
        """Reset document-local coherence state before filling its placeholders."""
        if seed is not None:
            self.reseed_document(seed)
        tags = []
        for placeholder, info in ph_map.items():
            match = SRC_TOK.fullmatch(placeholder)
            if match:
                tags.append(info.get("tag", match.group(1)))
        self._document_cache = {}
        self._preserve_linked_locations = sum(tag in LOCATION_TAGS for tag in tags) >= 2
        self.document_rejection_reasons = []
        if self.reject_unlocalized_categorical:
            unsupported = sorted(set(tags) & UNLOCALIZED_CATEGORICAL_TAGS)
            self.document_rejection_reasons.extend(f"unlocalized_categorical:{tag}" for tag in unsupported)

    def _natural(self, tag, *, force=False):
        if self.surface_pool is None:
            if force:
                raise ValueError(f"exact empirical route requires a surface pool for {self.lang}/{tag}")
            return None
        cell = self.surface_policy.for_tag(self.lang, tag) if self.surface_policy else None
        distinct = self.surface_pool.distinct_count(self.lang, tag)
        rate = cell.natural_surface_rate if cell else self.natural_surface_rate
        minimum = cell.natural_min_distinct_full_rate if cell else self.natural_min_distinct_full_rate
        temperature = cell.natural_count_temperature if cell else self.natural_count_temperature
        effective_rate = rate * min(1.0, distinct / minimum)
        if not distinct:
            if force:
                raise ValueError(f"exact empirical route has no entries for {self.lang}/{tag}")
            return None
        if not force and self.rng.random() >= effective_rate:
            return None
        return self.surface_pool.draw(
            self.lang,
            tag,
            self.rng,
            count_temperature=temperature,
        )

    def _name_parts(self, tag, mode=None, *, allow_natural=True):
        if mode is None:
            cell = self.surface_policy.for_tag(self.lang, tag) if self.surface_policy else None
            native_rate = cell.native_name_rate if cell else 0.80
            mode = "native" if self.rng.random() < native_rate else "latin-kept"
        if mode == "native":
            given = self._natural("GIVEN_NAME") if allow_natural else None
            family = self._natural("FAMILY_NAME") if allow_natural else None
            if given is not None:
                given = (given[0], f"name:native:{given[1]}")
            else:
                given = (self.loc.first_name(), f"name:native:faker:{self.lang}")
            if family is not None:
                family = (family[0], f"name:native:{family[1]}")
            else:
                family = (self.loc.last_name(), f"name:native:faker:{self.lang}")
            return given, family, "native"
        return (
            (self.en.first_name(), "name:latin-kept:faker:en"),
            (self.en.last_name(), "name:latin-kept:faker:en"),
            "latin-kept",
        )

    def _digits(self, length):
        return "".join(str(self.rng.randrange(10)) for _ in range(length))

    def _letters(self, length):
        return "".join(self.rng.choice(string.ascii_uppercase) for _ in range(length))

    def _date_between(self, start: date, end: date) -> date:
        return date.fromordinal(self.rng.randint(start.toordinal(), end.toordinal()))

    def _org_person_id(self):
        """A broad IT-system surface family shared by MRN-like person IDs."""
        generators = (
            lambda: self._digits(8),
            lambda: f"{self._digits(2)}-{self._digits(4)}-{self._digits(2)}",
            lambda: f"{self._letters(2)}{self._digits(7)}",
            lambda: f"{self._digits(4)}/{self._digits(6)}",
            lambda: f"{self._letters(1)}-{self._digits(3)}-{self._digits(4)}",
        )
        return self.rng.choice(generators)()

    def _monetary(self):
        amount = self.rng.randint(100, 99999)
        if self.locale_renderer is not None:
            surface, form = self.locale_renderer.render_monetary(amount, self.rng)
            return surface, f"monetary:locale-render-v1:{form}"
        return str(amount), "lex"

    def _loc_or_en(self, attr, *a, **k):
        for f, prov in ((self.loc, f"faker:{self.lang}"), (self.en, "faker:en")):
            try:
                return str(getattr(f, attr)(*a, **k)), prov
            except AttributeError:
                continue
        return None

    def fill(
        self,
        tag,
        orig_value,
        *,
        seed=None,
        name_mode=None,
        entity_key=None,
        surface_route="policy",
    ):
        if surface_route not in {"policy", "fresh-faker", "exact-empirical"}:
            raise ValueError(f"unsupported surface route: {surface_route!r}")
        if surface_route == "fresh-faker" and tag not in FAKER_SURFACE_TAGS:
            raise ValueError(f"fresh Faker route is unavailable for {tag}")
        key = (tag, orig_value if entity_key is None else entity_key, surface_route)
        if self._document_cache is not None and key in self._document_cache:
            return self._document_cache[key]
        if seed is not None:
            self.reseed_document(seed)
        value = self._fill_uncached(
            tag,
            orig_value,
            name_mode=name_mode,
            surface_route=surface_route,
        )
        if self._document_cache is not None:
            self._document_cache[key] = value
        return value

    def _fill_uncached(self, tag, orig_value, *, name_mode=None, surface_route="policy"):
        """(value, provenance) for one placeholder TAG."""
        if surface_route == "exact-empirical":
            value, provenance = self._natural(tag, force=True)
            return value, f"selective-exact-empirical:{provenance}"
        allow_natural = surface_route != "fresh-faker"
        if surface_route == "policy" and tag in TEMPORAL_TAGS:
            # Historically the original English surface passed through
            # verbatim for temporal coherence — which trained every
            # language on English date conventions and cost Korean a
            # measured -.05 span F1. With a locale profile the *value*
            # still passes through (coherence holds), but the surface
            # re-renders per the language's convention distribution.
            temporal_cell = self.surface_policy.for_tag(self.lang, tag) if self.surface_policy else None
            localize = temporal_cell is None or temporal_cell.localize_temporal_surfaces
            if self.locale_renderer is not None and tag != "TIME" and localize:
                rendered, form = self.locale_renderer.render_date(orig_value, self.rng)
                if form != "keep":
                    return rendered, f"temporal:locale-render-v1:{form}"
            return orig_value, "orig:coherence-temporal-v1"
        # Faker has no cross-locale county contract.  The old fallback used
        # ``city()``, which made the surface contradict the fine COUNTY label
        # whenever COUNTY was the document's only location field.  Retain the
        # source value just as we already do for a linked location tuple.
        if surface_route == "policy" and tag == "COUNTY":
            return orig_value, "orig:coherence-county-v2"
        if surface_route == "policy" and self._preserve_linked_locations and tag in LOCATION_TAGS:
            return orig_value, "orig:coherence-location-v1"
        cell = self.surface_policy.for_tag(self.lang, tag) if self.surface_policy else None
        if allow_natural and cell is not None and cell.non_empirical_fallback == "source":
            natural = self._natural(tag)
            if natural is not None:
                return natural
            return orig_value, "orig:surface-policy-source-v1"
        (g, g_prov), (f, f_prov), mode = self._name_parts(
            tag,
            mode=name_mode,
            allow_natural=allow_natural,
        )
        en = self.en
        # CJK native names concatenate family+given; others space-join.
        full = (f + g) if (mode == "native" and self.lang in ("zh", "ja", "ko")) else f"{g} {f}"
        full_prov = f"name:{mode}"
        if "natural-pool:" in g_prov or "natural-pool:" in f_prov:
            full_prov = f"name:{mode}:composed:{g_prov}|{f_prov}"
        direct = {
            "GIVEN_NAME": (g, g_prov),
            "FAMILY_NAME": (f, f_prov),
            "PERSON_NAME": (full, full_prov),
            "MIDDLE_NAME": (en.first_name(), "name:latin"),
            "COUNTRY": (COUNTRY[self.lang], "lex"),
            "EMAIL": (en.ascii_email(), "faker:ascii"),
            "URL": (en.uri(), "faker:ascii"),
            "USERNAME": (en.user_name(), "faker:ascii"),
            "CARD_NUMBER": (en.credit_card_number(), "faker:ascii"),
            "IBAN": (en.iban(), "faker:ascii"),
            "IP_ADDRESS": (en.ipv4(), "faker:ascii"),
            "IPV4": (en.ipv4(), "faker:ascii"),
            "IPV6": (en.ipv6(), "faker:ascii"),
            "MAC_ADDRESS": (en.mac_address(), "faker:ascii"),
            "MEDICAL_RECORD_NUMBER": (
                self._org_person_id(),
                "bespoke:org-person-id-v1",
            ),
            "AGE": (str(self.rng.randint(18, 90)), "lex"),
            "MONETARY_AMOUNT": self._monetary(),
            # label-catalog decision 1 (2026-08-07): ssn is strictly the
            # US SSN. The locale-parameterized Faker ssn() provider used
            # to leave e.g. German Steuer-ID surfaces under the ssn
            # label; US-style always. NATIONAL_ID keeps the locale
            # provider deliberately.
            "SSN": (en.ssn(), "faker:en-us-ssn-v1"),
        }
        if allow_natural and tag in {"PERSON_NAME", "ORGANIZATION", "OCCUPATION", "JOB_TITLE"}:
            natural = self._natural(tag)
            if natural is not None:
                return natural
        if tag in direct:
            v, prov = direct[tag]
            return str(v), prov
        by_attr = {
            "PHONE_NUMBER": ("phone_number", (), {}),
            "FAX_NUMBER": ("phone_number", (), {}),
            "STREET_ADDRESS": ("street_address", (), {}),
            "ADDRESS": ("address", (), {}),
            "CITY": ("city", (), {}),
            "REGION": ("administrative_unit", (), {}),
            "STATE": ("administrative_unit", (), {}),
            "POSTAL_CODE": ("postcode", (), {}),
            "NATIONAL_ID": ("ssn", (), {}),
            "ORGANIZATION": ("company", (), {}),
            "HEALTHCARE_ORG": ("company", (), {}),
            "FINANCIAL_ORG": ("company", (), {}),
            "OCCUPATION": ("job", (), {}),
            "JOB_TITLE": ("job", (), {}),
        }
        if tag == "DATE":
            return self._date_between(date(1970, 1, 1), date(2030, 1, 1)).strftime("%Y-%m-%d"), (
                "bespoke:seeded-temporal-v1"
            )
        if tag == "DATE_TIME":
            day = self._date_between(date(1970, 1, 1), date(2030, 1, 1))
            return f"{day:%Y-%m-%d} {self.rng.randrange(24):02d}:{self.rng.randrange(60):02d}", (
                "bespoke:seeded-temporal-v1"
            )
        if tag == "TIME":
            return f"{self.rng.randrange(24):02d}:{self.rng.randrange(60):02d}", (
                "bespoke:seeded-temporal-v1"
            )
        if tag == "DATE_OF_BIRTH":
            return self._date_between(date(1935, 1, 1), date(2008, 1, 1)).strftime("%Y-%m-%d"), (
                "bespoke:seeded-temporal-v1"
            )
        if tag in by_attr:
            attr, a, k = by_attr[tag]
            got = self._loc_or_en(attr, *a, **k)
            if got:
                v, prov = got
                return v.replace("\n", " "), prov
        return orig_value, "orig"

    def surface_metadata(self, tag, provenance):
        cell = self.surface_policy.for_tag(self.lang, tag) if self.surface_policy else None
        return {
            "policy": (
                {
                    "recipe_version": self.surface_policy.version,
                    "natural_surface_rate": cell.natural_surface_rate,
                    "natural_min_distinct_full_rate": cell.natural_min_distinct_full_rate,
                    "natural_count_temperature": cell.natural_count_temperature,
                    "native_name_rate": cell.native_name_rate,
                    "non_empirical_fallback": cell.non_empirical_fallback,
                    "fresh_faker_beta": cell.fresh_faker_beta,
                    "replacement_char_alpha": cell.replacement_char_alpha,
                }
                if cell is not None
                else None
            ),
            "pool_entries": (self.surface_pool.provenance_metadata(provenance) if self.surface_pool else []),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=sorted(FAKER_LOCALE))
    ap.add_argument(
        "--slot-morphology",
        default=None,
        help="path to a slot-morphology config (scripts/pii_slot_morphology.yaml). Inflects the "
        "drawn surface for the case its slot demands, per language. Omitted means the layer is "
        "off; a language absent from the config is never altered.",
    )
    ap.add_argument("--wave", default=None)
    ap.add_argument("--ph", default=os.path.join(PROD, "nemotron-ph-7000.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "legacy single-stream seed; explicit use preserves historical draw order. "
            "Omit to use the configured reproducibility seed tree"
        ),
    )
    ap.add_argument("--reproducibility-config", type=Path, default=REPRODUCIBILITY_CONFIG_PATH)
    ap.add_argument(
        "--reproducibility-seed",
        type=int,
        default=None,
        help="override the configured root seed while retaining named, per-document forks",
    )
    ap.add_argument(
        "--realization-index",
        type=int,
        default=0,
        help=(
            "late-materialization replica index; separate indices redraw entity values "
            "without changing the admitted carrier (default: 0)"
        ),
    )
    ap.add_argument(
        "--source-language-code",
        default="en",
        help="carrier language used for source-scoped repairs when wave metadata omits it (default: en)",
    )
    ap.add_argument("--materializer-config", type=Path, default=MATERIALIZER_POLICY_PATH)
    ap.add_argument("--surface-pool", type=Path)
    ap.add_argument(
        "--surface-pool-split",
        choices=("train", "audit"),
        default="train",
        help="partition of --surface-pool eligible for draws (default: train)",
    )
    ap.add_argument(
        "--surface-recipe",
        type=Path,
        help="validated per-language, per-family realization policy YAML",
    )
    ap.add_argument(
        "--placeholder-identity",
        default=None,
        help="optional placeholder identity policy recorded in every materialized row",
    )
    ap.add_argument(
        "--materialization-version",
        default=None,
        help="optional new dataset revision identity; defaults to the established policy-derived version",
    )
    ap.add_argument(
        "--natural-surface-rate",
        type=float,
        default=None,
        help="probability of drawing an available natural-pool value; defaults to 0.5 with --surface-pool",
    )
    ap.add_argument(
        "--natural-min-distinct-full-rate",
        type=int,
        default=100,
        help=(
            "distinct values required for the full --natural-surface-rate; smaller groups scale the rate "
            "linearly (default: 100)"
        ),
    )
    ap.add_argument(
        "--natural-count-temperature",
        type=float,
        default=0.5,
        help="exponent applied to observed pool counts when sampling (0=uniform, 1=empirical; default: 0.5)",
    )
    ap.add_argument(
        "--unlocalized-categorical-policy",
        choices=("drop", "keep"),
        default="drop",
        help="drop rows containing categorical values without a reviewed target-language mapping (default: drop)",
    )
    ap.add_argument(
        "--locale-profile",
        type=Path,
        default=DEFAULT_LOCALE_PROFILE,
        help=(
            "locale surface-convention profile YAML; renders temporal and monetary surfaces per "
            "the target language's convention distribution instead of passing English source "
            "forms through. Defaults to the project profile because omitting it left 28.7%% of "
            "Polish spans carrying English month names against 8.6%% with it (2026-08-20). The "
            "resolved path is recorded in each row's materialization block, so the default is "
            "visible in provenance rather than implicit. Pass 'none' to disable."
        ),
    )
    args = ap.parse_args()
    slot_morphology = SlotMorphology.load(args.slot_morphology) if args.slot_morphology else None
    if args.seed is not None and args.reproducibility_seed is not None:
        ap.error("--seed is the legacy stream; it cannot be combined with --reproducibility-seed")
    if args.realization_index < 0:
        ap.error("--realization-index must be nonnegative")
    if args.seed is not None and args.realization_index:
        ap.error("--realization-index requires the configured seed tree; omit legacy --seed")

    seed_tree = (
        None
        if args.seed is not None
        else load_seed_tree(args.reproducibility_config, root_seed=args.reproducibility_seed)
    )
    initial_seed = args.seed if args.seed is not None else seed_tree.root_seed
    wave = args.wave or os.path.join(PROD, f"tg-{args.lang}-7k.jsonl")
    out = args.out or os.path.join(PROD, f"{args.lang}-instantiated-v1.jsonl")
    repair_policy = load_repair_policy(args.materializer_config)

    ph = {json.loads(l)["id"]: json.loads(l)["ph_map"] for l in open(args.ph)}
    surface_pool = (
        SurfacePool.load(args.surface_pool, split=args.surface_pool_split) if args.surface_pool else None
    )
    surface_policy = SurfaceMixPolicy.load(args.surface_recipe) if args.surface_recipe else None
    natural_surface_rate = (
        args.natural_surface_rate
        if args.natural_surface_rate is not None
        else (0.5 if surface_pool is not None else 0.0)
    )
    if str(args.locale_profile).lower() in {"none", ""}:
        args.locale_profile = None
    locale_renderer = None
    if args.locale_profile is not None:
        from pii_locale_render import LocaleRenderer

        locale_renderer = LocaleRenderer(args.lang, args.locale_profile)
    filler = Filler(
        args.lang,
        initial_seed,
        surface_pool,
        natural_surface_rate,
        args.natural_min_distinct_full_rate,
        args.natural_count_temperature,
        args.unlocalized_categorical_policy == "drop",
        locale_renderer,
        surface_policy,
    )
    surface_pool_sha256 = None
    if args.surface_pool:
        surface_pool_sha256 = hashlib.sha256(args.surface_pool.read_bytes()).hexdigest()
    locale_profile_sha256 = None
    if args.locale_profile is not None:
        locale_profile_sha256 = hashlib.sha256(args.locale_profile.read_bytes()).hexdigest()
    materializer_config_sha256 = hashlib.sha256(args.materializer_config.read_bytes()).hexdigest()
    n = n_spans = n_skipped = n_semantic_filtered = n_cjk_name_reordered = 0
    semantic_filter_counts = Counter()
    prov_counts = Counter()
    with open(out, "w") as fo:
        for line in open(wave):
            r = json.loads(line)
            source_language_code = r.get("source_lang_code") or args.source_language_code
            source_tokens = SRC_TOK.findall(r["src"])
            src_idx = {}
            source_contract_valid = True
            for tag, idx in source_tokens:
                prior = src_idx.setdefault(idx, tag)
                if prior != tag:
                    source_contract_valid = False
            hyp_toks = HYP_TOK.findall(r["hyp"])
            source_counts = Counter(idx for _, idx in source_tokens)
            hypothesis_counts = Counter(idx for _, idx in hyp_toks)
            if not source_contract_valid or hypothesis_counts != source_counts:
                n_skipped += 1
                continue
            # '#k' suffixes come from --split-over pieces; the ph_map is
            # per original doc. Linked revisions may repeat a placeholder
            # within a piece, but its index still resolves to one map entry.
            carrier_id = carrier_document_id(r["id"])
            ph_map = ph[carrier_id]
            name_groups = name_mode_groups(ph_map)
            document_seed = None
            if seed_tree is not None:
                document_seed = seed_tree.fork(
                    "named-entity-materializer",
                    args.lang,
                    carrier_id,
                    args.realization_index,
                )
            filler.begin_document(ph_map, seed=document_seed)
            if filler.document_rejection_reasons:
                n_semantic_filtered += 1
                semantic_filter_counts.update(filler.document_rejection_reasons)
                continue
            spans, span_provenance, parts, last, output_length = [], [], [], 0, 0
            document_prov_counts = Counter()
            for m in HYP_TOK.finditer(r["hyp"]):
                idx = m.group(2)
                tag = src_idx[idx]
                placeholder = f"[{tag}_{idx}]"
                info = ph_map.get(placeholder, {})
                source_value = info.get("value", "")
                entity_seed = (
                    None
                    if seed_tree is None
                    else entity_realization_seed(
                        seed_tree,
                        args.lang,
                        carrier_id,
                        args.realization_index,
                        tag,
                        source_value,
                    )
                )
                name_group = name_groups.get(placeholder)
                name_mode = None
                if seed_tree is not None and name_group is not None:
                    mode_seed = seed_tree.fork(
                        "named-entity-materializer",
                        args.lang,
                        carrier_id,
                        args.realization_index,
                        "name-script-mode",
                        *name_group,
                    )
                    name_mode = "native" if random.Random(mode_seed).random() < 0.80 else "latin-kept"
                val, prov = filler.fill(
                    tag,
                    source_value,
                    seed=entity_seed,
                    name_mode=name_mode,
                )
                document_prov_counts[prov.split(":")[0]] += 1
                prefix = r["hyp"][last : m.start()]
                slot_meta = {}
                span_chars = None
                if slot_morphology is not None:
                    realized = slot_morphology.realize(
                        args.lang,
                        val,
                        prefix,
                        r["hyp"][m.end() :],
                        tag=info.get("src_label") or tag.lower(),
                    )
                    val = realized.surface
                    span_chars = realized.span_length
                    slot_meta = realized.metadata
                parts.append(prefix)
                output_length += len(prefix)
                start = output_length
                parts.append(val)
                output_length += len(val)
                # a separable clitic is written into the text but stays outside
                # the entity span, so the span may be shorter than the surface
                end = start + (len(val) if span_chars is None else span_chars)
                output_length = start + len(val)
                spans.append([start, end, info.get("src_label") or tag.lower()])
                span_provenance.append(
                    {
                        "start": start,
                        "end": end,
                        "placeholder": placeholder,
                        "generator": prov,
                        **filler.surface_metadata(tag, prov),
                        **slot_meta,
                    }
                )
                last = m.end()
            parts.append(r["hyp"][last:])
            text = "".join(parts)
            # Repair only source families whose separate name placeholders are
            # known to inherit a defective order. Target language alone does
            # not authorize rewriting a native or future source.
            row_view = {
                "lang": args.lang,
                "text": text,
                "spans": spans,
                "materialization": {"source_language_code": source_language_code},
            }
            if requires_family_first_repair(row_view, repair_policy):
                if reorder_cjk_names(row_view, max_gap=2):
                    text = row_view["text"]
                    n_cjk_name_reordered += 1
                    for span, record in zip(spans, span_provenance):
                        record["start"], record["end"] = span[0], span[1]
            if args.unlocalized_categorical_policy == "drop":
                guard_reasons = semantic_guard_reasons(text, spans)
                if guard_reasons:
                    n_semantic_filtered += 1
                    semantic_filter_counts.update(guard_reasons)
                    continue
            prov_counts.update(document_prov_counts)
            fo.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "lang": args.lang,
                        "text": text,
                        "spans": spans,
                        "span_provenance": span_provenance,
                        "materialization": {
                            "version": (
                                args.materialization_version
                                or (
                                    SEMANTIC_FILTER_VERSION
                                    if args.unlocalized_categorical_policy == "drop"
                                    else (
                                        SURFACE_FEEDBACK_VERSION if surface_pool else MATERIALIZATION_VERSION
                                    )
                                )
                            ),
                            "seed": initial_seed,
                            **(
                                {}
                                if seed_tree is None
                                else {
                                    "seed_tree": {
                                        "scheme": seed_tree.scheme,
                                        "root_seed": seed_tree.root_seed,
                                        "document_seed": document_seed,
                                        "entity_fork": "tag-source-value-v1",
                                        "name_mode_fork": "consecutive-name-components-v1",
                                        "realization_index": args.realization_index,
                                    }
                                }
                            ),
                            "surface_pool": str(args.surface_pool) if args.surface_pool else None,
                            "surface_pool_sha256": surface_pool_sha256,
                            "surface_pool_version": surface_pool.version if surface_pool else None,
                            "surface_pool_split": args.surface_pool_split if surface_pool else None,
                            "surface_recipe": surface_policy.receipt() if surface_policy else None,
                            "natural_surface_rate": natural_surface_rate,
                            "natural_min_distinct_full_rate": args.natural_min_distinct_full_rate,
                            "natural_count_temperature": args.natural_count_temperature,
                            "unlocalized_categorical_policy": args.unlocalized_categorical_policy,
                            "locale_profile": str(args.locale_profile) if args.locale_profile else None,
                            # recorded for the same reason: a defaulted or
                            # omitted config must be readable from the output
                            "slot_morphology": (str(args.slot_morphology) if args.slot_morphology else None),
                            "locale_profile_sha256": locale_profile_sha256,
                            "materializer_config": str(args.materializer_config),
                            "materializer_config_sha256": materializer_config_sha256,
                            "semantic_guard_ruleset": (
                                SEMANTIC_GUARD_RULESET
                                if args.unlocalized_categorical_policy == "drop"
                                else None
                            ),
                            "temporal_policy": "retain-source-coherence-v1",
                            "county_policy": "retain-source-coherence-v2",
                            "linked_location_policy": "retain-source-coherence-v1",
                            "translator": r.get("translator"),
                            "source_language": r.get("source_lang"),
                            "source_language_code": source_language_code,
                            "placeholder_recovered": r.get("recovered"),
                            **(
                                {"placeholder_identity": args.placeholder_identity}
                                if args.placeholder_identity
                                else {}
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
            n_spans += len(spans)
    print(
        f"INSTANTIATE {args.lang}: {n} docs, {n_spans} spans, {n_skipped} index-broken skipped "
        f"and {n_semantic_filtered} semantic-risk rows filtered, "
        f"{n_cjk_name_reordered} CJK name orders corrected "
        f"(seed={initial_seed}, realization={args.realization_index}) -> {out}"
    )
    print(f"INSTANTIATE {args.lang}: fill provenance {dict(prov_counts)}")
    print(f"INSTANTIATE {args.lang}: semantic filters {dict(semantic_filter_counts)}")


if __name__ == "__main__":
    main()
