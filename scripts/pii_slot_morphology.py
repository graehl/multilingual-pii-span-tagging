#!/usr/bin/env python
"""Inflect a realized transport surface for the slot its placeholder sits in.

A transported placeholder occupies a slot whose target language may demand a
case the placeholder token cannot carry (gaps/pii-transport-placeholder-morphology.md).
Turkish writes that case as a separable clitic, so the surface keeps its
citation form and the span excludes the suffix. Slavic rewrites the ending of
the name itself, so the span must cover the inflected form. Both are driven
entirely from scripts/pii_slot_morphology.yaml.

**This module contains no language knowledge.** Every governor, vowel class and
ending lives in the config, so a language absent from it is left exactly as
drawn and nothing changes through an invisible default. Callers opt in by
passing a config path; passing none disables the whole layer.

It never changes the tagging objective. The caller keeps its span label
unchanged and records the detected slot as provenance, so a surface model can
bin one entity's realizations across cases rather than memorizing them as
unrelated strings.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEMA = "pii-slot-morphology-v1"
# last alphabetic token before the slot, and the first after it
_LEFT = re.compile(r"([^\W\d_]+)\W*$", re.UNICODE)
_RIGHT = re.compile(r"^\W*([^\W\d_]+)", re.UNICODE)


@dataclass(frozen=True)
class Realization:
    """What the layer did, so the caller can record it rather than guess."""

    surface: str
    slot: str | None = None
    applied: bool = False
    reason: str = "no-config"
    # How many leading characters of ``surface`` the entity span covers. None
    # means all of it. A Turkish clitic is written into the text but sits
    # outside the span, which is what makes "Ankara'ya" annotate as "Ankara".
    span_chars: int | None = None

    @property
    def span_length(self) -> int:
        return len(self.surface) if self.span_chars is None else self.span_chars

    @property
    def metadata(self) -> dict:
        meta = {"slot_morphology": self.reason}
        if self.slot:
            meta["slot_case"] = self.slot
        return meta


class SlotMorphology:
    def __init__(self, config: dict):
        if config.get("schema") != SCHEMA:
            raise ValueError(f"expected schema {SCHEMA}, got {config.get('schema')!r}")
        self.languages = config.get("languages") or {}
        # absent means no tag is eligible: the layer must be told what it may
        # touch rather than defaulting to everything
        self.applies_to_tags = frozenset(config.get("applies_to_tags") or ())

    @classmethod
    def load(cls, path: str | Path) -> "SlotMorphology":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def knows(self, lang: str) -> bool:
        return lang in self.languages

    def prompt_frame_instruction(self, lang: str) -> str | None:
        """Extra prompt text asking for a frame that needs no inflected name.

        Only meaningful for a translator whose prompt can carry instructions;
        TranslateGemma's structured interface cannot, which is why the caller
        decides whether to use it.
        """
        spec = self.languages.get(lang) or {}
        text = spec.get("prompt_frame_instruction")
        return str(text).strip() if text else None

    def detect_slot(self, lang: str, left_context: str, right_context: str = "") -> str | None:
        """Which case the slot demands, from the config's governor tables."""
        spec = self.languages.get(lang)
        if not spec:
            return None
        left = spec.get("governors_left") or {}
        if left and (m := _LEFT.search(left_context or "")):
            if (case := left.get(m.group(1).casefold())) is not None:
                return case
        right = spec.get("governors_right") or {}
        if right and (m := _RIGHT.search(right_context or "")):
            if (case := right.get(m.group(1).casefold())) is not None:
                return case
        return None

    def realize(
        self,
        lang: str,
        surface: str,
        left_context: str,
        right_context: str = "",
        tag: str | None = None,
    ) -> Realization:
        if tag is not None and tag not in self.applies_to_tags:
            # a date or an identifier is never inflected, whatever slot it is in
            return Realization(surface, reason=f"tag-not-eligible:{tag}")
        spec = self.languages.get(lang)
        if not spec:
            return Realization(surface, reason="language-not-configured")
        slot = self.detect_slot(lang, left_context, right_context)
        if slot is None:
            return Realization(surface, reason="no-governing-slot")
        strategy = spec.get("strategy")
        if strategy == "clitic_suffix":
            return self._clitic(spec, surface, slot)
        if strategy == "fusional_rewrite":
            return self._fusional(spec, surface, slot)
        return Realization(surface, slot, reason=f"unknown-strategy:{strategy}")

    def _clitic(self, spec: dict, surface: str, slot: str) -> Realization:
        table = (spec.get("suffixes") or {}).get(slot)
        if not table:
            return Realization(surface, slot, reason="no-suffix-for-slot")
        back = set(spec.get("vowels", {}).get("back", ""))
        front = set(spec.get("vowels", {}).get("front", ""))
        harmony = None
        for ch in reversed(surface.casefold()):
            if ch in back:
                harmony = "back"
                break
            if ch in front:
                harmony = "front"
                break
        if harmony is None:
            return Realization(surface, slot, reason="no-vowel-for-harmony")
        last = surface.casefold()[-1:]
        is_vowel = last in back | front
        if "after_vowel" in table:
            branch = table["after_vowel"] if is_vowel else table.get("after_consonant", {})
        else:
            voiceless = set(spec.get("voiceless", ""))
            branch = table["after_voiceless"] if last in voiceless else table.get("after_voiced", {})
        suffix = branch.get(harmony)
        if not suffix:
            return Realization(surface, slot, reason="no-suffix-branch")
        # separable clitic: joined in the text, excluded from the span, which is
        # exactly what Turkish orthography wants -- the apostrophe is the delimiter
        excludes = bool(spec.get("span_excludes_suffix"))
        return Realization(
            surface + suffix,
            slot,
            applied=True,
            reason=f"clitic:{slot}:{harmony}",
            span_chars=len(surface) if excludes else None,
        )

    def _fusional(self, spec: dict, surface: str, slot: str) -> Realization:
        for bad in spec.get("reject_if_contains") or []:
            if bad in surface:
                # punctuated strings in these slots were untranslated English,
                # not names, and inflecting them produces nonsense
                return Realization(surface, slot, reason="rejected-punctuation")
        tokens = surface.split()
        max_tokens = spec.get("max_tokens")
        if max_tokens and len(tokens) > max_tokens:
            return Realization(surface, slot, reason=f"too-many-tokens:{len(tokens)}")
        if spec.get("all_tokens_must_match") and len(tokens) > 1:
            # a fusional language inflects every word of a name, so a partial
            # match would emit "Witold Szkodę"; all or nothing
            out = []
            for token in tokens:
                sub = self._fusional_token(spec, token, slot)
                if not sub.applied:
                    return Realization(surface, slot, reason=f"partial-name:{sub.reason}")
                out.append(sub.surface)
            return Realization(" ".join(out), slot, applied=True, reason=f"fusional:{slot}")
        return self._fusional_token(spec, surface, slot)

    def _fusional_token(self, spec: dict, surface: str, slot: str) -> Realization:
        for rule in spec.get("rewrite") or []:
            ending = rule.get("ends_with")
            if not ending or not surface.casefold().endswith(ending.casefold()):
                continue
            replacement = (rule.get("cases") or {}).get(slot)
            if replacement is None:
                # matched the stem class but this case is deliberately unencoded
                return Realization(surface, slot, reason=f"case-not-encoded:{slot}")
            stem = surface[: len(surface) - len(ending)]
            return Realization(stem + replacement, slot, applied=True, reason=f"fusional:{slot}")
        return Realization(surface, slot, reason="no-matching-stem-class")


def _self_test() -> int:
    here = Path(__file__).with_name("pii_slot_morphology.yaml")
    m = SlotMorphology.load(here)
    cases = [
        # Turkish: separable clitic, harmony from the surface's last vowel
        ("tr", "Ankara", "", " kadar", "Ankara'ya", True),
        ("tr", "İzmir", "", " kadar", "İzmir'e", True),
        ("tr", "Bodrum", "", " sonra", "Bodrum'dan", True),
        ("tr", "Ankara", "", " ve", "Ankara", False),
        # Polish: fusional, velar stems take -i not -y
        ("pl", "Warszawa", "pojechać do ", "", "Warszawy", True),
        ("pl", "Praga", "do ", "", "Pragi", True),
        ("pl", "Warszawa", "widzę ", "", "Warszawa", False),
        # Russian
        ("ru", "Москва", "из ", "", "Москвы", True),
        ("ru", "Рига", "из ", "", "Риги", True),
        ("ru", "Москва", "в ", "", "Москве", True),
        # Ukrainian
        ("uk", "Полтава", "до ", "", "Полтави", True),
        # unconfigured language is never touched
        ("de", "Berlin", "nach ", "", "Berlin", False),
    ]
    # regressions from the 2026-08-20 Polish instantiation smoke, where the
    # first rule set produced six wrong inflections out of eight
    regressions = [
        ("pl", "Austria", "do ", "location", "Austrii", True),
        ("pl", "Romania", "do ", "location", "Romanii", True),
        ("ru", "Австрия", "из ", "location", "Австрии", True),
        ("uk", "Австрія", "до ", "location", "Австрії", True),
        # every token of a name inflects, or none does
        ("pl", "Kaja Drela", "przez ", "person_name", "Kaję Drelę", True),
        ("pl", "Witold Szkoda", "przez ", "person_name", "Witold Szkoda", False),
        # untranslated English is not a name to inflect
        ("pl", "People ' s Republic of China", "z ", "location", "People ' s Republic of China", False),
    ]
    # a date in a genitive slot must survive untouched
    date = m.realize("pl", "25 January 2001", "od ", "", tag="date")
    print(
        f"{'ok ' if not date.applied else 'FAIL'} pl date in genitive slot -> {date.surface!r} ({date.reason})"
    )
    eligible = m.realize("pl", "Warszawa", "do ", "", tag="location")
    print(
        f"{'ok ' if eligible.applied else 'FAIL'} pl location in genitive slot -> {eligible.surface!r} ({eligible.reason})"
    )
    extra_bad = int(date.applied) + int(not eligible.applied)
    bad = 0
    for lang, surface, left, right, want, want_applied in cases:
        got = m.realize(lang, surface, left, right)
        ok = got.surface == want and got.applied == want_applied
        if lang == "tr" and got.applied:
            # the clitic is in the text but must not be inside the span
            ok = ok and got.span_length == len(surface)
        bad += not ok
        print(
            f"{'ok ' if ok else 'FAIL'} {lang} {surface!r} -> {got.surface!r} ({got.reason})"
            + ("" if ok else f"  expected {want!r} applied={want_applied}")
        )
    for lang, surface, left, tag, want, want_applied in regressions:
        got = m.realize(lang, surface, left, "", tag=tag)
        ok = got.surface == want and got.applied == want_applied
        bad += not ok
        print(
            f"{'ok ' if ok else 'FAIL'} {lang} {surface!r} -> {got.surface!r} ({got.reason})"
            + ("" if ok else f"  expected {want!r} applied={want_applied}")
        )
    bad += extra_bad
    print("PASS" if not bad else f"{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_self_test())
