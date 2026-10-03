#!/usr/bin/env python
"""Value-preserving locale rendering of convention-bearing PII surfaces.

Reads the policy distributions in `pii_locale_profiles.yaml` and
re-renders parsed values (dates, monetary amounts) in locale-native
conventions: native month-name forms (genitive where required), local
numeric orders, Japanese era years, Minguo, tabular civil Hijri, CJK
myriad grouping (万/億/만), Chinese bankers' numerals. The *value*
never changes (Hijri/era are the same date expressed in another
calendar); only the surface does, so document-internal temporal
coherence — the reason the materializer historically passed original
English surfaces through — is preserved with on-distribution surfaces.

Draws are made from a caller-supplied `random.Random`, so materializer
seeding governs reproducibility.
"""

import datetime
import re
from pathlib import Path

import yaml
from pii_fresh13_date_localize import MONTHS as _F13_MONTHS
from pii_fresh13_date_localize import WEEKDAYS as _F13_WEEKDAYS
from pii_fresh13_date_localize import native as _f13_native
from pii_fresh13_date_localize import numeric_local as _f13_numeric

PROFILE_PATH = Path(__file__).resolve().parent / "pii_locale_profiles.yaml"

MONTHS = dict(_F13_MONTHS)
MONTHS.update(
    {
        "de": "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split(),
        "en": "January February March April May June July August September October November December".split(),
        "es": "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(),
        "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    }
)

WEEKDAYS = dict(_F13_WEEKDAYS)
WEEKDAYS.update(
    {
        "de": "Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag".split(),
        "en": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
        "es": "lunes martes miércoles jueves viernes sábado domingo".split(),
        "fr": "lundi mardi mercredi jeudi vendredi samedi dimanche".split(),
        "ja": "月曜日 火曜日 水曜日 木曜日 金曜日 土曜日 日曜日".split(),
        "ko": "월요일 화요일 수요일 목요일 금요일 토요일 일요일".split(),
        "zh": "星期一 星期二 星期三 星期四 星期五 星期六 星期日".split(),
    }
)

# explicit list: several display names contain internal spaces
HIJRI_MONTHS = [
    "محرم",
    "صفر",
    "ربيع الأول",
    "ربيع الآخر",
    "جمادى الأولى",
    "جمادى الآخرة",
    "رجب",
    "شعبان",
    "رمضان",
    "شوال",
    "ذو القعدة",
    "ذو الحجة",
]

DATE_FORMS = {
    "keep",
    "native",
    "local_numeric",
    "weekday_native",
    "era",
    "minguo",
    "hijri",
    "full_char",
}
DATE_FORM_LANG = {"era": {"ja"}, "minguo": {"zh"}, "hijri": {"ar"}, "full_char": {"zh"}}
MONETARY_FORMS = {"digits", "latin_sep", "native_sep", "myriad", "bankers", "char_num"}
MONETARY_FORM_LANG = {"myriad": {"zh", "ja", "ko"}, "bankers": {"zh"}, "char_num": {"zh"}}

# native thousands-separator style (frontier spot-check 2026-08-07:
# comma grouping is Anglo; most Final20 languages group with dot or
# a space in ordinary text)
NATIVE_GROUPING = {
    "de": ".",
    "es": ".",
    "id": ".",
    "it": ".",
    "nl": ".",
    "pt": ".",
    "tr": ".",
    "vi": ".",
    "cs": "\u00a0",
    "fr": "\u00a0",
    "pl": "\u00a0",
    "ru": "\u00a0",
    "sv": "\u00a0",
    "uk": "\u00a0",
    "ar": ",",
    "en": ",",
    "hi": ",",
    "ja": ",",
    "ko": ",",
    "zh": ",",
}

ARABIC_INDIC_DIGITS = str.maketrans("0123456789,", "٠١٢٣٤٥٦٧٨٩٬")


def _indian_group(n):
    digits = str(n)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    pairs = []
    while head:
        pairs.append(head[-2:])
        head = head[:-2]
    return ",".join(reversed(pairs)) + "," + tail


def _native_sep(lang, n):
    if lang == "ar":
        return f"{n:,}".translate(ARABIC_INDIC_DIGITS)
    if lang == "hi":
        return _indian_group(n)
    sep = NATIVE_GROUPING.get(lang, ",")
    return f"{n:,}".replace(",", sep)


ISO_YMD = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
NUM_3 = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$")
MONTH_MDY = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$")
MONTH_DMY = re.compile(r"^(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$")
DATETIME_SPLIT = re.compile(r"^(\S+)([ T]\d{1,2}:\d{2}(?::\d{2})?)$")
EN_MONTHS = {m.lower(): i + 1 for i, m in enumerate(MONTHS["en"])}
EN_MONTHS.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS["en"])})


def parse_date(surface):
    """Return (y, m, d) or None; plain unambiguous-enough dates only."""
    s = surface.strip()
    m = ISO_YMD.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = NUM_3.match(s)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a > 12 and b <= 12:
                mo, d = b, a
            else:
                mo, d = a, b  # MDY read for ambiguous; value truth moot for NER
        else:
            m = MONTH_MDY.match(s)
            if m and m.group(1).lower() in EN_MONTHS:
                y, mo, d = int(m.group(3)), EN_MONTHS[m.group(1).lower()], int(m.group(2))
            else:
                m = MONTH_DMY.match(s)
                if m and m.group(2).lower() in EN_MONTHS:
                    y, mo, d = int(m.group(3)), EN_MONTHS[m.group(2).lower()], int(m.group(1))
                else:
                    return None
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return None
    return y, mo, d


def _native(lang, y, mo, d):
    if lang in ("ja", "zh"):
        return f"{y}年{mo}月{d}日"
    if lang == "ko":
        return f"{y}년 {mo}월 {d}일"
    if lang == "de":
        return f"{d}. {MONTHS['de'][mo - 1]} {y}"
    if lang == "es":
        return f"{d} de {MONTHS['es'][mo - 1]} de {y}"
    if lang in ("fr", "en"):
        n = f"{d} {MONTHS[lang][mo - 1]} {y}"
        return n if lang == "fr" else f"{MONTHS['en'][mo - 1]} {d}, {y}"
    return _f13_native(lang, y, mo, d)


def _local_numeric(lang, y, mo, d):
    if lang in ("ja", "zh", "ko"):
        return f"{y}.{mo:02d}.{d:02d}" if lang == "ko" else f"{y}/{mo}/{d}"
    if lang in ("de",):
        return f"{d:02d}.{mo:02d}.{y}"
    if lang in ("es", "fr", "en"):
        return f"{d:02d}/{mo:02d}/{y}" if lang != "en" else f"{mo:02d}/{d:02d}/{y}"
    return _f13_numeric(lang, y, mo, d)


def _era_ja(y, mo, d):
    if y >= 2019:
        era, ey = "令和", y - 2018
    elif y >= 1989:
        era, ey = "平成", y - 1988
    elif y >= 1926:
        era, ey = "昭和", y - 1925
    else:
        era, ey = "大正", y - 1911
    return f"{era}{'元' if ey == 1 else ey}年{mo}月{d}日"


def _gregorian_to_hijri(y, mo, d):
    """Convert Gregorian to the deterministic tabular Islamic civil calendar."""
    adjustment = (14 - mo) // 12
    gregorian_year = y + 4800 - adjustment
    gregorian_month = mo + 12 * adjustment - 3
    julian_day = (
        d
        + (153 * gregorian_month + 2) // 5
        + 365 * gregorian_year
        + gregorian_year // 4
        - gregorian_year // 100
        + gregorian_year // 400
        - 32045
    )
    days = julian_day - 1948440 + 10632
    cycles = (days - 1) // 10631
    days = days - 10631 * cycles + 354
    leap_position = ((10985 - days) // 5316) * ((50 * days) // 17719) + (days // 5670) * (
        (43 * days) // 15238
    )
    days = (
        days
        - ((30 - leap_position) // 15) * ((17719 * leap_position) // 50)
        - (leap_position // 16) * ((15238 * leap_position) // 43)
        + 29
    )
    hijri_month = (24 * days) // 709
    hijri_day = days - (709 * hijri_month) // 24
    hijri_year = 30 * cycles + leap_position - 30
    return hijri_year, hijri_month, hijri_day


def _hijri_ar(y, mo, d):
    hy, hm, hd = _gregorian_to_hijri(y, mo, d)
    return f"{hd} {HIJRI_MONTHS[hm - 1]} {hy} هـ"


_ZH_DIGITS = "〇一二三四五六七八九"
_ZH_BANK_DIGITS = "零壹贰叁肆伍陆柒捌玖"


def _zh_char_num(n, digits=_ZH_DIGITS, units=("", "十", "百", "千"), myriad="万"):
    if n == 0:
        return digits[0]
    parts = []
    if n >= 10000:
        parts.append(_zh_char_num(n // 10000, digits, units, myriad) + myriad)
        n %= 10000
        if 0 < n < 1000:
            parts.append(digits[0])
    s = ""
    zero_pending = False
    for pos in range(3, -1, -1):
        q, n = divmod(n, 10**pos)
        if q == 0:
            zero_pending = bool(s)
            continue
        if zero_pending:
            s += digits[0]
            zero_pending = False
        s += digits[q] + units[pos]
    parts.append(s)
    out = "".join(parts)
    # idiomatic: 一十X -> 十X at the very front
    return out[1:] if out.startswith("一十") else out


def _zh_bankers(n):
    return _zh_char_num(n, _ZH_BANK_DIGITS, ("", "拾", "佰", "仟"), "万") + "元整"


def _myriad(lang, n):
    if n < 10000:
        return None  # caller falls back to digits (and tags it so)
    high, low = divmod(n, 10000)
    unit = {"zh": "万", "ja": "万", "ko": "만"}[lang]
    if lang == "ko":
        return f"{high}{unit} {low}" if low else f"{high}{unit}"
    return f"{high}{unit}{low if low else ''}"


class LocaleRenderer:
    """Profile-driven surface renderer; validated at load."""

    def __init__(self, lang, profile_path=PROFILE_PATH):
        cfg = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
        if cfg.get("schema_version") != 1:
            raise ValueError(f"unsupported locale-profile schema: {cfg.get('schema_version')}")
        if lang not in cfg["languages"]:
            raise ValueError(f"no locale profile for language {lang!r}")
        self.lang = lang
        merged = dict(cfg["defaults"])
        merged.update(cfg["languages"][lang] or {})
        self.families = {}
        for family, allowed, gated in (
            ("date", DATE_FORMS, DATE_FORM_LANG),
            ("monetary", MONETARY_FORMS, MONETARY_FORM_LANG),
        ):
            shares = merged.get(family)
            if not shares:
                continue
            total = sum(shares.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{lang}/{family} shares sum to {total}, not 1.0")
            for form in shares:
                if form not in allowed:
                    raise ValueError(f"{lang}/{family}: unknown form {form!r}")
                if form in gated and lang not in gated[form]:
                    raise ValueError(f"{lang}/{family}: form {form!r} not supported for {lang}")
            self.families[family] = sorted(shares.items())

    def _draw(self, family, rng):
        x = rng.random()
        acc = 0.0
        for form, share in self.families[family]:
            acc += share
            if x < acc:
                return form
        return self.families[family][-1][0]

    def render_date(self, surface, rng):
        """(new_surface, form) — form 'keep' when kept or unparseable."""
        if "date" not in self.families:
            return surface, "keep"
        stripped = surface.strip()
        time_part = ""
        m = DATETIME_SPLIT.match(stripped)
        if m:
            stripped, time_part = m.group(1), m.group(2)
        parsed = parse_date(stripped)
        if parsed is None:
            return surface, "keep"
        y, mo, d = parsed
        form = self._draw("date", rng)
        if form == "keep":
            return surface, "keep"
        if form == "native":
            out = _native(self.lang, y, mo, d)
        elif form == "local_numeric":
            out = _local_numeric(self.lang, y, mo, d)
        elif form == "weekday_native":
            wd = WEEKDAYS[self.lang][datetime.date(y, mo, d).weekday()]
            body = _native(self.lang, y, mo, d)
            out = f"{body}（{wd}）" if self.lang in ("ja", "zh") else f"{wd}, {body}"
        elif form == "era":
            out = _era_ja(y, mo, d)
        elif form == "minguo":
            if y <= 1911:
                return surface, "keep"
            out = f"民國{y - 1911}年{mo}月{d}日"
        elif form == "hijri":
            out = _hijri_ar(y, mo, d)
        elif form == "full_char":
            ystr = "".join(_ZH_DIGITS[int(c)] for c in str(y))
            out = f"{ystr}年{_zh_char_num(mo)}月{_zh_char_num(d)}日"
        else:  # pragma: no cover
            return surface, "keep"
        if time_part.startswith("T"):
            time_part = " " + time_part[1:]
        return out + time_part, form

    def render_monetary(self, amount, rng):
        """(surface, form) for an integer amount."""
        if "monetary" not in self.families:
            return str(amount), "digits"
        form = self._draw("monetary", rng)
        if form == "digits":
            return str(amount), form
        if form == "latin_sep":
            return f"{amount:,}", form
        if form == "native_sep":
            return _native_sep(self.lang, amount), form
        if form == "myriad":
            surface = _myriad(self.lang, amount)
            if surface is None:
                return str(amount), "digits"
            return surface, form
        if form == "bankers":
            return _zh_bankers(amount), form
        if form == "char_num":
            return _zh_char_num(amount), form
        return str(amount), "digits"  # pragma: no cover
