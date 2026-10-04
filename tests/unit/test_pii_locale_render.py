import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pii_locale_render import (
    LocaleRenderer,
    _era_ja,
    _hijri_ar,
    _myriad,
    _zh_bankers,
    _zh_char_num,
    parse_date,
)


def test_parse_date_forms():
    assert parse_date("1970-05-26") == (1970, 5, 26)
    assert parse_date("15/07/2024") == (2024, 7, 15)
    assert parse_date("March 5, 2021") == (2021, 3, 5)
    assert parse_date("5 March 2021") == (2021, 3, 5)
    assert parse_date("2024-13-40") is None
    assert parse_date("R05/07/15") is None


def test_era_boundaries():
    assert _era_ja(2019, 6, 1) == "令和元年6月1日"
    assert _era_ja(2026, 3, 14) == "令和8年3月14日"
    assert _era_ja(1989, 6, 1) == "平成元年6月1日"
    assert _era_ja(1975, 2, 3) == "昭和50年2月3日"


def test_hijri_civil_conversion_preserves_calendar_date():
    assert _hijri_ar(2024, 3, 11) == "1 رمضان 1445 هـ"
    assert _hijri_ar(2020, 9, 10) == "22 محرم 1442 هـ"


def test_zh_char_and_bankers_numbers():
    assert _zh_char_num(0) == "〇"
    assert _zh_char_num(15) == "十五"
    assert _zh_char_num(305) == "三百〇五"
    assert _zh_char_num(3500) == "三千五百"
    assert _zh_char_num(35000) == "三万五千"
    assert _zh_char_num(100305) == "十万〇三百〇五"
    assert _zh_bankers(12345) == "壹万贰仟叁佰肆拾伍元整"


def test_myriad_hybrid():
    assert _myriad("zh", 35000) == "3万5000"
    assert _myriad("ja", 120000) == "12万"
    assert _myriad("ko", 35000) == "3만 5000"
    assert _myriad("zh", 9999) is None  # caller falls back to digits


def test_native_separator_grouping():
    from pii_locale_render import _native_sep

    assert _native_sep("nl", 987654) == "987.654"
    assert _native_sep("ru", 987654) == "987\u00a0654"
    assert _native_sep("en", 987654) == "987,654"
    assert _native_sep("hi", 1234567) == "12,34,567"
    assert _native_sep("ar", 1234567) == "١٬٢٣٤٬٥٦٧"


def test_renderer_ko_distribution_and_value_preservation():
    r = LocaleRenderer("ko")
    rng = random.Random(7)
    forms = {}
    for _ in range(400):
        out, form = r.render_date("1970-05-26", rng)
        forms[form] = forms.get(form, 0) + 1
        if form == "native":
            assert out == "1970년 5월 26일"
        if form == "keep":
            assert out == "1970-05-26"
    assert set(forms) <= {"native", "local_numeric", "keep", "weekday_native"}
    assert forms["native"] > forms["local_numeric"]


def test_renderer_datetime_keeps_time_part():
    r = LocaleRenderer("ko")
    rng = random.Random(3)
    for _ in range(30):
        out, form = r.render_date("1988-09-30 08:20", rng)
        assert out.endswith(" 08:20")


def test_renderer_datetime_normalizes_iso_separator_after_transformation():
    r = LocaleRenderer("ko")
    rng = random.Random(3)
    for _ in range(30):
        out, form = r.render_date("1988-09-30T08:20", rng)
        if form == "keep":
            assert out == "1988-09-30T08:20"
        else:
            assert out.endswith(" 08:20")
            assert "T08:20" not in out


def test_renderer_unparseable_keeps_surface():
    r = LocaleRenderer("ja")
    out, form = r.render_date("around March", random.Random(0))
    assert (out, form) == ("around March", "keep")


def test_profile_validation_rejects_bad_form():
    import tempfile

    bad = """
schema_version: 1
defaults:
  date: {era: 1.0}
languages:
  ko: {}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(bad)
    with pytest.raises(ValueError, match="not supported"):
        LocaleRenderer("ko", f.name)


def test_all_profile_languages_load_and_render():
    import yaml
    from pii_locale_render import PROFILE_PATH

    cfg = yaml.safe_load(Path(PROFILE_PATH).read_text())
    rng = random.Random(1)
    for lang in cfg["languages"]:
        r = LocaleRenderer(lang)
        for _ in range(50):
            out, form = r.render_date("2003-11-07", rng)
            assert out
            amt, mform = r.render_monetary(35000, rng)
            assert amt
