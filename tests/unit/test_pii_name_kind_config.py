from dataclasses import replace
from pathlib import Path

import pytest

from scripts.pii_name_annotation_qc import NameAnnotationQc, load_config


@pytest.mark.parametrize(
    "language,text,title",
    [
        ("ru", "проф. Иванов", "проф"),
        ("ru", "д-р Петров", "д-р"),
        ("ja", "山田 太郎 医師", "医師"),
        ("ja", "佐藤 花子 様", "様"),
        ("ko", "김민수 교수", "교수"),
        ("ko", "박지영 과장", "과장"),
        ("ko", "이서준 님", "님"),
        ("en", "Lord Palmerston", "Lord"),
        ("en", "Lady Astor", "Lady"),
    ],
)
def test_configured_titles_do_not_enter_name_components(language, text, title):
    config = load_config(
        Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    )
    processor = NameAnnotationQc(replace(config, name_kind_model=None))
    result = processor.analyze(
        row_id="title-rule",
        text=text,
        language=language,
        annotations=[{"start": 0, "end": len(text), "label": "person_name"}],
    )["candidate"]
    components = result["subclass_spans"]
    assert [(s["start"], s["end"]) for s in result["primary_spans"]] == [(0, len(text))]
    start = text.index(title)
    end = start + len(title)
    assert any(s["start"] == start and s["end"] == end and s["value"] == "Q" for s in components)
    assert not any(s["value"] != "Q" and s["start"] < end and start < s["end"] for s in components)


@pytest.mark.parametrize(
    "language,text,expected",
    [
        ("zh", "王小明医生", {("family_name", "王"), ("given_name", "小明"), ("Q", "医生")}),
        ("zh", "欧阳锋教授", {("family_name", "欧"), ("given_name", "阳锋"), ("Q", "教授")}),
        ("ko", "박서준님", {("family_name", "박"), ("given_name", "서준"), ("Q", "님")}),
        ("ko", "최지우 선생님", {("family_name", "최"), ("given_name", "지우"), ("Q", "선생님")}),
        ("ja", "山田 太郎さん", {("family_name", "山田"), ("given_name", "太郎"), ("Q", "さん")}),
        ("es", "Ana Pérez Gómez", {("given_name", "Ana"), ("family_name", "Pérez Gómez")}),
        (
            "es",
            "Dra. Lucía Ortega Vidal",
            {("Q", "Dra"), ("given_name", "Lucía"), ("family_name", "Ortega Vidal")},
        ),
        (
            "ru",
            "Петрова Анна Сергеевна",
            {("family_name", "Петрова"), ("given_name", "Анна"), ("middle_name", "Сергеевна")},
        ),
        (
            "ru",
            "Анна Сергеевна Петрова",
            {("given_name", "Анна"), ("middle_name", "Сергеевна"), ("family_name", "Петрова")},
        ),
        (
            "ru",
            "Смирнов Игорь Павлович",
            {("family_name", "Смирнов"), ("given_name", "Игорь"), ("middle_name", "Павлович")},
        ),
    ],
)
def test_attached_titles_compound_surnames_and_patronymics(language, text, expected):
    config = load_config(
        Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    )
    processor = NameAnnotationQc(replace(config, name_kind_model=None))
    result = processor.analyze(
        row_id="general-rule",
        text=text,
        language=language,
        annotations=[{"start": 0, "end": len(text), "label": "person_name"}],
    )["candidate"]
    assert [(s["start"], s["end"]) for s in result["primary_spans"]] == [(0, len(text))]
    components = {(s["value"], text[s["start"] : s["end"]]) for s in result["subclass_spans"]}
    assert components == expected


def test_latin_honorifics_never_split_latin_atoms():
    config = load_config(
        Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    )
    processor = NameAnnotationQc(replace(config, name_kind_model=None))
    text = "Alexandr Petrov"
    result = processor.analyze(
        row_id="no-split",
        text=text,
        language="ru",
        annotations=[{"start": 0, "end": len(text), "label": "person_name"}],
    )["candidate"]
    assert not any(s["value"] == "Q" for s in result["subclass_spans"])


@pytest.mark.parametrize(
    "language,text,family",
    [
        ("en", "Úna Ní Bhriain", "Ní Bhriain"),
        ("en", "Seán Ó Faoláin", "Ó Faoláin"),
        ("en", "Máire Nic Giolla", "Nic Giolla"),
        ("en", "Bríd Uí Cheallaigh", "Uí Cheallaigh"),
        ("en", "Ewan Mac Gregor", "Mac Gregor"),
    ],
)
def test_gaelic_surname_particles_stay_inside_the_family_name(language, text, family):
    config = load_config(
        Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    )
    processor = NameAnnotationQc(replace(config, name_kind_model=None))
    result = processor.analyze(
        row_id="particle-rule",
        text=text,
        language=language,
        annotations=[{"start": 0, "end": len(text), "label": "person_name"}],
    )["candidate"]
    components = {(s["value"], text[s["start"] : s["end"]]) for s in result["subclass_spans"]}
    given = text.split(" ", 1)[0]
    assert components == {("given_name", given), ("family_name", family)}


def test_language_limits_may_bound_component_spaces(tmp_path: Path):
    import json

    source = Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["languages"]["en"].setdefault("limits", {})["maximum_family_spaces"] = 1
    payload["languages"]["en"]["limits"]["maximum_given_spaces"] = 1
    target = tmp_path / "config.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    profile = NameAnnotationQc(replace(load_config(target), name_kind_model=None)).profile("en")
    assert profile.maximum_family_spaces == 1
    assert profile.maximum_given_spaces == 1
    payload["languages"]["en"]["limits"]["maximum_given_spaces"] = -1
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="maximum_given_spaces"):
        load_config(target)


@pytest.mark.parametrize("weight", [0, 1.5, -1, float("nan")])
def test_middle_presence_is_optional_and_language_specific(tmp_path: Path, weight: float):
    import json

    source = Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    payload = json.loads(source.read_text())
    payload["languages"]["en"]["scoring"]["weights"]["middle_presence"] = weight
    target = tmp_path / "config.json"
    target.write_text(json.dumps(payload))
    if weight < 0 or weight != weight:
        with pytest.raises(ValueError, match="middle_presence"):
            load_config(target)
        return
    processor = NameAnnotationQc(replace(load_config(target), name_kind_model=None))
    assert processor.profile("en-US").weights["middle_presence"] == weight
    assert processor.profile("und").weights.get("middle_presence", 0) == 0
    assert processor.profile("es").weights.get("middle_presence", 0) == 0
    del payload["languages"]["en"]["scoring"]["weights"]["middle_presence"]
    target.write_text(json.dumps(payload))
    processor = NameAnnotationQc(replace(load_config(target), name_kind_model=None))
    assert processor.profile("en").weights.get("middle_presence", 0) == 0


@pytest.mark.parametrize(
    "text",
    ["his brother Mr. B. Bodén", "Ms İ. Melikoff Mr T. Öker"],
)
def test_no_component_spans_an_interior_honorific(text):
    config = load_config(
        Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    )
    processor = NameAnnotationQc(replace(config, name_kind_model=None))
    result = processor.analyze(
        row_id="interior-honorific",
        text=text,
        language="en",
        annotations=[{"start": 0, "end": len(text), "label": "person_name"}],
    )
    components = sorted(result["candidate"]["subclass_spans"], key=lambda s: (s["start"], s["end"]))
    for left, right in zip(components, components[1:]):
        assert right["start"] >= left["end"], (text, components)
    honorifics = [s for s in components if s["value"] == "Q"]
    for q in honorifics:
        assert not any(
            s["value"] != "Q" and s["start"] < q["end"] and q["start"] < s["end"] for s in components
        ), (text, components)


@pytest.mark.parametrize(
    "text,expected",
    [
        # ª and º (U+00AA, U+00BA) are Latin-script letters, common in the Spanish
        # abbreviations Mª and Dª. They must not push a carrier off the Spanish
        # grammar onto the default one, which still has a middle-name slot.
        ("Mª José Vela Muñoz", {("given_name", "Mª José"), ("family_name", "Vela Muñoz")}),
        ("Ana Mª Gómez García", {("given_name", "Ana Mª"), ("family_name", "Gómez García")}),
        ("José María Hernández Pérez", {("given_name", "José María"), ("family_name", "Hernández Pérez")}),
    ],
)
def test_spanish_never_yields_a_middle_name(text, expected):
    config = load_config(
        Path("research/pii/frontier/models/name-components/name-postprocessor-name-kind.json")
    )
    processor = NameAnnotationQc(replace(config, name_kind_model=None))
    result = processor.analyze(
        row_id="es-no-middle",
        text=text,
        language="es",
        annotations=[{"start": 0, "end": len(text), "label": "person_name"}],
    )
    assert result["proposals"][0]["grammar"] == "given_first_no_middle"
    components = {(s["value"], text[s["start"] : s["end"]]) for s in result["candidate"]["subclass_spans"]}
    assert components == expected
