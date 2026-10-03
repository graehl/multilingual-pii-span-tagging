#!/usr/bin/env python
"""Repair fresh13 gold so dates exercise locale-native conventions.

The fresh13-frontier-final-v2 gold writes every date span in ISO
`YYYY-MM-DD`, so the suite never tests native month-name forms, local
numeric orders, or weekday-adjacent dates in any of its thirteen
languages — the exact convention axis where Korean was hiding a
25%-coverage hole. This produces a versioned sibling dataset per
language (`fresh13-datesfix-v1-<lang>`) with value-preserving rewrites
of parseable date spans:

- 5/8 native month-name form (genitive where the language requires it);
- 1/8 native form with a correct computed weekday inserted *before* the
  span (the weekday stays outside the span so boundary semantics do not
  change);
- 1/8 the language's conventional numeric order (Swedish keeps ISO —
  that genuinely is its convention);
- 1/8 kept as the original ISO.

The choice is a deterministic per-span hash; offsets shift by the
length delta; the underlying date value never changes. Original v2
files are untouched; comparisons using the repaired suite must
re-predict every model on it.
"""

import argparse
import datetime
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DATE_LABELS = {"date", "date_of_birth", "date_time"}
ISO_YMD = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")

MONTHS = {
    "ar": "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split(),
    "cs": "ledna února března dubna května června července srpna září října listopadu prosince".split(),
    "hi": "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(),
    "id": "Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember".split(),
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split(),
    "nl": "januari februari maart april mei juni juli augustus september oktober november december".split(),
    "pl": "stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia".split(),
    "pt": "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro".split(),
    "ru": "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(),
    "sv": "januari februari mars april maj juni juli augusti september oktober november december".split(),
    "tr": "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık".split(),
    "uk": "січня лютого березня квітня травня червня липня серпня вересня жовтня листопада грудня".split(),
    "vi": None,  # vi renders numerically: ngày D tháng M năm Y
}

# Monday-first, matching datetime.weekday()
WEEKDAYS = {
    "ar": "الاثنين الثلاثاء الأربعاء الخميس الجمعة السبت الأحد".split(),
    "cs": "pondělí úterý středa čtvrtek pátek sobota neděle".split(),
    "hi": "सोमवार मंगलवार बुधवार गुरुवार शुक्रवार शनिवार रविवार".split(),
    "id": "Senin Selasa Rabu Kamis Jumat Sabtu Minggu".split(),
    "it": "lunedì martedì mercoledì giovedì venerdì sabato domenica".split(),
    "nl": "maandag dinsdag woensdag donderdag vrijdag zaterdag zondag".split(),
    "pl": "poniedziałek wtorek środa czwartek piątek sobota niedziela".split(),
    "pt": "segunda-feira terça-feira quarta-feira quinta-feira sexta-feira sábado domingo".split(),
    "ru": "понедельник вторник среда четверг пятница суббота воскресенье".split(),
    "sv": "måndag tisdag onsdag torsdag fredag lördag söndag".split(),
    "tr": "Pazartesi Salı Çarşamba Perşembe Cuma Cumartesi Pazar".split(),
    "uk": "понеділок вівторок середа четвер п'ятниця субота неділя".split(),
    "vi": ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"],
}


def native(lang, y, mo, d):
    m = MONTHS[lang]
    if lang == "vi":
        return f"ngày {d} tháng {mo} năm {y}"
    if lang == "cs":
        return f"{d}. {m[mo - 1]} {y}"
    if lang == "pl":
        return f"{d} {m[mo - 1]} {y} r."
    if lang == "ru":
        return f"{d} {m[mo - 1]} {y} г."
    if lang == "uk":
        return f"{d} {m[mo - 1]} {y} р."
    if lang == "pt":
        return f"{d} de {m[mo - 1]} de {y}"
    return f"{d} {m[mo - 1]} {y}"


def numeric_local(lang, y, mo, d):
    if lang == "sv":
        return f"{y}-{mo:02d}-{d:02d}"  # ISO is the Swedish convention
    if lang == "cs":
        return f"{d}. {mo}. {y}"
    if lang in ("pl", "ru", "uk", "tr"):
        return f"{d:02d}.{mo:02d}.{y}"
    if lang == "nl":
        return f"{d:02d}-{mo:02d}-{y}"
    return f"{d:02d}/{mo:02d}/{y}"


def choose_form(row_id, start):
    digest = hashlib.sha256(f"{row_id}:{start}".encode()).digest()
    slot = digest[0] % 8
    if slot < 5:
        return "native"
    return ("weekday", "numeric", "keep")[slot - 5]


def localize_row(row, lang, stats):
    spans = sorted(row.get("spans", []), key=lambda s: (s["start"], s["end"]))
    if any(spans[i]["end"] > spans[i + 1]["start"] for i in range(len(spans) - 1)):
        stats["rows_skipped_overlap"] += 1
        return 0
    text = row["text"]
    out = []
    cursor = 0
    delta = 0
    changed = 0
    for sp in spans:
        start, end = sp["start"], sp["end"]
        out.append(text[cursor:start])
        surface = text[start:end]
        replacement = surface
        prefix = ""
        m = ISO_YMD.match(surface.strip())
        if sp.get("type") in DATE_LABELS and m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                weekday = datetime.date(y, mo, d).weekday()
            except ValueError:
                weekday = None
            if weekday is not None:
                form = choose_form(str(row.get("id", "")), start)
                if form == "native":
                    replacement = native(lang, y, mo, d)
                elif form == "weekday":
                    prefix = WEEKDAYS[lang][weekday] + ", "
                    replacement = native(lang, y, mo, d)
                elif form == "numeric":
                    replacement = numeric_local(lang, y, mo, d)
                stats[f"form:{form}"] += 1
                changed += form != "keep"
            else:
                stats["spans_invalid_date"] += 1
        elif sp.get("type") in DATE_LABELS:
            stats["spans_non_iso"] += 1
        out.append(prefix + replacement)
        cursor = end
        sp["start"] = start + delta + len(prefix)
        delta += len(prefix) + len(replacement) - len(surface)
        sp["end"] = end + delta
    out.append(text[cursor:])
    row["text"] = "".join(out)
    row["spans"] = spans
    return changed


def verify_row(row):
    for sp in row.get("spans", []):
        if not 0 <= sp["start"] < sp["end"] <= len(row["text"]):
            raise AssertionError(f"span {sp} escapes text after localization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", default="untracked/pii-eval/gold")
    parser.add_argument("--source-prefix", default="fresh13-frontier-final-v2")
    parser.add_argument("--output-prefix", default="fresh13-datesfix-v1")
    parser.add_argument("--languages", nargs="+", default=sorted(set(MONTHS) | {"vi"}))
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    total = Counter()
    manifest = {"schema_version": 1, "kind": "fresh13_date_localize", "languages": {}}
    for lang in args.languages:
        src = gold_dir / f"{args.source_prefix}-{lang}.jsonl"
        dst = gold_dir / f"{args.output_prefix}-{lang}.jsonl"
        if dst.exists():
            raise SystemExit(f"{dst} already exists; remove it explicitly first")
        stats = Counter()
        samples = []
        with src.open(encoding="utf-8") as reader, dst.open("w", encoding="utf-8") as writer:
            for line in reader:
                row = json.loads(line)
                changed = localize_row(row, lang, stats)
                verify_row(row)
                if changed and len(samples) < 3:
                    for sp in row["spans"]:
                        if sp.get("type") in DATE_LABELS:
                            samples.append(row["text"][max(0, sp["start"] - 15) : sp["end"] + 5])
                            break
                stats["rows"] += 1
                writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest["languages"][lang] = {"counts": dict(stats), "samples": samples}
        total.update(stats)
        print(lang, dict(stats))
        for s in samples:
            print(f"  {lang} sample: …{s}…")
    manifest["total"] = dict(total)
    out = gold_dir / f"{args.output_prefix}-manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("manifest:", out)


if __name__ == "__main__":
    main()
