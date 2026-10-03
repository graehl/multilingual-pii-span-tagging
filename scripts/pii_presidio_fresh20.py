#!/usr/bin/env python3
"""Run Microsoft Presidio Analyzer over the Fresh20 gold sets, one language at a time.

Runs inside the dedicated `untracked/presidio-venv` (presidio-analyzer from the
data-privacy-stack fork plus spaCy models); it uses only the standard library
and Presidio so it never imports project modules. Routing: a language with an
installed spaCy `*_core_*_lg` model uses it; every other language uses the
multilingual `xx_ent_wiki_sm` model registered under that language code, so
Presidio's language-independent pattern recognizers still run. Output rows
keep Presidio's own entity labels in `label`; the scorer projects them through
the declared source schema. A manifest per language records versions, models,
recognizer inventory and latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path

warnings.filterwarnings("ignore")

SPACY_LG = {
    "de": "de_core_news_lg",
    "es": "es_core_news_lg",
    "fr": "fr_core_news_lg",
    "it": "it_core_news_lg",
    "nl": "nl_core_news_lg",
    "pl": "pl_core_news_lg",
    "pt": "pt_core_news_lg",
    "ru": "ru_core_news_lg",
    "zh": "zh_core_web_lg",
    "ja": "ja_core_news_lg",
    "ko": "ko_core_news_lg",
    "sv": "sv_core_news_lg",
    "uk": "uk_core_news_lg",
    "en": "en_core_web_lg",
}
MULTILINGUAL = "xx_ent_wiki_sm"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_for(language: str) -> tuple[str, str]:
    """(model name, route) where route is 'native-lg' or 'multilingual-fallback'."""
    name = SPACY_LG.get(language)
    if name:
        try:
            import spacy.util

            if spacy.util.is_package(name):
                return name, "native-lg"
        except Exception:  # noqa: BLE001 - a missing package is the only expected failure
            pass
    return MULTILINGUAL, "multilingual-fallback"


def build_engine(language: str, model_name: str, score_threshold: float | None):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": model_name}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    kwargs = {"nlp_engine": nlp_engine, "supported_languages": [language]}
    if score_threshold is not None:
        kwargs["default_score_threshold"] = score_threshold
    return AnalyzerEngine(**kwargs)


def run_dataset(args, dataset: str) -> dict:
    from presidio_analyzer import RecognizerResult

    gold_path = args.gold_dir / f"{dataset}.jsonl"
    rows = read_jsonl(gold_path)
    languages = {row["lang"] for row in rows}
    if len(languages) != 1:
        raise SystemExit(f"{dataset}: expected one language, found {sorted(languages)}")
    language = languages.pop()
    model_name, route = model_for(language)
    engine = build_engine(language, model_name, args.score_threshold)
    entities = sorted(engine.get_supported_entities(language))
    output_path = args.output_dir / f"{args.stem}.{dataset}.jsonl"
    manifest_path = args.output_dir / f"{args.stem}.{dataset}.manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path} / {manifest_path}")
    entity_counts: Counter[str] = Counter()
    recognizer_counts: Counter[str] = Counter()
    predictions = []
    started = time.perf_counter()
    latencies = []
    for row in rows:
        document_started = time.perf_counter()
        results = engine.analyze(text=row["text"], language=language)
        spans = []
        for result in results:
            recognizer = (result.recognition_metadata or {}).get(
                RecognizerResult.RECOGNIZER_NAME_KEY, "unknown"
            )
            entity_counts[result.entity_type] += 1
            recognizer_counts[recognizer] += 1
            spans.append(
                {
                    "start": result.start,
                    "end": result.end,
                    "label": result.entity_type,
                    "score": result.score,
                    "recognizer": recognizer,
                }
            )
        spans.sort(key=lambda s: (s["start"], s["end"], s["label"]))
        latency = time.perf_counter() - document_started
        latencies.append(latency)
        predictions.append({"id": row["id"], "preds": spans, "latency_s": round(latency, 6)})
    elapsed = time.perf_counter() - started
    with output_path.open("w", encoding="utf-8") as stream:
        for item in predictions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    latencies.sort()
    manifest = {
        "schema": "pii-presidio-fresh20-manifest/v1",
        "dataset": dataset,
        "language": language,
        "presidio_analyzer_version": version("presidio-analyzer"),
        "presidio_source": args.presidio_source,
        "spacy_version": version("spacy"),
        "spacy_model": model_name,
        "spacy_model_version": version(model_name),
        "route": route,
        "score_threshold": args.score_threshold,
        "default_score_threshold": engine.default_score_threshold,
        "supported_entities": entities,
        "recognizers": dict(sorted(recognizer_counts.items())),
        "entity_counts": dict(sorted(entity_counts.items())),
        "gold": {"path": str(gold_path), "sha256": sha256(gold_path), "rows": len(rows)},
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "latency_s": {
            "mean": sum(latencies) / len(latencies),
            "p50": latencies[len(latencies) // 2],
            "p90": latencies[int(len(latencies) * 0.9)],
            "total": elapsed,
        },
        "python": sys.version.split()[0],
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return {
        "dataset": dataset,
        "language": language,
        "route": route,
        "model": model_name,
        "rows": len(rows),
        "spans": sum(entity_counts.values()),
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        required=True,
        help="comma-separated dataset names (<name>.jsonl under --gold-dir, one language each)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stem",
        required=True,
        help="prediction file stem, e.g. presidio-analyzer-2.2.364-fork-multilingual-v1",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="AnalyzerEngine default_score_threshold; omit for the library default",
    )
    parser.add_argument(
        "--presidio-source",
        default="git+https://github.com/data-privacy-stack/presidio#subdirectory=presidio-analyzer",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        print(json.dumps(run_dataset(args, dataset)), flush=True)


if __name__ == "__main__":
    main()
