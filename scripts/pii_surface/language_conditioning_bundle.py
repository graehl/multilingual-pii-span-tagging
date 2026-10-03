"""Train language-local surface generators with and without ontology-cut conditions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.pii_language_policy import validate_core_language_support

from .context_generator import (
    TAG_CONDITIONING_EXACT,
    TAG_CONDITIONING_P20_P9,
    LoadedContextInputs,
    LoadedContextSurfaceGenerator,
    ModelDimensions,
    file_sha256,
    stable_seed,
    train_context_generator,
    write_headline,
)
from .expert_bundle import atomic_write_json, metric_by_pair, validated_control

SCHEMA = "pii-surface-language-tag-conditioning-bundle-v1"
CONDITION_MODES = (TAG_CONDITIONING_EXACT, TAG_CONDITIONING_P20_P9)
INTRINSIC_WINNER = "intrinsic-winner"
ROUTE_MODES = (*CONDITION_MODES, INTRINSIC_WINNER)
MODE_SLUGS = {
    TAG_CONDITIONING_EXACT: "exact",
    TAG_CONDITIONING_P20_P9: "exact-p20-p9-additive",
}


class LoadedLanguageConditioningSurfaceGenerator:
    """Route one fitted language-local generator per declared language."""

    def __init__(
        self,
        path: Path,
        *,
        route_mode: str = INTRINSIC_WINNER,
        device: str = "cpu",
    ):
        if route_mode not in ROUTE_MODES:
            raise ValueError(f"unknown language-conditioning route mode {route_mode!r}")
        self.path = Path(path)
        self.report_path = self.path / "report.json"
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if report.get("schema") != SCHEMA:
            raise ValueError(f"{self.report_path}: expected schema {SCHEMA!r}")
        comparisons = report.get("language_comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError(f"{self.report_path}: missing language comparisons")
        self.report = report
        self.route_mode = route_mode
        self.context_chars = int(report["configuration"]["context_chars_per_side"])
        self.max_target_chars = int(report["configuration"]["max_target_chars"])
        self.routes: dict[str, LoadedContextSurfaceGenerator] = {}
        self.route_receipts = []
        for comparison in comparisons:
            language = str(comparison["language"])
            if language in self.routes:
                raise ValueError(f"{self.report_path}: duplicate language route {language!r}")
            selected_mode = (
                str(comparison["intrinsic_winner"]) if route_mode == INTRINSIC_WINNER else route_mode
            )
            if selected_mode not in CONDITION_MODES:
                raise ValueError(
                    f"{self.report_path}: invalid selected tag-conditioning mode {selected_mode!r}"
                )
            record_key = "exact" if selected_mode == TAG_CONDITIONING_EXACT else "hierarchical"
            record = comparison[record_key]
            model_path = self.path / str(record["relative_path"])
            generator = LoadedContextSurfaceGenerator(model_path, device=device)
            if generator.languages[1:] != (language,):
                raise ValueError(f"{model_path}: expected only language {language!r}")
            if generator.tag_conditioning.mode != selected_mode:
                raise ValueError(
                    f"{model_path}: tag-conditioning mode differs from selected route {selected_mode!r}"
                )
            config_hash = file_sha256(generator.config_path)
            weights_hash = file_sha256(generator.weights_path)
            if config_hash != record["config_sha256"] or weights_hash != record["weights_sha256"]:
                raise ValueError(f"{model_path}: route hashes differ from bundle report")
            self.routes[language] = generator
            self.route_receipts.append(
                {
                    "language": language,
                    "tag_conditioning": selected_mode,
                    "relative_path": str(record["relative_path"]),
                    "model_version": generator.config["model_version"],
                    "config_sha256": config_hash,
                    "weights_sha256": weights_hash,
                }
            )

    def generate(
        self,
        language: str,
        tag: str,
        left: str,
        right: str,
        *,
        seed: int,
    ) -> tuple[str, str] | None:
        generator = self.routes.get(language)
        if generator is None:
            return None
        return generator.generate(language, tag, left, right, seed=seed)

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bundle_version": self.report["bundle_version"],
            "report_sha256": file_sha256(self.report_path),
            "route_mode": self.route_mode,
            "context_chars": self.context_chars,
            "routes": self.route_receipts,
        }


def language_support_receipts(
    inputs: LoadedContextInputs,
    *,
    output: Path,
    bundle_version: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    train_counts = Counter(example.language for example in inputs.examples if example.split == "train")
    total = sum(train_counts.values())
    if total <= 0:
        raise ValueError("language-local conditioning bundle has no training examples")
    manifest = {
        "schema_version": 1,
        "version": f"{bundle_version}-language-routes-v1",
        "basis": (
            "one independently trained component per language; joint mass is the component's "
            "source training-example exposure before the common epoch multiplier"
        ),
        "routes": {
            language: {
                "languages": [language],
                "joint_sampling_mass": count / total,
                "train_examples_before_epoch_multiplier": count,
            }
            for language, count in sorted(train_counts.items())
        },
    }
    manifest_path = output / "language-route-manifest.json"
    atomic_write_json(manifest_path, manifest)
    receipts = {
        language: validate_core_language_support(
            {language: 1.0},
            split_manifest=manifest_path,
            split_component=language,
        )
        for language in sorted(train_counts)
    }
    required = set(next(iter(receipts.values()))["language_round"]["core_languages"])
    if set(receipts) != required:
        raise ValueError(
            "language-local components must exactly cover the core inventory: "
            f"components={sorted(receipts)} core_languages={sorted(required)}"
        )
    return (
        {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "version": manifest["version"],
            "basis": manifest["basis"],
        },
        receipts,
    )


def _candidate_record(
    *,
    language: str,
    mode: str,
    relative_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "language": language,
        "tag_conditioning": mode,
        "relative_path": str(relative_path),
        "model_version": report["model_version"],
        "train_examples": report["data"]["train_examples"],
        "audit_examples": report["data"]["audit_examples"],
        "audit_supported_fraction": report["data"]["audit_examples_supported_for_generation_fraction"],
        "audit_mean_nll": report["metrics"]["audit_with_context"]["mean_nll"],
        "audit_perplexity": report["metrics"]["audit_with_context"]["perplexity"],
        "blank_context_perplexity": report["metrics"]["audit_blank_context"]["perplexity"],
        "report_sha256": file_sha256(Path(report["artifacts"]["config"]["path"]).parent / "report.json"),
        "config_sha256": report["artifacts"]["config"]["sha256"],
        "weights_sha256": report["artifacts"]["weights"]["sha256"],
    }


def language_comparisons(
    records: Sequence[Mapping[str, Any]],
    *,
    output: Path,
) -> list[dict[str, Any]]:
    by_language: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_language.setdefault(str(record["language"]), {})[str(record["tag_conditioning"])] = record
    comparisons = []
    for language, candidates in sorted(by_language.items()):
        if set(candidates) != set(CONDITION_MODES):
            raise ValueError(f"{language}: incomplete tag-conditioning comparison")
        exact = candidates[TAG_CONDITIONING_EXACT]
        hierarchical = candidates[TAG_CONDITIONING_P20_P9]
        exact_report = json.loads(
            (output / str(exact["relative_path"]) / "report.json").read_text(encoding="utf-8")
        )
        hierarchical_report = json.loads(
            (output / str(hierarchical["relative_path"]) / "report.json").read_text(encoding="utf-8")
        )
        exact_pairs = metric_by_pair(exact_report)
        hierarchical_pairs = metric_by_pair(hierarchical_report)
        if set(exact_pairs) != set(hierarchical_pairs):
            raise ValueError(f"{language}: exact and hierarchical audit pair sets differ")
        pair_rows = []
        for pair in sorted(exact_pairs):
            exact_metric = exact_pairs[pair]
            hierarchical_metric = hierarchical_pairs[pair]
            pair_rows.append(
                {
                    "tag": pair[1],
                    "characters": exact_metric["characters"],
                    "exact_mean_nll": exact_metric["mean_nll"],
                    "hierarchical_mean_nll": hierarchical_metric["mean_nll"],
                    "exact_minus_hierarchical_mean_nll": (
                        exact_metric["mean_nll"] - hierarchical_metric["mean_nll"]
                    ),
                }
            )
        winner = min(
            (exact, hierarchical),
            key=lambda item: (float(item["audit_mean_nll"]), str(item["tag_conditioning"])),
        )
        comparisons.append(
            {
                "language": language,
                "train_examples": exact["train_examples"],
                "audit_examples": exact["audit_examples"],
                "audit_supported_fraction": exact["audit_supported_fraction"],
                "exact": dict(exact),
                "hierarchical": dict(hierarchical),
                "exact_minus_hierarchical_mean_nll": (
                    float(exact["audit_mean_nll"]) - float(hierarchical["audit_mean_nll"])
                ),
                "intrinsic_winner": winner["tag_conditioning"],
                "pair_comparisons": pair_rows,
                "training_admitted": False,
            }
        )
    return comparisons


def write_qualitative_comparison(
    records: Sequence[Mapping[str, Any]],
    *,
    output: Path,
) -> tuple[Path, int]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        mode = str(record["tag_conditioning"])
        samples_path = output / str(record["relative_path"]) / "samples.jsonl"
        with samples_path.open(encoding="utf-8") as source:
            for line in source:
                sample = json.loads(line)
                key = str(sample["language"]), str(sample["candidate_id"])
                row = rows.setdefault(
                    key,
                    {
                        "language": sample["language"],
                        "tag": sample["tag"],
                        "candidate_id": sample["candidate_id"],
                        "left": sample["left"],
                        "target": sample["target"],
                        "right": sample["right"],
                        "target_seen_in_train": sample["target_seen_in_train"],
                    },
                )
                row[MODE_SLUGS[mode]] = sample["generated"]
    path = output / "qualitative-comparison.jsonl"
    with path.open("w", encoding="utf-8") as target:
        for key in sorted(rows):
            target.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    return path, len(rows)


def train_language_conditioning_bundle(
    *,
    inputs: LoadedContextInputs,
    control: Path,
    output: Path,
    bundle_version: str,
    context_chars: int = 20,
    max_target_chars: int = 64,
    dimensions: ModelDimensions = ModelDimensions(),
    batch_size: int = 64,
    bucket_width: int = 16,
    epochs: int = 20,
    learning_rate: float = 2e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    gradient_clip: float = 1.0,
    seed: int = 20260815,
    device: str = "auto",
    decoding_temperature: float = 0.8,
    decoding_top_k: int = 20,
) -> dict[str, Any]:
    if not bundle_version:
        raise ValueError("bundle_version must be nonempty")
    control_config, control_report = validated_control(control, inputs)
    if context_chars != int(control_config["context_chars"]):
        raise ValueError("language-local comparison must match the shared control context span")
    if max_target_chars != int(control_config["max_target_chars"]):
        raise ValueError("language-local comparison and control maximum target lengths must match")
    if asdict(dimensions) != control_config["dimensions"]:
        raise ValueError("language-local comparison and control model dimensions must match")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite language-conditioning bundle {output}")
    output.mkdir(parents=True)
    route_manifest, route_receipts = language_support_receipts(
        inputs,
        output=output,
        bundle_version=bundle_version,
    )
    languages = sorted(route_receipts)
    vocabulary_receipt = {
        "kind": "shared-control-vocabulary",
        "control_path": str(control),
        "control_config_sha256": file_sha256(control / "config.json"),
    }
    progress_path = output / "progress.json"
    records = []
    total_candidates = len(languages) * len(CONDITION_MODES)
    for language in languages:
        language_seed = stable_seed(seed, "language-local", language) % (2**31)
        for mode in CONDITION_MODES:
            relative_path = Path("models") / MODE_SLUGS[mode] / language
            model_path = output / relative_path
            index = len(records) + 1
            write_headline(f"language-local tag conditioning {index}/{total_candidates}: {language} {mode}")
            candidate_report = train_context_generator(
                agreement_path=None,
                agreement_report_path=None,
                output=model_path,
                model_version=f"{bundle_version}-{language}-{MODE_SLUGS[mode]}",
                context_chars=context_chars,
                max_target_chars=max_target_chars,
                dimensions=dimensions,
                batch_size=batch_size,
                bucket_width=bucket_width,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                gradient_clip=gradient_clip,
                seed=language_seed,
                device=device,
                decoding_temperature=decoding_temperature,
                decoding_top_k=decoding_top_k,
                tag_conditioning_mode=mode,
                loaded_inputs=inputs,
                include_languages=(language,),
                fixed_vocabulary=control_config["vocabulary"],
                scope={
                    "kind": "language_local",
                    "language": language,
                    "tag_conditioning": mode,
                    "language_support": route_receipts[language],
                },
                vocabulary_receipt=vocabulary_receipt,
            )
            records.append(
                _candidate_record(
                    language=language,
                    mode=mode,
                    relative_path=relative_path,
                    report=candidate_report,
                )
            )
            atomic_write_json(
                progress_path,
                {
                    "schema": SCHEMA,
                    "status": "training",
                    "bundle_version": bundle_version,
                    "completed_candidates": len(records),
                    "total_candidates": total_candidates,
                    "last_candidate": {"language": language, "tag_conditioning": mode},
                    "candidates": records,
                },
            )
            print(
                f"SURFACE-LANGUAGE-CONDITIONING candidate={index}/{total_candidates} "
                f"language={language} mode={mode} "
                f"audit_ppl={candidate_report['metrics']['audit_with_context']['perplexity']:.4f}",
                flush=True,
            )
    comparisons = language_comparisons(records, output=output)
    qualitative_path, qualitative_rows = write_qualitative_comparison(records, output=output)
    wins = Counter(row["intrinsic_winner"] for row in comparisons)
    report = {
        "schema": SCHEMA,
        "status": "intrinsic_candidates_not_training_admitted",
        "bundle_version": bundle_version,
        "control": {
            "path": str(control),
            "model_version": control_report["model_version"],
            "config_sha256": file_sha256(control / "config.json"),
            "report_sha256": file_sha256(control / "report.json"),
            "weights_sha256": file_sha256(control / "model.pt"),
        },
        "input": inputs.input_receipt,
        "configuration": {
            "condition_modes": list(CONDITION_MODES),
            "hierarchical_condition": (
                "freely learned exact-tag embedding plus additive redaction_20_v1 and "
                "redaction_9_v1 embeddings"
            ),
            "context_chars_per_side": context_chars,
            "max_target_chars": max_target_chars,
            "dimensions": asdict(dimensions),
            "batch_size": batch_size,
            "bucket_width": bucket_width,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "gradient_clip": gradient_clip,
            "seed": seed,
            "device": device,
            "decoding": {"temperature": decoding_temperature, "top_k": decoding_top_k},
            "vocabulary": vocabulary_receipt,
        },
        "language_support": {
            "manifest": route_manifest,
            "receipts": route_receipts,
            "aggregate_completion": (
                "each conditioning mode completed one independently parameterized component "
                "for every core language"
            ),
        },
        "candidate_count": len(records),
        "candidates": records,
        "language_comparisons": comparisons,
        "intrinsic_wins_by_condition": dict(sorted(wins.items())),
        "artifacts": {
            "qualitative_comparison": {
                "path": str(qualitative_path),
                "sha256": file_sha256(qualitative_path),
                "rows": qualitative_rows,
            }
        },
        "contract": {
            "parameterization": "each language owns an independent parameter set",
            "matched_control": (
                "within a language, both arms use identical examples, source-document-heldout "
                "audit rows, vocabulary, dimensions, context span, seed, data order, and schedule"
            ),
            "shared_initialization": (
                "hierarchical-only embedding tables are initialized after all shared modules, "
                "so the common parameter tensors have identical initial values"
            ),
            "admission": (
                "intrinsic winners remain diagnostic until qualitative review and a matched "
                "downstream tagger treatment improve"
            ),
        },
    }
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        output / "routes.json",
        {
            "schema": SCHEMA,
            "status": "not_training_admitted",
            "bundle_version": bundle_version,
            "routes": [
                {
                    "language": row["language"],
                    "tag_conditioning": row["intrinsic_winner"],
                    "relative_path": row[
                        "exact" if row["intrinsic_winner"] == TAG_CONDITIONING_EXACT else "hierarchical"
                    ]["relative_path"],
                    "training_admitted": False,
                }
                for row in comparisons
            ],
        },
    )
    write_headline(
        "language-local tag conditioning complete: "
        + ", ".join(f"{mode}={count}" for mode, count in sorted(wins.items()))
    )
    return report
