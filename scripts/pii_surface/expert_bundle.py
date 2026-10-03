"""Train and compare independently parameterized surface-realizer experts."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.pii_language_policy import validate_core_language_support

from .context_generator import (
    SCHEMA as MODEL_SCHEMA,
)
from .context_generator import (
    LoadedContextInputs,
    LoadedContextSurfaceGenerator,
    ModelDimensions,
    file_sha256,
    stable_seed,
    train_context_generator,
    write_headline,
)
from .mix_policy import surface_expert_group

SCHEMA = "pii-surface-context-expert-bundle-v1"


@dataclass(frozen=True)
class ExpertCandidate:
    kind: str
    language: str
    name: str
    tags: tuple[str, ...]
    train_examples: int
    audit_examples: int

    @property
    def candidate_id(self) -> str:
        return f"{self.kind}:{self.language}:{self.name}"


def expert_candidates(
    inputs: LoadedContextInputs,
    *,
    minimum_train_examples: int,
    minimum_audit_examples: int,
) -> list[ExpertCandidate]:
    if minimum_train_examples <= 0 or minimum_audit_examples <= 0:
        raise ValueError("expert support floors must be positive")
    train_counts = Counter(
        (example.language, example.tag) for example in inputs.examples if example.split == "train"
    )
    audit_counts = Counter(
        (example.language, example.tag) for example in inputs.examples if example.split == "audit"
    )
    candidates = []
    group_tags: dict[tuple[str, str], set[str]] = defaultdict(set)
    for language, tag in train_counts:
        group = surface_expert_group(tag)
        if group is not None:
            group_tags[language, group].add(tag)
    for (language, group), tags in sorted(group_tags.items()):
        train = sum(train_counts[language, tag] for tag in tags)
        audit = sum(audit_counts[language, tag] for tag in tags)
        if train >= minimum_train_examples and audit >= minimum_audit_examples:
            candidates.append(
                ExpertCandidate(
                    kind="ontology_group",
                    language=language,
                    name=group,
                    tags=tuple(sorted(tags)),
                    train_examples=train,
                    audit_examples=audit,
                )
            )
    for (language, tag), train in sorted(train_counts.items()):
        audit = audit_counts[language, tag]
        if train >= minimum_train_examples and audit >= minimum_audit_examples:
            candidates.append(
                ExpertCandidate(
                    kind="exact_pair",
                    language=language,
                    name=tag,
                    tags=(tag,),
                    train_examples=train,
                    audit_examples=audit,
                )
            )
    return sorted(candidates, key=lambda item: (item.kind, item.language, item.name))


def metric_by_pair(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    groups = report["metrics"]["audit_with_context"]["groups"]
    return {(row["language"], row["tag"]): dict(row) for row in groups}


def support_by_pair(inputs: LoadedContextInputs) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    characters: dict[tuple[str, str], int] = Counter()
    for example in inputs.examples:
        counts[example.language, example.tag][example.split] += 1
        if example.split == "audit":
            characters[example.language, example.tag] += len(example.target) + 1
    return {
        pair: {
            "train_examples": values["train"],
            "audit_examples": values["audit"],
            "audit_characters_including_eos": characters[pair],
        }
        for pair, values in sorted(counts.items())
    }


def candidate_relative_path(candidate: ExpertCandidate, context_chars: int) -> Path:
    name = candidate.name.casefold().replace("_", "-")
    return Path("models") / candidate.kind / candidate.language / f"{name}-c{context_chars}"


def candidate_model_version(bundle_version: str, candidate: ExpertCandidate, context_chars: int) -> str:
    name = candidate.name.casefold().replace("_", "-")
    kind = "pair" if candidate.kind == "exact_pair" else "group"
    return f"{bundle_version}-{kind}-{candidate.language}-{name}-c{context_chars}"


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def language_route_receipts(
    candidates: Sequence[ExpertCandidate],
    *,
    output: Path,
    bundle_version: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Freeze and validate the aggregate language-component training budget."""
    budgets = Counter()
    for candidate in candidates:
        budgets[candidate.language] += candidate.train_examples
    total = sum(budgets.values())
    if total <= 0:
        raise ValueError("expert candidate plan has no training-example budget")
    manifest = {
        "schema_version": 1,
        "version": f"{bundle_version}-language-routes-v1",
        "basis": (
            "candidate train-example exposures before the common epoch multiplier; "
            "alternative pair/group candidates are grouped into one component per language"
        ),
        "routes": {
            language: {
                "languages": [language],
                "joint_sampling_mass": budget / total,
                "candidate_train_examples_before_epoch_multiplier": budget,
            }
            for language, budget in sorted(budgets.items())
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
        for language in sorted(budgets)
    }
    required = set(next(iter(receipts.values()))["language_round"]["core_languages"])
    if set(receipts) != required:
        raise ValueError(
            "expert candidate language routes must exactly cover the core inventory: "
            f"candidate_languages={sorted(receipts)} core_languages={sorted(required)}"
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


def validated_control(control: Path, inputs: LoadedContextInputs) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = control / "config.json"
    report_path = control / "report.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if config.get("schema") != MODEL_SCHEMA or report.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"{control}: not a context-character generator")
    if config.get("weights_sha256") != file_sha256(control / "model.pt"):
        raise ValueError(f"{control}: control weights drifted")
    if report.get("input") != inputs.input_receipt:
        raise ValueError(f"{control}: control and expert corpus receipts differ")
    return config, report


def _candidate_record(
    candidate: ExpertCandidate,
    *,
    relative_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "relative_path": str(relative_path),
        "model_version": report["model_version"],
        "report_sha256": file_sha256(Path(report["artifacts"]["config"]["path"]).parent / "report.json"),
        "config_sha256": report["artifacts"]["config"]["sha256"],
        "weights_sha256": report["artifacts"]["weights"]["sha256"],
        "audit_mean_nll": report["metrics"]["audit_with_context"]["mean_nll"],
        "audit_perplexity": report["metrics"]["audit_with_context"]["perplexity"],
        "blank_context_perplexity": report["metrics"]["audit_blank_context"]["perplexity"],
    }


def pair_comparisons(
    inputs: LoadedContextInputs,
    *,
    control_report: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    support = support_by_pair(inputs)
    choices: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair, metric in metric_by_pair(control_report).items():
        choices[pair].append(
            {
                "route": "shared_control",
                "candidate_id": None,
                "relative_path": None,
                **metric,
            }
        )
    for record in candidate_records:
        report = json.loads((output / record["relative_path"] / "report.json").read_text(encoding="utf-8"))
        for pair, metric in metric_by_pair(report).items():
            choices[pair].append(
                {
                    "route": record["kind"],
                    "candidate_id": record["candidate_id"],
                    "relative_path": record["relative_path"],
                    **metric,
                }
            )
    comparisons = []
    for pair, pair_support in support.items():
        if pair not in choices:
            continue
        ranked = sorted(
            choices[pair],
            key=lambda item: (item["mean_nll"], item["route"], item["candidate_id"] or ""),
        )
        shared = next(item for item in ranked if item["route"] == "shared_control")
        winner = ranked[0]
        comparisons.append(
            {
                "language": pair[0],
                "tag": pair[1],
                "ontology_group": surface_expert_group(pair[1]),
                **pair_support,
                "candidates": ranked,
                "intrinsic_winner": winner["route"],
                "intrinsic_winner_candidate_id": winner["candidate_id"],
                "shared_minus_winner_mean_nll": shared["mean_nll"] - winner["mean_nll"],
                "training_admitted": False,
            }
        )
    return comparisons


def train_expert_bundle(
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
    minimum_train_examples: int = 100,
    minimum_audit_examples: int = 10,
) -> dict[str, Any]:
    if not bundle_version:
        raise ValueError("bundle_version must be nonempty")
    control_config, control_report = validated_control(control, inputs)
    if context_chars != int(control_config["context_chars"]):
        raise ValueError("the first expert comparison must match the control context span")
    if max_target_chars != int(control_config["max_target_chars"]):
        raise ValueError("expert and control maximum target lengths must match")
    if asdict(dimensions) != control_config["dimensions"]:
        raise ValueError("expert and control model dimensions must match")
    candidates = expert_candidates(
        inputs,
        minimum_train_examples=minimum_train_examples,
        minimum_audit_examples=minimum_audit_examples,
    )
    output.mkdir(parents=True, exist_ok=True)
    language_route_manifest, language_receipts = language_route_receipts(
        candidates,
        output=output,
        bundle_version=bundle_version,
    )
    progress_path = output / "progress.json"
    candidate_records = []
    vocabulary_receipt = {
        "kind": "shared-control-vocabulary",
        "control_path": str(control),
        "control_config_sha256": file_sha256(control / "config.json"),
    }
    for index, candidate in enumerate(candidates, 1):
        relative_path = candidate_relative_path(candidate, context_chars)
        candidate_path = output / relative_path
        model_version = candidate_model_version(bundle_version, candidate, context_chars)
        write_headline(
            f"surface experts {index}/{len(candidates)}: {candidate.candidate_id} "
            f"({candidate.train_examples} train / {candidate.audit_examples} audit)"
        )
        report = train_context_generator(
            agreement_path=None,
            agreement_report_path=None,
            output=candidate_path,
            model_version=model_version,
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
            seed=stable_seed(seed, candidate.candidate_id) % (2**31),
            device=device,
            decoding_temperature=decoding_temperature,
            decoding_top_k=decoding_top_k,
            loaded_inputs=inputs,
            include_languages=(candidate.language,),
            include_tags=candidate.tags,
            fixed_vocabulary=control_config["vocabulary"],
            scope={
                "kind": candidate.kind,
                "language": candidate.language,
                "name": candidate.name,
                "tags": list(candidate.tags),
                "language_support": language_receipts[candidate.language],
            },
            vocabulary_receipt=vocabulary_receipt,
        )
        candidate_records.append(_candidate_record(candidate, relative_path=relative_path, report=report))
        atomic_write_json(
            progress_path,
            {
                "schema": SCHEMA,
                "status": "training",
                "bundle_version": bundle_version,
                "completed_candidates": len(candidate_records),
                "total_candidates": len(candidates),
                "last_candidate": candidate.candidate_id,
                "candidates": candidate_records,
            },
        )
        print(
            f"SURFACE-EXPERT candidate={index}/{len(candidates)} id={candidate.candidate_id} "
            f"audit_ppl={report['metrics']['audit_with_context']['perplexity']:.4f}",
            flush=True,
        )
    comparisons = pair_comparisons(
        inputs,
        control_report=control_report,
        candidate_records=candidate_records,
        output=output,
    )
    wins = Counter(row["intrinsic_winner"] for row in comparisons)
    report = {
        "schema": SCHEMA,
        "status": "intrinsic_route_candidates_not_training_admitted",
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
            "minimum_train_examples": minimum_train_examples,
            "minimum_audit_examples": minimum_audit_examples,
            "vocabulary": vocabulary_receipt,
        },
        "language_support": {
            "manifest": language_route_manifest,
            "receipts": language_receipts,
            "aggregate_completion": "every core-language component completed at least one candidate",
        },
        "candidate_count": len(candidate_records),
        "candidate_counts_by_kind": dict(sorted(Counter(row["kind"] for row in candidate_records).items())),
        "candidates": candidate_records,
        "pair_comparisons": comparisons,
        "intrinsic_wins_by_route": dict(sorted(wins.items())),
        "contract": {
            "parameterization": "every candidate path owns a wholly independent parameter set",
            "group_conditioning": "ontology-group experts receive the exact canonical tag at every decoder step",
            "comparison": "all candidates use the control vocabulary, architecture, context span, and source-document-heldout audit rows",
            "admission": "intrinsic winners are diagnostic routes only; qualitative review and a matched downstream tagger treatment are still required",
        },
    }
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        output / "routes.json",
        {
            "schema": SCHEMA,
            "status": "not_training_admitted",
            "bundle_version": bundle_version,
            "control_relative_path": None,
            "control_path": str(control),
            "routes": [
                {
                    "language": row["language"],
                    "tag": row["tag"],
                    "route": row["intrinsic_winner"],
                    "relative_path": next(
                        item["relative_path"]
                        for item in row["candidates"]
                        if item["route"] == row["intrinsic_winner"]
                    ),
                    "training_admitted": False,
                }
                for row in comparisons
            ],
        },
    )
    write_headline(
        f"surface expert bundle complete: {len(candidate_records)} candidates; "
        + ", ".join(f"{route}={count}" for route, count in sorted(wins.items()))
    )
    return report


class LoadedContextSurfaceBundle:
    """Lazy pair router over independent context-character checkpoints."""

    def __init__(
        self,
        path: Path,
        *,
        device: str = "cpu",
        require_training_admitted: bool = True,
    ):
        self.path = Path(path)
        self.routes_path = self.path / "routes.json"
        raw = json.loads(self.routes_path.read_text(encoding="utf-8"))
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"{self.routes_path}: expected schema {SCHEMA!r}")
        if require_training_admitted and raw.get("status") != "training_admitted":
            raise ValueError(f"{self.routes_path}: bundle is not admitted for training")
        self.device = device
        self.control_path = Path(raw["control_path"])
        self.routes = {(row["language"], row["tag"]): row for row in raw["routes"]}
        self._models: dict[str, LoadedContextSurfaceGenerator] = {}

    def _model(self, route: Mapping[str, Any]) -> LoadedContextSurfaceGenerator:
        relative_path = route.get("relative_path")
        path = self.control_path if relative_path is None else self.path / str(relative_path)
        key = str(path)
        if key not in self._models:
            self._models[key] = LoadedContextSurfaceGenerator(path, device=self.device)
        return self._models[key]

    def generate(
        self,
        language: str,
        tag: str,
        left: str,
        right: str,
        *,
        seed: int,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> tuple[str, str] | None:
        route = self.routes.get((language, tag))
        if route is None:
            return None
        return self._model(route).generate(
            language,
            tag,
            left,
            right,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "routes_sha256": file_sha256(self.routes_path),
            "routed_pairs": len(self.routes),
        }


def perplexity(mean_nll: float) -> float:
    return math.exp(min(20.0, mean_nll))


def routed_pairs(rows: Iterable[Mapping[str, Any]], route: str) -> int:
    return sum(row.get("intrinsic_winner") == route for row in rows)
