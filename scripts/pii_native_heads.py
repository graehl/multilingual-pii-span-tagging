"""Independent source-native gold batches alongside the primary Ont3 stream."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForTokenClassification


def native_head_coverage(spec: dict[str, Any]) -> str:
    return f"{spec['ontology']}; {spec['coverage']}; boundaries: {spec['boundary_policy']}"


def validate_native_rows(rows: list[dict[str, Any]], spec: dict[str, Any], *, split: str) -> None:
    """Validate source-native spans without projecting another ontology's O."""
    name, types = spec["name"], spec["types"]
    ids = set()
    for row in rows:
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"native head {name}: missing or duplicate source identity")
        ids.add(row_id)
        if row["split"] != split or row["supervision_ontology"] != spec["ontology"]:
            raise ValueError(f"native head {name}/{row_id}: source split or ontology mismatch")
        if row.get("supervision", "complete") != "complete":
            raise ValueError(f"native head {name}/{row_id}: only source-complete supervision is supported")
        if row.get("label_space", "v1") != "v1" or row.get("objective", "tag") != "tag":
            raise ValueError(f"native head {name}/{row_id}: mapped or non-tagging source row")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"native head {name}/{row_id}: empty source text")
        previous_end = 0
        for start, end, entity_type in sorted(row["spans"]):
            if (
                type(start) is not int
                or type(end) is not int
                or not previous_end <= start < end <= len(row["text"])
                or entity_type not in types
            ):
                raise ValueError(f"native head {name}/{row_id}: invalid or overlapping native span")
            previous_end = end
    if not rows:
        raise ValueError(f"native head {name}: empty {split} file")


def load_native_head_config(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "pii-native-heads/v1" or not document.get("heads"):
        raise ValueError("native heads require a nonempty pii-native-heads/v1 configuration")
    specs = []
    rows_by_head = {}
    for entry in document["heads"]:
        spec = dict(entry)
        name = spec["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in rows_by_head:
            raise ValueError(f"invalid or duplicate native head name: {name!r}")
        for field in ("ontology", "coverage", "boundary_policy"):
            if not isinstance(spec[field], str) or not spec[field].strip():
                raise ValueError(f"native head {name}: {field} must be nonempty")
        types = spec["types"]
        if (
            not isinstance(types, list)
            or not types
            or any(not isinstance(value, str) or not value or value == "O" for value in types)
        ):
            raise ValueError(f"native head {name}: types must be nonempty entity names")
        if len(set(types)) != len(types):
            raise ValueError(f"native head {name}: duplicate entity types")
        for field in ("probability", "loss_weight"):
            value = spec[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"native head {name}: {field} must be finite and nonnegative")
        if not 0 < spec["probability"] <= 1:
            raise ValueError(f"native head {name}: probability must be in (0, 1]")
        source = (path.parent / spec["train"]).resolve()
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != spec["train_sha256"]:
            raise ValueError(f"native head {name}: training file hash mismatch")
        rows = [json.loads(line) for line in payload.splitlines()]
        validate_native_rows(rows, spec, split="train")
        spec["train"] = str(source)
        specs.append(spec)
        rows_by_head[name] = rows
    if sum(spec["probability"] for spec in specs) > 1:
        raise ValueError("native head probabilities must sum to at most one")
    return specs, rows_by_head


def validate_native_tokenization(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
    for row in rows:
        encoded = tokenizer(row["text"], truncation=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(encoded["input_ids"]) > max_length:
            raise ValueError(
                f"native source {row['id']}: sentence exceeds --max-len; explicit segmentation required"
            )
        if not any(start < end for start, end in offsets):
            raise ValueError(f"native source {row['id']}: no supervised tokens")
        occupied = set()
        for start, end, _ in row["spans"]:
            positions = {
                i for i, (left, right) in enumerate(offsets) if left < right and left < end and start < right
            }
            if not positions or occupied.intersection(positions):
                raise ValueError(f"native source {row['id']}: missing or conflicting token span")
            occupied.update(positions)


class NativeHeadDataset(Dataset):
    requires_draw_nonce = True

    def __init__(self, primary: Dataset) -> None:
        self.primary = primary
        self.epoch = 0
        weights = primary.sampling_weights
        self.sampling_weights = weights if weights is not None else [1.0] * len(primary)

    def __len__(self) -> int:
        return len(self.primary)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "primary"), name)

    def __getitem__(self, index: tuple[int, bool | None, int]) -> dict[str, Any]:
        if not isinstance(index, tuple) or len(index) != 3:
            raise ValueError("native-head training requires a sampler draw nonce")
        return {**self.primary[index], "_native_draw_nonce": index[2]}


class NativeHeadBatchSampler:
    """Reconstruct each epoch so Trainer can skip completed batches exactly."""

    def __init__(
        self,
        dataset: NativeHeadDataset,
        sampler_factory: Callable[..., Any],
        *,
        seed: int,
        epoch_examples: int,
        batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.sampler_factory = sampler_factory
        self.seed = seed
        self.epoch_examples = epoch_examples
        self.batch_size = batch_size
        self.drop_last = False

    def __len__(self) -> int:
        return math.ceil(self.epoch_examples / self.batch_size)

    def __iter__(self) -> Iterator[list[tuple[int, None, int]]]:
        epoch = self.dataset.epoch
        nonce = epoch * self.epoch_examples
        for batch in self.sampler_factory(seed=self.seed + epoch):
            yield [(int(index), None, nonce + offset) for offset, index in enumerate(batch)]
            nonce += len(batch)

    def summary(self) -> str:
        return "native_epoch_nonce=v1 " + self.sampler_factory(seed=self.seed).summary()


class NativeHeadCollator:
    def __init__(
        self,
        primary_collator: Any,
        tokenizer: Any,
        specs: list[dict[str, Any]],
        datasets: dict[str, Dataset],
        *,
        seed: int,
    ) -> None:
        self.primary_collator = primary_collator
        self.native_collator = DataCollatorForTokenClassification(tokenizer)
        self.specs = specs
        self.datasets = datasets
        self.seed = seed

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        nonces = [feature.get("_native_draw_nonce") for feature in features]
        primary = self.primary_collator(
            [
                {key: value for key, value in feature.items() if key != "_native_draw_nonce"}
                for feature in features
            ]
        )
        if all(nonce is None for nonce in nonces):
            return primary
        if any(nonce is None for nonce in nonces):
            raise ValueError("native-head batch mixes training and evaluation examples")
        # Local RNG makes replay independent of DataLoader worker scheduling.
        rng = random.Random(f"native-head-v1:{self.seed}:{','.join(map(str, nonces))}")
        choice = rng.random()
        for spec in self.specs:
            choice -= spec["probability"]
            if choice >= 0:
                continue
            dataset = self.datasets[spec["name"]]
            indices = [rng.randrange(len(dataset)) for _ in features]
            primary["native_head_batch"] = {
                "head": spec["name"],
                "inputs": self.native_collator([dataset[index] for index in indices]),
                "source_ids": [dataset.rows[index]["id"] for index in indices],
                "draw_nonce": nonces,
            }
            break
        return primary


class NativeHeadLossMixin:
    def __init__(
        self,
        *args: Any,
        native_head_specs: list[dict[str, Any]],
        native_logical_normalization: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise ValueError("native-head training currently requires one process and one device")
        self.native_head_specs = {spec["name"]: spec for spec in native_head_specs}
        self.native_logical_normalization = native_logical_normalization
        if not native_logical_normalization:
            self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None
    ) -> Any:
        inputs = dict(inputs)
        native = inputs.pop("native_head_batch", None)
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if native is not None:
            if not model.training:
                raise ValueError("native-head supervision must not enter primary evaluation")
            spec = self.native_head_specs[native["head"]]
            weight = spec["loss_weight"]
            # Both controls consume the same forward RNG; zero weight creates no
            # gradients, so AdamW cannot decay otherwise unused native parameters.
            with torch.set_grad_enabled(torch.is_grad_enabled() and weight > 0):
                native_loss = model(**native["inputs"], native_head=native["head"]).loss
            scale = 1.0 / num_items_in_batch.physical_batches if self.native_logical_normalization else 1.0
            if weight > 0:
                loss = loss + native_loss * (weight * scale)
            native_labels = native["inputs"]["labels"]
            record = {
                "phase": "train-native-head",
                "step": self.state.global_step,
                "head": native["head"],
                "source_ids": native["source_ids"],
                "draw_nonce": native["draw_nonce"],
                "supervised_tokens": int(native_labels.ne(-100).sum().item()),
                "entity_tokens": int(((native_labels != -100) & (native_labels != 0)).sum().item()),
                "loss": float(native_loss.detach().item()),
                "loss_weight": weight,
                "normalization_scale": scale,
                "weighted_loss": float(native_loss.detach().item()) * weight * scale,
            }
            with (Path(self.args.output_dir) / "native_head_exposure.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return (loss, outputs) if return_outputs else loss
