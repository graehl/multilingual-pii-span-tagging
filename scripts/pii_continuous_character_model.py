#!/usr/bin/env python3
"""Joint XLM-R and continuous-character token classifier."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from scripts.pii_character_projection import (
    CharacterPairProjection,
    load_character_projection,
)
from scripts.pii_continuous_character_cnn import (
    ContinuousCharacterCnn,
    pool_character_states,
)

LEGACY_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE = "continuous_character_concat"
CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE = "continuous_character_split_logits"
CONTINUOUS_CHARACTER_DECODER_NAME = "continuous-character-split-logits"
SUPPORTED_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURES = {
    LEGACY_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE,
    CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE,
}
CHARACTER_PROJECTION_FILENAME = "character_projection.json"


def configured_parameter_dtype(config: object, name: str) -> torch.dtype | None:
    """Resolve an explicitly recorded module dtype without accepting aliases."""
    value = getattr(config, name, None)
    if value is None:
        return None
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return supported[str(value)]
    except KeyError as error:
        raise ValueError(f"unsupported {name} value {value!r}") from error


def uses_continuous_character_head(config: object) -> bool:
    """Return whether a model config names a loadable character head."""
    return getattr(config, "pii_head_architecture", None) in SUPPORTED_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURES


def rescale_imported_character_readout(
    model: ContinuousCharacterForTokenClassification,
    target_scale: float,
) -> float:
    """Set an imported tagger readout to an absolute inference-time scale."""
    if getattr(model.config, "pii_character_logit_initialization", None) != "tagger":
        raise ValueError("character readout scaling requires imported tagger logits")
    if getattr(model.config, "pii_character_runtime_logit_scale", None) is not None:
        raise RuntimeError("character readout was already rescaled for inference")
    stored_scale = float(getattr(model.config, "pii_character_logit_scale", 0.0))
    target_scale = float(target_scale)
    if stored_scale <= 0 or target_scale <= 0:
        raise ValueError("stored and target character logit scales must be positive")
    multiplier = target_scale / stored_scale
    with torch.no_grad():
        model.character_classifier.weight.mul_(multiplier)
    model.config.pii_character_runtime_logit_scale = target_scale
    return multiplier


@dataclass
class ContinuousCharacterTokenClassifierOutput(TokenClassifierOutput):
    """Token-classifier output with a training-only character readout."""

    character_logits: torch.Tensor | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def character_encoder_from_configuration(
    configuration: Mapping[str, object],
) -> ContinuousCharacterCnn:
    """Construct the character encoder named by a pretraining receipt."""
    return ContinuousCharacterCnn(
        int(configuration["primary_vocab_size"]),
        int(configuration["secondary_vocab_size"]),
        embedding_dim=int(configuration["embedding_dim"]),
        channels=int(configuration["channels"]),
        kernel_size=int(configuration["kernel_size"]),
        depth=int(configuration["depth"]),
        dropout=float(configuration["dropout"]),
        input_layout=str(configuration.get("character_input_layout", "paired-concat")),
        pair_interaction_rank=int(configuration.get("pair_interaction_rank", 0)),
    )


def load_character_pretrain(
    path: str | Path,
    *,
    initialization: str,
) -> tuple[ContinuousCharacterCnn, CharacterPairProjection, dict[str, object], dict[str, str]]:
    """Load either the pretrained encoder or its deterministic random ancestor."""
    path = Path(path)
    config_path = path / "config.json"
    summary_path = path / "summary.json"
    model_path = path / "model.safetensors"
    projection_path = path / CHARACTER_PROJECTION_FILENAME
    for required in (config_path, summary_path, model_path, projection_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if initialization not in {"pretrained", "random"}:
        raise ValueError(f"unsupported character initialization {initialization!r}")

    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("configuration") != configuration:
        raise ValueError("character pretraining config differs from its summary receipt")
    if configuration.get("character_text_view") != "raw-prefix":
        raise ValueError("joint token tagging currently requires the raw-prefix character view")
    projection = load_character_projection(projection_path)
    expected_projection_sha256 = configuration.get("character_projection", {}).get("sha256")
    if projection.metadata.get("sha256") != expected_projection_sha256:
        raise ValueError("character projection differs from the pretraining receipt")
    if projection.primary_vocab_size != int(
        configuration["primary_vocab_size"]
    ) or projection.secondary_vocab_size != int(configuration["secondary_vocab_size"]):
        raise ValueError("character projection vocabulary differs from the pretraining receipt")

    seed = int(configuration["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = character_encoder_from_configuration(configuration)
    if initialization == "pretrained":
        full_state = load_file(model_path)
        encoder_state = {
            name.removeprefix("encoder."): value
            for name, value in full_state.items()
            if name.startswith("encoder.")
        }
        encoder.load_state_dict(encoder_state, strict=True)
    source = {
        "path": str(path.resolve()),
        "config_sha256": file_sha256(config_path),
        "summary_sha256": file_sha256(summary_path),
        "model_sha256": file_sha256(model_path),
        "projection_sha256": file_sha256(projection_path),
    }
    return encoder, projection, configuration, source


def load_character_tagger_state(
    path: str | Path,
    *,
    label_names: Sequence[str],
    character_configuration: Mapping[str, object],
    character_projection_sha256: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, str]]:
    """Load a compatible isolated tagger's encoder and bias-free readout."""
    path = Path(path)
    config_path = path / "config.json"
    model_path = path / "model.safetensors"
    projection_path = path / CHARACTER_PROJECTION_FILENAME
    for required in (config_path, model_path, projection_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    for name, expected in character_configuration.items():
        if configuration.get(name) != expected:
            raise ValueError(f"character tagger configuration differs at {name!r}")
    observed_labels = configuration.get("character_tag_prediction", {}).get("label_names")
    if observed_labels != list(label_names):
        raise ValueError("character tagger and target model label vocabularies differ")
    if file_sha256(projection_path) != character_projection_sha256:
        raise ValueError("character tagger and pretraining projections differ")

    state = load_file(model_path)
    encoder_state = {
        name.removeprefix("encoder."): value for name, value in state.items() if name.startswith("encoder.")
    }
    weight = state.get("tag_classifier.weight")
    if weight is None:
        raise ValueError("character tagger artifact has no tag_classifier.weight")
    expected_shape = (len(label_names), int(character_configuration["channels"]))
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"character tagger readout shape {tuple(weight.shape)} differs from {expected_shape}"
        )
    source = {
        "path": str(path.resolve()),
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(model_path),
        "projection_sha256": file_sha256(projection_path),
        "bias_import_policy": "never",
    }
    return encoder_state, weight, source


def load_character_tag_head(
    path: str | Path,
    *,
    label_names: Sequence[str],
) -> torch.Tensor:
    """Load a fine-label character readout after verifying its label contract."""
    path = Path(path)
    configuration = json.loads((path / "config.json").read_text(encoding="utf-8"))
    observed_labels = configuration.get("character_tag_prediction", {}).get("label_names")
    if observed_labels != list(label_names):
        raise ValueError("character tagger and target model label vocabularies differ")
    state = load_file(path / "model.safetensors")
    weight = state.get("tag_classifier.weight")
    if weight is None:
        raise ValueError("character tagger artifact has no tag_classifier.weight")
    expected = (len(label_names), int(configuration["channels"]))
    if tuple(weight.shape) != expected:
        raise ValueError(f"character tagger readout shape {tuple(weight.shape)} differs from {expected}")
    return weight


def load_checkpoint_character_projection(path: str | Path) -> CharacterPairProjection:
    """Load and verify the character projection shipped with a joint checkpoint."""
    path = Path(path)
    config = AutoConfig.from_pretrained(path)
    if (
        getattr(config, "pii_head_architecture", None)
        not in SUPPORTED_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURES
    ):
        raise ValueError(f"{path} is not a continuous-character checkpoint")
    projection = load_character_projection(path / CHARACTER_PROJECTION_FILENAME)
    expected = getattr(config, "pii_character_projection_sha256", None)
    if projection.metadata.get("sha256") != expected:
        raise ValueError("checkpoint character projection hash differs from model config")
    return projection


class ContinuousCharacterForTokenClassification(PreTrainedModel):
    """Add a whole-segment character readout to the incumbent typed logits."""

    base_model_prefix = "encoder"
    main_input_name = "input_ids"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        *,
        encoder: nn.Module | None = None,
        character_encoder: ContinuousCharacterCnn | None = None,
        classifier: nn.Linear | None = None,
        character_classifier: nn.Linear | None = None,
        projection_bytes: bytes | None = None,
    ) -> None:
        super().__init__(config)
        architecture = getattr(config, "pii_head_architecture", None)
        if architecture not in SUPPORTED_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURES:
            raise ValueError(f"config is not a continuous-character classifier: {architecture!r}")
        self.num_labels = len(config.id2label)
        self.encoder = encoder if encoder is not None else AutoModel.from_config(config)
        self.character_encoder = (
            character_encoder
            if character_encoder is not None
            else character_encoder_from_configuration(config.pii_character_configuration)
        )
        self.pooling = str(config.pii_character_pooling)
        self.dropout = nn.Dropout(float(config.pii_classifier_dropout))
        self.classifier = classifier or nn.Linear(int(config.hidden_size), self.num_labels)
        self.character_classifier = character_classifier or nn.Linear(
            self.character_encoder.channels,
            self.num_labels,
            bias=False,
        )
        if (
            self.classifier.in_features != int(config.hidden_size)
            or self.classifier.out_features != self.num_labels
        ):
            raise ValueError(
                "incumbent classifier shape does not match the encoder: "
                f"got {self.classifier.in_features}x{self.classifier.out_features}"
            )
        if (
            self.character_classifier.in_features != self.character_encoder.channels
            or self.character_classifier.out_features != self.num_labels
            or self.character_classifier.bias is not None
        ):
            raise ValueError("character classifier must be a bias-free character-width-to-label affine")
        if projection_bytes is None:
            raise ValueError("joint character checkpoints require serialized projection bytes")
        observed_projection_sha256 = hashlib.sha256(projection_bytes).hexdigest()
        if observed_projection_sha256 != config.pii_character_projection_sha256:
            raise ValueError("serialized character projection differs from model config")
        self._character_projection_bytes = projection_bytes
        if architecture != CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE:
            config.pii_head_architecture_migrated_from = architecture
        config.pii_head_architecture = CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE
        ignored = list(getattr(config, "keys_to_ignore_at_inference", []) or [])
        if "character_logits" not in ignored:
            ignored.append("character_logits")
        config.keys_to_ignore_at_inference = ignored
        config.pii_head_parameters = self.task_head_parameter_count()

    def task_head_parameter_count(self) -> int:
        return (
            sum(parameter.numel() for parameter in self.character_encoder.parameters())
            + sum(parameter.numel() for parameter in self.classifier.parameters())
            + sum(parameter.numel() for parameter in self.character_classifier.parameters())
        )

    @classmethod
    def from_incumbent(
        cls,
        incumbent: PreTrainedModel,
        character_pretrain: str | Path,
        *,
        initialization: str,
        pooling: str = "max",
        character_tagger: str | Path | None = None,
        logit_initialization: str = "zero",
        logit_scale: float = 1.0,
    ) -> ContinuousCharacterForTokenClassification:
        if pooling not in {"max", "mean", "max-mean"}:
            raise ValueError(f"unsupported character pooling {pooling!r}")
        if initialization not in {"pretrained", "random", "tagger"}:
            raise ValueError(f"unsupported character initialization {initialization!r}")
        if logit_initialization not in {"zero", "tagger"}:
            raise ValueError(f"unsupported character logit initialization {logit_initialization!r}")
        if logit_scale <= 0:
            raise ValueError("character logit scale must be positive")
        if (initialization == "tagger" or logit_initialization == "tagger") and character_tagger is None:
            raise ValueError("tagger initialization requires a character tagger artifact")
        classifier = getattr(incumbent, "classifier", None)
        if not isinstance(classifier, nn.Linear):
            raise TypeError("incumbent must expose one linear classifier")
        if classifier.in_features != int(incumbent.config.hidden_size):
            raise ValueError("incumbent classifier is not a final-layer affine")
        character_encoder, projection, configuration, source = load_character_pretrain(
            character_pretrain,
            initialization="pretrained" if initialization == "tagger" else initialization,
        )
        projection_path = Path(character_pretrain) / CHARACTER_PROJECTION_FILENAME
        id2label = incumbent.config.id2label
        if id2label is None:
            raise ValueError("incumbent config has no id2label vocabulary")
        labels_by_index = {int(index): label for index, label in id2label.items()}
        expected_indices = set(range(len(labels_by_index)))
        if set(labels_by_index) != expected_indices:
            raise ValueError("incumbent id2label indices are not contiguous from zero")
        label_names = [labels_by_index[index] for index in range(len(labels_by_index))]
        tagger_source = None
        tagger_weight = None
        if character_tagger is not None:
            tagger_encoder_state, tagger_weight, tagger_source = load_character_tagger_state(
                character_tagger,
                label_names=label_names,
                character_configuration=configuration,
                character_projection_sha256=source["projection_sha256"],
            )
            if initialization == "tagger":
                character_encoder.load_state_dict(tagger_encoder_state, strict=True)
        config = copy.deepcopy(incumbent.config)
        config.architectures = [cls.__name__]
        config.pii_head_architecture = CONTINUOUS_CHARACTER_HEAD_ARCHITECTURE
        config.pii_character_configuration = configuration
        config.pii_character_initialization = initialization
        config.pii_character_pooling = pooling
        config.pii_character_projection_name = projection.name
        config.pii_character_projection_view = configuration["character_projection_view"]
        config.pii_character_projection_sha256 = source["projection_sha256"]
        config.pii_character_pretrain = source
        config.pii_character_tagger = tagger_source
        config.pii_character_logit_initialization = logit_initialization
        config.pii_character_logit_scale = logit_scale
        incumbent_classifier = copy.deepcopy(classifier)
        character_classifier = nn.Linear(
            character_encoder.channels,
            len(label_names),
            bias=False,
            device=classifier.weight.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            if logit_initialization == "zero":
                character_classifier.weight.zero_()
            else:
                if tagger_weight is None:
                    raise RuntimeError("tagger logit initialization did not load a tagger readout")
                character_classifier.weight.copy_(tagger_weight.float() * logit_scale)
        model = cls(
            config,
            encoder=incumbent.base_model,
            character_encoder=character_encoder,
            classifier=incumbent_classifier,
            character_classifier=character_classifier,
            projection_bytes=projection_path.read_bytes(),
        )
        model.character_encoder.to(dtype=torch.float32)
        model.config.pii_character_parameter_dtype = "float32"
        model.config.pii_character_classifier_dtype = "float32"
        return model

    @classmethod
    def from_local_checkpoint(
        cls,
        path: str | Path,
        *,
        dtype: torch.dtype | None = None,
    ) -> ContinuousCharacterForTokenClassification:
        path = Path(path)
        config = AutoConfig.from_pretrained(path)
        projection_path = path / CHARACTER_PROJECTION_FILENAME
        model = cls(config, projection_bytes=projection_path.read_bytes())
        safe_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if safe_path.is_file():
            state_dict = load_file(safe_path)
        elif bin_path.is_file():
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"no model weights found under {path}")
        joint_classifier_weight = state_dict.get("classifier.weight")
        if (
            "character_classifier.weight" not in state_dict
            and joint_classifier_weight is not None
            and joint_classifier_weight.shape[1] == int(config.hidden_size) + model.character_encoder.channels
        ):
            state_dict["classifier.weight"] = joint_classifier_weight[:, : int(config.hidden_size)]
            state_dict["character_classifier.weight"] = joint_classifier_weight[:, int(config.hidden_size) :]
        encoder_dtypes = {
            value.dtype
            for name, value in state_dict.items()
            if name.startswith("encoder.") and value.is_floating_point()
        }
        character_dtypes = {
            value.dtype
            for name, value in state_dict.items()
            if name.startswith("character_encoder.") and value.is_floating_point()
        }
        classifier_dtypes = {
            value.dtype
            for name, value in state_dict.items()
            if name.startswith("classifier.") and value.is_floating_point()
        }
        character_classifier_dtypes = {
            value.dtype
            for name, value in state_dict.items()
            if name.startswith("character_classifier.") and value.is_floating_point()
        }
        if (
            len(encoder_dtypes) != 1
            or len(character_dtypes) != 1
            or len(classifier_dtypes) != 1
            or len(character_classifier_dtypes) != 1
        ):
            raise RuntimeError(
                "checkpoint modules must each use one floating dtype: "
                f"encoder={encoder_dtypes} character={character_dtypes} "
                f"classifier={classifier_dtypes} character_classifier={character_classifier_dtypes}"
            )
        declared_classifier_dtype = configured_parameter_dtype(config, "pii_classifier_dtype")
        saved_classifier_dtype = next(iter(classifier_dtypes))
        if declared_classifier_dtype is not None and declared_classifier_dtype != saved_classifier_dtype:
            raise RuntimeError(
                "classifier checkpoint dtype differs from config: "
                f"weights={saved_classifier_dtype} config={declared_classifier_dtype}"
            )
        model.encoder.to(dtype=dtype or next(iter(encoder_dtypes)))
        model.character_encoder.to(dtype=next(iter(character_dtypes)))
        model.classifier.to(dtype=declared_classifier_dtype or dtype or saved_classifier_dtype)
        model.character_classifier.to(dtype=next(iter(character_classifier_dtypes)))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return model

    def save_pretrained(self, save_directory, *args, **kwargs) -> None:
        super().save_pretrained(save_directory, *args, **kwargs)
        destination = Path(save_directory) / CHARACTER_PROJECTION_FILENAME
        destination.write_bytes(self._character_projection_bytes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
        token_offsets: torch.Tensor,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **_kwargs,
    ) -> ContinuousCharacterTokenClassifierOutput:
        del token_type_ids
        return_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        character_states = self.character_encoder(character_ids, character_mask)
        token_character_states = pool_character_states(
            character_states,
            token_offsets,
            mode=self.pooling,
        )
        if token_character_states.shape[:2] != outputs.last_hidden_state.shape[:2]:
            raise ValueError("token offsets and encoder states must name the same token lattice")
        token_states = outputs.last_hidden_state.to(dtype=self.classifier.weight.dtype)
        incumbent_logits = self.classifier(self.dropout(token_states))
        token_character_states = token_character_states.to(dtype=self.character_classifier.weight.dtype)
        character_logits = self.character_classifier(self.dropout(token_character_states))
        logits = incumbent_logits + character_logits.to(dtype=incumbent_logits.dtype)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1),
            )
        return ContinuousCharacterTokenClassifierOutput(
            loss=cast(torch.FloatTensor | None, loss),
            logits=logits,
            hidden_states=outputs.hidden_states if return_hidden_states else None,
            character_logits=character_logits,
        )
