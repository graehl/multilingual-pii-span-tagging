"""Token classifier over concatenated hidden states from selected encoder layers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

if __package__:
    from scripts.pii_head_input_norm import CONFIG_KEY as HEAD_INPUT_NORM_KEY
    from scripts.pii_head_input_norm import HeadInputNorm, fold_into_linear
    from scripts.pii_head_residual_mlp import CONFIG_KEY as HEAD_RESIDUAL_MLP_KEY
    from scripts.pii_head_residual_mlp import ResidualHeadBlock
    from scripts.pii_prompt_slots import from_config as prompt_slots_from_config
    from scripts.pii_prompt_slots import install as install_prompt_slots
    from scripts.pii_soft_registers import (
        REGISTER_CONFIG_KEY,
        REGISTER_LAYERWISE_KEY,
        conditions_from_config,
    )
    from scripts.pii_soft_registers import install as install_soft_registers
else:
    from pii_head_input_norm import CONFIG_KEY as HEAD_INPUT_NORM_KEY
    from pii_head_input_norm import HeadInputNorm, fold_into_linear
    from pii_head_residual_mlp import CONFIG_KEY as HEAD_RESIDUAL_MLP_KEY
    from pii_head_residual_mlp import ResidualHeadBlock
    from pii_prompt_slots import from_config as prompt_slots_from_config
    from pii_prompt_slots import install as install_prompt_slots
    from pii_soft_registers import (
        REGISTER_CONFIG_KEY,
        REGISTER_LAYERWISE_KEY,
        conditions_from_config,
    )
    from pii_soft_registers import install as install_soft_registers

CONCAT_HEAD_ARCHITECTURE = "concat_encoder_layers"
DEFERRED_REFERENCE_TYPES = ("person_reference", "organization_reference")


@dataclass
class DualHeadTokenClassifierOutput(TokenClassifierOutput):
    """Token classifier output carrying optional auxiliary logits.

    ``logits`` stays the model's own vocabulary so every existing consumer --
    decode, span metrics, checkpoint tooling -- reads the same field it always
    did. ``secondary_logits`` is the added head while a run trains two
    ontologies at once. ``predicate_logits`` contains independent sigmoid
    channels when a successor predicate objective is attached. A
    primary-type-conditioned head adds a condition axis:
    ``[batch, token, primary type, predicate]``. ``subclass_logits`` is one
    flat affine bank whose contiguous blocks are declared by carrier type and
    categorical family in the saved config. ``reference_type_logits`` carries
    one separately supervised semantic score per configured reference type;
    each score is also added to that type's B/I/E/S primary rows.
    """

    secondary_logits: torch.Tensor | None = None
    predicate_logits: torch.Tensor | None = None
    subclass_logits: torch.Tensor | None = None
    reference_type_logits: torch.Tensor | None = None


def reference_type_label_index(config, reference_types: list[str]) -> torch.Tensor:
    """Map each primary BIOES row to its shared reference-type score."""
    if len(set(reference_types)) != len(reference_types) or any(
        not isinstance(primary_type, str) or not primary_type for primary_type in reference_types
    ):
        raise ValueError("reference residual types must be unique nonempty strings")
    labels = [config.id2label[index] for index in range(config.num_labels)]
    label2ids: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        label2ids.setdefault(label, []).append(index)
    type_by_label = [-1] * len(labels)
    for type_index, primary_type in enumerate(reference_types):
        for boundary in "BIES":
            label = f"{boundary}-{primary_type}"
            rows = label2ids.get(label, ())
            if len(rows) != 1:
                raise ValueError(f"reference residual type {primary_type!r} needs exactly one {label!r} row")
            type_by_label[rows[0]] = type_index
    return torch.tensor(type_by_label, dtype=torch.long)


def subclass_head_rows(blocks: list[dict] | tuple[dict, ...]) -> int:
    """Validate contiguous carrier-conditioned blocks and return row count."""
    next_start = 0
    seen = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or set(block) != {
            "family",
            "primary_type",
            "start",
            "width",
        }:
            raise ValueError(f"subclass block {index} has an invalid shape")
        key = (block["family"], block["primary_type"])
        if (
            any(not isinstance(value, str) or not value for value in key)
            or key in seen
            or isinstance(block["start"], bool)
            or not isinstance(block["start"], int)
            or block["start"] != next_start
            or isinstance(block["width"], bool)
            or not isinstance(block["width"], int)
            or block["width"] <= 0
        ):
            raise ValueError(f"subclass block {index} is invalid or noncontiguous")
        seen.add(key)
        next_start += block["width"]
    if not next_start:
        raise ValueError("subclass head needs at least one nonempty block")
    return next_start


def resolve_encoder_layers(indices: list[int] | tuple[int, ...], num_hidden_layers: int) -> tuple[int, ...]:
    """Resolve hidden-state indices while preserving explicit embedding output 0.

    Positive indices are 1-based encoder layers, 0 is the embedding-stack
    output, and negative indices count backward through encoder layers only.
    """
    resolved = []
    for index in indices:
        layer = num_hidden_layers + 1 + index if index < 0 else index
        minimum_layer = 1 if index < 0 else 0
        if layer < minimum_layer or layer > num_hidden_layers:
            raise ValueError(
                f"hidden-state layer {index} is outside 0..{num_hidden_layers}; "
                "0 selects the embedding output"
            )
        if layer in resolved:
            raise ValueError(f"layer indices select the same hidden state more than once: {indices}")
        resolved.append(layer)
    if not resolved:
        raise ValueError("select at least one hidden-state layer")
    return tuple(resolved)


def resolve_token_offsets(offsets: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Validate relative token positions used as classifier inputs."""
    resolved = tuple(int(offset) for offset in offsets)
    if not resolved:
        raise ValueError("select at least one token offset")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"token offsets must be unique: {offsets}")
    if 0 not in resolved:
        raise ValueError(f"token offsets must include the current token (0): {offsets}")
    return resolved


def shift_token_features(
    features: torch.Tensor,
    offset: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """At target position t, return features from source position t + offset."""
    if features.ndim != 3:
        raise ValueError(f"token features must have shape [batch, tokens, width], got {features.shape}")
    if attention_mask.shape != features.shape[:2]:
        raise ValueError(
            f"attention mask shape {attention_mask.shape} does not match token features {features.shape[:2]}"
        )
    token_count = features.shape[1]
    distance = abs(offset)
    shifted = torch.zeros_like(features)
    source_valid = torch.zeros_like(attention_mask, dtype=torch.bool)
    if offset == 0:
        shifted = features
        source_valid = attention_mask.bool()
    elif distance < token_count and offset < 0:
        shifted[:, distance:, :] = features[:, :-distance, :]
        source_valid[:, distance:] = attention_mask[:, :-distance].bool()
    elif distance < token_count:
        shifted[:, :-distance, :] = features[:, distance:, :]
        source_valid[:, :-distance] = attention_mask[:, distance:].bool()
    valid = source_valid & attention_mask.bool()
    return shifted * valid.unsqueeze(-1).to(dtype=features.dtype)


def _encoder_from_config(config):
    """Build the bare encoder from a saved config, matching the pretrained loader's
    attention choice: `from_pretrained` silently uses eager attention for an
    architecture without an SDPA kernel (DeBERTa-v2), while `from_config` raises."""
    try:
        return AutoModel.from_config(config)
    except ValueError as error:
        if "scaled_dot_product_attention" not in str(error):
            raise
        return AutoModel.from_config(config, attn_implementation="eager")


class LayerConcatForTokenClassification(PreTrainedModel):
    """AutoModel encoder and a task head over separately retained layer states."""

    base_model_prefix = "encoder"
    main_input_name = "input_ids"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    supports_gradient_checkpointing = True

    def __init__(self, config, encoder: nn.Module | None = None) -> None:
        super().__init__(config)
        if getattr(config, "pii_head_architecture", None) != CONCAT_HEAD_ARCHITECTURE:
            raise ValueError(f"config is not a {CONCAT_HEAD_ARCHITECTURE} token classifier")
        if config.pii_head_kind not in ("affine", "factorized_linear", "mlp"):
            raise ValueError(f"unsupported concatenated-layer head: {config.pii_head_kind}")

        self.num_labels = config.num_labels
        self.encoder = encoder if encoder is not None else _encoder_from_config(config)
        self.layer_indices = resolve_encoder_layers(config.pii_encoder_layers, config.num_hidden_layers)
        config.pii_encoder_layers = list(self.layer_indices)
        self.token_offsets = resolve_token_offsets(getattr(config, "pii_token_offsets", [0]))
        config.pii_token_offsets = list(self.token_offsets)
        if self.token_offsets != (0,):
            if config.pii_head_kind != "affine":
                raise ValueError("multi-token-state concatenation currently supports only an affine head")
            if len(self.layer_indices) != 1:
                raise ValueError("multi-token-state concatenation requires exactly one encoder layer")
            self.token_offset_normalizations = nn.ModuleList(
                nn.LayerNorm(config.hidden_size) for _ in self.token_offsets
            )
        else:
            self.token_offset_normalizations = None
        self.final_layer_only = self.layer_indices == (config.num_hidden_layers,)
        input_width = len(self.layer_indices) * len(self.token_offsets) * config.hidden_size
        rank = int(config.pii_head_rank)
        modernbert_head = bool(getattr(config, "pii_modernbert_native_head", False))
        if modernbert_head and not (
            config.model_type == "modernbert"
            and config.pii_head_kind == "mlp"
            and self.final_layer_only
            and self.token_offsets == (0,)
            and rank == config.hidden_size
            and config.classifier_activation == "gelu"
        ):
            raise ValueError("native ModernBert head requires final-layer, hidden-width GELU")

        if config.pii_head_kind == "affine":
            self.projection = None
            self.activation = nn.Identity()
            self.normalization = nn.Identity()
            classifier_width = input_width
        else:
            if rank <= 0:
                raise ValueError("low-rank head requires a positive rank")
            self.projection = nn.Linear(
                input_width, rank, bias=config.classifier_bias if modernbert_head else True
            )
            self.activation = nn.GELU() if config.pii_head_kind == "mlp" else nn.Identity()
            self.normalization = (
                nn.LayerNorm(
                    rank,
                    eps=config.norm_eps if modernbert_head else 1e-5,
                    bias=config.norm_bias if modernbert_head else True,
                )
                if config.pii_head_kind == "mlp"
                else nn.Identity()
            )
            classifier_width = rank
        norm_spec = getattr(config, HEAD_INPUT_NORM_KEY, None)
        self.head_input_norm = HeadInputNorm.from_config(classifier_width, norm_spec) if norm_spec else None
        block_spec = getattr(config, HEAD_RESIDUAL_MLP_KEY, None)
        self.head_residual_mlp = (
            ResidualHeadBlock.from_config(classifier_width, block_spec) if block_spec else None
        )
        self.dropout = nn.Dropout(float(config.pii_classifier_dropout))
        self.classifier = nn.Linear(classifier_width, config.num_labels)
        self.native_classifiers = nn.ModuleDict()
        self.language_bias = None
        language_inventory = list(getattr(config, "pii_language_bias_languages", ()) or ())
        secondary_labels = getattr(config, "pii_secondary_labels", None)
        self.secondary_classifier = (
            nn.Linear(classifier_width, len(secondary_labels)) if secondary_labels else None
        )
        predicate_channels = getattr(config, "pii_predicate_channels", None)
        predicate_condition_types = list(getattr(config, "pii_predicate_condition_types", ()) or ())
        if len(set(predicate_condition_types)) != len(predicate_condition_types):
            raise ValueError("predicate condition types must be unique")
        predicate_rows = len(predicate_channels or ()) * max(1, len(predicate_condition_types))
        self.predicate_classifier = (
            nn.Linear(classifier_width, predicate_rows) if predicate_channels else None
        )
        subclass_blocks = list(getattr(config, "pii_subclass_blocks", ()) or ())
        subclass_rows = subclass_head_rows(subclass_blocks) if subclass_blocks else 0
        self.subclass_classifier = nn.Linear(classifier_width, subclass_rows) if subclass_rows else None
        reference_type_residual_types = list(getattr(config, "pii_reference_type_residual_types", ()) or ())
        self.reference_type_classifier = (
            nn.Linear(classifier_width, len(reference_type_residual_types))
            if reference_type_residual_types
            else None
        )
        self.register_buffer(
            "_reference_type_index_by_label",
            reference_type_label_index(config, reference_type_residual_types),
            persistent=False,
        )
        self.register_buffer(
            "_deferred_reference_labels", torch.zeros(config.num_labels, dtype=torch.bool), persistent=False
        )
        self.defer_reference_training(bool(getattr(config, "pii_defer_reference_training", False)))
        self._initialize_task_head()
        native_specs = dict(getattr(config, "pii_native_heads", {}) or {})
        config.pii_native_heads = {}
        for name, spec in native_specs.items():
            self.attach_native_head(name, spec["labels"], coverage=spec["coverage"])
        if language_inventory:
            self.attach_language_bias(
                language_inventory, strength=float(getattr(config, "pii_language_bias_strength", 1.0))
            )
        config.pii_head_parameters = self.task_head_parameter_count()

    def defer_reference_training(self, enabled: bool) -> None:
        """Keep checkpoint rows but exclude references from the primary distribution."""
        if enabled and self.secondary_classifier is not None:
            raise ValueError("reference deferral requires a single primary head")
        types = list(DEFERRED_REFERENCE_TYPES) if enabled else []
        self._deferred_reference_labels = (reference_type_label_index(self.config, types) >= 0).to(
            device=self.classifier.weight.device
        )
        self.config.pii_defer_reference_training = enabled

    def task_head_parameter_count(self) -> int:
        """Count every trained parameter outside the pretrained encoder."""
        return sum(
            parameter.numel()
            for module in (
                self.projection,
                self.normalization,
                self.token_offset_normalizations,
                self.head_input_norm,
                self.head_residual_mlp,
                self.classifier,
                self.native_classifiers,
                self.secondary_classifier,
                self.predicate_classifier,
                self.subclass_classifier,
                self.reference_type_classifier,
                self.language_bias,
            )
            if module is not None
            for parameter in module.parameters()
        )

    def _initialize_task_head(self) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        for module in (
            self.projection,
            self.classifier,
            self.secondary_classifier,
            self.predicate_classifier,
            self.subclass_classifier,
        ):
            if module is None:
                continue
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.reference_type_classifier is not None:
            nn.init.zeros_(self.reference_type_classifier.weight)
            nn.init.zeros_(self.reference_type_classifier.bias)

    def attach_language_bias(self, languages: list[str], *, strength: float = 1.0) -> None:
        """Add a zero-initialized BIOES bias per explicitly routed language.

        A route of -1 uses the shared head. The caller owns language-ID routing;
        a missing route is an error when the conditioned head is present.
        """
        if not languages or any(not isinstance(lang, str) or not lang for lang in languages):
            raise ValueError("language bias needs nonempty language names")
        if len(set(languages)) != len(languages) or not math.isfinite(strength):
            raise ValueError("language bias needs unique languages and finite strength")
        existing = list(getattr(self.config, "pii_language_bias_languages", ()) or ())
        if self.language_bias is not None:
            if existing != languages:
                raise ValueError("language-bias inventory cannot change on an existing head")
        else:
            self.language_bias = nn.Embedding(
                len(languages),
                self.num_labels,
                device=self.classifier.weight.device,
                dtype=self.classifier.weight.dtype,
            )
            nn.init.zeros_(self.language_bias.weight)
        self.config.pii_language_bias_languages = list(languages)
        self.config.pii_language_bias_strength = strength
        self.config.pii_head_parameters = self.task_head_parameter_count()

    def head_feature_consumers(self) -> list[nn.Linear]:
        """Every affine map that reads the post-dropout head features."""
        consumers = [
            self.classifier,
            self.secondary_classifier,
            self.predicate_classifier,
            self.subclass_classifier,
            self.reference_type_classifier,
            *self.native_classifiers.values(),
        ]
        return [module for module in consumers if module is not None]

    def attach_head_input_norm(self, spec: dict, mean: torch.Tensor, var: torch.Tensor) -> None:
        """Insert per-channel head-input normalization, identity at these moments."""
        if self.head_input_norm is not None:
            raise ValueError("model already carries head input normalization")
        width = self.classifier.in_features
        norm = HeadInputNorm.from_config(width, spec).to(device=self.classifier.weight.device)
        norm.calibrate(mean, var)
        norm.train(self.training)
        self.head_input_norm = norm
        setattr(self.config, HEAD_INPUT_NORM_KEY, norm.config())
        self.config.pii_head_parameters = self.task_head_parameter_count()

    def attach_head_residual_mlp(self, spec: dict, mean: torch.Tensor | None = None, var=None) -> None:
        """Insert a zero-initialized residual GELU block; the served function is unchanged."""
        if self.head_residual_mlp is not None:
            raise ValueError("model already carries a residual head block")
        block = ResidualHeadBlock.from_config(self.classifier.in_features, spec)
        block.to(device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)
        if block.norm_kind == "batch":
            if mean is None or var is None:
                raise ValueError("a BatchNorm residual block needs calibration moments")
            block.calibrate(mean, var)
        block.train(self.training)
        self.head_residual_mlp = block
        setattr(self.config, HEAD_RESIDUAL_MLP_KEY, block.config())
        self.config.pii_head_parameters = self.task_head_parameter_count()

    def fold_head_input_norm(self) -> None:
        """Absorb the evaluation-mode normalization into every consumer and drop it."""
        if self.head_input_norm is None:
            raise ValueError("model carries no head input normalization to fold")
        scale, shift = self.head_input_norm.eval_affine()
        for linear in self.head_feature_consumers():
            fold_into_linear(linear, scale.to(linear.weight.device), shift.to(linear.weight.device))
        self.head_input_norm = None
        setattr(self.config, HEAD_INPUT_NORM_KEY, None)
        self.config.pii_head_parameters = self.task_head_parameter_count()

    def text_token_mask(self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor) -> torch.Tensor:
        """Real text positions: attended, and not the sentence start or end marker."""
        mask = attention_mask.bool()
        if input_ids is None:
            return mask
        for special in (self.config.bos_token_id, self.config.eos_token_id):
            if special is not None:
                mask = mask & (input_ids != special)
        return mask

    def apply_language_bias(self, logits: torch.Tensor, language_ids: torch.Tensor | None) -> torch.Tensor:
        if self.language_bias is None:
            if language_ids is not None:
                raise ValueError("language_ids supplied to an unconditioned model")
            return logits
        if language_ids is None or language_ids.shape != (logits.shape[0],):
            raise ValueError("conditioned model requires one language_ids entry per batch row")
        if (
            language_ids.dtype != torch.long
            or torch.any(language_ids < -1)
            or torch.any(language_ids >= self.language_bias.num_embeddings)
        ):
            raise ValueError("language_ids must be long inventory indices or -1 for shared")
        strength = float(self.config.pii_language_bias_strength)
        if not math.isfinite(strength):
            raise ValueError("language bias strength must be finite")
        if strength == 0:
            return logits
        residual = self.language_bias(language_ids.clamp_min(0))
        residual = residual.masked_fill((language_ids == -1).unsqueeze(-1), 0)
        return logits + strength * residual.unsqueeze(1)

    def attach_predicate_head(
        self,
        channels: list[str],
        *,
        condition_types: list[str] | None = None,
    ) -> bool:
        """Attach predicate rows, optionally independent per semantic primary type.

        Converting an existing unconditional head replicates its rows into each
        condition block. This changes parameter sharing without changing any
        condition's initial logits.
        """
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("predicate channels must be a nonempty unique list")
        condition_types = list(condition_types or ())
        if len(set(condition_types)) != len(condition_types):
            raise ValueError("predicate condition types must be unique")
        existing = list(getattr(self.config, "pii_predicate_channels", ()) or ())
        existing_condition_types = list(getattr(self.config, "pii_predicate_condition_types", ()) or ())
        if self.predicate_classifier is not None:
            if existing != channels:
                raise ValueError(f"predicate channel mismatch: checkpoint={existing}, requested={channels}")
            if existing_condition_types != condition_types:
                if existing_condition_types or not condition_types:
                    raise ValueError(
                        "predicate condition mismatch: "
                        f"checkpoint={existing_condition_types}, requested={condition_types}"
                    )
                source = self.predicate_classifier
                head = nn.Linear(
                    source.in_features,
                    len(channels) * len(condition_types),
                    bias=source.bias is not None,
                    device=source.weight.device,
                    dtype=source.weight.dtype,
                )
                with torch.no_grad():
                    head.weight.copy_(source.weight.repeat(len(condition_types), 1))
                    if head.bias is not None:
                        head.bias.copy_(source.bias.repeat(len(condition_types)))
                self.predicate_classifier = head
                self.config.pii_predicate_conditioning = "primary_type"
                self.config.pii_predicate_condition_types = condition_types
                self.config.pii_head_parameters = self.task_head_parameter_count()
                return True
            ignored = list(getattr(self.config, "keys_to_ignore_at_inference", ()) or ())
            if "predicate_logits" not in ignored:
                ignored.append("predicate_logits")
            self.config.keys_to_ignore_at_inference = ignored
            return False
        if existing:
            raise ValueError("config declares predicate channels but the model has no predicate head")
        head = nn.Linear(
            self.classifier.in_features,
            len(channels) * max(1, len(condition_types)),
        )
        nn.init.normal_(
            head.weight,
            mean=0.0,
            std=float(getattr(self.config, "initializer_range", 0.02)),
        )
        nn.init.zeros_(head.bias)
        self.predicate_classifier = head.to(
            dtype=self.classifier.weight.dtype,
            device=self.classifier.weight.device,
        )
        self.config.pii_predicate_channels = list(channels)
        self.config.pii_predicate_conditioning = "primary_type" if condition_types else "none"
        self.config.pii_predicate_condition_types = condition_types
        ignored = list(getattr(self.config, "keys_to_ignore_at_inference", ()) or ())
        if "predicate_logits" not in ignored:
            ignored.append("predicate_logits")
        self.config.keys_to_ignore_at_inference = ignored
        self.config.pii_head_parameters = self.task_head_parameter_count()
        return True

    def attach_subclass_head(
        self,
        blocks: list[dict],
        *,
        spec_sha256: str,
    ) -> bool:
        """Attach one flat affine bank partitioned by carrier and family."""
        rows = subclass_head_rows(blocks)
        if not isinstance(spec_sha256, str) or len(spec_sha256) != 64:
            raise ValueError("subclass spec SHA-256 must be a 64-character string")
        normalized_blocks = [dict(block) for block in blocks]
        existing_blocks = list(getattr(self.config, "pii_subclass_blocks", ()) or ())
        existing_sha256 = getattr(self.config, "pii_subclass_spec_sha256", None)
        if self.subclass_classifier is not None:
            if existing_blocks != normalized_blocks or existing_sha256 != spec_sha256:
                raise ValueError(
                    "subclass head mismatch: "
                    f"checkpoint blocks/SHA={existing_blocks}/{existing_sha256}, "
                    f"requested={normalized_blocks}/{spec_sha256}"
                )
            return False
        if existing_blocks or existing_sha256 is not None:
            raise ValueError("config declares subclasses but the model has no subclass head")
        head = nn.Linear(self.classifier.in_features, rows)
        nn.init.normal_(
            head.weight,
            mean=0.0,
            std=float(getattr(self.config, "initializer_range", 0.02)),
        )
        nn.init.zeros_(head.bias)
        self.subclass_classifier = head.to(
            dtype=self.classifier.weight.dtype,
            device=self.classifier.weight.device,
        )
        self.config.pii_subclass_blocks = normalized_blocks
        self.config.pii_subclass_spec_sha256 = spec_sha256
        ignored = list(getattr(self.config, "keys_to_ignore_at_inference", ()) or ())
        if "subclass_logits" not in ignored:
            ignored.append("subclass_logits")
        self.config.keys_to_ignore_at_inference = ignored
        self.config.pii_head_parameters = self.task_head_parameter_count()
        return True

    def attach_reference_type_residual(self, primary_types: list[str]) -> bool:
        """Attach zero-initialized semantic scores shared across BIOES rows."""
        if not primary_types:
            raise ValueError("reference residual needs at least one primary type")
        type_by_label = reference_type_label_index(self.config, primary_types)
        existing = list(getattr(self.config, "pii_reference_type_residual_types", ()) or ())
        if self.reference_type_classifier is not None:
            if existing != primary_types:
                raise ValueError(
                    f"reference residual type mismatch: checkpoint={existing}, requested={primary_types}"
                )
            return False
        if existing:
            raise ValueError("config declares reference residual types but the model has no residual head")
        head = nn.Linear(self.classifier.in_features, len(primary_types))
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.reference_type_classifier = head.to(
            dtype=self.classifier.weight.dtype,
            device=self.classifier.weight.device,
        )
        self._reference_type_index_by_label = type_by_label.to(device=self.classifier.weight.device)
        self.config.pii_reference_type_residual_types = list(primary_types)
        ignored = list(getattr(self.config, "keys_to_ignore_at_inference", ()) or ())
        if "reference_type_logits" not in ignored:
            ignored.append("reference_type_logits")
        self.config.keys_to_ignore_at_inference = ignored
        self.config.pii_head_parameters = self.task_head_parameter_count()
        return True

    def attach_native_head(self, name: str, labels: list[str], *, coverage: str) -> None:
        """Keep a source ontology's affine readout without projecting its O."""
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in self.native_classifiers:
            raise ValueError(f"invalid or duplicate native head: {name!r}")
        if not coverage or not labels or labels[0] != "O" or len(set(labels)) != len(labels):
            raise ValueError("native heads require coverage and unique labels beginning with O")
        types = set()
        for label in labels[1:]:
            match = re.fullmatch(r"[BIES]-(.+)", label)
            if match is None:
                raise ValueError(f"invalid native BIOES label: {label!r}")
            types.add(match[1])
        if not types or set(labels[1:]) != {f"{boundary}-{tag}" for tag in types for boundary in "BIES"}:
            raise ValueError("native label inventory requires all four BIOES boundaries per type")
        head = nn.Linear(self.classifier.in_features, len(labels))
        nn.init.normal_(head.weight, mean=0.0, std=float(getattr(self.config, "initializer_range", 0.02)))
        nn.init.zeros_(head.bias)
        self.native_classifiers[name] = head.to(
            dtype=self.classifier.weight.dtype, device=self.classifier.weight.device
        )
        self.config.pii_native_heads = {
            **self.config.pii_native_heads,
            name: {"labels": list(labels), "coverage": coverage},
        }
        self.config.pii_head_parameters = self.task_head_parameter_count()

    def attach_secondary_head(self, labels: list[str]) -> None:
        """Add a second label space's classifier over the existing features.

        Used after a warm start: the checkpoint supplies the encoder and its
        own head, and the new ontology's head is created here, fresh, on the
        same features and in the same dtype and device.
        """
        if self.secondary_classifier is not None:
            raise ValueError("this model already carries a secondary head")
        if not labels:
            raise ValueError("a secondary head needs a non-empty label inventory")
        head = nn.Linear(self.classifier.in_features, len(labels))
        nn.init.normal_(head.weight, mean=0.0, std=float(getattr(self.config, "initializer_range", 0.02)))
        nn.init.zeros_(head.bias)
        self.secondary_classifier = head.to(
            dtype=self.classifier.weight.dtype, device=self.classifier.weight.device
        )
        self.config.pii_secondary_labels = list(labels)
        self.config.pii_head_parameters = self.task_head_parameter_count()

    def retire_primary_head(self) -> list[nn.Parameter]:
        """Promote the secondary head in place and delete the primary one.

        Called once the transition schedule has taken the old ontology's
        authority to zero for good. The old head is not merely unweighted: its
        module is dropped, so it costs no forward pass, no activation memory,
        no gradient, and no slot in anything saved afterwards. From here the
        model is an ordinary single-head tagger over the new vocabulary, which
        is what a run past its fade should export.

        The surviving head keeps its parameter *objects*, so an optimizer
        already tracking them keeps their state; the returned list is the
        parameters that no longer exist, for the caller to evict from the
        optimizer.
        """
        if self.secondary_classifier is None:
            raise ValueError("this model carries no secondary head to promote")
        retired = list(self.classifier.parameters())
        labels = list(self.config.pii_secondary_labels)
        self.classifier = self.secondary_classifier
        self.secondary_classifier = None
        self.num_labels = len(labels)
        self.config.num_labels = len(labels)
        self.config.id2label = dict(enumerate(labels))
        self.config.label2id = {label: index for index, label in enumerate(labels)}
        self.config.pii_secondary_labels = None
        self.config.pii_head_parameters = self.task_head_parameter_count()
        return retired

    @classmethod
    def from_encoder_pretrained(
        cls,
        model_name: str | Path,
        label_names: list[str],
        *,
        layers: list[int] | tuple[int, ...],
        token_offsets: list[int] | tuple[int, ...] = (0,),
        head_kind: str,
        rank: int,
        dtype: torch.dtype = torch.bfloat16,
        dropout: float = 0.1,
    ) -> LayerConcatForTokenClassification:
        config = AutoConfig.from_pretrained(model_name)
        resolved_layers = resolve_encoder_layers(layers, config.num_hidden_layers)
        config.num_labels = len(label_names)
        config.id2label = dict(enumerate(label_names))
        config.label2id = {label: index for index, label in enumerate(label_names)}
        config.architectures = [cls.__name__]
        config.pii_head_architecture = CONCAT_HEAD_ARCHITECTURE
        config.pii_encoder_model = str(model_name)
        config.pii_encoder_layers = list(resolved_layers)
        config.pii_token_offsets = list(resolve_token_offsets(token_offsets))
        config.pii_head_kind = head_kind
        config.pii_head_rank = rank
        config.pii_classifier_dropout = dropout
        encoder = AutoModel.from_pretrained(model_name, config=config, dtype=dtype)
        return cls(config, encoder=encoder).to(dtype=dtype)

    @classmethod
    def from_local_checkpoint(
        cls,
        path: str | Path,
        *,
        dtype: torch.dtype | None = None,
    ) -> LayerConcatForTokenClassification:
        path = Path(path)
        config = AutoConfig.from_pretrained(path)
        if getattr(config, "pii_head_architecture", None) != CONCAT_HEAD_ARCHITECTURE:
            raise ValueError(f"{path} is not a {CONCAT_HEAD_ARCHITECTURE} checkpoint")
        model = cls(config)
        # Registers are model parameters, so a checkpoint trained with them carries
        # `pii_soft_registers.*`. Install before loading: the strict check below treats
        # those keys as an unexpected checkpoint otherwise, and the module has to exist
        # for its hooks to be in place on the first forward.
        register_count = int(getattr(config, REGISTER_CONFIG_KEY, 0) or 0)
        if register_count:
            install_soft_registers(
                model,
                register_count,
                layerwise=bool(getattr(config, REGISTER_LAYERWISE_KEY, True)),
                conditions=conditions_from_config(config),
            )
        # Prompt slots are model parameters too; rebuild them before the strict load.
        prompt = prompt_slots_from_config(config, config.hidden_size)
        if prompt is not None:
            install_prompt_slots(model, prompt)
        safe_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if safe_path.is_file():
            state_dict = load_file(safe_path)
        elif bin_path.is_file():
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"no model weights found under {path}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if dtype is not None:
            model.to(dtype=dtype)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        language_ids: torch.Tensor | None = None,
        native_head: str | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_kwargs,
    ) -> TokenClassifierOutput:
        del token_type_ids
        if native_head is not None and native_head not in self.native_classifiers:
            raise ValueError(f"unknown native head: {native_head!r}")
        return_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **({"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {}),
            **({"position_ids": position_ids} if position_ids is not None else {}),
            output_hidden_states=return_hidden_states or not self.final_layer_only,
            return_dict=True,
        )
        hidden = self.classifier_features(
            outputs,
            attention_mask,
            token_mask=self.text_token_mask(input_ids, attention_mask)
            if self.head_input_norm is not None or self.head_residual_mlp is not None
            else None,
        )
        dropped = self.dropout(hidden)
        if native_head is not None:
            logits = self.native_classifiers[native_head](dropped)
            loss = (
                None
                if labels is None
                else nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
            )
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if return_hidden_states else None,
            )
        logits = self.classifier(dropped)
        predicate_logits = None if self.predicate_classifier is None else self.predicate_classifier(dropped)
        subclass_logits = None if self.subclass_classifier is None else self.subclass_classifier(dropped)
        reference_type_logits = (
            None if self.reference_type_classifier is None else self.reference_type_classifier(dropped)
        )
        if reference_type_logits is not None:
            zero = reference_type_logits.new_zeros(*reference_type_logits.shape[:-1], 1)
            residual_by_type = torch.cat((zero, reference_type_logits), dim=-1)
            logits = logits + residual_by_type.index_select(
                -1,
                self._reference_type_index_by_label + 1,
            )
        logits = self.apply_language_bias(logits, language_ids)
        if self.config.pii_defer_reference_training:
            # A finite mask avoids inf*0 in masked multi-objective losses while
            # making reference probability zero at the supported float dtypes.
            logits = logits.masked_fill(self._deferred_reference_labels, -10000.0)
        predicate_condition_types = list(getattr(self.config, "pii_predicate_condition_types", ()) or ())
        if predicate_logits is not None and predicate_condition_types:
            predicate_logits = predicate_logits.reshape(
                *predicate_logits.shape[:-1],
                len(predicate_condition_types),
                len(self.config.pii_predicate_channels),
            )
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1),
            )
        if (
            self.secondary_classifier is None
            and predicate_logits is None
            and subclass_logits is None
            and reference_type_logits is None
        ):
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if return_hidden_states else None,
            )
        # The two heads share one dropout draw, so the blend the dual-head loss
        # takes compares them on the same sampled features.
        return DualHeadTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if return_hidden_states else None,
            secondary_logits=(
                None if self.secondary_classifier is None else self.secondary_classifier(dropped)
            ),
            predicate_logits=predicate_logits,
            subclass_logits=subclass_logits,
            reference_type_logits=reference_type_logits,
        )

    def classifier_features(
        self,
        outputs,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the deterministic features immediately before head dropout.

        Affine-head transport and diagnostic probes need the exact coordinate
        system consumed by ``classifier``.  Keeping that construction here
        prevents those consumers from reimplementing layer selection, token
        offsets, projection, activation, or normalization differently from
        the model's forward path.
        """
        if self.final_layer_only:
            features = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            features = torch.cat([hidden_states[index] for index in self.layer_indices], dim=-1)
        if self.token_offset_normalizations is not None:
            features = torch.cat(
                [
                    shift_token_features(normalization(features), offset, attention_mask)
                    for normalization, offset in zip(
                        self.token_offset_normalizations,
                        self.token_offsets,
                        strict=True,
                    )
                ],
                dim=-1,
            )
        hidden = features
        if self.projection is not None:
            hidden = self.projection(hidden)
        hidden = self.normalization(self.activation(hidden))
        if self.head_input_norm is not None:
            # Batch-dependent kinds need the real-text mask only in training;
            # evaluation applies the fixed running-moment affine map.
            hidden = self.head_input_norm(hidden, token_mask)
        if self.head_residual_mlp is not None:
            hidden = self.head_residual_mlp(hidden, token_mask)
        return hidden
