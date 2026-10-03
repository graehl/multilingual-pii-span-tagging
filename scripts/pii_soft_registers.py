#!/usr/bin/env python3
"""Learned register positions for an encoder tagger, re-injected at every layer.

Extra input positions the encoder attends to and is never scored on. They occupy real
token slots, so attention can both read them and write into them, which is the property
that distinguishes this from a key/value prefix: the registers participate.

Two parameter groups:

  shared        the layer-0 embedding written into each register position
  layer_delta   a vector added to each register position after every encoder block

The second group is why this is not input-only prompt tuning. With input-only prompts a
few thousand parameters at the bottom must steer the whole depth through one entry
point, and published results put that configuration at its weakest on small encoders
doing token-level tasks. Re-injecting per layer gives direct control at every depth for
roughly `layers` times the parameters, while keeping the registers as ordinary
participating tokens rather than attention-only key/value entries.

The encoder must be trainable for any of this to matter: the registers are
out-of-distribution inputs, and a frozen encoder has no way to learn to consult them.
"""

from __future__ import annotations

import torch
from torch import nn

REGISTER_CONFIG_KEY = "pii_soft_register_count"
REGISTER_LAYERWISE_KEY = "pii_soft_register_layerwise"
REGISTER_CONDITIONS_KEY = "pii_soft_register_conditions"
# Index 0 of every condition vocabulary. Production feeds it, training-time dropout
# substitutes it, and effectiveness is scored under it, so it is trained rather than
# extrapolated.
UNKNOWN_CONDITION = "unknown"


class SoftRegisters(nn.Module):
    def __init__(
        self,
        count: int,
        hidden: int,
        layers: int,
        layerwise: bool = True,
        conditions: tuple[tuple[str, tuple[str, ...]], ...] = (),
    ):
        super().__init__()
        if count < 1:
            raise ValueError("register count must be positive")
        if len(conditions) > count:
            raise ValueError(f"{len(conditions)} conditions need at least that many registers, got {count}")
        for name, vocabulary in conditions:
            if not vocabulary or vocabulary[0] != UNKNOWN_CONDITION:
                raise ValueError(f"condition {name!r} must list {UNKNOWN_CONDITION!r} first")
            if len(set(vocabulary)) != len(vocabulary):
                raise ValueError(f"condition {name!r} has a duplicated value")
        self.count = count
        self.layerwise = layerwise
        # Small init: registers start close to nothing so an untrained run reproduces the
        # no-register model up to the extra positions, which keeps the contrast honest.
        self.shared = nn.Parameter(torch.randn(count, hidden) * 0.02)
        self.layer_delta = nn.Parameter(torch.zeros(layers, count, hidden)) if layerwise else None
        # Condition i occupies register slot i and contributes a residual on top of that
        # slot's shared vector, rather than replacing it. Zero init means an unseen or
        # undertrained condition degrades exactly to the shared register instead of to
        # noise, which is what makes a rare convention class safe to add.
        #
        # The vocabularies live here, not only in the caller, because a checkpoint that
        # records a cardinality but not the values cannot be scored: nothing would say
        # which id a row's source or language maps to at inference.
        self.condition_names = tuple(name for name, _ in conditions)
        self.condition_vocabularies = {name: tuple(values) for name, values in conditions}
        self.condition_residuals = nn.ModuleDict(
            {name: nn.Embedding(len(values), hidden) for name, values in conditions}
        )
        for residual in self.condition_residuals.values():
            nn.init.zeros_(residual.weight)
        self._condition_ids: dict[str, torch.Tensor] = {}

    def set_conditions(self, ids: dict[str, torch.Tensor]) -> None:
        """Supply this batch's conditions: ids of shape [batch], or [batch, classes] weights.

        Held on the instance rather than passed through, because the values are consumed
        inside an embedding forward hook that receives no per-row context.
        """
        missing = set(self.condition_names) - set(ids)
        if missing:
            raise ValueError(f"missing condition ids for {sorted(missing)}")
        for name in self.condition_names:
            value = ids[name]
            width = self.condition_residuals[name].num_embeddings
            if value.dim() == 2 and value.shape[1] != width:
                raise ValueError(
                    f"condition {name!r} expects a distribution of width {width}, got {value.shape[1]}"
                )
            if value.dim() > 2:
                raise ValueError(f"condition {name!r} takes ids or one distribution per row")
        self._condition_ids = {name: ids[name] for name in self.condition_names}

    def clear_conditions(self) -> None:
        self._condition_ids = {}

    def write_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Replace positions 1..count with the learned register vectors.

        Position 0 is the leading special token; the dataset places the register
        placeholders immediately after it, so the slice is fixed and needs no mask.
        """
        if embeddings.size(1) < self.count + 1:
            raise ValueError(
                f"sequence of {embeddings.size(1)} is shorter than {self.count} registers plus a "
                "leading special token"
            )
        batch = embeddings.size(0)
        prefix = self.shared.to(embeddings.dtype).unsqueeze(0).expand(batch, -1, -1)
        if self.condition_names:
            if not self._condition_ids:
                raise ValueError(
                    "conditioned registers need set_conditions() before every forward; "
                    f"expected {list(self.condition_names)}"
                )
            prefix = prefix.clone()
            for slot, name in enumerate(self.condition_names):
                value = self._condition_ids[name].to(embeddings.device)
                weight = self.condition_residuals[name].weight
                if value.dim() == 2:
                    # A distribution over the vocabulary rather than a choice from it: the
                    # residual is the posterior-weighted mixture of the class vectors. A
                    # hard id is the special case where that distribution is one-hot, so
                    # both paths share one parameter set and mean the same thing.
                    residual = value.to(weight.dtype) @ weight
                else:
                    residual = self.condition_residuals[name](value)
                prefix[:, slot] = prefix[:, slot] + residual.to(embeddings.dtype)
        return torch.cat([embeddings[:, :1], prefix, embeddings[:, 1 + self.count :]], dim=1)

    def add_layer_delta(self, hidden_states: torch.Tensor, layer: int) -> torch.Tensor:
        if self.layer_delta is None:
            return hidden_states
        delta = self.layer_delta[layer].to(hidden_states.dtype).unsqueeze(0)
        head = hidden_states[:, :1]
        registers = hidden_states[:, 1 : 1 + self.count] + delta
        tail = hidden_states[:, 1 + self.count :]
        return torch.cat([head, registers, tail], dim=1)


def encoder_blocks(model) -> nn.ModuleList:
    """The transformer blocks of a token-classification model, by the usual layout."""
    base = model.base_model
    for path in (("encoder", "layer"), ("encoder", "layers"), ("layers",)):
        node = base
        for name in path:
            node = getattr(node, name, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList):
            return node
    raise ValueError(f"cannot locate encoder blocks on {type(base).__name__}")


def install(
    model,
    count: int,
    layerwise: bool = True,
    conditions: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> SoftRegisters:
    """Attach registers to a token-classification model and hook them into its forward.

    Returns the module so the caller can put it in its own parameter group; it is also
    registered on the model, so an ordinary checkpoint save carries it.

    With ``conditions``, a forward pre-hook lifts each condition's ids out of the call's
    inputs before the embedding hook needs them. Doing it here rather than in the trainer
    means training, evaluation, and the standalone evaluator all take the same path, and a
    caller that forgets to supply ids fails loudly in `write_embeddings` instead of
    silently scoring an unconditioned model.
    """
    blocks = encoder_blocks(model)
    embeddings = model.base_model.get_input_embeddings()
    registers = SoftRegisters(count, model.config.hidden_size, len(blocks), layerwise, conditions)
    registers.to(next(model.parameters()).device)
    model.add_module("pii_soft_registers", registers)
    model.config.update(
        {
            REGISTER_CONFIG_KEY: count,
            REGISTER_LAYERWISE_KEY: bool(layerwise),
            REGISTER_CONDITIONS_KEY: [[name, list(values)] for name, values in conditions],
        }
    )

    def write_registers(_module, _inputs, output):
        return registers.write_embeddings(output)

    handles = [embeddings.register_forward_hook(write_registers)]

    for index, block in enumerate(blocks):

        def add_delta(_module, _inputs, output, index=index):
            # Blocks return either a tensor or a tuple whose first entry is the states.
            if isinstance(output, tuple):
                return (registers.add_layer_delta(output[0], index), *output[1:])
            return registers.add_layer_delta(output, index)

        handles.append(block.register_forward_hook(add_delta))

    if conditions:

        def take_condition_ids(_module, args, kwargs):
            # The model's forward swallows unknown keywords, so without this the ids would
            # be dropped in silence and the registers would score unconditioned.
            ids = {}
            for name, _values in conditions:
                key = f"{name}_id"
                if key in kwargs:
                    ids[name] = kwargs.pop(key)
            if ids:
                registers.set_conditions(ids)
            return args, kwargs

        handles.append(model.register_forward_pre_hook(take_condition_ids, with_kwargs=True))

    # nn.Module.__setattr__ only accepts tensors and modules, so the handles live in the
    # instance dict; they must outlive this call or the hooks are silently removed.
    object.__setattr__(registers, "hook_handles", handles)
    return registers


def conditions_from_config(config) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The condition vocabularies a checkpoint declares, in the slot order it trained with."""
    declared = getattr(config, REGISTER_CONDITIONS_KEY, None) or ()
    return tuple((str(name), tuple(str(value) for value in values)) for name, values in declared)


def condition_ids(vocabulary: tuple[str, ...], values, device=None) -> torch.Tensor:
    """Map raw condition values onto ids, sending anything unseen to `unknown`.

    Unseen values fall back rather than raising because the vocabulary is fixed at
    training time: a new annotation source appearing later must score as `unknown`,
    which is the condition production uses anyway.
    """
    index = {value: position for position, value in enumerate(vocabulary)}
    return torch.tensor([index.get(value, 0) for value in values], dtype=torch.long, device=device)


def reserve_positions(input_ids: list[int], offsets: list[tuple[int, int]], count: int, filler: int):
    """Insert `count` placeholder positions after the leading special token.

    The offsets are zero width, which is how this trainer already marks special tokens,
    so every per-token supervision channel skips the registers without knowing they
    exist. The placeholder ids are never read: their embeddings are overwritten.
    """
    if count < 1:
        return input_ids, offsets
    return (
        [*input_ids[:1], *([filler] * count), *input_ids[1:]],
        [*offsets[:1], *([(0, 0)] * count), *offsets[1:]],
    )


def placeholder_id(tokenizer) -> int:
    """The token id the reserved slots hold before their embeddings are overwritten.

    Training and inference must agree on this only in the sense that both must choose a
    real id the tokenizer knows; the value is never read by the model.
    """
    for candidate in (tokenizer.mask_token_id, tokenizer.unk_token_id):
        if candidate is not None:
            return int(candidate)
    return 0


def reserve_window(model_inputs: dict[str, list[int]], offsets, count: int, filler: int):
    """Apply `reserve_positions` to one tokenized window, keeping every field aligned.

    Unknown fields raise rather than pass through at the wrong length: a silently
    mismatched `token_type_ids` would misalign every position after the registers.
    """
    if count < 1:
        return model_inputs, offsets
    unknown = set(model_inputs) - {"input_ids", "attention_mask", "token_type_ids"}
    if unknown:
        raise ValueError(f"cannot reserve register slots in unknown tokenizer fields: {sorted(unknown)}")
    input_ids, offsets = reserve_positions(model_inputs["input_ids"], offsets, count, filler)
    reserved = {"input_ids": input_ids}
    if "attention_mask" in model_inputs:
        reserved["attention_mask"] = [1] * count + list(model_inputs["attention_mask"])
    if "token_type_ids" in model_inputs:
        # Registers belong to the first segment, which is what position 0 already is.
        head = model_inputs["token_type_ids"][:1]
        reserved["token_type_ids"] = [*head, *(head * count), *model_inputs["token_type_ids"][1:]]
    return reserved, offsets
