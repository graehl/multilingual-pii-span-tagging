#!/usr/bin/env python3
"""Learned prompt slots spliced into an encoder tagger's input at fixed anchors.

The tag-status pilot prepended per-type "maybe" slots and training-only status
slots to the input embeddings. This module does the same inside the standard
trainer, with each slot anchored where the layout puts it:

  start          before the leading special token
  before_target  just before the target sentence, after any preceding context
  after_target   just after the target sentence

A slot is an input embedding the encoder attends to and never scores. A
training-only slot is hidden from attention outside training, and in training
while its gold status is unknown; hiding never moves a position, so inference
sees the text where training saw it. Outputs are gathered back to the original
tokens, so labels and every per-token head keep their alignment.
"""

from __future__ import annotations

import torch
from torch import nn

CONFIG_KEY = "pii_prompt_slots"
ANCHORS = ("start", "before_target", "after_target")
STATUS_VALUES = ("unknown", "absent", "present")
ROLES = ("maybe", "status", "neutral", "language", "language_training")
PER_TOKEN_OUTPUTS = (
    "logits",
    "secondary_logits",
    "predicate_logits",
    "subclass_logits",
    "reference_type_logits",
)
# English names initialize language slots, as type names initialize type slots.
LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish", "fil": "Filipino", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sv": "Swedish", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}  # fmt: skip


def layout_slots(types: tuple[str, ...], layout: str, gold: bool, language: bool) -> tuple[dict, ...]:
    """Slot specs, in sequence order, for a named layout.

    Every type gets a visible "maybe" slot and a training-only slot holding its gold
    status (gold) or a plain learned vector (neutral control). With ``language`` a
    visible and a training-only language slot join the maybe and training-only groups.

      maybe-first     maybe | context | target | training-only
      gold-first      training-only | context | maybe | target
      adjacent-pairs  context | (maybe, training-only) pairs | target
      grouped         maybe | training-only | context | target (the pilot's prefix)
    """
    hidden = "status" if gold else "neutral"
    maybe = [{"role": "maybe", "label": name, "training_only": False} for name in types]
    hide = [{"role": hidden, "label": name, "training_only": True} for name in types]
    if language:
        maybe.insert(0, {"role": "language", "label": "language", "training_only": False})
        hide.insert(0, {"role": "language_training", "label": "language", "training_only": True})
    if layout == "maybe-first":
        placed = [(s, "start") for s in maybe] + [(s, "after_target") for s in hide]
    elif layout == "gold-first":
        placed = [(s, "start") for s in hide] + [(s, "before_target") for s in maybe]
    elif layout == "adjacent-pairs":
        placed = [(s, "before_target") for pair in zip(maybe, hide, strict=True) for s in pair]
    elif layout == "grouped":
        placed = [(s, "start") for s in (*maybe, *hide)]
    else:
        raise ValueError(f"unknown prompt-slot layout {layout!r}")
    return tuple({**spec, "anchor": anchor} for spec, anchor in placed)


class PromptSlots(nn.Module):
    def __init__(
        self,
        hidden: int,
        slots: tuple[dict, ...],
        types: tuple[str, ...],
        languages: tuple[str, ...] = (),
        initial: torch.Tensor | None = None,
        language_initial: torch.Tensor | None = None,
    ):
        super().__init__()
        ranks = [ANCHORS.index(slot["anchor"]) for slot in slots]
        if ranks != sorted(ranks):
            raise ValueError("slots must be listed in sequence order: start, before, after the target")
        if any(slot["role"] not in ROLES for slot in slots):
            raise ValueError(f"slot roles must be among {ROLES}")
        self.slots = tuple(dict(slot) for slot in slots)
        self.types = tuple(types)
        self.languages = tuple(languages)
        uses_language = any(slot["role"].startswith("language") for slot in slots)
        if uses_language != bool(self.languages):
            raise ValueError(
                "language slots need a language inventory, and an inventory needs language slots"
            )
        self.type_index = {name: i for i, name in enumerate(self.types)}
        for slot in self.slots:
            if slot["role"] in ("maybe", "status", "neutral") and slot["label"] not in self.type_index:
                raise ValueError(f"slot label {slot['label']!r} is not a primary type")
        # One learned vector per slot. A status slot adds a zero-initialized residual
        # for absent or present, so each status starts from its type's vector.
        if initial is None:
            initial = torch.zeros(len(slots), hidden)
        if initial.shape != (len(slots), hidden):
            raise ValueError(f"slot initialization must be {(len(slots), hidden)}")
        self.vectors = nn.Parameter(initial.detach().clone().float())
        self.status_slots = tuple(i for i, slot in enumerate(self.slots) if slot["role"] == "status")
        self.status_residual = nn.Parameter(torch.zeros(len(self.status_slots), len(STATUS_VALUES), hidden))
        self.language_slots = tuple(
            i for i, slot in enumerate(self.slots) if slot["role"].startswith("language")
        )
        if self.languages:
            if language_initial is None or language_initial.shape != (len(self.languages), hidden):
                raise ValueError("language slots need one initialization row per language")
            table = language_initial.detach().clone().float()
            self.language_vectors = nn.Parameter(table.unsqueeze(0).repeat(len(self.language_slots), 1, 1))
        self._keep: torch.Tensor | None = None

    @property
    def width(self) -> int:
        return len(self.slots)

    def slot_vectors(self, batch: int, statuses, language_ids, device, dtype) -> torch.Tensor:
        vectors = self.vectors.to(device).unsqueeze(0).expand(batch, -1, -1).clone()
        if self.status_slots:
            if statuses is None:
                statuses = torch.zeros(batch, len(self.types), dtype=torch.long, device=device)
            for k, slot in enumerate(self.status_slots):
                values = statuses[:, self.type_index[self.slots[slot]["label"]]]
                vectors[:, slot] = vectors[:, slot] + self.status_residual[k][values]
        for k, slot in enumerate(self.language_slots):
            table = self.language_vectors[k]
            rows = table[language_ids.clamp_min(0)]
            vectors[:, slot] = torch.where((language_ids == -1)[:, None], table.mean(0), rows)
        return vectors.to(dtype)

    def slot_visibility(self, batch: int, statuses, language_ids, training: bool, device) -> torch.Tensor:
        visible = torch.ones(batch, self.width, dtype=torch.long, device=device)
        for i, slot in enumerate(self.slots):
            if not slot["training_only"]:
                continue
            if not training:
                visible[:, i] = 0
            elif slot["role"] == "status":
                visible[:, i] = statuses[:, self.type_index[slot["label"]]].ne(0).long()
            elif slot["role"] == "language_training":
                visible[:, i] = language_ids.ne(-1).long()
        return visible

    def splice(
        self, words, attention_mask, target_start, target_end, statuses, language_ids, training, pad_id
    ):
        """Insert the slots; return embeddings, mask, positions and the original tokens' new indices."""
        batch, length, _ = words.shape
        device = words.device
        anchors = {
            "start": torch.zeros_like(target_start),
            "before_target": target_start,
            "after_target": target_end + 1,
        }
        # Slot i sits before original index anchor_i, after the i slots listed before it.
        insert_before = torch.stack([anchors[slot["anchor"]] for slot in self.slots], dim=1)
        slot_index = insert_before + torch.arange(self.width, device=device)
        original = torch.arange(length, device=device).expand(batch, -1)
        shift = (insert_before.unsqueeze(1) <= original.unsqueeze(2)).sum(dim=2)
        keep = original + shift
        total = length + self.width
        rows = torch.arange(batch, device=device).unsqueeze(1)
        embeds = words.new_zeros(batch, total, words.size(2))
        embeds[rows, keep] = words
        embeds[rows, slot_index] = self.slot_vectors(batch, statuses, language_ids, device, words.dtype)
        mask = attention_mask.new_zeros(batch, total)
        mask[rows, keep] = attention_mask
        mask[rows, slot_index] = self.slot_visibility(batch, statuses, language_ids, training, device).to(
            mask.dtype
        )
        # Every real position, hidden slots included, keeps a stable index, so hiding a
        # hint never moves the text; padding takes the pad position as usual.
        occupied = torch.zeros(batch, total, dtype=torch.bool, device=device)
        occupied[rows, keep] = attention_mask.bool()
        occupied[rows, slot_index] = True
        positions = occupied.long().cumsum(dim=1) * occupied.long() + pad_id
        return embeds, mask, positions, keep


def install(model, prompt: PromptSlots) -> PromptSlots:
    """Attach prompt slots to a token classifier and splice them into its forward."""
    model.add_module("pii_prompt_slots", prompt)
    model.config.update(
        {
            CONFIG_KEY: {
                "slots": [dict(slot) for slot in prompt.slots],
                "types": list(prompt.types),
                "languages": list(prompt.languages),
            }
        }
    )
    embeddings = model.base_model.get_input_embeddings()
    status_keys = [f"status_{name}_id" for name in prompt.types]

    def splice_inputs(module, args, kwargs):
        if args:
            raise ValueError("prompt slots need keyword inputs")
        if kwargs.get("labels") is not None:
            raise ValueError("prompt slots gather outputs after the forward; score labels outside the model")
        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs["attention_mask"]
        start = kwargs.pop("prompt_target_start")
        end = kwargs.pop("prompt_target_end")
        statuses = None
        if all(key in kwargs for key in status_keys):
            statuses = torch.stack([kwargs.pop(key) for key in status_keys], dim=1)
        else:
            for key in status_keys:
                kwargs.pop(key, None)
        language_ids = kwargs.pop("prompt_language_id", None)
        if prompt.languages and language_ids is None:
            raise ValueError("language prompt slots need prompt_language_id")
        if module.training and prompt.status_slots and statuses is None:
            raise ValueError("gold status slots need every status id in training")
        embeds, mask, positions, keep = prompt.splice(
            embeddings(input_ids),
            attention_mask,
            start,
            end,
            statuses,
            language_ids,
            module.training,
            module.config.pad_token_id,
        )
        prompt._keep = keep
        kwargs.update(input_ids=None, inputs_embeds=embeds, attention_mask=mask, position_ids=positions)
        return args, kwargs

    def gather_outputs(_module, _args, output):
        keep = prompt._keep
        rows = torch.arange(keep.size(0), device=keep.device).unsqueeze(1)
        for key in PER_TOKEN_OUTPUTS:
            value = getattr(output, key, None)
            if value is not None:
                output[key] = value[rows, keep]
        if getattr(output, "hidden_states", None) is not None:
            output.hidden_states = tuple(state[rows, keep] for state in output.hidden_states)
        return output

    handles = [
        model.register_forward_pre_hook(splice_inputs, with_kwargs=True),
        model.register_forward_hook(gather_outputs),
    ]
    object.__setattr__(prompt, "hook_handles", handles)
    return prompt


def from_config(config, hidden: int) -> PromptSlots | None:
    """Rebuild the module a checkpoint declares (parameters load from its state dict)."""
    spec = getattr(config, CONFIG_KEY, None)
    if spec is None:
        return None
    languages = tuple(spec["languages"])
    return PromptSlots(
        hidden,
        tuple(dict(slot) for slot in spec["slots"]),
        tuple(spec["types"]),
        languages,
        language_initial=torch.zeros(len(languages), hidden) if languages else None,
    )


def target_bounds(offsets) -> tuple[int, int]:
    """First and last token index of the target text: the only non-zero-width offsets.

    A target with no tokens (an empty or content-free segment, which training data
    and serving inputs both contain) is an empty span just before the trailing
    special token: both target-relative anchors then fall there, (n-1, n-2).
    """
    if len(offsets) < 2:
        raise ValueError("prompt slots need at least the leading and trailing special tokens")
    target = [i for i, (a, b) in enumerate(offsets) if b > a]
    if not target:
        return len(offsets) - 1, len(offsets) - 2
    return target[0], target[-1]
