"""Per-type maybe/present/absent embeddings with explicit unknown inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

if __package__:
    from scripts.pii_soft_prompt import encode_prompted
else:
    from pii_soft_prompt import encode_prompted


LAYOUTS = ("grouped", "alternating")
CONFIG_KEY = "pii_tag_status_prompt"
STATE_FILE = "tag_status_prompt.pt"
REFERENCE_TYPES = ("person_reference", "organization_reference")


class TagStatusPrompt(nn.Module):
    """Visible "maybe" slots plus training-only status slots, one pair per type.

    Each type owns a visible slot and a training-only slot. The training-only
    slot holds a gold present/absent embedding (or a shared neutral one) and is
    attention-masked at inference without shifting text positions. With a
    language inventory, a leading pair of language slots precedes the types:
    one visible throughout, one training-only. ``grouped`` places all visible
    slots before all training-only slots; ``alternating`` places each visible
    slot directly before its own training-only slot.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        layout: str = "grouped",
        languages: tuple[str, ...] = (),
        type_initial: torch.Tensor | None = None,
        language_initial: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if base.config.model_type != "xlm-roberta" or base.token_offsets != (0,):
            raise ValueError("Tag status pilot requires current-token XLM-R")
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown prompt layout {layout!r}")
        if len(set(languages)) != len(languages):
            raise ValueError("Duplicate prompt languages")
        self.base = base
        self.layout = layout
        self.languages = tuple(languages)
        self.types = sorted(
            {
                label[2:]
                for label in base.config.id2label.values()
                if label != "O" and label[2:] not in REFERENCE_TYPES
            }
        )
        if not self.types:
            raise ValueError("No active entity types")
        embeddings = base.encoder.get_input_embeddings().weight
        if type_initial is None:
            initial = (
                embeddings[torch.randint(5, len(embeddings), (len(self.types),), device=embeddings.device)]
                .detach()
                .clone()
            )
        else:
            if type_initial.shape != (len(self.types), embeddings.shape[1]):
                raise ValueError("Type initialization must have one row per type")
            initial = type_initial.detach().clone().to(embeddings)
        self.maybe = nn.Parameter(initial)
        self.present = nn.Parameter(initial.clone())
        self.absent = nn.Parameter(initial.clone())
        self.neutral = nn.Parameter(initial.clone())
        if self.languages:
            if language_initial is None or language_initial.shape != (
                len(self.languages),
                embeddings.shape[1],
            ):
                raise ValueError("Language slots need one initialization row per language")
            language_initial = language_initial.detach().clone().to(embeddings)
            self.language_visible = nn.Parameter(language_initial)
            self.language_training = nn.Parameter(language_initial.clone())
        self.register_buffer(
            "reference_columns",
            torch.tensor(
                [i for i, label in base.config.id2label.items() if label[2:] in REFERENCE_TYPES],
                dtype=torch.long,
                device=embeddings.device,
            ),
        )
        self.register_buffer(
            "type_by_label",
            torch.tensor(
                [
                    self.types.index(base.config.id2label[i][2:])
                    if base.config.id2label[i][2:] in self.types
                    else -1
                    for i in range(base.num_labels)
                ],
                device=embeddings.device,
            ),
        )

    def gold_status(self, targets: torch.Tensor) -> torch.Tensor:
        """Only complete flat targets may call this; -100 is padding/special."""
        mapped = self.type_by_label[targets.clamp_min(0)].masked_fill(targets == -100, -1)
        return torch.stack([(mapped == i).any(1) for i in range(len(self.types))], dim=1).long()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        routes: torch.Tensor | None = None,
        statuses: torch.Tensor | None = None,
        neutral: bool = False,
        output_hidden_states: bool = False,
        language_ids: torch.Tensor | None = None,
        bias_language_ids: torch.Tensor | None = None,
    ):
        """Training passes ``statuses``; inference omits them, masking every training-only slot.

        ``language_ids`` index this prompt's language slots; ``bias_language_ids``
        index the base head's own language-bias inventory and pass through to it.
        """
        count, batch = len(self.types), len(input_ids)
        training = statuses is not None
        if statuses is None:
            statuses = input_ids.new_full((batch, count), -1)
        if (
            statuses.shape != (batch, count)
            or statuses.dtype != torch.long
            or torch.any((statuses < -1) | (statuses > 1))
        ):
            raise ValueError("Each tag status must be unknown(-1), absent(0) or present(1)")
        maybe = self.maybe.unsqueeze(0).expand(batch, -1, -1)
        certainty = torch.where((statuses == 1)[:, :, None], self.present, self.absent)
        if neutral:
            certainty = self.neutral.unsqueeze(0).expand_as(certainty)
        visible_mask = attention_mask.new_ones(batch, count)
        training_mask = statuses.ne(-1).long()
        if self.languages:
            if (
                language_ids is None
                or language_ids.shape != (batch,)
                or language_ids.dtype != torch.long
                or torch.any((language_ids < -1) | (language_ids >= len(self.languages)))
            ):
                raise ValueError("Language slots need one inventory index per row, or -1 for unknown")
            unknown = (language_ids == -1)[:, None]
            rows = language_ids.clamp_min(0)
            # An unlisted language gets the mean learned embedding of its slot.
            visible = torch.where(unknown, self.language_visible.mean(0), self.language_visible[rows])
            hidden = torch.where(unknown, self.language_training.mean(0), self.language_training[rows])
            maybe = torch.cat((visible[:, None], maybe), dim=1)
            certainty = torch.cat((hidden[:, None], certainty), dim=1)
            visible_mask = torch.cat((attention_mask.new_ones(batch, 1), visible_mask), dim=1)
            training_mask = torch.cat(
                (attention_mask.new_full((batch, 1), int(training)), training_mask), dim=1
            )
        elif language_ids is not None:
            raise ValueError("This prompt has no language slots")
        if self.layout == "grouped":
            prefix = torch.cat((maybe, certainty), dim=1)
            prefix_mask = torch.cat((visible_mask, training_mask), dim=1)
        else:
            prefix = torch.stack((maybe, certainty), dim=2).flatten(1, 2)
            prefix_mask = torch.stack((visible_mask, training_mask), dim=2).flatten(1)
        width = prefix.shape[1]
        # Preserve positions across hint removal: unknown certainty slots are
        # invisible attention keys, but still reserve the same text positions.
        positions = torch.arange(1, width + input_ids.shape[1] + 1, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch, -1) + self.base.config.pad_token_id
        positions = positions.clone()
        positions[:, width:] = positions[:, width:].masked_fill(
            ~attention_mask.bool(), self.base.config.pad_token_id
        )
        embedding_dtype = self.base.encoder.get_input_embeddings().weight.dtype
        output = encode_prompted(
            self.base,
            input_ids,
            attention_mask,
            prefix.to(embedding_dtype),
            prefix_mask,
            positions,
            output_hidden_states=output_hidden_states,
            **({} if bias_language_ids is None else {"language_ids": bias_language_ids}),
        )
        if len(self.reference_columns):
            # Reference supervision is disabled: its columns can never win.
            output["logits"] = output.logits.index_fill(-1, self.reference_columns, -1e4)
        return output

    @property
    def prefix_width(self) -> int:
        return 2 * (len(self.types) + bool(self.languages))

    def prompt_parameters(self) -> list[nn.Parameter]:
        return list(self._prompt_tensors().values())

    def save_prompt(self, directory: str | Path) -> None:
        """Store prompt tensors beside a saved base checkpoint and describe them in its config."""
        directory = Path(directory)
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in self._prompt_tensors().items()},
            directory / STATE_FILE,
        )
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config[CONFIG_KEY] = {"layout": self.layout, "languages": list(self.languages), "types": self.types}
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    def _prompt_tensors(self) -> dict[str, torch.Tensor]:
        names = ["maybe", "present", "absent", "neutral"]
        if self.languages:
            names += ["language_visible", "language_training"]
        return {name: getattr(self, name) for name in names}

    @classmethod
    def load_for(cls, base: nn.Module, directory: str | Path) -> TagStatusPrompt:
        """Rebuild the prompt a checkpoint's config declares, on an already loaded base."""
        spec = getattr(base.config, CONFIG_KEY)
        hidden = base.encoder.get_input_embeddings().weight.shape[1]
        # Placeholders only: every prompt tensor is overwritten from the saved state below.
        prompt = cls(
            base,
            layout=spec["layout"],
            languages=tuple(spec["languages"]),
            type_initial=torch.zeros(len(spec["types"]), hidden),
            language_initial=torch.zeros(len(spec["languages"]), hidden) if spec["languages"] else None,
        )
        if prompt.types != spec["types"]:
            raise ValueError("Checkpoint prompt types differ from the loaded head")
        state = torch.load(Path(directory) / STATE_FILE, map_location="cpu", weights_only=True)
        expected = prompt._prompt_tensors()
        if set(state) != set(expected):
            raise ValueError(f"Prompt state keys {sorted(state)} differ from {sorted(expected)}")
        with torch.no_grad():
            for name, tensor in expected.items():
                tensor.copy_(state[name].to(tensor))
        return prompt

    def oracle_logits(self, logits: torch.Tensor, targets: torch.Tensor, policy: str) -> torch.Tensor:
        """Diagnostic only: apply complete gold type presence after encoding."""
        statuses = self.gold_status(targets)
        active = self.type_by_label >= 0
        present = statuses[:, self.type_by_label.clamp_min(0)].bool() & active
        if policy == "gold_mask":
            return logits.masked_fill((active & ~present)[:, None, :], -torch.inf)
        if policy == "positive_bias":
            return logits + present[:, None, :].to(logits.dtype)
        raise ValueError(f"Unknown output hint policy: {policy}")

    def predicted_presence(
        self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fixed uncalibrated rule: any lexical token has type probability >=0.9."""
        valid = attention_mask.bool()
        for token in (
            self.base.config.bos_token_id,
            self.base.config.eos_token_id,
            self.base.config.pad_token_id,
        ):
            valid = valid & input_ids.ne(token)
        probabilities = logits.softmax(-1)
        return torch.stack(
            [
                probabilities[:, :, self.type_by_label == i].sum(-1).masked_fill(~valid, 0).amax(1) >= 0.9
                for i in range(len(self.types))
            ],
            dim=1,
        )

    def positive_logits(self, logits: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        present = presence[:, self.type_by_label.clamp_min(0)] & (self.type_by_label >= 0)
        return logits + present[:, None, :].to(logits.dtype)


def training_status(
    model: TagStatusPrompt, targets: torch.Tensor, condition: str, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, bool]:
    statuses = model.gold_status(targets)
    if condition == "maybe":
        statuses.fill_(-1)
    elif condition == "dropout":
        # Half of individual hints retained; one quarter of examples carry none.
        keep = torch.rand(statuses.shape, device=statuses.device, generator=generator) < 0.5
        keep &= torch.rand((len(statuses), 1), device=statuses.device, generator=generator) >= 0.25
        statuses = statuses.masked_fill(~keep, -1)
    elif condition not in ("gold", "neutral"):
        raise ValueError(f"Unknown status condition {condition!r}")
    return statuses, condition == "neutral"
