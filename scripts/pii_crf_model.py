"""mmBERT token classifier with an efficient, constrained BIOES CRF.

The unified PII tagset has hundreds of composite BIOES labels.  A dense CRF
would perform O(labels^2) work at every token and learn mostly meaningless
cross-entity-type bigrams.  This CRF is the exact linear-chain model induced by
factorizing transitions into:

- a learned 5x5 boundary-state transition matrix (O/B/I/E/S), and
- a hard constraint that I/E may only continue the current entity type.

The partition function and Viterbi decoder are therefore O(tokens * labels).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

PREFIXES = ("O", "B", "I", "E", "S")
PREFIX_TO_ID = {prefix: index for index, prefix in enumerate(PREFIXES)}
O, B, I, E, S = range(len(PREFIXES))
COMPLETE_PREFIXES = (O, E, S)
START_PREFIXES = (O, B, S)
END_PREFIXES = (O, E, S)
INSIDE_PREFIXES = (B, I)
CONTINUE_PREFIXES = (I, E)
INVALID_SCORE = -10_000.0
CRF_ARCHITECTURE = "factorized_bioes_crf"


def _label_parts(label: str) -> tuple[int, str | None]:
    if label == "O":
        return O, None
    prefix, entity_type = label.split("-", 1)
    if prefix not in PREFIX_TO_ID or prefix == "O":
        raise ValueError(f"invalid BIOES label: {label}")
    return PREFIX_TO_ID[prefix], entity_type


class FactorizedBIOESCRF(nn.Module):
    """Typed BIOES linear-chain CRF with shared boundary transitions."""

    def __init__(self, label_names: list[str]) -> None:
        super().__init__()
        if not label_names or label_names[0] != "O":
            raise ValueError("BIOES CRF requires O as label 0")

        type_names = sorted({_label_parts(label)[1] for label in label_names if label != "O"})
        if not type_names:
            raise ValueError("BIOES CRF requires at least one entity type")
        type_to_id = {name: index for index, name in enumerate(type_names)}

        prefix_ids = []
        type_ids = []
        by_prefix_and_type: dict[tuple[int, int], int] = {}
        for label_id, label in enumerate(label_names):
            prefix, entity_type = _label_parts(label)
            type_id = -1 if entity_type is None else type_to_id[entity_type]
            prefix_ids.append(prefix)
            type_ids.append(type_id)
            if prefix != O:
                key = (prefix, type_id)
                if key in by_prefix_and_type:
                    raise ValueError(f"duplicate BIOES label for {label}")
                by_prefix_and_type[key] = label_id

        for type_name, type_id in type_to_id.items():
            missing = [
                PREFIXES[prefix] for prefix in (B, I, E, S) if (prefix, type_id) not in by_prefix_and_type
            ]
            if missing:
                raise ValueError(f"{type_name} lacks BIOES labels: {', '.join(missing)}")

        self.label_names = label_names
        self.num_labels = len(label_names)
        self.num_types = len(type_names)
        self.transitions = nn.Parameter(torch.zeros(len(PREFIXES), len(PREFIXES)))
        self.start_transitions = nn.Parameter(torch.zeros(len(PREFIXES)))
        self.end_transitions = nn.Parameter(torch.zeros(len(PREFIXES)))
        self.register_buffer("prefix_ids", torch.tensor(prefix_ids, dtype=torch.long), persistent=False)
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long), persistent=False)
        self.register_buffer(
            "complete_prefix_ids",
            torch.tensor(COMPLETE_PREFIXES, dtype=torch.long),
            persistent=False,
        )
        for prefix, name in ((B, "b_ids"), (I, "i_ids"), (E, "e_ids"), (S, "s_ids")):
            ids = [by_prefix_and_type[(prefix, type_id)] for type_id in range(self.num_types)]
            self.register_buffer(name, torch.tensor(ids, dtype=torch.long), persistent=False)

    def _start_scores(self) -> torch.Tensor:
        scores = self.start_transitions[self.prefix_ids]
        valid = torch.zeros_like(scores, dtype=torch.bool)
        for prefix in START_PREFIXES:
            valid |= self.prefix_ids == prefix
        return scores.masked_fill(~valid, INVALID_SCORE)

    def _end_scores(self) -> torch.Tensor:
        scores = self.end_transitions[self.prefix_ids]
        valid = torch.zeros_like(scores, dtype=torch.bool)
        for prefix in END_PREFIXES:
            valid |= self.prefix_ids == prefix
        return scores.masked_fill(~valid, INVALID_SCORE)

    def _transition_scores(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        previous_prefix = self.prefix_ids[previous]
        current_prefix = self.prefix_ids[current]
        previous_type = self.type_ids[previous]
        current_type = self.type_ids[current]

        starts_new = ((previous_prefix == O) | (previous_prefix == E) | (previous_prefix == S)) & (
            (current_prefix == O) | (current_prefix == B) | (current_prefix == S)
        )
        continues = (
            ((previous_prefix == B) | (previous_prefix == I))
            & ((current_prefix == I) | (current_prefix == E))
            & (previous_type == current_type)
        )
        scores = self.transitions[previous_prefix, current_prefix]
        return scores.masked_fill(~(starts_new | continues), INVALID_SCORE)

    @staticmethod
    def _validate_inputs(emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> None:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [batch, time, labels]")
        if tags.shape != emissions.shape[:2] or mask.shape != tags.shape:
            raise ValueError("tags and mask must match emissions batch/time dimensions")
        if not torch.all(mask[:, 0]):
            raise ValueError("every CRF sequence must contain at least one token")
        if torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("CRF mask must be contiguous")

    def _closed_group_scores(self, alpha: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                alpha[:, 0],
                torch.logsumexp(alpha[:, self.e_ids], dim=1),
                torch.logsumexp(alpha[:, self.s_ids], dim=1),
            ),
            dim=1,
        )

    def _advance(self, alpha: torch.Tensor, emission: torch.Tensor) -> torch.Tensor:
        next_alpha = torch.empty_like(alpha)
        closed = self._closed_group_scores(alpha)

        for target_prefix, target_ids in ((O, None), (B, self.b_ids), (S, self.s_ids)):
            base = torch.logsumexp(
                closed + self.transitions[self.complete_prefix_ids, target_prefix].unsqueeze(0),
                dim=1,
            )
            if target_ids is None:
                next_alpha[:, 0] = base + emission[:, 0]
            else:
                next_alpha[:, target_ids] = base.unsqueeze(1) + emission[:, target_ids]

        for target_prefix, target_ids in ((I, self.i_ids), (E, self.e_ids)):
            within = torch.stack(
                (
                    alpha[:, self.b_ids] + self.transitions[B, target_prefix],
                    alpha[:, self.i_ids] + self.transitions[I, target_prefix],
                ),
                dim=2,
            )
            next_alpha[:, target_ids] = torch.logsumexp(within, dim=2) + emission[:, target_ids]
        return next_alpha

    def _log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        alpha = emissions[:, 0] + self._start_scores()
        for timestep in range(1, emissions.shape[1]):
            advanced = self._advance(alpha, emissions[:, timestep])
            alpha = torch.where(mask[:, timestep].unsqueeze(1), advanced, alpha)
        return torch.logsumexp(alpha + self._end_scores(), dim=1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_tags = tags.masked_fill(~mask, 0)
        emission_scores = emissions.gather(2, safe_tags.unsqueeze(2)).squeeze(2)
        score = (emission_scores * mask).sum(dim=1)
        score += self._start_scores()[safe_tags[:, 0]]
        for timestep in range(1, emissions.shape[1]):
            transition = self._transition_scores(safe_tags[:, timestep - 1], safe_tags[:, timestep])
            score += transition * mask[:, timestep]
        lengths = mask.long().sum(dim=1)
        last_tags = safe_tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        return score + self._end_scores()[last_tags]

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(emissions, tags, mask)
        emissions = emissions.float()
        return (self._log_partition(emissions, mask) - self._gold_score(emissions, tags, mask)).mean()

    def _advance_viterbi(
        self,
        score: torch.Tensor,
        emission: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_score = torch.empty_like(score)
        backpointer = torch.empty(self.num_labels, dtype=torch.long, device=score.device)

        group_scores = torch.stack(
            (
                score[0],
                score[self.e_ids].max(),
                score[self.s_ids].max(),
            )
        )
        group_sources = torch.stack(
            (
                torch.zeros((), dtype=torch.long, device=score.device),
                self.e_ids[score[self.e_ids].argmax()],
                self.s_ids[score[self.s_ids].argmax()],
            )
        )
        for target_prefix, target_ids in ((O, None), (B, self.b_ids), (S, self.s_ids)):
            candidates = group_scores + self.transitions[self.complete_prefix_ids, target_prefix]
            best_group = candidates.argmax()
            source = group_sources[best_group]
            if target_ids is None:
                next_score[0] = candidates[best_group] + emission[0]
                backpointer[0] = source
            else:
                next_score[target_ids] = candidates[best_group] + emission[target_ids]
                backpointer[target_ids] = source

        for target_prefix, target_ids in ((I, self.i_ids), (E, self.e_ids)):
            candidates = torch.stack(
                (
                    score[self.b_ids] + self.transitions[B, target_prefix],
                    score[self.i_ids] + self.transitions[I, target_prefix],
                ),
                dim=1,
            )
            choices = candidates.argmax(dim=1)
            next_score[target_ids] = candidates.gather(1, choices.unsqueeze(1)).squeeze(1)
            next_score[target_ids] += emission[target_ids]
            backpointer[target_ids] = torch.where(choices == 0, self.b_ids, self.i_ids)
        return next_score, backpointer

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        if emissions.ndim != 3 or mask.shape != emissions.shape[:2]:
            raise ValueError("decode expects emissions [batch,time,labels] and matching mask")
        if not torch.all(mask[:, 0]) or torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("decode mask must be non-empty and contiguous")

        paths = []
        for batch_index in range(emissions.shape[0]):
            length = int(mask[batch_index].sum())
            sequence = emissions[batch_index, :length].float()
            score = sequence[0] + self._start_scores()
            backpointers = []
            for timestep in range(1, length):
                score, backpointer = self._advance_viterbi(score, sequence[timestep])
                backpointers.append(backpointer)
            last = int((score + self._end_scores()).argmax())
            path = [last]
            for backpointer in reversed(backpointers):
                last = int(backpointer[last])
                path.append(last)
            paths.append(list(reversed(path)))
        return paths


class MMBertCrfForTokenClassification(PreTrainedModel):
    """AutoModel encoder, linear emissions, and factorized BIOES CRF."""

    base_model_prefix = "encoder"
    main_input_name = "input_ids"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    supports_gradient_checkpointing = True

    def __init__(self, config, encoder: nn.Module | None = None) -> None:
        super().__init__(config)
        label_names = [config.id2label[index] for index in range(config.num_labels)]
        self.num_labels = config.num_labels
        self.encoder = encoder if encoder is not None else AutoModel.from_config(config)
        dropout = float(getattr(config, "pii_classifier_dropout", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = FactorizedBIOESCRF(label_names)
        self._init_weights(self.classifier)

    @classmethod
    def from_encoder_pretrained(
        cls,
        model_name: str,
        label_names: list[str],
        *,
        dtype: torch.dtype = torch.bfloat16,
        dropout: float = 0.1,
    ) -> MMBertCrfForTokenClassification:
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = len(label_names)
        config.id2label = dict(enumerate(label_names))
        config.label2id = {label: index for index, label in enumerate(label_names)}
        config.architectures = [cls.__name__]
        config.pii_decoder = CRF_ARCHITECTURE
        config.pii_encoder_model = model_name
        config.pii_classifier_dropout = dropout
        encoder = AutoModel.from_pretrained(model_name, config=config, dtype=dtype)
        return cls(config, encoder=encoder)

    @classmethod
    def from_local_checkpoint(
        cls,
        path: str | Path,
        *,
        encoder_dtype: torch.dtype | None = None,
    ) -> MMBertCrfForTokenClassification:
        path = Path(path)
        config = AutoConfig.from_pretrained(path)
        if getattr(config, "pii_decoder", None) != CRF_ARCHITECTURE:
            raise ValueError(f"{path} is not a {CRF_ARCHITECTURE} checkpoint")
        model = cls(config)
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
        if encoder_dtype is not None:
            model.encoder.to(dtype=encoder_dtype)
            model.classifier.to(dtype=encoder_dtype)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **_kwargs,
    ) -> TokenClassifierOutput:
        del token_type_ids
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        loss = None
        if labels is not None:
            crf_mask = labels != -100
            safe_labels = labels.masked_fill(~crf_mask, 0)
            compact_emissions, compact_labels, compact_mask = compact_valid_tokens(
                emissions, safe_labels, crf_mask
            )
            loss = self.crf.neg_log_likelihood(compact_emissions, compact_labels, compact_mask)
        return TokenClassifierOutput(loss=loss, logits=emissions)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        compact_emissions, _, compact_mask = compact_valid_tokens(
            emissions,
            torch.zeros_like(mask, dtype=torch.long),
            mask,
        )
        return self.crf.decode(compact_emissions, compact_mask)


def compact_valid_tokens(
    emissions: torch.Tensor,
    tags: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove special/padded positions and return contiguous CRF sequences."""
    lengths = mask.long().sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("CRF batch contains an empty token sequence")
    max_length = int(lengths.max())
    batch_size, sequence_length, num_labels = emissions.shape
    positions = torch.arange(sequence_length, device=mask.device).expand(batch_size, -1)
    # Valid positions sort before invalid ones while retaining left-to-right
    # order.  This avoids one GPU synchronization and indexed copy per row.
    order = torch.argsort(torch.where(mask, positions, positions + sequence_length), dim=1)
    order = order[:, :max_length]
    compact_emissions = emissions.gather(1, order.unsqueeze(2).expand(-1, -1, num_labels))
    compact_tags = tags.gather(1, order)
    compact_mask = torch.arange(max_length, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    return compact_emissions, compact_tags, compact_mask
