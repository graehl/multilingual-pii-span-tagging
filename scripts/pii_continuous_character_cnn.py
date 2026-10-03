#!/usr/bin/env python3
"""Continuous character-CNN components for multilingual PII token tagging.

The encoder runs once over the tokenizer-visible raw character sequence,
including whitespace and punctuation.  Token offsets are used only after the
character sequence has been encoded, when character-center states are pooled
onto the incumbent token lattice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.pii_character_projection import (
    CHARACTER_PADDING_ID,
    CHARACTER_RESERVED_IDS,
    CHARACTER_UNKNOWN_ID,
    CharacterPairProjection,
)


@dataclass(frozen=True)
class ContinuousCharacterInputs:
    """Padded whole-row character inputs and their token alignment."""

    character_ids: torch.Tensor
    character_mask: torch.Tensor
    token_offsets: torch.Tensor


def _content_pair(
    projection: CharacterPairProjection,
    character: str,
    projection_view: str,
) -> tuple[int, int]:
    primary, secondary = projection.pair(character)
    if projection_view == "pair":
        return primary, secondary
    if projection_view == "block-only":
        return primary, CHARACTER_UNKNOWN_ID
    if projection_view == "index-only":
        return CHARACTER_UNKNOWN_ID, secondary
    if projection_view == "constant":
        return CHARACTER_UNKNOWN_ID, CHARACTER_UNKNOWN_ID
    raise ValueError(f"unsupported character projection view {projection_view!r}")


def effective_projection_vocab_sizes(
    projection: CharacterPairProjection,
    projection_view: str,
) -> tuple[int, int]:
    """Return the embedding sizes actually needed by one projection view."""
    if projection_view not in {"pair", "block-only", "index-only", "constant"}:
        raise ValueError(f"unsupported character projection view {projection_view!r}")
    primary = (
        CHARACTER_RESERVED_IDS
        if projection_view in {"index-only", "constant"}
        else projection.primary_vocab_size
    )
    secondary = (
        CHARACTER_RESERVED_IDS
        if projection_view in {"block-only", "constant"}
        else projection.secondary_vocab_size
    )
    return primary, secondary


def project_continuous_characters(
    rows: list[dict],
    offsets: torch.Tensor,
    *,
    projection: CharacterPairProjection,
    projection_view: str = "pair",
) -> ContinuousCharacterInputs:
    """Project each tokenizer-visible raw prefix without deleting inter-token text.

    Character position ``p`` remains position ``p`` from the original string.
    Spaces and punctuation not owned by a tokenizer token therefore remain in
    ``character_ids``. Overlapping offsets are preserved: XLM-R fast
    tokenization can assign one raw character to both a standalone metaspace
    piece and the following content piece.
    """
    if offsets.ndim != 3 or offsets.shape[-1] != 2 or offsets.shape[0] != len(rows):
        raise ValueError("offsets must have shape [rows, tokens, 2]")
    effective_projection_vocab_sizes(projection, projection_view)
    token_count = offsets.shape[1]
    visible_lengths: list[int] = []
    for row_index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"row {row_index} lacks string text")
        row_offsets = offsets[row_index]
        visible_length = int(row_offsets[:, 1].max().item()) if token_count else 0
        if not 0 <= visible_length <= len(text):
            raise ValueError(
                f"row {row_index} tokenizer offset ends at {visible_length} outside {len(text)} characters"
            )
        visible_lengths.append(visible_length)

    padded_characters = max(1, max(visible_lengths, default=0))
    character_ids = torch.zeros(
        len(rows),
        padded_characters,
        2,
        dtype=torch.long,
    )
    character_mask = torch.zeros(len(rows), padded_characters, dtype=torch.bool)
    for row_index, (row, visible_length) in enumerate(zip(rows, visible_lengths, strict=True)):
        text = row["text"]
        pairs = [_content_pair(projection, character, projection_view) for character in text[:visible_length]]
        if pairs:
            character_ids[row_index, :visible_length] = torch.tensor(pairs, dtype=torch.long)
            character_mask[row_index, :visible_length] = True
        for start, end in offsets[row_index].tolist():
            if start == end:
                continue
            if not 0 <= start < end <= visible_length:
                raise ValueError(
                    f"row {row_index} token offset {(start, end)} is outside "
                    f"the {visible_length}-character tokenizer-visible prefix"
                )
    return ContinuousCharacterInputs(
        character_ids=character_ids,
        character_mask=character_mask,
        token_offsets=offsets.to(dtype=torch.long).clone(),
    )


def five_token_receptive_field_target(offsets: torch.Tensor) -> int:
    """Estimate a representative five-token character span from train offsets."""
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError("offsets must have shape [rows, tokens, 2]")
    widths: list[int] = []
    for row_offsets in offsets.tolist():
        content = [(start, end) for start, end in row_offsets if start < end]
        if not content:
            continue
        window = min(5, len(content))
        widths.extend(
            content[start + window - 1][1] - content[start][0] for start in range(len(content) - window + 1)
        )
    if not widths:
        raise ValueError("cannot estimate a receptive field without content-token offsets")
    return max(1, math.ceil(median(widths)))


def convolution_depth_for_receptive_field(target: int, kernel_size: int) -> int:
    """Return the minimum same-padded convolution depth reaching ``target``."""
    if target <= 0:
        raise ValueError("target receptive field must be positive")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel size must be an odd integer greater than one")
    return max(1, math.ceil((target - 1) / (kernel_size - 1)))


class PreNormGatedConvolution(nn.Module):
    """One same-padded pre-LayerNorm residual gated-convolution block."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel size must be an odd integer greater than one")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.normalization = nn.LayerNorm(channels)
        self.input_dropout = nn.Dropout(dropout)
        self.convolution = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        update = self.input_dropout(self.normalization(hidden)).transpose(1, 2)
        update = F.glu(self.convolution(update), dim=1).transpose(1, 2)
        hidden = residual + self.output_dropout(update)
        return hidden.masked_fill(~mask.unsqueeze(-1), 0)


class TypedPairComposer(nn.Module):
    """Collapse alternating typed component symbols into codepoint states.

    The two component streams already use disjoint embedding tables.  The
    multiplicative path lets a codepoint state depend on the particular
    ``(primary, secondary)`` combination without introducing a pair-specific
    lookup table.
    """

    def __init__(
        self,
        component_dim: int,
        channels: int,
        interaction_rank: int,
    ) -> None:
        super().__init__()
        if component_dim <= 0 or channels <= 0 or interaction_rank <= 0:
            raise ValueError("pair-composer dimensions must be positive")
        self.primary_additive = nn.Linear(component_dim, channels, bias=False)
        self.secondary_additive = nn.Linear(component_dim, channels, bias=True)
        self.primary_interaction = nn.Linear(component_dim, interaction_rank, bias=False)
        self.secondary_interaction = nn.Linear(component_dim, interaction_rank, bias=False)
        self.interaction_output = nn.Linear(interaction_rank, channels, bias=False)
        self.interaction_rank = int(interaction_rank)

    def forward(
        self,
        typed_states: torch.Tensor,
        typed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if typed_states.ndim != 3 or typed_states.shape[1] % 2:
            raise ValueError("typed pair states must have shape [batch, 2 * codepoints, channels]")
        if typed_mask.shape != typed_states.shape[:2]:
            raise ValueError("typed pair mask must match the typed-state lattice")
        primary_mask = typed_mask[:, 0::2]
        secondary_mask = typed_mask[:, 1::2]
        if not torch.equal(primary_mask, secondary_mask):
            raise ValueError("each codepoint must contain both typed component symbols")
        primary = typed_states[:, 0::2]
        secondary = typed_states[:, 1::2]
        additive = self.primary_additive(primary) + self.secondary_additive(secondary)
        interaction = self.interaction_output(
            self.primary_interaction(primary) * self.secondary_interaction(secondary)
        )
        composed = additive + interaction
        return composed.masked_fill(~primary_mask.unsqueeze(-1), 0), primary_mask


class ContinuousCharacterCnn(nn.Module):
    """Bidirectional whole-segment character encoder with fixed positions."""

    def __init__(
        self,
        primary_vocab_size: int,
        secondary_vocab_size: int,
        *,
        embedding_dim: int = 64,
        channels: int = 64,
        kernel_size: int = 5,
        depth: int = 6,
        dropout: float = 0.1,
        input_layout: str = "paired-concat",
        pair_interaction_rank: int = 0,
    ) -> None:
        super().__init__()
        if primary_vocab_size <= 0 or secondary_vocab_size <= 0:
            raise ValueError("character vocabulary sizes must be positive")
        if embedding_dim <= 0 or embedding_dim % 2:
            raise ValueError("embedding dimension must be a positive even integer")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_layout not in {"paired-concat", "typed-pair-composer"}:
            raise ValueError(f"unsupported character input layout {input_layout!r}")
        if pair_interaction_rank < 0:
            raise ValueError("pair interaction rank must be nonnegative")
        if input_layout == "paired-concat" and pair_interaction_rank:
            raise ValueError("pair interaction rank applies only to typed-pair-composer")
        component_dim = embedding_dim // 2
        self.primary_vocab_size = int(primary_vocab_size)
        self.secondary_vocab_size = int(secondary_vocab_size)
        self.primary_mask_id = self.primary_vocab_size
        self.secondary_mask_id = self.secondary_vocab_size
        self.primary_embedding = nn.Embedding(
            self.primary_vocab_size + 1,
            component_dim,
            padding_idx=0,
        )
        self.secondary_embedding = nn.Embedding(
            self.secondary_vocab_size + 1,
            component_dim,
            padding_idx=0,
        )
        self.input_layout = input_layout
        if input_layout == "paired-concat":
            self.input_projection = (
                nn.Identity() if embedding_dim == channels else nn.Linear(embedding_dim, channels)
            )
            self.pair_composer = None
            self.pair_interaction_rank = 0
        else:
            self.input_projection = None
            self.pair_interaction_rank = int(pair_interaction_rank or channels)
            self.pair_composer = TypedPairComposer(
                component_dim,
                channels,
                self.pair_interaction_rank,
            )
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            PreNormGatedConvolution(channels, kernel_size, dropout) for _ in range(depth)
        )
        self.output_normalization = nn.LayerNorm(channels)
        self.output_dropout = nn.Dropout(dropout)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.depth = int(depth)
        self.receptive_field = 1 + self.depth * (self.kernel_size - 1)

    def typed_symbol_embeddings(
        self,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the explicit ``primary, secondary`` typed-symbol lattice."""
        if character_ids.ndim != 3 or character_ids.shape[-1] != 2:
            raise ValueError("character IDs must have shape [batch, characters, 2]")
        if character_mask.shape != character_ids.shape[:2]:
            raise ValueError("character mask must match the first two character-ID dimensions")
        primary = self.primary_embedding(character_ids[..., 0])
        secondary = self.secondary_embedding(character_ids[..., 1])
        typed_states = torch.stack((primary, secondary), dim=2).flatten(1, 2)
        typed_mask = character_mask.unsqueeze(-1).expand(-1, -1, 2).flatten(1, 2)
        return typed_states, typed_mask

    def forward(self, character_ids: torch.Tensor, character_mask: torch.Tensor) -> torch.Tensor:
        if character_ids.ndim != 3 or character_ids.shape[-1] != 2:
            raise ValueError("character IDs must have shape [batch, characters, 2]")
        if character_mask.shape != character_ids.shape[:2]:
            raise ValueError("character mask must match the first two character-ID dimensions")
        if self.input_layout == "paired-concat":
            embedded = torch.cat(
                (
                    self.primary_embedding(character_ids[..., 0]),
                    self.secondary_embedding(character_ids[..., 1]),
                ),
                dim=-1,
            )
            assert self.input_projection is not None
            hidden = self.input_projection(embedded)
        else:
            typed_states, typed_mask = self.typed_symbol_embeddings(character_ids, character_mask)
            assert self.pair_composer is not None
            hidden, composed_mask = self.pair_composer(typed_states, typed_mask)
            if not torch.equal(composed_mask, character_mask):
                raise AssertionError("typed pair composition changed the codepoint mask")
        hidden = self.input_dropout(hidden)
        hidden = hidden.masked_fill(~character_mask.unsqueeze(-1), 0)
        for block in self.blocks:
            hidden = block(hidden, character_mask)
        hidden = self.output_dropout(self.output_normalization(hidden))
        return hidden.masked_fill(~character_mask.unsqueeze(-1), 0)


class PerCharacterTagger(nn.Module):
    """Predict one fine BIOES label per Unicode codepoint from character IDs alone.

    The module signature is the deployed tokenizer-free contract itself: one
    ``[batch, characters]`` stream of projected codepoint IDs in, one
    ``[batch, characters, labels]`` logit tensor out.  There is no tokenizer,
    no token lattice, and no separate mask input that could disagree with the
    IDs: padding is the projection's reserved ID 0, so the character mask is
    recoverable from the IDs at every position.  Training and ONNX export
    therefore run this same forward rather than two aligned copies of it.

    One ID drives both learned component embedding tables, which is exact only
    for a diagonal projection.  The caller owns that contract; the trainer and
    the exporter each verify it against the projection file before use.
    """

    def __init__(
        self,
        encoder: ContinuousCharacterCnn,
        *,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("per-character tagging requires a positive label count")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if encoder.input_layout != "paired-concat":
            raise ValueError(
                "the per-character tagger reads both component embedding tables from one ID "
                f"stream, which requires the paired-concat layout, not {encoder.input_layout!r}"
            )
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.tag_classifier = nn.Linear(int(encoder.channels), int(num_labels))

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        if char_ids.ndim != 2:
            raise ValueError("character IDs must have shape [batch, characters]")
        character_mask = char_ids.ne(CHARACTER_PADDING_ID)
        character_states = self.encoder(
            torch.stack((char_ids, char_ids), dim=-1),
            character_mask,
        )
        return self.tag_classifier(self.dropout(character_states))


def sample_character_span_mask(
    character_mask: torch.Tensor,
    *,
    probability: float,
    mean_span_length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Select same-length masked spans, with at least one target per nonempty row."""
    if character_mask.ndim != 2:
        raise ValueError("character mask must have shape [batch, characters]")
    if not 0 < probability <= 1:
        raise ValueError("mask probability must be in (0, 1]")
    if mean_span_length <= 0:
        raise ValueError("mean span length must be positive")
    selected = torch.zeros_like(character_mask)
    for row_index, valid in enumerate(character_mask.cpu()):
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        target = max(1, round(valid_indices.numel() * probability))
        starts = valid_indices[torch.randperm(valid_indices.numel(), generator=generator)].tolist()
        for start in starts:
            if int(selected[row_index].sum()) >= target:
                break
            span_length = int(
                torch.randint(
                    1,
                    2 * mean_span_length + 1,
                    (),
                    generator=generator,
                ).item()
            )
            end = min(start + span_length, character_mask.shape[1])
            eligible = character_mask[row_index, start:end]
            selected[row_index, start:end] |= eligible
        if int(selected[row_index].sum()) > target:
            chosen = torch.nonzero(selected[row_index], as_tuple=False).flatten()
            selected[row_index, chosen[target:]] = False
    return selected


def masked_character_ids(
    character_ids: torch.Tensor,
    prediction_mask: torch.Tensor,
    *,
    primary_mask_id: int,
    secondary_mask_id: int,
) -> torch.Tensor:
    """Replace selected character pairs without changing sequence positions."""
    if prediction_mask.shape != character_ids.shape[:2]:
        raise ValueError("prediction mask must match the character-ID sequence dimensions")
    corrupted = character_ids.clone()
    corrupted[..., 0].masked_fill_(prediction_mask, primary_mask_id)
    corrupted[..., 1].masked_fill_(prediction_mask, secondary_mask_id)
    return corrupted


class MaskedCharacterPrediction(nn.Module):
    """Dual projection-component prediction head for character-CNN pretraining."""

    def __init__(self, encoder: ContinuousCharacterCnn) -> None:
        super().__init__()
        self.encoder = encoder
        self.primary_head = nn.Linear(encoder.channels, encoder.primary_vocab_size)
        self.secondary_head = nn.Linear(encoder.channels, encoder.secondary_vocab_size)

    def forward(
        self,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not torch.any(prediction_mask):
            raise ValueError("masked character prediction requires at least one target")
        corrupted = masked_character_ids(
            character_ids,
            prediction_mask,
            primary_mask_id=self.encoder.primary_mask_id,
            secondary_mask_id=self.encoder.secondary_mask_id,
        )
        hidden = self.encoder(corrupted, character_mask)
        selected = hidden[prediction_mask]
        primary_logits = self.primary_head(selected)
        secondary_logits = self.secondary_head(selected)
        primary_loss = F.cross_entropy(primary_logits, character_ids[..., 0][prediction_mask])
        secondary_loss = F.cross_entropy(secondary_logits, character_ids[..., 1][prediction_mask])
        loss = 0.5 * (primary_loss + secondary_loss)
        return loss, {
            "primary_loss": primary_loss.detach(),
            "secondary_loss": secondary_loss.detach(),
            "masked_characters": prediction_mask.sum().detach(),
        }


class MaskedLiteralCharacterPrediction(nn.Module):
    """One exact-character prediction head over a diagonal literal projection."""

    def __init__(self, encoder: ContinuousCharacterCnn) -> None:
        super().__init__()
        if encoder.primary_vocab_size != encoder.secondary_vocab_size:
            raise ValueError("literal character prediction requires equal component vocabularies")
        self.encoder = encoder
        self.character_head = nn.Linear(encoder.channels, encoder.primary_vocab_size)

    def forward(
        self,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not torch.any(prediction_mask):
            raise ValueError("masked literal-character prediction requires at least one target")
        if not torch.equal(character_ids[..., 0], character_ids[..., 1]):
            raise ValueError("literal character prediction requires diagonal character IDs")
        corrupted = masked_character_ids(
            character_ids,
            prediction_mask,
            primary_mask_id=self.encoder.primary_mask_id,
            secondary_mask_id=self.encoder.secondary_mask_id,
        )
        hidden = self.encoder(corrupted, character_mask)
        logits = self.character_head(hidden[prediction_mask])
        targets = character_ids[..., 0][prediction_mask]
        target_losses = F.cross_entropy(logits, targets, reduction="none")
        loss = target_losses.mean()
        top_k = min(5, logits.shape[-1])
        top_indices = logits.topk(top_k, dim=-1).indices
        target_rows = torch.nonzero(prediction_mask, as_tuple=True)[0]
        batch_size = character_ids.shape[0]

        def per_row_sum(values: torch.Tensor) -> torch.Tensor:
            totals = torch.zeros(batch_size, dtype=values.dtype, device=values.device)
            totals.scatter_add_(0, target_rows, values)
            return totals.detach()

        return loss, {
            "character_loss": loss.detach(),
            "top1_correct": top_indices[:, 0].eq(targets).sum().detach(),
            "top5_correct": top_indices.eq(targets.unsqueeze(-1)).any(dim=-1).sum().detach(),
            "unknown_targets": targets.eq(CHARACTER_UNKNOWN_ID).sum().detach(),
            "masked_characters": prediction_mask.sum().detach(),
            "row_loss_sum": per_row_sum(target_losses),
            "row_top1_correct": per_row_sum(top_indices[:, 0].eq(targets).to(torch.long)),
            "row_top5_correct": per_row_sum(top_indices.eq(targets.unsqueeze(-1)).any(dim=-1).to(torch.long)),
            "row_unknown_targets": per_row_sum(targets.eq(CHARACTER_UNKNOWN_ID).to(torch.long)),
            "row_masked_characters": prediction_mask.sum(dim=1).detach(),
        }


def pool_character_states(
    character_states: torch.Tensor,
    token_offsets: torch.Tensor,
    *,
    mode: str = "max",
    max_mean_weight: float = 0.5,
) -> torch.Tensor:
    """Pool each token's raw character interval, including shared centers."""
    if character_states.ndim != 3:
        raise ValueError("character states must have shape [batch, characters, channels]")
    if (
        token_offsets.ndim != 3
        or token_offsets.shape[0] != character_states.shape[0]
        or token_offsets.shape[-1] != 2
    ):
        raise ValueError("token offsets must have shape [batch, tokens, 2]")
    if mode not in {"max", "mean", "max-mean"}:
        raise ValueError(f"unsupported token pooling mode {mode!r}")
    if not 0 <= max_mean_weight <= 1:
        raise ValueError("max-mean weight must be in [0, 1]")
    batch_size, character_count, channels = character_states.shape
    token_count = token_offsets.shape[1]
    offsets = token_offsets.to(character_states.device)
    starts = offsets[..., 0]
    ends = offsets[..., 1]
    invalid = (starts < 0) | (ends < starts) | (ends > character_count)
    if torch.any(invalid):
        raise ValueError("token offset is outside the character-state sequence")
    positions = torch.arange(character_count, device=character_states.device)
    membership = (
        (starts < ends).unsqueeze(-1) & (starts.unsqueeze(-1) <= positions) & (positions < ends.unsqueeze(-1))
    )
    edge_batch, edge_token, edge_character = torch.nonzero(membership, as_tuple=True)
    flat_tokens = edge_batch * token_count + edge_token
    flat_states = character_states[edge_batch, edge_character]
    output_shape = (batch_size * token_count, channels)
    scatter_indices = flat_tokens.unsqueeze(-1).expand(-1, channels)

    mean_states = torch.zeros(output_shape, device=character_states.device, dtype=character_states.dtype)
    mean_states.scatter_add_(0, scatter_indices, flat_states)
    counts = torch.zeros(batch_size * token_count, device=character_states.device, dtype=torch.long)
    counts.scatter_add_(0, flat_tokens, torch.ones_like(flat_tokens))
    mean_states = mean_states / counts.clamp_min(1).unsqueeze(-1)
    if mode == "mean":
        return mean_states.reshape(batch_size, token_count, channels)

    max_states = torch.full(
        output_shape,
        -torch.inf,
        device=character_states.device,
        dtype=character_states.dtype,
    ).scatter_reduce(0, scatter_indices, flat_states, reduce="amax", include_self=True)
    max_states = max_states.masked_fill(counts.eq(0).unsqueeze(-1), 0)
    if mode == "max":
        return max_states.reshape(batch_size, token_count, channels)
    mixed = max_mean_weight * max_states + (1 - max_mean_weight) * mean_states
    return mixed.reshape(batch_size, token_count, channels)


def max_pool_character_intervals(
    character_states: torch.Tensor,
    token_starts: torch.Tensor,
    token_ends: torch.Tensor,
) -> torch.Tensor:
    """Max-pool each token's half-open character interval with dense tensors.

    This is the graph-export twin of ``pool_character_states(mode="max")``.
    It keeps the same membership rule—``start < end`` and
    ``start <= position < end``, so shared characters reach every owning
    token—but expresses it as a boolean mask plus a masked maximum instead of
    ``nonzero``/``scatter_reduce``.  Those two operators produce
    data-dependent shapes that an ONNX trace cannot express, whereas every
    step here has a shape derived only from the input dimensions.  Tokens with
    an empty interval pool to exact zeros.

    Peak memory is ``batch * tokens * characters * channels`` elements, which
    is the price of removing the data-dependent gather.

    The caller owns the offset contract ``0 <= start <= end <= characters``;
    an ONNX graph cannot raise, so this function deliberately does not add a
    data-dependent check that would vanish from the exported graph.
    """
    if character_states.ndim != 3:
        raise ValueError("character states must have shape [batch, characters, channels]")
    if token_starts.ndim != 2 or token_ends.ndim != 2:
        raise ValueError("token starts and ends must both have shape [batch, tokens]")
    if not torch.jit.is_tracing():
        # A tracer turns these size comparisons into tensor operations it then
        # warns about folding into constants, and the exported graph has no way
        # to raise. Keep them as eager-caller checks.
        if token_ends.shape != token_starts.shape:
            raise ValueError("token starts and ends must both have shape [batch, tokens]")
        if token_starts.shape[0] != character_states.shape[0]:
            raise ValueError("token offsets and character states must name the same batch")
    positions = torch.arange(
        character_states.shape[1],
        device=character_states.device,
        dtype=token_starts.dtype,
    )
    membership = (
        (token_starts < token_ends).unsqueeze(-1)
        & (token_starts.unsqueeze(-1) <= positions)
        & (positions < token_ends.unsqueeze(-1))
    )
    candidates = torch.where(
        membership.unsqueeze(-1),
        character_states.unsqueeze(1),
        torch.full(
            (),
            torch.finfo(character_states.dtype).min,
            device=character_states.device,
            dtype=character_states.dtype,
        ),
    )
    pooled = candidates.amax(dim=2)
    return torch.where(
        membership.any(dim=2).unsqueeze(-1),
        pooled,
        torch.zeros((), device=character_states.device, dtype=character_states.dtype),
    )


def initialized_expanded_classifier(
    incumbent: nn.Linear,
    character_width: int,
) -> nn.Linear:
    """Append zero character columns to an incumbent typed affine head."""
    if character_width <= 0:
        raise ValueError("character width must be positive")
    expanded = nn.Linear(
        incumbent.in_features + character_width,
        incumbent.out_features,
        bias=incumbent.bias is not None,
        device=incumbent.weight.device,
        dtype=incumbent.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, : incumbent.in_features].copy_(incumbent.weight)
        if incumbent.bias is not None:
            assert expanded.bias is not None
            expanded.bias.copy_(incumbent.bias)
    return expanded


class CharacterAugmentedTokenClassifier(nn.Module):
    """Pool continuous character states and concatenate them before typed logits."""

    def __init__(
        self,
        encoder: ContinuousCharacterCnn,
        incumbent_classifier: nn.Linear,
        *,
        pooling: str = "max",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.encoder = encoder
        self.pooling = pooling
        self.character_dropout = nn.Dropout(dropout)
        self.classifier = initialized_expanded_classifier(incumbent_classifier, encoder.channels)

    def forward(
        self,
        token_states: torch.Tensor,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
        token_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if token_states.ndim != 3:
            raise ValueError("token states must have shape [batch, tokens, channels]")
        character_states = self.encoder(character_ids, character_mask)
        token_character_states = pool_character_states(
            character_states,
            token_offsets,
            mode=self.pooling,
        )
        if token_character_states.shape[:2] != token_states.shape[:2]:
            raise ValueError("token offsets and token states must name the same token lattice")
        features = torch.cat(
            (token_states, self.character_dropout(token_character_states)),
            dim=-1,
        )
        return self.classifier(features)
