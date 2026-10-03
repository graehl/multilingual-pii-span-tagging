"""Shared causal-decoding batches and reusable prompt-prefix state."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

TokenIds = Sequence[int] | torch.Tensor


def _run_metrics():
    """The shared GPU-run metrics helpers, or inert stand-ins when unavailable."""
    here = Path(__file__).resolve().parent
    if str(here / "scripts") not in sys.path:
        sys.path.insert(0, str(here / "scripts"))
    import agents_run_metrics

    return agents_run_metrics


class DecodeMetrics:
    """Publish a platform-upgrade baseline around a decode pass.

    Decoding has no trainer loop to hang a callback on, so the caller brackets its own
    pass. Everything it needs is countable at the call site and nothing here is decode
    theory: rows and generated tokens go in, a rate and the peak memory come out.

        metrics = DecodeMetrics().start()
        ...
        metrics.finish(rows=len(prompts), tokens=generated, batch_size=32, dtype="bf16")

    A no-op outside a tracked run, and never raises.
    """

    def __init__(self, phase: str = "decode"):
        self._metrics = _run_metrics().RunMetrics(phase)

    def start(self) -> DecodeMetrics:
        self._metrics.start()
        return self

    def finish(self, **facts: Any) -> None:
        self._metrics.finish(**facts)

    def __enter__(self) -> DecodeMetrics:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.finish()


def _token_list(token_ids: TokenIds) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError("token ids must be one-dimensional")
        return token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def common_prefix_length(sequences: Sequence[TokenIds]) -> int:
    """Return the exact token prefix shared by every nonempty sequence."""
    if not sequences:
        raise ValueError("at least one token sequence is required")
    first = _token_list(sequences[0])
    if not first:
        raise ValueError("token sequences must be nonempty")
    shared = len(first)
    for sequence in sequences[1:]:
        tokens = _token_list(sequence)
        if not tokens:
            raise ValueError("token sequences must be nonempty")
        shared = min(shared, len(tokens))
        index = 0
        while index < shared and tokens[index] == first[index]:
            index += 1
        shared = index
        if not shared:
            break
    return shared


def split_common_prefix(
    sequences: Sequence[TokenIds],
    *,
    minimum_tokens: int = 1,
    retained_suffix_tokens: int = 1,
) -> tuple[list[int], list[list[int]]]:
    """Split an exact shared prefix while keeping every suffix nonempty."""
    if minimum_tokens <= 0:
        raise ValueError("minimum_tokens must be positive")
    if retained_suffix_tokens <= 0:
        raise ValueError("retained_suffix_tokens must be positive")
    token_lists = [_token_list(sequence) for sequence in sequences]
    shared = common_prefix_length(token_lists)
    max_prefix = min(len(tokens) - retained_suffix_tokens for tokens in token_lists)
    prefix_length = min(shared, max_prefix)
    if prefix_length < minimum_tokens:
        raise ValueError(
            f"exact shared prefix has {prefix_length} usable tokens; minimum is {minimum_tokens}"
        )
    return token_lists[0][:prefix_length], [tokens[prefix_length:] for tokens in token_lists]


@dataclass(frozen=True)
class PrefilledPrefix:
    """One-row model KV state for an exact token prefix."""

    past_key_values: Any
    token_ids: tuple[int, ...]
    token_count: int
    model_identity: int

    def for_batch(
        self,
        model: Any,
        batch_size: int,
        *,
        leading_pad_counts: Sequence[int] | None = None,
        fork: bool = True,
    ) -> Any:
        if id(model) != self.model_identity:
            raise ValueError("prefix state belongs to a different model instance")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not fork:
            if batch_size != 1 or (leading_pad_counts is not None and any(leading_pad_counts)):
                raise ValueError("a consumed prefix must remain an unpadded one-row cache")
            return self.past_key_values
        cache = _fork_cache(self.past_key_values)
        repeat = getattr(cache, "batch_repeat_interleave", None)
        if not callable(repeat):
            raise TypeError(f"{type(cache).__name__} cannot expand a one-row prefix cache to a batch")
        if batch_size > 1:
            repeat(batch_size)
        if leading_pad_counts is not None and any(leading_pad_counts):
            _shift_cache_behind_padding(cache, leading_pad_counts, self.token_count)
        return cache

    def copy_to(
        self,
        model: Any,
        device: torch.device | str,
    ) -> PrefilledPrefix:
        if id(model) != self.model_identity:
            raise ValueError("prefix state belongs to a different model instance")
        cache = _fork_cache(self.past_key_values)
        _move_dynamic_cache(cache, torch.device(device))
        return _prefix_from_exact_cache(model, cache, self.token_ids)


def _fork_cache(cache: Any) -> Any:
    layers = getattr(cache, "layers", None)
    if (
        isinstance(cache, DynamicCache)
        and isinstance(layers, list)
        and all(isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)) for layer in layers)
    ):
        fork = copy.copy(cache)
        fork.layers = [copy.copy(layer) for layer in layers]
        return fork
    return copy.deepcopy(cache)


def _move_dynamic_cache(cache: Any, device: torch.device) -> None:
    layers = getattr(cache, "layers", None)
    if not isinstance(cache, DynamicCache) or not isinstance(layers, list):
        raise TypeError(f"{type(cache).__name__} cannot be copied between context-handle devices")
    for layer in layers:
        if not isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)):
            raise TypeError(f"{type(layer).__name__} is not an append-only dynamic cache layer")
        if not layer.is_initialized:
            continue
        layer.keys = layer.keys.to(device=device, copy=True)
        layer.values = layer.values.to(device=device, copy=True)


def _cache_token_count(cache: Any) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise TypeError(f"{type(cache).__name__} does not report its cached token count")
    return int(get_seq_length())


def _prefix_from_exact_cache(
    model: Any,
    cache: Any,
    token_ids: Sequence[int],
) -> PrefilledPrefix:
    tokens = tuple(int(token_id) for token_id in token_ids)
    cached = _cache_token_count(cache)
    if cached != len(tokens):
        raise RuntimeError(f"cache contains {cached} tokens; expected {len(tokens)}")
    repeat = getattr(cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise TypeError(f"{type(cache).__name__} does not support context-handle forks")
    return PrefilledPrefix(
        past_key_values=cache,
        token_ids=tokens,
        token_count=len(tokens),
        model_identity=id(model),
    )


@dataclass(frozen=True)
class DecodeBatch:
    """A one-use padded causal-generation batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any | None = None
    prefix_token_count: int = 0

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def output_prompt_width(self) -> int:
        """Width to remove from ``generate`` sequences before decoding output."""
        return int(self.input_ids.shape[1])

    @property
    def total_prompt_width(self) -> int:
        """Padded prompt width including a prefix represented only in KV state."""
        return self.prefix_token_count + self.output_prompt_width

    def model_inputs(self) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.past_key_values is not None:
            inputs["past_key_values"] = self.past_key_values
        return inputs


def prefill_prefix(
    model: Any,
    prefix_token_ids: TokenIds,
    *,
    device: torch.device | str | None = None,
) -> PrefilledPrefix:
    """Compute one reusable KV state without generating a token."""
    tokens = _token_list(prefix_token_ids)
    if not tokens:
        raise ValueError("prefix must contain at least one token")
    target_device = torch.device(device) if device is not None else torch.device(model.device)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=target_device)
    attention_mask = torch.ones_like(input_ids)
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise RuntimeError("model did not return past_key_values for prefix prefill")
    return _prefix_from_exact_cache(model, cache, tokens)


def extend_prefix(
    model: Any,
    prefix: PrefilledPrefix,
    suffix_token_ids: TokenIds,
    *,
    device: torch.device | str | None = None,
) -> PrefilledPrefix:
    """Fork a context handle and append exact tokens without replaying its prefix."""
    suffix = _token_list(suffix_token_ids)
    if not suffix:
        return prefix
    target_device = torch.device(device) if device is not None else torch.device(model.device)
    input_ids = torch.tensor([suffix], dtype=torch.long, device=target_device)
    attention_mask = torch.ones(
        (1, prefix.token_count + len(suffix)),
        dtype=torch.long,
        device=target_device,
    )
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=prefix.for_batch(model, 1),
            use_cache=True,
        )
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise RuntimeError("model did not return past_key_values while extending context")
    return _prefix_from_exact_cache(model, cache, [*prefix.token_ids, *suffix])


def prefix_from_generate(
    model: Any,
    committed_token_ids: TokenIds,
    past_key_values: Any,
    *,
    device: torch.device | str | None = None,
) -> PrefilledPrefix:
    """Return the exact reusable prefix already present in a generation cache.

    Transformers generation normally leaves the returned cache one token behind
    ``sequences``. The caller retains the full committed token stream and feeds that
    pending final token with the next turn instead of running an otherwise unnecessary
    full-model forward solely to cache an end-of-turn marker.
    """
    tokens = _token_list(committed_token_ids)
    if not tokens:
        raise ValueError("generated context must contain at least one token")
    if past_key_values is None:
        raise RuntimeError("generate did not return past_key_values for context reuse")
    cached = _cache_token_count(past_key_values)
    if cached == len(tokens):
        return _prefix_from_exact_cache(model, past_key_values, tokens)
    if cached != len(tokens) - 1:
        raise RuntimeError(f"generated cache contains {cached} tokens for a {len(tokens)}-token context")
    return _prefix_from_exact_cache(model, past_key_values, tokens[:-1])


def _shift_cache_behind_padding(
    cache: Any,
    leading_pad_counts: Sequence[int],
    prefix_token_count: int,
) -> None:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list):
        raise TypeError(f"{type(cache).__name__} does not expose transformable cache layers")
    for layer in layers:
        if not getattr(layer, "is_initialized", False):
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError(f"{type(layer).__name__} does not expose tensor key/value state")
        if keys.shape[0] != len(leading_pad_counts) or values.shape[0] != len(leading_pad_counts):
            raise RuntimeError("expanded prefix cache batch size does not match padding plan")
        if keys.shape[-2] != prefix_token_count or values.shape[-2] != prefix_token_count:
            raise RuntimeError("prefix prefill cache did not retain its complete token history")
        shifted_keys = torch.zeros_like(keys)
        shifted_values = torch.zeros_like(values)
        for row, pad_count in enumerate(leading_pad_counts):
            retained = prefix_token_count - pad_count
            shifted_keys[row, :, pad_count:, :] = keys[row, :, :retained, :]
            shifted_values[row, :, pad_count:, :] = values[row, :, :retained, :]
        layer.keys = shifted_keys
        layer.values = shifted_values


def prepare_batch(
    model: Any,
    sequences: Sequence[TokenIds],
    *,
    pad_token_id: int,
    padding_side: str = "left",
    prefix: PrefilledPrefix | None = None,
    device: torch.device | str | None = None,
    fork_prefix: bool = True,
) -> DecodeBatch:
    """Pad token sequences and attach an expanded prefix cache when supplied."""
    if not sequences:
        raise ValueError("at least one token sequence is required")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    token_lists = [_token_list(sequence) for sequence in sequences]
    if any(not tokens for tokens in token_lists):
        raise ValueError("token sequences must be nonempty")
    target_device = torch.device(device) if device is not None else torch.device(model.device)
    suffix_width = max(map(len, token_lists))
    leading_pad_counts = [suffix_width - len(tokens) for tokens in token_lists]
    if prefix is not None and padding_side == "left":
        if max(leading_pad_counts) >= prefix.token_count:
            return prepare_batch(
                model,
                [list(prefix.token_ids) + tokens for tokens in token_lists],
                pad_token_id=pad_token_id,
                padding_side=padding_side,
                device=target_device,
            )
        token_lists = [
            list(prefix.token_ids[prefix.token_count - pad_count :]) + tokens
            for pad_count, tokens in zip(leading_pad_counts, token_lists, strict=True)
        ]
    width = max(len(tokens) for tokens in token_lists)
    input_ids = torch.full(
        (len(token_lists), width),
        int(pad_token_id),
        dtype=torch.long,
        device=target_device,
    )
    suffix_mask = torch.zeros(
        (len(token_lists), width),
        dtype=torch.long,
        device=target_device,
    )
    for row, tokens in enumerate(token_lists):
        start = width - len(tokens) if padding_side == "left" else 0
        end = start + len(tokens)
        input_ids[row, start:end] = torch.tensor(tokens, dtype=torch.long, device=target_device)
        suffix_mask[row, start:end] = 1

    if prefix is None:
        return DecodeBatch(input_ids=input_ids, attention_mask=suffix_mask)
    prefix_mask = torch.ones((len(token_lists), prefix.token_count), dtype=torch.long, device=target_device)
    if padding_side == "left":
        for row, pad_count in enumerate(leading_pad_counts):
            prefix_mask[row, :pad_count] = 0
    return DecodeBatch(
        input_ids=input_ids,
        attention_mask=torch.cat((prefix_mask, suffix_mask), dim=1),
        past_key_values=prefix.for_batch(
            model,
            len(token_lists),
            leading_pad_counts=leading_pad_counts if padding_side == "left" else None,
            fork=fork_prefix,
        ),
        prefix_token_count=prefix.token_count,
    )


def prepare_cached_batch(
    model: Any,
    full_sequences: Sequence[TokenIds],
    *,
    pad_token_id: int,
    prefix: PrefilledPrefix,
    padding_side: str = "left",
    device: torch.device | str | None = None,
    fork_prefix: bool = True,
) -> DecodeBatch:
    """Prepare complete sequences by stripping one exact reusable context handle."""
    prefix_tokens = list(prefix.token_ids)
    suffixes: list[list[int]] = []
    for row, sequence in enumerate(full_sequences):
        tokens = _token_list(sequence)
        if tokens[: prefix.token_count] != prefix_tokens:
            raise ValueError(f"sequence {row} does not extend the supplied context handle")
        suffix = tokens[prefix.token_count :]
        if not suffix:
            raise ValueError(f"sequence {row} has no uncached token to decode")
        suffixes.append(suffix)
    batch = prepare_batch(
        model,
        suffixes,
        pad_token_id=pad_token_id,
        padding_side=padding_side,
        prefix=prefix,
        device=device,
        fork_prefix=fork_prefix,
    )
    if batch.past_key_values is None:
        raise RuntimeError("cached batch preparation fell back to full-sequence prefill")
    return batch
