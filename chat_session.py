"""Turn-taking chat session over a HF causal LM with KV-cache reuse.

Vocabulary (topics/chatroom-translation.md): a *model definition* — the shipped
tokenizer, `chat_template.jinja`, and generation config — combined with
session-level *format options* resolves to a *model format*: the template and
turn mechanics the session actually renders with. Options that affect turn text
(thinking on/off, whether history keeps think pads, stable-vs-stock history
template) are session properties, not fixed model properties.

ChatSession is a longest-common-prefix token machine. It owns the committed
token stream and KV cache for one conversation. Each turn it renders the
message list, reuses the longest common prefix with the committed stream
(cropping the cache when the tail was rewritten — e.g. stock Qwen3 templates
render past assistant turns without the empty think pad the generation prompt
inserted), feeds only the suffix, and generates in chunks until an end-of-turn
token. Models are unaware of ``max_new_tokens``, so a turn only ends at an
end-of-turn token, a per-turn cap, or a degenerate empty chunk; all three are
counted. Per-turn metrics record cached vs fed vs rewritten tokens, so template
prefix-instability shows up as a measured cost, never a silent one.

The default ``history_template="auto"`` keeps the committed stream
prefix-identical for ChatML+thinking models by rendering history with the same
empty think pad the generation prompt used (``stable``); ``stock`` re-renders
with the model's shipped template and pays the measured tail rescan instead.
"""

import time
from dataclasses import dataclass

import prompt

# ChatML history template that renders past assistant turns exactly as the
# thinking-off generation prompt produced them (empty think pad kept), so the
# committed stream stays prefix-identical across turns. Text-only subset: the
# chatroom session never uses tools or multimodal content.
STABLE_CHATML_NOTHINK = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['role'] == 'assistant' -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}"
    "{{- message['content'] -}}{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}"
    "{%- endif -%}"
)

# End-of-turn tag tokens some templates use; resolved against the vocab, never assumed.
TURN_END_TAGS = ("<|im_end|>", "<end_of_turn>")

HISTORY_TEMPLATE_MODES = ("auto", "stock", "stable")


@dataclass(frozen=True)
class FormatOptions:
    """Session-level choices that, applied to a model definition, resolve the model format.

    thinking: chat-template thinking mode (prompt.THINKING_MODES).
    history_template: auto|stock|stable. stable renders history prefix-identical
        to what generation produced (currently the ChatML thinking-off pad);
        stock uses the model's shipped template and accepts tail rescan.
    strip_thinking_output: strip a leading think block from the reply text
        returned to the caller (history handling is the template's job).
    """

    thinking: str = "off"
    history_template: str = "auto"
    strip_thinking_output: bool = True


class ModelFormat:
    """Model definition + format options -> resolved template and turn mechanics."""

    def __init__(
        self, tokenizer, *, options: FormatOptions | None = None, processor=None, generation_config=None
    ):
        self.tokenizer = tokenizer
        self.options = options or FormatOptions()
        if self.options.thinking not in prompt.THINKING_MODES:
            raise ValueError(f"unsupported thinking mode {self.options.thinking!r}")
        if self.options.history_template not in HISTORY_TEMPLATE_MODES:
            raise ValueError(f"unsupported history_template {self.options.history_template!r}")
        stock = getattr(tokenizer, "chat_template", None) or ""
        self.history_template_mode = self._resolve_history_template(stock)
        template_override = STABLE_CHATML_NOTHINK if self.history_template_mode == "stable" else None
        # Templates without an enable_thinking knob (e.g. Gemma) get no kwarg at all;
        # the stable override hard-codes the pad, so it needs no kwarg either.
        omit_off = "enable_thinking" not in stock
        thinking_kwargs = (
            {}
            if template_override
            else prompt.chat_template_thinking_kwargs(self.options.thinking, omit_off=omit_off)
        )
        self.renderer = prompt.TemplateRenderer(
            tokenizer,
            processor=processor,
            thinking_kwargs=thinking_kwargs,
            template_override=template_override,
        )
        self.stop_token_ids = self._stop_token_ids(generation_config)
        self.turn_end_id = self._turn_end_id(stock or STABLE_CHATML_NOTHINK)

    def _resolve_history_template(self, stock_template: str) -> str:
        mode = self.options.history_template
        if mode == "stock":
            return "stock"
        stable_ok = (
            self.options.thinking == "off"
            and "<|im_start|>" in stock_template
            and "enable_thinking" in stock_template
        )
        if mode == "stable":
            if not stable_ok:
                raise ValueError(
                    "history_template=stable requires thinking=off and a ChatML template "
                    "with an enable_thinking knob (e.g. Qwen3); use stock for this model"
                )
            return "stable"
        return "stable" if stable_ok else "stock"

    def _stop_token_ids(self, generation_config) -> list[int]:
        ids: set[int] = set()
        sources = [
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(generation_config, "eos_token_id", None),
        ]
        for src in sources:
            if src is None:
                continue
            for tid in src if isinstance(src, (list, tuple)) else [src]:
                if tid is not None and tid >= 0:
                    ids.add(int(tid))
        unk = getattr(self.tokenizer, "unk_token_id", None)
        for tag in TURN_END_TAGS:
            tid = self.tokenizer.convert_tokens_to_ids(tag)
            if tid is not None and tid >= 0 and tid != unk:
                ids.add(int(tid))
        if not ids:
            raise ValueError("no end-of-turn/EOS token ids resolvable from tokenizer or generation config")
        return sorted(ids)

    def _turn_end_id(self, template: str) -> int:
        """The token the template ends assistant turns with (used to close simulated turns)."""
        unk = getattr(self.tokenizer, "unk_token_id", None)
        for tag in TURN_END_TAGS:
            if tag in template:
                tid = self.tokenizer.convert_tokens_to_ids(tag)
                if tid is not None and tid >= 0 and tid != unk:
                    return int(tid)
        return self.stop_token_ids[0]

    def render(self, msgs: list, *, add_generation_prompt: bool) -> list[int]:
        return self.renderer.token_ids(msgs, add_generation_prompt=add_generation_prompt)

    def visible_reply(self, text: str) -> str:
        if self.options.strip_thinking_output:
            return prompt.strip_leading_think_block(text)
        return text


@dataclass
class TurnMetrics:
    """Per-turn cache/latency record; the cache-proof measurement unit."""

    utterance_id: object = None
    render_tokens: int = 0  # full rendered history incl. new turn + generation prompt
    cached_tokens: int = 0  # longest common prefix reused from the committed stream
    fed_tokens: int = 0  # render_tokens - cached_tokens (prefill work this turn)
    rewritten_tokens: int = 0  # committed tokens dropped because the render rewrote the tail
    output_tokens: int = 0
    committed_tokens: int = 0  # stream length after committing this turn
    gen_chunks: int = 0
    stop_reason: str = ""  # eot | max-turn-tokens | empty-chunk | reference
    prefix_stable: bool = True  # entire previous committed stream was reused
    first_token_ms: float | None = None
    turn_ms: float = 0.0
    decode_tps: float | None = None
    cache_mib: float | None = None


def _lcp_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _iter_cache_tensors(value, seen=None):
    """Best-effort walk of a transformers Cache object's tensors (API varies by version)."""
    import torch

    if seen is None:
        seen = set()
    if value is None or id(value) in seen:
        return []
    seen.add(id(value))
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_iter_cache_tensors(item, seen))
        return tensors
    tensors = []
    for attr in ("key_cache", "value_cache", "keys", "values", "layers"):
        if hasattr(value, attr):
            tensors.extend(_iter_cache_tensors(getattr(value, attr), seen))
    return tensors


def cache_mib(cache) -> float | None:
    tensors = _iter_cache_tensors(cache)
    if not tensors:
        return None
    return sum(t.numel() * t.element_size() for t in tensors) / (1024 * 1024)


class _FirstTokenTimer:
    """Duck-typed generate() streamer that records time-to-first-token."""

    def __init__(self, t0: float):
        self.t0 = t0
        self.first_token_ms: float | None = None
        self._seen_prompt = False

    def put(self, value):
        if not self._seen_prompt:
            self._seen_prompt = True  # first put() is the prompt ids
            return
        if self.first_token_ms is None:
            self.first_token_ms = (time.perf_counter() - self.t0) * 1000

    def end(self):
        pass


class ChatSession:
    """One live conversation: message list, committed token stream, KV cache, metrics.

    ``model=None`` supports tokenize-only dry runs via commit_reference_reply().
    """

    def __init__(
        self,
        fmt: ModelFormat,
        model=None,
        *,
        session_cache: bool = True,
        chunk_tokens: int = 256,
        max_turn_tokens: int = 1024,
        temperature: float = 0.0,
        suppress_token_ids: list[int] | None = None,
    ):
        self.fmt = fmt
        self.model = model
        self.session_cache = session_cache
        self.chunk_tokens = chunk_tokens
        self.max_turn_tokens = max_turn_tokens
        self.temperature = temperature
        # Token ids the decoder may never emit (logit -> -inf), e.g. reasoning-channel
        # open tokens for a thinking model asked to translate only.
        self.suppress_token_ids = list(suppress_token_ids) if suppress_token_ids else None
        self.msgs: list[dict] = []
        self.tokens: list[int] = []  # committed stream the cache state corresponds to
        self.cache = None
        self.cache_unsupported = False  # set when the backend never returns a reusable cache
        self.metrics: list[TurnMetrics] = []
        self._committed_msgs = 0  # msg count as of the last commit (for append-only targets)
        self._prelude_snapshot = None  # (cache, tokens, n_prelude_msgs) for start-over restore

    def add(self, role: str, content: str) -> None:
        self.msgs.append(prompt.message(role, content))

    def truncate_history(self, keep_msgs: int) -> None:
        """Drop all but the first keep_msgs messages (e.g. the prelude).

        The committed token stream and cache are left alone: the next render's
        longest-common-prefix alignment crops them and counts the dropped tail
        as rewritten_tokens, so the truncation cost is measured, never silent.
        """
        if keep_msgs < len(self.msgs):
            self.msgs = self.msgs[:keep_msgs]

    def prime_prelude(self, n_prelude_msgs: int) -> None:
        """Prefill the prelude alone and snapshot its KV cache as the start-over base.

        Taken when context is just the prelude (far under any sliding window), so nothing
        is evicted and the snapshot is fully valid to restore later -- including for
        sliding-window caches (gemma-4). restart_recent() restores this instead of
        re-prefilling the prelude; only [last turn + current] is then prefilled on top."""
        import copy

        import torch

        if not (self.model and self.session_cache) or not self.msgs:
            return
        target = self.fmt.render(self.msgs, add_generation_prompt=False)
        if not target:
            return
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        ids = torch.tensor([target], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        self.tokens = list(target)
        self.cache = getattr(out, "past_key_values", None)
        self._committed_msgs = n_prelude_msgs
        try:
            self._prelude_snapshot = (copy.deepcopy(self.cache), list(target), n_prelude_msgs)
        except Exception:
            self._prelude_snapshot = None  # non-cloneable cache: fall back to re-prefill

    def restart_recent(self, n_prelude_msgs: int) -> None:
        """Start-over: keep the prelude + the most recent completed exchange + the current
        turn, drop the middle. Restores the snapshotted prelude KV cache (prime_prelude)
        so only [last turn + current] is prefilled -- never falling all the way back to
        no-conversation context, and never re-prefilling the prelude.

        Called after the current (not-yet-generated) user turn is added, so the tail is
        [..prelude.., <older history..>, current_user]."""
        import copy

        cur = self.msgs[-1:]  # current user turn (just added, not yet generated)
        body = self.msgs[n_prelude_msgs:-1]  # committed exchanges before it
        keep = body[-2:] if len(body) >= 2 else body  # last completed user+assistant exchange
        self.msgs = self.msgs[:n_prelude_msgs] + keep + cur
        if self._prelude_snapshot is not None:
            snap_cache, snap_tokens, snap_np = self._prelude_snapshot
            self.cache = copy.deepcopy(snap_cache)  # fresh copy; snapshot stays pristine
            self.tokens = list(snap_tokens)
            self._committed_msgs = snap_np  # prelude committed; next render appends [keep+cur]
        else:
            # No snapshot: keep the full cache and let the next render's LCP crop to the
            # shared prefix (full-attention) or re-prefill the small reduced prompt.
            self._committed_msgs = len(self.msgs)

    def _build_target(self) -> list[int]:
        """Model input for this turn. Prefer an append-only extension of the committed
        stream so already-generated reply tokens are never re-encoded; else full render."""
        inc = self._incremental_target()
        return inc if inc is not None else self.fmt.render(self.msgs, add_generation_prompt=True)

    def _incremental_target(self) -> list[int] | None:
        """Committed tokens + only this turn's newly rendered text, encoded once.

        Never re-encodes prior turns: decode->encode is not idempotent for SentencePiece
        (a generated reply can be a non-canonical segmentation of its own text, acute for
        CJK), so re-tokenizing committed replies diverges from the KV cache and forces a
        full re-prefill the sliding-window cache cannot crop back (see
        topics/chatroom-translation.md). Only applies to a pure single-user-turn
        extension of the committed stream; truncated/bounded history falls back to a full
        render + LCP crop. The new turn's text is delimited by turn special tokens, so
        encoding it in isolation is stable."""
        if not (self.session_cache and self.tokens):
            return None
        if len(self.msgs) != self._committed_msgs + 1 or self.msgs[-1].get("role") != "user":
            return None
        base = self.fmt.renderer.text(self.msgs[:-1], add_generation_prompt=False).rstrip("\n")
        full = self.fmt.renderer.text(self.msgs, add_generation_prompt=True)
        if not full.startswith(base):
            return None
        delta = self.fmt.tokenizer.encode(full[len(base):], add_special_tokens=False)
        return self.tokens + delta

    def _align_to(self, target: list[int], *, conceptual: bool = False) -> tuple[int, int]:
        """Crop stream+cache to the longest common prefix with target; return (cached, rewritten).

        conceptual=True (tokenize-only dry run) keeps the LCP accounting even though
        no cache object exists; a real run downgrades cached to 0 whenever the cache
        cannot actually be cropped to the common prefix, so metrics stay honest.
        """
        if not self.session_cache:
            self.tokens = []
            self.cache = None
            return 0, 0
        cached = _lcp_len(self.tokens, target)
        rewritten = len(self.tokens) - cached
        if not conceptual and self.tokens and self.cache is None:
            cached = 0  # backend never handed back a reusable cache: full re-prefill
        if rewritten:
            if self.cache is not None:
                if hasattr(self.cache, "crop"):
                    try:
                        self.cache.crop(cached)
                    except Exception:
                        self.cache = None
                        cached = 0
                else:
                    self.cache = None
                    cached = 0
            self.tokens = self.tokens[:cached]
        return cached, rewritten

    def _base_metrics(self, utterance_id, target: list[int], cached: int, rewritten: int) -> TurnMetrics:
        return TurnMetrics(
            utterance_id=utterance_id,
            render_tokens=len(target),
            cached_tokens=cached,
            fed_tokens=len(target) - cached,
            rewritten_tokens=rewritten,
            prefix_stable=not rewritten,
        )

    def generate_reply(self, utterance_id=None) -> tuple[str, TurnMetrics]:
        """Generate the assistant reply for the current message list; commit and record."""
        import torch

        assert self.model is not None, (
            "generate_reply requires a model; use commit_reference_reply for dry runs"
        )
        target = self._build_target()
        cached, rewritten = self._align_to(target)
        m = self._base_metrics(utterance_id, target, cached, rewritten)

        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        ids = torch.tensor([target], dtype=torch.long, device=device)
        stop_ids = self.fmt.stop_token_ids
        generated: list[int] = []
        m.stop_reason = "max-turn-tokens"
        t0 = time.perf_counter()
        with torch.inference_mode():
            while len(generated) < self.max_turn_tokens:
                budget = min(self.chunk_tokens, self.max_turn_tokens - len(generated))
                gen_kwargs: dict = dict(
                    max_new_tokens=budget,
                    eos_token_id=stop_ids,
                    pad_token_id=stop_ids[0],
                    return_dict_in_generate=True,
                )
                if self.temperature and self.temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=self.temperature)
                else:
                    gen_kwargs.update(do_sample=False)
                if self.suppress_token_ids:
                    gen_kwargs["suppress_tokens"] = self.suppress_token_ids
                streamer = _FirstTokenTimer(t0) if m.gen_chunks == 0 else None
                if streamer is not None:
                    gen_kwargs["streamer"] = streamer
                if self.cache is not None:
                    gen_kwargs["past_key_values"] = self.cache
                out = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    **gen_kwargs,
                )
                m.gen_chunks += 1
                sequences = out.sequences if hasattr(out, "sequences") else out
                new = sequences[0, ids.shape[1] :].tolist()
                if streamer is not None:
                    m.first_token_ms = streamer.first_token_ms
                if self.session_cache:
                    self.cache = getattr(out, "past_key_values", None)
                    if self.cache is None and not self.cache_unsupported:
                        self.cache_unsupported = True
                if not new:
                    m.stop_reason = "empty-chunk"
                    break
                generated.extend(int(t) for t in new)
                ids = sequences
                if any(t in stop_ids for t in new):
                    m.stop_reason = "eot"
                    break
        m.turn_ms = (time.perf_counter() - t0) * 1000
        m.output_tokens = len(generated)
        if m.first_token_ms is not None and len(generated) > 1:
            decode_s = (m.turn_ms - m.first_token_ms) / 1000
            if decode_s > 0:
                m.decode_tps = (len(generated) - 1) / decode_s
        if generated and generated[-1] not in stop_ids:
            generated.append(self.fmt.turn_end_id)  # keep the committed stream turn-closed
        text = self.fmt.tokenizer.decode(
            [t for t in generated if t not in stop_ids], skip_special_tokens=True
        )
        visible = self.fmt.visible_reply(text.strip())
        self._commit(target, generated, visible, m)
        if self.session_cache and self.cache is not None:
            m.cache_mib = cache_mib(self.cache)
        return visible, m

    def generate_code_lines_reply(
        self, target_langs: list[str], utterance_id=None
    ) -> tuple[str, TurnMetrics]:
        """Generate one assistant reply as forced ``code: text`` lines.

        The language-code prefixes are deterministic surface structure; the model
        still generates each line's translation text. This is an explicit
        constrained-emission mode for multi-target chatroom turns, not cleanup of
        a malformed free-form reply.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnLineBreak(StoppingCriteria):
            def __init__(self, tokenizer, start_len: int):
                self.tokenizer = tokenizer
                self.start_len = start_len

            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001, ANN003
                text = self.tokenizer.decode(
                    input_ids[0, self.start_len :].tolist(), skip_special_tokens=False
                )
                return "\n" in text or "\r" in text

        assert self.model is not None, (
            "generate_code_lines_reply requires a model; use commit_reference_reply for dry runs"
        )
        if not target_langs:
            raise ValueError("generate_code_lines_reply requires at least one target language")
        target = self._build_target()
        cached, rewritten = self._align_to(target)
        m = self._base_metrics(utterance_id, target, cached, rewritten)

        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        ids = torch.tensor([target], dtype=torch.long, device=device)
        stop_ids = self.fmt.stop_token_ids
        generated: list[int] = []
        line_cap = max(8, self.max_turn_tokens // len(target_langs))
        m.stop_reason = "forced-lines"
        t0 = time.perf_counter()

        def trim_trailing_line_break() -> None:
            nonlocal ids
            while generated:
                text = self.fmt.tokenizer.decode([generated[-1]], skip_special_tokens=False)
                if not (("\n" in text or "\r" in text) and not text.strip()):
                    break
                generated.pop()
                ids = ids[:, :-1]
                if self.cache is not None and hasattr(self.cache, "crop"):
                    try:
                        self.cache.crop(ids.shape[1])
                    except Exception:
                        self.cache = None
                        break

        def trim_trailing_stop() -> None:
            nonlocal ids
            while generated and generated[-1] in stop_ids:
                generated.pop()
                ids = ids[:, :-1]
                if self.cache is not None and hasattr(self.cache, "crop"):
                    try:
                        self.cache.crop(ids.shape[1])
                    except Exception:
                        self.cache = None
                        break

        with torch.inference_mode():
            for line_i, lang in enumerate(target_langs):
                prefix = f"{lang}: " if line_i == 0 else f"\n{lang}: "
                prefix_ids = self.fmt.tokenizer.encode(prefix, add_special_tokens=False)
                if prefix_ids:
                    generated.extend(int(t) for t in prefix_ids)
                    prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
                    ids = torch.cat([ids, prefix_t], dim=1)

                line_tokens = 0
                line_stop = ""
                while len(generated) < self.max_turn_tokens and line_tokens < line_cap:
                    budget = min(
                        self.chunk_tokens,
                        self.max_turn_tokens - len(generated),
                        line_cap - line_tokens,
                    )
                    if budget <= 0:
                        break
                    gen_kwargs: dict = dict(
                        max_new_tokens=budget,
                        eos_token_id=stop_ids,
                        pad_token_id=stop_ids[0],
                        return_dict_in_generate=True,
                        stopping_criteria=StoppingCriteriaList(
                            [StopOnLineBreak(self.fmt.tokenizer, ids.shape[1])]
                        ),
                    )
                    if self.temperature and self.temperature > 0:
                        gen_kwargs.update(do_sample=True, temperature=self.temperature)
                    else:
                        gen_kwargs.update(do_sample=False)
                    streamer = _FirstTokenTimer(t0) if m.gen_chunks == 0 else None
                    if streamer is not None:
                        gen_kwargs["streamer"] = streamer
                    if self.cache is not None:
                        gen_kwargs["past_key_values"] = self.cache
                    out = self.model.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        **gen_kwargs,
                    )
                    m.gen_chunks += 1
                    sequences = out.sequences if hasattr(out, "sequences") else out
                    new = sequences[0, ids.shape[1] :].tolist()
                    if streamer is not None:
                        m.first_token_ms = streamer.first_token_ms
                    if self.session_cache:
                        self.cache = getattr(out, "past_key_values", None)
                        if self.cache is None and not self.cache_unsupported:
                            self.cache_unsupported = True
                    if not new:
                        line_stop = "empty-chunk"
                        break
                    generated.extend(int(t) for t in new)
                    line_tokens += len(new)
                    ids = sequences
                    if any(t in stop_ids for t in new):
                        line_stop = "eot"
                        break
                    text = self.fmt.tokenizer.decode(new, skip_special_tokens=False)
                    if "\n" in text or "\r" in text:
                        line_stop = "newline"
                        break
                if line_stop == "empty-chunk":
                    m.stop_reason = "empty-chunk"
                    break
                if line_stop == "eot" and line_i + 1 < len(target_langs):
                    trim_trailing_stop()
                    continue
                if line_stop == "eot":
                    m.stop_reason = "eot"
                    break
                if line_stop == "newline":
                    trim_trailing_line_break()
                if line_tokens >= line_cap and line_i + 1 < len(target_langs):
                    m.stop_reason = "max-turn-tokens"

        m.turn_ms = (time.perf_counter() - t0) * 1000
        m.output_tokens = len(generated)
        if m.first_token_ms is not None and len(generated) > 1:
            decode_s = (m.turn_ms - m.first_token_ms) / 1000
            if decode_s > 0:
                m.decode_tps = (len(generated) - 1) / decode_s
        text = self.fmt.tokenizer.decode(
            [t for t in generated if t not in stop_ids], skip_special_tokens=True
        )
        visible = self.fmt.visible_reply(text.strip())
        self._commit(target, generated, visible, m)
        if self.session_cache and self.cache is not None:
            m.cache_mib = cache_mib(self.cache)
        return visible, m

    def commit_reference_reply(self, text: str, utterance_id=None) -> TurnMetrics:
        """Tokenize-only dry run: commit a reference reply as if the model had generated it.

        Simulates the committed stream (reply tokens + turn-end token) so template
        prefix-stability and per-turn rescan cost are measurable without a model.
        """
        target = self.fmt.render(self.msgs, add_generation_prompt=True)
        cached, rewritten = self._align_to(target, conceptual=True)
        m = self._base_metrics(utterance_id, target, cached, rewritten)
        generated = self.fmt.tokenizer.encode(text, add_special_tokens=False) + [self.fmt.turn_end_id]
        m.stop_reason = "reference"
        m.output_tokens = len(generated)
        self._commit(target, generated, text, m)
        return m

    def _commit(self, target: list[int], generated: list[int], visible: str, m: TurnMetrics) -> None:
        if self.session_cache:
            self.tokens = list(target) + list(generated)
        m.committed_tokens = len(target) + len(generated)
        self.add("assistant", visible)
        self._committed_msgs = len(self.msgs)
        self.metrics.append(m)

    def summary(self, warmup_turns: int = 0) -> dict:
        """Aggregate turn metrics; warmup_turns excludes leading turns from latency stats."""
        turns = self.metrics
        timed = turns[warmup_turns:] or turns
        ttfts = [t.first_token_ms for t in timed if t.first_token_ms is not None]
        out = {
            "turns": len(turns),
            "render_tokens": sum(t.render_tokens for t in turns),
            "fed_tokens": sum(t.fed_tokens for t in turns),
            "cached_tokens": sum(t.cached_tokens for t in turns),
            "rewritten_tokens": sum(t.rewritten_tokens for t in turns),
            "output_tokens": sum(t.output_tokens for t in turns),
            "prefix_stable_turns": sum(1 for t in turns if t.prefix_stable),
            "stop_reasons": {
                r: sum(1 for t in turns if t.stop_reason == r) for r in {t.stop_reason for t in turns}
            },
            "cache_unsupported": self.cache_unsupported,
        }
        if ttfts:
            out["first_token_ms_mean"] = sum(ttfts) / len(ttfts)
            out["first_token_ms_max"] = max(ttfts)
        turn_times = [t.turn_ms for t in timed if t.turn_ms]
        if turn_times:
            out["turn_ms_mean"] = sum(turn_times) / len(turn_times)
        tps = [t.decode_tps for t in timed if t.decode_tps]
        if tps:
            out["decode_tps_mean"] = sum(tps) / len(tps)
        return out


if __name__ == "__main__":
    # Tokenizer-free self-test of the LCP/commit machinery using a stub tokenizer.
    class _StubTok:
        chat_template = "<|im_start|>enable_thinking"
        eos_token_id = 9
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return {"<|im_end|>": 9}.get(tok, 0)

        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 50 + 10 for c in text]

        def apply_chat_template(self, msgs, **kw):
            ids = []
            for msg in msgs:
                ids += [1] + self.encode(msg["content"]) + [9]
            if kw.get("add_generation_prompt"):
                ids += [1, 2]
            return ids

    fmt = ModelFormat(_StubTok(), options=FormatOptions(history_template="stock"))
    sess = ChatSession(fmt)
    sess.add("user", "hola")
    m1 = sess.commit_reference_reply("hi", utterance_id=1)
    assert m1.cached_tokens == 0 and m1.prefix_stable
    sess.add("user", "que tal")
    m2 = sess.commit_reference_reply("fine", utterance_id=2)
    # stub template re-renders the assistant turn without the [1,2] gen prompt: tail rescan
    assert m2.rewritten_tokens > 0 and m2.cached_tokens > 0
    s = sess.summary()
    assert s["turns"] == 2 and s["stop_reasons"] == {"reference": 2}
    print("chat_session self-test ok:", s)
