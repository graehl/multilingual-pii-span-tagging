import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SYSTEM_PROMPT = ""
# Note: LoRAs trained with unsloth behave incorrectly when a system prompt is present at
# inference (even the same prompt used during training). Use empty system prompt for all
# LoRA-based chi→en translation inference. The task is unambiguous from Chinese user content.

HELPFUL_SYSTEM_PROMPT = "You are a helpful assistant."

THINKING_MODES = ("off", "on", "auto")
_THINK_BLOCK_RE = re.compile(r"^\s*(?:<think>\s*)?.*?</think>\s*", re.DOTALL)


def chat_template_thinking_kwargs(mode: str, *, omit_off: bool = False) -> dict[str, bool]:
    if mode == "auto":
        return {}
    if mode == "on":
        return {"enable_thinking": True}
    if mode == "off":
        return {} if omit_off else {"enable_thinking": False}
    raise ValueError(f"unsupported thinking mode {mode!r}")


def strip_leading_think_block(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text, count=1).lstrip()


IM_END = "<|im_end|>"
IM_START = "<|im_start|>"


def qwen_role_start(role):
    return f"{IM_START}{role}\n"


ROLES = ["system", "user", "assistant"]
ROLE_STARTS = {r: qwen_role_start(r) for r in ROLES}
ROLE_ENDS = [f"{IM_END}\n"]


def strip_end(x):
    for t in ROLE_ENDS:
        if x.endswith(t):
            x = x[: -len(t)]
    return x


def role_unstart(x: str, role="user") -> str:
    if role in ROLE_STARTS:
        s = ROLE_STARTS[role]
        if x.startswith(s):
            for t in ROLE_ENDS:
                if x.endswith(t):
                    x = x[: -len(t)]
            x = x[len(s) :]
    return x


def extract_first_im_block(
    text: str, role="user", start_token="<|im_start|>", end_token="<|im_end|>"
) -> str | None:
    start_token += role
    start_idx = text.find(start_token)
    if start_idx == -1:
        return None
    content_start = start_idx + len(start_token)
    if content_start >= len(text):
        return None
    if text[content_start] == "\n":
        content_start += 1
    end_idx = text.find(end_token, content_start)
    if end_idx == -1:
        return None
    return text[content_start:end_idx]


def message(role, content):
    return {"role": role, "content": content}


def _content_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c["text"] for c in content if isinstance(c, dict) and "text" in c)
    return str(content)


def merge_leading_system_into_user(msgs: list[dict]) -> list[dict]:
    """Merge only an adjacent leading system+user pair."""
    if len(msgs) < 2 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        return msgs
    system_text = _content_text(msgs[0].get("content", ""))
    user_text = _content_text(msgs[1].get("content", ""))
    merged = [message("user", f"{system_text}\n\n{user_text}")]
    merged.extend(msgs[2:])
    return merged


def normalize_msgs(msgs: list) -> list:
    """Wrap multimodal content dicts as plain strings for tokenizer fallback.
    Preserves system role — Gemma-4 IT supports it natively."""
    out = []
    for msg in msgs:
        content = msg["content"]
        if isinstance(content, list):
            out.append({"role": msg["role"], "content": _content_text(content)})
        else:
            out.append(msg)
    return out


class TemplateRenderer:
    """Chat-template rendering shared by hf-translate batch decode and chat sessions.

    Wraps apply_chat_template with the fallbacks this repo needs: prefer the
    processor's template (translategemma processor-swap, multimodal content),
    fall back to the base tokenizer with normalized plain-text messages, and
    merge a leading system role into the first user turn for templates that
    reject system. Raw-text rendering directives and postedit rules from prompt
    strategies bypass the chat template entirely.

    ``template_override`` replaces the template string for both render paths
    (e.g. a prefix-stable chat-session history template); thinking kwargs are
    only passed on the primary path, matching the historical closure behavior.
    """

    def __init__(self, tokenizer, processor=None, thinking_kwargs=None, template_override: str | None = None):
        self.tokenizer = tokenizer
        self.processor = processor if processor is not None else tokenizer
        self.thinking_kwargs = dict(thinking_kwargs or {})
        self.template_override = template_override

    def _template_kwargs(self) -> dict:
        return {"chat_template": self.template_override} if self.template_override else {}

    def _tokenizer_fallback_text(self, msgs: list, *, add_generation_prompt: bool = True) -> str:
        normalized = normalize_msgs(msgs)
        try:
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **self._template_kwargs(),
            )
        except Exception:
            merged = merge_leading_system_into_user(normalized)
            if merged == normalized:
                raise
            return self.tokenizer.apply_chat_template(
                merged,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **self._template_kwargs(),
            )

    def token_ids(self, msgs: list, *, add_generation_prompt: bool = True) -> list[int]:
        _, raw_text, postedit_rules = split_rendering_directives(msgs)
        if raw_text is not None or postedit_rules:
            return self.tokenizer.encode(
                self.text(msgs, add_generation_prompt=add_generation_prompt),
                add_special_tokens=False,
            )
        try:
            return self.processor.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors=None,
                return_dict=False,  # transformers 5.x changed default to True
                **self._template_kwargs(),
                **self.thinking_kwargs,
            )
        except Exception:
            # Gemma-4 IT: processor may be overridden (e.g. translategemma template)
            # and also requires multimodal content dicts / no leading system role.
            # Fall back to the base-model tokenizer with normalized plain-text messages.
            text = self._tokenizer_fallback_text(msgs, add_generation_prompt=add_generation_prompt)
            return self.tokenizer.encode(text, add_special_tokens=False)

    def text(self, msgs: list, *, add_generation_prompt: bool = True) -> str:
        chat_msgs, raw_text, postedit_rules = split_rendering_directives(msgs)
        if raw_text is not None:
            return apply_postedits(raw_text, postedit_rules)
        if postedit_rules:
            return apply_postedits(
                self.text(chat_msgs, add_generation_prompt=add_generation_prompt), postedit_rules
            )
        try:
            return self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
                **self._template_kwargs(),
                **self.thinking_kwargs,
            )
        except Exception:
            return self._tokenizer_fallback_text(msgs, add_generation_prompt=add_generation_prompt)


@dataclass(frozen=True)
class PromptChoice:
    prompt_id: int
    text: str
    weight: float = 1.0
    source: str | None = None


@dataclass(frozen=True)
class PromptMessageTemplate:
    role: str
    content: str
    source: str | None = None


PROMPT_POLICIES = ("fixed", "rotate", "sample", "expand", "blend")


def read_prompt_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def load_prompt_set(list_file: str | Path) -> list[PromptChoice]:
    list_path = Path(list_file)
    prompts: list[PromptChoice] = []
    for i, raw in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompt_path = Path(line)
        if not prompt_path.is_absolute():
            prompt_path = list_path.parent / prompt_path
        prompts.append(PromptChoice(prompt_id=i, text=read_prompt_file(prompt_path), source=str(prompt_path)))
    return prompts


def load_prompt_prefix(prefix: str | Path) -> list[PromptChoice]:
    base = Path(prefix)
    prompts: list[PromptChoice] = []
    i = 1
    while True:
        path = Path(f"{base}.{i}")
        if not path.exists():
            break
        prompts.append(PromptChoice(prompt_id=i, text=read_prompt_file(path), source=str(path)))
        i += 1
    return prompts


def parse_prompt_choice(spec: str | None, max_prompt_id: int) -> list[tuple[int, float]]:
    if max_prompt_id < 1:
        return []
    if spec is None:
        return [(1, 1.0)]
    spec = spec.strip()
    if not spec:
        return [(1, 1.0)]
    if spec.upper() == "ALL":
        return [(i, 1.0) for i in range(1, max_prompt_id + 1)]
    merged: dict[int, float] = {}
    order: list[int] = []
    for piece in spec.split(","):
        token = piece.strip()
        if not token:
            continue
        sid, sep, sweight = token.partition(":")
        prompt_id = int(sid)
        if prompt_id < 1 or prompt_id > max_prompt_id:
            raise ValueError(f"prompt id {prompt_id} out of range 1..{max_prompt_id}")
        weight = float(sweight) if sep else 1.0
        if weight <= 0:
            raise ValueError(f"prompt weight must be >0, got {weight} for prompt id {prompt_id}")
        if prompt_id not in merged:
            order.append(prompt_id)
            merged[prompt_id] = 0.0
        merged[prompt_id] += weight
    return [(prompt_id, merged[prompt_id]) for prompt_id in order]


def resolve_prompt_choices(
    system_prompt: str,
    prompt_set: str | None = None,
    prompt_prefix: str | None = None,
    prompt_choice: str | None = None,
) -> list[PromptChoice]:
    if prompt_prefix:
        prompts = load_prompt_prefix(prompt_prefix)
    elif prompt_set:
        prompts = load_prompt_set(prompt_set)
    else:
        return [PromptChoice(prompt_id=1, text=system_prompt, weight=1.0, source=None)]
    if not prompts:
        source = prompt_prefix if prompt_prefix else prompt_set
        raise ValueError(f"prompt source {source} had no prompts")
    selected = parse_prompt_choice(prompt_choice or "ALL", len(prompts))
    by_id = {p.prompt_id: p for p in prompts}
    return [
        PromptChoice(
            prompt_id=prompt_id, text=by_id[prompt_id].text, weight=weight, source=by_id[prompt_id].source
        )
        for prompt_id, weight in selected
    ]


def _read_template_content(content: str, base_dir: Path | None = None) -> str:
    if content.startswith("@"):
        path = Path(content[1:])
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return read_prompt_file(path)
    return content


def parse_prompt_message_spec(spec: str) -> PromptMessageTemplate:
    for sep in (":", "="):
        role, found, content = spec.partition(sep)
        if found:
            role = role.strip()
            if not role:
                raise ValueError(f"prompt message spec has empty role: {spec!r}")
            return PromptMessageTemplate(role=role, content=_read_template_content(content))
    raise ValueError(f"prompt message spec must be ROLE:template or ROLE=template, got {spec!r}")


# Rendering directives: pseudo-roles in a prompt strategy that change how the
# final prompt string is produced rather than adding a chat message.
# - raw: full template bypass; content is the model's wire-format prompt with
#   the usual placeholders. For surfaces the chat template cannot express
#   (e.g. a 1-shot exemplar containing a <think> block, which Qwen templates
#   strip from history). enable_thinking does not apply: the raw text decides.
# - postedit: content is a JSON object {"pattern": regex, "replace": str,
#   "count": int (0=all)} applied, in file order, to the *rendered* template
#   output. For surgical edits where the template is otherwise right.
RAW_ROLE = "raw"
POSTEDIT_ROLE = "postedit"


def _parse_postedit_rule(content: str, where: str) -> str:
    try:
        rule = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: postedit content must be JSON: {exc}") from exc
    if not isinstance(rule, dict) or "pattern" not in rule:
        raise ValueError(f'{where}: postedit needs {{"pattern": ..., "replace": ..., "count": ...}}')
    try:
        re.compile(rule["pattern"])
    except re.error as exc:
        raise ValueError(f"{where}: bad postedit pattern: {exc}") from exc
    rule.setdefault("replace", "")
    rule.setdefault("count", 0)
    return json.dumps(rule)


def load_prompt_strategy_jsonl(path: str | Path) -> list[PromptMessageTemplate]:
    strategy_path = Path(path)
    templates: list[PromptMessageTemplate] = []
    with strategy_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{strategy_path}:{lineno}: expected a JSON object")
            role = str(rec.get("role", "")).strip()
            if not role:
                raise ValueError(f"{strategy_path}:{lineno}: missing nonempty role")
            if "content" in rec:
                content = str(rec["content"])
            elif "text" in rec:
                content = str(rec["text"])
            elif "path" in rec:
                content = "@" + str(rec["path"])
            else:
                raise ValueError(f"{strategy_path}:{lineno}: expected content, text, or path")
            content = _read_template_content(content, strategy_path.parent)
            if role == POSTEDIT_ROLE:
                content = _parse_postedit_rule(content, f"{strategy_path}:{lineno}")
            templates.append(
                PromptMessageTemplate(
                    role=role,
                    content=content,
                    source=f"{strategy_path}:{lineno}",
                )
            )
    return templates


def resolve_prompt_strategy(
    strategy_jsonl: str | None = None,
    message_specs: list[str] | None = None,
) -> list[PromptMessageTemplate] | None:
    templates: list[PromptMessageTemplate] = []
    if strategy_jsonl:
        templates.extend(load_prompt_strategy_jsonl(strategy_jsonl))
    for spec in message_specs or []:
        templates.append(parse_prompt_message_spec(spec))
    if not templates:
        return None
    raw_templates = [t for t in templates if t.role == RAW_ROLE]
    if raw_templates:
        if any(t.role not in (RAW_ROLE, POSTEDIT_ROLE) for t in templates):
            raise ValueError("a raw prompt strategy cannot also contain chat-message templates")
        if not any("{source}" in t.content for t in raw_templates):
            raise ValueError("raw prompt strategy must contain {source}")
        return templates
    substantive = [t for t in templates if t.role != POSTEDIT_ROLE]
    if not any("{source}" in template.content for template in substantive):
        templates.append(PromptMessageTemplate(role="user", content="{source}", source="auto-source"))
    return templates


def _format_prompt_template(
    template: str,
    *,
    line: str,
    system: str | None = None,
    sl: str | None = None,
    tl: str | None = None,
) -> str:
    return (
        template.replace("{source}", line)
        .replace("{system}", system or "")
        .replace("{source_lang}", sl or "")
        .replace("{target_lang}", tl or "")
    )


RAW_CARRIER_ROLE = "__raw__"
POSTEDIT_CARRIER_ROLE = "__postedit__"


def render_messages(
    line: str,
    *,
    sl=None,
    tl=None,
    system=None,
    strategy: list[PromptMessageTemplate] | None = None,
) -> list[dict]:
    if not strategy:
        msgs = []
        if system:
            msgs.append(message("system", system))
        msgs.append(message("user", usercontent(line, sl, tl)))
        return msgs

    msgs = []
    for template in strategy:
        if template.role == POSTEDIT_ROLE:
            msgs.append(message(POSTEDIT_CARRIER_ROLE, template.content))
        elif template.role == RAW_ROLE:
            msgs.append(
                message(
                    RAW_CARRIER_ROLE,
                    _format_prompt_template(template.content, line=line, system=system, sl=sl, tl=tl),
                )
            )
        elif template.role == "user" and template.content.strip() == "{source}":
            msgs.append(message(template.role, usercontent(line, sl, tl)))
        else:
            msgs.append(
                message(
                    template.role,
                    _format_prompt_template(template.content, line=line, system=system, sl=sl, tl=tl),
                )
            )
    return msgs


def split_rendering_directives(msgs: list[dict]) -> tuple[list[dict], str | None, list[dict]]:
    """Separate rendered messages from rendering directives.

    Returns (chat_msgs, raw_text, postedit_rules); raw_text is the full
    template-bypass prompt when the strategy used a raw directive (multiple
    raw lines concatenate in order).
    """
    chat_msgs: list[dict] = []
    raw_parts: list[str] = []
    rules: list[dict] = []
    for msg in msgs:
        role = msg.get("role")
        if role == RAW_CARRIER_ROLE:
            raw_parts.append(str(msg.get("content", "")))
        elif role == POSTEDIT_CARRIER_ROLE:
            rules.append(json.loads(str(msg.get("content", "{}"))))
        else:
            chat_msgs.append(msg)
    return chat_msgs, ("".join(raw_parts) if raw_parts else None), rules


def apply_postedits(text: str, rules: list[dict]) -> str:
    for rule in rules:
        text = re.sub(rule["pattern"], rule.get("replace", ""), text, count=int(rule.get("count", 0)))
    return text


def merge_leading_system_into_first_user(msgs: list[dict]) -> list[dict]:
    """Merge a leading system message into the first later user message."""
    if not msgs or msgs[0].get("role") != "system":
        return msgs
    user_idx = next((i for i, msg in enumerate(msgs[1:], start=1) if msg.get("role") == "user"), None)
    if user_idx is None:
        return msgs
    system_text = _content_text(msgs[0].get("content", ""))
    user_text = _content_text(msgs[user_idx].get("content", ""))
    merged: list[dict] = []
    for i, msg in enumerate(msgs[1:], start=1):
        if i == user_idx:
            merged.append(message("user", f"{system_text}\n\n{user_text}"))
        else:
            merged.append(dict(msg))
    return merged


def prompt_groups(
    count: int,
    choices: list[PromptChoice],
    policy: str,
    seed: int = 0,
) -> list[list[PromptChoice]]:
    if not choices:
        return [[] for _ in range(count)]
    if policy not in PROMPT_POLICIES:
        raise ValueError(f"unknown prompt policy {policy!r}; expected one of {PROMPT_POLICIES}")
    if policy == "fixed":
        return [[choices[0]] for _ in range(count)]
    if policy == "rotate":
        n = len(choices)
        return [[choices[i % n]] for i in range(count)]
    if policy == "sample":
        rng = random.Random(seed)
        weights = [c.weight for c in choices]
        return [[rng.choices(choices, weights=weights, k=1)[0]] for _ in range(count)]
    if policy in ("expand", "blend"):
        return [list(choices) for _ in range(count)]
    raise ValueError(f"unsupported prompt policy {policy!r}")


def usercontent(line: str, sl=None, tl=None) -> str | list[dict]:
    """multimodal translategemma content if sl, tl else text"""
    if sl and tl:
        d = {}
        d.update(type="text", source_lang_code=sl, target_lang_code=tl, text=line)
        return [d]
    return line


def promptmsgs(msgs: list[dict], tokenizer, tensors=False, enable_thinking=None) -> str:
    chat_msgs, raw_text, rules = split_rendering_directives(msgs)
    if raw_text is not None or rules:
        if tensors:
            raise ValueError("raw/postedit prompt directives do not support tensors=True")
        text = (
            raw_text
            if raw_text is not None
            else promptmsgs(chat_msgs, tokenizer, enable_thinking=enable_thinking)
        )
        return apply_postedits(text, rules)
    extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    if tensors:
        kwargs = dict(
            tokenize=False, add_generation_prompt=True, return_tensors="pt", return_dict=True, **extra
        )
    else:
        kwargs = dict(tokenize=False, add_generation_prompt=True, **extra)
    try:
        return tokenizer.apply_chat_template(msgs, **kwargs)
    except TypeError:
        # tokenizer doesn't support enable_thinking (e.g., translategemma)
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(msgs, **kwargs)


def promptusercontent(content, tokenizer, tensors=False, system=None, enable_thinking=None) -> str:
    msgs = []
    if system:
        msgs.append(message("system", system))
    msgs.append(message("user", content))
    try:
        return promptmsgs(msgs, tokenizer, tensors=tensors, enable_thinking=enable_thinking)
    except Exception:
        if system:
            # Template doesn't support a leading system role (e.g. Gemma-4);
            # merge system content into the first user message and retry.
            return promptmsgs(
                merge_leading_system_into_user(msgs),
                tokenizer,
                tensors=tensors,
                enable_thinking=enable_thinking,
            )
        raise


def promptline(
    line: str,
    tokenizer,
    sl=None,
    tl=None,
    tensors=False,
    system=None,
    enable_thinking=None,
    strategy: list[PromptMessageTemplate] | None = None,
) -> str:
    msgs = render_messages(line, sl=sl, tl=tl, system=system, strategy=strategy)
    try:
        return promptmsgs(msgs, tokenizer, tensors=tensors, enable_thinking=enable_thinking)
    except Exception:
        fallback = merge_leading_system_into_first_user(msgs)
        if fallback != msgs:
            return promptmsgs(fallback, tokenizer, tensors=tensors, enable_thinking=enable_thinking)
        raise


def messageline(
    line: str,
    sl=None,
    tl=None,
    system=None,
    strategy: list[PromptMessageTemplate] | None = None,
):
    return render_messages(line, sl=sl, tl=tl, system=system, strategy=strategy)


def write_yaml_docs(file, xs, key):
    if isinstance(file, io.IOBase):
        for i, x in enumerate(xs):
            if i:
                file.write("---\n")
            yaml.safe_dump(
                {key: x} if key else x,
                file,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
    else:
        with open(file, "w", encoding="utf-8") as f:
            return write_yaml_docs(f, xs, key)
