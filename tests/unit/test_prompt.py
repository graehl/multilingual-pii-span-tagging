import pytest

import prompt


class FakeTokenizer:
    def apply_chat_template(
        self,
        msgs,
        *,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        parts = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                rendered = []
                for item in content:
                    rendered.append(
                        f"[{item.get('source_lang_code', '-')}"
                        f"->{item.get('target_lang_code', '-')}]"
                        f"{item.get('text', '')}"
                    )
                content = "\n".join(rendered)
            parts.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        text = "".join(parts)
        return [ord(c) for c in text] if tokenize else text


@pytest.mark.unit
def test_chat_template_thinking_kwargs_modes():
    assert prompt.chat_template_thinking_kwargs("off") == {"enable_thinking": False}
    assert prompt.chat_template_thinking_kwargs("off", omit_off=True) == {}
    assert prompt.chat_template_thinking_kwargs("on") == {"enable_thinking": True}
    assert prompt.chat_template_thinking_kwargs("auto") == {}


@pytest.mark.unit
def test_strip_leading_think_block_keeps_final_answer():
    text = "<think>\nCheck source terms.\n</think>\n\nFinal translation."

    assert prompt.strip_leading_think_block(text) == "Final translation."


@pytest.mark.unit
def test_strip_generated_think_prefix_when_template_opened_tag():
    text = "Check source terms.\n</think>\n\nFinal translation."

    assert prompt.strip_leading_think_block(text) == "Final translation."


@pytest.mark.unit
def test_strip_leading_think_block_leaves_plain_answer():
    assert prompt.strip_leading_think_block("Final translation.") == "Final translation."


@pytest.mark.unit
def test_promptline_matches_messages_plus_template_for_multiline_gemma_input():
    tokenizer = FakeTokenizer()
    line = "Paragraph one.\n\nParagraph two."
    system = "Translate Chinese to English."

    via_promptline = prompt.promptline(
        line,
        tokenizer,
        sl="zh",
        tl="en",
        system=system,
        enable_thinking=None,
    )
    via_messages = prompt.promptmsgs(
        prompt.messageline(line, sl="zh", tl="en", system=system),
        tokenizer,
        enable_thinking=None,
    )

    assert via_promptline == via_messages
    assert "[zh->en]Paragraph one.\n\nParagraph two." in via_promptline
    assert "<system>Translate Chinese to English.</system>" in via_promptline


@pytest.mark.unit
def test_prompt_strategy_rollout_appends_final_source_user():
    tokenizer = FakeTokenizer()
    strategy = prompt.resolve_prompt_strategy(
        message_specs=[
            "system:{system}",
            "user:You are a professional translator.",
            "assistant:Understood. I will translate faithfully.",
        ]
    )

    rendered = prompt.promptline(
        "你好",
        tokenizer,
        system="Translate Chinese to English.",
        strategy=strategy,
    )

    assert rendered == (
        "<system>Translate Chinese to English.</system>"
        "<user>You are a professional translator.</user>"
        "<assistant>Understood. I will translate faithfully.</assistant>"
        "<user>你好</user>"
        "<assistant>"
    )


@pytest.mark.unit
def test_prompt_strategy_jsonl_embeds_source(tmp_path):
    strategy_path = tmp_path / "prompt-strategy.jsonl"
    strategy_path.write_text(
        '{"role":"system","content":"{system}"}\n'
        '{"role":"user","content":"Translate from {source_lang} to {target_lang}:\\n{source}"}\n',
        encoding="utf-8",
    )

    strategy = prompt.resolve_prompt_strategy(strategy_jsonl=str(strategy_path))
    rendered = prompt.promptline(
        "病原体",
        FakeTokenizer(),
        sl="zh",
        tl="en",
        system="Translate.",
        strategy=strategy,
    )

    assert rendered == ("<system>Translate.</system><user>Translate from zh to en:\n病原体</user><assistant>")


@pytest.mark.unit
def test_prompt_strategy_plain_source_preserves_lang_code_content():
    strategy = prompt.resolve_prompt_strategy(message_specs=["user:{source}"])

    rendered = prompt.promptline(
        "病原体",
        FakeTokenizer(),
        sl="zh",
        tl="en",
        system="ignored without {system}",
        strategy=strategy,
    )

    assert rendered == "<user>[zh->en]病原体</user><assistant>"


@pytest.mark.unit
def test_raw_strategy_bypasses_template(tmp_path):
    strategy_file = tmp_path / "raw.jsonl"
    strategy_file.write_text(
        '{"role": "raw", "content": "<|im_start|>user\\nExample: {source}\\n<think>brief check</think>\\nAnswer<|im_end|>\\n<|im_start|>assistant\\n"}\n'
    )
    strategy = prompt.resolve_prompt_strategy(strategy_jsonl=str(strategy_file))
    rendered = prompt.promptline("你好", FakeTokenizer(), strategy=strategy)
    assert (
        rendered
        == "<|im_start|>user\nExample: 你好\n<think>brief check</think>\nAnswer<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.mark.unit
def test_raw_strategy_requires_source(tmp_path):
    strategy_file = tmp_path / "raw.jsonl"
    strategy_file.write_text('{"role": "raw", "content": "no placeholder"}\n')
    with pytest.raises(ValueError, match="raw prompt strategy must contain"):
        prompt.resolve_prompt_strategy(strategy_jsonl=str(strategy_file))


@pytest.mark.unit
def test_raw_strategy_rejects_mixed_chat_messages(tmp_path):
    strategy_file = tmp_path / "raw.jsonl"
    strategy_file.write_text('{"role": "system", "content": "hi"}\n{"role": "raw", "content": "{source}"}\n')
    with pytest.raises(ValueError, match="cannot also contain chat-message"):
        prompt.resolve_prompt_strategy(strategy_jsonl=str(strategy_file))


@pytest.mark.unit
def test_postedit_rule_rewrites_rendered_template(tmp_path):
    strategy_file = tmp_path / "pe.jsonl"
    strategy_file.write_text(
        '{"role": "system", "content": "Translate."}\n'
        '{"role": "user", "content": "{source}"}\n'
        '{"role": "postedit", "content": "{\\"pattern\\": \\"<assistant>$\\", \\"replace\\": \\"<assistant><think>brief</think>\\"}"}\n'
    )
    strategy = prompt.resolve_prompt_strategy(strategy_jsonl=str(strategy_file))
    rendered = prompt.promptline("你好", FakeTokenizer(), strategy=strategy)
    assert rendered.endswith("<assistant><think>brief</think>")
    assert "<system>Translate.</system>" in rendered


@pytest.mark.unit
def test_postedit_bad_json_rejected(tmp_path):
    strategy_file = tmp_path / "pe.jsonl"
    strategy_file.write_text('{"role": "postedit", "content": "not json"}\n')
    with pytest.raises(ValueError, match="postedit content must be JSON"):
        prompt.resolve_prompt_strategy(strategy_jsonl=str(strategy_file))
