"""Input soft prompts for the existing Ont3 token classifier.

Added positions enter the encoder but are removed from every returned token
tensor. The original tokenizer offsets and supervised positions stay intact.
"""

from __future__ import annotations

import torch
from torch import nn


def encode_prompted(
    base,
    input_ids,
    attention_mask,
    prefix,
    prefix_mask=None,
    position_ids=None,
    output_hidden_states=False,
    **base_conditions,
):
    """Encode added positions and return only original-token outputs.

    ``base_conditions`` pass through to the base forward, e.g. the row
    ``language_ids`` a language-bias head requires.
    """
    count = prefix.shape[1]
    words = base.encoder.get_input_embeddings()(input_ids)
    embeds = torch.cat((prefix, words), dim=1)
    if prefix_mask is None:
        prefix_mask = attention_mask.new_ones(len(input_ids), count)
    mask = torch.cat((prefix_mask, attention_mask), dim=1)
    if position_ids is None:
        position_ids = mask.long().cumsum(1) * mask.long() + base.config.pad_token_id
    if int(position_ids.max()) >= base.config.max_position_embeddings:
        raise ValueError("Prompt plus text exceeds encoder position capacity")
    output = base(
        input_ids=None,
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=position_ids,
        output_hidden_states=output_hidden_states,
        **base_conditions,
    )
    for key in ("logits", "secondary_logits", "predicate_logits", "subclass_logits", "reference_type_logits"):
        value = getattr(output, key, None)
        if value is not None:
            output[key] = value[:, count:]
    if output.hidden_states is not None:
        output.hidden_states = tuple(value[:, count:] for value in output.hidden_states)
    return output


class SoftPromptClassifier(nn.Module):
    def __init__(self, base: nn.Module, shared: int, per_language: int, languages: list[str]) -> None:
        super().__init__()
        if shared < 0 or per_language < 0 or len(set(languages)) != len(languages):
            raise ValueError("Invalid prompt sizes or language inventory")
        if per_language and not languages:
            raise ValueError("Language prompts require an inventory")
        if base.config.model_type != "xlm-roberta" or base.token_offsets != (0,):
            raise ValueError("Pilot supports XLM-R with current-token features only")
        self.base = base
        self.languages = tuple(languages)
        self.prompt_count = shared + per_language
        embedding = base.encoder.get_input_embeddings().weight
        # The same ten sampled vocabulary embeddings initialize both arms.
        initial = (
            embedding[torch.randint(5, embedding.shape[0], (self.prompt_count,), device=embedding.device)]
            .detach()
            .clone()
        )
        self.shared = nn.Parameter(initial[:shared])
        self.language = nn.Parameter(initial[shared:].unsqueeze(0).repeat(len(languages), 1, 1))

    def prompt(self, routes: torch.Tensor) -> torch.Tensor:
        if routes.ndim != 1 or routes.dtype != torch.long:
            raise ValueError("Routes must be one long index per example")
        if torch.any(routes < -1) or torch.any(routes >= len(self.languages)):
            raise ValueError("Route outside inventory; use -1 for uncertain language")
        common = self.shared.unsqueeze(0).expand(len(routes), -1, -1)
        if self.language.shape[1] == 0:
            return common
        routed = self.language[routes.clamp_min(0)]
        # The uncertain route uses the mean of learned language prompts.
        routed = torch.where((routes == -1)[:, None, None], self.language.mean(0), routed)
        return torch.cat((common, routed), dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        routes: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        if routes.shape != (input_ids.shape[0],):
            raise ValueError("One language route required per input row")
        if not self.prompt_count:
            return self.base(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states
            )
        return encode_prompted(
            self.base,
            input_ids,
            attention_mask,
            self.prompt(routes),
            output_hidden_states=output_hidden_states,
        )
