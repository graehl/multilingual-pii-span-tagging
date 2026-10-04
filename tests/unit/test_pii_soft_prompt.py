import copy

import pytest
import torch
from transformers import XLMRobertaConfig

from scripts.pii_layered_head_model import LayerConcatForTokenClassification
from scripts.pii_soft_prompt import SoftPromptClassifier


def model():
    config = XLMRobertaConfig(
        vocab_size=40,
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=16,
        num_labels=5,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    config.pii_head_architecture = "concat_encoder_layers"
    config.pii_head_kind = "affine"
    config.pii_head_rank = 0
    config.pii_encoder_layers = [2]
    config.pii_classifier_dropout = 0
    return LayerConcatForTokenClassification(config).eval()


def test_prompt_gradients_routes_and_original_alignment():
    base = model()
    base.requires_grad_(False)
    wrapped = SoftPromptClassifier(base, 8, 2, ["en", "ar"])
    ids = torch.tensor([[0, 5, 6, 2, 1], [0, 7, 2, 1, 1]])
    mask = ids.ne(1).long()
    output = wrapped(ids, mask, torch.tensor([0, 0]), output_hidden_states=True)
    assert output.logits.shape == (2, 5, 5)
    assert all(value.shape == (2, 5, 12) for value in output.hidden_states)
    output.logits[:, 1:3].square().sum().backward()
    assert wrapped.shared.grad.abs().sum() > 0
    assert wrapped.language.grad[0].abs().sum() > 0
    assert wrapped.language.grad[1].abs().sum() == 0
    assert all(p.grad is None for p in base.parameters())
    with pytest.raises(ValueError, match="outside inventory"):
        wrapped(ids, mask, torch.tensor([2, 0]))
    assert torch.isfinite(wrapped(ids, mask, torch.tensor([-1, -1])).logits).all()


def test_controls_initial_equivalence_padding_and_reload(tmp_path):
    base = model()
    ids = torch.tensor([[0, 5, 6, 2, 1]])
    mask = ids.ne(1).long()
    torch.manual_seed(123)
    shared = SoftPromptClassifier(copy.deepcopy(base), 10, 0, ["en", "ar"])
    torch.manual_seed(123)
    language = SoftPromptClassifier(copy.deepcopy(base), 8, 2, ["en", "ar"])
    routes = torch.tensor([1])
    torch.testing.assert_close(shared(ids, mask, routes).logits, language(ids, mask, routes).logits)
    none = SoftPromptClassifier(base, 0, 0, ["en", "ar"])
    assert torch.equal(none(ids, mask, routes).logits, base(input_ids=ids, attention_mask=mask).logits)
    short = language(ids[:, :4], mask[:, :4], routes).logits
    torch.testing.assert_close(short, language(ids, mask, routes).logits[:, :4])
    torch.save(language.state_dict(), tmp_path / "state.pt")
    restored = SoftPromptClassifier(model(), 8, 2, ["en", "ar"])
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    torch.testing.assert_close(short, restored(ids[:, :4], mask[:, :4], routes).logits)
