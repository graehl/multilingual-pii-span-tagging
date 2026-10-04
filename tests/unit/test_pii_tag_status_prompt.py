import copy

import pytest
import torch

from scripts.pii_tag_status_prompt import TagStatusPrompt, training_status
from tests.unit.test_pii_soft_prompt import model


def configured():
    base = model()
    base.config.id2label = dict(enumerate(["O", "B-person", "I-person", "E-person", "S-person"]))
    base.requires_grad_(False)
    return TagStatusPrompt(base).eval()


def test_absent_present_unknown_gradients_and_inference():
    wrapper = configured()
    ids = torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]])
    mask = torch.ones_like(ids)
    targets = torch.tensor([[-100, 4, 0, -100], [-100, 0, 0, -100]])
    gold = wrapper.gold_status(targets)
    assert gold.tolist() == [[1], [0]]
    before = wrapper(ids, mask).logits.detach()
    # Changing invisible certainty vectors must not alter maybe-only inference.
    with torch.no_grad():
        wrapper.present.add_(3)
        wrapper.absent.sub_(2)
    torch.testing.assert_close(wrapper(ids, mask).logits, before, rtol=0, atol=0)
    output = wrapper(ids, mask, statuses=gold)
    output.logits.square().sum().backward()
    for parameter in (wrapper.maybe, wrapper.present, wrapper.absent):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert wrapper.neutral.grad is None
    assert all(p.grad is None for p in wrapper.base.parameters())
    with pytest.raises(ValueError, match="status"):
        wrapper(ids, mask, statuses=torch.tensor([[2], [0]]))


def test_neutral_has_no_label_information_and_dropout_contains_unknown():
    wrapper = configured()
    ids = torch.tensor([[0, 5, 6, 2]])
    mask = torch.ones_like(ids)
    yes, no = torch.tensor([[1]]), torch.tensor([[0]])
    torch.testing.assert_close(
        wrapper(ids, mask, statuses=yes).logits, wrapper(ids, mask, statuses=yes, neutral=True).logits
    )
    torch.testing.assert_close(
        wrapper(ids, mask, statuses=yes, neutral=True).logits,
        wrapper(ids, mask, statuses=no, neutral=True).logits,
    )
    statuses, _ = training_status(
        wrapper, torch.zeros(100, 4, dtype=torch.long), "dropout", torch.Generator().manual_seed(123)
    )
    assert set(statuses.flatten().tolist()) == {-1, 0}
    restored = copy.deepcopy(wrapper)
    restored.load_state_dict(wrapper.state_dict())
    torch.testing.assert_close(restored(ids, mask).logits, wrapper(ids, mask).logits)


def test_output_presence_preserves_o_and_unknown_channels():
    wrapper = configured()
    logits = torch.tensor([[[0.0, -2.0, -2.0, -2.0, 0.5]], [[0.0, -2.0, -2.0, -2.0, 0.5]]])
    targets = torch.tensor([[4], [0]])
    masked = wrapper.oracle_logits(logits, targets, "gold_mask")
    assert logits.argmax(-1).tolist() == [[4], [4]]
    assert masked.argmax(-1).tolist() == [[4], [0]]
    torch.testing.assert_close(masked[0], logits[0])
    torch.testing.assert_close(masked[:, :, 0], logits[:, :, 0])
    boosted = wrapper.oracle_logits(logits, targets, "positive_bias")
    torch.testing.assert_close(boosted[1], logits[1])
    torch.testing.assert_close(boosted[:, :, 0], logits[:, :, 0])
    torch.testing.assert_close(boosted[0, :, 1:], logits[0, :, 1:] + 1)
    assert torch.isfinite(torch.nn.functional.cross_entropy(masked[:, 0], targets[:, 0]))


def test_predicted_presence_ignores_special_and_padding_tokens():
    wrapper = configured()
    # Both rows have a strong special-token score, but only row0 has lexical evidence.
    ids = torch.tensor([[0, 5, 6, 2, 1], [0, 5, 6, 2, 1]])
    mask = ids.ne(1).long()
    logits = torch.full((2, 5, 5), -20.0)
    logits[:, :, 0] = 0
    logits[:, [0, 3, 4], 4] = 20
    logits[0, 1, 4] = 10
    logits[:, 2, 4] = -0.5
    presence = wrapper.predicted_presence(logits, ids, mask)
    assert presence.tolist() == [[True], [False]]
    boosted = wrapper.positive_logits(logits, presence)
    assert logits[:, 2].argmax(-1).tolist() == [0, 0]
    assert boosted[:, 2].argmax(-1).tolist() == [4, 0]
    torch.testing.assert_close(boosted[:, :, 0], logits[:, :, 0])
    torch.testing.assert_close(boosted[1], logits[1])


def test_status_feature_export_preserves_original_logits_and_positions():
    wrapper = configured()
    ids = torch.tensor([[0, 5, 6, 2, 1]])
    mask = ids.ne(1).long()
    plain = wrapper(ids, mask).logits
    exported = wrapper(ids, mask, output_hidden_states=True)
    torch.testing.assert_close(exported.logits, plain, rtol=0, atol=0)
    assert all(value.shape[:2] == ids.shape for value in exported.hidden_states)
