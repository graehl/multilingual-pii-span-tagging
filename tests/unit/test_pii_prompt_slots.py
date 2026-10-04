import pytest
import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

from scripts.pii_prompt_slots import (
    CONFIG_KEY,
    PromptSlots,
    from_config,
    install,
    layout_slots,
    target_bounds,
)

TYPES = ("email", "person_name")
PAD = 1


def _prompt(layout, gold=True, language=False, hidden=4):
    slots = layout_slots(TYPES, layout, gold, language)
    initial = torch.arange(len(slots), dtype=torch.float).unsqueeze(1).repeat(1, hidden) + 100
    languages = ("de", "en") if language else ()
    language_initial = torch.full((2, hidden), 50.0) if language else None
    return PromptSlots(hidden, slots, TYPES, languages, initial=initial, language_initial=language_initial)


def _splice(prompt, training, statuses=None, language_ids=None):
    # Row: <s> ctx ctx T T </s>, target at token indices 3..4; a padded second row
    # holds <s> T </s> pad pad pad with its target at index 1.
    words = torch.arange(12, dtype=torch.float).view(2, 6, 1).repeat(1, 1, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    start = torch.tensor([3, 1])
    end = torch.tensor([4, 1])
    return prompt.splice(words, mask, start, end, statuses, language_ids, training, PAD)


def _roles_in_order(prompt, embeds, keep, row):
    """Label every position of the spliced row: t<index> for text, the slot role otherwise."""
    labels = [None] * embeds.size(1)
    for j, index in enumerate(keep[row].tolist()):
        labels[index] = f"t{j}"
    slot = 0
    for i, value in enumerate(labels):
        if value is None:
            labels[i] = f"{prompt.slots[slot]['role']}:{prompt.slots[slot]['label']}"
            slot += 1
    return labels


def test_maybe_first_puts_maybe_before_everything_and_gold_after_the_target():
    prompt = _prompt("maybe-first")
    embeds, _, _, keep = _splice(prompt, training=False)
    assert _roles_in_order(prompt, embeds, keep, 0) == [
        "maybe:email",
        "maybe:person_name",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "status:email",
        "status:person_name",
        "t5",
    ]


def test_gold_first_puts_maybe_between_context_and_target():
    prompt = _prompt("gold-first")
    embeds, _, _, keep = _splice(prompt, training=False)
    assert _roles_in_order(prompt, embeds, keep, 0) == [
        "status:email",
        "status:person_name",
        "t0",
        "t1",
        "t2",
        "maybe:email",
        "maybe:person_name",
        "t3",
        "t4",
        "t5",
    ]


def test_an_empty_target_places_every_slot_and_keeps_the_text():
    # <s> </s> pad: a content-free segment, as training data and serving inputs contain.
    prompt = _prompt("maybe-first")
    words = torch.arange(3, dtype=torch.float).view(1, 3, 1).repeat(1, 1, 4)
    mask = torch.tensor([[1, 1, 0]])
    start, end = (torch.tensor([value]) for value in target_bounds([(0, 0), (0, 0)]))
    embeds, _, _, keep = prompt.splice(words, mask, start, end, None, None, False, PAD)
    assert _roles_in_order(prompt, embeds, keep, 0)[:6] == [
        "maybe:email",
        "maybe:person_name",
        "t0",
        "status:email",
        "status:person_name",
        "t1",
    ]


def test_adjacent_pairs_sit_between_context_and_target():
    prompt = _prompt("adjacent-pairs")
    embeds, _, _, keep = _splice(prompt, training=False)
    assert _roles_in_order(prompt, embeds, keep, 0)[3:7] == [
        "maybe:email",
        "status:email",
        "maybe:person_name",
        "status:person_name",
    ]


def test_training_only_slots_hide_at_inference_and_for_unknown_status_without_moving_text():
    prompt = _prompt("maybe-first")
    statuses = torch.tensor([[2, 0], [1, 2]])  # row 0: email present, person unknown
    _, train_mask, train_positions, keep = _splice(prompt, training=True, statuses=statuses)
    _, eval_mask, eval_positions, _ = _splice(prompt, training=False, statuses=statuses)
    gold = [7, 8]  # row 0 positions of the two status slots
    assert train_mask[0, gold].tolist() == [1, 0]
    assert eval_mask[0, gold].tolist() == [0, 0]
    # Hiding a hint never moves a text position.
    assert torch.equal(train_positions, eval_positions)
    assert train_positions[0, keep[0]].tolist() == [PAD + 1 + i for i in (2, 3, 4, 5, 6, 9)]


def test_padding_keeps_the_pad_position_and_after_target_slots_precede_it():
    prompt = _prompt("maybe-first")
    _, mask, positions, keep = _splice(prompt, training=False)
    # Row 1: <s> T </s> pad pad pad; gold slots land after T, before </s>.
    assert keep[1].tolist() == [2, 3, 6, 7, 8, 9]
    assert mask[1, keep[1]].tolist() == [1, 1, 1, 0, 0, 0]
    assert positions[1, keep[1][3:]].tolist() == [PAD, PAD, PAD]


def test_language_slots_use_the_row_language_and_the_mean_for_unknown():
    prompt = _prompt("grouped", language=True)
    with torch.no_grad():
        prompt.language_vectors[0, 0] = 10.0
        prompt.language_vectors[0, 1] = 20.0
    vectors = prompt.slot_vectors(2, None, torch.tensor([1, -1]), "cpu", torch.float)
    assert vectors[0, 0, 0].item() == 20.0
    assert vectors[1, 0, 0].item() == 15.0


class _Stub(nn.Module):
    """Stands in for the tagger: logits are the summed input embedding per position."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(20, 4)
        self.config = type(
            "C", (), {"pad_token_id": PAD, "update": lambda self, d: self.__dict__.update(d)}
        )()
        self.seen = None

    @property
    def base_model(self):
        return self

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None, **_kwargs):
        assert input_ids is None
        self.seen = (attention_mask, position_ids)
        return TokenClassifierOutput(logits=inputs_embeds.sum(dim=-1, keepdim=True))


def test_installed_slots_leave_the_logits_aligned_with_the_original_tokens():
    model = _Stub().eval()
    install(model, _prompt("gold-first"))
    ids = torch.tensor([[0, 5, 6, 7, 8, 2]])
    out = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        prompt_target_start=torch.tensor([3]),
        prompt_target_end=torch.tensor([4]),
    )
    expected = model.embed(ids).sum(dim=-1, keepdim=True)
    assert torch.allclose(out.logits, expected)
    assert model.config.__dict__[CONFIG_KEY]["slots"][0]["anchor"] == "start"


def test_labels_must_not_reach_the_spliced_forward():
    model = _Stub()
    install(model, _prompt("gold-first"))
    ids = torch.tensor([[0, 5, 2]])
    with pytest.raises(ValueError, match="score labels outside the model"):
        model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=torch.zeros_like(ids),
            prompt_target_start=torch.tensor([1]),
            prompt_target_end=torch.tensor([1]),
        )


def test_config_rebuilds_the_same_slots():
    prompt = _prompt("adjacent-pairs", language=True)
    config = type("C", (), {})()
    setattr(
        config, CONFIG_KEY, {"slots": list(prompt.slots), "types": list(TYPES), "languages": ["de", "en"]}
    )
    rebuilt = from_config(config, 4)
    assert rebuilt.slots == prompt.slots and rebuilt.languages == ("de", "en")


def test_target_bounds_are_the_nonzero_width_offsets():
    assert target_bounds([(0, 0), (0, 0), (0, 3), (4, 7), (0, 0)]) == (2, 3)
    # An empty target anchors both target-relative slots before the trailing token.
    start, end = target_bounds([(0, 0), (0, 0)])
    assert (start, end + 1) == (1, 1)
    with pytest.raises(ValueError):
        target_bounds([(0, 0)])
