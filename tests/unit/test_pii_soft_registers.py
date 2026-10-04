import pytest
import torch
from torch import nn

from scripts.pii_soft_registers import (
    REGISTER_CONDITIONS_KEY,
    REGISTER_CONFIG_KEY,
    SoftRegisters,
    condition_ids,
    conditions_from_config,
    encoder_blocks,
    install,
    placeholder_id,
    reserve_positions,
    reserve_window,
)


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, *_args, **_kwargs):
        # Mean pooling stands in for attention: without some mixing across positions a
        # register cannot reach the tokens that are scored, and the test below would be
        # asserting a property the stub is incapable of.
        mixed = hidden_states + hidden_states.mean(dim=1, keepdim=True)
        return (self.dense(mixed),)


class _Base(nn.Module):
    def __init__(self, hidden, layers, vocab):
        super().__init__()
        self.embeddings = nn.Embedding(vocab, hidden)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(_Block(hidden) for _ in range(layers))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids):
        states = self.embeddings(input_ids)
        for block in self.encoder.layer:
            states = block(states)[0]
        return states


class _Config:
    """Stores what `update` is given, the way a real model config does."""

    def __init__(self, hidden):
        self.hidden_size = hidden

    def update(self, values):
        self.__dict__.update(values)


class _Model(nn.Module):
    def __init__(self, hidden=8, layers=3, vocab=20):
        super().__init__()
        self.roberta = _Base(hidden, layers, vocab)
        self.config = _Config(hidden)

    @property
    def base_model(self):
        return self.roberta

    def forward(self, input_ids):
        return self.roberta(input_ids)


def test_reserve_positions_keeps_target_offsets_and_marks_registers_zero_width():
    ids = [0, 5, 6, 7, 1]
    offsets = [(0, 0), (0, 4), (5, 8), (9, 12), (0, 0)]
    new_ids, new_offsets = reserve_positions(ids, offsets, 3, filler=9)

    assert new_ids == [0, 9, 9, 9, 5, 6, 7, 1]
    assert new_offsets[:4] == [(0, 0), (0, 0), (0, 0), (0, 0)]
    # Every offset that names real characters is preserved exactly.
    assert [o for o in new_offsets if o[1] > o[0]] == [o for o in offsets if o[1] > o[0]]


def test_reserve_positions_is_a_no_op_without_registers():
    ids, offsets = [0, 5, 1], [(0, 0), (0, 3), (0, 0)]
    assert reserve_positions(ids, offsets, 0, filler=9) == (ids, offsets)


def test_write_embeddings_replaces_only_the_register_slots():
    registers = SoftRegisters(count=2, hidden=4, layers=2)
    embeddings = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    written = registers.write_embeddings(torch.cat([embeddings, torch.zeros(2, 2, 4)], dim=1)[:, :5])
    assert written.shape == (2, 5, 4)
    assert torch.equal(written[:, 0], embeddings[:, 0])
    assert torch.equal(written[0, 1:3], registers.shared)
    assert torch.equal(written[1, 1:3], registers.shared)


def test_write_embeddings_rejects_a_sequence_too_short_for_its_registers():
    registers = SoftRegisters(count=4, hidden=4, layers=2)
    with pytest.raises(ValueError, match="shorter than 4 registers"):
        registers.write_embeddings(torch.zeros(1, 3, 4))


def test_layer_delta_is_absent_when_input_only():
    assert SoftRegisters(count=2, hidden=4, layers=3, layerwise=False).layer_delta is None
    states = torch.zeros(1, 5, 4)
    unchanged = SoftRegisters(count=2, hidden=4, layers=3, layerwise=False).add_layer_delta(states, 0)
    assert torch.equal(unchanged, states)


def test_installed_registers_reach_the_content_tokens_and_receive_gradient():
    torch.manual_seed(0)
    model = _Model()
    registers = install(model, count=2, layerwise=True)
    ids = torch.tensor([[0, 9, 9, 5, 6, 1]])

    before = model(ids)
    with torch.no_grad():
        registers.layer_delta.add_(1.0)
    after = model(ids)
    # A register that only changed itself would be useless: the scored positions must move.
    assert (after[:, 3:] - before[:, 3:]).abs().max() > 1e-6

    after.sum().backward()
    assert registers.shared.grad is not None and registers.shared.grad.abs().sum() > 0
    assert registers.layer_delta.grad is not None and registers.layer_delta.grad.abs().sum() > 0
    assert model.config.__dict__.get(REGISTER_CONFIG_KEY, 2) == 2


def test_installed_registers_ignore_the_placeholder_token_id():
    torch.manual_seed(0)
    model = _Model()
    install(model, count=2, layerwise=True)
    with torch.no_grad():
        one = model(torch.tensor([[0, 9, 9, 5, 6, 1]]))
        two = model(torch.tensor([[0, 3, 4, 5, 6, 1]]))
    assert torch.allclose(one, two)


def test_encoder_blocks_reports_an_unfamiliar_layout():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.thing = nn.Linear(2, 2)

        @property
        def base_model(self):
            return self

    with pytest.raises(ValueError, match="cannot locate encoder blocks"):
        encoder_blocks(Odd())


class _Tokenizer:
    def __init__(self, mask_token_id=None, unk_token_id=None):
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id


def test_placeholder_id_prefers_mask_then_unk_then_zero():
    assert placeholder_id(_Tokenizer(mask_token_id=250001, unk_token_id=3)) == 250001
    assert placeholder_id(_Tokenizer(unk_token_id=3)) == 3
    assert placeholder_id(_Tokenizer()) == 0


def test_reserve_window_keeps_every_field_aligned():
    window = {"input_ids": [0, 7, 8, 1], "attention_mask": [1, 1, 1, 1], "token_type_ids": [0, 0, 0, 0]}
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
    reserved, moved = reserve_window(window, offsets, 2, filler=250001)
    assert reserved["input_ids"] == [0, 250001, 250001, 7, 8, 1]
    assert reserved["attention_mask"] == [1, 1, 1, 1, 1, 1]
    assert reserved["token_type_ids"] == [0, 0, 0, 0, 0, 0]
    assert moved == [(0, 0), (0, 0), (0, 0), (0, 3), (4, 7), (0, 0)]
    # Every scored offset must survive unmoved, or predictions land on the wrong characters.
    assert [span for span in moved if span[0] != span[1]] == [(0, 3), (4, 7)]


def test_reserve_window_refuses_a_field_it_cannot_align():
    window = {"input_ids": [0, 7, 1], "special_tokens_mask": [1, 0, 1]}
    with pytest.raises(ValueError, match="unknown tokenizer fields"):
        reserve_window(window, [(0, 0), (0, 3), (0, 0)], 2, filler=0)


def test_reserve_window_is_a_no_op_without_registers():
    window = {"input_ids": [0, 7, 1]}
    offsets = [(0, 0), (0, 3), (0, 0)]
    assert reserve_window(window, offsets, 0, filler=0) == (window, offsets)


def test_conditions_add_a_residual_to_their_own_slot_only():
    torch.manual_seed(0)
    model = _Model()
    registers = install(
        model,
        count=3,
        layerwise=True,
        conditions=(
            ("source", ("unknown", "terra", "nemotron", "openpii")),
            ("language", ("unknown", "en", "de")),
        ),
    )
    # A plain tensor rather than the model's own embedding call: that call is hooked, so it
    # would run write_embeddings a second time and obscure what is being asserted.
    embeddings = torch.randn(1, 7, 8)
    with torch.no_grad():
        registers.condition_residuals["source"].weight[2] = 1.0

    registers.set_conditions({"source": torch.tensor([0]), "language": torch.tensor([0])})
    zero = registers.write_embeddings(embeddings)
    registers.set_conditions({"source": torch.tensor([2]), "language": torch.tensor([0])})
    moved = registers.write_embeddings(embeddings)

    # Slot 0 carries source, slot 1 language, slot 2 is plain shared; only source moved.
    assert not torch.allclose(zero[:, 1], moved[:, 1])
    assert torch.allclose(zero[:, 2], moved[:, 2])
    assert torch.allclose(zero[:, 3], moved[:, 3])


def test_zero_initialized_conditions_reproduce_the_unconditioned_registers():
    torch.manual_seed(0)
    model = _Model()
    registers = install(
        model, count=3, layerwise=True, conditions=(("source", ("unknown", "terra", "nemotron", "openpii")),)
    )
    embeddings = torch.randn(1, 7, 8)
    registers.set_conditions({"source": torch.tensor([3])})
    conditioned = registers.write_embeddings(embeddings)
    plain = torch.cat([embeddings[:, :1], registers.shared.unsqueeze(0), embeddings[:, 4:]], dim=1)
    # An untrained condition must degrade to the shared register, not to noise; that is
    # what makes a rare convention class safe to add to the vocabulary.
    assert torch.allclose(conditioned, plain)


def test_conditioned_registers_refuse_a_forward_without_ids():
    torch.manual_seed(0)
    model = _Model()
    registers = install(
        model, count=2, layerwise=False, conditions=(("source", ("unknown", "terra", "nemotron")),)
    )
    registers.clear_conditions()
    with pytest.raises(ValueError, match="set_conditions"):
        registers.write_embeddings(model.base_model.get_input_embeddings()(torch.tensor([[0, 9, 9, 5, 1]])))


def test_set_conditions_reports_a_missing_condition():
    torch.manual_seed(0)
    model = _Model()
    registers = install(
        model,
        count=2,
        layerwise=False,
        conditions=(("source", ("unknown", "terra", "nemotron")), ("language", ("unknown", "en", "de"))),
    )
    with pytest.raises(ValueError, match="language"):
        registers.set_conditions({"source": torch.tensor([1])})


def test_more_conditions_than_registers_is_rejected():
    with pytest.raises(ValueError, match="at least that many registers"):
        SoftRegisters(1, 8, 2, True, (("source", ("unknown", "a")), ("language", ("unknown", "en"))))


def test_install_records_conditions_in_the_config():
    torch.manual_seed(0)
    model = _Model()
    install(
        model,
        count=2,
        layerwise=True,
        conditions=(("source", ("unknown", "terra")), ("language", ("unknown", "en", "de"))),
    )
    assert model.config.__dict__[REGISTER_CONDITIONS_KEY] == [
        ["source", ["unknown", "terra"]],
        ["language", ["unknown", "en", "de"]],
    ]
    assert conditions_from_config(model.config) == (
        ("source", ("unknown", "terra")),
        ("language", ("unknown", "en", "de")),
    )


def test_unknown_must_be_the_first_condition_value():
    with pytest.raises(ValueError, match="must list 'unknown' first"):
        SoftRegisters(2, 8, 2, True, (("source", ("terra", "unknown")),))


def test_condition_ids_send_unseen_values_to_unknown():
    vocabulary = ("unknown", "terra", "nemotron")
    ids = condition_ids(vocabulary, ["terra", "nemotron", "a-source-added-later", "unknown"])
    # A source that appears after training must score as unknown rather than raise: the
    # vocabulary is frozen at training time and unknown is what production feeds anyway.
    assert ids.tolist() == [1, 2, 0, 0]


def test_a_soft_condition_mixes_the_class_vectors():
    torch.manual_seed(0)
    model = _Model()
    registers = install(model, count=3, layerwise=True, conditions=(("domain", ("unknown", "a", "b")),))
    with torch.no_grad():
        registers.condition_residuals["domain"].weight[1] = 1.0
        registers.condition_residuals["domain"].weight[2] = -1.0
    embeddings = torch.randn(1, 7, 8)

    registers.set_conditions({"domain": torch.tensor([[0.0, 1.0, 0.0]])})
    hard = registers.write_embeddings(embeddings)
    registers.set_conditions({"domain": torch.tensor([1])})
    by_id = registers.write_embeddings(embeddings)
    # A hard id is the one-hot special case, so both routes must agree exactly.
    assert torch.allclose(hard, by_id)

    registers.set_conditions({"domain": torch.tensor([[0.0, 0.5, 0.5]])})
    mixed = registers.write_embeddings(embeddings)
    # Equal weight on opposite vectors cancels back to the plain shared register.
    assert torch.allclose(mixed[:, 1], embeddings.new_tensor(registers.shared[0]), atol=1e-6)


def test_an_unknown_weighted_condition_contributes_nothing():
    torch.manual_seed(0)
    model = _Model()
    registers = install(model, count=2, layerwise=False, conditions=(("domain", ("unknown", "a")),))
    embeddings = torch.randn(1, 6, 8)
    registers.set_conditions({"domain": torch.tensor([[1.0, 0.0]])})
    conditioned = registers.write_embeddings(embeddings)
    plain = torch.cat([embeddings[:, :1], registers.shared.unsqueeze(0), embeddings[:, 3:]], dim=1)
    assert torch.allclose(conditioned, plain)


def test_a_distribution_of_the_wrong_width_is_rejected():
    torch.manual_seed(0)
    model = _Model()
    registers = install(model, count=2, layerwise=False, conditions=(("domain", ("unknown", "a", "b")),))
    with pytest.raises(ValueError, match="width 3"):
        registers.set_conditions({"domain": torch.tensor([[0.5, 0.5]])})
