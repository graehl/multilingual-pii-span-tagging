import pytest
import torch

from scripts.pii_encoder_train import exclude_unsupervised_items, mapped_head_objective_mass


def test_unsupervised_items_lose_their_weight_to_their_own_pool():
    # Pool p: items 0 and 1 supervised, item 2 fully masked. Pool q untouched.
    weights = [1.0, 3.0, 4.0, 2.0]
    scaled, excluded = exclude_unsupervised_items(weights, ["p", "p", "p", "q"], [5.0, 2.0, 0.0, 7.0])

    assert excluded == [2]
    # p's total weight 8 now sits on items 0 and 1 in their 1:3 ratio.
    assert scaled == pytest.approx([2.0, 6.0, 0.0, 2.0])
    assert sum(scaled[:3]) == pytest.approx(8.0)


def test_a_pool_with_nothing_supervised_is_an_error():
    with pytest.raises(ValueError, match="no supervised item"):
        exclude_unsupervised_items([1.0, 1.0], ["p", "q"], [3.0, 0.0])


def test_masked_context_positions_carry_no_mass():
    # Positions: [CLS] ctx ctx [SEP] target(O) target(PER) [SEP]; O weight 0.75.
    own = torch.tensor([-100, -100, -100, -100, 0, 2, -100])
    cross = torch.full_like(own, -100)
    with_context = mapped_head_objective_mass(
        own, cross, own_o_label_id=0, cross_o_label_id=-1, o_token_weight=0.75
    )
    isolated = mapped_head_objective_mass(
        own[[0, 4, 5, 6]], cross[[0, 4, 5, 6]], own_o_label_id=0, cross_o_label_id=-1, o_token_weight=0.75
    )
    assert float(with_context) == float(isolated) == 1.75
