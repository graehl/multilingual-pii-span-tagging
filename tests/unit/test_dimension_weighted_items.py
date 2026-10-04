import pytest

from trainlib import dimension_weighted_items

CORPUS = [
    {"source": "refinement", "lang": "en"},
    {"source": "refinement", "lang": "en"},
    {"source": "refinement", "lang": "ta"},
    {"source": "native", "lang": "en"},
    {"source": "native", "lang": "ta"},
]


def test_no_weights_means_every_item_is_equally_likely():
    weights, receipt = dimension_weighted_items(CORPUS)

    assert weights == pytest.approx([0.2] * 5)
    assert receipt["dimensions"]["lang"]["achieved_share"]["ta"] == pytest.approx(0.4)
    assert receipt["dimensions"]["lang"]["label_weights"] == {"en": 1.0, "ta": 1.0}


def test_an_unlisted_label_weighs_one():
    # Only Tamil is named; English keeps 1 rather than becoming unspecified.
    weights, receipt = dimension_weighted_items(CORPUS, {"lang": {"ta": 3.0}})

    assert receipt["dimensions"]["lang"]["label_weights"] == {"en": 1.0, "ta": 3.0}
    # Three English items at 1 and two Tamil at 3 gives 3/9 and 6/9.
    assert receipt["dimensions"]["lang"]["achieved_share"]["ta"] == pytest.approx(6 / 9)
    assert sum(weights) == pytest.approx(1.0)


def test_weighting_one_dimension_leaves_the_other_untouched_within_it():
    plain, _ = dimension_weighted_items(CORPUS)
    lifted, _ = dimension_weighted_items(CORPUS, {"lang": {"ta": 4.0}})

    # Among English items the source split is unchanged; only the language mass moved.
    english = [0, 1, 3]
    plain_en = [plain[i] / sum(plain[i] for i in english) for i in english]
    lifted_en = [lifted[i] / sum(lifted[i] for i in english) for i in english]
    assert lifted_en == pytest.approx(plain_en)


def test_weights_multiply_across_dimensions():
    weights, _ = dimension_weighted_items(CORPUS, {"lang": {"ta": 3.0}, "source": {"native": 5.0}})

    # The Tamil native item carries both multipliers; the English refinement item carries
    # neither, so their ratio is exactly the product of the two weights.
    assert weights[4] / weights[0] == pytest.approx(15.0)


def test_a_zero_weight_removes_a_label_without_removing_the_dimension():
    weights, receipt = dimension_weighted_items(CORPUS, {"lang": {"en": 0.0}})

    assert receipt["dimensions"]["lang"]["achieved_share"]["en"] == pytest.approx(0.0)
    assert weights[2] == pytest.approx(0.5)


def test_the_receipt_contrasts_intent_with_consequence():
    _weights, receipt = dimension_weighted_items(CORPUS, {"lang": {"ta": 3.0}})
    lang = receipt["dimensions"]["lang"]

    # A multiplier is intent; the share is what the run actually trained on.
    assert lang["equal_weight_share"]["ta"] == pytest.approx(0.4)
    assert lang["achieved_share"]["ta"] == pytest.approx(6 / 9)


def test_weights_naming_an_absent_label_are_rejected():
    with pytest.raises(ValueError, match="labels with no items"):
        dimension_weighted_items(CORPUS, {"lang": {"xx": 2.0}})


def test_an_unknown_dimension_is_rejected():
    with pytest.raises(ValueError, match="no item carries a 'domain' label"):
        dimension_weighted_items(CORPUS, {"domain": {"legal": 2.0}})


def test_an_item_missing_a_dimension_is_rejected():
    with pytest.raises(ValueError, match="lacks labels"):
        dimension_weighted_items([{"source": "a", "lang": "en"}, {"source": "b"}])


def test_zeroing_everything_is_rejected_rather_than_normalized():
    with pytest.raises(ValueError, match="every item weighs zero"):
        dimension_weighted_items(CORPUS, {"lang": {"en": 0.0, "ta": 0.0}})


def test_floor_multipliers_hit_the_floor_exactly():
    from trainlib import label_weights_for_floor

    # Ninety English, nine German, one Tamil; only Tamil is under a 5% floor.
    labels = ["en"] * 90 + ["de"] * 9 + ["ta"]
    weights = label_weights_for_floor(labels, 0.05)
    items = [{"lang": label} for label in labels]
    _, receipt = dimension_weighted_items(items, {"lang": weights})
    share = receipt["dimensions"]["lang"]["achieved_share"]

    # Raised to the floor and no further: a minimum is not a target.
    assert share["ta"] == pytest.approx(0.05)
    # Above the floor the two donors keep their ratio to each other.
    assert share["en"] / share["de"] == pytest.approx(90 / 9)
    assert sum(share.values()) == pytest.approx(1.0)


def test_floor_leaves_an_already_balanced_dimension_alone():
    from trainlib import label_weights_for_floor

    weights = label_weights_for_floor(["en", "de", "ta"], 0.10)

    assert weights == {"en": 1.0, "de": 1.0, "ta": 1.0}


def test_every_starved_label_lands_on_the_floor_not_above_it():
    from trainlib import label_weights_for_floor

    # At a 10% floor both German and Tamil are starved, so both land on 10% exactly
    # and English keeps the rest; nothing is lifted past what the minimum requires.
    labels = ["en"] * 90 + ["de"] * 9 + ["ta"]
    weights = label_weights_for_floor(labels, 0.10)
    items = [{"lang": label} for label in labels]
    _, receipt = dimension_weighted_items(items, {"lang": weights})
    share = receipt["dimensions"]["lang"]["achieved_share"]

    assert share["ta"] == pytest.approx(0.10)
    assert share["de"] == pytest.approx(0.10)
    assert share["en"] == pytest.approx(0.80)


def test_floor_refuses_what_cannot_fit():
    from trainlib import label_weights_for_floor

    with pytest.raises(ValueError, match="infeasible for 3 labels"):
        label_weights_for_floor(["en", "de", "ta"], 0.4)


def test_a_donor_pushed_under_the_floor_is_raised_in_turn():
    from trainlib import label_weights_for_floor

    # Four labels at 46/46/5/3 percent with a 10% floor. Lifting the two starved ones
    # would scale the donors by 0.84, which would leave neither below the floor here,
    # but the tight case is 30/30/30/10 at a 25% floor: raising the last one alone
    # would drop the other three to 25% exactly, so the set must stop growing there.
    labels = ["a"] * 30 + ["b"] * 30 + ["c"] * 30 + ["d"] * 10
    weights = label_weights_for_floor(labels, 0.25)
    items = [{"lang": label} for label in labels]
    _, receipt = dimension_weighted_items(items, {"lang": weights})
    share = receipt["dimensions"]["lang"]["achieved_share"]

    assert min(share.values()) >= 0.25 - 1e-9
    assert sum(share.values()) == pytest.approx(1.0)


def test_floor_reconsiders_a_donor_that_becomes_starved():
    from trainlib import label_weights_for_floor

    counts = {"a": 80, "b": 11, "c": 8, "d": 1}
    weights = label_weights_for_floor(list(counts), 0.10, list(counts.values()))
    mass = {key: count * weights[key] for key, count in counts.items()}
    shares = {key: value / sum(mass.values()) for key, value in mass.items()}
    assert shares == pytest.approx({"a": 0.70, "b": 0.10, "c": 0.10, "d": 0.10})
