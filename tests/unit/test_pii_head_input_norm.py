import pytest
import torch

from scripts.pii_layered_head_model import LayerConcatForTokenClassification
from tests.unit.test_pii_layered_head_model import layered_config


def model_with_norm(kind: str, *, seed: int = 0) -> LayerConcatForTokenClassification:
    torch.manual_seed(seed)
    config = layered_config(layers=(4,), head_kind="affine")
    config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, 2, 0
    # Deterministic training-mode passes: only the head normalization may differ.
    config.hidden_dropout_prob = config.attention_probs_dropout_prob = 0.0
    model = LayerConcatForTokenClassification(config)
    if kind != "none":
        width = model.classifier.in_features
        model.attach_head_input_norm(
            {"kind": kind, "momentum": 0.1}, torch.randn(width) * 0.1, torch.rand(width) + 0.5
        )
    return model


def batch(rows: list[list[int]]) -> dict:
    width = max(map(len, rows))
    return {
        "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
        "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
    }


ROWS = [[1, 5, 6, 7, 2], [1, 8, 9, 2], [1, 10, 11, 12, 13, 2]]


@pytest.mark.parametrize("kind", ["fixed", "batch", "renorm"])
def test_insertion_is_identity_in_evaluation_at_calibration(kind) -> None:
    reference = model_with_norm("none").eval()
    model = model_with_norm("none").eval()
    width = model.classifier.in_features
    model.attach_head_input_norm({"kind": kind}, torch.randn(width), torch.rand(width) + 0.1)
    inputs = batch(ROWS)
    torch.testing.assert_close(model(**inputs).logits, reference(**inputs).logits, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["batch", "renorm"])
def test_training_moments_ignore_padding_and_specials_but_depend_on_companions(kind) -> None:
    model = model_with_norm(kind).train()
    start = {k: v.clone() for k, v in model.head_input_norm.state_dict().items()}
    alone = model(**batch(ROWS[:2])).logits
    model.head_input_norm.load_state_dict(start)
    padded = batch(ROWS[:2])
    padded["input_ids"] = torch.cat([padded["input_ids"], torch.zeros(2, 3, dtype=torch.long)], 1)
    padded["attention_mask"] = torch.cat([padded["attention_mask"], torch.zeros(2, 3, dtype=torch.long)], 1)
    torch.testing.assert_close(model(**padded).logits[:, :5], alone)
    model.head_input_norm.load_state_dict(start)
    with_companion = model(**batch(ROWS)).logits
    assert not torch.allclose(with_companion[:2, :4], alone[:, :4])
    assert int(model.head_input_norm.updates) == 1


def test_fixed_kind_and_evaluation_mode_do_not_depend_on_companions() -> None:
    for kind, training in (("fixed", True), ("batch", False), ("renorm", False)):
        model = model_with_norm(kind).train(training)
        alone = model(**batch(ROWS[:1])).logits
        together = model(**batch(ROWS)).logits
        torch.testing.assert_close(together[:1, :5], alone)


def test_renorm_with_running_moments_equal_to_batch_matches_running_output() -> None:
    model = model_with_norm("renorm").train()
    inputs = batch(ROWS)
    features = model.encoder(**inputs, return_dict=True).last_hidden_state  # affine final-layer head
    mask = model.text_token_mask(inputs["input_ids"], inputs["attention_mask"])
    selected = features[mask]
    model.head_input_norm.calibrate(selected.mean(0), selected.var(0, unbiased=False))
    # Evaluate first: the training pass then moves the running moments.
    evaluated = model.eval()(**inputs).logits
    trained = model.train()(**inputs).logits
    torch.testing.assert_close(trained, evaluated, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["fixed", "batch", "renorm"])
def test_reload_and_fold_preserve_evaluation_logits(tmp_path, kind) -> None:
    model = model_with_norm(kind)
    model.train()
    model(**batch(ROWS))  # move running moments away from calibration
    with torch.no_grad():
        model.head_input_norm.weight.mul_(1.3)
        model.head_input_norm.bias.add_(0.2)
    model.eval()
    inputs = batch(ROWS)
    expected = model(**inputs).logits
    model.save_pretrained(tmp_path / "normed")
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path / "normed").eval()
    torch.testing.assert_close(restored(**inputs).logits, expected)
    restored.fold_head_input_norm()
    assert restored.head_input_norm is None
    restored.save_pretrained(tmp_path / "folded")
    folded = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path / "folded").eval()
    assert folded.head_input_norm is None
    torch.testing.assert_close(folded(**inputs).logits, expected, rtol=1e-5, atol=1e-5)


def test_norm_parameters_are_counted_and_receive_gradients() -> None:
    plain = model_with_norm("none")
    model = model_with_norm("batch").train()
    assert model.task_head_parameter_count() == plain.task_head_parameter_count() + 2 * (
        model.classifier.in_features
    )
    inputs = batch(ROWS)
    logits = model(**inputs).logits
    logits.square().mean().backward()
    assert model.head_input_norm.weight.grad is not None
    assert model.head_input_norm.weight.grad.abs().sum() > 0


def test_training_batch_norm_without_mask_is_an_error() -> None:
    model = model_with_norm("batch").train()
    inputs = batch(ROWS)
    outputs = model.encoder(**inputs, return_dict=True)
    with pytest.raises(ValueError, match="token mask"):
        model.classifier_features(outputs, inputs["attention_mask"])


def model_with_block(norm: str) -> LayerConcatForTokenClassification:
    model = model_with_norm("none")
    width = model.classifier.in_features
    model.attach_head_residual_mlp(
        {"hidden": 16, "norm": norm}, torch.randn(width) * 0.1, torch.rand(width) + 0.5
    )
    return model


@pytest.mark.parametrize("norm", ["none", "layer", "batch"])
def test_residual_block_starts_as_identity_in_training_and_evaluation(norm) -> None:
    reference = model_with_norm("none")
    model = model_with_block(norm)
    inputs = batch(ROWS)
    for training in (True, False):
        reference.train(training)
        model.train(training)
        torch.testing.assert_close(model(**inputs).logits, reference(**inputs).logits)


@pytest.mark.parametrize("norm", ["none", "layer", "batch"])
def test_residual_block_learns_and_round_trips(tmp_path, norm) -> None:
    model = model_with_block(norm).train()
    inputs = batch(ROWS)
    model(**inputs).logits.square().mean().backward()
    assert model.head_residual_mlp.up.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.head_residual_mlp.up.weight.normal_(std=0.1)
    model.eval()
    expected = model(**inputs).logits
    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    torch.testing.assert_close(restored(**inputs).logits, expected)


def test_batch_residual_block_depends_on_companions_only_in_training() -> None:
    model = model_with_block("batch")
    with torch.no_grad():
        model.head_residual_mlp.up.weight.normal_(std=0.1)
    for training, dependent in ((True, True), (False, False)):
        model.train(training)
        start = {k: v.clone() for k, v in model.head_residual_mlp.state_dict().items()}
        alone = model(**batch(ROWS[:1])).logits
        model.head_residual_mlp.load_state_dict(start)
        together = model(**batch(ROWS)).logits
        model.head_residual_mlp.load_state_dict(start)
        assert torch.allclose(together[:1, :5], alone, atol=1e-6) != dependent
