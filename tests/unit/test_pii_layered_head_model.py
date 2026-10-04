import pytest
import torch
from transformers import AutoModel, BertConfig

from scripts.pii_layered_head_model import (
    CONCAT_HEAD_ARCHITECTURE,
    LayerConcatForTokenClassification,
    resolve_encoder_layers,
    resolve_token_offsets,
    shift_token_features,
)

LABELS = ["O", "B-name", "I-name", "E-name", "S-name"]


class RecordingEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.output_hidden_states = None

    def forward(self, *args, output_hidden_states=None, **kwargs):
        self.output_hidden_states = output_hidden_states
        return self.model(*args, output_hidden_states=output_hidden_states, **kwargs)


def test_hidden_state_indices_resolve_embeddings_and_encoder_layers() -> None:
    assert resolve_encoder_layers([0, 1, -1, -3], num_hidden_layers=4) == (0, 1, 4, 2)
    with pytest.raises(ValueError, match="same hidden state"):
        resolve_encoder_layers([4, -1], num_hidden_layers=4)
    with pytest.raises(ValueError, match="outside"):
        resolve_encoder_layers([-5], num_hidden_layers=4)


def test_language_bias_routes_preserve_shared_path_and_reload(tmp_path) -> None:
    model = LayerConcatForTokenClassification(layered_config(layers=(4,), head_kind="affine")).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 5, 2], [1, 6, 2]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    original = model(**inputs).logits.detach()
    original_parameters = model.task_head_parameter_count()
    model.attach_language_bias(["en", "ar"], strength=0.5)
    assert model.task_head_parameter_count() == original_parameters + 2 * len(LABELS)
    ids = torch.tensor([0, -1])
    assert torch.equal(original, model(**inputs, language_ids=ids).logits)
    with torch.no_grad():
        model.language_bias.weight[0, 1] = 2
        model.language_bias.weight[1, 2] = 3
    shifted = model(**inputs, language_ids=ids).logits
    assert torch.equal(original[1], shifted[1])
    torch.testing.assert_close(shifted[0, :, 1], original[0, :, 1] + 1)
    model.config.pii_language_bias_strength = 0
    assert torch.equal(original, model(**inputs, language_ids=ids).logits)
    model.config.pii_language_bias_strength = 0.5
    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    torch.testing.assert_close(restored(**inputs, language_ids=ids).logits, shifted)
    with pytest.raises(ValueError, match="one language_ids"):
        restored(**inputs)
    with pytest.raises(ValueError, match="inventory indices"):
        restored(**inputs, language_ids=torch.tensor([0, 2]))


def test_language_bias_learning_changes_only_routed_parameters() -> None:
    model = LayerConcatForTokenClassification(layered_config(layers=(4,), head_kind="affine"))
    model.attach_language_bias(["en", "ar"])
    logits = torch.zeros(2, 3, len(LABELS))
    shifted = model.apply_language_bias(logits, torch.tensor([1, -1]))
    torch.nn.functional.cross_entropy(shifted.flatten(0, 1), torch.ones(6, dtype=torch.long)).backward()
    assert torch.count_nonzero(model.language_bias.weight.grad[0]) == 0
    assert torch.count_nonzero(model.language_bias.weight.grad[1]) > 0


def layered_config(
    *,
    layers=(2, 4),
    token_offsets=(0,),
    head_kind="mlp",
    head_rank=3,
    labels=LABELS,
):
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
    )
    config.architectures = [LayerConcatForTokenClassification.__name__]
    config.pii_head_architecture = CONCAT_HEAD_ARCHITECTURE
    config.pii_encoder_layers = list(layers)
    config.pii_token_offsets = list(token_offsets)
    config.pii_head_kind = head_kind
    config.pii_head_rank = head_rank
    config.pii_classifier_dropout = 0.0
    return config


def test_token_offsets_require_unique_current_position() -> None:
    assert resolve_token_offsets([-1, 0, 1]) == (-1, 0, 1)
    with pytest.raises(ValueError, match="unique"):
        resolve_token_offsets([0, 0])
    with pytest.raises(ValueError, match="current token"):
        resolve_token_offsets([-1, 1])


def test_deferred_references_do_not_compete_or_backpropagate_and_reload(tmp_path) -> None:
    labels = LABELS + [f"{p}-{t}" for t in ("person_reference", "organization_reference") for p in "BIES"]
    model = LayerConcatForTokenClassification(
        layered_config(layers=(4,), head_kind="affine", labels=labels)
    ).eval()
    model.attach_reference_type_residual(["person_reference", "organization_reference"])
    inputs = {"input_ids": torch.tensor([[1, 5, 2]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.no_grad():
        model.classifier.bias[5:] = 100
        model.reference_type_classifier.bias.fill_(100)
    original = model(**inputs).logits
    assert original.argmax(-1).min() >= 5
    model.defer_reference_training(True)
    projected = model(**inputs).logits
    assert torch.equal(projected[..., :5], original[..., :5])
    assert projected.argmax(-1).max() < 5
    target = torch.zeros(3, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(projected.flatten(0, 1), target)
    expected = torch.nn.functional.cross_entropy(original[..., :5].flatten(0, 1), target)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(model.classifier.weight.grad[5:]) == 0
    assert torch.count_nonzero(model.reference_type_classifier.weight.grad) == 0
    actual_encoder_grads = [p.grad.clone() for p in model.encoder.parameters() if p.grad is not None]
    model.zero_grad()
    expected.backward()
    expected_encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    for actual, wanted in zip(actual_encoder_grads, expected_encoder_grads, strict=True):
        torch.testing.assert_close(actual, wanted)
    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    torch.testing.assert_close(restored(**inputs).logits, projected)
    restored.defer_reference_training(False)
    torch.testing.assert_close(restored(**inputs).logits, original)


def test_token_feature_shift_uses_relative_sources_without_padding_or_wraparound() -> None:
    features = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    previous = shift_token_features(features, -1, attention_mask)
    current = shift_token_features(features, 0, attention_mask)
    following = shift_token_features(features, 1, attention_mask)

    assert torch.equal(previous[..., 0], torch.tensor([[0.0, 1.0, 2.0, 0.0]]))
    assert torch.equal(current[..., 0], torch.tensor([[1.0, 2.0, 3.0, 0.0]]))
    assert torch.equal(following[..., 0], torch.tensor([[2.0, 3.0, 0.0, 0.0]]))


def test_concatenated_layer_mlp_produces_token_logits_and_loss() -> None:
    config = layered_config()
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config))
    inputs = {
        "input_ids": torch.tensor([[1, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 3, -100]]),
    }

    output = model(**inputs)
    output.loss.backward()

    assert output.logits.shape == (1, 4, len(LABELS))
    assert torch.isfinite(output.loss)
    assert model.projection.in_features == 2 * config.hidden_size
    assert model.projection.out_features == config.pii_head_rank
    assert model.classifier.in_features == config.pii_head_rank


def test_embedding_output_can_be_concatenated_with_final_encoder_layer() -> None:
    config = layered_config(layers=(0, 4), head_kind="affine")
    encoder = RecordingEncoder(config)
    model = LayerConcatForTokenClassification(config, encoder=encoder)

    output = model(
        input_ids=torch.tensor([[1, 5, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert output.logits.shape == (1, 3, len(LABELS))
    assert model.layer_indices == (0, 4)
    assert model.classifier.in_features == 2 * config.hidden_size
    assert encoder.output_hidden_states is True


def test_time_concat_affine_normalizes_each_token_slice_separately() -> None:
    config = layered_config(
        layers=(-1,),
        token_offsets=(-1, 0, 1),
        head_kind="affine",
    )
    encoder = RecordingEncoder(config)
    model = LayerConcatForTokenClassification(config, encoder=encoder)

    output = model(
        input_ids=torch.tensor([[1, 5, 6, 2]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    assert output.logits.shape == (1, 4, len(LABELS))
    assert model.token_offsets == (-1, 0, 1)
    assert len(model.token_offset_normalizations) == 3
    assert all(isinstance(module, torch.nn.LayerNorm) for module in model.token_offset_normalizations)
    assert model.classifier.in_features == 3 * config.hidden_size
    expected_parameters = sum(
        parameter.numel()
        for module in (model.token_offset_normalizations, model.classifier)
        for parameter in module.parameters()
    )
    assert model.config.pii_head_parameters == expected_parameters
    assert encoder.output_hidden_states is False


def test_time_concat_rejects_multiple_encoder_layers() -> None:
    config = layered_config(
        layers=(2, 4),
        token_offsets=(-1, 0, 1),
        head_kind="affine",
    )
    with pytest.raises(ValueError, match="exactly one encoder layer"):
        LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config))


@pytest.mark.parametrize("head_kind", ["affine", "factorized_linear"])
def test_concatenated_layer_linear_heads_preserve_selected_layer_width(head_kind) -> None:
    config = layered_config(head_kind=head_kind)
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config))
    output = model(
        input_ids=torch.tensor([[1, 5, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert output.logits.shape == (1, 3, len(LABELS))
    if head_kind == "affine":
        assert model.classifier.in_features == 2 * config.hidden_size
        assert model.projection is None
    else:
        assert model.projection.in_features == 2 * config.hidden_size
        assert isinstance(model.activation, torch.nn.Identity)
        assert isinstance(model.normalization, torch.nn.Identity)


def test_concatenated_layer_checkpoint_round_trip_preserves_logits(tmp_path) -> None:
    config = layered_config(layers=(-1, -3), head_kind="factorized_linear", head_rank=4)
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 7, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    with torch.no_grad():
        before = model(**inputs).logits

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        after = restored(**inputs).logits

    assert restored.layer_indices == (4, 2)
    assert restored.config.pii_encoder_layers == [4, 2]
    assert torch.allclose(before, after)


def test_predicate_head_round_trip_preserves_independent_logits(tmp_path) -> None:
    config = layered_config(layers=(-1,), head_kind="affine")
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    assert model.attach_predicate_head(["given_name", "family_name", "care_provider"])
    assert not model.attach_predicate_head(["given_name", "family_name", "care_provider"])
    inputs = {
        "input_ids": torch.tensor([[1, 7, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    with torch.no_grad():
        before = model(**inputs)

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        after = restored(**inputs)

    assert before.predicate_logits.shape == (1, 3, 3)
    assert torch.allclose(before.logits, after.logits)
    assert torch.allclose(before.predicate_logits, after.predicate_logits)
    assert restored.config.pii_predicate_channels == [
        "given_name",
        "family_name",
        "care_provider",
    ]
    assert "predicate_logits" in restored.config.keys_to_ignore_at_inference


def test_primary_type_conditioning_replicates_then_trains_independent_blocks(tmp_path) -> None:
    config = layered_config(layers=(-1,), head_kind="affine")
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    assert model.attach_predicate_head(["role", "family_name"])
    source_weight = model.predicate_classifier.weight.detach().clone()
    source_bias = model.predicate_classifier.bias.detach().clone()

    assert model.attach_predicate_head(
        ["role", "family_name"],
        condition_types=["person_name", "person_reference"],
    )
    assert torch.equal(model.predicate_classifier.weight[:2], source_weight)
    assert torch.equal(model.predicate_classifier.weight[2:], source_weight)
    assert torch.equal(model.predicate_classifier.bias[:2], source_bias)
    assert torch.equal(model.predicate_classifier.bias[2:], source_bias)

    inputs = {
        "input_ids": torch.tensor([[1, 7, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    with torch.no_grad():
        before = model(**inputs).predicate_logits
    assert before.shape == (1, 3, 2, 2)
    assert torch.equal(before[:, :, 0], before[:, :, 1])

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        after = restored(**inputs).predicate_logits
    assert torch.equal(before, after)
    assert restored.config.pii_predicate_conditioning == "primary_type"
    assert restored.config.pii_predicate_condition_types == [
        "person_name",
        "person_reference",
    ]


def test_subclass_head_round_trip_preserves_flat_conditioned_blocks(tmp_path) -> None:
    config = layered_config(layers=(-1,), head_kind="affine")
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    blocks = [
        {"family": "name_component", "primary_type": "person_name", "start": 0, "width": 4},
        {"family": "place_coarseness", "primary_type": "locality", "start": 4, "width": 6},
    ]
    assert model.attach_subclass_head(blocks, spec_sha256="0" * 64)
    assert not model.attach_subclass_head(blocks, spec_sha256="0" * 64)
    inputs = {
        "input_ids": torch.tensor([[1, 7, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    with torch.no_grad():
        before = model(**inputs)

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        after = restored(**inputs)

    assert before.subclass_logits.shape == (1, 3, 10)
    assert torch.equal(before.subclass_logits, after.subclass_logits)
    assert restored.config.pii_subclass_blocks == blocks
    assert restored.config.pii_subclass_spec_sha256 == "0" * 64
    assert "subclass_logits" in restored.config.keys_to_ignore_at_inference


def test_reference_type_residual_is_zero_noop_then_broadcasts_and_round_trips(tmp_path) -> None:
    reference_labels = [
        "O",
        *[f"{boundary}-person_name" for boundary in "BIES"],
        *[f"{boundary}-organization_reference" for boundary in "BIES"],
        *[f"{boundary}-person_reference" for boundary in "BIES"],
    ]
    config = layered_config(layers=(-1,), head_kind="affine", labels=reference_labels)
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 7, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    with torch.no_grad():
        base_logits = model(**inputs).logits

    primary_types = ["organization_reference", "person_reference"]
    assert model.attach_reference_type_residual(primary_types)
    assert not model.attach_reference_type_residual(primary_types)
    assert torch.count_nonzero(model.reference_type_classifier.weight) == 0
    assert torch.count_nonzero(model.reference_type_classifier.bias) == 0
    with torch.no_grad():
        noop_output = model(**inputs)
    assert torch.equal(noop_output.logits, base_logits)
    assert torch.equal(noop_output.reference_type_logits, torch.zeros(1, 3, 2))

    with torch.no_grad():
        model.reference_type_classifier.bias.copy_(torch.tensor([1.25, -0.75]))
        changed = model(**inputs)
    label2id = {label: index for index, label in enumerate(reference_labels)}
    for boundary in "BIES":
        assert torch.allclose(
            changed.logits[..., label2id[f"{boundary}-organization_reference"]]
            - base_logits[..., label2id[f"{boundary}-organization_reference"]],
            torch.full((1, 3), 1.25),
        )
        assert torch.allclose(
            changed.logits[..., label2id[f"{boundary}-person_reference"]]
            - base_logits[..., label2id[f"{boundary}-person_reference"]],
            torch.full((1, 3), -0.75),
        )
        assert torch.equal(
            changed.logits[..., label2id[f"{boundary}-person_name"]],
            base_logits[..., label2id[f"{boundary}-person_name"]],
        )
    assert torch.equal(changed.logits[..., 0], base_logits[..., 0])

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        restored_output = restored(**inputs)
    assert torch.equal(changed.logits, restored_output.logits)
    assert torch.equal(changed.reference_type_logits, restored_output.reference_type_logits)
    assert restored.config.pii_reference_type_residual_types == primary_types
    assert "reference_type_logits" in restored.config.keys_to_ignore_at_inference


def test_time_concat_checkpoint_round_trip_preserves_logits(tmp_path) -> None:
    config = layered_config(
        layers=(-1,),
        token_offsets=(-1, 0, 1),
        head_kind="affine",
    )
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 7, 8, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    with torch.no_grad():
        before = model(**inputs).logits

    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    with torch.no_grad():
        after = restored(**inputs).logits

    assert restored.token_offsets == (-1, 0, 1)
    assert torch.allclose(before, after)


def test_build_from_encoder_checkpoint_records_reproducible_head_metadata(tmp_path) -> None:
    base_path = tmp_path / "base"
    config = layered_config()
    del config.pii_head_architecture
    del config.pii_encoder_layers
    del config.pii_head_kind
    del config.pii_head_rank
    del config.pii_classifier_dropout
    del config.pii_token_offsets
    AutoModel.from_config(config).save_pretrained(base_path)

    model = LayerConcatForTokenClassification.from_encoder_pretrained(
        base_path,
        LABELS,
        layers=[2, -1],
        token_offsets=[0],
        head_kind="mlp",
        rank=3,
        dtype=torch.float32,
        dropout=0.2,
    )

    assert model.layer_indices == (2, 4)
    assert model.config.pii_encoder_model == str(base_path)
    assert model.config.pii_head_kind == "mlp"
    assert model.config.pii_head_rank == 3
    assert model.config.pii_classifier_dropout == 0.2
    assert model.config.pii_token_offsets == [0]


def test_final_layer_only_head_does_not_retain_every_encoder_state() -> None:
    config = layered_config(layers=(-1,), head_kind="mlp")
    encoder = RecordingEncoder(config)
    model = LayerConcatForTokenClassification(config, encoder=encoder)

    output = model(
        input_ids=torch.tensor([[1, 5, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert output.logits.shape == (1, 3, len(LABELS))
    assert output.hidden_states is None
    assert encoder.output_hidden_states is False


def test_classifier_features_are_the_exact_pre_head_values() -> None:
    config = layered_config(layers=(-1,), head_kind="affine")
    model = LayerConcatForTokenClassification(config, encoder=AutoModel.from_config(config)).eval()
    input_ids = torch.tensor([[1, 5, 2]])
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    with torch.no_grad():
        encoded = model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        features = model.classifier_features(encoded, attention_mask)
        expected = model.classifier(features)
        actual = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.equal(actual, expected)


def test_layered_head_returns_hidden_states_when_requested() -> None:
    config = layered_config(layers=(-1,), head_kind="mlp")
    encoder = RecordingEncoder(config)
    model = LayerConcatForTokenClassification(config, encoder=encoder)

    output = model(
        input_ids=torch.tensor([[1, 5, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        output_hidden_states=True,
    )

    assert output.logits.shape == (1, 3, len(LABELS))
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    assert output.hidden_states[0].shape == (1, 3, config.hidden_size)
    assert encoder.output_hidden_states is True
