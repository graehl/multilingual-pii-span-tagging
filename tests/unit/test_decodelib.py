import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import decodelib
from decodelib import (
    common_prefix_length,
    prefill_prefix,
    prefix_from_generate,
    prepare_batch,
    prepare_cached_batch,
    split_common_prefix,
)


def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(17)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=101,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()


def test_split_common_prefix_retains_a_real_suffix_token():
    sequences = [[1, 5, 6, 11], [1, 5, 6, 21, 22], [1, 5, 6, 31]]

    assert common_prefix_length(sequences) == 3
    assert split_common_prefix(sequences, minimum_tokens=3) == (
        [1, 5, 6],
        [[11], [21, 22], [31]],
    )


def test_prefilled_ragged_batch_matches_full_prompt_greedy_decode():
    model = tiny_model()
    prompts = [[1, 5, 6, 7, 11, 12, 13], [1, 5, 6, 7, 21, 22]]
    prefix_ids, suffixes = split_common_prefix(prompts, minimum_tokens=4)

    with torch.inference_mode():
        baseline_batch = prepare_batch(model, prompts, pad_token_id=0)
        baseline = model.generate(
            **baseline_batch.model_inputs(),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=0,
        )
        prefix = prefill_prefix(model, prefix_ids)
        cached_batch = prepare_batch(model, suffixes, pad_token_id=0, prefix=prefix)
        assert cached_batch.input_ids.tolist() == [[11, 12, 13], [7, 21, 22]]
        assert cached_batch.attention_mask.tolist() == [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
        ]
        cached = model.generate(
            **cached_batch.model_inputs(),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=0,
        )

    baseline_new = baseline[:, baseline_batch.output_prompt_width :]
    cached_new = cached[:, cached_batch.output_prompt_width :]
    assert prefix.token_count == 4
    assert baseline_batch.total_prompt_width == cached_batch.total_prompt_width == 7
    assert torch.equal(baseline_new, cached_new)


def test_prepare_batch_preserves_requested_padding_side():
    model = tiny_model()
    sequences = [[5, 6, 7], [8]]

    left = prepare_batch(model, sequences, pad_token_id=0, padding_side="left")
    right = prepare_batch(model, sequences, pad_token_id=0, padding_side="right")

    assert left.input_ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert left.attention_mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert right.input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert right.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_left_padding_falls_back_when_suffix_gap_exhausts_prefix():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5])

    batch = prepare_batch(model, [[6, 7, 8], [9]], pad_token_id=0, prefix=prefix)

    assert batch.past_key_values is None
    assert batch.input_ids.tolist() == [[1, 5, 6, 7, 8], [0, 0, 1, 5, 9]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]


def test_prefix_state_rejects_a_different_model_instance():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])

    try:
        prepare_batch(tiny_model(), [[7]], pad_token_id=0, prefix=prefix)
    except ValueError as error:
        assert "different model instance" in str(error)
    else:
        raise AssertionError("a cache from another model instance was accepted")


def test_prefill_constructs_cache_for_model_attention_layout(monkeypatch):
    class FakeCache:
        def __init__(self, *, config):
            self.config = config
            self.token_count = 0

        def get_seq_length(self):
            return self.token_count

        def batch_repeat_interleave(self, _repeats):
            return None

    class FakeModel:
        device = torch.device("cpu")
        config = object()

        def __call__(self, *, input_ids, past_key_values, **_kwargs):
            assert past_key_values.config is self.config
            past_key_values.token_count = input_ids.shape[1]
            return type("Output", (), {"past_key_values": past_key_values})()

    monkeypatch.setattr(decodelib, "DynamicCache", FakeCache)
    model = FakeModel()

    prefix = prefill_prefix(model, [1, 5, 6])

    assert prefix.token_count == 3


def test_zero_padding_does_not_require_transformable_cache_layers():
    class FakeCache:
        def __deepcopy__(self, _memo):
            return FakeCache()

        def batch_repeat_interleave(self, _repeats):
            return None

    model = object()
    prefix = decodelib.PrefilledPrefix(
        past_key_values=FakeCache(),
        token_ids=(1, 5, 6),
        token_count=3,
        model_identity=id(model),
    )

    prefix.for_batch(model, 1, leading_pad_counts=[0])


def test_generated_context_handle_advances_final_token_and_reuses_exact_prefix():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])
    first_target = [1, 5, 6, 7, 8]
    first_batch = prepare_cached_batch(
        model,
        [first_target],
        pad_token_id=0,
        prefix=prefix,
    )

    with torch.inference_mode():
        first = model.generate(
            **first_batch.model_inputs(),
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=100,
            pad_token_id=0,
            use_cache=True,
            return_dict_in_generate=True,
        )

    committed = [*prefix.token_ids, *first.sequences[0].tolist()]
    assert first.past_key_values.get_seq_length() == len(committed) - 1
    handle = prefix_from_generate(model, committed, first.past_key_values)
    assert handle.token_ids == tuple(committed[:-1])
    assert handle.past_key_values.get_seq_length() == len(committed) - 1

    next_target = [*committed, 9]
    with torch.inference_mode():
        baseline_batch = prepare_batch(model, [next_target], pad_token_id=0)
        baseline = model.generate(
            **baseline_batch.model_inputs(),
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=100,
            pad_token_id=0,
        )
        cached_batch = prepare_cached_batch(
            model,
            [next_target],
            pad_token_id=0,
            prefix=handle,
        )
        assert cached_batch.input_ids[0, :2].tolist() == [committed[-1], 9]
        cached = model.generate(
            **cached_batch.model_inputs(),
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=100,
            pad_token_id=0,
        )

    assert torch.equal(
        baseline[:, baseline_batch.output_prompt_width :],
        cached[:, cached_batch.output_prompt_width :],
    )


def test_cached_retry_batches_fork_without_mutating_pre_segment_handle():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])
    target = [1, 5, 6, 7, 8]

    first = prepare_cached_batch(model, [target], pad_token_id=0, prefix=prefix)
    second = prepare_cached_batch(model, [target], pad_token_id=0, prefix=prefix)
    with torch.inference_mode():
        first_output = model.generate(
            **first.model_inputs(), max_new_tokens=2, do_sample=False, pad_token_id=0
        )
        second_output = model.generate(
            **second.model_inputs(), max_new_tokens=2, do_sample=False, pad_token_id=0
        )

    assert torch.equal(first_output, second_output)
    assert prefix.past_key_values.get_seq_length() == prefix.token_count == 3


def test_dynamic_cache_fork_shares_prefix_tensors_but_not_layer_objects():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])

    fork = prefix.for_batch(model, 1)

    assert fork is not prefix.past_key_values
    assert fork.layers[0] is not prefix.past_key_values.layers[0]
    assert fork.layers[0].keys is prefix.past_key_values.layers[0].keys
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[7]]),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            past_key_values=fork,
            use_cache=True,
        )
    assert output.past_key_values.get_seq_length() == 4
    assert prefix.past_key_values.get_seq_length() == prefix.token_count == 3
    assert output.past_key_values.layers[0].keys is not prefix.past_key_values.layers[0].keys


def test_context_handle_cpu_backup_owns_independent_tensors():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])

    backup = prefix.copy_to(model, "cpu")

    assert backup.token_ids == prefix.token_ids
    assert backup.past_key_values.layers[0].keys is not prefix.past_key_values.layers[0].keys
    assert torch.equal(
        backup.past_key_values.layers[0].keys,
        prefix.past_key_values.layers[0].keys,
    )


def test_cached_batch_can_consume_one_row_prefix_in_place():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])
    target = [1, 5, 6, 7]

    batch = prepare_cached_batch(
        model,
        [target],
        pad_token_id=0,
        prefix=prefix,
        fork_prefix=False,
    )

    assert batch.past_key_values is prefix.past_key_values
    with torch.inference_mode():
        output = model(**batch.model_inputs(), use_cache=True)
    assert output.past_key_values.get_seq_length() == 4


def test_cached_batch_rejects_a_divergent_sequence():
    model = tiny_model()
    prefix = prefill_prefix(model, [1, 5, 6])

    with pytest.raises(ValueError, match="does not extend"):
        prepare_cached_batch(model, [[1, 5, 9]], pad_token_id=0, prefix=prefix)
