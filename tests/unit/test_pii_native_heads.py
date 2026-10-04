import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import BertConfig

from scripts.pii_layered_head_model import LayerConcatForTokenClassification


def small_model() -> LayerConcatForTokenClassification:
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    config.pii_head_architecture = "concat_encoder_layers"
    config.pii_head_kind = "affine"
    config.pii_encoder_layers = [-1]
    config.pii_head_rank = 0
    config.pii_classifier_dropout = 0.0
    return LayerConcatForTokenClassification(config)


def test_native_o_updates_only_selected_head_and_shared_encoder_then_reloads(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = small_model().eval()
    inputs = {"input_ids": torch.tensor([[1, 7, 9, 2]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    primary_before_attach = model(**inputs).logits.detach()
    labels = ["O", "B-PER", "I-PER", "E-PER", "S-PER"]
    model.attach_native_head(
        "uner", labels, coverage="Proper names only; native O excludes other Ont3 categories"
    )
    model.attach_native_head("other", labels, coverage="Distinct source boundary convention")
    assert torch.equal(primary_before_attach, model(**inputs).logits)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    primary_weights = model.classifier.weight.detach().clone()
    other_weights = model.native_classifiers["other"].weight.detach().clone()
    native_weights = model.native_classifiers["uner"].weight.detach().clone()
    before_encoder = {name: value.detach().clone() for name, value in model.encoder.named_parameters()}
    model.train()
    model(**inputs, labels=torch.tensor([[-100, 0, 0, -100]]), native_head="uner").loss.backward()
    assert model.classifier.weight.grad is None
    assert model.native_classifiers["other"].weight.grad is None
    assert model.native_classifiers["uner"].weight.grad.abs().sum() > 0
    optimizer.step()
    assert torch.equal(primary_weights, model.classifier.weight)
    assert torch.equal(other_weights, model.native_classifiers["other"].weight)
    assert not torch.equal(native_weights, model.native_classifiers["uner"].weight)
    assert any(
        not torch.equal(before_encoder[name], value) for name, value in model.encoder.named_parameters()
    )
    model.eval()
    expected_native = model(**inputs, native_head="uner").logits
    expected_primary = model(**inputs).logits
    model.save_pretrained(tmp_path)
    restored = LayerConcatForTokenClassification.from_local_checkpoint(tmp_path).eval()
    assert restored.config.pii_native_heads == model.config.pii_native_heads
    torch.testing.assert_close(restored(**inputs, native_head="uner").logits, expected_native)
    torch.testing.assert_close(restored(**inputs).logits, expected_primary)


def test_native_batch_draw_preserves_primary_labels_and_replays_by_nonce() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast

    from scripts.pii_encoder_train import SpanDataset
    from scripts.pii_native_heads import NativeHeadCollator, NativeHeadDataset

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "Alice": 2, "Bob": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    labels = {label: i for i, label in enumerate(["O", "B-name", "I-name", "E-name", "S-name"])}
    primary = SpanDataset([{"id": "p", "text": "Alice", "spans": [[0, 5, "name"]]}], tokenizer, labels, 32)
    native = SpanDataset([{"id": "n0", "text": "Bob", "spans": []}], tokenizer, labels, 32)
    wrapped = NativeHeadDataset(primary)
    spec = {"name": "uner", "probability": 1.0, "loss_weight": 0.25}
    collator = NativeHeadCollator(
        DataCollatorForTokenClassification(tokenizer), tokenizer, [spec], {"uner": native}, seed=17
    )
    first = collator([wrapped[(0, None, 10)]])
    # Worker scheduling and earlier calls cannot alter the same draw.
    collator([wrapped[(0, None, 99)]])
    replay = collator([wrapped[(0, None, 10)]])
    assert first["labels"].tolist() == [[4]]
    assert first["native_head_batch"]["inputs"]["labels"].tolist() == [[0]]
    assert first["native_head_batch"]["source_ids"] == ["n0"]
    assert torch.equal(
        first["native_head_batch"]["inputs"]["input_ids"], replay["native_head_batch"]["inputs"]["input_ids"]
    )
    assert "native_head_batch" not in collator([primary[0]])

    from functools import partial

    from accelerate.data_loader import SkipBatchSampler

    from scripts.pii_native_heads import NativeHeadBatchSampler
    from trainlib import WeightedLengthBatchSampler

    primary_rows = [{"text": " ".join(["Alice"] * (i + 1)), "spans": [[0, 5, "name"]]} for i in range(8)]
    drawn = NativeHeadDataset(SpanDataset(primary_rows, tokenizer, labels, 32))
    factory = partial(
        WeightedLengthBatchSampler,
        lengths=drawn.token_lengths(),
        weights=[1.0] * 8,
        batch_size=2,
        gradient_accumulation_steps=2,
        epoch_examples=7,
    )
    sampler = NativeHeadBatchSampler(drawn, factory, seed=17, epoch_examples=7, batch_size=2)
    initial = list(sampler)
    assert [[request[0] for request in batch] for batch in initial] == list(factory(seed=17))
    drawn.set_epoch(3)
    expected = list(sampler)
    assert [request[2] for batch in expected for request in batch] == list(range(21, 28))
    assert sorted(len(batch) for batch in expected) == [1, 2, 2, 2]
    drawn.set_epoch(9)
    list(sampler)
    drawn.set_epoch(3)
    assert list(SkipBatchSampler(sampler, skip_batches=2)) == expected[2:]
    drawn.set_epoch(0)
    assert list(sampler) == initial


def test_native_source_hash_ontology_and_split_are_enforced(tmp_path: Path) -> None:
    from scripts.pii_native_heads import load_native_head_config

    source = tmp_path / "gold.jsonl"
    config = tmp_path / "heads.json"
    row = {
        "id": "original",
        "text": "Alice",
        "lang": "en",
        "spans": [[0, 5, "PER"]],
        "split": "train",
        "supervision_ontology": "names",
    }
    spec = {
        "name": "names",
        "types": ["PER"],
        "ontology": "names",
        "coverage": "Proper names only",
        "boundary_policy": "Original source",
        "train": source.name,
        "probability": 1,
        "loss_weight": 0.25,
    }

    def write() -> None:
        source.write_text(json.dumps(row) + "\n")
        spec["train_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        config.write_text(json.dumps({"schema": "pii-native-heads/v1", "heads": [spec]}))

    write()
    specs, rows = load_native_head_config(config)
    assert rows["names"] == [row]
    assert specs[0]["train"] == str(source)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_native_head_config(config)
    for field, value in (("supervision_ontology", "other"), ("split", "test")):
        original = row[field]
        row[field] = value
        write()
        with pytest.raises(ValueError, match="source split or ontology mismatch"):
            load_native_head_config(config)
        row[field] = original
