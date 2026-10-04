import copy
import json
import random

import numpy as np
import pytest
import torch

import scripts.pii_name_role_publish as publisher
from scripts.pii_name_role_char_pilot import Example, NameRoleCharCNN, NameRoleDataset


def _training_fixture(seed: int) -> tuple[NameRoleCharCNN, NameRoleDataset]:
    examples = [
        Example(surface="Alice", key="alice", role="given", language="en"),
        Example(surface="Smith", key="smith", role="family", language="en"),
        Example(surface="Akira", key="akira", role="given", language="ja"),
        Example(surface="Sato", key="sato", role="family", language="ja"),
    ]
    vocabulary = ["<PAD>", "<BOS>", "<EOS>", "<TRUNC>", *sorted(set("AliceSmithAkiraSato"))]
    character_ids = {character: index for index, character in enumerate(vocabulary)}
    model = NameRoleCharCNN(
        vocabulary_size=len(vocabulary),
        language_count=2,
        embedding_dim=4,
        convolution_channels=4,
        language_dim=2,
        language_dropout=0.1,
        language_noise=0.1,
        shared_mixture_alpha=0.2,
    )
    dataset = NameRoleDataset(
        examples,
        character_ids=character_ids,
        language_ids={"en": 0, "ja": 1},
        max_characters=12,
        case_weights=(1.0, 0.0, 0.0),
        unknown_character_backoff="typed",
        rare_characters=frozenset(),
        rare_character_backoff_probability=1.0,
        frequent_character_backoff_probability=0.0,
        seed=seed,
    )
    return model, dataset


def _train(model, dataset, checkpoint, *, resume):
    return publisher.train_fixed_horizon(
        model,
        dataset,
        [1.0] * len(dataset),
        batch_size=2,
        samples_per_epoch=8,
        epochs=4,
        learning_rate=0.01,
        length_window_steps=2,
        seed=19,
        checkpoint_path=checkpoint,
        checkpoint_signature={"test": "exact-resume"},
        resume=resume,
    )


def test_all_data_training_resumes_exactly_at_epoch_boundary(tmp_path, monkeypatch) -> None:
    random.seed(13)
    torch.manual_seed(13)
    initial, _unused = _training_fixture(19)
    initial_state = copy.deepcopy(initial.state_dict())

    full, full_dataset = _training_fixture(19)
    full.load_state_dict(initial_state)
    random.seed(23)
    torch.manual_seed(23)
    full_history, full_steps, full_resume_epoch = _train(
        full,
        full_dataset,
        tmp_path / "full.pt",
        resume=False,
    )

    interrupted, interrupted_dataset = _training_fixture(19)
    interrupted.load_state_dict(initial_state)
    random.seed(23)
    torch.manual_seed(23)
    checkpoint = tmp_path / "interrupted.pt"
    write_checkpoint = publisher._write_training_checkpoint

    def stop_after_epoch_two(path, state):
        write_checkpoint(path, state)
        if state["completed_epoch"] == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(publisher, "_write_training_checkpoint", stop_after_epoch_two)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _train(interrupted, interrupted_dataset, checkpoint, resume=False)

    monkeypatch.setattr(publisher, "_write_training_checkpoint", write_checkpoint)
    resumed, resumed_dataset = _training_fixture(19)
    resumed.load_state_dict(initial_state)
    random.seed(999)
    torch.manual_seed(999)
    resumed_history, resumed_steps, resumed_from_epoch = _train(
        resumed,
        resumed_dataset,
        checkpoint,
        resume=True,
    )

    assert full_steps == resumed_steps
    assert full_resume_epoch == 0
    assert resumed_from_epoch == 2
    for full_point, resumed_point in zip(full_history, resumed_history, strict=True):
        for key in ("epoch", "steps", "rows", "train_loss", "learning_rate"):
            assert resumed_point[key] == pytest.approx(full_point[key])
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])


def test_deployment_json_drives_python_onnx_inference(tmp_path) -> None:
    script_encoding = tmp_path / "name-kind-script-encoding.json"
    script_encoding.write_text(
        json.dumps(
            {
                "version": "1.0",
                "num_blocks": 1,
                "num_index_tokens": 1,
                "blocks": [[0, "Latin", "Letter", 0, "Ω"]],
                "source": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "name-kind.onnx").write_bytes(b"test graph placeholder")
    config = {
        "schema": "pii-name-role-character-pilot-v1",
        "roles": ["given", "family"],
        "character_vocabulary": [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<TRUNC>",
            "<UNK_SCRIPT_BLOCK_000>",
            "<UNK_SCRIPT_UNASSIGNED>",
            "A",
        ],
        "languages": ["en"],
        "max_characters": 8,
        "shared_mixture_alpha": 0.2,
        "character_backoff": {
            "unknown_character_backoff": "script-block",
            "script_projection": {
                "path": script_encoding.name,
                "sha256": publisher.file_sha256(script_encoding),
            },
        },
        "onnx": {
            "path": "name-kind.onnx",
            "inputs": {
                "char_ids": ["batch", 8],
                "language_probs": ["batch", 1],
                "mixture_alpha": ["batch", 1],
            },
            "output": {"name": "log_probabilities", "shape": ["batch", 2]},
        },
    }
    config_path = tmp_path / "name-kind.config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Session:
        def __init__(self):
            self.received = None

        def get_inputs(self):
            return [
                Tensor("char_ids", ["batch", 8]),
                Tensor("language_probs", ["batch", 1]),
                Tensor("mixture_alpha", ["batch", 1]),
            ]

        def get_outputs(self):
            return [Tensor("log_probabilities", ["batch", 2])]

        def run(self, _outputs, inputs):
            self.received = inputs
            return [np.asarray([[-0.1, -2.0]], dtype=np.float32)]

    session = Session()
    deployment = publisher.NameKindOnnxDeployment.load(
        config_path,
        session_factory=lambda _path: session,
    )
    normalized, scores = deployment.infer(["Ａ"], "en-US")

    assert normalized == ["A"]
    assert scores.argmax(axis=1).tolist() == [0]
    assert session.received["char_ids"].tolist()[0][:3] == [1, 6, 2]
    assert session.received["language_probs"].tolist() == [[1.0]]
    assert session.received["mixture_alpha"][0, 0] == pytest.approx(0.2)
