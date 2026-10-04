"""Checkpoint precision survives loading without an intermediate BF16 round.

The environment's SWIG import/shutdown warnings are tracked separately in
research/pii/frontier/gaps/python-swig-test-shutdown.md; do not filter them.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaForTokenClassification

from scripts.pii_encoder_train import load_local_token_classifier, resolve_training_parameter_precision
from scripts.pii_layered_head_model import LayerConcatForTokenClassification


@pytest.mark.parametrize("layered", [False, True])
def test_fp32_checkpoint_loading_preserves_sub_bf16_updates(tmp_path: Path, layered: bool) -> None:
    config = XLMRobertaConfig(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=5,
    )
    if layered:
        config.pii_head_architecture = "concat_encoder_layers"
        config.pii_encoder_layers = [1]
        config.pii_head_kind = "affine"
        config.pii_head_rank = 0
        config.pii_classifier_dropout = 0
        model = LayerConcatForTokenClassification(config)
    else:
        model = XLMRobertaForTokenClassification(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.000123)
    model.save_pretrained(tmp_path)
    restored = load_local_token_classifier(tmp_path, dtype=torch.float32)
    legacy = load_local_token_classifier(tmp_path)
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name
        assert torch.equal(legacy.state_dict()[name].float(), value.bfloat16().float()), name
        assert not torch.equal(legacy.state_dict()[name].float(), value), name


def test_precision_defaults_and_exact_resume_contract() -> None:
    assert resolve_training_parameter_precision(None) == "float32"
    assert resolve_training_parameter_precision(None, evaluate_only=True) == "legacy"
    assert resolve_training_parameter_precision("legacy") == "legacy"
    assert resolve_training_parameter_precision(None, resume_config=SimpleNamespace()) == "legacy"
    for saved in ("float32", "legacy"):
        config = SimpleNamespace(pii_training_parameter_precision=saved)
        assert resolve_training_parameter_precision(None, resume_config=config) == saved
        assert resolve_training_parameter_precision(saved, resume_config=config) == saved
        other = "legacy" if saved == "float32" else "float32"
        with pytest.raises(ValueError, match="exact resume training parameter precision mismatch"):
            resolve_training_parameter_precision(other, resume_config=config)
    for invalid in (None, "float16", ""):
        with pytest.raises(ValueError, match="unsupported checkpoint"):
            resolve_training_parameter_precision(
                None, resume_config=SimpleNamespace(pii_training_parameter_precision=invalid)
            )
