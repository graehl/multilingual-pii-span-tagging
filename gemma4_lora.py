"""Gemma-4 LoRA glue.

PEFT cannot wrap gemma-4's ``Gemma4ClippableLinear`` out of the box: it is a plain
``nn.Module`` holding an inner ``nn.Linear`` (``.linear``) plus +-inf clip buffers
(a no-op passthrough in the text tower). Register a custom LoRA class that adapts the
inner ``.linear`` so ``target_modules=q_proj,k_proj,v_proj,o_proj`` attaches cleanly.

Used at train time (register on the ``LoraConfig`` before building the PeftModel) and at
load time (register on a ``PeftConfig`` loaded from the adapter dir, then pass it via
``PeftModel.from_pretrained(..., config=cfg)``). No-op for non-gemma-4 models.
"""


def register_gemma4_clippable_lora(peft_config) -> bool:
    """Register a LoRA custom module for gemma-4's ``Gemma4ClippableLinear`` on
    ``peft_config`` (a ``LoraConfig`` or a loaded ``PeftConfig``). Returns True when the
    gemma-4 class is importable and registration ran, False otherwise. Harmless for
    non-gemma-4 models (the mapping is only consulted for matching module types)."""
    if not hasattr(peft_config, "_register_custom_module"):
        # Prompt-learning configs (prompt/prefix tuning) have no module targeting;
        # only LoRA-family configs support custom module registration.
        return False
    try:
        from peft.tuners.lora.layer import Linear as _PeftLoraLinear
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except Exception:
        return False

    class _LoraGemma4Clippable(_PeftLoraLinear):
        def __init__(self, base_layer, adapter_name, **kw):
            super().__init__(base_layer.linear, adapter_name, **kw)

    peft_config._register_custom_module({Gemma4ClippableLinear: _LoraGemma4Clippable})
    return True
