from typing import Callable, List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

DECODER_TYPES = {
    "qwen",
    "qwen2",
    "qwen3",
    "llama",
    "mistral",
    "falcon",
    "gemma",
    "phi",
    "gpt2",
    "gpt_neox",
    "opt",
    "bloom",
}


def is_decoder_only(config) -> bool:
    model_type = getattr(config, "model_type", "").lower()
    arch = getattr(config, "architectures", []) or []
    causal_lm = any("ForCausalLM" in a for a in arch)
    return model_type in DECODER_TYPES or causal_lm


def last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, left_padded=True
) -> torch.Tensor:
    # For left-padded decoder-only: last token is always at index -1 (after EOS append)
    if left_padded:
        return hidden_states[:, -1]
    else:
        masked_hidden = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Masked mean for encoder/encoder-decoder
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def load_embedder(
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    device: str | None = None,
    dtype=torch.bfloat16,
    max_length: int = 8192,
    log: bool = True,
    trust_left_padded: bool = True,
) -> Callable[[List[str]], torch.Tensor]:
    """
    Returns a single callable embed(texts: List[str]) -> torch.Tensor [batch, dim] (L2-normalized).
    Auto-detects decoder-only vs. encoder and uses correct pooling.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # Left for decoder efficiency
    )
    model = (
        AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        .to(device)
        .eval()
    )

    decoder = is_decoder_only(config)
    if not decoder:
        tokenizer.padding_side = "right"  # Standard for encoder models

    if decoder:
        pool_func = lambda hs, mask: F.normalize(last_token_pool(hs, mask, trust_left_padded), p=2, dim=1)
        if log:
            print(
                f"[embed_utils] Detected decoder-only ({config.model_type}) → last-token pooling + EOS append"
            )
    else:
        pool_func = lambda hs, mask: F.normalize(mean_pool(hs, mask), p=2, dim=1)
        if log:
            print("[embed_utils] Detected encoder/encoder-decoder → mean pooling")

    eos_token = tokenizer.eos_token or "<|endoftext|>"  # Fallback for Qwen

    def embed(texts: List[str]) -> torch.Tensor:
        # Append EOS for decoder-only if missing (critical for Qwen3-Embedding)
        if decoder:
            texts = [t.rstrip() + eos_token if not t.endswith(eos_token) else t for t in texts]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = pool_func(outputs.last_hidden_state, inputs["attention_mask"])
            return embeddings

    return embed


if __name__ == "__main__":
    embedder = load_embedder("Qwen/Qwen3-Embedding-0.6B")
    hyps = ["Hello world.", "Hello world!"]
    embeds = embedder(hyps)
    sims = embeds @ embeds.T
    print(sims)
