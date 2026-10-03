"""Per-channel normalization of task-head input features.

Experiment module for `research/pii/frontier/gaps/sketches/batchnorm-continuation.md`.
One module sits between `classifier_features` and the shared head dropout, so
every affine consumer of the head features sees the same transform.

Kinds:

- ``fixed``: normalize with fixed calibration moments, then a trainable
  per-channel scale and offset. A pure affine reparameterization.
- ``batch``: BatchNorm. In training, normalize with the current physical
  batch's moments over real text tokens, differentiably, and update running
  moments; in evaluation use the running moments.
- ``renorm``: Batch Renormalization (Ioffe, 2017). Training normalizes with
  batch moments corrected by the clipped, gradient-free factors
  ``r = sigma_B / sigma_run`` and ``d = (mu_B - mu_run) / sigma_run``, so the
  output equals the running-moment output wherever the clip does not bind.

Moments are token-weighted over the supplied token mask, in float32. Every
kind starts as the identity in evaluation mode: scale is the calibration
standard deviation and offset the calibration mean. In evaluation mode the
transform is a fixed affine map, which `fold_into_linear` absorbs into each
consumer so serving checkpoints carry no extra module.
"""

from __future__ import annotations

import torch
from torch import nn

CONFIG_KEY = "pii_head_input_norm"
KINDS = ("fixed", "batch", "renorm")


class HeadInputNorm(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        kind: str,
        eps: float = 1e-5,
        momentum: float = 0.1,
        r_max: float = 3.0,
        d_max: float = 5.0,
    ) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"head input norm kind must be one of {KINDS}, got {kind!r}")
        if not (0.0 < momentum <= 1.0) or eps <= 0 or r_max < 1.0 or d_max < 0.0:
            raise ValueError("invalid head input norm hyperparameters")
        self.kind = kind
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.r_max = float(r_max)
        self.d_max = float(d_max)
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("running_mean", torch.zeros(width))
        self.register_buffer("running_var", torch.ones(width))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def config(self) -> dict:
        return {
            "kind": self.kind,
            "eps": self.eps,
            "momentum": self.momentum,
            "r_max": self.r_max,
            "d_max": self.d_max,
        }

    @classmethod
    def from_config(cls, width: int, spec: dict) -> HeadInputNorm:
        return cls(width, **spec)

    @torch.no_grad()
    def calibrate(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """Make the evaluation transform the identity at these moments."""
        mean = mean.to(self.running_mean)
        var = var.to(self.running_var)
        self.running_mean.copy_(mean)
        self.running_var.copy_(var)
        self.weight.copy_(torch.sqrt(var + self.eps).to(self.weight))
        self.bias.copy_(mean.to(self.bias))

    def eval_affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluation transform as y = scale * x + shift, float32."""
        scale = self.weight.float() / torch.sqrt(self.running_var.float() + self.eps)
        shift = self.bias.float() - self.running_mean.float() * scale
        return scale, shift

    def forward(self, features: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if self.kind == "fixed" or not self.training:
            scale, shift = self.eval_affine()
            return (features.float() * scale + shift).to(features.dtype)
        if token_mask is None:
            raise ValueError("batch-dependent head normalization needs a token mask in training")
        mask = token_mask.bool()
        selected = features.float()[mask]
        if selected.shape[0] < 2:
            raise ValueError("batch-dependent head normalization needs at least two valid tokens")
        mean = selected.mean(0)
        var = selected.var(0, unbiased=False)
        std = torch.sqrt(var + self.eps)
        normalized = (features.float() - mean) / std
        if self.kind == "renorm":
            running_std = torch.sqrt(self.running_var.float() + self.eps)
            r = (std.detach() / running_std).clamp(1.0 / self.r_max, self.r_max)
            d = ((mean.detach() - self.running_mean.float()) / running_std).clamp(-self.d_max, self.d_max)
            normalized = normalized * r + d
        out = normalized * self.weight.float() + self.bias.float()
        with torch.no_grad():
            count = selected.shape[0]
            unbiased = var.detach() * count / (count - 1)
            self.running_mean.lerp_(mean.detach().to(self.running_mean), self.momentum)
            self.running_var.lerp_(unbiased.to(self.running_var), self.momentum)
            self.updates += 1
        return out.to(features.dtype)


@torch.no_grad()
def calibration_moments(model, encodings, tokenizer, *, device, chunk: int = 32) -> dict:
    """Token-weighted head-feature moments over encoded training items.

    ``encodings`` are dataset items carrying ``input_ids`` and
    ``attention_mask``. The model runs in evaluation mode (encoder dropout
    off) and the features are taken before any head-input normalization.
    Returns float32 mean, biased variance and the counts behind them.
    """
    if getattr(model, "head_input_norm", None) is not None:
        raise ValueError("calibrate before attaching head input normalization")
    was_training = model.training
    model.eval()
    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]))
    total = total_sq = None
    tokens = 0
    try:
        for start in range(0, len(order), chunk):
            batch = tokenizer.pad(
                [
                    {
                        "input_ids": encodings[i]["input_ids"],
                        "attention_mask": encodings[i]["attention_mask"],
                    }
                    for i in order[start : start + chunk]
                ],
                return_tensors="pt",
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=not model.final_layer_only,
                return_dict=True,
            )
            features = model.classifier_features(outputs, attention_mask).double()
            selected = features[model.text_token_mask(input_ids, attention_mask)]
            total = selected.sum(0) if total is None else total + selected.sum(0)
            squares = selected.pow(2).sum(0)
            total_sq = squares if total_sq is None else total_sq + squares
            tokens += selected.shape[0]
    finally:
        model.train(was_training)
    if tokens < 2:
        raise ValueError("head input calibration found fewer than two text tokens")
    mean = total / tokens
    var = (total_sq / tokens - mean.pow(2)).clamp_min(0.0)
    return {"mean": mean.float(), "var": var.float(), "items": len(encodings), "tokens": tokens}


@torch.no_grad()
def fold_into_linear(linear: nn.Linear, scale: torch.Tensor, shift: torch.Tensor) -> None:
    """Absorb y = scale * x + shift into a following affine map, in place."""
    weight = linear.weight.float()
    folded_bias = weight @ shift
    if linear.bias is None:
        raise ValueError("folding a shift needs a bias on the consumer")
    linear.bias.copy_((linear.bias.float() + folded_bias).to(linear.bias))
    linear.weight.copy_((weight * scale).to(linear.weight))
