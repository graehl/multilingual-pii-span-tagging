"""Zero-initialized residual GELU block on task-head features.

Experiment module for the one-more-layer follow-up in
`research/pii/frontier/gaps/sketches/batchnorm-continuation.md`:

    h = x + W_up GELU(W_down N(x) + b_down) + b_up

``W_up`` and ``b_up`` start at zero, so inserting the block into a trained
affine head leaves its function unchanged; arms that differ only in ``N``
start from the identical served model. ``N`` is the identity, LayerNorm per
token, or masked BatchNorm over real text tokens (standard initialization
gamma=1, beta=0, running moments from calibration).
"""

from __future__ import annotations

import torch
from torch import nn

if __package__:
    from scripts.pii_head_input_norm import HeadInputNorm
else:
    from pii_head_input_norm import HeadInputNorm

CONFIG_KEY = "pii_head_residual_mlp"
NORMS = ("none", "layer", "batch")


class ResidualHeadBlock(nn.Module):
    def __init__(
        self, width: int, *, hidden: int, norm: str, momentum: float = 0.1, eps: float = 1e-5
    ) -> None:
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"residual head norm must be one of {NORMS}, got {norm!r}")
        if hidden <= 0:
            raise ValueError("residual head hidden width must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.norm_kind = norm
        self.momentum = float(momentum)
        self.eps = float(eps)
        if norm == "layer":
            self.norm = nn.LayerNorm(width, eps=eps)
        elif norm == "batch":
            self.norm = HeadInputNorm(width, kind="batch", momentum=momentum, eps=eps)
        else:
            self.norm = None
        self.down = nn.Linear(width, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def config(self) -> dict:
        return {"hidden": self.hidden, "norm": self.norm_kind, "momentum": self.momentum, "eps": self.eps}

    @classmethod
    def from_config(cls, width: int, spec: dict) -> ResidualHeadBlock:
        return cls(width, **spec)

    @torch.no_grad()
    def calibrate(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """Seed BatchNorm running moments; scale and offset keep their standard init."""
        if self.norm_kind != "batch":
            raise ValueError("only the BatchNorm block takes calibration moments")
        self.norm.running_mean.copy_(mean.to(self.norm.running_mean))
        self.norm.running_var.copy_(var.to(self.norm.running_var))

    def forward(self, features: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if self.norm_kind == "batch":
            normalized = self.norm(features, token_mask)
        elif self.norm_kind == "layer":
            normalized = self.norm(features)
        else:
            normalized = features
        return features + self.up(self.activation(self.down(normalized)))
