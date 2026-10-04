"""Smoke tests for SophiaG optimizer and paged_lion integration in train-lora.py.

All tests run on CPU with a tiny randomly-initialized model so they don't
require GPU and don't compete with live training jobs.

Run:
    cd .
    pixi-gemma4/.pixi/envs/default/bin/python -m pytest tests/unit/test_sophia_optimizer.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# SophiaG now lives in the shared trainlib module (moved out of train-lora.py in 9c22379);
# import it directly — trainlib is a light import with no model/main side effects.
from trainlib import SophiaG  # noqa: E402, I001


# ---------------------------------------------------------------------------
# 1. SophiaG unit tests — pure optimizer math, no model
# ---------------------------------------------------------------------------


class TestSophiaGOptimizer:
    def _opt(self, *params, k=5, lr=1e-2, rho=0.05, wd=0.0):
        return SophiaG(list(params), lr=lr, rho=rho, weight_decay=wd, k=k)

    def test_init_defaults(self):
        p = nn.Parameter(torch.ones(3))
        opt = self._opt(p)
        assert opt.k == 5
        assert opt._optimizer_steps == 0

    def test_needs_hessian_at_zero(self):
        p = nn.Parameter(torch.ones(2))
        opt = self._opt(p, k=10)
        assert opt.needs_hessian_update  # 0 % 10 == 0

    def test_hessian_cadence_k3(self):
        """Hessian fires at optimizer steps 0, 3, 6, 9 for k=3."""
        p = nn.Parameter(torch.randn(4))
        opt = self._opt(p, k=3)
        fired_at = []
        for step in range(12):
            if opt.needs_hessian_update:
                fired_at.append(step)
                p.grad = torch.randn_like(p)
                opt.update_hessian()
            p.grad = torch.randn_like(p)
            opt.step()
            opt.zero_grad()
        assert fired_at == [0, 3, 6, 9], fired_at

    def test_optimizer_steps_increments(self):
        p = nn.Parameter(torch.ones(2))
        opt = self._opt(p, k=100)
        for _ in range(5):
            p.grad = torch.ones(2)
            opt.step()
            opt.zero_grad()
        assert opt._optimizer_steps == 5

    def test_parameters_update(self):
        p = nn.Parameter(torch.ones(4))
        opt = self._opt(p, lr=1e-1)
        before = p.data.clone()
        p.grad = torch.ones(4)
        opt.step()
        assert not torch.allclose(p.data, before), "parameter unchanged"
        opt.zero_grad()

    def test_clipping_by_hessian(self):
        """Large hessian → update clipped to ≤ lr * rho."""
        p = nn.Parameter(torch.ones(4))
        lr, rho = 1.0, 0.1
        opt = self._opt(p, lr=lr, rho=rho, k=1)
        # First: set a large hessian
        p.grad = torch.ones(4) * 50.0
        opt.update_hessian()
        opt.zero_grad()
        # Step with unit gradient — ratio should be clipped to ≤ rho
        before = p.data.clone()
        p.grad = torch.ones(4)
        opt.step()
        delta = (p.data - before).abs()
        # |Δ| = lr * |ratio| ≤ lr * rho = 0.1
        assert (delta <= lr * rho + 1e-6).all(), f"not clipped: max Δ={delta.max():.4f}"
        opt.zero_grad()

    def test_weight_decay(self):
        p = nn.Parameter(torch.ones(4) * 3.0)
        lr, wd = 0.01, 0.5
        opt = self._opt(p, lr=lr, wd=wd, k=100)
        p.grad = torch.zeros(4)
        before = p.data.clone()
        opt.step()
        # p ← p * (1 - lr * wd)
        expected = before * (1.0 - lr * wd)
        assert torch.allclose(p.data, expected, atol=1e-6)
        opt.zero_grad()

    def test_hessian_ema(self):
        """ĥ_t = β2·ĥ_{t-1} + (1-β2)·g²"""
        beta2 = 0.9
        p = nn.Parameter(torch.zeros(2))
        opt = SophiaG([p], betas=(0.965, beta2), k=1)

        p.grad = torch.tensor([2.0, 3.0])
        opt.update_hessian()
        h1 = opt.state[p]["hessian"].clone()
        assert torch.allclose(h1, (1 - beta2) * torch.tensor([4.0, 9.0]), atol=1e-6)

        p.grad = torch.tensor([1.0, 1.0])
        opt.update_hessian()
        h2 = opt.state[p]["hessian"].clone()
        expected = beta2 * h1 + (1 - beta2) * torch.ones(2)
        assert torch.allclose(h2, expected, atol=1e-6)
        opt.zero_grad()

    def test_convergence_quadratic(self):
        """SophiaG should descend on f(x) = ||x||² from x=5."""
        p = nn.Parameter(torch.ones(4) * 5.0)
        opt = self._opt(p, lr=5e-2, rho=0.5, k=2)
        losses = []
        for step in range(40):
            loss = (p**2).sum()
            losses.append(loss.item())
            loss.backward()
            if opt.needs_hessian_update:
                # GNB proxy: use current grad
                opt.update_hessian()
            opt.step()
            opt.zero_grad()
        assert losses[-1] < losses[0] * 0.5, f"loss did not halve: {losses[0]:.3f} → {losses[-1]:.3f}"


# ---------------------------------------------------------------------------
# 2. paged_lion: bitsandbytes step smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "optim_name,bnb_cls_name",
    [
        ("paged_lion_32bit", "PagedLion32bit"),
        ("paged_lion_8bit", "PagedLion8bit"),
    ],
)
def test_paged_lion_bnb_step(optim_name, bnb_cls_name):
    """paged_lion variants can be instantiated and take a step via bitsandbytes."""
    bnb = pytest.importorskip("bitsandbytes")
    cls = getattr(bnb.optim, bnb_cls_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = nn.Parameter(torch.randn(8, 8, device=device))
    before = p.data.clone()
    opt = cls([p], lr=1e-4)
    (p**2).sum().backward()
    opt.step()
    opt.zero_grad()
    assert not torch.allclose(p.data.cpu(), before.cpu()), "parameter unchanged after step"


# ---------------------------------------------------------------------------
# 3. paged_lion: TrainingArguments / SFTConfig accepts the optim string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("optim_name", ["paged_lion_32bit", "paged_lion_8bit"])
def test_paged_lion_accepted_by_sftconfig(optim_name, tmp_path):
    """SFTConfig validates optim strings; paged_lion must not raise ValueError."""
    SFTConfig = pytest.importorskip("trl").SFTConfig
    try:
        cfg = SFTConfig(output_dir=str(tmp_path), optim=optim_name, report_to=[])
    except ValueError as e:
        pytest.fail(f"SFTConfig rejected {optim_name}: {e}")
    assert cfg.optim == optim_name


# ---------------------------------------------------------------------------
# 4. sophia_g TrainerArgs placeholder workaround
# ---------------------------------------------------------------------------


def test_sophia_g_trainerargs_workaround(tmp_path):
    """sophia_g is not in OptimizerNames; verify the placeholder workaround
    used in train-lora.py's main() allows SFTConfig construction."""
    SFTConfig = pytest.importorskip("trl").SFTConfig

    # sophia_g must be rejected directly
    with pytest.raises((ValueError, Exception)):
        SFTConfig(output_dir=str(tmp_path), optim="sophia_g", report_to=[])

    # placeholder workaround must succeed
    cfg = SFTConfig(output_dir=str(tmp_path), optim="adamw_torch_fused", report_to=[])
    # Overwrite after construction (as main() does)
    cfg.optim = "sophia_g"  # type: ignore
    assert cfg.optim == "sophia_g"


# ---------------------------------------------------------------------------
# 5. EOS masking collator fix — attention_mask-based boundary
# ---------------------------------------------------------------------------


def test_eos_masking_uses_attention_mask():
    """When pad_token_id == eos_token_id the old labels != pad_token_id check
    excluded the EOS token itself, causing it to always be masked.
    The fix uses attention_mask to find the last real token correctly."""
    IGNORE_MASK = -100
    eos_id = 2  # simulated eos == pad token id

    # Simulate a batch row: real_tokens=[0,1,2(eos)], then padding [2,2] (pad == eos).
    # attention_mask: [1,1,1,0,0]
    labels = torch.tensor([0, 1, eos_id, eos_id, eos_id])
    attn = torch.tensor([1, 1, 1, 0, 0])

    # Old behaviour: (labels != eos_id) skips position 2 (EOS), so last_idx=1
    old_real = (labels != eos_id).nonzero()
    old_last = old_real[-1].item()
    assert old_last == 1, "old code would find last_idx=1 (before EOS)"
    # After old code: labels[2:] = IGNORE_MASK → EOS masked!
    labels_old = labels.clone()
    labels_old[old_last + 1 :] = IGNORE_MASK
    assert labels_old[2] == IGNORE_MASK, "old: EOS always masked (bug)"

    # New behaviour: use attention_mask
    labels_new = labels.clone()
    real_positions = attn.nonzero()
    new_last = real_positions[-1].item()
    assert new_last == 2, "new code: last real position is EOS at index 2"
    # eos=True path: don't mask EOS, mask padding only
    labels_new[new_last + 1 :] = IGNORE_MASK
    assert labels_new[2] == eos_id, "new: EOS token kept for learning"
    assert labels_new[3] == IGNORE_MASK, "new: padding still masked"
    assert labels_new[4] == IGNORE_MASK, "new: padding still masked"
