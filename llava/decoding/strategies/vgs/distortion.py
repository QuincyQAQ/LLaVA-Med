"""Image distortion for VGS-Decoding (arXiv:2603.20314 §2 / §3.3): Gaussian + Poisson on model input tensors."""

from __future__ import annotations

import torch


def distort_images_clip_tensor(
    images: torch.Tensor,
    sigma: float = 0.07,
    poisson_lambda: float = 70.0,
) -> torch.Tensor:
    """
    Apply corruption in the same domain as `process_images` output (CLIP-style normalized batch).

    Paper defaults: σ=0.07, λ=70 for Gaussian + Poisson. Poisson is ill-defined on signed normalized
    tensors; we apply a standard *scaled Poisson reprojection* on a per-image min–max channel view,
    then map back — preserves the spirit of photon noise without requiring uint8 round-trip.

    Args:
        images: float tensor, typically [B, 3, H, W], same dtype/device as input.
        sigma: additive Gaussian std (paper default 0.07 in normalized feature domain).
        poisson_lambda: strength of Poisson-like term (paper 70); set 0 to disable.
    """
    x = images.clone()
    if sigma > 0:
        x = x + sigma * torch.randn_like(x)

    if poisson_lambda and poisson_lambda > 0:
        # Poisson in float32 (half often unsupported / unstable for poisson)
        b, c, h, w = x.shape
        xf = x.float()
        flat = xf.view(b, c, -1)
        # (B, C, 1) — do not add extra dims or broadcasting blows up u / flat2 vs (B,C,H,W)
        mn = flat.amin(dim=-1, keepdim=True)
        mx = flat.amax(dim=-1, keepdim=True)
        u = (flat - mn) / (mx - mn + 1e-6)
        u = torch.clamp(u, 1e-4, 1.0)
        lam_t = u * float(poisson_lambda)
        noisy = torch.poisson(lam_t) / float(poisson_lambda)
        flat2 = noisy * (mx - mn + 1e-6) + mn
        xf = flat2.view(b, c, h, w).to(dtype=x.dtype)
        x = xf

    return x
