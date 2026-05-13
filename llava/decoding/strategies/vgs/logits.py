"""Visual Grounding Score and multiplicative reweighting (arXiv:2603.20314, Algorithm 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vgs_reweighted_probs(
    logits_orig: torch.Tensor,
    logits_dist: torch.Tensor,
    alpha: float = 1.0,
    delta: float = 0.01,
) -> torch.Tensor:
    """
    Args:
        logits_orig: [..., V] next-token logits under original image.
        logits_dist: [..., V] same under distorted image.
        alpha: reweighting strength (paper default 1.0).
        delta: floor inside max(..., delta) to avoid zeros.

    Returns:
        Renormalized probability distribution [..., V].
    """
    p_orig = F.softmax(logits_orig.float(), dim=-1)
    p_dist = F.softmax(logits_dist.float(), dim=-1)
    p_orig = torch.nan_to_num(p_orig, nan=0.0, posinf=0.0, neginf=0.0)
    p_dist = torch.nan_to_num(p_dist, nan=0.0, posinf=0.0, neginf=0.0)
    p_orig = p_orig / p_orig.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p_dist = p_dist / p_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    denom = p_orig + p_dist + 1e-8
    vgs = (p_orig - p_dist) / denom
    factor = torch.clamp(1.0 + float(alpha) * vgs, min=float(delta))
    p_final = p_orig * factor
    p_final = p_final / p_final.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p_final
