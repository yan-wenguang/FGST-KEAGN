from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_stage1_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sparse_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
    sparse_loss = mask.abs().mean()
    loss = cls_loss + sparse_weight * sparse_loss
    metrics = {
        "cls_loss": float(cls_loss.detach().cpu()),
        "sparse_loss": float(sparse_loss.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }
    return loss, metrics

