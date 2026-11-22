from __future__ import annotations

import torch
import torch.nn.functional as F


def _context_smoothness_loss(
    gate: torch.Tensor,
    context_embedding: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    context_distance = torch.cdist(context_embedding, context_embedding, p=2).pow(2)
    similarity = torch.exp(-gamma * context_distance)
    diagonal_mask = 1.0 - torch.eye(similarity.shape[0], device=similarity.device, dtype=similarity.dtype)
    similarity = similarity * diagonal_mask
    gate_distance = torch.cdist(gate, gate, p=2).pow(2)
    return (similarity * gate_distance).sum() / (similarity.sum() + 1e-8)


def compute_stage2_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    w_data: torch.Tensor,
    w_knowledge: torch.Tensor,
    gate: torch.Tensor,
    context_embedding: torch.Tensor,
    align_weight: float,
    context_weight: float,
    context_gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
    align_loss = F.mse_loss(w_data, w_knowledge)
    ctx_loss = _context_smoothness_loss(gate, context_embedding, gamma=context_gamma)
    total = cls_loss + align_weight * align_loss + context_weight * ctx_loss
    return total, {
        "cls_loss": float(cls_loss.detach().cpu()),
        "align_loss": float(align_loss.detach().cpu()),
        "ctx_loss": float(ctx_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
