from __future__ import annotations

import torch
from torch import nn

from src.models.gat_layer import DenseGraphAttentionLayer


def _minmax_scale(values: torch.Tensor) -> torch.Tensor:
    min_value = values.min()
    max_value = values.max()
    return (values - min_value) / (max_value - min_value + 1e-8)


class FGATFeatureSelector(nn.Module):
    def __init__(
        self,
        node_features: torch.Tensor,
        relation_matrices: dict[str, torch.Tensor],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        alpha_balance: float,
        temperature: float,
    ) -> None:
        super().__init__()
        feature_dim = node_features.shape[0]
        self.alpha_balance = alpha_balance
        self.temperature = temperature
        self.register_buffer("node_inputs", node_features)
        self.register_buffer("mi_matrix", relation_matrices["mi"])
        self.register_buffer("corr_matrix", relation_matrices["corr"])
        self.register_buffer("cls_matrix", relation_matrices["cls"])
        self.relation_logits = nn.Parameter(torch.zeros(3))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.gat1 = DenseGraphAttentionLayer(node_features.shape[1], hidden_dim, num_heads, dropout)
        self.gat2 = DenseGraphAttentionLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout)
        self.classifier = nn.Linear(feature_dim, 1)


    def build_adjacency(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.relation_logits, dim=0)
        adjacency = (
            weights[0] * self.mi_matrix
            + weights[1] * self.corr_matrix
            + weights[2] * self.cls_matrix
        )
        adjacency = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
        adjacency = adjacency / adjacency.max().clamp(min=1e-8)
        return adjacency, weights


    def forward(self, batch_x: torch.Tensor) -> dict[str, torch.Tensor]:
        adjacency, relation_weights = self.build_adjacency()
        hidden_1, attention_1 = self.gat1(self.node_inputs, adjacency)
        hidden_2, attention_2 = self.gat2(hidden_1, adjacency)
        attention_score = 0.5 * (attention_1.mean(dim=1) + attention_2.mean(dim=1))
        embedding_score = torch.norm(hidden_2, p=2, dim=1)
        scores = self.alpha_balance * _minmax_scale(attention_score) + (1.0 - self.alpha_balance) * _minmax_scale(embedding_score)
        mask = torch.sigmoid((scores - self.threshold) / self.temperature)
        logits = self.classifier(batch_x * mask.unsqueeze(0))
        return {
            "logits": logits,
            "scores": scores,
            "mask": mask,
            "adjacency": adjacency,
            "relation_weights": relation_weights,
            "node_embeddings": hidden_2,
            "attention_matrix": attention_2,
        }

