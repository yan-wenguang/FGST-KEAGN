from __future__ import annotations

import torch
from torch import nn

from src.models.context_encoder import ContextEncoder


class LLMKEAGN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        context_input_dim: int,
        hidden_dim: int,
        context_dim: int,
        dropout: float = 0.10,
        use_knowledge_branch: bool = True,
        use_adaptive_gate: bool = True,
        fixed_gate_value: float = 0.5,
        weight_mode: str = "softmax",
        residual_raw_weight: float = 0.0,
        raw_branch_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_knowledge_branch = use_knowledge_branch
        self.use_adaptive_gate = use_adaptive_gate
        self.fixed_gate_value = fixed_gate_value
        self.weight_mode = weight_mode
        self.residual_raw_weight = residual_raw_weight
        self.raw_branch_weight = raw_branch_weight
        self.data_branch = nn.Linear(feature_dim, feature_dim)
        self.knowledge_branch = nn.Linear(feature_dim * 2, feature_dim) if use_knowledge_branch else None
        self.context_encoder = ContextEncoder(context_input_dim, context_dim, dropout=dropout)
        self.gate_network = (
            nn.Sequential(
                nn.Linear(feature_dim * 2 + context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim),
            )
            if use_adaptive_gate
            else None
        )
        self.output_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_head = (
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if raw_branch_weight > 0.0
            else None
        )

    def _normalize_feature_weights(self, logits: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        selected_features: torch.Tensor,
        context_features: torch.Tensor,
        global_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if global_prior.dim() == 1:
            global_prior = global_prior.unsqueeze(0).expand(selected_features.shape[0], -1)

        data_logits = self.data_branch(selected_features)
        w_data = self._normalize_feature_weights(data_logits)

        if self.use_knowledge_branch and self.knowledge_branch is not None:
            knowledge_input = torch.cat([selected_features, global_prior], dim=-1)
            knowledge_logits = self.knowledge_branch(knowledge_input)
            w_knowledge = self._normalize_feature_weights(knowledge_logits)
        else:
            w_knowledge = global_prior

        context_embedding = self.context_encoder(context_features)
        if self.use_knowledge_branch:
            if self.use_adaptive_gate and self.gate_network is not None:
                gate_input = torch.cat([w_data, w_knowledge, context_embedding], dim=-1)
                gate = torch.sigmoid(self.gate_network(gate_input))
            else:
                gate = torch.full_like(w_data, fill_value=self.fixed_gate_value)
            fused_weights = gate * w_data + (1.0 - gate) * w_knowledge
        else:
            gate = torch.ones_like(w_data)
            fused_weights = w_data

        weighted_features = fused_weights * selected_features + self.residual_raw_weight * selected_features
        logits = self.output_head(weighted_features)
        if self.raw_head is not None:
            logits = logits + self.raw_branch_weight * self.raw_head(selected_features)

        return {
            "logits": logits,
            "w_data": w_data,
            "w_knowledge": w_knowledge,
            "gate": gate,
            "fused_weights": fused_weights,
            "weighted_features": weighted_features,
            "context_embedding": context_embedding,
        }
