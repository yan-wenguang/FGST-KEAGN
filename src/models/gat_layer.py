from __future__ import annotations

import torch
from torch import nn


class DenseGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.projections = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)])
        self.attn_src = nn.ParameterList([nn.Parameter(torch.empty(out_dim)) for _ in range(num_heads)])
        self.attn_dst = nn.ParameterList([nn.Parameter(torch.empty(out_dim)) for _ in range(num_heads)])
        self.activation = nn.ELU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        for projection, attn_src, attn_dst in zip(self.projections, self.attn_src, self.attn_dst):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.xavier_uniform_(attn_src.unsqueeze(0))
            nn.init.xavier_uniform_(attn_dst.unsqueeze(0))


    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = adjacency.clamp(min=0.0)
        mask = adjacency > 0
        outputs = []
        attentions = []

        for projection, attn_src, attn_dst in zip(self.projections, self.attn_src, self.attn_dst):
            transformed = projection(node_features)
            src_score = (transformed * attn_src).sum(dim=-1).unsqueeze(1)
            dst_score = (transformed * attn_dst).sum(dim=-1).unsqueeze(0)
            logits = self.leaky_relu(src_score + dst_score)
            logits = logits + torch.log(adjacency + 1e-8)
            logits = logits.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(logits, dim=-1)
            alpha = self.dropout(alpha)
            head_output = self.activation(alpha @ transformed)
            outputs.append(head_output)
            attentions.append(alpha)

        merged = torch.cat(outputs, dim=-1)
        stacked_attention = torch.stack(attentions, dim=0).mean(dim=0)
        return merged, stacked_attention

