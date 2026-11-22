from __future__ import annotations

from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )


    def forward(self, context):
        return self.network(context)

