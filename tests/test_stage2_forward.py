from __future__ import annotations

import unittest

import torch

from src.models.llm_keagn import LLMKEAGN


class Stage2ForwardTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        batch_size = 32
        feature_dim = 7
        context_dim = 10
        model = LLMKEAGN(
            feature_dim=feature_dim,
            context_input_dim=context_dim,
            hidden_dim=16,
            context_dim=8,
        )
        x_selected = torch.randn(batch_size, feature_dim)
        x_context = torch.randn(batch_size, context_dim)
        global_prior = torch.softmax(torch.randn(feature_dim), dim=0)
        output = model(x_selected, x_context, global_prior)
        self.assertEqual(output["logits"].shape, (batch_size, 1))
        self.assertEqual(output["fused_weights"].shape, (batch_size, feature_dim))


if __name__ == "__main__":
    unittest.main()
