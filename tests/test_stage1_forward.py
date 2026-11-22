from __future__ import annotations

import unittest

import torch

from src.data.loader import load_data_bundle
from src.graph.adjacency import build_relation_matrices
from src.graph.node_features import build_node_features
from src.models.fgat_fs import FGATFeatureSelector


class Stage1ForwardTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        bundle = load_data_bundle(
            "examples/synthetic.csv", "label", ["context_level"]
        )
        frame = bundle.dataframe.sample(n=128, random_state=42).reset_index(drop=True)
        node_features = build_node_features(frame, bundle.schema.feature_columns, bundle.schema.label_column)
        relation = build_relation_matrices(frame, bundle.schema.feature_columns, bundle.schema.label_column, bins=6, max_iter=100)
        model = FGATFeatureSelector(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            relation_matrices={key: torch.tensor(value, dtype=torch.float32) for key, value in relation.items()},
            hidden_dim=8,
            num_heads=2,
            dropout=0.1,
            alpha_balance=0.5,
            temperature=0.5,
        )
        x = torch.tensor(frame[bundle.schema.feature_columns].to_numpy(dtype="float32"))
        output = model(x)
        self.assertEqual(output["logits"].shape, (len(frame), 1))
        self.assertEqual(output["mask"].shape[0], len(bundle.schema.feature_columns))


if __name__ == "__main__":
    unittest.main()
