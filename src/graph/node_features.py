from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def _quantile_discretize(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    unique_values = np.unique(values)
    if unique_values.size <= 1:
        return np.zeros(values.shape[0], dtype=int)

    effective_bins = min(bins, unique_values.size)
    quantiles = np.linspace(0.0, 1.0, effective_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size <= 2:
        return np.searchsorted(np.sort(unique_values), values)
    return np.digitize(values, bins=edges[1:-1], right=False)


def compute_entropy(values: np.ndarray, bins: int = 8) -> float:
    discrete = _quantile_discretize(values, bins)
    if np.unique(discrete).size <= 1:
        return 0.0
    return float(mutual_info_score(discrete, discrete))


def build_node_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    bins: int = 8,
) -> np.ndarray:
    labels = dataframe[label_column].to_numpy(dtype=float)
    features = []
    for column in feature_columns:
        values = dataframe[column].to_numpy(dtype=float)
        std_value = float(np.std(values))
        if std_value < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, labels)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        features.append(
            [
                float(np.mean(values)),
                std_value,
                corr,
                compute_entropy(values, bins=bins),
            ]
        )
    return np.asarray(features, dtype=np.float32)
