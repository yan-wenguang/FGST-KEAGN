from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score


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


def _discretize_features(matrix: np.ndarray, bins: int) -> np.ndarray:
    transformed = np.zeros_like(matrix, dtype=int)
    for index in range(matrix.shape[1]):
        transformed[:, index] = _quantile_discretize(matrix[:, index], bins)
    return transformed


def _feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    if np.allclose(values.std(), 0.0):
        return 0.5
    auc = roc_auc_score(labels, values)
    return float(max(auc, 1.0 - auc))


def compute_mi_matrix(matrix: np.ndarray, bins: int = 8) -> np.ndarray:
    discrete = _discretize_features(matrix, bins)
    num_features = discrete.shape[1]
    entropies = np.zeros(num_features, dtype=float)
    mi_matrix = np.zeros((num_features, num_features), dtype=np.float32)

    for index in range(num_features):
        entropies[index] = mutual_info_score(discrete[:, index], discrete[:, index])

    for i in range(num_features):
        for j in range(i, num_features):
            mi = mutual_info_score(discrete[:, i], discrete[:, j])
            denom = np.sqrt(max(entropies[i] * entropies[j], 1e-12))
            value = float(mi / denom) if denom > 0 else 0.0
            mi_matrix[i, j] = value
            mi_matrix[j, i] = value

    return mi_matrix


def compute_corr_matrix(matrix: np.ndarray) -> np.ndarray:
    corr = np.abs(np.corrcoef(matrix, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def compute_class_synergy_matrix(
    matrix: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 300,
) -> np.ndarray:
    num_features = matrix.shape[1]
    synergy = np.zeros((num_features, num_features), dtype=np.float32)
    single_aucs = np.asarray([_feature_auc(matrix[:, index], labels) for index in range(num_features)])

    for i, j in combinations(range(num_features), 2):
        pair_x = matrix[:, [i, j]]
        try:
            model = LogisticRegression(max_iter=max_iter, solver="lbfgs")
            model.fit(pair_x, labels)
            pair_prob = model.predict_proba(pair_x)[:, 1]
            pair_auc = roc_auc_score(labels, pair_prob)
        except Exception:
            pair_auc = max(single_aucs[i], single_aucs[j])

        score = max(0.0, float(pair_auc - max(single_aucs[i], single_aucs[j])))
        synergy[i, j] = score
        synergy[j, i] = score

    np.fill_diagonal(synergy, single_aucs.astype(np.float32))
    return synergy


def build_relation_matrices(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    bins: int = 8,
    max_iter: int = 300,
) -> dict[str, np.ndarray]:
    matrix = dataframe[feature_columns].to_numpy(dtype=np.float32)
    labels = dataframe[label_column].to_numpy(dtype=int)
    mi = compute_mi_matrix(matrix, bins=bins)
    corr = compute_corr_matrix(matrix)
    cls = compute_class_synergy_matrix(matrix, labels, max_iter=max_iter)
    return {"mi": mi, "corr": corr, "cls": cls}
