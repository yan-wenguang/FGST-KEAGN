from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _subset_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    selected_indices: list[int],
) -> float:
    subset_train = x_train[:, selected_indices]
    subset_val = x_val[:, selected_indices]

    if len(selected_indices) == 1:
        scores = subset_val[:, 0]
        auc = roc_auc_score(y_val, scores)
        return float(max(auc, 1.0 - auc))

    model = LogisticRegression(max_iter=300, solver="lbfgs")
    model.fit(subset_train, y_train)
    probabilities = model.predict_proba(subset_val)[:, 1]
    return float(roc_auc_score(y_val, probabilities))


def select_features_with_redundancy(
    scores: np.ndarray,
    redundancy_matrix: np.ndarray,
    feature_names: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    min_features: int,
    max_features: int,
    penalty: float,
) -> dict[str, object]:
    ranking = list(np.argsort(scores)[::-1])
    selected = [ranking[0]]
    candidates = ranking[1:]
    best_subset = list(selected)
    best_auc = -np.inf
    history: list[dict[str, float | int]] = []

    while candidates and len(selected) < max_features:
        candidate_scores = []
        for candidate in candidates:
            redundancy = float(np.mean(redundancy_matrix[candidate, selected])) if selected else 0.0
            value = float(scores[candidate] - penalty * redundancy)
            candidate_scores.append((value, candidate))

        candidate_scores.sort(reverse=True)
        _, chosen = candidate_scores[0]
        selected.append(chosen)
        candidates.remove(chosen)
        current_auc = _subset_auc(x_train, y_train, x_val, y_val, selected)
        history.append(
            {
                "num_features": len(selected),
                "val_auc": current_auc,
                "selected_indices": [int(index) for index in selected],
                "selected_names": [feature_names[index] for index in selected],
            }
        )

        if len(selected) >= min_features and current_auc >= best_auc:
            best_auc = current_auc
            best_subset = list(selected)

    names = [feature_names[index] for index in best_subset]
    return {
        "selected_indices": best_subset,
        "selected_names": names,
        "best_val_auc": float(best_auc if best_auc > -np.inf else 0.0),
        "history": history,
    }
