from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_true = np.where(y_true > 0, 1, 0).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(int)
    unique_count = np.unique(y_true).size
    auc = 0.5 if unique_count < 2 else float(roc_auc_score(y_true, y_prob))
    pr_auc = float(y_true.mean()) if unique_count < 2 else float(average_precision_score(y_true, y_prob))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if unique_count >= 1 else 0.0
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": mcc,
    }


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / (np.sum(exp_values) + 1e-12)
