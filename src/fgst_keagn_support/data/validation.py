from __future__ import annotations

import numpy as np


def normalize_binary_labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all():
        raise ValueError("Labels must contain finite numeric binary values.")
    unique_labels = set(np.unique(labels).tolist())
    if unique_labels.issubset({0, 1}):
        return labels.astype(np.int64)
    if unique_labels.issubset({-1, 1}):
        return (labels == 1).astype(np.int64)
    raise ValueError("Supported label encodings are 0/1 or -1/+1; mixed encodings are invalid.")


def validate_probability_threshold(threshold: float) -> float:
    value = np.asarray(threshold)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise ValueError("Probability threshold must be a finite number in [0, 1].")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("Probability threshold must be a finite number in [0, 1].")
    return result


def validate_feature_columns(feature_columns: list[str], allowed_columns: list[str]) -> None:
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Feature columns must be a nonempty list.")
    if any(not isinstance(name, str) for name in feature_columns):
        raise ValueError("Feature columns must be named strings.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be unique.")
    if any(name not in allowed_columns for name in feature_columns):
        raise ValueError("Feature columns must belong to the non-label feature set.")
