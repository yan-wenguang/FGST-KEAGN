from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fgst_keagn_support.data.validation import normalize_binary_labels, validate_probability_threshold

METRIC_NAMES = (
    "roc_auc", "average_precision", "accuracy", "precision", "recall", "f1", "mcc"
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = normalize_binary_labels(y_true)
    probabilities = np.asarray(y_prob)
    if probabilities.ndim != 1 or probabilities.size != labels.size:
        raise ValueError("Probabilities must be one-dimensional and aligned with labels.")
    if probabilities.dtype.kind not in "iuf" or not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite real numbers in [0, 1].")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite real numbers in [0, 1].")
    probabilities = probabilities.astype(float)
    threshold = validate_probability_threshold(threshold)
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    positives = tp + fn
    negatives = tn + fp
    predicted_positives = tp + fp
    f1_denominator = 2 * tp + fp + fn
    mcc_denominator_squared = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    result = {
        "roc_auc": float(roc_auc_score(labels, probabilities)) if positives and negatives else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if positives else None,
        "accuracy": float((tp + tn) / labels.size),
        "precision": float(tp / predicted_positives) if predicted_positives else None,
        "recall": float(tp / positives) if positives else None,
        "f1": float(2 * tp / f1_denominator) if f1_denominator else None,
        "mcc": float((tp * tn - fp * fn) / np.sqrt(float(mcc_denominator_squared)))
        if mcc_denominator_squared else None,
    }
    undefined_reasons = {
        "roc_auc": "requires_both_true_classes",
        "average_precision": "no_positive_truth",
        "precision": "no_predicted_positives",
        "recall": "no_positive_truth",
        "f1": "zero_f1_denominator",
        "mcc": "zero_mcc_denominator",
    }
    result["status"] = {
        name: "defined" if result[name] is not None else undefined_reasons[name]
        for name in METRIC_NAMES
    }
    result["support"] = {
        "samples": int(labels.size),
        "positive": positives,
        "negative": negatives,
        "predicted_positive": predicted_positives,
    }
    result["confusion_matrix"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return result


def stable_softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError("Softmax input must be a nonempty one-dimensional real numeric array.")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError("Softmax input must contain only finite values.")
    with np.errstate(over="ignore", under="ignore"):
        shifted = array - np.max(array)
        weights = np.exp(shifted)
    return weights / np.sum(weights)
