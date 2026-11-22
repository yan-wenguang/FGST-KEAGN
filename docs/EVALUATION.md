# Binary Evaluation Contract

Contract version: **1.0**. Entry point: `fgst_keagn_support.utils.metrics.compute_binary_metrics`.

## Valid inputs

- Truth labels are a nonempty one-dimensional finite numeric array. Boolean labels are accepted.
- Values must belong entirely to either `{0, 1}` or `{-1, +1}`. Signed labels are normalized explicitly. Mixed encodings, unsupported values, missing labels, strings and complex values are rejected.
- Predicted probabilities are a one-dimensional array of the same length, with finite real numeric values in `[0, 1]`. Boolean or string probability arrays are rejected.
- The threshold is a finite real scalar in `[0, 1]`. Prediction uses `probability >= threshold` for the positive class.
- The caller must supply the probability of the class encoded as positive after normalization. The conventional example uses `predict_proba(X)[:, 1]` for classes 0/1.

No labels or failed predictions are silently repaired. A score from `decision_function` is not interchangeable with a probability in this particular helper, even though some ranking APIs accept unrestricted decision scores.

## Values and validity statuses

Each result includes seven metric values, a `status` mapping, `support` counts and a `confusion_matrix` containing TP, TN, FP and FN. Undefined metrics are Python `None` / JSON `null`; they must not be averaged as ordinary numerical observations.

| Metric | Definition | Undefined condition |
|---|---|---|
| `roc_auc` | scikit-learn binary ROC-AUC | Either true class absent |
| `average_precision` | Noninterpolated sum of precision weighted by recall increments | No positive truth |
| `accuracy` | (TP + TN) / n | None after nonempty input validation |
| `precision` | TP / (TP + FP) | No predicted positives |
| `recall` | TP / (TP + FN) | No positive truth |
| `f1` | 2TP / (2TP + FP + FN) | Denominator zero |
| `mcc` | (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Denominator zero |

All-positive truth can have average precision 1, while ROC-AUC remains undefined. When a mixed-class set has no predicted positives, precision and MCC are undefined, but recall and F1 are defined as zero. Validity is evaluated per metric rather than with a blanket single-class fallback.

**Average precision is not the trapezoidal area under a precision-recall curve.** The former `pr_auc` name is replaced by `average_precision`; `auc` is replaced by `roc_auc`. This is an intentional API change, not a reinterpretation of historical study results.

Status reason values are `defined`, `requires_both_true_classes`, `no_positive_truth`, `no_predicted_positives`, `zero_f1_denominator` and `zero_mcc_denominator`.

## Data and field safety

`infer_schema` rejects missing labels, missing context references, duplicate column names and the label as context. CSV headers are checked before pandas can rename duplicate columns. The example's baseline input list must be nonempty, unique and a subset of declared static, non-label features; context is deliberately excluded.

These boundary checks prevent the previously identified label-as-feature configuration error in the public example. They are not a substitute for reviewing semantic leakage or dependence in a real research dataset.

## Serialization and related helpers

`save_json` validates standard JSON serialization with `allow_nan=False` before opening the output file, so invalid numeric payloads do not truncate an existing target. This is not an atomic-write or crash-recovery guarantee. Payloads must use standard JSON-compatible Python values; callers should convert NumPy-specific values explicitly.

`stable_softmax` accepts a nonempty finite real one-dimensional numeric array, converts integer inputs to floating point, subtracts its maximum and normalizes exponentials. Very small weights may underflow to zero. It is not a multidimensional tensor softmax API.

## Evidence and scope

Tests cover known nonperfect rankings, invalid-input rejections, per-metric undefined conditions, confusion counts and the synthetic baseline. This validates the selected public helper behavior, not the private FGST-KEAGN evaluator or paper results.

References: [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
