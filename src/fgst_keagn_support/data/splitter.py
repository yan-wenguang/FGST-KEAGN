from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fgst_keagn_support.data.schema import infer_schema
from fgst_keagn_support.data.validation import normalize_binary_labels


def stratified_split(
    dataframe: pd.DataFrame,
    label_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio])
    if ratios.shape != (3,) or ratios.dtype.kind not in "iuf" or not np.isfinite(ratios).all():
        raise ValueError("Split ratios must be finite numeric values.")
    total = float(ratios.sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    if not ((ratios > 0.0) & (ratios < 1.0)).all():
        raise ValueError("All three split ratios must be strictly between 0 and 1.")
    infer_schema(dataframe, label_column, [])
    labels = normalize_binary_labels(dataframe[label_column].to_numpy())
    train_count = round(len(dataframe) * float(train_ratio))
    val_count = round(len(dataframe) * float(val_ratio))
    test_count = len(dataframe) - train_count - val_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("Each split must contain at least one row.")
    try:
        train_df, remaining_df = train_test_split(
            dataframe,
            test_size=val_count + test_count,
            stratify=labels,
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError(f"Training/holdout stratification is infeasible: {error}") from error
    try:
        val_df, test_df = train_test_split(
            remaining_df,
            test_size=test_count,
            stratify=normalize_binary_labels(remaining_df[label_column].to_numpy()),
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError(f"Validation/test stratification is infeasible: {error}") from error
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
