from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    dataframe: pd.DataFrame,
    label_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")

    train_df, remaining_df = train_test_split(
        dataframe,
        test_size=(1.0 - train_ratio),
        stratify=dataframe[label_column],
        random_state=seed,
    )
    remaining_ratio = val_ratio + test_ratio
    val_share = val_ratio / remaining_ratio
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=(1.0 - val_share),
        stratify=remaining_df[label_column],
        random_state=seed,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

