from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSchema:
    label_column: str
    dynamic_context_columns: list[str]
    feature_columns: list[str]
    static_feature_columns: list[str]


def infer_schema(
    dataframe: pd.DataFrame,
    label_column: str,
    dynamic_context_columns: list[str],
) -> DatasetSchema:
    feature_columns = [column for column in dataframe.columns if column != label_column]
    static_feature_columns = [column for column in feature_columns if column not in dynamic_context_columns]
    return DatasetSchema(
        label_column=label_column,
        dynamic_context_columns=dynamic_context_columns,
        feature_columns=feature_columns,
        static_feature_columns=static_feature_columns,
    )

