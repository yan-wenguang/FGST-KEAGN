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
    if not dataframe.columns.is_unique:
        raise ValueError("Dataframe column names must be unique.")
    if any(not isinstance(name, str) for name in dataframe.columns):
        raise ValueError("Dataframe column names must be strings.")
    if label_column not in dataframe.columns:
        raise ValueError("The label column must exist in the dataframe.")
    if not isinstance(dynamic_context_columns, list):
        raise ValueError("Context columns must be a list of named non-label columns.")
    if any(not isinstance(name, str) for name in dynamic_context_columns):
        raise ValueError("Context columns must contain string names.")
    if len(set(dynamic_context_columns)) != len(dynamic_context_columns):
        raise ValueError("Context column names must be unique.")
    if label_column in dynamic_context_columns:
        raise ValueError("The label column cannot also be a context column.")
    if any(name not in dataframe.columns for name in dynamic_context_columns):
        raise ValueError("Every context column must exist in the dataframe.")
    feature_columns = [name for name in dataframe.columns if name != label_column]
    static_feature_columns = [name for name in feature_columns if name not in dynamic_context_columns]
    return DatasetSchema(
        label_column=label_column,
        dynamic_context_columns=list(dynamic_context_columns),
        feature_columns=feature_columns,
        static_feature_columns=static_feature_columns,
    )
