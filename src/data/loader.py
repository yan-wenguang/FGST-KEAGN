from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.schema import DatasetSchema, infer_schema


@dataclass
class DataBundle:
    dataframe: pd.DataFrame
    schema: DatasetSchema


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_data_bundle(
    csv_path: str | Path,
    label_column: str,
    dynamic_context_columns: list[str],
) -> DataBundle:
    dataframe = load_dataframe(csv_path)
    if label_column in dataframe.columns:
        labels = dataframe[label_column].to_numpy()
        unique_labels = set(np.unique(labels).tolist())
        if unique_labels.issubset({-1, 1}):
            dataframe[label_column] = (dataframe[label_column] > 0).astype(int)
    schema = infer_schema(dataframe, label_column, dynamic_context_columns)
    return DataBundle(dataframe=dataframe, schema=schema)


def frame_to_tensors(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    context_columns: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    context_columns = context_columns or []
    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    labels = dataframe[label_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    bundle = {
        "x": torch.tensor(features, dtype=torch.float32),
        "y": torch.tensor(labels, dtype=torch.float32),
    }
    if context_columns:
        context = dataframe[context_columns].to_numpy(dtype=np.float32)
        bundle["context"] = torch.tensor(context, dtype=torch.float32)
    return bundle
