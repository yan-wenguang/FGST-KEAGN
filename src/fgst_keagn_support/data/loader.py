from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fgst_keagn_support.data.schema import DatasetSchema, infer_schema
from fgst_keagn_support.data.validation import normalize_binary_labels, validate_feature_columns

if TYPE_CHECKING:
    import torch


@dataclass
class DataBundle:
    dataframe: pd.DataFrame
    schema: DatasetSchema


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header or len(set(header)) != len(header) or any(not name for name in header):
        raise ValueError("CSV must have a nonempty header with unique named columns.")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def load_data_bundle(
    csv_path: str | Path,
    label_column: str,
    dynamic_context_columns: list[str],
) -> DataBundle:
    dataframe = load_dataframe(csv_path)
    schema = infer_schema(dataframe, label_column, dynamic_context_columns)
    dataframe[label_column] = normalize_binary_labels(dataframe[label_column].to_numpy())
    return DataBundle(dataframe=dataframe, schema=schema)


def frame_to_tensors(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    context_columns: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    context_columns = [] if context_columns is None else context_columns
    schema = infer_schema(dataframe, label_column, context_columns)
    validate_feature_columns(feature_columns, schema.feature_columns)
    labels = normalize_binary_labels(dataframe[label_column].to_numpy()).astype(np.float32).reshape(-1, 1)
    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    context = dataframe[context_columns].to_numpy(dtype=np.float32) if context_columns else None
    if not np.isfinite(features).all() or (context is not None and not np.isfinite(context).all()):
        raise ValueError("Tensor features and context must be finite numeric values.")
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise ModuleNotFoundError(
            "frame_to_tensors requires PyTorch. Install torch separately to use this helper."
        ) from error
    bundle = {
        "x": torch.tensor(features, dtype=torch.float32),
        "y": torch.tensor(labels, dtype=torch.float32),
    }
    if context is not None:
        bundle["context"] = torch.tensor(context, dtype=torch.float32)
    return bundle
