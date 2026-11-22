from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_data_bundle
from src.utils.io import load_json, load_yaml, save_json


def _top_related_features(
    dataframe: pd.DataFrame,
    feature_name: str,
    feature_names: list[str],
    top_k: int = 3,
) -> list[dict]:
    correlations = dataframe[feature_names].corr().abs()[feature_name].drop(labels=[feature_name])
    top = correlations.sort_values(ascending=False).head(top_k)
    return [{"feature_name": name, "correlation": float(value)} for name, value in top.items()]


def generate_feature_cards(
    dataframe: pd.DataFrame,
    feature_names: list[str],
    label_column: str,
    feature_metadata: dict[str, dict],
) -> dict[str, dict]:
    cards: dict[str, dict] = {}
    labels = dataframe[label_column].to_numpy(dtype=float)
    for feature_name in feature_names:
        values = dataframe[feature_name].to_numpy(dtype=float)
        corr = float(np.corrcoef(values, labels)[0, 1]) if np.std(values) > 1e-8 else 0.0
        if np.isnan(corr):
            corr = 0.0
        metadata = feature_metadata.get(feature_name, {})
        cards[feature_name] = {
            "feature_name": feature_name,
            "display_name": metadata.get("display_name", feature_name),
            "description": metadata.get("description", f"Auto-generated feature card for {feature_name}."),
            "category": metadata.get("category", "unknown"),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "label_correlation": corr,
            "direction_hint": "positive" if corr >= 0 else "negative",
            "top_related_features": _top_related_features(dataframe, feature_name, feature_names),
        }
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature cards from the dataset.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    parser.add_argument("--selected", default="", help="Optional selected features JSON path.")
    parser.add_argument("--output", default="knowledge/feature_cards.json", help="Output JSON path.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_bundle = load_data_bundle(
        config["data"]["csv_path"],
        config["data"]["label_column"],
        config["data"]["dynamic_context_columns"],
    )

    feature_metadata = load_json("knowledge/feature_metadata.json", default={})
    feature_names = data_bundle.schema.feature_columns
    if args.selected and Path(args.selected).exists():
        payload = load_json(args.selected, default={})
        feature_names = payload.get("selected_features", feature_names)

    cards = generate_feature_cards(
        dataframe=data_bundle.dataframe,
        feature_names=feature_names,
        label_column=data_bundle.schema.label_column,
        feature_metadata=feature_metadata,
    )
    save_json(args.output, cards)
    print(f"Saved {len(cards)} feature cards to {args.output}")


if __name__ == "__main__":
    main()
