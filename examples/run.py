from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fgst_keagn_support import __version__
from fgst_keagn_support.data.loader import load_data_bundle
from fgst_keagn_support.data.splitter import stratified_split
from fgst_keagn_support.data.validation import validate_feature_columns, validate_probability_threshold
from fgst_keagn_support.utils.io import load_json, save_json
from fgst_keagn_support.utils.metrics import compute_binary_metrics
from fgst_keagn_support.utils.seed import set_global_seed

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIRECTORY.parent / "configs" / "synthetic_example.json"


def validate_configuration(configuration: dict) -> None:
    sections = (
        ("configuration", configuration, {
            "experiment_id", "scope", "baseline_feature_columns", "split", "model",
            "probability_threshold", "tuning",
        }),
        ("split", configuration.get("split") if isinstance(configuration, dict) else None, {
            "seed", "train_ratio", "val_ratio", "test_ratio",
        }),
        ("model", configuration.get("model") if isinstance(configuration, dict) else None, {
            "algorithm", "solver", "penalty", "C", "fit_intercept", "max_iter",
        }),
    )
    for name, values, expected_keys in sections:
        if not isinstance(values, dict) or set(values) != expected_keys:
            raise ValueError(f"Unsupported or missing keys in the {name} configuration.")


def run_example(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    configuration_bytes = Path(config_path).read_bytes()
    configuration = json.loads(configuration_bytes)
    validate_configuration(configuration)
    contract = load_json(EXAMPLE_DIRECTORY / "schema.json")
    fixture_path = EXAMPLE_DIRECTORY / "synthetic.csv"
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if fixture_sha256 != contract["sha256"]:
        raise ValueError("Synthetic fixture checksum differs from the declared data contract.")
    bundle = load_data_bundle(
        fixture_path, contract["label_column"], contract["dynamic_context_columns"]
    )
    expected_schema = {key: contract[key] for key in (
        "label_column", "dynamic_context_columns", "feature_columns", "static_feature_columns"
    )}
    if asdict(bundle.schema) != expected_schema or list(bundle.dataframe.columns) != contract["columns"]:
        raise ValueError("Synthetic CSV schema differs from its declared contract.")
    if len(bundle.dataframe) != contract["row_count"]:
        raise ValueError("Synthetic CSV row count differs from its declared contract.")
    features = configuration["baseline_feature_columns"]
    validate_feature_columns(features, bundle.schema.static_feature_columns)
    if not np.isfinite(bundle.dataframe[bundle.schema.feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Synthetic features must contain finite numeric values.")
    threshold = validate_probability_threshold(configuration["probability_threshold"])
    split = configuration["split"]
    set_global_seed(split["seed"])
    splits = stratified_split(
        bundle.dataframe, contract["label_column"], split["train_ratio"],
        split["val_ratio"], split["test_ratio"], split["seed"],
    )
    model = configuration["model"]
    if model["algorithm"] != "LogisticRegression":
        raise ValueError("This illustration supports only the conventional LogisticRegression baseline.")
    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver=model["solver"], penalty=model["penalty"], C=model["C"],
            fit_intercept=model["fit_intercept"], max_iter=model["max_iter"],
            random_state=split["seed"],
        ),
    )
    label = contract["label_column"]
    baseline.fit(splits["train"][features], splits["train"][label])
    result = {
        "scope": "Synthetic illustration only; not FGST-KEAGN or a study result.",
        "evaluation_contract": "1.0",
        "support_package_version": __version__,
        "fixture_sha256": fixture_sha256,
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "configuration": configuration,
        "row_counts": {name: len(frame) for name, frame in splits.items()},
        "environment": {
            "python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in (
                "numpy", "pandas", "scikit-learn", "scipy", "PyYAML"
            )},
        },
    }
    for name in ("val", "test"):
        frame = splits[name]
        probabilities = baseline.predict_proba(frame[features])[:, 1]
        result[f"{name}_metrics"] = compute_binary_metrics(frame[label].to_numpy(), probabilities, threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic-only conventional baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Optional aggregate JSON output; no row-level predictions.")
    arguments = parser.parse_args()
    result = run_example(arguments.config)
    if arguments.output is not None:
        save_json(arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
