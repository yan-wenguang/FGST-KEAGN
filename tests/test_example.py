from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.generate import COLUMNS, DATA_PATH, ROW_COUNT, synthetic_csv_bytes, write_synthetic_data
from examples.run import DEFAULT_CONFIG, EXAMPLE_DIRECTORY, run_example
from fgst_keagn_support.data.loader import load_data_bundle
from fgst_keagn_support.utils.io import load_json
from fgst_keagn_support.utils.metrics import METRIC_NAMES


class ExampleTests(unittest.TestCase):
    def test_committed_fixture_matches_generator_bytes(self) -> None:
        self.assertEqual(DATA_PATH.read_bytes(), synthetic_csv_bytes())

    def test_generator_refuses_different_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            write_synthetic_data(path)
            write_synthetic_data(path)
            path.write_bytes(b"different content\n")
            with self.assertRaises(FileExistsError):
                write_synthetic_data(path)
            self.assertEqual(path.read_bytes(), b"different content\n")

    def test_fixture_and_schema_contract(self) -> None:
        contract = load_json(EXAMPLE_DIRECTORY / "schema.json")
        bundle = load_data_bundle(DATA_PATH, "label", ["context_level"])
        frame = bundle.dataframe
        self.assertEqual(list(frame.columns), COLUMNS)
        self.assertEqual(len(frame), ROW_COUNT)
        self.assertEqual(len(frame.drop_duplicates()), ROW_COUNT)
        self.assertFalse(frame.isna().any().any())
        self.assertEqual(set(frame["label"]), {0, 1})
        self.assertEqual(hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(), contract["sha256"])
        self.assertEqual(asdict(bundle.schema), {key: contract[key] for key in (
            "label_column", "dynamic_context_columns", "feature_columns", "static_feature_columns",
        )})

    def test_baseline_reports_finite_defined_aggregate_metrics(self) -> None:
        result = run_example()
        self.assertEqual(result["row_counts"], {"train": 144, "val": 48, "test": 48})
        self.assertIn("Synthetic", result["scope"])
        self.assertEqual(result["evaluation_contract"], "1.0")
        self.assertEqual(result["configuration"]["baseline_feature_columns"], ["feature_a", "feature_b", "feature_c"])
        self.assertEqual(result["configuration_sha256"], hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest())
        for name in ("val_metrics", "test_metrics"):
            for metric in METRIC_NAMES:
                value = result[name][metric]
                self.assertTrue(np.isfinite(value))
                self.assertEqual(result[name]["status"][metric], "defined")
                self.assertGreaterEqual(value, -1.0 if metric == "mcc" else 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertNotIn("predictions", result)
        json.dumps(result, allow_nan=False)

    def test_baseline_rejects_target_and_context_as_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for feature in ("label", "context_level", "missing"):
                configuration = load_json(DEFAULT_CONFIG)
                configuration["baseline_feature_columns"] = [feature]
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(configuration), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-label"):
                    run_example(path)

    def test_baseline_rejects_nonfinite_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = load_json(DEFAULT_CONFIG)
            configuration["probability_threshold"] = float("nan")
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(configuration), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_example(path)

    def test_baseline_rejects_unsupported_or_missing_model_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for changes in ({"class_weight": "balanced"}, {"tol": 0.01}):
                configuration = load_json(DEFAULT_CONFIG)
                configuration["model"].update(changes)
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(configuration), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unsupported or missing keys"):
                    run_example(path)
            configuration = load_json(DEFAULT_CONFIG)
            del configuration["model"]["C"]
            path.write_text(json.dumps(configuration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported or missing keys"):
                run_example(path)

    def test_reference_aggregate_result_matches_with_numeric_tolerance(self) -> None:
        expected = load_json(EXAMPLE_DIRECTORY.parent / "docs" / "reference_synthetic_result.json")
        actual = run_example()
        for key in ("fixture_sha256", "configuration_sha256", "configuration", "row_counts", "evaluation_contract"):
            self.assertEqual(actual[key], expected[key])
        for partition in ("val_metrics", "test_metrics"):
            for key in ("status", "support", "confusion_matrix"):
                self.assertEqual(actual[partition][key], expected[partition][key])
            for metric in METRIC_NAMES:
                np.testing.assert_allclose(actual[partition][metric], expected[partition][metric], rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
