from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from fgst_keagn_support.data.loader import frame_to_tensors, load_data_bundle, load_dataframe
from fgst_keagn_support.data.schema import infer_schema
from fgst_keagn_support.data.splitter import stratified_split
from fgst_keagn_support.data.validation import normalize_binary_labels, validate_feature_columns
from fgst_keagn_support.utils.io import load_json, load_yaml, save_dataframe, save_json
from fgst_keagn_support.utils.metrics import METRIC_NAMES, compute_binary_metrics, stable_softmax
from fgst_keagn_support.utils.seed import set_global_seed


class UtilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "feature_a": np.arange(100, dtype=float),
            "context_level": np.linspace(0.0, 1.0, 100),
            "label": [0, 1] * 50,
        })

    def test_all_public_modules_import(self) -> None:
        for name in (
            "data.loader", "data.schema", "data.splitter", "data.validation",
            "utils.io", "utils.metrics", "utils.seed",
        ):
            self.assertIsNotNone(importlib.import_module(f"fgst_keagn_support.{name}"))

    def test_label_normalization_supported_encodings(self) -> None:
        for values in ([0, 1], [-1, 1], [False, True]):
            np.testing.assert_array_equal(normalize_binary_labels(np.array(values)), [0, 1])
        for values, expected in (([0, 0], [0, 0]), ([1, 1], [1, 1]), ([-1, -1], [0, 0])):
            np.testing.assert_array_equal(normalize_binary_labels(np.array(values)), expected)

    def test_labels_reject_missing_unsupported_and_mixed_encodings(self) -> None:
        for values in ([np.nan, 1], [np.inf, 1], [1, 2], [-1, 0, 1], [0.0, 0.5], ["0", "1"], [1j, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_binary_labels(np.array(values))

    def test_labels_reject_empty_or_multidimensional_input(self) -> None:
        for values in ([], [[0], [1]], 1):
            with self.assertRaises(ValueError):
                normalize_binary_labels(np.array(values))

    def test_schema_excludes_label_and_identifies_context(self) -> None:
        self.assertEqual(asdict(infer_schema(self.frame, "label", ["context_level"])), {
            "label_column": "label", "dynamic_context_columns": ["context_level"],
            "feature_columns": ["feature_a", "context_level"], "static_feature_columns": ["feature_a"],
        })

    def test_schema_rejects_missing_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "label column must exist"):
            infer_schema(self.frame, "missing", [])

    def test_schema_rejects_invalid_context_references(self) -> None:
        for context in (["missing"], ["label"], ["context_level", "context_level"], [1], "context_level"):
            with self.assertRaises(ValueError):
                infer_schema(self.frame, "label", context)

    def test_schema_rejects_duplicate_dataframe_columns(self) -> None:
        frame = pd.DataFrame([[1, 2, 0]], columns=["feature", "feature", "label"])
        with self.assertRaisesRegex(ValueError, "unique"):
            infer_schema(frame, "label", [])

    def test_feature_allowlist_rejects_target_missing_and_duplicates(self) -> None:
        for values in (["label"], ["missing"], ["feature_a", "feature_a"], [], "feature_a"):
            with self.assertRaises(ValueError):
                validate_feature_columns(values, ["feature_a"])

    def test_io_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"purpose": "synthetic", "undefined": None, "items": [1, 2, 3]}
            save_json(root / "nested" / "sample.json", payload)
            self.assertEqual(load_json(root / "nested" / "sample.json"), payload)
            self.assertEqual(load_json(root / "absent.json"), {})
            self.assertEqual(load_json(root / "absent.json", default=[]), [])
            yaml_path = root / "example.yaml"
            yaml_path.write_text("purpose: synthetic\ncount: 3\n", encoding="utf-8")
            self.assertEqual(load_yaml(yaml_path), {"purpose": "synthetic", "count": 3})
            csv_path = root / "nested" / "sample.csv"
            save_dataframe(csv_path, self.frame)
            assert_frame_equal(load_data_bundle(csv_path, "label", ["context_level"]).dataframe, self.frame)

    def test_json_rejects_nonfinite_before_touching_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_bytes(b'{"preserved": true}\n')
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.assertRaises(ValueError):
                    save_json(path, {"metric": value})
                self.assertEqual(path.read_bytes(), b'{"preserved": true}\n')

    def test_loader_normalizes_signed_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signed.csv"
            signed = self.frame.copy()
            signed["label"] = np.where(signed["label"] == 1, 1, -1)
            signed.to_csv(path, index=False)
            self.assertEqual(load_data_bundle(path, "label", []).dataframe["label"].tolist(), self.frame["label"].tolist())

    def test_loader_rejects_unsupported_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            invalid = self.frame.copy()
            invalid["label"] += 1
            invalid.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data_bundle(path, "label", [])

    def test_loader_rejects_duplicate_raw_csv_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("feature,feature,label\n1,2,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_dataframe(path)

    def test_split_disjoint_coverage_sizes_and_repeatability(self) -> None:
        first = stratified_split(self.frame, "label", 0.6, 0.2, 0.2, 91357)
        second = stratified_split(self.frame, "label", 0.6, 0.2, 0.2, 91357)
        self.assertEqual({key: len(frame) for key, frame in first.items()}, {"train": 60, "val": 20, "test": 20})
        memberships = [set(frame["feature_a"]) for frame in first.values()]
        self.assertEqual(set.union(*memberships), set(self.frame["feature_a"]))
        for i, current in enumerate(memberships):
            for other in memberships[i + 1:]:
                self.assertTrue(current.isdisjoint(other))
        for name, frame in first.items():
            assert_frame_equal(frame, second[name])
            self.assertEqual(set(frame["label"]), {0, 1})
            self.assertAlmostEqual(float(frame["label"].mean()), 0.5)

    def test_split_integral_seventy_fifteen_fifteen(self) -> None:
        result = stratified_split(self.frame, "label", 0.7, 0.15, 0.15, 91357)
        self.assertEqual({key: len(frame) for key, frame in result.items()}, {"train": 70, "val": 15, "test": 15})

    def test_split_rejects_invalid_ratios(self) -> None:
        for ratios in ((0.6, 0.3, 0.3), (1.0, 0.0, 0.0), (0.6, -0.1, 0.5), (np.nan, 0.2, 0.2), (np.inf, 0.2, 0.2)):
            with self.assertRaises(ValueError):
                stratified_split(self.frame, "label", *ratios, 91357)

    def test_split_reports_small_or_imbalanced_infeasibility(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row"):
            stratified_split(self.frame.iloc[:2], "label", 0.6, 0.2, 0.2, 91357)
        frame = pd.DataFrame({"feature": range(30), "label": [0] * 27 + [1] * 3})
        with self.assertRaisesRegex(ValueError, "Validation/test stratification is infeasible"):
            stratified_split(frame, "label", 0.6, 0.2, 0.2, 91357)

    def test_perfect_predictions(self) -> None:
        result = compute_binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        for name in METRIC_NAMES:
            self.assertAlmostEqual(result[name], 1.0)
            self.assertEqual(result["status"][name], "defined")
        self.assertEqual(result["support"], {"samples": 4, "positive": 2, "negative": 2, "predicted_positive": 2})
        self.assertEqual(result["confusion_matrix"], {"tp": 2, "tn": 2, "fp": 0, "fn": 0})

    def test_nonperfect_ranking_average_precision_and_confusion(self) -> None:
        result = compute_binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["mcc"], 1 / np.sqrt(3))

    def test_metrics_reject_missing_and_wrong_truth(self) -> None:
        for labels in ([np.nan, 1], [1, 2], [-1, 0, 1], ["0", "1"]):
            with self.assertRaises(ValueError):
                compute_binary_metrics(np.array(labels), np.zeros(len(labels)))

    def test_metrics_reject_invalid_probabilities(self) -> None:
        for probabilities in ([np.nan, 0.9], [np.inf, 0.9], [float("-inf"), 0.9], [-0.1, 0.9], [0.1, 1.1], ["0.1", "0.9"], [True, False]):
            with self.assertRaises(ValueError):
                compute_binary_metrics(np.array([0, 1]), np.array(probabilities))

    def test_metrics_reject_unaligned_or_multidimensional_input(self) -> None:
        for probabilities in ([], [0.1], [[0.1], [0.9]]):
            with self.assertRaises(ValueError):
                compute_binary_metrics(np.array([0, 1]), np.array(probabilities))

    def test_metrics_reject_invalid_threshold(self) -> None:
        for threshold in (np.nan, np.inf, -0.1, 1.1, "0.5", [0.5], True):
            with self.assertRaises(ValueError):
                compute_binary_metrics(np.array([0, 1]), np.array([0.1, 0.9]), threshold)

    def test_threshold_uses_inclusive_positive_boundary(self) -> None:
        result = compute_binary_metrics(np.array([0, 1]), np.array([0.1, 0.5]), 0.5)
        self.assertEqual(result["confusion_matrix"]["tp"], 1)

    def test_single_negative_class_reports_undefined_metrics(self) -> None:
        result = compute_binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
        for name in ("roc_auc", "average_precision", "precision", "recall", "f1", "mcc"):
            self.assertIsNone(result[name])
            self.assertNotEqual(result["status"][name], "defined")
        self.assertEqual(result["accuracy"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_negative_truth_with_positive_predictions_has_defined_zero_precision_and_f1(self) -> None:
        result = compute_binary_metrics(np.array([0, 0]), np.array([0.8, 0.9]))
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["status"]["precision"], "defined")
        self.assertEqual(result["status"]["f1"], "defined")
        for name in ("roc_auc", "average_precision", "recall", "mcc"):
            self.assertIsNone(result[name])

    def test_single_positive_class_has_defined_average_precision(self) -> None:
        result = compute_binary_metrics(np.array([1, 1]), np.array([0.8, 0.9]))
        self.assertIsNone(result["roc_auc"])
        self.assertEqual(result["average_precision"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertIsNone(result["mcc"])

    def test_constant_negative_predictions_have_defined_zero_f1(self) -> None:
        result = compute_binary_metrics(np.array([0, 1]), np.array([0.1, 0.2]))
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertIsNone(result["mcc"])

    def test_softmax_large_float_and_unsigned_integer(self) -> None:
        np.testing.assert_allclose(stable_softmax(np.array([1000.0, 1001.0])), stable_softmax(np.array([0.0, 1.0])))
        weights = stable_softmax(np.array([0, 255], dtype=np.uint8))
        self.assertLess(weights[0], 1e-100)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_softmax_rejects_invalid_inputs(self) -> None:
        for values in ([], [[1, 2]], [np.nan, 1], [np.inf, 1], [1j, 0], ["0", "1"]):
            with self.assertRaises(ValueError):
                stable_softmax(np.array(values))

    def test_python_and_numpy_seed_repeatability(self) -> None:
        set_global_seed(91357)
        first = (random.random(), np.random.random())
        set_global_seed(91357)
        self.assertEqual(first, (random.random(), np.random.random()))

    @unittest.skipIf(importlib.util.find_spec("torch") is not None, "Checks the no-PyTorch path.")
    def test_tensor_conversion_explains_optional_dependency(self) -> None:
        with self.assertRaisesRegex(ModuleNotFoundError, "requires PyTorch"):
            frame_to_tensors(self.frame, ["feature_a"], "label")

    def test_tensor_conversion_rejects_label_as_feature_before_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-label"):
            frame_to_tensors(self.frame, ["label"], "label")


if __name__ == "__main__":
    unittest.main()
