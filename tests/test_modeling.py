import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.modeling.data import make_sequences, select_feature_columns, split_chronologically
from src.modeling.metrics import calculate_metrics
from src.modeling.trainers import xgboost_model


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=6),
                "coffee_price": [10, 11, 12, 13, 14, 15],
                "next_day_price": [11, 12, 13, 14, 15, 16],
                "date_gap_days": [1] * 6,
                "category": ["coffee"] * 6,
            }
        )

    def test_feature_selection_excludes_target_and_non_numeric_columns(self):
        self.assertEqual(select_feature_columns(self.dataset), ["coffee_price"])

    def test_chronological_split_preserves_row_order(self):
        split = split_chronologically(self.dataset, train_ratio=0.5)

        self.assertEqual(split.train["coffee_price"].tolist(), [10, 11, 12])
        self.assertEqual(split.test["coffee_price"].tolist(), [13, 14, 15])

    def test_sequence_targets_align_with_last_row_in_each_window(self):
        sequences, targets = make_sequences(
            self.dataset,
            ["coffee_price"],
            "next_day_price",
            sequence_length=3,
        )

        np.testing.assert_array_equal(
            sequences[:, :, 0],
            [[10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15]],
        )
        np.testing.assert_array_equal(targets, [13, 14, 15, 16])

    def test_sequence_length_cannot_exceed_dataset_size(self):
        with self.assertRaises(ValueError):
            make_sequences(
                self.dataset.iloc[:2],
                ["coffee_price"],
                "next_day_price",
                sequence_length=3,
            )


class MetricTests(unittest.TestCase):
    def test_calculates_regression_metrics(self):
        result = calculate_metrics(np.array([1.0, 3.0]), np.array([2.0, 2.0]))

        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["r2"], 0.0)


class XGBoostTrainerTests(unittest.TestCase):
    def test_trains_on_price_change_and_adds_it_to_current_price(self):
        train = pd.DataFrame(
            {
                "coffee_price": np.arange(100.0, 108.0),
                "feature": np.arange(8.0),
                "next_day_price": np.arange(101.0, 109.0),
            }
        )
        test = pd.DataFrame(
            {
                "coffee_price": [108.0, 109.0],
                "feature": [8.0, 9.0],
                "next_day_price": [109.0, 110.0],
            }
        )
        fitted_targets = []

        class FakeRegressor:
            def __init__(self, **_parameters):
                del _parameters

            def fit(self, _features, targets):
                del _features
                fitted_targets.append(np.asarray(targets).copy())

            def predict(self, features):
                return np.full(len(features), 77.0)

        with (
            patch.object(xgboost_model, "XGBRegressor", FakeRegressor),
            patch.object(xgboost_model, "save_pickle_model"),
        ):
            result = xgboost_model.train_xgboost(train, test)

        expected_change = train["next_day_price"] - train["coffee_price"]
        np.testing.assert_array_equal(fitted_targets[0], expected_change.iloc[:6])
        np.testing.assert_array_equal(fitted_targets[-1], expected_change)
        self.assertEqual(result["mae"], 76.0)


if __name__ == "__main__":
    unittest.main()
