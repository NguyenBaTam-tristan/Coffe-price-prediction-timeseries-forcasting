import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import pipeline, train_models, train_models_coffee_only


class ModelPreparationTests(unittest.TestCase):
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
        self.assertEqual(
            train_models.select_feature_columns(self.dataset),
            ["coffee_price"],
        )

    def test_sequence_targets_align_with_last_row_in_each_window(self):
        sequences, targets = train_models.prepare_lstm_sequences(
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

    def test_test_sequences_keep_training_rows_as_historical_context(self):
        train = self.dataset.iloc[:3]
        test = self.dataset.iloc[3:]

        sequences, targets = train_models.prepare_lstm_test_sequences(
            train,
            test,
            ["coffee_price"],
            "next_day_price",
            sequence_length=3,
        )

        self.assertEqual(len(sequences), len(test))
        np.testing.assert_array_equal(
            sequences[:, :, 0],
            [[11, 12, 13], [12, 13, 14], [13, 14, 15]],
        )
        np.testing.assert_array_equal(targets, [14, 15, 16])

    def test_sequence_length_must_fit_dataset(self):
        with self.assertRaises(ValueError):
            train_models.prepare_lstm_sequences(
                self.dataset.iloc[:2],
                ["coffee_price"],
                "next_day_price",
                sequence_length=3,
            )

    def test_coffee_only_extraction_drops_weather_before_feature_engineering(self):
        coffee = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=40),
                "price": np.arange(40.0),
                "temperature_2m_mean": np.arange(40.0),
                "rain_sum": np.arange(40.0),
            }
        )

        features = pipeline.build_coffee_only_features(coffee)

        self.assertTrue(
            all(
                not any(
                    token in column
                    for token in (
                        "temperature",
                        "rain",
                        "precipitation",
                        "humidity",
                        "soil_moisture",
                        "radiation",
                    )
                )
                for column in features.columns
            )
        )
        self.assertIn("price_lag_30", features.columns)
        self.assertIn("coffee_price", features.columns)
        self.assertIn("next_day_price", features.columns)

    def test_metrics_calculate_regression_scores(self):
        result = train_models.metrics(
            np.array([1.0, 3.0]),
            np.array([2.0, 2.0]),
        )

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
            def __init__(self, **parameters):
                self.parameters = parameters

            def fit(self, features, targets):
                fitted_targets.append(np.asarray(targets).copy())

            def predict(self, features):
                return np.full(len(features), 77.0)

        with (
            patch.object(train_models, "XGBRegressor", FakeRegressor),
            patch.object(train_models, "save_pickle_model"),
        ):
            result = train_models.train_xgboost(train, test)

        expected_change = train["next_day_price"] - train["coffee_price"]
        np.testing.assert_array_equal(fitted_targets[0], expected_change.iloc[:6])
        np.testing.assert_array_equal(fitted_targets[-1], expected_change)
        self.assertEqual(result["mae"], 76.0)


class CoffeeOnlyEntrypointTests(unittest.TestCase):
    def test_direct_script_execution_can_import_train_models(self):
        script_path = Path(train_models_coffee_only.__file__)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import runpy; "
                    f"runpy.run_path({str(script_path)!r}, "
                    "run_name='coffee_only_import_check')"
                ),
            ],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
