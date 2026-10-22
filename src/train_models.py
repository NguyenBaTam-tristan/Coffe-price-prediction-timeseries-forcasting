"""Train and compare regression models for coffee prices.

The dataset is split chronologically without shuffling. Models train on the
first 80% of observations and are evaluated on the last 20%.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

try:
    from tensorflow import keras
except ImportError:  # TensorFlow is optional when the LSTM model is not needed.
    keras = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "features" / "training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "next_day_price"
EXCLUDED_FEATURE_COLUMNS = {
    TARGET_COLUMN,
    "date",
    "date_gap_days",
    "province",
    "product",
    "unit",
    "source",
    "source_url",
}

LSTM_SEQUENCE_LENGTH = 30
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 16
LSTM_VALIDATION_SPLIT = 0.15
LSTM_EARLY_STOPPING_PATIENCE = 8
LSTM_TARGET_SCALE = 1000.0

XGBOOST_COMMON_PARAMETERS = {
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "reg_lambda": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 2,
}
XGBOOST_CANDIDATES = [
    {
        "name": f"depth_{depth}_estimators_{n_estimators}",
        "parameters": {"n_estimators": n_estimators, "max_depth": depth},
    }
    for depth, n_estimators in ((2, 80), (2, 120), (2, 200), (3, 80), (3, 120), (3, 200))
]


def metrics(actual, predicted):
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((actual - np.mean(actual)) ** 2)),
    }


def select_feature_columns(dataset):
    """Select numeric features available at the prediction date."""
    return [
        column
        for column in dataset.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def prepare_lstm_sequences(dataset, feature_columns, target_column, sequence_length):
    """Create rolling feature sequences and align each target to its last row."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")
    if len(dataset) < sequence_length:
        raise ValueError(
            f"At least {sequence_length} rows are required to create sequences; "
            f"received {len(dataset)}."
        )

    features = dataset[feature_columns].to_numpy(dtype=np.float32)
    targets = dataset[target_column].to_numpy(dtype=np.float32)
    sequences = np.stack(
        [
            features[index : index + sequence_length]
            for index in range(len(dataset) - sequence_length + 1)
        ]
    )
    return sequences, targets[sequence_length - 1 :]


def prepare_lstm_test_sequences(
    train, test, feature_columns, target_column, sequence_length
):
    """Keep historical rows from train as context for the first test window."""
    context = pd.concat(
        [train.tail(sequence_length - 1), test],
        ignore_index=True,
    )
    return prepare_lstm_sequences(
        context, feature_columns, target_column, sequence_length
    )


def save_pickle_model(
    model, name, result, feature_columns, config=None, model_dir=MODEL_DIR
):
    output_dir = model_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )
    if config is not None:
        (output_dir / "model_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )


def train_linear_regression(train, test, model_dir=MODEL_DIR):
    feature_columns = select_feature_columns(train)
    model = LinearRegression()
    model.fit(train[feature_columns], train[TARGET_COLUMN])
    prediction = model.predict(test[feature_columns])
    result = {
        "model": "linear_regression",
        **metrics(test[TARGET_COLUMN].to_numpy(), prediction),
    }
    save_pickle_model(
        model, "linear_regression", result, feature_columns, model_dir=model_dir
    )
    return result





def build_lstm_model(sequence_length, feature_count):
    if keras is None:
        raise ImportError("TensorFlow is required to build the LSTM model.")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, feature_count)),
            keras.layers.Normalization(axis=-1, name="feature_normalization"),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(train, test, model_dir=MODEL_DIR):
    if keras is None:
        print("TensorFlow not installed; skipping LSTM model.")
        return None

    feature_columns = select_feature_columns(train)
    train_sequences, train_targets = prepare_lstm_sequences(
        train, feature_columns, TARGET_COLUMN, LSTM_SEQUENCE_LENGTH
    )
    test_sequences, test_targets = prepare_lstm_test_sequences(
        train, test, feature_columns, TARGET_COLUMN, LSTM_SEQUENCE_LENGTH
    )

    keras.utils.set_random_seed(42)
    model = build_lstm_model(LSTM_SEQUENCE_LENGTH, len(feature_columns))
    fit_sequence_count = int(
        len(train_sequences) * (1 - LSTM_VALIDATION_SPLIT)
    )
    model.get_layer("feature_normalization").adapt(
        train_sequences[:fit_sequence_count]
    )
    train_price = train["coffee_price"].to_numpy(dtype=np.float32)[
        LSTM_SEQUENCE_LENGTH - 1 :
    ]
    train_change = (train_targets - train_price) / LSTM_TARGET_SCALE
    model.fit(
        train_sequences,
        train_change,
        validation_split=LSTM_VALIDATION_SPLIT,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=LSTM_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
    )
    predicted_change = (
        model.predict(test_sequences, verbose=0).ravel() * LSTM_TARGET_SCALE
    )
    prediction = test["coffee_price"].to_numpy() + predicted_change
    result = {"model": "lstm", **metrics(test_targets, prediction)}

    output_dir = model_dir / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "sequence_length": LSTM_SEQUENCE_LENGTH,
                "target_mode": (
                    "next_day_price - coffee_price; add predicted change to current price"
                ),
                "target_scale": LSTM_TARGET_SCALE,
                "architecture": [
                    "Normalization(axis=-1)",
                    "LSTM(32)",
                    "Dropout(0.2)",
                    "Dense(16, relu)",
                    "Dense(1)",
                ],
                "loss": "mse",
                "optimizer": "adam",
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": LSTM_EARLY_STOPPING_PATIENCE,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def train_xgboost(train, test, model_dir=MODEL_DIR):
    feature_columns = select_feature_columns(train)
    validation_split = int(len(train) * 0.8)
    tuning_train = train.iloc[:validation_split]
    validation = train.iloc[validation_split:]
    validation_results = []

    for candidate in XGBOOST_CANDIDATES:
        candidate_model = XGBRegressor(
            **XGBOOST_COMMON_PARAMETERS,
            **candidate["parameters"],
        )
        tuning_target = (
            tuning_train[TARGET_COLUMN] - tuning_train["coffee_price"]
        )
        candidate_model.fit(tuning_train[feature_columns], tuning_target)
        predicted_change = candidate_model.predict(validation[feature_columns])
        validation_prediction = (
            validation["coffee_price"].to_numpy() + predicted_change
        )
        validation_metrics = metrics(
            validation[TARGET_COLUMN].to_numpy(),
            validation_prediction,
        )
        validation_results.append(
            {
                "name": candidate["name"],
                "parameters": candidate["parameters"],
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
            }
        )

    selected_candidate = min(
        validation_results,
        key=lambda result: result["validation_mae"],
    )
    model = XGBRegressor(
        **XGBOOST_COMMON_PARAMETERS,
        **selected_candidate["parameters"],
    )
    train_change = train[TARGET_COLUMN] - train["coffee_price"]
    model.fit(train[feature_columns], train_change)
    predicted_change = model.predict(test[feature_columns])
    prediction = test["coffee_price"].to_numpy() + predicted_change
    result = {
        "model": "xgboost",
        **metrics(test[TARGET_COLUMN].to_numpy(), prediction),
    }
    config = {
        "selected_candidate": selected_candidate["name"],
        "parameters": {
            **XGBOOST_COMMON_PARAMETERS,
            **selected_candidate["parameters"],
        },
        "target_mode": (
            "next_day_price - coffee_price; add predicted change to current price"
        ),
        "selection_metric": "validation_mae",
        "validation_candidates": [
            {key: value for key, value in candidate.items() if key != "parameters"}
            for candidate in validation_results
        ],
    }
    save_pickle_model(
        model, "xgboost", result, feature_columns, config, model_dir=model_dir
    )
    return result


def train_models(dataset_path=DATASET_PATH, model_dir=MODEL_DIR):
    dataset = pd.read_csv(dataset_path, parse_dates=["date"])
    split_index = int(len(dataset) * 0.8)
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    trainers = (
        train_linear_regression,
        train_lstm,
        train_xgboost,
    )
    comparison = []
    for trainer in trainers:
        result = trainer(train, test, model_dir)
        if result is not None:
            comparison.append(result)

    comparison_frame = pd.DataFrame(comparison).sort_values("mae")
    model_dir.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(model_dir / "model_comparison.csv", index=False)
    print(comparison_frame.to_string(index=False))
    return comparison_frame


def main():
    train_models()


if __name__ == "__main__":
    main()
