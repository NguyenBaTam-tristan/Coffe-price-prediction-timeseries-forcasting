"""LSTM sequence model trainer."""

import pandas as pd

from ..artifacts import save_keras_model
from ..config import (
    LSTM_BATCH_SIZE,
    LSTM_EARLY_STOPPING_PATIENCE,
    LSTM_EPOCHS,
    LSTM_VALIDATION_SPLIT,
    LSTM_SEQUENCE_LENGTH,
    MODEL_DIR,
    TARGET_COLUMN,
)
from ..data import make_sequences, select_feature_columns
from ..metrics import calculate_metrics

try:
    from tensorflow import keras
except ImportError:  # TensorFlow is optional in environments without LSTM support.
    keras = None


def _build_model(sequence_length, feature_count):
    if keras is None:
        raise ImportError("TensorFlow is required to build the LSTM model.")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, feature_count)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(train: pd.DataFrame, test: pd.DataFrame):
    if keras is None:
        print("TensorFlow not installed; skipping LSTM model.")
        return None

    features = select_feature_columns(train)
    train_sequences, train_targets = make_sequences(
        train,
        features,
        TARGET_COLUMN,
        LSTM_SEQUENCE_LENGTH,
    )
    test_sequences, test_targets = make_sequences(
        test,
        features,
        TARGET_COLUMN,
        LSTM_SEQUENCE_LENGTH,
    )

    model = _build_model(LSTM_SEQUENCE_LENGTH, len(features))
    model.fit(
        train_sequences,
        train_targets,
        validation_split=LSTM_VALIDATION_SPLIT,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=LSTM_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
    )
    prediction = model.predict(test_sequences, verbose=0).ravel()
    result = {"model": "lstm", **calculate_metrics(test_targets, prediction)}
    config = {
        "sequence_length": LSTM_SEQUENCE_LENGTH,
        "architecture": [
            "LSTM(64, return_sequences=True)",
            "Dropout(0.2)",
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
    }
    save_keras_model(
        model,
        result["model"],
        result,
        features,
        MODEL_DIR,
        config,
    )
    return result
