"""Shared configuration for model training."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "features" / "training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "next_day_price"
CURRENT_PRICE_COLUMN = "coffee_price"
TRAIN_RATIO = 0.8
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
XGBOOST_CANDIDATES = tuple(
    {
        "name": f"depth_{depth}_estimators_{n_estimators}",
        "parameters": {"n_estimators": n_estimators, "max_depth": depth},
    }
    for depth, n_estimators in ((2, 80), (2, 120), (2, 200), (3, 80), (3, 120), (3, 200))
)
