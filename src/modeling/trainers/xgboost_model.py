"""XGBoost trainer with chronological validation-based candidate selection."""

import pandas as pd
from xgboost import XGBRegressor

from ..artifacts import save_pickle_model
from ..config import (
    CURRENT_PRICE_COLUMN,
    MODEL_DIR,
    TARGET_COLUMN,
    TRAIN_RATIO,
    XGBOOST_CANDIDATES,
    XGBOOST_COMMON_PARAMETERS,
)
from ..data import select_feature_columns
from ..metrics import calculate_metrics


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    features = select_feature_columns(train)

    validation_split = int(len(train) * TRAIN_RATIO)
    tuning_train = train.iloc[:validation_split]
    validation = train.iloc[validation_split:]
    validation_results = []

    for candidate in XGBOOST_CANDIDATES:
        candidate_model = XGBRegressor(
            **XGBOOST_COMMON_PARAMETERS,
            **candidate["parameters"],
        )
        tuning_target = (
            tuning_train[TARGET_COLUMN] - tuning_train[CURRENT_PRICE_COLUMN]
        )
        candidate_model.fit(tuning_train[features], tuning_target)
        predicted_change = candidate_model.predict(validation[features])
        validation_prediction = (
            validation[CURRENT_PRICE_COLUMN].to_numpy() + predicted_change
        )
        candidate_metrics = calculate_metrics(
            validation[TARGET_COLUMN].to_numpy(),
            validation_prediction,
        )
        validation_results.append(
            {
                "name": candidate["name"],
                "parameters": candidate["parameters"],
                "validation_mae": candidate_metrics["mae"],
                "validation_rmse": candidate_metrics["rmse"],
                "validation_r2": candidate_metrics["r2"],
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
    train_change = train[TARGET_COLUMN] - train[CURRENT_PRICE_COLUMN]
    model.fit(train[features], train_change)
    predicted_change = model.predict(test[features])
    prediction = test[CURRENT_PRICE_COLUMN].to_numpy() + predicted_change
    result = {
        "model": "xgboost",
        **calculate_metrics(test[TARGET_COLUMN].to_numpy(), prediction),
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
        model,
        result["model"],
        result,
        features,
        MODEL_DIR,
        config,
    )
    return result
