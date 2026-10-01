"""Huấn luyện và so sánh các model regression cho coffee price.

Dữ liệu được chia theo thời gian, tuyệt đối không shuffle. Model chỉ được học
trên 80% quan sát đầu; 20% cuối là test mô phỏng dữ liệu tương lai.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "features" / "training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def metrics(actual, predicted):
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((actual - np.mean(actual)) ** 2)),
    }


def main():
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    target_column = "next_day_price"
    # Chỉ số đo được và biết tại ngày dự báo mới được dùng làm đầu vào.
    excluded_columns = {
        target_column,
        "date",
        "date_gap_days",
        "province",
        "product",
        "unit",
        "source",
        "source_url",
    }
    feature_columns = [
        column
        for column in dataset.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    split_index = int(len(dataset) * 0.8)

    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]
    X_train, y_train = train[feature_columns], train[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
    }

    comparison = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        result = {"model": name, **metrics(y_test.to_numpy(), prediction)}
        comparison.append(result)

        output_dir = MODEL_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "model.pkl").open("wb") as model_file:
            pickle.dump(model, model_file)
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    # Dùng phần cuối của train làm validation để chọn XGBoost mà không nhìn vào test.
    validation_split = int(len(train) * 0.8)
    tuning_train = train.iloc[:validation_split]
    validation = train.iloc[validation_split:]
    xgboost_candidates = [
        {"name": f"depth_{depth}_estimators_{n_estimators}", "parameters": {"n_estimators": n_estimators, "max_depth": depth}}
        for depth, n_estimators in ((2, 80), (2, 120), (2, 200), (3, 80), (3, 120), (3, 200))
    ]
    common_parameters = {
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "reg_lambda": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 2,
    }
    validation_results = []
    # Forecasting the change is more stable than fitting the changing price level.
    # Convert the predicted change back to a price using the known price at time T.
    tuning_target = tuning_train[target_column] - tuning_train["coffee_price"]
    for candidate in xgboost_candidates:
        candidate_model = XGBRegressor(**common_parameters, **candidate["parameters"])
        candidate_model.fit(tuning_train[feature_columns], tuning_target)
        predicted_change = candidate_model.predict(validation[feature_columns])
        validation_prediction = validation["coffee_price"].to_numpy() + predicted_change
        validation_metrics = metrics(validation[target_column].to_numpy(), validation_prediction)
        validation_results.append(
            {
                "name": candidate["name"],
                "parameters": candidate["parameters"],
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
            }
        )

    selected_candidate = min(validation_results, key=lambda result: result["validation_mae"])
    selected_features = feature_columns
    xgboost_model = XGBRegressor(**common_parameters, **selected_candidate["parameters"])
    train_change = y_train - X_train["coffee_price"]
    xgboost_model.fit(X_train[selected_features], train_change)
    predicted_change = xgboost_model.predict(X_test[selected_features])
    xgboost_prediction = X_test["coffee_price"].to_numpy() + predicted_change
    xgboost_result = {"model": "xgboost", **metrics(y_test.to_numpy(), xgboost_prediction)}
    comparison.append(xgboost_result)

    xgboost_dir = MODEL_DIR / "xgboost"
    xgboost_dir.mkdir(parents=True, exist_ok=True)
    with (xgboost_dir / "model.pkl").open("wb") as model_file:
        pickle.dump(xgboost_model, model_file)
    (xgboost_dir / "metrics.json").write_text(
        json.dumps(xgboost_result, indent=2), encoding="utf-8"
    )
    (xgboost_dir / "feature_columns.json").write_text(
        json.dumps(selected_features, indent=2), encoding="utf-8"
    )
    (xgboost_dir / "model_config.json").write_text(
        json.dumps(
            {
                "selected_candidate": selected_candidate["name"],
                "parameters": {**common_parameters, **selected_candidate["parameters"]},
                "target_mode": "next_day_price - coffee_price; add predicted change to current price",
                "selection_metric": "validation_mae",
                "validation_candidates": [
                    {key: value for key, value in result.items() if key != "parameters"}
                    for result in validation_results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison_frame = pd.DataFrame(comparison).sort_values("mae")
    comparison_frame.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
