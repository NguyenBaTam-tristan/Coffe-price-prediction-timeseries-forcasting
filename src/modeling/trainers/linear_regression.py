"""Linear regression trainer."""

import pandas as pd
from sklearn.linear_model import LinearRegression

from ..artifacts import save_pickle_model
from ..config import MODEL_DIR, TARGET_COLUMN
from ..data import select_feature_columns
from ..metrics import calculate_metrics


def train_linear_regression(train: pd.DataFrame, test: pd.DataFrame):
    features = select_feature_columns(train)
    model = LinearRegression()
    model.fit(train[features], train[TARGET_COLUMN])
    prediction = model.predict(test[features])
    result = {
        "model": "linear_regression",
        **calculate_metrics(test[TARGET_COLUMN].to_numpy(), prediction),
    }
    save_pickle_model(
        model,
        result["model"],
        result,
        features,
        MODEL_DIR,
    )
    return result
