"""Common regression metrics."""

import numpy as np


def calculate_metrics(actual, predicted):
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((actual - np.mean(actual)) ** 2)),
    }
