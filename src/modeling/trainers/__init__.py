"""Registry of the model trainers used by the training pipeline."""

from .linear_regression import train_linear_regression
from .lstm import train_lstm
from .xgboost_model import train_xgboost


MODEL_TRAINERS = (
    train_linear_regression,
    train_lstm,
    train_xgboost,
)
