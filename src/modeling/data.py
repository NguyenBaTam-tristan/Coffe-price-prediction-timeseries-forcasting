"""Dataset loading, feature selection, and chronological splitting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    DATASET_PATH,
    EXCLUDED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_RATIO,
)


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def load_dataset(dataset_path=DATASET_PATH):
    """Load the prepared feature dataset."""
    return pd.read_csv(dataset_path, parse_dates=["date"])


def select_feature_columns(dataset):
    """Select numeric inputs that are available at the prediction date."""
    return [
        column
        for column in dataset.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def split_chronologically(dataset, target_column=TARGET_COLUMN, train_ratio=TRAIN_RATIO):
    """Split rows by time without shuffling."""
    feature_columns = select_feature_columns(dataset)
    split_index = int(len(dataset) * train_ratio)
    return DatasetSplit(
        train=dataset.iloc[:split_index],
        test=dataset.iloc[split_index:],
        feature_columns=feature_columns,
        target_column=target_column,
    )


def make_sequences(dataset, feature_columns, target_column, sequence_length):
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
