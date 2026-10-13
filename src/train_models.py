"""Command-line entry point for training and comparing regression models."""

import pandas as pd

from modeling.config import DATASET_PATH, MODEL_DIR, TARGET_COLUMN
from modeling.data import load_dataset, split_chronologically
from modeling.trainers import MODEL_TRAINERS

def main():
    def train_all_models(dataset_path=DATASET_PATH):
        dataset = load_dataset(dataset_path)
        split = split_chronologically(dataset, TARGET_COLUMN)

        results = []
        for trainer in MODEL_TRAINERS:
            result = trainer(split.train, split.test)
            if result is not None:
                results.append(result)
        comparison = pd.DataFrame(results).sort_values("mae")
        comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
        print(comparison.to_string(index=False))
        return comparison

    train_all_models()

if __name__ == "__main__":
    main()
