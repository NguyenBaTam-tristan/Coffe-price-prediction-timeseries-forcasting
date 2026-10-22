"""Train models using coffee price and calendar features only."""

import pandas as pd

if __package__:
    from . import train_models
else:
    import train_models


DATASET_PATH = (
    train_models.PROJECT_ROOT
    / "data"
    / "features"
    / "training_dataset_coffee_only.csv"
)
MODEL_DIR = train_models.MODEL_DIR / "coffee_only"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run src/pipeline.py first to build it."
        )

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    split_index = int(len(dataset) * 0.8)
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    trainers = (
        train_models.train_linear_regression,
        train_models.train_lstm,
        train_models.train_xgboost,
    )
    comparison = []
    for trainer in trainers:
        result = trainer(train, test, MODEL_DIR)
        if result is not None:
            comparison.append(result)

    comparison_frame = pd.DataFrame(comparison).sort_values("mae")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
