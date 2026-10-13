"""Persistence helpers shared by model trainers."""

import json
import pickle
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_pickle_model(model, model_name, result, feature_columns, model_dir, config=None):
    output_dir = Path(model_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "feature_columns.json", feature_columns)
    if config is not None:
        write_json(output_dir / "model_config.json", config)


def save_keras_model(model, model_name, result, feature_columns, model_dir, config):
    output_dir = Path(model_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "feature_columns.json", feature_columns)
    write_json(output_dir / "model_config.json", config)
