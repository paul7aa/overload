import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

MODEL_NAME = "overload-predictor"
ALIAS_PROD = "Production"

_ROOT = Path(__file__).parent.parent.parent
EXERCISE_MAP_PATH = _ROOT / "data" / "exercise_map.json"

client = MlflowClient()


def load_model():
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_PROD)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS_PROD}")
    return model, version


def load_exercise_map() -> dict[str, int]:
    with open(EXERCISE_MAP_PATH) as f:
        return json.load(f)
