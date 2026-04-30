import json
import os

from prefect import flow, task

from src.data.preprocess import preprocess
from src.models.tune import run_tuning
from src.models.train import run_training, DEFAULT_HYPERPARAMS
from src.models.evaluate import evaluate_and_promote

BEST_PARAMS_PATH = "data/best_params.json"


@task(name="preprocess", log_prints=True)
def preprocess_task():
    preprocess()


@task(name="tune", log_prints=True)
def tune_task() -> dict:
    return run_tuning()


@task(name="save-params", log_prints=True)
def save_params_task(params: dict):
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)


@task(name="load-params", log_prints=True)
def load_params_task() -> dict:
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    return DEFAULT_HYPERPARAMS


@task(name="train", log_prints=True)
def train_task(hyperparams: dict):
    run_training(hyperparams)


@task(name="evaluate-and-promote", log_prints=True)
def evaluate_task():
    evaluate_and_promote()


@flow(name="overload-tuning-pipeline", log_prints=True)
def tuning_pipeline():
    """Run occasionally (e.g. monthly) to find best hyperparameters."""
    preprocess_task()
    best_params = tune_task()
    save_params_task(best_params)


@flow(name="overload-retraining-pipeline", log_prints=True)
def retraining_pipeline():
    """Run frequently (triggered by user log threshold) using stored best params."""
    preprocess_task()
    best_params = load_params_task()
    train_task(best_params)
    evaluate_task()


if __name__ == "__main__":
    retraining_pipeline()
