import json
import os

import requests
from prefect import flow, task

from src.data.preprocess import preprocess
from src.models.tune import run_tuning
from src.models.train import run_training, DEFAULT_HYPERPARAMS
from src.models.evaluate import evaluate_and_promote

BEST_PARAMS_PATH = "data/best_params.json"

_API_URL = os.environ.get("API_URL", "http://api:8000")
_API_KEY = os.environ.get("API_KEY", "")
_EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


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
def evaluate_task() -> bool:
    return evaluate_and_promote()


@task(name="notify-and-reload", log_prints=True)
def notify_and_reload_task(promoted: bool):
    if not promoted:
        print("Challenger not promoted — skipping reload and notifications.")
        return

    # Hot-swap the API to the newly promoted model
    try:
        resp = requests.post(
            f"{_API_URL}/reload",
            headers={"X-API-Key": _API_KEY},
            timeout=10,
        )
        print(f"[reload] {resp.json()}")
    except Exception as exc:
        print(f"[reload] failed: {exc}")

    # Fetch all registered push tokens from the database
    from src.api.db import SessionLocal, PushToken, create_tables
    create_tables()
    db = SessionLocal()
    try:
        tokens = [row.token for row in db.query(PushToken).all()]
    finally:
        db.close()

    if not tokens:
        print("[push] no tokens registered — skipping notifications.")
        return

    # Send push notifications via Expo's push API
    messages = [
        {
            "to": token,
            "title": "Model updated",
            "body": "Your AI predictions just got smarter based on your workouts.",
            "sound": "default",
            "channelId": "workout",
        }
        for token in tokens
    ]
    try:
        resp = requests.post(
            _EXPO_PUSH_URL,
            json=messages,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        print(f"[push] sent {len(tokens)} notification(s): HTTP {resp.status_code}")
    except Exception as exc:
        print(f"[push] failed: {exc}")


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
    promoted = evaluate_task()
    notify_and_reload_task(promoted)


if __name__ == "__main__":
    # Registers this flow as a scheduled Prefect deployment and serves it.
    # The worker container runs this file directly; Prefect picks up both
    # scheduled runs (weekly cron) and manually triggered runs from the UI.
    retraining_pipeline.serve(
        name="scheduled-retraining",
        cron="0 3 * * 0",  # every Sunday at 03:00 UTC
    )
