import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prefect import flow, task

from data.preprocess import preprocess
from models.train import run_training
from models.evaluate import evaluate_and_promote


@task(name="preprocess", log_prints=True)
def preprocess_task():
    preprocess()


@task(name="train", log_prints=True)
def train_task():
    run_training()


@task(name="evaluate-and-promote", log_prints=True)
def evaluate_task():
    evaluate_and_promote()


@flow(name="overload-retraining-pipeline", log_prints=True)
def retraining_pipeline():
    preprocess_task()
    train_task()
    evaluate_task()


if __name__ == "__main__":
    retraining_pipeline()
