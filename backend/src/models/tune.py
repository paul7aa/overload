import os
import mlflow
import optuna
from src.models.train import train, _load_data, EXPERIMENT_NAME

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_tuning(n_trials: int = 20) -> dict:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    train_X, train_Y, val_X, val_Y, train_weights = _load_data()

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state":      42,
            "n_jobs":            -1,
        }
        return train(f"trial_{trial.number}", params, train_X, train_Y, val_X, val_Y, train_weights)

    mlflow.set_experiment(EXPERIMENT_NAME)
    study = optuna.create_study(direction="minimize")
    with mlflow.start_run(run_name="tuning"):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_weighted_rmse", study.best_value)

    return study.best_params


if __name__ == "__main__":
    best = run_tuning()
    print(f"Best params: {best}")
