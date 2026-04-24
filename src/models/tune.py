import mlflow
import optuna
from train import train, EXPERIMENT_NAME

def objective(trial : optuna.Trial):
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

    return train(f"trial_{trial.number}", params)


mlflow.set_experiment(EXPERIMENT_NAME)

study = optuna.create_study(direction="minimize")

with mlflow.start_run(run_name="tuning"):
    study.optimize(objective, n_trials=20)

    print(f"\nBest trial:  {study.best_trial.number}")
    print(f"Best score:  {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_weighted_rmse", study.best_value)