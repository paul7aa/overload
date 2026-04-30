import os
import logging
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

EXPERIMENT_NAME = "overload-predictor"

TARGETS   = ["delta_sets", "delta_reps", "delta_pct_1rm"]
DROP_COLS = ["program_id", "sets", "reps", "intensity", "pct_1rm", "volume", "sample_weight"] + TARGETS
WEIGHTS   = {"delta_pct_1rm": 0.50, "delta_reps": 0.35, "delta_sets": 0.15}

def _load_data():
    train_data = pd.read_csv("data/train.csv")
    val_data   = pd.read_csv("data/val.csv")
    train_weights = train_data["sample_weight"] if "sample_weight" in train_data.columns else None
    train_X = train_data.drop(columns=DROP_COLS, errors="ignore")
    train_Y = train_data[TARGETS]
    val_X   = val_data.drop(columns=DROP_COLS, errors="ignore")
    val_Y   = val_data[TARGETS]
    return train_X, train_Y, val_X, val_Y, train_weights


def train(run_name, hyperparams, train_X, train_Y, val_X, val_Y, train_weights=None):
    model = MultiOutputRegressor(LGBMRegressor(**hyperparams))

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(hyperparams)

        model.fit(train_X, train_Y, sample_weight=train_weights)

        val_preds = model.predict(val_X)

        for i, target in enumerate(TARGETS):
            y_true = val_Y[target]
            y_pred = val_preds[:, i]

            rmse      = root_mean_squared_error(y_true, y_pred)
            mae       = mean_absolute_error(y_true, y_pred)
            r2        = r2_score(y_true, y_pred)
            dir_acc   = float(((y_true * y_pred) >= 0).mean())

            mlflow.log_metric(f"val_rmse_{target}",     round(rmse,    4))
            mlflow.log_metric(f"val_mae_{target}",      round(mae,     4))
            mlflow.log_metric(f"val_r2_{target}",       round(r2,      4))
            mlflow.log_metric(f"val_dir_acc_{target}",  round(dir_acc, 4))

            logger.info("%s: RMSE=%.4f  MAE=%.4f  R²=%.4f  DirAcc=%.4f", target, rmse, mae, r2, dir_acc)

        mlflow.sklearn.log_model(model, name="model")

        # feature importance — averaged across the three sub-models
        feature_names = train_X.columns.tolist()
        importances = pd.DataFrame(
            {target: estimator.feature_importances_
            for target, estimator in zip(TARGETS, model.estimators_)},
            index=feature_names
        )
        importances["mean"] = importances.mean(axis=1)
        importances = importances.sort_values("mean", ascending=False)

        rmse_scores = {
            target: root_mean_squared_error(val_Y[target], val_preds[:, i])
            for i, target in enumerate(TARGETS)
        }
        score = sum(WEIGHTS[t] * rmse_scores[t] for t in TARGETS)
        mlflow.log_metric("weighted_rmse", round(score, 4))
        return score


DEFAULT_HYPERPARAMS = {
    "n_estimators":      531,
    "learning_rate":     0.016,
    "num_leaves":        20,
    "min_child_samples": 18,
    "subsample":         0.60,
    "colsample_bytree":  0.84,
    "random_state":      42,
    "n_jobs":            -1,
}


def run_training(hyperparams: dict = None):
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_X, train_Y, val_X, val_Y, train_weights = _load_data()
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    train("training", params, train_X, train_Y, val_X, val_Y, train_weights)


if __name__ == "__main__":
    run_training()