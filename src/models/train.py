import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

EXPERIMENT_NAME = "overload-predictor"

TARGETS   = ["delta_sets", "delta_reps", "delta_pct_1rm"]
DROP_COLS = ["program_id", "sets", "reps", "intensity", "pct_1rm", "volume"] + TARGETS
WEIGHTS   = {"delta_pct_1rm": 0.50, "delta_reps": 0.35, "delta_sets": 0.15}

mlflow.set_experiment(EXPERIMENT_NAME)

train_data = pd.read_csv("data/train.csv")
val_data   = pd.read_csv("data/val.csv")

train_X = train_data.drop(columns=DROP_COLS)
train_Y = train_data[TARGETS]
val_X   = val_data.drop(columns=DROP_COLS)
val_Y   = val_data[TARGETS]


def train(run_name, hyperparams):
    model = MultiOutputRegressor(LGBMRegressor(**hyperparams))

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(hyperparams)

        model.fit(train_X, train_Y)

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

            print(f"{target}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  DirAcc={dir_acc:.4f}")

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


if __name__ == "__main__":
    hyperparams = {
        "n_estimators":      531,
        "learning_rate":     0.016,
        "num_leaves":        20,
        "min_child_samples": 18,
        "subsample":         0.60,
        "colsample_bytree":  0.84,
        "random_state":      42,
        "n_jobs":            -1,
    }
    train("training", hyperparams)