"""
Champion vs challenger evaluation.

Loads metrics from the latest training run (challenger) and the current Production
model (champion), computes a weighted composite RMSE score, and promotes the
challenger if it improves on the champion by more than IMPROVEMENT_THRESHOLD.

Weights reflect business priority:
  delta_pct_1rm — most important, maps directly to kg to lift
  delta_reps    — drives training volume
  delta_sets    — least variable, rarely changes week-to-week
"""

import mlflow
import mlflow.exceptions
from mlflow import MlflowClient

EXPERIMENT_NAME = "overload-predictor"
MODEL_NAME      = "overload-predictor"
ALIAS_PROD      = "Production"

TARGETS = ["delta_sets", "delta_reps", "delta_pct_1rm"]

WEIGHTS = {
    "delta_pct_1rm": 0.50,
    "delta_reps":    0.35,
    "delta_sets":    0.15,
}

# challenger must beat champion by at least this much to be promoted (guards against noise)
IMPROVEMENT_THRESHOLD = 0.02


def weighted_rmse(metrics: dict[str, float]) -> float:
    return sum(WEIGHTS[t] * metrics[f"val_rmse_{t}"] for t in TARGETS)


METRIC_PREFIXES = ("val_rmse_", "val_mae_", "val_r2_", "val_dir_acc_")

def get_run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    run = client.get_run(run_id)
    return {
        k: v for k, v in run.data.metrics.items()
        if any(k.startswith(p) for p in METRIC_PREFIXES)
    }


client = MlflowClient()

#load the challenger
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment.experiment_id,
    filter_string="tags.mlflow.runName = 'training'",
    order_by=["start_time DESC"],
    max_results=1
)
if not runs:
    raise RuntimeError("No training runs found. Run train.py first.")

challenger_run_id = runs[0].info.run_id
challenger_metrics = get_run_metrics(client, challenger_run_id)
challenger_score   = weighted_rmse(challenger_metrics)

print("Challenger metrics:")
for t in TARGETS:
    print(f"  val_rmse_{t}: {challenger_metrics[f'val_rmse_{t}']:.4f}")
print(f"  weighted RMSE: {challenger_score:.4f}")

#load champion (current prod)
try:
    champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_PROD)
    champion_metrics = get_run_metrics(client, champion_version.run_id)
    champion_score   = weighted_rmse(champion_metrics)

    print("\nChampion metrics:")
    for t in TARGETS:
        print(f"  val_rmse_{t}: {champion_metrics[f'val_rmse_{t}']:.4f}")
    print(f"  weighted RMSE: {champion_score:.4f}")

except mlflow.exceptions.MlflowException:
    champion_version = None
    champion_score   = None
    print("\nNo Production model found — first run.")

#compare models
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metrics({f"challenger_{k}": v for k, v in challenger_metrics.items()})
    mlflow.log_metric("challenger_weighted_rmse", challenger_score)

    if champion_score is not None:
        mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_metric("champion_weighted_rmse", champion_score)

        improvement = (champion_score - challenger_score) / champion_score
        mlflow.log_metric("relative_improvement", improvement)

        print(f"\nRelative improvement: {improvement*100:.2f}%")

        per_target = {
            t: champion_metrics[f"val_rmse_{t}"] - challenger_metrics[f"val_rmse_{t}"]
            for t in TARGETS
        }
        targets_improved = sum(1 for v in per_target.values() if v > 0)
        print(f"Targets improved: {targets_improved}/3")
        for t, delta in per_target.items():
            direction = "better" if delta > 0 else "worse" if delta < 0 else "= equal"
            print(f"  {t}: {direction} ({delta:+.4f})")

        promote = (improvement >= IMPROVEMENT_THRESHOLD and targets_improved >= 2) \
                    or targets_improved == 3
        mlflow.log_param("promoted", promote)
        mlflow.log_param("reason",
            f"improvement={improvement*100:.2f}%, targets_improved={targets_improved}/3"
        )
    else:
        promote = True
        mlflow.log_param("promoted", True)
        mlflow.log_param("reason", "no existing Production model")

    if promote:
        result = mlflow.register_model(f"runs:/{challenger_run_id}/model", MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, ALIAS_PROD, result.version)
        print(f"\nPromoted challenger to Production (version {result.version})")
    else:
        print("\nChallenger not promoted — champion retained.")
