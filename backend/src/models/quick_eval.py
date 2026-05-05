"""
Trains on data/train.csv and evaluates on data/val.csv without MLflow.
Run from backend/:  python -m src.models.quick_eval
"""

import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.models.train import TARGETS, DROP_COLS, DEFAULT_HYPERPARAMS, WEIGHTS

# ── Load data ─────────────────────────────────────────────────────────────────

train = pd.read_csv("data/train.csv")
val   = pd.read_csv("data/val.csv")

train_weights = train["sample_weight"] if "sample_weight" in train.columns else None
train_X = train.drop(columns=DROP_COLS, errors="ignore")
train_Y = train[TARGETS]
val_X   = val.drop(columns=DROP_COLS, errors="ignore")
val_Y   = val[TARGETS]

print(f"Train: {len(train_X):,} rows   Val: {len(val_X):,} rows   Features: {train_X.shape[1]}")
print(f"Feature columns:\n  {list(train_X.columns)}\n")

# ── Train ─────────────────────────────────────────────────────────────────────

model = MultiOutputRegressor(LGBMRegressor(**DEFAULT_HYPERPARAMS, verbose=-1))
model.fit(train_X, train_Y, sample_weight=train_weights)

preds = model.predict(val_X)

# ── Metrics ───────────────────────────────────────────────────────────────────

print("=" * 55)
print(f"{'TARGET':<18} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'DirAcc':>8}")
print("-" * 55)
weighted = 0.0
for i, t in enumerate(TARGETS):
    y_true = val_Y[t].values
    y_pred = preds[:, i]
    rmse    = root_mean_squared_error(y_true, y_pred)
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    dir_acc = float(((y_true * y_pred) >= 0).mean())
    weighted += WEIGHTS[t] * rmse
    print(f"{t:<18} {rmse:>7.4f} {mae:>7.4f} {r2:>7.4f} {dir_acc:>8.3f}")
print("-" * 55)
print(f"{'Weighted RMSE':<18} {weighted:>7.4f}")

# ── Feature importance ────────────────────────────────────────────────────────

feat_names = train_X.columns.tolist()
importance_df = pd.DataFrame(
    {t: est.feature_importances_ for t, est in zip(TARGETS, model.estimators_)},
    index=feat_names,
)
importance_df["mean"] = importance_df.mean(axis=1)
top = importance_df.sort_values("mean", ascending=False).head(15)

print("\nTop 15 features by mean importance:")
print(f"  {'feature':<30} {'mean':>6}  {'reps':>6}  {'pct_1rm':>8}")
print("  " + "-" * 48)
for feat, row in top.iterrows():
    print(f"  {feat:<30} {row['mean']:>6.0f}  {row['delta_reps']:>6.0f}  {row['delta_pct_1rm']:>8.0f}")

# ── Sample predictions ────────────────────────────────────────────────────────

id_to_name = {v: k for k, v in json.load(open("data/exercise_map.json")).items()}

sample = val.sample(20, random_state=42).copy()
sample_X = sample.drop(columns=DROP_COLS, errors="ignore")
sample_preds = model.predict(sample_X)

print("\n" + "=" * 72)
print(f"{'Exercise':<28} {'Wk':>3}  {'Δreps act/pred':>15}  {'Δpct_1rm act/pred':>19}")
print("-" * 72)
for i, (_, row) in enumerate(sample.iterrows()):
    name = id_to_name.get(int(row["exercise_id"]), f"id={int(row['exercise_id'])}")[:27]
    dr_a, dp_a = row["delta_reps"], row["delta_pct_1rm"]
    dr_p, dp_p = sample_preds[i]
    print(f"{name:<28} {int(row['week']):>3}  "
          f"{dr_a:>+6.1f} / {dr_p:>+5.1f}    "
          f"{dp_a:>+8.4f} / {dp_p:>+7.4f}")
