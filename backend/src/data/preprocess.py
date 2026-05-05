"""
Cleans and feature-engineers the Boostcamp program dataset into train/val splits
ready for model training. Outputs data/train.csv and data/val.csv.

Train/val split is by program (not by row) to prevent data leakage.
User logs from PostgreSQL are appended to the train set with higher sample weights.

Output fields
─────────────
Context
  program_id          label-encoded program identifier (train/val split key, dropped during training)
  program_length      total weeks in the program
  time_per_workout    planned session duration (minutes)
  week                current week number
  day                 day within the week
  week_pct            week / program_length — relative position in the program arc
  week_pct2           week_pct² — captures non-linear periodization curves
  number_of_exercises number of exercises in the session

Exercise
  exercise_id         label-encoded exercise name

Periodization intent
  is_deload           1 if this week's intensity is >12% below the 3-session rolling peak
  overload_linear     program classified as linear (steadily increasing intensity)
  overload_undulating program classified as undulating (alternating heavy/light weeks)
  overload_block      program classified as block (phases + periodic deloads)

Current prescription (dropped during training — used to compute targets and lag features)
  sets                number of sets this week
  reps                number of reps this week
  intensity           RPE (6-10)
  pct_1rm             estimated % of 1RM via Tuchscherer table
  volume              sets × reps × pct_1rm (intensity-weighted volume)

Lag features — last session
  lag_sets            sets from the previous occurrence of this exercise
  lag_reps            reps from the previous occurrence
  lag_pct_1rm         pct_1rm from the previous occurrence
  lag_volume          volume from the previous occurrence
  weeks_gap           weeks elapsed since the previous occurrence

Trend features — direction and acceleration (velocity entering this week)
  lag_delta_reps      lag_reps minus the session before that (reps velocity)
  lag2_delta_reps     two sessions back minus three sessions back (reps acceleration)
  lag_delta_pct_1rm   lag_pct_1rm minus the session before that (intensity velocity)
  lag2_delta_pct_1rm  two sessions back minus three sessions back (intensity acceleration)

Fitness level (multi-hot)
  level_Advanced, level_Beginner, level_Intermediate, level_Novice

Goal (multi-hot)
  goal_At-Home & Calisthenics, goal_Athletics, goal_Bodybuilding,
  goal_Bodyweight Fitness, goal_Muscle & Sculpting, goal_Olympic Weightlifting,
  goal_Powerbuilding, goal_Powerlifting

Equipment (one-hot)
  At Home, Dumbbell Only, Full Gym, Garage Gym

Targets (week-over-week deltas — what the model predicts)
  delta_reps          change in reps vs previous session
  delta_pct_1rm       change in relative load vs previous session
"""

import ast
import logging
import pandas as pd
from src.data.consts import lookup_pct_1rm, GOAL_COL_MAP, EQUIPMENT_COL_MAP
from sqlalchemy import create_engine
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", None)
DATASET_PATH = "data/programs_detailed_canonical.csv"
CLEAN_DATASET_PATH = "data/programs_detailed_cleaned.csv"


def multihot_encode_str_list(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    parsed = df[col].apply(ast.literal_eval)
    encoded = pd.DataFrame(
        mlb.fit_transform(parsed),
        columns=[f"{col}_{c}" for c in mlb.classes_],
        index=df.index
    )
    return df.drop(columns=[col]).join(encoded)

def load_user_logs() -> pd.DataFrame:
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — skipping user log ingestion")
        return None
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM workout_logs", engine)
        logger.info("Loaded %d user log rows from Postgres", len(df))
        df["sample_weight"] = 50.0
        return df
    except Exception as e:
        logger.warning("Could not load user logs from Postgres: %s", e)
        return None


def process_user_logs(raw_df: pd.DataFrame, exercise_map: dict) -> pd.DataFrame:
    """Convert raw workout_logs rows into the same column schema as Boostcamp train data."""
    df = raw_df.copy()

    # Drop rows whose exercise isn't in the model's known vocabulary
    before = len(df)
    df = df[df["exercise"].isin(exercise_map)]
    if len(df) < before:
        logger.warning("Dropped %d user log rows with unknown exercises", before - len(df))

    # Tuchscherer table covers RPE 6-10, reps 1-12
    df = df[
        df["rpe"].between(6, 10) & df["reps"].between(1, 12) &
        df["lag_rpe"].between(6, 10) & df["lag_reps"].between(1, 12)
    ]

    if df.empty:
        return df

    df["exercise_id"]       = df["exercise"].map(exercise_map)
    df["pct_1rm"]           = df.apply(lambda r: lookup_pct_1rm(r["reps"],     r["rpe"]),     axis=1)
    df["lag_pct_1rm"]       = df.apply(lambda r: lookup_pct_1rm(r["lag_reps"], r["lag_rpe"]), axis=1)
    df["lag_volume"]        = df["lag_sets"] * df["lag_reps"] * df["lag_pct_1rm"]
    df["volume"]            = df["sets"] * df["reps"] * df["pct_1rm"]
    df["week_pct"]          = (df["week"] / df["program_length"]).round(3)
    df["week_pct2"]         = df["week_pct"] * df["week_pct"]
    df["delta_sets"]        = df["sets"] - df["lag_sets"]
    df["delta_reps"]        = df["reps"] - df["lag_reps"]
    df["delta_pct_1rm"]     = df["pct_1rm"] - df["lag_pct_1rm"]
    df["lag_delta_pct_1rm"] = 0.0
    df["intensity"]         = df["rpe"]
    df["program_id"]        = -1

    df = df.rename(columns={**GOAL_COL_MAP, **EQUIPMENT_COL_MAP})

    drop_cols = ["id", "user_id", "logged_at", "one_rm", "exercise", "rpe", "reps", "sets"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.info("Processed %d user log rows into training format", len(df))
    return df
      

def tag_deload_weeks(data: pd.DataFrame) -> pd.DataFrame:
    """Flag weeks where the current prescribed intensity is a significant drop from recent peak.

    Rolling max covers the 3 previous sessions for this exercise in this program.
    The 88% threshold (~12% drop) captures intentional deload reductions while
    ignoring normal session-to-session noise. Early rows with <2 prior sessions get 0.
    """
    rolling_peak = data.groupby(["title", "exercise_id"])["pct_1rm"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).max()
    )
    data["is_deload"] = (
        (data["pct_1rm"] < rolling_peak * 0.88) &
        rolling_peak.notna() &
        (rolling_peak > 0.65)
    ).astype(int)
    return data


def classify_programs(data: pd.DataFrame) -> pd.DataFrame:
    """Classify each program's periodization style from its intensity pattern.

    linear:     intensity steadily increases (>55% of week-over-week changes positive)
    block:      positive trend punctuated by large drops - 10%+ weeks drop >7% of 1RM
    undulating: roughly balanced mix of up and down weeks
    """
    def _classify_from_diffs(diffs: pd.Series) -> str:
        if len(diffs) < 3:
            return "linear"
        large_neg = (diffs < -0.07).mean()
        pos = (diffs > 0.02).mean()
        neg = (diffs < -0.02).mean()
        if large_neg > 0.10:
            return "block"
        if abs(pos - neg) < 0.20:
            return "undulating"
        return "linear"

    types = {}
    for title, group in data.groupby("title"):
        all_diffs = pd.concat([
            ex.sort_values("week")["pct_1rm"].diff().dropna()
            for _, ex in group.groupby("exercise_id")
        ])
        types[title] = _classify_from_diffs(all_diffs)

    data["overload_type"] = data["title"].map(types)
    dummies = pd.get_dummies(data["overload_type"], prefix="overload")
    data = data.drop(columns=["overload_type"]).join(dummies)

    for col in ["overload_linear", "overload_undulating", "overload_block"]:
        if col not in data.columns:
            data[col] = 0

    return data


def preprocess():
    logger.info("Loading dataset from %s", DATASET_PATH)
    data = pd.read_csv(DATASET_PATH)
    logger.info("Loaded %d raw rows", len(data))

    data = data.drop(columns=["description", "created", "last_edit"])
    data = data.drop_duplicates()
    data = data.dropna(subset=["program_length", "equipment"])

    # rep-based only; cap at 12 where RPE → pct_1rm mapping is reliable
    data = data[data["reps"] > 0]
    data = data[data["reps"] <= 12].copy()

    # Tuchscherer table covers RPE 6-10
    data = data[data["intensity"].between(6, 10)].copy()

    # sets > 10 are data entry errors
    data = data[data["sets"] <= 10].copy()
    logger.info("%d rows after cleaning", len(data))

    data["pct_1rm"] = data.apply(
        lambda row: lookup_pct_1rm(row["reps"], row["intensity"]), axis=1
    )

    data = multihot_encode_str_list(data, "level")
    data = multihot_encode_str_list(data, "goal")

    data = data.drop("equipment", axis=1).join(pd.get_dummies(data["equipment"]))

    exercise_map = json.load(open("data/exercise_map.json"))

    data["exercise_id"] = data["exercise_name"].map(exercise_map)
    data = data.drop(columns=["exercise_name"])

    data["week_pct"] = (data["week"] / data["program_length"]).round(3)
    data["week_pct2"] = data["week_pct"] * data["week_pct"]
    data["volume"] = data["reps"] * data["sets"] * data["pct_1rm"]

    data = data.sort_values(by=["title", "exercise_id", "week", "day"])
    data = tag_deload_weeks(data)
    data = classify_programs(data)

    for col in ["sets", "reps", "pct_1rm", "volume"]:
        data[f"lag_{col}"] = data.groupby(["title", "exercise_id"])[col].shift(1)

    # Two-step trend for both intensity and reps: direction + acceleration.
    # Sessions with fewer than 2/3 prior rows fall back to 0 (no trend yet).
    lag2_pct_1rm = data.groupby(["title", "exercise_id"])["pct_1rm"].shift(2)
    lag3_pct_1rm = data.groupby(["title", "exercise_id"])["pct_1rm"].shift(3)
    data["lag_delta_pct_1rm"]  = (data["lag_pct_1rm"] - lag2_pct_1rm).fillna(0)
    data["lag2_delta_pct_1rm"] = (lag2_pct_1rm - lag3_pct_1rm).fillna(0)

    lag2_reps = data.groupby(["title", "exercise_id"])["reps"].shift(2)
    lag3_reps = data.groupby(["title", "exercise_id"])["reps"].shift(3)
    data["lag_delta_reps"]  = (data["lag_reps"] - lag2_reps).fillna(0)
    data["lag2_delta_reps"] = (lag2_reps - lag3_reps).fillna(0)

    data["lag_week"] = data.groupby(["title", "exercise_id"])["week"].shift(1)
    data["weeks_gap"] = data["week"] - data["lag_week"]
    data = data.drop(columns=["lag_week"])

    data = data.dropna(subset=["lag_sets", "lag_reps", "lag_pct_1rm", "lag_volume"])
    data = data[data["weeks_gap"] <= 4]

    le_program = LabelEncoder()
    data["program_id"] = le_program.fit_transform(data["title"])
    data = data.drop(columns=["title"])

    data["delta_sets"] = data["sets"] - data["lag_sets"]
    data["delta_reps"] = data["reps"] - data["lag_reps"]
    data["delta_pct_1rm"] = data["pct_1rm"] - data["lag_pct_1rm"]
    logger.info("%d rows after feature engineering", len(data))

    unique_programs = data["program_id"].unique()
    train_programs, val_programs = train_test_split(unique_programs, test_size=0.2, random_state=42)

    train = data[data["program_id"].isin(train_programs)].copy()
    val   = data[data["program_id"].isin(val_programs)].copy()

    # Boostcamp rows get weight 1.0; user log rows get 50.0 (set in load_user_logs)
    train["sample_weight"] = 1.0
    val["sample_weight"]   = 1.0

    raw_logs = load_user_logs()
    if raw_logs is not None and not raw_logs.empty:
        exercise_map_dict = json.load(open("data/exercise_map.json"))
        user_logs = process_user_logs(raw_logs, exercise_map_dict)
        if not user_logs.empty:
            user_logs = user_logs.reindex(columns=train.columns, fill_value=0)
            train = pd.concat([train, user_logs], axis=0, ignore_index=True)
            logger.info("Appended %d user log rows to train set", len(user_logs))

    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    data.to_csv(CLEAN_DATASET_PATH, index=False)

    logger.info("Train: %d rows | %d programs", len(train), len(train_programs))
    logger.info("Val:   %d rows | %d programs", len(val), len(val_programs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    preprocess()
