"""
Cleans and feature-engineers the raw Boostcamp program dataset into train/val splits
ready for model training. Outputs data/train.csv and data/val.csv.

Output fields
─────────────
Context
  program_id          label-encoded program identifier (used for train/val split only)
  program_length      total weeks in the program
  time_per_workout    planned session duration (minutes)
  week                current week number
  day                 day within the week
  week_pct            week / program_length — relative position in the program arc
  number_of_exercises number of exercises in the session

Exercise
  exercise_id         label-encoded exercise name

Current prescription
  sets                number of sets this week
  reps                number of reps this week
  intensity           RPE (6-10)
  pct_1rm             estimated % of 1RM derived from reps + RPE via Tuchscherer table
  volume              sets × reps

Previous week (lag features)
  lag_sets            sets from the previous occurrence of this exercise
  lag_reps            reps from the previous occurrence
  lag_pct_1rm         pct_1rm from the previous occurrence
  lag_volume          volume from the previous occurrence
  weeks_gap           weeks elapsed since the previous occurrence (0 = same week, different day)

Fitness level (multi-hot)
  level_Advanced, level_Beginner, level_Intermediate, level_Novice

Goal (multi-hot)
  goal_Athletics, goal_Bodybuilding, goal_Bodyweight Fitness,
  goal_Muscle & Sculpting, goal_Olympic Weightlifting,
  goal_Powerbuilding, goal_Powerlifting

Equipment (one-hot)
  At Home, Dumbbell Only, Full Gym, Garage Gym

Targets (week-over-week deltas — what the model predicts)
  delta_sets          change in sets vs previous week
  delta_reps          change in reps vs previous week
  delta_pct_1rm       change in relative load vs previous week
"""

import ast
import pandas as pd
from consts import lookup_pct_1rm
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/programs_detailed.csv"
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


data = pd.read_csv(DATASET_PATH)

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

data["pct_1rm"] = data.apply(
    lambda row: lookup_pct_1rm(row["reps"], row["intensity"]), axis=1
)

data = multihot_encode_str_list(data, "level")
data = multihot_encode_str_list(data, "goal")

data = data.drop("equipment", axis=1).join(pd.get_dummies(data["equipment"]))

le_exercise = LabelEncoder()
data["exercise_id"] = le_exercise.fit_transform(data["exercise_name"])
data = data.drop(columns=["exercise_name"])

data["week_pct"] = (data["week"] / data["program_length"]).round(3)
data["volume"] = data["reps"] * data["sets"]

data = data.sort_values(by=["title", "exercise_id", "week", "day"])
for col in ["sets", "reps", "pct_1rm", "volume"]:
    data[f"lag_{col}"] = data.groupby(["title", "exercise_id"])[col].shift(1)

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

unique_programs = data["program_id"].unique()
train_programs, val_programs = train_test_split(unique_programs, test_size=0.2, random_state=42)

train = data[data["program_id"].isin(train_programs)]
val   = data[data["program_id"].isin(val_programs)]

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
data.to_csv(CLEAN_DATASET_PATH, index=False)

print(f"Train: {len(train):,} rows | {len(train_programs)} programs")
print(f"Val:   {len(val):,} rows | {len(val_programs)} programs")
