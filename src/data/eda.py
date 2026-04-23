import pandas as pd

DATASET_PATH = "data/programs_detailed.csv"

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("BASIC INFO")
print("=" * 60)
print(f"Rows (sample): {len(df):,}  |  Columns: {df.shape[1]}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum():,}")


# ── Reps: two regimes ─────────────────────────────────────────────────────────
# Negative reps encode duration in seconds (e.g. -180 = 3-min hold).
# Positive reps are standard rep counts.

print("\n" + "=" * 60)
print("REPS ANALYSIS")
print("=" * 60)

time_based = df[df["reps"] < 0].copy()
rep_based  = df[df["reps"] >= 0].copy()
time_based["duration_s"] = time_based["reps"].abs()

print(f"Time-based rows (reps < 0): {len(time_based):,}  "
      f"({len(time_based)/len(df)*100:.1f}%)")
print(f"Rep-based rows  (reps >= 0): {len(rep_based):,}  "
      f"({len(rep_based)/len(df)*100:.1f}%)")

print("\nRep-based reps distribution:")
print(rep_based["reps"].describe().round(2))

print("\nTime-based duration (seconds) distribution:")
print(time_based["duration_s"].describe().round(2))
print("Common durations (s):", time_based["duration_s"].value_counts().head(8).to_dict())


# ── Intensity (RPE 0-10) ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("INTENSITY (RPE)")
print("=" * 60)
print(df["intensity"].describe().round(2))
print("\nValue counts (top 10):", df["intensity"].value_counts().head(10).to_dict())
print(f"Rows with intensity == 0: {(df['intensity'] == 0).sum():,}  "
      f"(likely unset / bodyweight)")


# ── Sets ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SETS")
print("=" * 60)
print(df["sets"].describe().round(2))
print("\nValue counts:", df["sets"].value_counts().sort_index().to_dict())


# ── Categorical columns ───────────────────────────────────────────────────────
# level and goal are stored as Python list literals, e.g. "['Beginner', 'Advanced']"

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS")
print("=" * 60)

for col in ["level", "goal", "equipment"]:
    if col in ["level", "goal"]:
        # explode list-encoded strings
        exploded = (
            df[col]
            .dropna()
            .apply(lambda x: [v.strip().strip("'") for v in x.strip("[]").split(",")])
            .explode()
            .str.strip()
        )
        print(f"\n{col} (unique values):\n{exploded.value_counts().to_string()}")
    else:
        print(f"\n{col}:\n{df[col].value_counts().to_string()}")


# ── Program structure ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PROGRAM STRUCTURE")
print("=" * 60)
print(f"Unique programs:  {df['title'].nunique():,}")
print(f"Unique exercises: {df['exercise_name'].nunique():,}")
print(f"\nprogram_length distribution:\n{df['program_length'].value_counts().sort_index().to_string()}")
print(f"\nWeek range: {df['week'].min():.0f} – {df['week'].max():.0f}")
print(f"Day  range: {df['day'].min():.0f} – {df['day'].max():.0f}")


# ── Progressive overload signal ───────────────────────────────────────────────
# Core question: do sets/reps/intensity actually increase across weeks?

print("\n" + "=" * 60)
print("PROGRESSIVE OVERLOAD SIGNAL (rep-based exercises only)")
print("=" * 60)

weekly = (
    rep_based.groupby("week")[["sets", "reps", "intensity"]]
    .mean()
    .round(3)
)
weekly["volume"] = weekly["sets"] * weekly["reps"]
print(weekly.to_string())

# Per-exercise progression for a well-known lift
sample_exercise = "Barbell Back Squat"
ex_df = rep_based[rep_based["exercise_name"] == sample_exercise]
if not ex_df.empty:
    print(f"\nWeekly averages for '{sample_exercise}':")
    print(
        ex_df.groupby("week")[["sets", "reps", "intensity"]]
        .mean()
        .round(2)
        .to_string()
    )
else:
    top_exercise = rep_based["exercise_name"].value_counts().idxmax()
    print(f"\n'{sample_exercise}' not found — using '{top_exercise}' instead:")
    print(
        rep_based[rep_based["exercise_name"] == top_exercise]
        .groupby("week")[["sets", "reps", "intensity"]]
        .mean()
        .round(2)
        .to_string()
    )


# ── Volume metric & target preview ───────────────────────────────────────────
# Volume = sets × reps — the most common progressive overload proxy.
# The ML model will predict next-week's sets, reps, intensity.

print("\n" + "=" * 60)
print("TARGET VARIABLE PREVIEW (week-over-week deltas)")
print("=" * 60)

rep_based = rep_based.copy()
rep_based["volume"] = rep_based["sets"] * rep_based["reps"]

# Aggregate to (program, exercise, week) level
agg = (
    rep_based.groupby(["title", "exercise_name", "week"])[["sets", "reps", "intensity", "volume"]]
    .mean()
    .reset_index()
    .sort_values(["title", "exercise_name", "week"])
)

# Week-over-week deltas
for col in ["sets", "reps", "intensity", "volume"]:
    agg[f"delta_{col}"] = agg.groupby(["title", "exercise_name"])[col].diff()

deltas = agg[[c for c in agg.columns if c.startswith("delta_")]].dropna()
print("\nWeek-over-week delta statistics:")
print(deltas.describe().round(3).to_string())

positive_overload = (deltas["delta_volume"] > 0).mean()
print(f"\nWeeks where volume increased: {positive_overload*100:.1f}%")
print(f"Weeks where volume decreased: {(deltas['delta_volume'] < 0).mean()*100:.1f}%")
print(f"Weeks where volume unchanged: {(deltas['delta_volume'] == 0).mean()*100:.1f}%")


# ── RPE → % 1RM lookup (Tuchscherer / Helms) ─────────────────────────────────
# Rows = reps (1-12), Cols = RPE (6-10).
# Beyond 12 reps or below RPE 6, the relationship becomes unreliable.

print("\n" + "=" * 60)
print("RPE → % 1RM LOOKUP VALIDATION")
print("=" * 60)

from consts import  lookup_pct_1rm

# Apply to rep-based rows within the reliable reps/RPE range
eligible = rep_based[
    (rep_based["reps"].between(1, 12)) &
    (rep_based["intensity"].between(6, 10))
].copy()

eligible["pct_1rm"] = eligible.apply(
    lambda row: lookup_pct_1rm(row["reps"], row["intensity"]), axis=1
)

print(f"\nRows eligible for pct_1rm (reps 1-12, RPE 6-10): {len(eligible):,}  "
      f"({len(eligible)/len(rep_based)*100:.1f}% of rep-based)")
print(f"Rows outside reliable range (excluded): "
      f"{len(rep_based) - len(eligible):,}")

print("\npct_1rm distribution:")
print(eligible["pct_1rm"].describe().round(3))
print("\nValue counts (top 10):",
      (eligible["pct_1rm"] * 100).astype(int).value_counts().head(10).to_dict())

# Week-over-week delta_pct_1rm — the core progressive overload signal
agg_pct = (
    eligible.groupby(["title", "exercise_name", "week"])["pct_1rm"]
    .mean()
    .reset_index()
    .sort_values(["title", "exercise_name", "week"])
)
agg_pct["delta_pct_1rm"] = agg_pct.groupby(["title", "exercise_name"])["pct_1rm"].diff()
deltas_pct = agg_pct["delta_pct_1rm"].dropna()

print("\nWeek-over-week delta_pct_1rm statistics:")
print(deltas_pct.describe().round(4))
print(f"\nWeeks where relative load increased: {(deltas_pct > 0).mean()*100:.1f}%")
print(f"Weeks where relative load decreased: {(deltas_pct < 0).mean()*100:.1f}%")
print(f"Weeks where relative load unchanged: {(deltas_pct == 0).mean()*100:.1f}%")

# Mean pct_1rm by week — should show a general trend
print("\nMean pct_1rm per week (rep-based, eligible rows):")
print(eligible.groupby("week")["pct_1rm"].mean().round(3).to_string())
