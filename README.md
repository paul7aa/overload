# Progressive Overload Predictor

An end-to-end MLOps pipeline that predicts how a workout program should progress week-over-week — giving you the actual weight to lift based on your one rep max.

---

## The problem

Progressive overload (gradually increasing training stress over time) is the fundamental driver of strength and muscle gains. Knowing *how much* to increase each week is non-trivial: too little and you stagnate, too much and you risk injury or burnout.

Existing workout apps either follow rigid, pre-written programs or leave the decision entirely to the user. This project learns the progression patterns from hundreds of real, coach-designed programs and applies them to your own lifts.

---

## How it works

### The dataset

The model is trained on ~2.5 million rows of structured workout program data (Boostcamp, via Kaggle), covering 892 programs and 1,674 exercises. Each row is one exercise prescription: how many sets, reps, and at what intensity to perform it in a given week.

The dataset does not contain absolute weights — only relative intensity targets expressed as **RPE**.

### RPE — Rate of Perceived Exertion

RPE is a 0–10 scale that describes how hard a set was in terms of reps left in reserve:

| RPE | Meaning |
|-----|---------|
| 10 | Maximum — could not do another rep |
| 9 | 1 rep left in reserve |
| 8 | 2 reps left in reserve |
| 7 | 3 reps left in reserve |
| 6 | 4+ reps left in reserve |

Programs prescribe RPE rather than absolute weight so the same program scales to any athlete — a 60 kg lifter and a 140 kg lifter both train at "RPE 8", just with different plates on the bar.

### pct_1rm — Percentage of One Rep Max

Your **1RM** (one rep max) is the maximum weight you can lift for a single rep. Your working weight expressed as a fraction of that max is `pct_1rm`.

Example: bench press 1RM = 100 kg, working weight = 75 kg → `pct_1rm = 0.75`.

**RPE and pct_1rm are directly related.** If you do 5 reps and stop genuinely at RPE 8, sports science (Tuchscherer/Helms) tells us you were lifting approximately 76% of your 1RM — regardless of your absolute strength level:

| Reps | RPE 7 | RPE 8 | RPE 9 | RPE 10 |
|------|-------|-------|-------|--------|
| 1    | 83%   | 88%   | 94%   | 100%   |
| 3    | 77%   | 82%   | 88%   | 94%    |
| 5    | 71%   | 76%   | 82%   | 88%    |
| 8    | 63%   | 68%   | 74%   | 79%    |
| 10   | 59%   | 65%   | 71%   | 77%    |
| 12   | 53%   | 58%   | 63%   | 68%    |

This means we can derive `pct_1rm` from the data we already have (`reps` + `intensity`), with no absolute weights required.

### What the model learns

The model is trained entirely in relative terms:

```
Input:  (exercise, week, sets, reps, pct_1rm, level, goal, equipment, program_length, ...)
Output: (Δsets, Δreps, Δpct_1rm)  ←  week-over-week change
```

At prediction time, the user supplies their 1RM for an exercise. The predicted `pct_1rm` is multiplied by that value to produce an absolute weight:

```
next_weight = (current_pct_1rm + Δpct_1rm) × user_1rm
```

Example: model predicts +3% relative load next week, user's bench 1RM is 100 kg → **lift 79 kg next week**.

---

## Pipeline architecture

New data dropped into `data/incoming/` triggers a fully automated retraining pipeline:

```
data/incoming/  ← drop new CSV files here
      │
      ▼  Watchdog detects file
Prefect flow triggered
      │
      ├─ 1. Validate        Great Expectations checks schema + value ranges
      ├─ 2. Merge + version  DVC appends new data, commits new version
      ├─ 3. Preprocess       Feature engineering, train/val split
      ├─ 4. Train            MLflow experiment — LightGBM multi-output regression
      ├─ 5. Evaluate         Challenger vs current Production champion
      └─ 6. Promote          Best model tagged Production in MLflow Registry
                             FastAPI hot-swaps model with zero downtime
```

### Services

| Service | Tool | Purpose |
|---------|------|---------|
| Experiment tracking | MLflow | Log params, metrics, model artifacts |
| Data versioning | DVC | Version raw and processed datasets |
| Orchestration | Prefect | Automate the retraining flow |
| Data validation | Great Expectations | Catch bad data before training |
| API | FastAPI + Pydantic | Serve predictions, `/reload` endpoint |
| Containerisation | Docker Compose | Isolate all services |
| Automation trigger | Watchdog | Fire pipeline on new data arrival |

---

## API usage

```
POST /predict
{
  "exercise": "Bench Press (Barbell)",
  "current_week": 4,
  "sets": 3,
  "reps": 5,
  "rpe": 8,
  "one_rep_max_kg": 100,
  "program_level": "Intermediate",
  "goal": "Powerbuilding"
}

→ {
  "next_sets": 3,
  "next_reps": 5,
  "next_rpe": 8.5,
  "next_pct_1rm": 0.79,
  "next_weight_kg": 79.0
}
```

---

## Project structure

```
overload/
├── data/
│   ├── raw/                  # DVC-tracked original dataset
│   ├── processed/            # DVC-tracked feature-engineered parquet
│   └── incoming/             # Drop zone for new training data
├── src/
│   ├── data/
│   │   ├── eda.py            # Exploratory analysis
│   │   ├── validate.py       # Great Expectations suite
│   │   ├── preprocess.py     # Cleaning + feature engineering
│   │   └── merge.py          # Merge incoming → raw
│   ├── models/
│   │   ├── train.py          # MLflow training run
│   │   └── evaluate.py       # Champion vs challenger comparison
│   ├── pipeline/
│   │   ├── flows.py          # Prefect flow definition
│   │   └── watcher.py        # Watchdog → Prefect trigger
│   └── api/
│       ├── app.py            # FastAPI app
│       └── schemas.py        # Pydantic I/O models
├── docker-compose.yml
├── dvc.yaml
├── PLAN.md
└── README.md
```

---

## Quickstart

```bash
# Start all services
docker compose up

# Drop new training data to trigger retraining
cp my_new_data.csv data/incoming/

# MLflow UI
open http://localhost:5000

# Prefect UI
open http://localhost:4200

# API docs
open http://localhost:8000/docs
```
