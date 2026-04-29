# Progressive Overload Predictor

A mobile-first MLOps app that tells you exactly what weight to lift next week — and gets smarter the more you use it.

---

## The problem

Progressive overload (gradually increasing training stress over time) is the fundamental driver of strength and muscle gains. Knowing *how much* to increase each week is non-trivial: too little and you stagnate, too much and you risk injury or burnout.

Existing workout apps either follow rigid, pre-written programs or leave the decision entirely to the user. This project learns progression patterns from hundreds of real coach-designed programs, then continuously refines that knowledge from your own gym sessions.

---

## How it works

### Phase 1 — Cold start (Boostcamp dataset)

The model is bootstrapped on ~400k rows of structured workout program data (Boostcamp, via Kaggle), covering 2,500+ programs and 1,674 exercises. Each row is one exercise prescription: how many sets, reps, and at what intensity to perform it in a given week.

**Dataset:** [600K+ Fitness Exercise & Workout Program Dataset](https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset) — download `programs_detailed_boostcamp_kaggle.csv` and save it as `data/programs_detailed.csv`.

### Phase 2 — Continuous learning (user logs)

Every time a user logs a completed session via the mobile app, that data is stored in PostgreSQL. When enough new sessions have accumulated, a scheduled Prefect flow automatically retrains the model on the combined dataset — coach-designed programs plus real user progressions. The model gets promoted to Production if it beats the current champion.

---

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

## Architecture

### Retraining pipeline

```
User logs session via mobile app
        │
        ▼
PostgreSQL (session store)
        │
        ▼  500+ new sessions accumulated
Prefect scheduled flow triggered
        │
        ├─ 1. Preprocess    Merge Boostcamp + user logs → feature engineering
        ├─ 2. Train         MLflow experiment — LightGBM multi-output regression
        ├─ 3. Evaluate      Challenger vs current Production champion
        └─ 4. Promote       Best model tagged Production in MLflow Registry
                            FastAPI hot-swaps model via /reload
```

### Services

| Service | Tool | Purpose |
|---------|------|---------|
| Experiment tracking | MLflow | Log params, metrics, model artifacts |
| Orchestration | Prefect | Retraining flow + scheduling |
| Session store | PostgreSQL | Persist user gym logs |
| API | FastAPI + Pydantic | Predictions, session logging, model reload |
| Containerisation | Docker Compose | Run full stack locally |
| Mobile app | React Native + Expo | iOS/Android client |

### Hosting

The full stack runs locally on a home machine. The mobile app connects over local WiFi (`http://<machine-ip>:8000`).

---

## API

```
POST /predict       Get next week's prescription
POST /log           Record a completed session (feeds retraining)
POST /reload        Hot-swap to latest Production model
GET  /health        Model status + version
GET  /docs          Interactive API docs (Swagger UI)
```

### Example

```
POST /predict
{
  "exercise": "Bench Press (Barbell)",
  "one_rm": 100,
  "lag_sets": 3,
  "lag_reps": 5,
  "lag_rpe": 8,
  "week": 4,
  "program_length": 12,
  ...
}

→ {
  "next_sets": 3,
  "next_reps": 5,
  "delta_pct_1rm": 0.03,
  "next_weight_kg": 79.0
}
```

---

## Project structure

```
overload/                         # monorepo root
├── backend/
│   ├── data/
│   │   ├── programs_detailed.csv     # Boostcamp cold-start dataset (not committed)
│   │   ├── train.csv                 # Generated by preprocess.py
│   │   ├── val.csv                   # Generated by preprocess.py
│   │   └── exercise_map.json         # Exercise name → encoded ID
│   ├── src/
│   │   ├── data/
│   │   │   ├── consts.py             # Tuchscherer RPE → pct_1rm table
│   │   │   ├── eda.py                # Exploratory analysis
│   │   │   └── preprocess.py        # Cleaning + feature engineering
│   │   ├── models/
│   │   │   ├── train.py              # MLflow training run
│   │   │   ├── evaluate.py           # Champion vs challenger comparison
│   │   │   ├── tune.py               # Optuna hyperparameter search
│   │   │   └── utils.py              # Model loading helpers
│   │   ├── pipeline/
│   │   │   └── flow.py               # Prefect flow: preprocess → train → evaluate
│   │   └── api/
│   │       ├── app.py                # FastAPI app
│   │       └── schemas.py            # Pydantic I/O models
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                      # React Native app (Expo)
│   ├── src/
│   │   ├── screens/
│   │   │   ├── HomeScreen            # Active program + start workout
│   │   │   ├── ProgramsScreen        # List of saved programs
│   │   │   ├── AddProgramScreen      # Create program (goal, equipment, exercises)
│   │   │   ├── ActiveWorkoutScreen   # Current exercise + set tracker
│   │   │   └── WorkoutCompleteScreen # Session summary
│   │   ├── components/
│   │   └── api/                  # API client (points to backend over WiFi)
│   ├── app.json
│   └── package.json
└── README.md
```

---

## Mobile app

### Design principles

- **Minimal input mid-workout** — you're between sets, hands may be chalked. Taps, not typing.
- **Log what happened, not what you planned** — the app tells you what to do; you confirm what you actually did.
- **One screen at a time** — no navigation during a workout.
- **No manual 1RM entry** — the app back-calculates your 1RM from weight + reps + RPE automatically.

---

### 1RM — auto-calculated, never entered

The Tuchscherer table works in both directions:

```
1RM = working_weight / pct_1rm(reps, rpe)
```

Example: lifted 75 kg × 5 reps @ RPE 8 → `pct_1rm = 0.76` → estimated 1RM = **98.7 kg**

- **First session on an exercise** — user enters the weight they lifted, reps, and RPE. App back-calculates 1RM and stores it on-device.
- **Every subsequent session** — the API returns a predicted weight. User logs actual reps + RPE. 1RM is recalculated and updated automatically.
- **Self-correcting** — as the user gets stronger, 1RM updates after every session with no manual input.

---

### Screens

#### Programs
A list of your saved programs. Each program has:
- Name
- Goal (Powerlifting, Bodybuilding, etc.)
- Equipment (Full Gym, Garage Gym, etc.)
- Level
- Length in weeks
- List of exercises

#### Add Program
Step-by-step form:
1. Name + Level
2. Goal (single select)
3. Equipment (single select)
4. Program length (weeks) + time per workout
5. Add exercises — search by name

---

### Workout flow

This matches how a session actually happens in the gym:

```
Home
 └── Start Workout
       │
       ▼
 ┌─────────────────────────────────────┐
 │  Active Workout                     │
 │                                     │
 │  Exercise: Bench Press              │
 │  Target: 3×5 @ 79 kg                │  ← from API prediction
 │                                     │
 │  Set 1  [ reps ] [ RPE slider ]     │  ← filled in after completing the set
 │  Set 2  [ reps ] [ RPE slider ]     │
 │  Set 3  [ reps ] [ RPE slider ]     │
 │                                     │
 │         [ Next Exercise → ]         │
 └─────────────────────────────────────┘
       │
       ▼  (repeat for each exercise)
       │
       ▼
 Workout Complete
  - Summary of sets/reps/weight per exercise
  - 1RM recalculated for each exercise from logged reps + RPE
  - POST /log called once per exercise
```

**Key decisions:**
- Weight is **shown** from the prediction, not entered — user only logs reps and RPE.
- RPE entry is a slider (6–10 in 0.5 steps), not a text field.
- Rest timer runs automatically between sets (configurable, skippable).
- `POST /log` is called at the end of the workout, not after every set, to avoid partial writes if the user exits mid-session.
- **First session on a new exercise**: no prediction shown. User logs the weight they used — 1RM is calculated and stored. Prediction available from the next session.

---

### Development

Built with **React Native + Expo**. No emulator needed — install [Expo Go](https://expo.dev/go) on your phone, scan the QR code printed by `npx expo start`, and the app loads directly on your device over WiFi.

---

## Quickstart

```bash
# 1. Download the dataset and place it at data/programs_detailed.csv
# https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset

# 2. Start all services
docker compose up -d

# 3. Run the full pipeline (preprocess → train → evaluate → promote)
docker compose exec worker python src/pipeline/flow.py

# MLflow UI
open http://localhost:5001

# Prefect UI
open http://localhost:4200

# API docs
open http://localhost:8000/docs
```

## Roadmap

- [x] Boostcamp cold-start training pipeline
- [x] MLflow experiment tracking + model registry
- [x] Champion/challenger promotion logic
- [x] FastAPI prediction endpoint
- [x] Prefect orchestration + worker
- [ ] PostgreSQL session store
- [ ] `POST /log` endpoint
- [ ] Scheduled retraining flow on user log threshold
- [ ] React Native mobile app

