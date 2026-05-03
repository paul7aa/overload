# Progressive Overload Predictor

A mobile-first MLOps app that predicts what weight to lift next week — and gets smarter the more you use it.

---

## The problem

Progressive overload (gradually increasing training stress over time) is the fundamental driver of strength and muscle gains. Knowing *how much* to increase each week is non-trivial: too little and you stagnate, too much and you risk injury or burnout.

Existing workout apps either follow rigid, pre-written programs or leave the decision entirely to the user. This project learns progression patterns from hundreds of real coach-designed programs, then continuously refines that knowledge from your own gym sessions.

---

## How it works

### Phase 1 — Cold start (Boostcamp dataset)

The model is bootstrapped on ~400k rows of structured workout program data (Boostcamp, via Kaggle), covering 2,500+ programs and 1,674 exercises. Each row is one exercise prescription: sets, reps, and intensity for a given week of a program.

**Dataset:** [600K+ Fitness Exercise & Workout Program Dataset](https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset) — download `programs_detailed_boostcamp_kaggle.csv` and save it as `backend/data/programs_detailed.csv`.

### Phase 2 — Continuous learning (user logs)

Every completed session logged via the app is stored in PostgreSQL. When enough new sessions accumulate, a Prefect flow retrains the model on the combined dataset. The new model is promoted to Production if it beats the current champion on a held-out validation set.

---

## Key concepts

### RPE — Rate of Perceived Exertion

RPE is a 6–10 scale describing how hard a set was in terms of reps left in reserve:

| RPE | Meaning |
|-----|---------|
| 10  | Maximum — could not do another rep |
| 9   | 1 rep left in reserve |
| 8   | 2 reps left in reserve |
| 7   | 3 reps left in reserve |
| 6   | 4+ reps left in reserve |

### pct_1rm — Percentage of One Rep Max

Your **1RM** is the maximum weight you can lift for a single rep. `pct_1rm` is your working weight as a fraction of that max.

RPE and pct_1rm are directly related via the Tuchscherer table — if you do 5 reps at RPE 8, you were lifting approximately 76% of your 1RM regardless of your absolute strength:

| Reps | RPE 7 | RPE 8 | RPE 9 | RPE 10 |
|------|-------|-------|-------|--------|
| 1    | 83%   | 88%   | 94%   | 100%   |
| 3    | 77%   | 82%   | 88%   | 94%    |
| 5    | 71%   | 76%   | 82%   | 88%    |
| 8    | 63%   | 68%   | 74%   | 79%    |
| 10   | 59%   | 65%   | 71%   | 77%    |
| 12   | 53%   | 58%   | 63%   | 68%    |

### What the model learns

The model predicts week-over-week deltas in relative terms:

```
Input:  (exercise_id, week, lag_sets, lag_reps, lag_pct_1rm, level, goal, equipment, ...)
Output: (Δsets, Δreps, Δpct_1rm)
```

At prediction time the user's 1RM (estimated from their last session via the Tuchscherer table) is multiplied by the predicted `pct_1rm` to produce an absolute weight:

```
next_weight_kg = (lag_pct_1rm + Δpct_1rm) × estimated_1rm
estimated_1rm  = weight / pct_1rm(reps, rpe)   ← Tuchscherer table
```

The Epley formula (`weight × (1 + reps/30)`) is only used to rank sets within a session when identifying the "best" set — it is not used for prediction. Tuchscherer keeps the 1RM estimate consistent with `lag_pct_1rm`, which is also derived from the same table.

---

## Architecture

### Retraining pipeline

Two Prefect flows:

- **`tuning_pipeline`** — runs occasionally (e.g. monthly). Runs Optuna hyperparameter search and saves the best params to `data/best_params.json`.
- **`retraining_pipeline`** — runs frequently (triggered by user log threshold). Loads saved params, retrains, and runs champion/challenger evaluation.

```
User logs session via mobile app
        │
        ▼
PostgreSQL (session store)
        │
        ▼  threshold reached
Prefect retraining_pipeline
        │
        ├─ 1. Preprocess    Merge Boostcamp + user logs → feature engineering
        ├─ 2. Train         MLflow experiment — LightGBM multi-output regression
        ├─ 3. Evaluate      Challenger vs current Production champion
        └─ 4. Promote       Best model tagged Production in MLflow Registry
                            FastAPI hot-swaps via /reload
```

### Services

| Service | Tool | Purpose |
|---------|------|---------|
| Experiment tracking | MLflow | Log params, metrics, model artifacts |
| Orchestration | Prefect | Retraining flows + scheduling |
| Session store | PostgreSQL | Persist user gym logs |
| API | FastAPI + Pydantic | Predictions, logging, model reload |
| Containerisation | Docker Compose | Full stack, single command |
| Mobile app | React Native + Expo | iOS/Android client |

---

## API

```
POST /predict            Get next week's prescription
POST /log                Record a completed session (feeds retraining)
GET  /exercises          List all known exercises
GET  /exercise-info      Exercise description, step-by-step instructions, image and GIF URLs
GET  /media/{path}       Static serving of exercise images and GIFs
GET  /health             Model status + version
POST /reload             Hot-swap to latest Production model
GET  /docs               Interactive Swagger UI
```

### Example

```json
POST /predict
{
  "exercise": "Barbell Bench Press",
  "one_rm": 100,
  "lag_sets": 3, "lag_reps": 5, "lag_rpe": 8,
  "week": 4, "program_length": 12,
  "day": 1, "time_per_workout": 60, "number_of_exercises": 4,
  "level_Intermediate": 1, "goal_powerbuilding": 1, "equipment_full_gym": 1
}

→ { "next_sets": 3, "next_reps": 5, "delta_pct_1rm": 0.03, "next_weight_kg": 79.0 }
```

Exercise names sent to `/predict` and `/log` are matched case-insensitively against the exercise map.

---

## Exercise data

The app uses 484 canonical exercises sourced from the ExerciseDB dataset. Only exercises that appear in the Boostcamp training data are included. The exercise list is bundled statically in the mobile app — no network request needed for search.

The full ExerciseDB dataset (`backend/data/exercises_dataset/`) provides per-exercise descriptions, step-by-step instructions, a static image, and an animated GIF for each movement. These are served at runtime via FastAPI's `StaticFiles` mount (`GET /media/...`) — no on-device storage required. The `api` service mounts `./data` as a volume so the files are never baked into the Docker image.

---

## Project structure

```
overload/
├── backend/
│   ├── data/
│   │   ├── programs_detailed.csv         # Boostcamp dataset (not committed)
│   │   ├── exercise_map.json             # Canonical exercise name → integer ID
│   │   ├── best_params.json              # Saved Optuna hyperparameters (not committed)
│   │   ├── exercises_dataset/            # ExerciseDB — descriptions, images, GIFs (not committed)
│   │   │   ├── data/exercises.json
│   │   │   ├── images/
│   │   │   └── videos/
│   │   ├── map_exercises.py              # Maps messy training names → canonical names
│   │   └── generate_frontend_exercises.py
│   ├── src/
│   │   ├── data/
│   │   │   ├── consts.py                 # Tuchscherer RPE → pct_1rm table
│   │   │   └── preprocess.py             # Cleaning + feature engineering
│   │   ├── models/
│   │   │   ├── train.py                  # MLflow training run + DEFAULT_HYPERPARAMS
│   │   │   ├── tune.py                   # Optuna hyperparameter search
│   │   │   ├── evaluate.py               # Champion vs challenger
│   │   │   └── utils.py                  # Model loading helpers
│   │   ├── pipeline/
│   │   │   └── flow.py                   # tuning_pipeline + retraining_pipeline
│   │   └── api/
│   │       ├── app.py                    # FastAPI app
│   │       ├── schemas.py                # Pydantic I/O models
│   │       └── db.py                     # SQLAlchemy session log model
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── api/
    │   │   └── client.ts                 # predict(), logWorkout(), fetchExerciseInfo(), flag helpers
    │   ├── data/
    │   │   └── exercises.json            # 484 canonical exercises (bundled)
    │   ├── screens/
    │   │   ├── HomeScreen.tsx
    │   │   ├── AddProgramModal.tsx
    │   │   ├── ActiveWorkoutScreen.tsx
    │   │   ├── WorkoutCompleteScreen.tsx
    │   │   └── WorkoutHistoryScreen.tsx
    │   ├── types.ts
    │   └── theme.ts
    ├── App.tsx
    └── package.json
```

---

## Quickstart

```bash
# 1. Download the dataset → backend/data/programs_detailed.csv

# 2. Start all services
cd backend && docker compose up -d

# 3. Run the training pipeline
docker compose exec worker python src/pipeline/flow.py

# 4. Start the mobile app
cd frontend && npx expo start
```

`exercise_map.json` is committed — no mapping step or API keys needed.

The app connects to the backend over your local network. Set your machine's LAN IP in `frontend/.env`:

```
EXPO_PUBLIC_API_URL=http://<your-machine-ip>:8000
```

| UI | URL |
|----|-----|
| MLflow | http://localhost:5001 |
| Prefect | http://localhost:4200 |
| API docs | http://localhost:8000/docs |

---

## Roadmap

- [x] Boostcamp cold-start training pipeline
- [x] MLflow experiment tracking + model registry
- [x] Champion/challenger promotion logic
- [x] FastAPI prediction + logging endpoints
- [x] Prefect orchestration — tuning and retraining flows
- [x] PostgreSQL session store
- [x] React Native mobile app (Expo)
- [x] On-device workout history
- [x] `/predict` wired into active workout screen (week 2+)
- [ ] Push notifications for retraining trigger
- [ ] Scheduled Prefect deployment
