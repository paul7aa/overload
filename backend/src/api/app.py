import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.models.utils import load_model, load_exercise_map, MODEL_NAME, ALIAS_PROD
from src.api.schemas import PredictRequest, PredictResponse, LogRequest, _FIELD_TO_COL, _FEATURE_COLS
from src.api.db import SessionLocal, WorkoutLog, create_tables
from src.data.consts import lookup_pct_1rm

_model = None
_version = None
_exercise_map: dict[str, int] = {}
_exercise_list: list[dict] = []


def _build_exercise_list(exercise_map: dict[str, int]) -> list[dict]:
    try:
        with open("data/exercises_dataset/data/exercises.json") as f:
            dataset = json.load(f)
        name_to_meta = {ex["name"]: ex for ex in dataset}
    except (FileNotFoundError, json.JSONDecodeError):
        name_to_meta = {}

    exercises = [
        {
            "id": eid,
            "name": name.title(),
            "muscle": name_to_meta.get(name, {}).get("body_part", ""),
            "full": name.title(),
        }
        for name, eid in exercise_map.items()
    ]
    return sorted(exercises, key=lambda e: e["name"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _version, _exercise_map, _exercise_list
    create_tables()
    _model, _version = load_model()
    _exercise_map = load_exercise_map()
    _exercise_list = _build_exercise_list(_exercise_map)
    yield


app = FastAPI(title="Overload Predictor", lifespan=lifespan)

# ENDPOINTS

@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": _version.version,
        "alias": ALIAS_PROD,
    }


@app.get("/exercises")
def list_exercises():
    return _exercise_list


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    exercise_key = req.exercise.lower()
    if exercise_key not in _exercise_map:
        raise HTTPException(422, f"Unknown exercise: '{req.exercise}'")

    exercise_id = _exercise_map[exercise_key]
    lag_pct_1rm = lookup_pct_1rm(req.lag_reps, req.lag_rpe)
    if lag_pct_1rm is None:
        raise HTTPException(422, f"RPE {req.lag_rpe} with {req.lag_reps} reps is outside the Tuchscherer table (reps 1–12, RPE 6–10)")
    lag_volume  = req.lag_sets * req.lag_reps * lag_pct_1rm
    week_pct    = round(req.week / req.program_length, 3)

    # Build the feature row with training column names, then order correctly
    row = {_FIELD_TO_COL[k]: v for k, v in req.model_dump().items() if k in _FIELD_TO_COL}
    row["exercise_id"] = exercise_id
    row["week_pct"]    = week_pct
    row["lag_pct_1rm"] = lag_pct_1rm
    row["lag_volume"]  = lag_volume

    df = pd.DataFrame([row])[_FEATURE_COLS]

    # TARGETS order in train.py: ["delta_sets", "delta_reps", "delta_pct_1rm"]
    delta_sets, delta_reps, delta_pct_1rm = _model.predict(df)[0]

    return PredictResponse(
        delta_sets=round(delta_sets, 2),
        delta_reps=round(delta_reps, 2),
        delta_pct_1rm=round(delta_pct_1rm, 4),
        next_sets=max(1, min(10, round(req.lag_sets + delta_sets))),
        next_reps=max(1, min(12, round(req.lag_reps + delta_reps))),
        next_weight_kg=round((lag_pct_1rm + delta_pct_1rm) * req.one_rm, 2),
    )


@app.post("/log", status_code=201)
def log_session(req: LogRequest):
    exercise_key = req.exercise.lower()
    if exercise_key not in _exercise_map:
        raise HTTPException(422, f"Unknown exercise: '{req.exercise}'")

    db = SessionLocal()
    try:
        entry = WorkoutLog(**{**req.model_dump(), "exercise": exercise_key})
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return {"id": entry.id, "status": "logged"}
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"Failed to log session: {exc}")
    finally:
        db.close()


@app.post("/reload")
def reload():
    global _model, _version
    _model, _version = load_model()
    if _model is None:
        raise HTTPException(503, "No Production model found in MLflow")
    return {"status": "reloaded", "model_version": _version.version}
