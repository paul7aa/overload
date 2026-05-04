import gc
import json
import os
from contextlib import asynccontextmanager

import pandas as pd
import requests as _requests
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func

from src.models.utils import load_model, load_exercise_map, MODEL_NAME, ALIAS_PROD
from src.api.schemas import PredictRequest, PredictResponse, LogRequest, PushTokenRequest, _FIELD_TO_COL, _FEATURE_COLS
from src.api.db import SessionLocal, WorkoutLog, PushToken, create_tables
from src.data.consts import lookup_pct_1rm

_MEDIA_DIR = "data/exercises_dataset"
RETRAIN_EVERY = int(os.environ.get("RETRAIN_EVERY", "20"))
_PREFECT_API = os.environ.get("PREFECT_API_URL", "http://prefect:4200/api")
_DEPLOYMENT = "overload-retraining-pipeline/scheduled-retraining"
_API_KEY = os.environ.get("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_model = None
_version = None
_exercise_map: dict[str, int] = {}
_exercise_list: list[dict] = []
_exercise_info: dict[str, dict] = {}  # lowercase name → full dataset entry

def _trigger_retraining():
    """Fire-and-forget: ask Prefect to create a flow run for the retraining deployment."""
    try:
        resp = _requests.get(f"{_PREFECT_API}/deployments/name/{_DEPLOYMENT}", timeout=5)
        if resp.status_code != 200:
            print(f"[retrain] deployment not found ({resp.status_code}) — run flow.py first")
            return
        deployment_id = resp.json()["id"]
        _requests.post(f"{_PREFECT_API}/deployments/{deployment_id}/create_flow_run", json={}, timeout=5)
        print(f"[retrain] triggered retraining pipeline")
    except Exception as exc:
        print(f"[retrain] trigger failed: {exc}")


def verify_api_key(key: str | None = Security(_api_key_header)):
    if not _API_KEY:
        return  # not configured — open in dev
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


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
    global _model, _version, _exercise_map, _exercise_list, _exercise_info
    create_tables()
    _model, _version = load_model()
    _exercise_map = load_exercise_map()
    _exercise_list = _build_exercise_list(_exercise_map)
    try:
        with open("data/exercises_dataset/data/exercises.json") as f:
            _exercise_info = {ex["name"].lower(): ex for ex in json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    yield


app = FastAPI(title="Overload Predictor", lifespan=lifespan, dependencies=[Depends(verify_api_key)])
if os.path.isdir(_MEDIA_DIR):
    app.mount("/media", StaticFiles(directory=_MEDIA_DIR), name="media")

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


@app.get("/exercise-info")
def get_exercise_info(name: str = Query(..., description="Exercise name (case-insensitive)")):
    entry = _exercise_info.get(name.lower())
    if not entry:
        raise HTTPException(404, f"No info found for '{name}'")
    return {
        "name": entry["name"],
        "description": entry.get("instructions", {}).get("en", ""),
        "steps": entry.get("instruction_steps", {}).get("en", []),
        "image_url": f"/media/{entry['image']}" if entry.get("image") else None,
        "gif_url": f"/media/{entry['gif_url']}" if entry.get("gif_url") else None,
    }


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
        next_weight_kg=round(round((lag_pct_1rm + delta_pct_1rm) * req.one_rm / 5) * 5, 2),
    )


@app.post("/register-push-token", status_code=200)
def register_push_token(req: PushTokenRequest):
    db = SessionLocal()
    try:
        exists = db.query(PushToken).filter(PushToken.token == req.token).first()
        if not exists:
            db.add(PushToken(token=req.token))
            db.commit()
        return {"status": "ok"}
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"Failed to register token: {exc}")
    finally:
        db.close()


@app.post("/log", status_code=201)
def log_session(req: LogRequest, background: BackgroundTasks):
    exercise_key = req.exercise.lower()
    if exercise_key not in _exercise_map:
        raise HTTPException(422, f"Unknown exercise: '{req.exercise}'")

    db = SessionLocal()
    try:
        entry = WorkoutLog(**{**req.model_dump(), "exercise": exercise_key})
        db.add(entry)
        db.commit()
        db.refresh(entry)
        count = db.query(func.count(WorkoutLog.id)).scalar()
        if count % RETRAIN_EVERY == 0:
            background.add_task(_trigger_retraining)
        return {"id": entry.id, "status": "logged"}
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"Failed to log session: {exc}")
    finally:
        db.close()


@app.post("/reload")
def reload():
    global _model, _version
    new_model, new_version = load_model()
    if new_model is None:
        raise HTTPException(503, "No Production model found in MLflow")
    old_model = _model
    _model, _version = new_model, new_version
    del old_model
    gc.collect()
    return {"status": "reloaded", "model_version": _version.version}
