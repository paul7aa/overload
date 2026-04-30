"""
Generates frontend/src/data/exercises.json from the canonical exercise dataset.
Run once from the backend/data directory:

    cd backend/data
    python generate_frontend_exercises.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).parent

with open(ROOT / "exercises_dataset/data/exercises.json") as f:
    dataset = json.load(f)

with open(ROOT / "exercise_map.json") as f:
    exercise_map = json.load(f)  # { lowercase_name: integer_id }

name_to_meta = {ex["name"]: ex for ex in dataset}

def fix_encoding(s: str) -> str:
    # Fix double-encoded UTF-8: Â° → °, etc.
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s

result = []
for name, eid in exercise_map.items():
    meta = name_to_meta.get(name, {})
    display = fix_encoding(name).title()
    result.append({
        "id": eid,
        "name": display,
        "muscle": meta.get("body_part", ""),
        "full": display,
    })

result.sort(key=lambda e: e["name"])

out = ROOT.parent.parent / "frontend" / "src" / "data" / "exercises.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)

print(f"Wrote {len(result)} exercises to {out}")
