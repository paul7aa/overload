from pydantic import BaseModel


class PredictRequest(BaseModel):
    # Exercise being performed
    exercise: str          # human-readable name → looked up to exercise_id

    # User's current one-rep max for this exercise (kg)
    # Used only for the absolute weight output — not a model feature
    one_rm: float

    # ── Previous week's performance ─────────────────────────────────────────
    lag_sets: int
    lag_reps: float
    lag_rpe: float         # RPE 6–10; converted to lag_pct_1rm internally

    # ── Position in the program ─────────────────────────────────────────────
    week: int              # current week number (1-indexed)
    day: int               # day within the week (1-indexed)
    program_length: int    # total weeks in the program
    time_per_workout: float
    number_of_exercises: int

    # ── Weeks since last session on this exercise ────────────────────────────
    weeks_gap: int = 1

    # ── Fitness level (exactly one should be 1) ──────────────────────────────
    level_Advanced: int = 0
    level_Beginner: int = 0
    level_Intermediate: int = 0
    level_Novice: int = 0

    # ── Program goal (at least one should be 1) ──────────────────────────────
    goal_at_home_calisthenics: int = 0   # "goal_At-Home & Calisthenics"
    goal_athletics: int = 0              # "goal_Athletics"
    goal_bodybuilding: int = 0           # "goal_Bodybuilding"
    goal_bodyweight_fitness: int = 0     # "goal_Bodyweight Fitness"
    goal_muscle_sculpting: int = 0       # "goal_Muscle & Sculpting"
    goal_olympic_weightlifting: int = 0  # "goal_Olympic Weightlifting"
    goal_powerbuilding: int = 0          # "goal_Powerbuilding"
    goal_powerlifting: int = 0           # "goal_Powerlifting"

    # ── Equipment (at least one should be 1) ─────────────────────────────────
    equipment_at_home: int = 0           # "At Home"
    equipment_dumbbell_only: int = 0     # "Dumbbell Only"
    equipment_full_gym: int = 0          # "Full Gym"
    equipment_garage_gym: int = 0        # "Garage Gym"


class PredictResponse(BaseModel):
    delta_sets: float
    delta_reps: float
    delta_pct_1rm: float

    # Absolute prescription for next session
    next_sets: int
    next_reps: int
    next_weight_kg: float

# Maps Pydantic field names -> training DataFrame column names.
# Needed because pandas column names can contain spaces and special characters
# that aren't valid Python identifiers.
_FIELD_TO_COL = {
    "program_length":             "program_length",
    "time_per_workout":           "time_per_workout",
    "week":                       "week",
    "day":                        "day",
    "number_of_exercises":        "number_of_exercises",
    "level_Advanced":             "level_Advanced",
    "level_Beginner":             "level_Beginner",
    "level_Intermediate":         "level_Intermediate",
    "level_Novice":               "level_Novice",
    "goal_at_home_calisthenics":  "goal_At-Home & Calisthenics",
    "goal_athletics":             "goal_Athletics",
    "goal_bodybuilding":          "goal_Bodybuilding",
    "goal_bodyweight_fitness":    "goal_Bodyweight Fitness",
    "goal_muscle_sculpting":      "goal_Muscle & Sculpting",
    "goal_olympic_weightlifting": "goal_Olympic Weightlifting",
    "goal_powerbuilding":         "goal_Powerbuilding",
    "goal_powerlifting":          "goal_Powerlifting",
    "equipment_at_home":          "At Home",
    "equipment_dumbbell_only":    "Dumbbell Only",
    "equipment_full_gym":         "Full Gym",
    "equipment_garage_gym":       "Garage Gym",
    "lag_sets":                   "lag_sets",
    "lag_reps":                   "lag_reps",
    "weeks_gap":                  "weeks_gap",
}

# matches the exact column order the model was trained on.
_FEATURE_COLS = [
    "program_length", "time_per_workout", "week", "day", "number_of_exercises",
    "level_Advanced", "level_Beginner", "level_Intermediate", "level_Novice",
    "goal_At-Home & Calisthenics", "goal_Athletics", "goal_Bodybuilding",
    "goal_Bodyweight Fitness", "goal_Muscle & Sculpting", "goal_Olympic Weightlifting",
    "goal_Powerbuilding", "goal_Powerlifting",
    "At Home", "Dumbbell Only", "Full Gym", "Garage Gym",
    "exercise_id", "week_pct",
    "lag_sets", "lag_reps", "lag_pct_1rm", "lag_volume", "weeks_gap",
]
