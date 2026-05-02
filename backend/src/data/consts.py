RPE_TABLE = {
    #  reps: {rpe: pct_1rm}
    1:  {6: 78, 7: 83, 8: 88, 9: 94, 10: 100},
    2:  {6: 75, 7: 80, 8: 85, 9: 91, 10: 97},
    3:  {6: 72, 7: 77, 8: 82, 9: 88, 10: 94},
    4:  {6: 69, 7: 74, 8: 79, 9: 85, 10: 91},
    5:  {6: 66, 7: 71, 8: 76, 9: 82, 10: 88},
    6:  {6: 63, 7: 68, 8: 73, 9: 79, 10: 85},
    7:  {6: 61, 7: 66, 8: 71, 9: 77, 10: 82},
    8:  {6: 58, 7: 63, 8: 68, 9: 74, 10: 79},
    9:  {6: 56, 7: 61, 8: 65, 9: 71, 10: 76},
    10: {6: 53, 7: 59, 8: 65, 9: 71, 10: 77},
    11: {6: 51, 7: 56, 8: 60, 9: 66, 10: 71},
    12: {6: 48, 7: 53, 8: 58, 9: 63, 10: 68},
}

def lookup_pct_1rm(reps: float, rpe: float) -> float | None:
    r = int(reps + 0.5)  # round-half-up (avoids Python banker's rounding)
    if r not in RPE_TABLE:
        return None
    row = RPE_TABLE[r]
    rpe_low = int(rpe)
    frac = rpe - rpe_low
    if rpe_low not in row:
        return None
    if frac == 0 or (rpe_low + 1) not in row:
        return row[rpe_low] / 100.0
    # linear interpolation between adjacent integer RPE entries
    return (row[rpe_low] + frac * (row[rpe_low + 1] - row[rpe_low])) / 100.0


# Maps LogRequest field names → Boostcamp training column names
GOAL_COL_MAP = {
    "goal_at_home_calisthenics":  "goal_At-Home & Calisthenics",
    "goal_athletics":             "goal_Athletics",
    "goal_bodybuilding":          "goal_Bodybuilding",
    "goal_bodyweight_fitness":    "goal_Bodyweight Fitness",
    "goal_muscle_sculpting":      "goal_Muscle & Sculpting",
    "goal_olympic_weightlifting": "goal_Olympic Weightlifting",
    "goal_powerbuilding":         "goal_Powerbuilding",
    "goal_powerlifting":          "goal_Powerlifting",
}

EQUIPMENT_COL_MAP = {
    "equipment_at_home":       "At Home",
    "equipment_dumbbell_only": "Dumbbell Only",
    "equipment_full_gym":      "Full Gym",
    "equipment_garage_gym":    "Garage Gym",
}

