import { Program } from '../types';

const BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';
const HEADERS = {
  'Content-Type': 'application/json',
  'X-API-Key': process.env.EXPO_PUBLIC_API_KEY ?? '',
};

export type PredictRequest = {
  exercise: string;
  one_rm: number;
  lag_sets: number;
  lag_reps: number;
  lag_rpe: number;
  week: number;
  day: number;
  program_length: number;
  time_per_workout: number;
  number_of_exercises: number;
  weeks_gap?: number;
  level_Advanced?: number;
  level_Beginner?: number;
  level_Intermediate?: number;
  level_Novice?: number;
  goal_at_home_calisthenics?: number;
  goal_athletics?: number;
  goal_bodybuilding?: number;
  goal_bodyweight_fitness?: number;
  goal_muscle_sculpting?: number;
  goal_olympic_weightlifting?: number;
  goal_powerbuilding?: number;
  goal_powerlifting?: number;
  equipment_at_home?: number;
  equipment_dumbbell_only?: number;
  equipment_full_gym?: number;
  equipment_garage_gym?: number;
};

export type PredictResponse = {
  delta_sets: number;
  delta_reps: number;
  delta_pct_1rm: number;
  next_sets: number;
  next_reps: number;
  next_weight_kg: number;
};

export type LogRequest = {
  user_id: string;
  exercise: string;
  one_rm: number;
  week: number;
  day: number;
  program_length: number;
  time_per_workout: number;
  number_of_exercises: number;
  weeks_gap?: number;
  lag_sets: number;
  lag_reps: number;
  lag_rpe: number;
  sets: number;
  reps: number;
  rpe: number;
  level_Advanced?: number;
  level_Beginner?: number;
  level_Intermediate?: number;
  level_Novice?: number;
  goal_at_home_calisthenics?: number;
  goal_athletics?: number;
  goal_bodybuilding?: number;
  goal_bodyweight_fitness?: number;
  goal_muscle_sculpting?: number;
  goal_olympic_weightlifting?: number;
  goal_powerbuilding?: number;
  goal_powerlifting?: number;
  equipment_at_home?: number;
  equipment_dumbbell_only?: number;
  equipment_full_gym?: number;
  equipment_garage_gym?: number;
};

export function levelFlags(program: Program) {
  return {
    level_Advanced:     program.level.includes('Advanced')     ? 1 : 0,
    level_Beginner:     program.level.includes('Beginner')     ? 1 : 0,
    level_Intermediate: program.level.includes('Intermediate') ? 1 : 0,
    level_Novice:       program.level.includes('Novice')       ? 1 : 0,
  };
}

export function goalFlags(program: Program) {
  const g = program.goal;
  return {
    goal_at_home_calisthenics:  g.includes('At-Home & Calisthenics')  ? 1 : 0,
    goal_athletics:             g.includes('Athletics')                ? 1 : 0,
    goal_bodybuilding:          g.includes('Bodybuilding')             ? 1 : 0,
    goal_bodyweight_fitness:    g.includes('Bodyweight Fitness')       ? 1 : 0,
    goal_muscle_sculpting:      g.includes('Muscle & Sculpting')       ? 1 : 0,
    goal_olympic_weightlifting: g.includes('Olympic Weightlifting')    ? 1 : 0,
    goal_powerbuilding:         g.includes('Powerbuilding')            ? 1 : 0,
    goal_powerlifting:          g.includes('Powerlifting')             ? 1 : 0,
  };
}

export function equipmentFlags(program: Program) {
  const e = program.equipment;
  return {
    equipment_at_home:       e === 'At Home'       ? 1 : 0,
    equipment_dumbbell_only: e === 'Dumbbell Only' ? 1 : 0,
    equipment_full_gym:      e === 'Full Gym'      ? 1 : 0,
    equipment_garage_gym:    e === 'Garage Gym'    ? 1 : 0,
  };
}

export function epleyOneRm(weight: number, reps: number): number {
  return weight * (1 + reps / 30);
}

// Mirrors the backend RPE_TABLE — keep in sync with backend/src/data/consts.py
const RPE_TABLE: Record<number, Record<number, number>> = {
  1:  {6: 78, 7: 83, 8: 88,  9: 94,  10: 100},
  2:  {6: 75, 7: 80, 8: 85,  9: 91,  10: 97},
  3:  {6: 72, 7: 77, 8: 82,  9: 88,  10: 94},
  4:  {6: 69, 7: 74, 8: 79,  9: 85,  10: 91},
  5:  {6: 66, 7: 71, 8: 76,  9: 82,  10: 88},
  6:  {6: 63, 7: 68, 8: 73,  9: 79,  10: 85},
  7:  {6: 61, 7: 66, 8: 71,  9: 77,  10: 82},
  8:  {6: 58, 7: 63, 8: 68,  9: 74,  10: 79},
  9:  {6: 56, 7: 61, 8: 65,  9: 71,  10: 76},
  10: {6: 53, 7: 59, 8: 65,  9: 71,  10: 77},
  11: {6: 51, 7: 56, 8: 60,  9: 66,  10: 71},
  12: {6: 48, 7: 53, 8: 58,  9: 63,  10: 68},
};

function lookupPct1rm(reps: number, rpe: number): number | null {
  const r = Math.floor(reps + 0.5);
  const row = RPE_TABLE[r];
  if (!row) return null;
  const rpeFloor = Math.floor(rpe);
  const frac = rpe - rpeFloor;
  if (!(rpeFloor in row)) return null;
  if (frac === 0 || !((rpeFloor + 1) in row)) return row[rpeFloor] / 100;
  return (row[rpeFloor] + frac * (row[rpeFloor + 1] - row[rpeFloor])) / 100;
}

// Use Tuchscherer table (same as backend) so oneRm is consistent with lag_pct_1rm.
// Epley overestimates 1RM, which causes a systematic -10 to -20% weight bias.
export function tuchschererOneRm(weight: number, reps: number, rpe: number): number {
  const pct = lookupPct1rm(reps, rpe);
  return pct ? weight / pct : 0;
}

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: HEADERS,
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`predict ${res.status}`);
  return res.json();
}

export async function logWorkout(req: LogRequest): Promise<void> {
  const res = await fetch(`${BASE}/log`, {
    method: 'POST',
    headers: HEADERS,
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`log ${res.status}`);
}

export type ExerciseInfo = {
  name: string;
  description: string;
  steps: string[];
  image_url: string | null;
  gif_url: string | null;
};

export async function fetchExerciseInfo(name: string): Promise<ExerciseInfo | null> {
  try {
    const res = await fetch(`${BASE}/exercise-info?name=${encodeURIComponent(name)}`, { headers: HEADERS });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
