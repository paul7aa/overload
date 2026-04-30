import { Program } from '../types';

const BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';

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

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`predict ${res.status}`);
  return res.json();
}

export async function logWorkout(req: LogRequest): Promise<void> {
  const res = await fetch(`${BASE}/log`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`log ${res.status}`);
}
