import AsyncStorage from '@react-native-async-storage/async-storage';
import { Program, WorkoutRecord, LastSessionEntry } from '../types';
import { HISTORY_KEY } from '../screens/WorkoutCompleteScreen';

const PROGRAMS_KEY = 'programs';
const SELECTED_KEY = 'selected_program_id';
const PROGRAM_ID = 'dev-ppl-1';

const PROGRAM: Program = {
  id: PROGRAM_ID,
  name: 'PPL Hypertrophy',
  level: ['Intermediate'],
  goal: ['Bodybuilding'],
  equipment: 'Full Gym',
  lengthWeeks: 12,
  timePerWorkout: 75,
  overloadType: 'linear',
  days: [
    {
      dayNumber: 1,
      exercises: [
        { id: 98,  name: 'Barbell Bench Press',           muscle: 'chest',       full: 'Barbell Bench Press',           sets: 4, reps: 8  },
        { id: 130, name: 'Barbell Incline Bench Press',   muscle: 'chest',       full: 'Barbell Incline Bench Press',   sets: 3, reps: 10 },
        { id: 74,  name: 'Band Shoulder Press',           muscle: 'shoulders',   full: 'Band Shoulder Press',           sets: 3, reps: 12 },
        { id: 33,  name: 'Assisted Triceps Dip (Kneeling)', muscle: 'upper arms', full: 'Assisted Triceps Dip (Kneeling)', sets: 3, reps: 10 },
      ],
    },
    {
      dayNumber: 2,
      exercises: [
        { id: 101, name: 'Barbell Bent Over Row',    muscle: 'back',        full: 'Barbell Bent Over Row',    sets: 4, reps: 8  },
        { id: 7,   name: 'Archer Pull Up',           muscle: 'back',        full: 'Archer Pull Up',           sets: 3, reps: 6  },
        { id: 49,  name: 'Band Close-Grip Pulldown', muscle: 'back',        full: 'Band Close-Grip Pulldown', sets: 3, reps: 12 },
        { id: 106, name: 'Barbell Curl',             muscle: 'upper arms',  full: 'Barbell Curl',             sets: 3, reps: 10 },
      ],
    },
    {
      dayNumber: 3,
      exercises: [
        { id: 120,  name: 'Barbell Full Squat',          muscle: 'upper legs', full: 'Barbell Full Squat',          sets: 4, reps: 5  },
        { id: 182,  name: 'Barbell Romanian Deadlift',   muscle: 'upper legs', full: 'Barbell Romanian Deadlift',   sets: 3, reps: 8  },
        { id: 1158, name: 'Sled 45° Leg Press',          muscle: 'upper legs', full: 'Sled 45° Leg Press',          sets: 3, reps: 12 },
        { id: 955,  name: 'Lever Lying Leg Curl',        muscle: 'upper legs', full: 'Lever Lying Leg Curl',        sets: 3, reps: 12 },
      ],
    },
  ],
};

// Progressive week-by-week data per exercise: index 0 = oldest, 3 = most recent
type WeekEntry = { weight: number; reps: number; rpe: number };

const HISTORY: Record<string, WeekEntry[]> = {
  'Barbell Bench Press':             [{ weight: 75,  reps: 8,  rpe: 8   }, { weight: 77.5, reps: 8,  rpe: 8   }, { weight: 80,   reps: 8,  rpe: 8.5 }, { weight: 82.5, reps: 8,  rpe: 8.5 }],
  'Barbell Incline Bench Press':     [{ weight: 60,  reps: 10, rpe: 7.5 }, { weight: 62.5, reps: 10, rpe: 8   }, { weight: 62.5, reps: 11, rpe: 8   }, { weight: 65,   reps: 10, rpe: 8   }],
  'Band Shoulder Press':             [{ weight: 40,  reps: 12, rpe: 7   }, { weight: 42.5, reps: 12, rpe: 7.5 }, { weight: 42.5, reps: 13, rpe: 7.5 }, { weight: 45,   reps: 12, rpe: 7.5 }],
  'Assisted Triceps Dip (Kneeling)': [{ weight: 0,   reps: 10, rpe: 8   }, { weight: 0,    reps: 11, rpe: 8   }, { weight: 0,    reps: 12, rpe: 8   }, { weight: 5,    reps: 10, rpe: 8   }],
  'Barbell Bent Over Row':           [{ weight: 70,  reps: 8,  rpe: 8   }, { weight: 72.5, reps: 8,  rpe: 8   }, { weight: 75,   reps: 8,  rpe: 8.5 }, { weight: 75,   reps: 9,  rpe: 8.5 }],
  'Archer Pull Up':                  [{ weight: 0,   reps: 6,  rpe: 8.5 }, { weight: 0,    reps: 7,  rpe: 8.5 }, { weight: 0,    reps: 7,  rpe: 8   }, { weight: 0,    reps: 8,  rpe: 8   }],
  'Band Close-Grip Pulldown':        [{ weight: 50,  reps: 12, rpe: 7.5 }, { weight: 52.5, reps: 12, rpe: 7.5 }, { weight: 55,   reps: 12, rpe: 8   }, { weight: 57.5, reps: 12, rpe: 8   }],
  'Barbell Curl':                    [{ weight: 35,  reps: 10, rpe: 8   }, { weight: 37.5, reps: 10, rpe: 8   }, { weight: 37.5, reps: 11, rpe: 8   }, { weight: 40,   reps: 10, rpe: 8   }],
  'Barbell Full Squat':              [{ weight: 100, reps: 5,  rpe: 8   }, { weight: 102.5,reps: 5,  rpe: 8   }, { weight: 105,  reps: 5,  rpe: 8.5 }, { weight: 107.5,reps: 5,  rpe: 8.5 }],
  'Barbell Romanian Deadlift':       [{ weight: 80,  reps: 8,  rpe: 8   }, { weight: 82.5, reps: 8,  rpe: 8   }, { weight: 85,   reps: 8,  rpe: 8.5 }, { weight: 87.5, reps: 8,  rpe: 8.5 }],
  'Sled 45° Leg Press':              [{ weight: 120, reps: 12, rpe: 7.5 }, { weight: 125,  reps: 12, rpe: 7.5 }, { weight: 130,  reps: 12, rpe: 8   }, { weight: 135,  reps: 12, rpe: 8   }],
  'Lever Lying Leg Curl':            [{ weight: 40,  reps: 12, rpe: 7.5 }, { weight: 42.5, reps: 12, rpe: 7.5 }, { weight: 42.5, reps: 13, rpe: 7.5 }, { weight: 45,   reps: 12, rpe: 8   }],
};

const ONE_RM: Record<string, number> = {
  'Barbell Bench Press':             110,
  'Barbell Incline Bench Press':     86,
  'Band Shoulder Press':             62,
  'Assisted Triceps Dip (Kneeling)': 72,
  'Barbell Bent Over Row':           102,
  'Archer Pull Up':                  82,
  'Band Close-Grip Pulldown':        76,
  'Barbell Curl':                    52,
  'Barbell Full Squat':              135,
  'Barbell Romanian Deadlift':       115,
  'Sled 45° Leg Press':              185,
  'Lever Lying Leg Curl':            62,
};

function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  d.setHours(10, 30, 0, 0);
  return d.toISOString();
}

// Merged last-session map for all exercises at a given week index.
// Merging all days lets the AI suggest on any day picked, not just the most recent one.
function mergedLastSession(weekIdx: number): Record<string, LastSessionEntry> {
  const out: Record<string, LastSessionEntry> = {};
  for (const day of PROGRAM.days) {
    for (const ex of day.exercises) {
      const w = HISTORY[ex.name][weekIdx];
      out[ex.name] = {
        sets: ex.sets,
        reps: w.reps,
        rpe: w.rpe,
        oneRm: ONE_RM[ex.name] ?? 80,
        ...(w.weight > 0 && { weight: w.weight }),
      };
    }
  }
  return out;
}

// Runs once per install — skips if the dev program already exists.
export async function seedDevData(): Promise<void> {
  const existing = await AsyncStorage.getItem(PROGRAMS_KEY);
  const programs: Program[] = existing ? JSON.parse(existing) : [];
  if (programs.some(p => p.id === PROGRAM_ID)) return;

  // 4 weeks × 3 days = 12 sessions, spaced out over the past month
  const schedule = [
    [29, 27, 25],
    [22, 20, 18],
    [15, 13, 11],
    [8,  6,  4 ],
  ];

  const history: WorkoutRecord[] = [];
  for (let week = 0; week < 4; week++) {
    for (let dayIdx = 0; dayIdx < 3; dayIdx++) {
      const day = PROGRAM.days[dayIdx];
      history.push({
        id: `dev-w${week}-d${dayIdx}`,
        programId: PROGRAM_ID,
        programName: PROGRAM.name,
        dayNumber: day.dayNumber,
        weekNumber: week + 1,
        durationSeconds: 2700 + dayIdx * 300 + week * 120,
        completedAt: daysAgo(schedule[week][dayIdx]),
        exercises: day.exercises.map(ex => {
          const w = HISTORY[ex.name][week];
          return {
            name: ex.name,
            muscle: ex.muscle,
            sets: Array.from({ length: ex.sets }, () => ({ weight: w.weight, reps: w.reps, rpe: w.rpe })),
          };
        }),
      });
    }
  }

  await AsyncStorage.multiSet([
    [PROGRAMS_KEY,                           JSON.stringify([...programs, PROGRAM])],
    [SELECTED_KEY,                           PROGRAM_ID],
    [HISTORY_KEY,                            JSON.stringify(history)],
    [`last_session_${PROGRAM_ID}`,           JSON.stringify(mergedLastSession(3))],
    [`second_last_session_${PROGRAM_ID}`,    JSON.stringify(mergedLastSession(2))],
    [`third_last_session_${PROGRAM_ID}`,     JSON.stringify(mergedLastSession(1))],
    [`day_count_${PROGRAM_ID}_1`,            '4'],
    [`day_count_${PROGRAM_ID}_2`,            '4'],
    [`day_count_${PROGRAM_ID}_3`,            '4'],
  ]);

  console.log('[dev] seeded PPL Hypertrophy with 12 mock workouts');
}