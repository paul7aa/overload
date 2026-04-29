export type Exercise = {
  id: number;
  name: string;
  muscle: string;
  full: string;
};

export type ProgramExercise = Exercise & {
  sets: number;
  reps: number;
};

export type WorkoutDay = {
  dayNumber: number;
  exercises: ProgramExercise[];
};

export type Program = {
  id: string;
  name: string;
  level: string[];
  goal: string[];
  equipment: string;
  lengthWeeks: number;
  timePerWorkout: number;
  days: WorkoutDay[];
};

export type SetLog = {
  reps: number;
  rpe: number;
  weight: number;
};

export type ExerciseLog = {
  exercise: ProgramExercise;
  dayNumber: number;
  sets: SetLog[];
};

export type RootStackParamList = {
  Home: undefined;
  ActiveWorkout: { program: Program };
  WorkoutComplete: { logs: ExerciseLog[]; program: Program };
  AddProgram: { program?: Program };
};
