import { useEffect } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { ExerciseLog, LastSessionEntry, RootStackParamList, WorkoutRecord } from '../types';
import { epleyOneRm, equipmentFlags, goalFlags, levelFlags, logWorkout, tuchschererOneRm } from '../api/client';

type Props = NativeStackScreenProps<RootStackParamList, 'WorkoutComplete'>;

export const HISTORY_KEY = 'workout_history';

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

function buildRecord(
  logs: ExerciseLog[],
  program: Props['route']['params']['program'],
  durationSeconds: number,
  weekNumber: number,
): WorkoutRecord {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    programId: program.id,
    programName: program.name,
    dayNumber: logs[0]?.dayNumber ?? 1,
    weekNumber,
    durationSeconds,
    completedAt: new Date().toISOString(),
    exercises: logs
      .map(log => ({
        name: log.exercise.name,
        muscle: log.exercise.muscle,
        sets: log.sets.filter(s => s.completed),
      }))
      .filter(ex => ex.sets.length > 0),
  };
}

export default function WorkoutCompleteScreen({ route, navigation }: Props) {
  const { logs, program, durationSeconds, weekNumber } = route.params;
  const { bottom } = useSafeAreaInsets();

  const record = buildRecord(logs, program, durationSeconds, weekNumber);
  const totalSets = record.exercises.reduce((n, ex) => n + ex.sets.length, 0);
  const totalVolume = record.exercises.reduce(
    (n, ex) => n + ex.sets.reduce((s, set) => s + set.weight * set.reps, 0),
    0,
  );

  useEffect(() => {
    const save = async () => {
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      const history: WorkoutRecord[] = raw ? JSON.parse(raw) : [];
      history.unshift(record);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      console.log('[WorkoutComplete] saved record:', JSON.stringify(record, null, 2));

      const [lastRaw, secondLastRaw] = await Promise.all([
        AsyncStorage.getItem(`last_session_${program.id}`),
        AsyncStorage.getItem(`second_last_session_${program.id}`),
      ]);
      const lastSession: Record<string, LastSessionEntry> = lastRaw ? JSON.parse(lastRaw) : {};
      const newLastSession: Record<string, LastSessionEntry> = { ...lastSession };

      for (const log of logs) {
        const completed = log.sets.filter(s => s.completed);
        if (completed.length === 0) continue;

        const avgReps = completed.reduce((sum, s) => sum + s.reps, 0) / completed.length;
        const avgRpe  = completed.reduce((sum, s) => sum + s.rpe, 0)  / completed.length;
        const bestSet = completed.reduce((best, s) =>
          epleyOneRm(s.weight, s.reps) > epleyOneRm(best.weight, best.reps) ? s : best,
          completed[0],
        );
        const isBodyweight = bestSet.weight === 0;
        const oneRm = isBodyweight ? 0 : tuchschererOneRm(bestSet.weight, bestSet.reps, bestSet.rpe);

        const prev = lastSession[log.exercise.name];
        if (!isBodyweight && weekNumber >= 2 && prev && prev.oneRm > 0) {
          logWorkout({
            user_id: 'local',
            exercise: log.exercise.name,
            one_rm: prev.oneRm,
            week: weekNumber,
            day: log.dayNumber,
            program_length: program.lengthWeeks,
            time_per_workout: program.timePerWorkout,
            number_of_exercises: logs.length,
            weeks_gap: 1,
            lag_sets: prev.sets,
            lag_reps: prev.reps,
            lag_rpe:  prev.rpe,
            sets: completed.length,
            reps: avgReps,
            rpe:  avgRpe,
            ...levelFlags(program),
            ...goalFlags(program),
            ...equipmentFlags(program),
          }).catch(() => {});
        }

        newLastSession[log.exercise.name] = { sets: completed.length, reps: avgReps, rpe: avgRpe, oneRm, weight: bestSet.weight };
      }

      await AsyncStorage.setItem(`third_last_session_${program.id}`, secondLastRaw ?? '{}');
      await AsyncStorage.setItem(`second_last_session_${program.id}`, JSON.stringify(lastSession));
      await AsyncStorage.setItem(`last_session_${program.id}`, JSON.stringify(newLastSession));

      const dayCountKey = `day_count_${program.id}_${record.dayNumber}`;
      const prevCount = await AsyncStorage.getItem(dayCountKey);
      await AsyncStorage.setItem(dayCountKey, String((prevCount ? parseInt(prevCount) : 0) + 1));
    };
    save();
  }, []);

  return (
    <View className="flex-1 bg-background">
      <ScrollView contentContainerStyle={{ padding: 24, gap: 16 }}>
        <Text className="text-5xl text-center text-accent">✓</Text>
        <Text className="text-28 font-outfit-bold text-accent text-center">Workout Complete</Text>
        <Text className="text-13 font-outfit text-secondary text-center mb-2">{program.name} · Day {record.dayNumber}</Text>

        <View className="flex-row bg-surface rounded-xl border border-border p-5 items-center">
          <View className="flex-1 items-center gap-1">
            <Text className="text-lg font-outfit text-primary font-bold">{formatDuration(durationSeconds)}</Text>
            <Text className="text-xs font-outfit text-secondary">Duration</Text>
          </View>
          <View className="w-px h-9 bg-border" />
          <View className="flex-1 items-center gap-1">
            <Text className="text-lg font-outfit text-primary font-bold">{totalSets}</Text>
            <Text className="text-xs font-outfit text-secondary">Sets</Text>
          </View>
          <View className="w-px h-9 bg-border" />
          <View className="flex-1 items-center gap-1">
            <Text className="text-lg font-outfit text-primary font-bold">{Math.round(totalVolume).toLocaleString()} kg</Text>
            <Text className="text-xs font-outfit text-secondary">Volume</Text>
          </View>
        </View>

        {record.exercises.map((ex, i) => (
          <View key={i} className="bg-surface rounded-[10px] border border-border p-3.5 gap-2">
            <View className="flex-row justify-between items-center">
              <Text className="text-base font-outfit text-primary font-semibold flex-1">{ex.name}</Text>
              <Text className="text-xs font-outfit text-secondary">{ex.muscle}</Text>
            </View>
            {ex.sets.map((set, si) => (
              <View key={si} className="flex-row items-center gap-2">
                <Text className="text-13 font-outfit text-secondary w-11">Set {si + 1}</Text>
                <Text className="text-13 font-outfit text-secondary flex-1 text-center">{set.weight} kg</Text>
                <Text className="text-13 font-outfit text-secondary flex-1 text-center">{set.reps} reps</Text>
                <Text className="text-13 font-outfit text-secondary flex-1 text-center">RPE {set.rpe}</Text>
              </View>
            ))}
          </View>
        ))}
      </ScrollView>

      <View className="flex-row gap-2.5 p-4" style={{ paddingBottom: bottom || 16 }}>
        <Pressable
          className="flex-1 p-3.5 rounded-[10px] border border-border items-center"
          onPress={() => navigation.navigate('WorkoutHistory')}
        >
          <Text className="text-base font-outfit text-primary text-sm">View History</Text>
        </Pressable>
        <Pressable
          className="flex-1 p-3.5 rounded-[10px] bg-accent items-center"
          onPress={() => navigation.popToTop()}
        >
          <Text className="text-background font-bold text-sm">Done</Text>
        </Pressable>
      </View>
    </View>
  );
}
