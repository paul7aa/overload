import { useEffect } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, typography } from '../theme';
import { ExerciseLog, LastSessionEntry, RootStackParamList, WorkoutRecord } from '../types';
import { epleyOneRm, equipmentFlags, goalFlags, levelFlags, logWorkout } from '../api/client';

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
      // Save to workout history
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      const history: WorkoutRecord[] = raw ? JSON.parse(raw) : [];
      history.unshift(record);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));

      // Load previous session data for lag values
      const lastRaw = await AsyncStorage.getItem(`last_session_${program.id}`);
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
        const oneRm = isBodyweight ? 0 : epleyOneRm(bestSet.weight, bestSet.reps);

        // POST /log from week 2 onward, when lag data exists and exercise has weight
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
          }).catch(() => {}); // fire-and-forget
        }

        newLastSession[log.exercise.name] = { sets: completed.length, reps: avgReps, rpe: avgRpe, oneRm };
      }

      await AsyncStorage.setItem(`last_session_${program.id}`, JSON.stringify(newLastSession));

      // Record program start date on first completion (used to derive week number)
      const startKey = `program_start_${program.id}`;
      const existingStart = await AsyncStorage.getItem(startKey);
      if (!existingStart) {
        await AsyncStorage.setItem(startKey, new Date().toISOString());
      }
    };
    save();
  }, []);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.check}>✓</Text>
        <Text style={styles.title}>Workout Complete</Text>
        <Text style={styles.subtitle}>{program.name} · Day {record.dayNumber}</Text>

        <View style={styles.statsRow}>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{formatDuration(durationSeconds)}</Text>
            <Text style={styles.statLabel}>Duration</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.stat}>
            <Text style={styles.statValue}>{totalSets}</Text>
            <Text style={styles.statLabel}>Sets</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.stat}>
            <Text style={styles.statValue}>{Math.round(totalVolume).toLocaleString()} kg</Text>
            <Text style={styles.statLabel}>Volume</Text>
          </View>
        </View>

        {record.exercises.map((ex, i) => (
          <View key={i} style={styles.exerciseCard}>
            <View style={styles.exerciseHeader}>
              <Text style={styles.exerciseName}>{ex.name}</Text>
              <Text style={styles.exerciseMuscle}>{ex.muscle}</Text>
            </View>
            {ex.sets.map((set, si) => (
              <View key={si} style={styles.setRow}>
                <Text style={styles.setNum}>Set {si + 1}</Text>
                <Text style={styles.setDetail}>{set.weight} kg</Text>
                <Text style={styles.setDetail}>{set.reps} reps</Text>
                <Text style={styles.setDetail}>RPE {set.rpe}</Text>
              </View>
            ))}
          </View>
        ))}
      </ScrollView>

      <View style={[styles.footer, { paddingBottom: bottom || 16 }]}>
        <Pressable style={styles.historyBtn} onPress={() => navigation.navigate('WorkoutHistory')}>
          <Text style={styles.historyBtnText}>View History</Text>
        </Pressable>
        <Pressable style={styles.homeBtn} onPress={() => navigation.popToTop()}>
          <Text style={styles.homeBtnText}>Done</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  scroll: { padding: 24, gap: 16 },

  check: { fontSize: 48, textAlign: 'center', color: colors.accent },
  title: { ...typography.heading, textAlign: 'center', fontSize: 28 },
  subtitle: { ...typography.caption, textAlign: 'center', marginBottom: 8 },

  statsRow: {
    flexDirection: 'row', backgroundColor: colors.surface,
    borderRadius: 12, borderWidth: 1, borderColor: colors.border,
    padding: 20, alignItems: 'center',
  },
  stat: { flex: 1, alignItems: 'center', gap: 4 },
  statValue: { ...typography.body, fontSize: 18, fontWeight: '700' as const },
  statLabel: { ...typography.caption, fontSize: 12 },
  statDivider: { width: 1, height: 36, backgroundColor: colors.border },

  exerciseCard: {
    backgroundColor: colors.surface, borderRadius: 10,
    borderWidth: 1, borderColor: colors.border, padding: 14, gap: 8,
  },
  exerciseHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  exerciseName: { ...typography.body, fontWeight: '600' as const, flex: 1 },
  exerciseMuscle: { ...typography.caption, fontSize: 12 },

  setRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  setNum: { ...typography.caption, width: 44, fontSize: 12 },
  setDetail: { ...typography.caption, flex: 1, textAlign: 'center', fontSize: 13 },

  footer: { flexDirection: 'row', gap: 10, padding: 16 },
  historyBtn: {
    flex: 1, padding: 14, borderRadius: 10,
    borderWidth: 1, borderColor: colors.border, alignItems: 'center',
  },
  historyBtnText: { ...typography.body, fontSize: 14 },
  homeBtn: {
    flex: 1, padding: 14, borderRadius: 10,
    backgroundColor: colors.accent, alignItems: 'center',
  },
  homeBtnText: { color: colors.background, fontWeight: '700' as const, fontSize: 14 },
});