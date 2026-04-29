import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import Slider from '@react-native-community/slider';
import { useEffect, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { colors, typography } from '../theme';
import { ExerciseLog, ProgramExercise, RootStackParamList, WorkoutDay } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'ActiveWorkout'>;

const RPE_LABELS: Record<number, string> = {
  6: 'Very Easy', 6.5: 'Easy', 7: 'Moderate', 7.5: 'Somewhat Hard',
  8: 'Hard', 8.5: 'Very Hard', 9: 'Very Hard+', 9.5: 'Near Max', 10: 'Max Effort',
};

export default function ActiveWorkoutScreen({ route, navigation }: Props) {
  const { program } = route.params;
  const { bottom } = useSafeAreaInsets();
  const [selectedDay, setSelectedDay] = useState<WorkoutDay | null>(null);
  const [exerciseIndex, setExerciseIndex] = useState(0);
  const [logs, setLogs] = useState<ExerciseLog[]>([]);
  const [weekNumber, setWeekNumber] = useState(1);
  const [currentSetIndex, setCurrentSetIndex] = useState(0);
  const [currentWeight, setCurrentWeight] = useState('');
  const [currentReps, setCurrentReps] = useState(0);
  const [currentRpe, setCurrentRpe] = useState(8);

  useEffect(() => {
    AsyncStorage.getItem(`week_${program.id}`).then(val => {
      if (val) setWeekNumber(parseInt(val));
    });
  }, []);

  useEffect(() => {
    if (!selectedDay) {
      navigation.setOptions({ headerLeft: undefined });
      return;
    }
    navigation.setOptions({
      headerLeft: () => (
        <Pressable
          style={{ paddingRight: 12 }}
          onPress={() => Alert.alert('Exit Workout?', 'Your progress will be lost.', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Exit', style: 'destructive', onPress: () => navigation.goBack() },
          ])}
        >
          <Text style={styles.exitBtn}>Exit</Text>
        </Pressable>
      ),
    });
  }, [selectedDay, navigation]);

  // Reset current-set state when moving to a new exercise
  useEffect(() => {
    if (!logs.length) return;
    setCurrentSetIndex(0);
    const set = logs[exerciseIndex]?.sets[0];
    if (set) {
      setCurrentReps(set.reps);
      setCurrentRpe(set.rpe);
      setCurrentWeight('');
    }
  }, [exerciseIndex]);

  if (!selectedDay) {
    return (
      <View style={styles.container}>
        <Text style={styles.heading}>Which day?</Text>
        <ScrollView contentContainerStyle={[styles.dayGrid, { paddingBottom: bottom + 16 }]}>
          {program.days.map((day, i) => (
            <Pressable
              key={i}
              style={styles.dayCard}
              onPress={() => {
                const newLogs: ExerciseLog[] = day.exercises.map(ex => ({
                  exercise: ex,
                  dayNumber: day.dayNumber,
                  sets: Array.from({ length: ex.sets }, () => ({ reps: ex.reps, rpe: 8, weight: 0 })),
                }));
                setLogs(newLogs);
                setExerciseIndex(0);
                setCurrentSetIndex(0);
                const first = newLogs[0]?.sets[0];
                if (first) { setCurrentReps(first.reps); setCurrentRpe(first.rpe); setCurrentWeight(''); }
                setSelectedDay(day);
              }}
            >
              <Text style={styles.dayCardTitle}>Day {day.dayNumber}</Text>
              <View style={styles.dayExerciseList}>
                {day.exercises.map(ex => (
                  <View key={ex.id} style={styles.dayExerciseRow}>
                    <Text style={styles.dayExerciseName}>{ex.name}</Text>
                    <Text style={styles.dayExerciseSets}>{ex.sets} × {ex.reps}</Text>
                  </View>
                ))}
              </View>
            </Pressable>
          ))}
        </ScrollView>
      </View>
    );
  }

  const currentLog = logs[exerciseIndex];
  const ex: ProgramExercise = currentLog.exercise;
  const isLast = exerciseIndex === logs.length - 1;
  const allSetsComplete = currentSetIndex >= currentLog.sets.length;

  const completeSet = () => {
    const weight = parseFloat(currentWeight) || 0;
    const nextIdx = currentSetIndex + 1;
    setLogs(prev => prev.map((log, i) => {
      if (i !== exerciseIndex) return log;
      const sets = log.sets.map((s, si) =>
        si === currentSetIndex ? { reps: currentReps, rpe: currentRpe, weight } : s
      );
      return { ...log, sets };
    }));
    if (nextIdx < currentLog.sets.length) {
      const next = currentLog.sets[nextIdx];
      setCurrentReps(next.reps);
      setCurrentRpe(next.rpe);
      setCurrentWeight('');
    }
    setCurrentSetIndex(nextIdx);
  };

  return (
    <View style={styles.container}>
      <View style={styles.exerciseHeader}>
        <Text style={styles.exerciseCounter}>{exerciseIndex + 1} / {logs.length}</Text>
        <Text style={styles.exerciseName}>{ex.name}</Text>
        <Text style={styles.exerciseMuscle}>{ex.muscle}</Text>
      </View>

      <View style={styles.targetRow}>
        <Text style={styles.targetLabel}>Target</Text>
        <Text style={styles.targetValue}>{ex.sets} × {ex.reps}</Text>
        <Text style={styles.targetSep}>·</Text>
        <Text style={styles.targetLabel}>Week {weekNumber}</Text>
        <Text style={styles.targetWeight}>— kg</Text>
      </View>

      <ScrollView contentContainerStyle={styles.setList} keyboardShouldPersistTaps="handled">
        {currentLog.sets.slice(0, currentSetIndex).map((set, si) => (
          <View key={si} style={styles.completedSet}>
            <Text style={styles.completedSetNum}>Set {si + 1}</Text>
            <Text style={styles.completedSetDetail}>{set.weight > 0 ? `${set.weight} kg` : '— kg'}</Text>
            <Text style={styles.completedSetDetail}>{set.reps} reps</Text>
            <Text style={styles.completedSetDetail}>RPE {set.rpe}</Text>
          </View>
        ))}

        {!allSetsComplete && (
          <View style={styles.activeSet}>
            <Text style={styles.activeSetNum}>Set {currentSetIndex + 1} of {currentLog.sets.length}</Text>

            <View style={styles.inputRow}>
              <Text style={styles.inputLabel}>Weight (kg)</Text>
              <TextInput
                style={styles.weightInput}
                value={currentWeight}
                onChangeText={setCurrentWeight}
                keyboardType="numeric"
                placeholder="—"
                placeholderTextColor={colors.secondary}
                returnKeyType="done"
              />
            </View>

            <View style={styles.inputRow}>
              <Text style={styles.inputLabel}>Reps</Text>
              <View style={styles.repsStepper}>
                <Pressable style={styles.stepBtn} onPress={() => setCurrentReps(r => Math.max(1, r - 1))}>
                  <Text style={styles.stepBtnText}>−</Text>
                </Pressable>
                <Text style={styles.repsValue}>{currentReps}</Text>
                <Pressable style={styles.stepBtn} onPress={() => setCurrentReps(r => r + 1)}>
                  <Text style={styles.stepBtnText}>+</Text>
                </Pressable>
              </View>
            </View>

            <View style={styles.rpeSection}>
              <View style={styles.rpeLabelRow}>
                <Text style={styles.inputLabel}>RPE</Text>
                <Text style={styles.rpeValue}>{currentRpe} — {RPE_LABELS[currentRpe]}</Text>
              </View>
              <Text style={styles.rpeHint}>Rate of Perceived Exertion · 6 = easy · 10 = max effort</Text>
              <Slider
                style={styles.slider}
                minimumValue={6}
                maximumValue={10}
                step={0.5}
                value={currentRpe}
                onValueChange={v => setCurrentRpe(v)}
                minimumTrackTintColor={colors.accent}
                maximumTrackTintColor={colors.border}
                thumbTintColor={colors.accent}
              />
            </View>
          </View>
        )}
      </ScrollView>

      <View style={[styles.footer, { paddingBottom: bottom || 16 }]}>
        {allSetsComplete ? (
          <>
            {exerciseIndex > 0 && (
              <Pressable style={styles.backBtn} onPress={() => setExerciseIndex(i => i - 1)}>
                <Text style={styles.backBtnText}>← Back</Text>
              </Pressable>
            )}
            <Pressable
              style={[styles.nextBtn, isLast && styles.finishBtn]}
              onPress={() => isLast
                ? navigation.replace('WorkoutComplete', { logs, program })
                : setExerciseIndex(i => i + 1)
              }
            >
              <Text style={[styles.nextBtnText, isLast && { color: colors.background }]}>
                {isLast ? 'Finish Workout' : 'Next →'}
              </Text>
            </Pressable>
          </>
        ) : (
          <Pressable style={styles.completeSetBtn} onPress={completeSet}>
            <Text style={styles.completeSetBtnText}>✓  Complete Set {currentSetIndex + 1}</Text>
          </Pressable>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },

  // day picker
  heading: { ...typography.heading, paddingHorizontal: 20, paddingTop: 32, marginBottom: 24 },
  dayGrid: { paddingHorizontal: 16, gap: 12 },
  dayCard: { backgroundColor: colors.surface, borderRadius: 10, padding: 20, borderWidth: 1, borderColor: colors.border },
  dayCardTitle: { ...typography.body, fontWeight: '700', marginBottom: 10 },
  dayExerciseList: { gap: 6 },
  dayExerciseRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  dayExerciseName: { ...typography.caption, flex: 1 },
  dayExerciseSets: { ...typography.caption, color: colors.secondary },

  // header
  exitBtn: { color: colors.secondary, fontSize: 15 },
  exerciseHeader: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 16, borderBottomWidth: 1, borderBottomColor: colors.border },
  exerciseCounter: { ...typography.caption, marginBottom: 4 },
  exerciseName: { ...typography.heading, fontSize: 24, marginBottom: 2 },
  exerciseMuscle: { ...typography.caption },

  // target
  targetRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 20, paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: colors.border },
  targetLabel: { ...typography.caption },
  targetValue: { ...typography.body, fontWeight: '700' },
  targetSep: { ...typography.caption },
  targetWeight: { ...typography.body, color: colors.secondary },

  // set list
  setList: { padding: 16, gap: 12 },

  // completed sets
  completedSet: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.surface, borderRadius: 8, padding: 12, gap: 8, borderWidth: 1, borderColor: colors.border, opacity: 0.7 },
  completedSetNum: { ...typography.caption, width: 44 },
  completedSetDetail: { ...typography.caption, flex: 1, textAlign: 'center' },

  // active set card
  activeSet: { backgroundColor: colors.surface, borderRadius: 10, padding: 16, gap: 20, borderWidth: 1, borderColor: colors.accent },
  activeSetNum: { ...typography.caption, color: colors.accent, fontWeight: '600' },

  // inputs
  inputRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  inputLabel: { ...typography.caption },
  weightInput: {
    width: 100, height: 44, borderWidth: 1, borderColor: colors.border, borderRadius: 8,
    backgroundColor: colors.background, color: colors.primary,
    textAlign: 'center', fontSize: 18, fontWeight: '700' as const,
  },

  // reps stepper
  repsStepper: { flexDirection: 'row', alignItems: 'center' },
  stepBtn: { width: 44, height: 44, backgroundColor: colors.background, borderWidth: 1, borderColor: colors.border, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
  stepBtnText: { color: colors.accent, fontSize: 20, fontWeight: '300' as const },
  repsValue: { width: 56, textAlign: 'center' as const, ...typography.heading, fontSize: 22 },

  // RPE
  rpeSection: { gap: 4 },
  rpeLabelRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  rpeValue: { ...typography.caption, color: colors.primary },
  rpeHint: { ...typography.caption, fontSize: 11, lineHeight: 16 },
  slider: { width: '100%' as const, height: 40 },

  // footer
  footer: { flexDirection: 'row', gap: 10, padding: 16 },
  completeSetBtn: { flex: 1, padding: 16, borderRadius: 10, backgroundColor: colors.accent, alignItems: 'center' },
  completeSetBtnText: { color: colors.background, fontSize: 15, fontWeight: '700' as const },
  backBtn: { flex: 1, padding: 16, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' },
  backBtnText: { color: colors.primary, fontSize: 15 },
  nextBtn: { flex: 2, padding: 16, borderRadius: 10, backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border, alignItems: 'center' },
  finishBtn: { backgroundColor: colors.accent, borderColor: colors.accent },
  nextBtnText: { color: colors.accent, fontSize: 15, fontWeight: '700' as const },
});
