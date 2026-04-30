import { Alert, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import Slider from '@react-native-community/slider';
import { useEffect, useRef, useState } from 'react';
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

function WorkoutTimer() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setElapsed(s => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss = String(elapsed % 60).padStart(2, '0');
  return <Text style={{ color: colors.secondary, fontSize: 14 }}>Workout · {mm}:{ss}</Text>;
}

export default function ActiveWorkoutScreen({ route, navigation }: Props) {
  const { program } = route.params;
  const { bottom } = useSafeAreaInsets();
  const [selectedDay, setSelectedDay] = useState<WorkoutDay | null>(null);
  const [exerciseIndex, setExerciseIndex] = useState(0);
  const [logs, setLogs] = useState<ExerciseLog[]>([]);
  const [weekNumber, setWeekNumber] = useState(1);
  const [currentWeight, setCurrentWeight] = useState('');
  const [currentReps, setCurrentReps] = useState(0);
  const [currentRpe, setCurrentRpe] = useState(8);
  const [editingSetIndex, setEditingSetIndex] = useState<number | null>(null);
  const [editWeight, setEditWeight] = useState('');
  const [editReps, setEditReps] = useState(0);
  const [editRpe, setEditRpe] = useState(8);
  const isFinishing = useRef(false);
  const startTime = useRef(0);

  useEffect(() => {
    AsyncStorage.getItem(`program_start_${program.id}`).then(val => {
      if (val) {
        const elapsed = (Date.now() - new Date(val).getTime()) / 86400000;
        const week = Math.min(Math.floor(elapsed / 7) + 1, program.lengthWeeks);
        setWeekNumber(week);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedDay) {
      navigation.setOptions({ headerLeft: undefined });
      return;
    }
    navigation.setOptions({
      headerBackVisible: false,
      headerLeft: () => null,
      headerTitle: () => <WorkoutTimer />,
      headerRight: () => (
        <Pressable
          style={{ paddingLeft: 12 }}
          onPress={() => Alert.alert('Exit Workout?', 'Your progress will be lost.', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Exit', style: 'destructive', onPress: () => { isFinishing.current = true; navigation.goBack(); } },
          ])}
        >
          <Text style={styles.exitBtn}>✕</Text>
        </Pressable>
      ),
    });
  }, [selectedDay, navigation]);


  // Intercept all back gestures and hardware back button during a workout
  useEffect(() => {
    if (!selectedDay) return;
    return navigation.addListener('beforeRemove', e => {
      if (isFinishing.current) return;
      e.preventDefault();
      Alert.alert('Exit Workout?', 'Your progress will be lost.', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Exit', style: 'destructive', onPress: () => navigation.dispatch(e.data.action) },
      ]);
    });
  }, [selectedDay, navigation]);

  // Reset input state when moving to a new exercise
  useEffect(() => {
    if (!logs.length) return;
    const idx = logs[exerciseIndex]?.sets.filter(s => s.weight > 0).length ?? 0;
    const set = logs[exerciseIndex]?.sets[idx];
    setEditingSetIndex(null);
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
                  sets: Array.from({ length: ex.sets }, () => ({ reps: ex.reps, rpe: 8, weight: 0, completed: false })),
                }));
                startTime.current = Date.now();
                setLogs(newLogs);
                setExerciseIndex(0);
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
  const currentSetIndex = currentLog.sets.filter(s => s.completed).length;
  const allSetsComplete = currentSetIndex >= currentLog.sets.length;

  const openEditSet = (si: number) => {
    const set = currentLog.sets[si];
    setEditingSetIndex(si);
    setEditWeight(set.weight > 0 ? String(set.weight) : '');
    setEditReps(set.reps);
    setEditRpe(set.rpe);
  };

  const saveEditedSet = () => {
    if (editingSetIndex === null) return;
    const weight = parseFloat(editWeight) || 0;
    setLogs(prev => prev.map((log, i) => {
      if (i !== exerciseIndex) return log;
      const sets = log.sets.map((s, si) =>
        si === editingSetIndex ? { reps: editReps, rpe: editRpe, weight, completed: true } : s
      );
      return { ...log, sets };
    }));
    setEditingSetIndex(null);
  };

  const completeSet = () => {
    const weight = parseFloat(currentWeight) || 0;
    const nextIdx = currentSetIndex + 1;
    setLogs(prev => prev.map((log, i) => {
      if (i !== exerciseIndex) return log;
      const sets = log.sets.map((s, si) =>
        si === currentSetIndex ? { reps: currentReps, rpe: currentRpe, weight, completed: true } : s
      );
      return { ...log, sets };
    }));
    if (nextIdx < currentLog.sets.length) {
      const next = currentLog.sets[nextIdx];
      setCurrentReps(next.reps);
      setCurrentRpe(next.rpe);
      setCurrentWeight(currentWeight); // carry weight from previous set
    }
  };

  const workoutComplete = logs.every(log => log.sets.every(s => s.completed));

  const finishWorkout = () => {
    const doFinish = () => {
      isFinishing.current = true;
      const durationSeconds = Math.floor((Date.now() - startTime.current) / 1000);
      navigation.replace('WorkoutComplete', { logs, program, durationSeconds, weekNumber });
    };
    if (workoutComplete) {
      doFinish();
    } else {
      Alert.alert(
        'Finish early?',
        "Some sets haven't been completed. Finish anyway?",
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Finish', style: 'destructive', onPress: doFinish },
        ]
      );
    }
  };

  return (
    <View style={styles.container}>
      {/* Tab bar — tap any exercise to jump to it */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.tabBar}
        contentContainerStyle={styles.tabBarContent}
      >
        {logs.map((log, i) => {
          const done = log.sets.every(s => s.completed);
          const active = i === exerciseIndex;
          return (
            <Pressable
              key={i}
              style={[styles.tab, active && styles.tabActive]}
              onPress={() => setExerciseIndex(i)}
            >
              <Text style={[styles.tabText, active && styles.tabTextActive]}>
                {done ? '✓ ' : ''}{log.exercise.name}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>

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

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <ScrollView contentContainerStyle={styles.setList} keyboardShouldPersistTaps="handled">
        {currentLog.sets.slice(0, currentSetIndex).map((set, si) =>
          editingSetIndex === si ? (
            <View key={si} style={styles.activeSet}>
              <View style={styles.editSetHeader}>
                <Text style={styles.activeSetNum}>Editing Set {si + 1}</Text>
                <Pressable onPress={() => setEditingSetIndex(null)}>
                  <Text style={styles.cancelEditBtn}>Cancel</Text>
                </Pressable>
              </View>

              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Weight (kg)</Text>
                <TextInput
                  style={styles.weightInput}
                  value={editWeight}
                  onChangeText={setEditWeight}
                  keyboardType="numeric"
                  placeholder="—"
                  placeholderTextColor={colors.secondary}
                  returnKeyType="done"
                  autoFocus
                />
              </View>

              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Reps</Text>
                <View style={styles.repsStepper}>
                  <Pressable style={styles.stepBtn} onPress={() => setEditReps(r => Math.max(1, r - 1))}>
                    <Text style={styles.stepBtnText}>−</Text>
                  </Pressable>
                  <Text style={styles.repsValue}>{editReps}</Text>
                  <Pressable style={styles.stepBtn} onPress={() => setEditReps(r => r + 1)}>
                    <Text style={styles.stepBtnText}>+</Text>
                  </Pressable>
                </View>
              </View>

              <View style={styles.rpeSection}>
                <View style={styles.rpeLabelRow}>
                  <Text style={styles.inputLabel}>RPE</Text>
                  <Text style={styles.rpeValue}>{editRpe} — {RPE_LABELS[editRpe]}</Text>
                </View>
                <Slider
                  style={styles.slider}
                  minimumValue={6}
                  maximumValue={10}
                  step={0.5}
                  value={editRpe}
                  onValueChange={v => setEditRpe(v)}
                  minimumTrackTintColor={colors.accent}
                  maximumTrackTintColor={colors.border}
                  thumbTintColor={colors.accent}
                />
              </View>

              <Pressable style={styles.saveEditBtn} onPress={saveEditedSet}>
                <Text style={styles.saveEditBtnText}>Save Changes</Text>
              </Pressable>
            </View>
          ) : (
            <Pressable key={si} style={styles.completedSet} onPress={() => openEditSet(si)}>
              <Text style={styles.completedSetNum}>Set {si + 1}</Text>
              <Text style={styles.completedSetDetail}>{set.weight > 0 ? `${set.weight} kg` : 'BW'}</Text>
              <Text style={styles.completedSetDetail}>{set.reps} reps</Text>
              <Text style={styles.completedSetDetail}>RPE {set.rpe}</Text>
              <Text style={styles.editHint}>✎</Text>
            </Pressable>
          )
        )}

        {!allSetsComplete && editingSetIndex === null && (
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
        {!allSetsComplete && editingSetIndex === null && (
          <Pressable style={styles.completeSetBtn} onPress={completeSet}>
            <Text style={styles.completeSetBtnText}>✓  Complete Set {currentSetIndex + 1}</Text>
          </Pressable>
        )}
        <View style={styles.footerRow}>
          {!allSetsComplete && editingSetIndex === null && (
            <Pressable style={styles.skipBtn} onPress={() => setExerciseIndex(i => isLast ? i : i + 1)}>
              <Text style={styles.skipBtnText}>Skip</Text>
            </Pressable>
          )}
          <Pressable style={[styles.finishBtn, allSetsComplete && styles.finishBtnComplete]} onPress={finishWorkout}>
            <Text style={[styles.finishBtnText, allSetsComplete && styles.finishBtnTextComplete]}>Finish Workout</Text>
          </Pressable>
        </View>
      </View>
      </KeyboardAvoidingView>
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

  // tab bar
  tabBar: { borderBottomWidth: 1, borderBottomColor: colors.border, flexGrow: 0 },
  tabBarContent: { paddingHorizontal: 12, paddingVertical: 8, gap: 8 },
  tab: { minWidth: 80, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.surface },
  tabActive: { borderColor: colors.accent, backgroundColor: colors.accent + '18' },
  tabText: { ...typography.caption, fontSize: 12 },
  tabTextActive: { color: colors.accent, fontWeight: '600' as const },

  // header
  exitBtn: { color: colors.secondary, fontSize: 15 },
  exerciseHeader: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 16, borderBottomWidth: 1, borderBottomColor: colors.border },
  exerciseHeaderTop: { flexDirection: 'row' as const, justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  exerciseCounter: { ...typography.caption },
  timerText: { ...typography.caption, fontWeight: '600' as const, color: colors.primary },
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
  editHint: { ...typography.caption, color: colors.secondary, fontSize: 14 },

  // inline set editor
  editSetHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  cancelEditBtn: { ...typography.caption, color: colors.secondary },
  saveEditBtn: { backgroundColor: colors.accent, borderRadius: 8, padding: 12, alignItems: 'center' },
  saveEditBtnText: { color: colors.background, fontWeight: '700' as const, fontSize: 14 },

  // active set card
  activeSet: { backgroundColor: colors.surface, borderRadius: 10, padding: 16, gap: 20, borderWidth: 1, borderColor: colors.accent },
  activeSetNum: { ...typography.caption, color: colors.accent, fontWeight: '600' },

  // inputs
  inputRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  inputLabel: { ...typography.caption, fontSize: 16 },
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
  rpeValue: { ...typography.caption, fontSize: 16, color: colors.primary },
  rpeHint: { ...typography.caption, fontSize: 13, lineHeight: 18 },
  slider: { width: '100%' as const, height: 40 },

  // footer
  footer: { gap: 8, padding: 16 },
  footerRow: { flexDirection: 'row' as const, gap: 8 },
  completeSetBtn: { padding: 16, borderRadius: 10, backgroundColor: colors.accent, alignItems: 'center' as const },
  completeSetBtnText: { color: colors.background, fontSize: 15, fontWeight: '700' as const },
  skipBtn: { flex: 1, padding: 12, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' as const },
  skipBtnText: { color: colors.secondary, fontSize: 14 },
  finishBtn: { flex: 1, padding: 14, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' as const },
  finishBtnText: { color: colors.primary, fontSize: 14, fontWeight: '600' as const },
  finishBtnComplete: { backgroundColor: colors.accent, borderColor: colors.accent },
  finishBtnTextComplete: { color: colors.background },
  nextBtnText: { color: colors.accent, fontSize: 15, fontWeight: '700' as const },
});
