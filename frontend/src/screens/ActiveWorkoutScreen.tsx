import { ActivityIndicator, Alert, Animated, Dimensions, FlatList, Image, KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import Slider from '@react-native-community/slider';
import { useEffect, useRef, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { colors, typography } from '../theme';
import { Exercise, ExerciseLog, LastSessionEntry, ProgramExercise, RootStackParamList, WorkoutRecord, WorkoutDay } from '../types';
import { HISTORY_KEY } from './WorkoutCompleteScreen';
import { equipmentFlags, ExerciseInfo, fetchExerciseInfo, goalFlags, levelFlags, predict, PredictResponse } from '../api/client';
import EXERCISES from '../data/exercises.json';

type Props = NativeStackScreenProps<RootStackParamList, 'ActiveWorkout'>;

const RPE_LABELS: Record<number, string> = {
  6: 'Very Easy', 6.5: 'Easy', 7: 'Moderate', 7.5: 'Somewhat Hard',
  8: 'Hard', 8.5: 'Very Hard', 9: 'Very Hard+', 9.5: 'Near Max', 10: 'Max Effort',
};

const RPE_HINTS: Record<number, string> = {
  6:   'Very easy — 4+ reps left in the tank',
  6.5: 'Easy — could definitely do 4 more',
  7:   'Comfortable — 3 reps left, form is solid',
  7.5: 'Getting there — could do 2–3 more',
  8:   'Hard — 2 reps left, starting to grind',
  8.5: 'Very hard — maybe 1–2 more, form holding',
  9:   'Near max — 1 rep left, form may break',
  9.5: 'Almost max — could squeeze out 1, barely',
  10:  'True max — couldn\'t do another rep',
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
  const [lastDayNumber, setLastDayNumber] = useState<number | null>(null);
  const [pickedDay, setPickedDay] = useState<WorkoutDay | null>(null);
  const [predictions, setPredictions] = useState<Record<string, PredictResponse>>({});
  const [predictionsStatus, setPredictionsStatus] = useState<'idle' | 'loading' | 'failed'>('idle');
  const [swapModalVisible, setSwapModalVisible] = useState(false);
  const [swapQuery, setSwapQuery] = useState('');
  const [infoModalVisible, setInfoModalVisible] = useState(false);
  const [exerciseInfo, setExerciseInfo] = useState<ExerciseInfo | null>(null);
  const lastSessionRef = useRef<Record<string, LastSessionEntry>>({});
  const isFinishing = useRef(false);
  const startTime = useRef(0);
  const slideAnim = useRef(new Animated.Value(Dimensions.get('window').width)).current;

  useEffect(() => {
    AsyncStorage.getItem(HISTORY_KEY).then(raw => {
      const history: WorkoutRecord[] = raw ? JSON.parse(raw) : [];
      const last = history.find(r => r.programId === program.id);
      if (last) setLastDayNumber(last.dayNumber);
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

  const startWorkout = async (day: WorkoutDay) => {
    const [dayCountRaw, lastSessionRaw] = await Promise.all([
      AsyncStorage.getItem(`day_count_${program.id}_${day.dayNumber}`),
      AsyncStorage.getItem(`last_session_${program.id}`),
    ]);
    const week = (dayCountRaw ? parseInt(dayCountRaw) : 0) + 1;
    setWeekNumber(week);

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
    slideAnim.setValue(Dimensions.get('window').width);
    setSelectedDay(day);
    Animated.timing(slideAnim, { toValue: 0, duration: 320, useNativeDriver: true }).start();

    setPredictions({});
    setPredictionsStatus('idle');

    const lastSession: Record<string, LastSessionEntry> = lastSessionRaw ? JSON.parse(lastSessionRaw) : {};
    lastSessionRef.current = lastSession;
    const eligibleExercises = day.exercises.filter(ex => lastSession[ex.name]?.oneRm > 0);
    if (eligibleExercises.length === 0) return;

    setPredictionsStatus('loading');
    const flags = { ...levelFlags(program), ...goalFlags(program), ...equipmentFlags(program) };
    const results = await Promise.all(
      eligibleExercises.map(ex => {
        const prev = lastSession[ex.name];
        return predict({
          exercise: ex.name,
          one_rm: prev.oneRm,
          lag_sets: prev.sets,
          lag_reps: prev.reps,
          lag_rpe: prev.rpe,
          week,
          day: day.dayNumber,
          program_length: program.lengthWeeks,
          time_per_workout: program.timePerWorkout,
          number_of_exercises: day.exercises.length,
          weeks_gap: 1,
          ...flags,
        }).then(res => ({ name: ex.name, res })).catch(err => {
            console.warn(`[predict] failed for "${ex.name}":`, err);
            return null;
          });
      })
    );

    const preds: Record<string, PredictResponse> = {};
    for (const r of results) {
      if (r) preds[r.name] = r.res;
    }
    if (Object.keys(preds).length === 0) {
      setPredictionsStatus('failed');
    } else {
      setPredictions(preds);
      setPredictionsStatus('idle');
    }
  };

  if (!selectedDay) {
    return (
      <View style={styles.container}>
        <Text style={styles.heading}>Which day?</Text>
        <ScrollView contentContainerStyle={[styles.dayGrid, { paddingBottom: bottom + 100 }]}>
          {program.days.map((day, i) => {
            const isPicked = pickedDay?.dayNumber === day.dayNumber;
            return (
              <Pressable
                key={i}
                style={[styles.dayCard, isPicked && styles.dayCardPicked]}
                onPress={() => setPickedDay(day)}
              >
                <View style={styles.dayCardHeader}>
                  <Text style={styles.dayCardTitle}>Day {day.dayNumber}</Text>
                  {lastDayNumber === day.dayNumber && (
                    <Text style={styles.lastDoneBadge}>Last done</Text>
                  )}
                </View>
                <View style={styles.dayExerciseList}>
                  {day.exercises.map(ex => (
                    <View key={ex.id} style={styles.dayExerciseRow}>
                      <Text style={styles.dayExerciseName}>{ex.name}</Text>
                      <Text style={styles.dayExerciseSets}>{ex.sets} × {ex.reps}</Text>
                    </View>
                  ))}
                </View>
              </Pressable>
            );
          })}
        </ScrollView>
        {pickedDay && (
          <View style={[styles.dayPickerFooter, { paddingBottom: bottom || 16 }]}>
            <Pressable style={styles.startBtn} onPress={() => startWorkout(pickedDay)}>
              <Text style={styles.startBtnText}>Start Day {pickedDay.dayNumber}</Text>
            </Pressable>
          </View>
        )}
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

  const swapResults = swapQuery.trim().length >= 2
    ? (EXERCISES as Exercise[])
        .filter(ex => ex.name.toLowerCase().includes(swapQuery.toLowerCase().trim()))
        .slice(0, 30)
    : [];

  const swapExercise = (newEx: Exercise) => {
    const oldEx = logs[exerciseIndex].exercise;
    const newProgramEx: ProgramExercise = { ...newEx, sets: oldEx.sets, reps: oldEx.reps };
    setLogs(prev => prev.map((log, i) => i !== exerciseIndex ? log : {
      ...log,
      exercise: newProgramEx,
      sets: Array.from({ length: oldEx.sets }, () => ({ reps: oldEx.reps, rpe: 8, weight: 0, completed: false })),
    }));
    setPredictions(prev => { const next = { ...prev }; delete next[oldEx.name]; return next; });
    setCurrentWeight('');
    setCurrentReps(newProgramEx.reps);
    setCurrentRpe(8);
    setEditingSetIndex(null);
    setSwapModalVisible(false);
    setSwapQuery('');

    const prev = lastSessionRef.current[newEx.name];
    if (prev?.oneRm > 0) {
      const flags = { ...levelFlags(program), ...goalFlags(program), ...equipmentFlags(program) };
      predict({
        exercise: newEx.name, one_rm: prev.oneRm,
        lag_sets: prev.sets, lag_reps: prev.reps, lag_rpe: prev.rpe,
        week: weekNumber, day: currentLog.dayNumber,
        program_length: program.lengthWeeks, time_per_workout: program.timePerWorkout,
        number_of_exercises: logs.length, weeks_gap: 1, ...flags,
      }).then(res => setPredictions(p => ({ ...p, [newEx.name]: res })))
        .catch(err => console.warn(`[predict] failed for "${newEx.name}":`, err));
    }
  };

  const openInfoModal = async () => {
    setInfoModalVisible(true);
    if (!exerciseInfo || exerciseInfo.name.toLowerCase() !== ex.name.toLowerCase()) {
      setExerciseInfo(null);
      const info = await fetchExerciseInfo(ex.name);
      setExerciseInfo(info);
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
    <Animated.View style={[styles.container, { transform: [{ translateX: slideAnim }] }]}>
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
        <View style={styles.exerciseHeaderTop}>
          <Text style={styles.exerciseCounter}>{exerciseIndex + 1} / {logs.length}</Text>
          <Pressable onPress={() => setSwapModalVisible(true)}>
            <Text style={styles.swapBtn}>Change exercise</Text>
          </Pressable>
        </View>
        <View style={styles.exerciseNameRow}>
          <Text style={[styles.exerciseName, { flex: 1, flexShrink: 1, marginRight: 12 }]} numberOfLines={2} ellipsizeMode="tail">{ex.name}</Text>
          <Pressable onPress={openInfoModal} style={styles.infoBtn} hitSlop={8}>
            <Text style={styles.infoBtnText}>ℹ</Text>
          </Pressable>
        </View>
        <Text style={styles.exerciseMuscle}>{ex.muscle}</Text>
      </View>

      <Modal visible={swapModalVisible} animationType="slide" onRequestClose={() => setSwapModalVisible(false)}>
        <View style={styles.swapModal}>
          <View style={styles.swapHeader}>
            <Text style={styles.swapTitle}>Change Exercise</Text>
            <Pressable onPress={() => { setSwapModalVisible(false); setSwapQuery(''); }}>
              <Text style={styles.swapClose}>✕</Text>
            </Pressable>
          </View>
          <TextInput
            style={styles.swapInput}
            placeholder="Search exercises…"
            placeholderTextColor={colors.secondary}
            value={swapQuery}
            onChangeText={setSwapQuery}
            autoFocus
          />
          <FlatList
            data={swapResults}
            keyExtractor={item => String(item.id)}
            keyboardShouldPersistTaps="handled"
            renderItem={({ item }) => (
              <Pressable style={styles.swapResult} onPress={() => swapExercise(item)}>
                <Text style={styles.swapResultName}>{item.name}</Text>
                <Text style={styles.swapResultMuscle}>{item.muscle}</Text>
              </Pressable>
            )}
          />
        </View>
      </Modal>

      <Modal visible={infoModalVisible} animationType="slide" onRequestClose={() => setInfoModalVisible(false)}>
        <View style={styles.infoModal}>
          <View style={styles.infoModalHeader}>
            <Text style={styles.infoModalTitle}>{ex.name}</Text>
            <Pressable onPress={() => setInfoModalVisible(false)}>
              <Text style={styles.swapClose}>✕</Text>
            </Pressable>
          </View>
          <ScrollView contentContainerStyle={styles.infoModalBody}>
            {exerciseInfo === null ? (
              <ActivityIndicator style={{ marginTop: 40 }} color={colors.accent} />
            ) : (
              <>
                {exerciseInfo.gif_url && (
                  <Image
                    source={{ uri: `${process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000'}${exerciseInfo.gif_url}` }}
                    style={styles.exerciseGif}
                    resizeMode="contain"
                  />
                )}
                {exerciseInfo.steps.map((step, i) => (
                  <View key={i} style={styles.infoStep}>
                    <Text style={styles.infoStepNum}>{i + 1}</Text>
                    <Text style={styles.infoStepText}>{step}</Text>
                  </View>
                ))}
              </>
            )}
          </ScrollView>
        </View>
      </Modal>

      <View style={styles.targetRow}>
        <Text style={styles.targetLabel}>Target</Text>
        {predictions[ex.name] ? (
          <>
            <Text style={styles.targetValue}>
              {predictions[ex.name].next_sets} × {predictions[ex.name].next_reps}
            </Text>
            <Text style={styles.targetSep}>·</Text>
            <Text style={styles.targetLabel}>Week {weekNumber}</Text>
            <Text style={styles.targetWeight}>{predictions[ex.name].next_weight_kg} kg</Text>
            <View style={styles.aiBadge}>
              <Text style={styles.aiBadgeText}>AI</Text>
            </View>
          </>
        ) : (
          <>
            <Text style={styles.targetValue}>{ex.sets} × {ex.reps}</Text>
            <Text style={styles.targetSep}>·</Text>
            <Text style={styles.targetLabel}>Week {weekNumber}</Text>
            <Text style={styles.targetWeight}>— kg</Text>
          </>
        )}
      </View>
      {predictionsStatus === 'failed' && (
        <View style={styles.predFailedBanner}>
          <Text style={styles.predFailedText}>AI predictions unavailable — check your connection</Text>
        </View>
      )}

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
                <Text style={styles.rpeHint}>{RPE_HINTS[editRpe]}</Text>
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
              <Text style={styles.rpeHint}>{RPE_HINTS[currentRpe]}</Text>
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
        {allSetsComplete && !isLast && (
          <Pressable style={styles.nextExerciseBtn} onPress={() => setExerciseIndex(i => i + 1)}>
            <Text style={styles.nextExerciseBtnText}>Next Exercise →</Text>
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
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },

  // day picker
  heading: { ...typography.heading, paddingHorizontal: 20, paddingTop: 32, marginBottom: 24 },
  dayGrid: { paddingHorizontal: 16, gap: 12 },
  dayCard: { backgroundColor: colors.surface, borderRadius: 10, padding: 20, borderWidth: 1, borderColor: colors.border },
  dayCardPicked: { borderColor: colors.accent, backgroundColor: colors.accent + '12' },
  dayPickerFooter: { position: 'absolute' as const, bottom: 0, left: 0, right: 0, padding: 16, backgroundColor: colors.background, borderTopWidth: 1, borderTopColor: colors.border },
  startBtn: { backgroundColor: colors.accent, borderRadius: 10, padding: 16, alignItems: 'center' as const },
  startBtnText: { color: colors.background, fontWeight: '700' as const, fontSize: 15 },
  dayCardHeader: { flexDirection: 'row' as const, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  dayCardTitle: { ...typography.body, fontWeight: '700' as const },
  lastDoneBadge: { ...typography.caption, fontSize: 11, color: colors.accent, borderWidth: 1, borderColor: colors.accent, borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
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
  exerciseName: { ...typography.heading, fontSize: 24, marginBottom: 2, marginTop:10 },
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
  completedSet: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.surface, borderRadius: 8, padding: 12, gap: 8, borderWidth: 1, borderColor: colors.border },
  completedSetNum: { ...typography.caption, width: 44, color: colors.secondary },
  completedSetDetail: { ...typography.caption, flex: 1, textAlign: 'center', color: colors.secondary },
  editHint: { ...typography.caption, color: colors.accent, fontSize: 17 },

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
  nextExerciseBtn: { padding: 14, borderRadius: 10, backgroundColor: colors.accent, alignItems: 'center' as const },
  nextExerciseBtnText: { color: colors.background, fontSize: 15, fontWeight: '700' as const },
  skipBtn: { flex: 1, padding: 12, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' as const },
  skipBtnText: { color: colors.secondary, fontSize: 14 },
  finishBtn: { flex: 1, padding: 14, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' as const },
  finishBtnText: { color: colors.primary, fontSize: 15, fontWeight: '600' as const },
  finishBtnComplete: { backgroundColor: colors.accent, borderColor: colors.accent },
  finishBtnTextComplete: { color: colors.background },
  nextBtnText: { color: colors.accent, fontSize: 15, fontWeight: '700' as const },

  aiBadge: {
    backgroundColor: colors.accent + '22',
    borderWidth: 1,
    borderColor: colors.accent,
    borderRadius: 4,
    paddingHorizontal: 5,
    paddingVertical: 1,
  },
  aiBadgeText: { color: colors.accent, fontSize: 10, fontWeight: '700' as const },

  predFailedBanner: {
    paddingHorizontal: 20,
    paddingVertical: 6,
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  predFailedText: { ...typography.caption, fontSize: 12, color: colors.secondary },

  swapBtn: { ...typography.caption, color: colors.accent, fontSize: 13 },

  swapModal: { flex: 1, backgroundColor: colors.background },
  swapHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, paddingTop: 60, paddingBottom: 16,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  swapTitle: { ...typography.heading, fontSize: 20 },
  swapClose: { color: colors.secondary, fontSize: 18, paddingLeft: 16 },
  swapInput: {
    margin: 16, padding: 12, borderRadius: 10,
    borderWidth: 1, borderColor: colors.border,
    backgroundColor: colors.surface, color: colors.primary, fontSize: 16,
  },
  swapResult: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  swapResultName: { ...typography.body, flex: 1 },
  swapResultMuscle: { ...typography.caption, fontSize: 12, color: colors.secondary },

  // exercise name row with info button
  exerciseNameRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 2 },
  infoBtn: { width: 24, height: 24, borderRadius: 12, borderWidth: 1, borderColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
  infoBtnText: { color: colors.accent, fontSize: 13, fontWeight: '700' as const, lineHeight: 16 },

  // exercise info modal
  infoModal: { flex: 1, backgroundColor: colors.background },
  infoModalHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, paddingTop: 60, paddingBottom: 16,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  infoModalTitle: { ...typography.heading, fontSize: 18, flex: 1, marginRight: 12 },
  infoModalBody: { padding: 20, gap: 16 },
  exerciseGif: { width: '100%', height: 220, borderRadius: 10, backgroundColor: colors.surface, marginBottom: 8 },
  infoStep: { flexDirection: 'row', gap: 12, alignItems: 'flex-start' },
  infoStepNum: {
    width: 24, height: 24, borderRadius: 12, backgroundColor: colors.accent + '22',
    borderWidth: 1, borderColor: colors.accent,
    textAlign: 'center', lineHeight: 22, fontSize: 12, fontWeight: '700' as const, color: colors.accent,
  },
  infoStepText: { ...typography.body, flex: 1, fontSize: 14, lineHeight: 22 },
});
