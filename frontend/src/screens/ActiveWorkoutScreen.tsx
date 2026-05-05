import { ActivityIndicator, Alert, Animated, Dimensions, FlatList, Image, KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import Slider from '@react-native-community/slider';
import { useEffect, useRef, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Notifications from 'expo-notifications';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { colors } from '../theme';
import { Exercise, ExerciseLog, LastSessionEntry, ProgramExercise, RootStackParamList, WorkoutRecord, WorkoutDay } from '../types';
import { HISTORY_KEY } from './WorkoutCompleteScreen';
import { equipmentFlags, ExerciseInfo, fetchExerciseInfo, goalFlags, levelFlags, overloadFlags, predict, PredictResponse } from '../api/client';
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

function WorkoutTimer({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(() => Math.floor((Date.now() - startedAt) / 1000));
  useEffect(() => {
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(id);
  }, [startedAt]);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss = String(elapsed % 60).padStart(2, '0');
  return <Text className="text-secondary text-sm">Workout · {mm}:{ss}</Text>;
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
  const [isDeload, setIsDeload] = useState(false);
  const lastSessionRef = useRef<Record<string, LastSessionEntry>>({});
  const secondLastSessionRef = useRef<Record<string, LastSessionEntry>>({});
  const thirdLastSessionRef = useRef<Record<string, LastSessionEntry>>({});
  const isFinishing = useRef(false);
  const startTime = useRef(0);
  const notifId = useRef<string | null>(null);
  const slideAnim = useRef(new Animated.Value(Dimensions.get('window').width)).current;

  const showWorkoutNotification = async () => {
    const id = await Notifications.scheduleNotificationAsync({
      content: {
        title: 'Workout in progress',
        body: `Started at ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
        sticky: true,
        autoDismiss: false,
      },
      trigger: null,
    });
    notifId.current = id;
  };

  const dismissWorkoutNotification = () => {
    if (notifId.current) {
      Notifications.dismissNotificationAsync(notifId.current);
      notifId.current = null;
    }
  };

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
      headerTitle: () => <WorkoutTimer startedAt={startTime.current} />,
      headerRight: () => (
        <Pressable
          style={{ paddingLeft: 12 }}
          onPress={() => Alert.alert('Exit Workout?', 'Your progress will be lost.', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Exit', style: 'destructive', onPress: () => { isFinishing.current = true; dismissWorkoutNotification(); navigation.goBack(); } },
          ])}
        >
          <Text className="text-secondary text-15">✕</Text>
        </Pressable>
      ),
    });
  }, [selectedDay, navigation]);

  useEffect(() => {
    if (!selectedDay) return;
    return navigation.addListener('beforeRemove', e => {
      if (isFinishing.current) return;
      e.preventDefault();
      Alert.alert('Exit Workout?', 'Your progress will be lost.', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Exit', style: 'destructive', onPress: () => { dismissWorkoutNotification(); navigation.dispatch(e.data.action); } },
      ]);
    });
  }, [selectedDay, navigation]);

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
    const [dayCountRaw, lastSessionRaw, secondLastSessionRaw, thirdLastSessionRaw] = await Promise.all([
      AsyncStorage.getItem(`day_count_${program.id}_${day.dayNumber}`),
      AsyncStorage.getItem(`last_session_${program.id}`),
      AsyncStorage.getItem(`second_last_session_${program.id}`),
      AsyncStorage.getItem(`third_last_session_${program.id}`),
    ]);
    const week = (dayCountRaw ? parseInt(dayCountRaw) : 0) + 1;
    setWeekNumber(week);

    const newLogs: ExerciseLog[] = day.exercises.map(ex => ({
      exercise: ex,
      dayNumber: day.dayNumber,
      sets: Array.from({ length: ex.sets }, () => ({ reps: ex.reps, rpe: 8, weight: 0, completed: false })),
    }));
    startTime.current = Date.now();
    showWorkoutNotification();
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
    secondLastSessionRef.current = secondLastSessionRaw ? JSON.parse(secondLastSessionRaw) : {};
    thirdLastSessionRef.current = thirdLastSessionRaw ? JSON.parse(thirdLastSessionRaw) : {};
    const eligibleExercises = day.exercises.filter(ex => lastSession[ex.name]?.oneRm > 0);
    if (eligibleExercises.length === 0) return;

    setPredictionsStatus('loading');
    const flags = { ...levelFlags(program), ...goalFlags(program), ...equipmentFlags(program), ...overloadFlags(program, isDeload) };
    const results = await Promise.all(
      eligibleExercises.map(ex => {
        const prev  = lastSession[ex.name];
        const prev2 = secondLastSessionRef.current[ex.name];
        const prev3 = thirdLastSessionRef.current[ex.name];
        return predict({
          exercise: ex.name,
          one_rm: prev.oneRm,
          lag_sets: prev.sets,
          lag_reps: prev.reps,
          lag_rpe: prev.rpe,
          ...(prev2 && { lag2_reps: prev2.reps, lag2_rpe: prev2.rpe }),
          ...(prev3 && { lag3_reps: prev3.reps, lag3_rpe: prev3.rpe }),
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
      <View className="flex-1 bg-background">
        <Text className="text-22 font-outfit-bold text-accent px-5 pt-8 mb-6">Which day?</Text>
        <ScrollView contentContainerStyle={{ paddingHorizontal: 16, gap: 12, paddingBottom: bottom + 100 }}>
          {program.days.map((day, i) => {
            const isPicked = pickedDay?.dayNumber === day.dayNumber;
            return (
              <Pressable
                key={i}
                className={`bg-surface rounded-[10px] p-5 border ${isPicked ? 'border-accent bg-accent/[7%]' : 'border-border'}`}
                onPress={() => setPickedDay(day)}
              >
                <View className="flex-row items-center justify-between mb-2.5">
                  <Text className="text-base font-outfit text-primary font-bold">Day {day.dayNumber}</Text>
                  {lastDayNumber === day.dayNumber && (
                    <Text className="text-11 font-outfit text-accent border border-accent rounded-md px-1.5 py-0.5">Last done</Text>
                  )}
                </View>
                <View className="gap-1.5">
                  {day.exercises.map(ex => (
                    <View key={ex.id} className="flex-row justify-between items-center">
                      <Text className="text-13 font-outfit text-secondary flex-1">{ex.name}</Text>
                      <Text className="text-13 font-outfit text-secondary">{ex.sets} × {ex.reps}</Text>
                    </View>
                  ))}
                </View>
              </Pressable>
            );
          })}
        </ScrollView>
        {pickedDay && (
          <View
            className="absolute bottom-0 left-0 right-0 bg-background border-t border-border"
            style={{ paddingBottom: bottom || 16 }}
          >
            <Pressable
              className="flex-row items-center justify-between px-5 py-3.5 border-b border-border"
              onPress={() => setIsDeload(v => !v)}
            >
              <View>
                <Text className="text-base font-outfit text-primary">Deload week</Text>
                <Text className="text-xs font-outfit text-secondary mt-0.5">Reduce volume & intensity for recovery</Text>
              </View>
              <View
                className={`w-12 h-7 rounded-full ${isDeload ? 'bg-accent' : 'bg-border'}`}
                style={{ padding: 2 }}
              >
                <View style={{
                  width: 23, height: 23, borderRadius: 12,
                  backgroundColor: 'white',
                  alignSelf: isDeload ? 'flex-end' : 'flex-start',
                }} />
              </View>
            </Pressable>
            <View className="p-4">
              <Pressable
                className="bg-accent rounded-[10px] p-4 items-center"
                onPress={() => startWorkout(pickedDay)}
              >
                <Text className="text-background font-bold text-15">Start Day {pickedDay.dayNumber}</Text>
              </Pressable>
            </View>
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
      setCurrentWeight(currentWeight);
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

    const prev  = lastSessionRef.current[newEx.name];
    const prev2 = secondLastSessionRef.current[newEx.name];
    const prev3 = thirdLastSessionRef.current[newEx.name];
    if (prev?.oneRm > 0) {
      const flags = { ...levelFlags(program), ...goalFlags(program), ...equipmentFlags(program), ...overloadFlags(program, isDeload) };
      predict({
        exercise: newEx.name, one_rm: prev.oneRm,
        lag_sets: prev.sets, lag_reps: prev.reps, lag_rpe: prev.rpe,
        ...(prev2 && { lag2_reps: prev2.reps, lag2_rpe: prev2.rpe }),
        ...(prev3 && { lag3_reps: prev3.reps, lag3_rpe: prev3.rpe }),
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

  const prevSession = lastSessionRef.current[ex.name];
  const workoutComplete = logs.every(log => log.sets.every(s => s.completed));

  const finishWorkout = () => {
    const doFinish = () => {
      isFinishing.current = true;
      dismissWorkoutNotification();
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
    <Animated.View className="flex-1 bg-background" style={{ transform: [{ translateX: slideAnim }] }}>
      {/* Exercise tab bar — tap to jump to any exercise */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        className="border-b border-border grow-0"
        contentContainerStyle={{ paddingHorizontal: 12, paddingVertical: 8, gap: 8 }}
      >
        {logs.map((log, i) => {
          const done = log.sets.every(s => s.completed);
          const active = i === exerciseIndex;
          return (
            <Pressable
              key={i}
              className={`min-w-[80px] px-3 py-1.5 rounded-[20px] border ${active ? 'border-accent bg-accent/[9%]' : 'border-border bg-surface'}`}
              onPress={() => setExerciseIndex(i)}
            >
              <Text className={`text-xs font-outfit ${active ? 'text-accent font-semibold' : 'text-secondary'}`}>
                {done ? '✓ ' : ''}{log.exercise.name}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>

      <View className="px-5 pt-5 pb-4 border-b border-border">
        <View className="flex-row justify-between items-center mb-1">
          <Text className="text-13 font-outfit text-secondary">{exerciseIndex + 1} / {logs.length}</Text>
          <Pressable onPress={() => setSwapModalVisible(true)}>
            <Text className="text-13 font-outfit text-accent">Change exercise</Text>
          </Pressable>
        </View>
        <View className="flex-row items-center mb-0.5">
          <Text className="text-22 font-outfit-bold text-accent flex-1 shrink mt-2.5 mr-3" numberOfLines={2} ellipsizeMode="tail">{ex.name}</Text>
          <Pressable onPress={openInfoModal} className="w-6 h-6 rounded-full border border-accent items-center justify-center" hitSlop={8}>
            <Text className="text-accent text-[13px] font-bold" style={{ lineHeight: 16 }}>ℹ</Text>
          </Pressable>
        </View>
        <Text className="text-13 font-outfit text-secondary">{ex.muscle}</Text>
      </View>

      {/* Change-exercise modal */}
      <Modal visible={swapModalVisible} animationType="slide" onRequestClose={() => setSwapModalVisible(false)}>
        <View className="flex-1 bg-background">
          <View className="flex-row items-center justify-between px-5 pt-[60px] pb-4 border-b border-border">
            <Text className="text-xl font-outfit-bold text-accent">Change Exercise</Text>
            <Pressable onPress={() => { setSwapModalVisible(false); setSwapQuery(''); }}>
              <Text className="text-secondary text-lg pl-4">✕</Text>
            </Pressable>
          </View>
          <TextInput
            className="m-4 p-3 rounded-[10px] border border-border bg-surface text-primary text-base"
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
              <Pressable
                className="flex-row items-center justify-between px-5 py-3.5 border-b border-border"
                onPress={() => swapExercise(item)}
              >
                <Text className="text-base font-outfit text-primary flex-1">{item.name}</Text>
                <Text className="text-xs font-outfit text-secondary">{item.muscle}</Text>
              </Pressable>
            )}
          />
        </View>
      </Modal>

      {/* Exercise info modal */}
      <Modal visible={infoModalVisible} animationType="slide" onRequestClose={() => setInfoModalVisible(false)}>
        <View className="flex-1 bg-background">
          <View className="flex-row items-center justify-between px-5 pt-[60px] pb-4 border-b border-border">
            <Text className="text-lg font-outfit-bold text-accent flex-1 mr-3">{ex.name}</Text>
            <Pressable onPress={() => setInfoModalVisible(false)}>
              <Text className="text-secondary text-lg pl-4">✕</Text>
            </Pressable>
          </View>
          <ScrollView contentContainerStyle={{ padding: 20, gap: 16 }}>
            {exerciseInfo === null ? (
              <ActivityIndicator style={{ marginTop: 40 }} color={colors.accent} />
            ) : (
              <>
                {exerciseInfo.gif_url && (
                  <Image
                    source={{ uri: `${process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000'}${exerciseInfo.gif_url}` }}
                    className="w-full h-[220px] rounded-[10px] bg-surface mb-2"
                    resizeMode="contain"
                  />
                )}
                {exerciseInfo.steps.map((step, i) => (
                  <View key={i} className="flex-row gap-3 items-start">
                    <Text
                      className="w-6 h-6 rounded-full text-center text-xs font-bold text-accent border border-accent bg-accent/[13%]"
                      style={{ lineHeight: 22 }}
                    >
                      {i + 1}
                    </Text>
                    <Text className="text-base font-outfit text-primary flex-1 text-sm leading-[22px]">{step}</Text>
                  </View>
                ))}
              </>
            )}
          </ScrollView>
        </View>
      </Modal>

      {/* Last session / AI prediction row */}
      <View className="px-5 py-3 gap-1.5 border-b border-border">
        {prevSession && (
          <View className="flex-row items-center gap-2">
            <Text className="text-xs font-outfit text-secondary">Last session</Text>
            <Text className="text-sm font-outfit text-secondary font-semibold">{prevSession.sets} × {Math.round(prevSession.reps)}</Text>
            {prevSession.weight != null && prevSession.weight > 0 && (
              <>
                <Text className="text-13 font-outfit text-secondary">@</Text>
                <Text className="text-sm font-outfit text-secondary">{prevSession.weight} kg</Text>
              </>
            )}
          </View>
        )}
        {predictions[ex.name] ? (
          <View className="flex-row items-center gap-2">
            <View className="bg-accent rounded px-1.5 py-0.5">
              <Text className="text-background text-11 font-bold">AI</Text>
            </View>
            <Text className="text-13 font-outfit text-accent">Suggests</Text>
            <Text className="text-17 font-outfit-bold text-accent">{prevSession?.sets} × {predictions[ex.name].next_reps}</Text>
            <Text className="text-13 font-outfit text-secondary">@</Text>
            <Text className="text-17 font-outfit-bold text-accent">{predictions[ex.name].next_weight_kg} kg</Text>
          </View>
        ) : (
          <View className="flex-row items-center gap-2">
            <Text className="text-xs font-outfit text-secondary">Target</Text>
            <Text className="text-sm font-outfit text-secondary font-semibold">{ex.sets} × {ex.reps}</Text>
            <Text className="text-13 font-outfit text-secondary">·</Text>
            <Text className="text-xs font-outfit text-secondary">Week {weekNumber}</Text>
          </View>
        )}
      </View>

      {predictionsStatus === 'failed' && (
        <View className="px-5 py-1.5 bg-surface border-b border-border">
          <Text className="text-xs font-outfit text-secondary">AI predictions unavailable — check your connection</Text>
        </View>
      )}

      <KeyboardAvoidingView className="flex-1" behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <ScrollView contentContainerStyle={{ padding: 16, gap: 12 }} keyboardShouldPersistTaps="handled">

          {currentLog.sets.slice(0, currentSetIndex).map((set, si) =>
            editingSetIndex === si ? (
              <View key={si} className="bg-surface rounded-[10px] p-4 gap-5 border border-accent">
                <View className="flex-row items-center justify-between">
                  <Text className="text-13 font-outfit text-accent font-semibold">Editing Set {si + 1}</Text>
                  <Pressable onPress={() => setEditingSetIndex(null)}>
                    <Text className="text-13 font-outfit text-secondary">Cancel</Text>
                  </Pressable>
                </View>

                <View className="flex-row items-center justify-between">
                  <Text className="text-base font-outfit text-secondary">Weight (kg)</Text>
                  <TextInput
                    className="w-[100px] h-14 border border-border rounded-lg bg-background text-primary text-center text-lg font-bold"
                    value={editWeight}
                    onChangeText={setEditWeight}
                    keyboardType="numeric"
                    placeholder="—"
                    placeholderTextColor={colors.secondary}
                    returnKeyType="done"
                    scrollEnabled={false}
                    autoFocus
                  />
                </View>

                <View className="flex-row items-center justify-between">
                  <Text className="text-base font-outfit text-secondary">Reps</Text>
                  <View className="flex-row items-center">
                    <Pressable className="w-11 h-11 bg-background border border-border rounded-lg items-center justify-center" onPress={() => setEditReps(r => Math.max(1, r - 1))}>
                      <Text className="text-accent text-xl font-light">−</Text>
                    </Pressable>
                    <Text className="w-14 text-center text-22 font-outfit-bold text-accent">{editReps}</Text>
                    <Pressable className="w-11 h-11 bg-background border border-border rounded-lg items-center justify-center" onPress={() => setEditReps(r => r + 1)}>
                      <Text className="text-accent text-xl font-light">+</Text>
                    </Pressable>
                  </View>
                </View>

                <View className="gap-1">
                  <View className="flex-row items-center justify-between">
                    <Text className="text-base font-outfit text-secondary">RPE</Text>
                    <Text className="text-base font-outfit text-primary">{editRpe} — {RPE_LABELS[editRpe]}</Text>
                  </View>
                  <Text className="text-13 font-outfit text-secondary leading-[18px]">{RPE_HINTS[editRpe]}</Text>
                  <Slider
                    style={{ width: '100%', height: 40 }}
                    minimumValue={6} maximumValue={10} step={0.5}
                    value={editRpe}
                    onValueChange={v => setEditRpe(v)}
                    minimumTrackTintColor={colors.accent}
                    maximumTrackTintColor={colors.border}
                    thumbTintColor={colors.accent}
                  />
                </View>

                <Pressable className="bg-accent rounded-lg p-3 items-center" onPress={saveEditedSet}>
                  <Text className="text-background font-bold text-sm">Save Changes</Text>
                </Pressable>
              </View>
            ) : (
              <Pressable key={si} className="flex-row items-center bg-surface rounded-lg p-3 gap-2 border border-border" onPress={() => openEditSet(si)}>
                <Text className="text-13 font-outfit text-secondary w-11">Set {si + 1}</Text>
                <Text className="text-13 font-outfit text-secondary flex-1 text-center">{set.weight > 0 ? `${set.weight} kg` : 'BW'}</Text>
                <Text className="text-13 font-outfit text-secondary flex-1 text-center">{set.reps} reps</Text>
                <Text className="text-13 font-outfit text-secondary flex-1 text-center">RPE {set.rpe}</Text>
                <Text className="text-accent text-17">✎</Text>
              </Pressable>
            )
          )}

          {!allSetsComplete && editingSetIndex === null && (
            <View className="bg-surface rounded-[10px] p-4 gap-5 border border-accent">
              <Text className="text-13 font-outfit text-accent font-semibold">Set {currentSetIndex + 1} of {currentLog.sets.length}</Text>

              <View className="flex-row items-center justify-between">
                <Text className="text-base font-outfit text-secondary">Weight (kg)</Text>
                <TextInput
                  className="w-[100px] h-13 border border-border rounded-lg bg-background text-primary text-center text-xl font-bold"
                  value={currentWeight}
                  onChangeText={setCurrentWeight}
                  keyboardType="numeric"
                  placeholder="—"
                  placeholderTextColor={colors.secondary}
                  returnKeyType="done"
                  scrollEnabled={false}
                />
              </View>

              <View className="flex-row items-center justify-between">
                <Text className="text-base font-outfit text-secondary">Reps</Text>
                <View className="flex-row items-center">
                  <Pressable className="w-11 h-11 bg-background border border-border rounded-lg items-center justify-center" onPress={() => setCurrentReps(r => Math.max(1, r - 1))}>
                    <Text className="text-accent text-xl font-light">−</Text>
                  </Pressable>
                  <Text className="w-14 text-center text-22 font-outfit-bold text-accent">{currentReps}</Text>
                  <Pressable className="w-11 h-11 bg-background border border-border rounded-lg items-center justify-center" onPress={() => setCurrentReps(r => r + 1)}>
                    <Text className="text-accent text-xl font-light">+</Text>
                  </Pressable>
                </View>
              </View>

              <View className="gap-1">
                <View className="flex-row items-center justify-between">
                  <Text className="text-base font-outfit text-secondary">RPE</Text>
                  <Text className="text-base font-outfit text-primary">{currentRpe} — {RPE_LABELS[currentRpe]}</Text>
                </View>
                <Text className="text-13 font-outfit text-secondary leading-[18px]">{RPE_HINTS[currentRpe]}</Text>
                <Slider
                  style={{ width: '100%', height: 40 }}
                  minimumValue={6} maximumValue={10} step={0.5}
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

        <View className="gap-2 p-4" style={{ paddingBottom: bottom || 16 }}>
          {!allSetsComplete && editingSetIndex === null && (
            <Pressable className="p-4 rounded-[10px] bg-accent items-center" onPress={completeSet}>
              <Text className="text-background text-15 font-bold">✓  Complete Set {currentSetIndex + 1}</Text>
            </Pressable>
          )}
          {allSetsComplete && !isLast && (
            <Pressable className="p-3.5 rounded-[10px] bg-accent items-center" onPress={() => setExerciseIndex(i => i + 1)}>
              <Text className="text-background text-15 font-bold">Next Exercise →</Text>
            </Pressable>
          )}
          <View className="flex-row gap-2">
            {!allSetsComplete && editingSetIndex === null && (
              <Pressable
                className="flex-1 p-3 rounded-[10px] border border-border items-center"
                onPress={() => setExerciseIndex(i => isLast ? i : i + 1)}
              >
                <Text className="text-secondary text-sm">Skip</Text>
              </Pressable>
            )}
            <Pressable
              className={`flex-1 p-3.5 rounded-[10px] border items-center ${allSetsComplete ? 'bg-accent border-accent' : 'border-border'}`}
              onPress={finishWorkout}
            >
              <Text className={`text-15 font-semibold ${allSetsComplete ? 'text-background' : 'text-primary'}`}>Finish Workout</Text>
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Animated.View>
  );
}
