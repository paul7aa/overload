import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated, KeyboardAvoidingView, Pressable, StyleSheet,
  Text, TextInput, View, TouchableOpacity, ScrollView
} from 'react-native';
import DragList, { DragListRenderItemInfo } from 'react-native-draglist';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, typography } from '../theme';
import { Exercise, ProgramExercise, Program, WorkoutDay, RootStackParamList } from '../types';
import EXERCISES from '../data/exercises.json';
import React from 'react';

function scoreExercise(ex: Exercise, words: string[]): number {
  const name = ex.name.toLowerCase();
  const muscle = ex.muscle.toLowerCase();
  const nameWords = name.split(/[\s\-]+/);

  if (words.every(w => nameWords.some(nw => nw.startsWith(w)))) {
    return nameWords[0].startsWith(words[0]) ? 90 : 75;
  }
  if (words.every(w => name.includes(w))) return 60;
  const hits = words.filter(w => nameWords.some(nw => nw.startsWith(w))).length;
  if (hits > 0) return 40 + (hits / words.length) * 15;
  if (words.every(w => muscle.includes(w))) return 20;
  return 0;
}

function searchExercises(query: string, exclude: Set<number>): Exercise[] {
  const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  return (EXERCISES as Exercise[])
    .filter(ex => !exclude.has(ex.id))
    .map(ex => ({ ex, score: scoreExercise(ex, words) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.ex.name.localeCompare(b.ex.name))
    .map(({ ex }) => ex)
    .slice(0, 40);
}

type Props = NativeStackScreenProps<RootStackParamList, 'AddProgram'>;

const LEVELS    = ['Beginner', 'Novice', 'Intermediate', 'Advanced'];
const GOALS     = ['Powerlifting', 'Powerbuilding', 'Bodybuilding', 'Muscle & Sculpting', 'Athletics', 'Bodyweight Fitness', 'At-Home & Calisthenics', 'Olympic Weightlifting'];
const EQUIPMENT = ['Full Gym', 'Garage Gym', 'Dumbbell Only', 'At Home'];
const MAX_DAYS  = 7;
const TOTAL_STEPS = 5;
const PROGRAMS_KEY = 'programs';

function AnimatedDay({ onRemove, children}: { onRemove: () => void; children: (remove: () => void) => React.ReactNode }) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(anim, { toValue: 1, duration: 260, useNativeDriver: true }).start();
  }, []);
  const animateRemove = () => {
    Animated.timing(anim, { toValue: 0, duration: 180, useNativeDriver: true }).start(() => onRemove());
  };
  return (
    <Animated.View 
    style={{
      opacity: anim,
      transform: [{ translateY: anim.interpolate({ inputRange: [0, 1], outputRange: [14, 0] }) }],
    }}>
      {children(animateRemove)}
    </Animated.View>
  );
}

interface ExerciseItemProps {
  ex: ProgramExercise;
  dayIndex: number;
  isActive: boolean;
  onDragStart: () => void;
  removeExercise: (dayIndex: number, exId: number) => void;
}

const ExerciseItem = React.memo(({
  ex, dayIndex, isActive, onDragStart, removeExercise
}: ExerciseItemProps) => {
  return (
    <View style={[styles.exRow, isActive && styles.exRowDragging]} collapsable={false}>
      <Pressable onPress={() => removeExercise(dayIndex, ex.id)} hitSlop={8}>
        <Text style={styles.exRemove}>×</Text>
      </Pressable>
      <View style={styles.exInfo}>
        <Text style={styles.exName}>{ex.name}</Text>
        <Text style={styles.exMeta}>{ex.sets} × {ex.reps}  ·  {ex.muscle}</Text>
      </View>
      <TouchableOpacity 
        onPressIn={onDragStart} // Only trigger the start, let the library handle the drop naturally
        hitSlop={8} 
        style={styles.dragHandle}
      >
        <Text style={styles.dragHandleIcon}>≡</Text>
      </TouchableOpacity>
    </View>
  );
});

interface DayExerciseListProps {
  day: WorkoutDay;
  dayIndex: number;
  reorderExercise: (dayIndex: number, fromIndex: number, toIndex: number) => void;
  removeExercise: (dayIndex: number, exId: number) => void;
  onDragBegin: () => void;
  onDragEnd: () => void;
}

const DayExerciseList = React.memo(({
  day, dayIndex, reorderExercise, removeExercise, onDragBegin, onDragEnd
}: DayExerciseListProps) => {

  const renderItem = useCallback(({ item: ex, onDragStart, isActive }: DragListRenderItemInfo<ProgramExercise>) => (
    <ExerciseItem
      ex={ex}
      dayIndex={dayIndex}
      isActive={isActive}
      onDragStart={onDragStart}
      removeExercise={removeExercise}
    />
  ), [dayIndex, removeExercise]);

  const keyExtractor = useCallback((ex: ProgramExercise) => `ex-${ex.id}`, []);

  const onReordered = useCallback((fromIndex: number, toIndex: number) => {
    reorderExercise(dayIndex, fromIndex, toIndex);
  }, [dayIndex, reorderExercise]);

  return (
    <DragList
      data={day.exercises}
      keyExtractor={keyExtractor}
      onReordered={onReordered}
      onDragBegin={onDragBegin} // Library tells us when the drag starts
      onDragEnd={onDragEnd}     // Library tells us when the drag fully finishes
      scrollEnabled={false}
      renderItem={renderItem}
    />
  );
});

export default function AddProgramModal({ navigation, route }: Props) {
  const editing = route.params?.program;
  const { bottom } = useSafeAreaInsets();
  const scrollRef = useRef<any>(null);
  useEffect(() => {
    navigation.setOptions({ title: editing ? 'Edit Program' : 'New Program' });
  }, []);
  const [step, setStep] = useState(1);
  const [name, setName] = useState(editing?.name ?? '');
  const [level, setLevel] = useState<string[]>(editing?.level ?? []);
  const [goal, setGoal] = useState<string[]>(editing?.goal ?? []);
  const [equipment, setEquipment] = useState<string | null>(editing?.equipment ?? null);
  const [lengthWeeks, setLengthWeeks] = useState(String(editing?.lengthWeeks ?? 12));
  const [timePerWorkout, setTimePerWorkout] = useState(String(editing?.timePerWorkout ?? 60));
  const [days, setDays] = useState<WorkoutDay[]>(editing?.days ?? [{ dayNumber: 1, exercises: [] }]);

// Drag state for standard ScrollView wrapper
  const [isDragging, setIsDragging] = useState(false);

const handleDragBegin = useCallback(() => setIsDragging(true), []);
  
  const handleDragEnd = useCallback(() => {
    // Give the layout engine 50ms to process the drop and array reorder 
    // before re-enabling the parent scroll view.
    setTimeout(() => {
      setIsDragging(false);
    }, 50);
  }, []);
  // exercise picker state
  const [addingToDay, setAddingToDay] = useState<number | null>(null);
  const [query, setQuery] = useState('');
  const [pendingEx, setPendingEx] = useState<Exercise | null>(null);
  const [pendingSets, setPendingSets] = useState('3');
  const [pendingReps, setPendingReps] = useState('8');

  const toggleMulti = (val: string, list: string[], set: (v: string[]) => void) =>
    set(list.includes(val) ? list.filter(v => v !== val) : [...list, val]);

  const filtered: Exercise[] = useMemo(() => {
    if (query.length < 2 || addingToDay === null) return [];
    const alreadyAdded = new Set(days[addingToDay].exercises.map(e => e.id));
    return searchExercises(query, alreadyAdded);
  }, [query, addingToDay, days]);

  const openPicker = (dayIndex: number) => {
    setAddingToDay(dayIndex);
    setQuery('');
    setPendingEx(null);
    setPendingSets('3');
    setPendingReps('8');
  };

  const closePicker = () => {
    setAddingToDay(null);
    setQuery('');
    setPendingEx(null);
  };

  const confirmAdd = () => {
    if (pendingEx === null || addingToDay === null) return;
    const ex: ProgramExercise = { ...pendingEx, sets: parseInt(pendingSets) || 3, reps: parseInt(pendingReps) || 8 };
    setDays(prev => prev.map((d, i) =>
      i === addingToDay ? { ...d, exercises: [...d.exercises, ex] } : d
    ));
    setPendingEx(null);
    setQuery('');
  };

  const removeExercise = useCallback((dayIndex: number, exId: number) => {
    setDays(prev => prev.map((d, i) =>
      i === dayIndex ? { ...d, exercises: d.exercises.filter(e => e.id !== exId) } : d
    ));
  }, []);

  const reorderExercise = useCallback((dayIndex: number, fromIndex: number, toIndex: number) => {
    setDays(prev => prev.map((d, i) => {
      if (i !== dayIndex) return d;
      const reordered = [...d.exercises];
      const [moved] = reordered.splice(fromIndex, 1);
      reordered.splice(toIndex, 0, moved);
      return { ...d, exercises: reordered };
    }));
  }, []);

  const addDay = () => {
    if (days.length >= MAX_DAYS) return;
    setDays(prev => [...prev, { dayNumber: prev.length + 1, exercises: [] }]);
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
  };

  const duplicateDay = (dayIndex: number) => {
    if (days.length >= MAX_DAYS) return;
    setDays(prev => {
      const clone = { ...prev[dayIndex], exercises: [...prev[dayIndex].exercises] };
      const next = [...prev, clone];
      return next.map((d, i) => ({ ...d, dayNumber: i + 1 }));
    });
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
  };

  const swapDays = (i: number, j: number) => {
    setDays(prev => {
      const next = [...prev];
      [next[i], next[j]] = [next[j], next[i]];
      return next.map((d, idx) => ({ ...d, dayNumber: idx + 1 }));
    });
  };

  const removeDay = (index: number) => {
    if (days.length === 1) return;
    if (addingToDay === index) closePicker();
    setDays(prev =>
      prev.filter((_, i) => i !== index).map((d, i) => ({ ...d, dayNumber: i + 1 }))
    );
  };

  const canAdvance = (): boolean => {
    if (step === 1) return name.trim().length > 0 && level.length > 0;
    if (step === 2) return goal.length > 0;
    if (step === 3) return equipment !== null;
    if (step === 4) return parseInt(lengthWeeks) > 0 && parseInt(timePerWorkout) > 0;
    if (step === 5) return days.every(d => d.exercises.length > 0);
    return false;
  };

  const save = async () => {
    const raw = await AsyncStorage.getItem(PROGRAMS_KEY);
    const programs: Program[] = raw ? JSON.parse(raw) : [];
    const updated: Program = {
      id: editing?.id ?? Date.now().toString(),
      name: name.trim(),
      level, goal,
      equipment: equipment!,
      lengthWeeks: parseInt(lengthWeeks),
      timePerWorkout: parseInt(timePerWorkout),
      days,
    };
    const next = editing
      ? programs.map(p => p.id === editing.id ? updated : p)
      : [...programs, updated];
    await AsyncStorage.setItem(PROGRAMS_KEY, JSON.stringify(next));
    navigation.goBack();
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior="padding">
      <View style={styles.progressRow}>
        {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
          <View key={i} style={[styles.pip, i < step && styles.pipDone]} />
        ))}
      </View>

      <ScrollView 
        ref={scrollRef} 
        contentContainerStyle={styles.body} 
        keyboardShouldPersistTaps="handled"
        scrollEnabled={!isDragging} // Disable parent scrolling while dragging
      >

        {step === 1 && <>
          <Text style={styles.heading}>Name your program</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 12-Week Strength Block"
            placeholderTextColor={colors.secondary}
            value={name}
            onChangeText={setName}
          />
          <Text style={[styles.heading, { marginTop: 28 }]}>Level</Text>
          <OptionRow options={LEVELS} selected={level} onSelect={v => toggleMulti(v, level, setLevel)} />
        </>}

        {step === 2 && <>
          <Text style={styles.heading}>Goal</Text>
          <OptionGrid options={GOALS} selected={goal} onSelect={v => toggleMulti(v, goal, setGoal)} />
        </>}

        {step === 3 && <>
          <Text style={styles.heading}>Equipment</Text>
          <OptionGrid options={EQUIPMENT} selected={equipment} onSelect={setEquipment} />
        </>}

        {step === 4 && <>
          <Text style={styles.heading}>Program length (weeks)</Text>
          <TextInput style={styles.input} keyboardType="numeric" value={lengthWeeks} onChangeText={setLengthWeeks} placeholderTextColor={colors.secondary} />
          <Text style={[styles.heading, { marginTop: 28 }]}>Time per workout (min)</Text>
          <TextInput style={styles.input} keyboardType="numeric" value={timePerWorkout} onChangeText={setTimePerWorkout} placeholderTextColor={colors.secondary} />
        </>}

        {step === 5 && <> 
          {days.map((day, di) => (
            <AnimatedDay key={`day-block-${day.dayNumber}`} onRemove={() => removeDay(di)}>
              {(animateRemove) => (
            <View style={styles.dayBlock}>
              <View style={styles.dayHeader}>
                <Text style={styles.dayTitle}>Day {day.dayNumber}</Text>
                <View style={styles.dayActions}>
                  {di > 0 && (
                    <Pressable onPress={() => swapDays(di, di - 1)} hitSlop={8}>
                      <Text style={styles.dayAction}>▲</Text>
                    </Pressable>
                  )}
                  {di < days.length - 1 && (
                    <Pressable onPress={() => swapDays(di, di + 1)} hitSlop={8}>
                      <Text style={styles.dayAction}>▼</Text>
                    </Pressable>
                  )}
                  {days.length < MAX_DAYS && (
                    <Pressable onPress={() => duplicateDay(di)} hitSlop={8}>
                      <Text style={styles.dayAction}>⧉</Text>
                    </Pressable>
                  )}
                  {days.length > 1 && (
                    <Pressable onPress={animateRemove} hitSlop={8}>
                      <Text style={styles.removeDay}>Remove</Text>
                    </Pressable>
                  )}
                </View>
              </View>

              <DayExerciseList
                day={day}
                dayIndex={di}
                reorderExercise={reorderExercise}
                removeExercise={removeExercise}
                onDragBegin={handleDragBegin}
                onDragEnd={handleDragEnd}
              />

              {addingToDay === di ? (
                <View style={styles.picker}>
                  {pendingEx === null ? (
                    <>
                      <TextInput
                        style={styles.input}
                        placeholder="Search exercises..."
                        placeholderTextColor={colors.secondary}
                        value={query}
                        onChangeText={setQuery}
                        autoFocus
                      />
                      {filtered.map(ex => (
                        <Pressable key={ex.id} style={styles.searchRow} onPress={() => setPendingEx(ex)}>
                          <Text style={styles.exName}>{ex.name}</Text>
                          <Text style={styles.exMeta}>{ex.muscle}</Text>
                        </Pressable>
                      ))}
                      <Pressable onPress={closePicker} style={styles.cancelBtn}>
                        <Text style={styles.cancelText}>Cancel</Text>
                      </Pressable>
                    </>
                  ) : (
                    <View style={styles.setsRepsRow}>
                      <Text style={styles.pendingName}>{pendingEx.name}</Text>
                      <View style={styles.setsRepsInputs}>
                        <View style={styles.numericField}>
                          <Text style={styles.numericLabel}>Sets</Text>
                          <TextInput style={styles.numericInput} keyboardType="numeric" value={pendingSets} onChangeText={setPendingSets} />
                        </View>
                        <Text style={styles.times}>×</Text>
                        <View style={styles.numericField}>
                          <Text style={styles.numericLabel}>Reps</Text>
                          <TextInput style={styles.numericInput} keyboardType="numeric" value={pendingReps} onChangeText={setPendingReps} />
                        </View>
                      </View>
                      <View style={styles.confirmRow}>
                        <Pressable style={styles.cancelBtn} onPress={() => setPendingEx(null)}>
                          <Text style={styles.cancelText}>Back</Text>
                        </Pressable>
                        <Pressable style={styles.confirmBtn} onPress={confirmAdd}>
                          <Text style={styles.confirmText}>Add</Text>
                        </Pressable>
                      </View>
                    </View>
                  )}
                </View>
              ) : (
                <Pressable style={styles.addExBtn} onPress={() => openPicker(di)}>
                  <Text style={styles.addExText}>+ Add Exercise</Text>
                </Pressable>
              )}
            </View>
              )}
            </AnimatedDay>
          ))}

          {days.length < MAX_DAYS && (
            <Pressable style={styles.addDayBtn} onPress={addDay}>
              <Text style={styles.addDayText}>+ Add Day</Text>
            </Pressable>
          )}
        </>}

      </ScrollView>

      <View style={[styles.footer, { paddingBottom: bottom || 16 }]}>
        {step > 1 && (
          <Pressable style={styles.backBtn} onPress={() => setStep(s => s - 1)}>
            <Text style={styles.backBtnText}>Back</Text>
          </Pressable>
        )}
        <Pressable
          style={[styles.nextBtn, !canAdvance() && styles.nextBtnDisabled]}
          disabled={!canAdvance()}
          onPress={() => step < TOTAL_STEPS ? setStep(s => s + 1) : save()}
        >
          <Text style={styles.nextBtnText}>{step < TOTAL_STEPS ? 'Next' : 'Save'}</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

function OptionRow({ options, selected, onSelect }: { options: string[]; selected: string[]; onSelect: (o: string) => void }) {
  return (
    <View style={styles.optionRow}>
      {options.map(o => (
        <Pressable key={o} style={[styles.pill, selected.includes(o) && styles.pillActive]} onPress={() => onSelect(o)}>
          <Text style={[styles.pillText, selected.includes(o) && styles.pillTextActive]}>{o}</Text>
        </Pressable>
      ))}
    </View>
  );
}

function OptionGrid({ options, selected, onSelect }: { options: string[]; selected: string | string[] | null; onSelect: (o: string) => void }) {
  const active = (o: string) => Array.isArray(selected) ? selected.includes(o) : o === selected;
  return (
    <View style={styles.grid}>
      {options.map(o => (
        <Pressable key={o} style={[styles.gridCard, active(o) && styles.gridCardActive]} onPress={() => onSelect(o)}>
          <Text style={[styles.gridText, active(o) && styles.gridTextActive]}>{o}</Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  progressRow: { flexDirection: 'row', gap: 6, paddingHorizontal: 20, paddingTop: 16 },
  pip: { flex: 1, height: 3, borderRadius: 2, backgroundColor: colors.border },
  pipDone: { backgroundColor: colors.accent },
  body: { padding: 20, gap: 8 },
  heading: { ...typography.body, fontWeight: '600', marginBottom: 12 },
  input: { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 12, color: colors.accent, fontSize: 15 },

  // option row (pills)
  optionRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  pill: { paddingVertical: 8, paddingHorizontal: 14, borderRadius: 20, borderWidth: 1, borderColor: colors.border },
  pillActive: { backgroundColor: colors.accent, borderColor: colors.accent },
  pillText: { color: colors.primary, fontSize: 14 },
  pillTextActive: { color: colors.background, fontWeight: '600' },

  // option grid (cards)
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  gridCard: { paddingVertical: 12, paddingHorizontal: 14, borderRadius: 8, borderWidth: 1, borderColor: colors.border, minWidth: '45%', flex: 1 },
  gridCardActive: { backgroundColor: colors.accent, borderColor: colors.accent },
  gridText: { color: colors.primary, fontSize: 14 },
  gridTextActive: { color: colors.background, fontWeight: '600' },

  // day blocks
  dayBlock: { backgroundColor: colors.surface, borderRadius: 10, borderWidth: 1, borderColor: colors.border, padding: 14, gap: 8 },
  dayHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  dayTitle: { ...typography.body, fontWeight: '700' },
  dayActions: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  dayAction: { color: colors.primary, fontSize: 16 },
  removeDay: { ...typography.caption, color: colors.secondary },

  // exercise rows
  exRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8, borderTopWidth: 1, borderTopColor: colors.border, gap: 8 },
  exRowDragging: { backgroundColor: colors.surface, borderRadius: 8, borderTopWidth: 0 },
  dragHandle: { paddingHorizontal: 4 },
  dragHandleIcon: { color: colors.secondary, fontSize: 20, letterSpacing: -1 },
  exInfo: { flex: 1 },
  exName: { ...typography.body },
  exMeta: { ...typography.caption, marginTop: 2 },
  exRemove: { color: colors.secondary, fontSize: 20, paddingLeft: 4 },

  // add exercise button
  addExBtn: { paddingVertical: 10, alignItems: 'center', borderWidth: 1, borderColor: colors.secondary, borderRadius: 8, marginTop: 4 },
  addExText: { color: colors.primary, fontSize: 14 },

  // picker
  picker: { gap: 8, marginTop: 4 },
  searchRow: { paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: colors.border },
  pendingName: { ...typography.body, fontWeight: '600', marginBottom: 12 },
  setsRepsRow: { gap: 12 },
  setsRepsInputs: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  numericField: { flex: 1, gap: 6 },
  numericLabel: { ...typography.caption },
  numericInput: { backgroundColor: colors.background, borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 12, color: colors.accent, fontSize: 18, textAlign: 'center' },
  times: { color: colors.secondary, fontSize: 18 },
  confirmRow: { flexDirection: 'row', gap: 8 },
  cancelBtn: { flex: 1, padding: 12, borderRadius: 8, borderWidth: 1, borderColor: colors.border, alignItems: 'center' },
  cancelText: { color: colors.secondary, fontSize: 14 },
  confirmBtn: { flex: 2, padding: 12, borderRadius: 8, backgroundColor: colors.accent, alignItems: 'center' },
  confirmText: { color: colors.background, fontSize: 14, fontWeight: '700' },

  // add day
  addDayBtn: { padding: 14, borderRadius: 10, borderWidth: 1, borderColor: colors.secondary, alignItems: 'center' },
  addDayText: { color: colors.primary, fontSize: 14 },

  // footer
  footer: { flexDirection: 'row', gap: 10, padding: 16 },
  backBtn: { flex: 1, padding: 16, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' },
  backBtnText: { color: colors.primary, fontSize: 15 },
  nextBtn: { flex: 2, padding: 16, borderRadius: 10, backgroundColor: colors.accent, alignItems: 'center' },
  nextBtnDisabled: { backgroundColor: colors.surface },
  nextBtnText: { color: colors.background, fontSize: 15, fontWeight: '700' },
});