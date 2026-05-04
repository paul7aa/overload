import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated, KeyboardAvoidingView, Pressable,
  Text, TextInput, View, TouchableOpacity, ScrollView
} from 'react-native';
import DragList, { DragListRenderItemInfo } from 'react-native-draglist';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors } from '../theme';
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
      }}
    >
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
    <View
      className={`flex-row items-center py-2 border-t border-border gap-2 ${isActive ? 'bg-surface rounded-lg border-t-0' : ''}`}
      collapsable={false}
    >
      <Pressable onPress={() => removeExercise(dayIndex, ex.id)} hitSlop={8}>
        <Text className="text-secondary text-xl pl-1">×</Text>
      </Pressable>
      <View className="flex-1">
        <Text className="text-base font-outfit text-primary">{ex.name}</Text>
        <Text className="text-13 font-outfit text-secondary mt-0.5">{ex.sets} × {ex.reps}  ·  {ex.muscle}</Text>
      </View>
      <TouchableOpacity
        onPressIn={onDragStart}
        hitSlop={8}
        className="px-1"
      >
        <Text className="text-secondary text-xl" style={{ letterSpacing: -1 }}>≡</Text>
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
      onDragBegin={onDragBegin}
      onDragEnd={onDragEnd}
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

  const [isDragging, setIsDragging] = useState(false);

  const handleDragBegin = useCallback(() => setIsDragging(true), []);

  const handleDragEnd = useCallback(() => {
    setTimeout(() => {
      setIsDragging(false);
    }, 50);
  }, []);

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
    <KeyboardAvoidingView className="flex-1 bg-background" behavior="padding">
      <View className="flex-row gap-1.5 px-5 pt-4">
        {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
          <View key={i} className={`flex-1 h-[3px] rounded-sm ${i < step ? 'bg-accent' : 'bg-border'}`} />
        ))}
      </View>

      <ScrollView
        ref={scrollRef}
        contentContainerStyle={{ padding: 20, gap: 8 }}
        keyboardShouldPersistTaps="handled"
        scrollEnabled={!isDragging}
      >
        {step === 1 && <>
          <Text className="text-base font-outfit text-primary font-semibold mb-3">Name your program</Text>
          <TextInput
            className="bg-surface border border-border rounded-lg p-3 text-accent text-15"
            placeholder="e.g. 12-Week Strength Block"
            placeholderTextColor={colors.secondary}
            value={name}
            onChangeText={setName}
          />
          <Text className="text-base font-outfit text-primary font-semibold mb-3 mt-7">Level</Text>
          <OptionRow options={LEVELS} selected={level} onSelect={v => toggleMulti(v, level, setLevel)} />
        </>}

        {step === 2 && <>
          <Text className="text-base font-outfit text-primary font-semibold mb-3">Goal</Text>
          <OptionGrid options={GOALS} selected={goal} onSelect={v => toggleMulti(v, goal, setGoal)} />
        </>}

        {step === 3 && <>
          <Text className="text-base font-outfit text-primary font-semibold mb-3">Equipment</Text>
          <OptionGrid options={EQUIPMENT} selected={equipment} onSelect={setEquipment} />
        </>}

        {step === 4 && <>
          <Text className="text-base font-outfit text-primary font-semibold mb-3">Program length (weeks)</Text>
          <TextInput
            className="bg-surface border border-border rounded-lg p-3 text-accent text-15"
            keyboardType="numeric"
            value={lengthWeeks}
            onChangeText={setLengthWeeks}
            placeholderTextColor={colors.secondary}
          />
          <Text className="text-base font-outfit text-primary font-semibold mb-3 mt-7">Time per workout (min)</Text>
          <TextInput
            className="bg-surface border border-border rounded-lg p-3 text-accent text-15"
            keyboardType="numeric"
            value={timePerWorkout}
            onChangeText={setTimePerWorkout}
            placeholderTextColor={colors.secondary}
          />
        </>}

        {step === 5 && <>
          {days.map((day, di) => (
            <AnimatedDay key={`day-block-${day.dayNumber}`} onRemove={() => removeDay(di)}>
              {(animateRemove) => (
                <View className="bg-surface rounded-[10px] border border-border p-3.5 gap-2">
                  <View className="flex-row justify-between items-center mb-1">
                    <Text className="text-base font-outfit text-primary font-bold">Day {day.dayNumber}</Text>
                    <View className="flex-row items-center gap-3.5">
                      {di > 0 && (
                        <Pressable onPress={() => swapDays(di, di - 1)} hitSlop={8}>
                          <Text className="text-primary text-base">▲</Text>
                        </Pressable>
                      )}
                      {di < days.length - 1 && (
                        <Pressable onPress={() => swapDays(di, di + 1)} hitSlop={8}>
                          <Text className="text-primary text-base">▼</Text>
                        </Pressable>
                      )}
                      {days.length < MAX_DAYS && (
                        <Pressable onPress={() => duplicateDay(di)} hitSlop={8}>
                          <Text className="text-primary text-base">⧉</Text>
                        </Pressable>
                      )}
                      {days.length > 1 && (
                        <Pressable onPress={animateRemove} hitSlop={8}>
                          <Text className="text-13 font-outfit text-secondary">Remove</Text>
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
                    <View className="gap-2 mt-1">
                      {pendingEx === null ? (
                        <>
                          <TextInput
                            className="bg-surface border border-border rounded-lg p-3 text-accent text-15"
                            placeholder="Search exercises..."
                            placeholderTextColor={colors.secondary}
                            value={query}
                            onChangeText={setQuery}
                            autoFocus
                          />
                          {filtered.map(ex => (
                            <Pressable
                              key={ex.id}
                              className="py-3 border-b border-border"
                              onPress={() => setPendingEx(ex)}
                            >
                              <Text className="text-base font-outfit text-primary">{ex.name}</Text>
                              <Text className="text-13 font-outfit text-secondary">{ex.muscle}</Text>
                            </Pressable>
                          ))}
                          <Pressable
                            className="flex-1 p-3 rounded-lg border border-border items-center"
                            onPress={closePicker}
                          >
                            <Text className="text-secondary text-sm">Cancel</Text>
                          </Pressable>
                        </>
                      ) : (
                        <View className="gap-3">
                          <Text className="text-base font-outfit text-primary font-semibold mb-3">{pendingEx.name}</Text>
                          <View className="flex-row items-center gap-3">
                            <View className="flex-1 gap-1.5">
                              <Text className="text-13 font-outfit text-secondary">Sets</Text>
                              <TextInput
                                className="bg-background border border-border rounded-lg p-3 text-accent text-lg text-center"
                                keyboardType="numeric"
                                value={pendingSets}
                                onChangeText={setPendingSets}
                              />
                            </View>
                            <Text className="text-secondary text-lg">×</Text>
                            <View className="flex-1 gap-1.5">
                              <Text className="text-13 font-outfit text-secondary">Reps</Text>
                              <TextInput
                                className="bg-background border border-border rounded-lg p-3 text-accent text-lg text-center"
                                keyboardType="numeric"
                                value={pendingReps}
                                onChangeText={setPendingReps}
                              />
                            </View>
                          </View>
                          <View className="flex-row gap-2">
                            <Pressable
                              className="flex-1 p-3 rounded-lg border border-border items-center"
                              onPress={() => setPendingEx(null)}
                            >
                              <Text className="text-secondary text-sm">Back</Text>
                            </Pressable>
                            <Pressable
                              className="flex-[2] p-3 rounded-lg bg-accent items-center"
                              onPress={confirmAdd}
                            >
                              <Text className="text-background text-sm font-bold">Add</Text>
                            </Pressable>
                          </View>
                        </View>
                      )}
                    </View>
                  ) : (
                    <Pressable
                      className="py-2.5 items-center border border-secondary rounded-lg mt-1"
                      onPress={() => openPicker(di)}
                    >
                      <Text className="text-primary text-sm">+ Add Exercise</Text>
                    </Pressable>
                  )}
                </View>
              )}
            </AnimatedDay>
          ))}

          {days.length < MAX_DAYS && (
            <Pressable
              className="p-3.5 rounded-[10px] border border-secondary items-center"
              onPress={addDay}
            >
              <Text className="text-primary text-sm">+ Add Day</Text>
            </Pressable>
          )}
        </>}
      </ScrollView>

      <View className="flex-row gap-2.5 p-4" style={{ paddingBottom: bottom || 16 }}>
        {step > 1 && (
          <Pressable
            className="flex-1 p-4 rounded-[10px] border border-border items-center"
            onPress={() => setStep(s => s - 1)}
          >
            <Text className="text-primary text-15">Back</Text>
          </Pressable>
        )}
        <Pressable
          className={`flex-[2] p-4 rounded-[10px] items-center ${!canAdvance() ? 'bg-surface' : 'bg-accent'}`}
          disabled={!canAdvance()}
          onPress={() => step < TOTAL_STEPS ? setStep(s => s + 1) : save()}
        >
          <Text className="text-background text-15 font-bold">{step < TOTAL_STEPS ? 'Next' : 'Save'}</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

function OptionRow({ options, selected, onSelect }: { options: string[]; selected: string[]; onSelect: (o: string) => void }) {
  return (
    <View className="flex-row flex-wrap gap-2">
      {options.map(o => (
        <Pressable
          key={o}
          className={`py-2 px-3.5 rounded-[20px] border ${selected.includes(o) ? 'bg-accent border-accent' : 'border-border'}`}
          onPress={() => onSelect(o)}
        >
          <Text className={`text-sm ${selected.includes(o) ? 'text-background font-semibold' : 'text-primary'}`}>{o}</Text>
        </Pressable>
      ))}
    </View>
  );
}

function OptionGrid({ options, selected, onSelect }: { options: string[]; selected: string | string[] | null; onSelect: (o: string) => void }) {
  const active = (o: string) => Array.isArray(selected) ? selected.includes(o) : o === selected;
  return (
    <View className="flex-row flex-wrap gap-2">
      {options.map(o => (
        <Pressable
          key={o}
          className={`py-3 px-3.5 rounded-lg border flex-1 min-w-[45%] ${active(o) ? 'bg-accent border-accent' : 'border-border'}`}
          onPress={() => onSelect(o)}
        >
          <Text className={`text-sm ${active(o) ? 'text-background font-semibold' : 'text-primary'}`}>{o}</Text>
        </Pressable>
      ))}
    </View>
  );
}
