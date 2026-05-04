import { useCallback, useRef, useState } from 'react';
import { Alert, Pressable, ScrollView, Text, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Swipeable } from 'react-native-gesture-handler';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../theme';
import { WorkoutRecord } from '../types';
import { HISTORY_KEY } from './WorkoutCompleteScreen';

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

function groupByDate(records: WorkoutRecord[]): { label: string; items: WorkoutRecord[] }[] {
  const now = new Date();
  const todayStr = now.toDateString();
  const yesterdayStr = new Date(now.getTime() - 86400000).toDateString();
  const weekAgo = now.getTime() - 7 * 86400000;

  const groups: Record<string, WorkoutRecord[]> = {};
  for (const r of records) {
    const d = new Date(r.completedAt);
    let label: string;
    if (d.toDateString() === todayStr) label = 'Today';
    else if (d.toDateString() === yesterdayStr) label = 'Yesterday';
    else if (d.getTime() > weekAgo) label = 'This week';
    else label = d.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' });
    (groups[label] ??= []).push(r);
  }

  const order = ['Today', 'Yesterday', 'This week'];
  return Object.entries(groups).sort(([a], [b]) => {
    const ai = order.indexOf(a), bi = order.indexOf(b);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return 0;
  }).map(([label, items]) => ({ label, items }));
}

function WorkoutCard({ record }: { record: WorkoutRecord }) {
  const [expanded, setExpanded] = useState(false);
  const totalSets = record.exercises.reduce((n, ex) => n + ex.sets.length, 0);
  const totalVolume = record.exercises.reduce(
    (n, ex) => n + ex.sets.reduce((s, set) => s + set.weight * set.reps, 0),
    0,
  );
  const time = new Date(record.completedAt).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

  return (
    <Pressable
      className="bg-surface rounded-xl border border-border p-3.5 gap-1.5"
      onPress={() => setExpanded(e => !e)}
    >
      <View className="flex-row items-start justify-between">
        <View className="flex-1 gap-0.5">
          <Text className="text-base font-outfit text-primary font-bold">{record.programName} · Day {record.dayNumber}</Text>
          <Text className="text-xs font-outfit text-secondary">Week {record.weekNumber} · {time} · {formatDuration(record.durationSeconds)}</Text>
        </View>
        <Text className="text-11 font-outfit text-secondary ml-2">{expanded ? '▲' : '▼'}</Text>
      </View>

      <View className="flex-row items-center gap-1.5">
        <Text className="text-xs font-outfit text-secondary">{record.exercises.length} exercises</Text>
        <Text className="text-xs font-outfit text-border">·</Text>
        <Text className="text-xs font-outfit text-secondary">{totalSets} sets</Text>
        <Text className="text-xs font-outfit text-border">·</Text>
        <Text className="text-xs font-outfit text-secondary">{Math.round(totalVolume).toLocaleString()} kg volume</Text>
      </View>

      {expanded && (
        <View className="mt-2 gap-2.5 border-t border-border pt-2.5">
          {record.exercises.map((ex, i) => (
            <View key={i} className="gap-1">
              <View className="flex-row justify-between items-center">
                <Text className="text-13 font-outfit text-primary font-semibold flex-1">{ex.name}</Text>
                <Text className="text-11 font-outfit text-secondary">{ex.muscle}</Text>
              </View>
              {ex.sets.map((set, si) => (
                <View key={si} className="flex-row items-center pl-2 gap-2">
                  <Text className="text-xs font-outfit text-secondary w-11">Set {si + 1}</Text>
                  <Text className="text-xs font-outfit text-secondary flex-1 text-center">{set.weight > 0 ? `${set.weight} kg` : 'BW'}</Text>
                  <Text className="text-xs font-outfit text-secondary flex-1 text-center">{set.reps} reps</Text>
                  <Text className="text-xs font-outfit text-secondary flex-1 text-center">RPE {set.rpe}</Text>
                </View>
              ))}
            </View>
          ))}
        </View>
      )}
    </Pressable>
  );
}

function LifetimeStats({ records }: { records: WorkoutRecord[] }) {
  const totalWorkouts = records.length;
  const totalSets = records.reduce((n, r) => n + r.exercises.reduce((m, ex) => m + ex.sets.length, 0), 0);
  const totalVolume = records.reduce(
    (n, r) => n + r.exercises.reduce((m, ex) => m + ex.sets.reduce((s, set) => s + set.weight * set.reps, 0), 0),
    0,
  );
  const totalMinutes = Math.round(records.reduce((n, r) => n + r.durationSeconds, 0) / 60);

  const fmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);

  return (
    <View className="mx-4 mb-0 rounded-[14px] bg-navy border border-accent/[19%] p-5 gap-4">
      <Text className="text-11 font-outfit text-accent tracking-widest uppercase">Lifetime</Text>
      <View className="flex-row flex-wrap">
        <View className="w-1/2 py-2.5 px-1 gap-1">
          <Text className="text-[28px] font-outfit-bold text-accent">{totalWorkouts}</Text>
          <Text className="text-xs font-outfit text-secondary">Workouts</Text>
        </View>
        <View className="w-1/2 py-2.5 px-1 gap-1">
          <Text className="text-[28px] font-outfit-bold text-accent">{fmt(totalSets)}</Text>
          <Text className="text-xs font-outfit text-secondary">Sets</Text>
        </View>
        <View className="w-1/2 py-2.5 px-1 gap-1">
          <Text className="text-[28px] font-outfit-bold text-accent">{fmt(Math.round(totalVolume))}</Text>
          <Text className="text-xs font-outfit text-secondary">kg Volume</Text>
        </View>
        <View className="w-1/2 py-2.5 px-1 gap-1">
          <Text className="text-[28px] font-outfit-bold text-accent">{fmt(totalMinutes)}</Text>
          <Text className="text-xs font-outfit text-secondary">Minutes</Text>
        </View>
      </View>
    </View>
  );
}

export default function WorkoutHistoryScreen() {
  const { bottom } = useSafeAreaInsets();
  const [records, setRecords] = useState<WorkoutRecord[]>([]);
  const [groups, setGroups] = useState<{ label: string; items: WorkoutRecord[] }[]>([]);
  const swipeRefs = useRef<Map<string, Swipeable>>(new Map());

  const load = useCallback(() => {
    AsyncStorage.getItem(HISTORY_KEY).then(raw => {
      const all: WorkoutRecord[] = raw ? JSON.parse(raw) : [];
      setRecords(all);
      setGroups(groupByDate(all));
    });
  }, []);

  useFocusEffect(load);

  const deleteRecord = useCallback((id: string) => {
    setRecords(prev => {
      const next = prev.filter(r => r.id !== id);
      AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      setGroups(groupByDate(next));
      return next;
    });
  }, []);

  if (groups.length === 0) {
    return (
      <View className="flex-1 items-center justify-center gap-3">
        <Text className="text-base font-outfit text-primary font-semibold">No workouts yet.</Text>
        <Text className="text-13 font-outfit text-secondary">Complete a workout to see your history here.</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={{ padding: 16, gap: 8, paddingBottom: bottom + 16 }}>
      <LifetimeStats records={records} />
      {groups.map(({ label, items }) => (
        <View key={label}>
          <Text className="text-13 font-outfit text-secondary font-semibold mt-3 mb-1.5 ml-1">{label}</Text>
          <View className="gap-2">
            {items.map(r => (
              <Swipeable
                key={r.id}
                ref={ref => { if (ref) swipeRefs.current.set(r.id, ref); else swipeRefs.current.delete(r.id); }}
                friction={2}
                rightThreshold={40}
                renderRightActions={() => (
                  <Pressable
                    className="w-16 justify-center items-center ml-2"
                    onPress={() => {
                      swipeRefs.current.get(r.id)?.close();
                      Alert.alert(
                        'Delete workout?',
                        'This will permanently remove this session from your history.',
                        [
                          { text: 'Cancel', style: 'cancel' },
                          { text: 'Delete', style: 'destructive', onPress: () => deleteRecord(r.id) },
                        ]
                      );
                    }}
                  >
                    <Ionicons name="trash" size={20} color={colors.danger} />
                  </Pressable>
                )}
              >
                <WorkoutCard record={r} />
              </Swipeable>
            ))}
          </View>
        </View>
      ))}
    </ScrollView>
  );
}
