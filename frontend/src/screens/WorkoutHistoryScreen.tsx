import { useCallback, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, typography } from '../theme';
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
    <Pressable style={styles.card} onPress={() => setExpanded(e => !e)}>
      <View style={styles.cardHeader}>
        <View style={styles.cardLeft}>
          <Text style={styles.cardTitle}>{record.programName} · Day {record.dayNumber}</Text>
          <Text style={styles.cardMeta}>{time} · {formatDuration(record.durationSeconds)}</Text>
        </View>
        <Text style={styles.chevron}>{expanded ? '▲' : '▼'}</Text>
      </View>

      <View style={styles.cardStats}>
        <Text style={styles.cardStat}>{record.exercises.length} exercises</Text>
        <Text style={styles.cardStatSep}>·</Text>
        <Text style={styles.cardStat}>{totalSets} sets</Text>
        <Text style={styles.cardStatSep}>·</Text>
        <Text style={styles.cardStat}>{Math.round(totalVolume).toLocaleString()} kg volume</Text>
      </View>

      {expanded && (
        <View style={styles.exerciseList}>
          {record.exercises.map((ex, i) => (
            <View key={i} style={styles.exerciseRow}>
              <View style={styles.exerciseRowHeader}>
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
        </View>
      )}
    </Pressable>
  );
}

export default function WorkoutHistoryScreen() {
  const { bottom } = useSafeAreaInsets();
  const [groups, setGroups] = useState<{ label: string; items: WorkoutRecord[] }[]>([]);

  useFocusEffect(useCallback(() => {
    AsyncStorage.getItem(HISTORY_KEY).then(raw => {
      const records: WorkoutRecord[] = raw ? JSON.parse(raw) : [];
      setGroups(groupByDate(records));
    });
  }, []));

  if (groups.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No workouts yet.</Text>
        <Text style={typography.caption}>Complete a workout to see your history here.</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={[styles.scroll, { paddingBottom: bottom + 16 }]}>
      {groups.map(({ label, items }) => (
        <View key={label}>
          <Text style={styles.groupLabel}>{label}</Text>
          {items.map(r => <WorkoutCard key={r.id} record={r} />)}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { padding: 16, gap: 8 },

  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 },
  emptyText: { ...typography.body, fontWeight: '600' as const },

  groupLabel: { ...typography.caption, fontWeight: '600' as const, marginTop: 12, marginBottom: 6, marginLeft: 4 },

  card: {
    backgroundColor: colors.surface, borderRadius: 12,
    borderWidth: 1, borderColor: colors.border, padding: 14, gap: 6,
  },
  cardHeader: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' },
  cardLeft: { flex: 1, gap: 2 },
  cardTitle: { ...typography.body, fontWeight: '700' as const },
  cardMeta: { ...typography.caption, fontSize: 12 },
  chevron: { ...typography.caption, fontSize: 11, marginLeft: 8 },

  cardStats: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  cardStat: { ...typography.caption, fontSize: 12, color: colors.secondary },
  cardStatSep: { ...typography.caption, fontSize: 12, color: colors.border },

  exerciseList: { marginTop: 8, gap: 10, borderTopWidth: 1, borderTopColor: colors.border, paddingTop: 10 },
  exerciseRow: { gap: 4 },
  exerciseRowHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  exerciseName: { ...typography.caption, fontWeight: '600' as const, color: colors.primary, flex: 1 },
  exerciseMuscle: { ...typography.caption, fontSize: 11 },
  setRow: { flexDirection: 'row', alignItems: 'center', paddingLeft: 8, gap: 8 },
  setNum: { ...typography.caption, width: 44, fontSize: 12 },
  setDetail: { ...typography.caption, flex: 1, textAlign: 'center', fontSize: 12 },
});