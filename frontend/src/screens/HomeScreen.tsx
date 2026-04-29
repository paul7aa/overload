import { useCallback, useRef, useState } from 'react';
import { FlatList, Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { Swipeable } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography } from '../theme';
import { Program, RootStackParamList } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

const PROGRAMS_KEY = 'programs';
const SELECTED_KEY = 'selected_program_id';

export default function HomeScreen({ navigation }: Props) {
  const { top, bottom } = useSafeAreaInsets();
  const [programs, setPrograms] = useState<Program[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Program | null>(null);
  const swipeRefs = useRef<Map<string, Swipeable>>(new Map());

  useFocusEffect(useCallback(() => {
    const load = async () => {
      const [raw, saved] = await Promise.all([
        AsyncStorage.getItem(PROGRAMS_KEY),
        AsyncStorage.getItem(SELECTED_KEY),
      ]);
      if (raw) setPrograms(JSON.parse(raw));
      if (saved) setSelectedId(saved);
    };
    load();
  }, []));

  const closeAll = () => swipeRefs.current.forEach(r => r.close());

  const selectProgram = async (id: string) => {
    closeAll();
    setSelectedId(id);
    await AsyncStorage.setItem(SELECTED_KEY, id);
  };

  const deleteProgram = async (id: string) => {
    const updated = programs.filter(p => p.id !== id);
    setPrograms(updated);
    setPendingDelete(null);
    await AsyncStorage.setItem(PROGRAMS_KEY, JSON.stringify(updated));
    if (selectedId === id) {
      setSelectedId(null);
      await AsyncStorage.removeItem(SELECTED_KEY);
    }
  };

  const selectedProgram = programs.find(p => p.id === selectedId);

  const renderActions = (item: Program) => (
    <View style={styles.actions}>
      <Pressable
        style={styles.action}
        onPress={() => { closeAll(); navigation.navigate('AddProgram', { program: item }); }}
      >
        <Ionicons name="pencil" size={20} color={colors.primary} />
      </Pressable>
      <Pressable
        style={styles.action}
        onPress={() => { closeAll(); setPendingDelete(item); }}
      >
        <Ionicons name="trash" size={20} color="#ff4444" />
      </Pressable>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={[styles.header, { paddingTop: top + 16 }]}>
        <Text style={styles.title}>overload</Text>
        <View style={styles.headerActions}>
          <Pressable onPress={() => navigation.navigate('WorkoutHistory')}>
            <Ionicons name="time-outline" size={24} color={colors.primary} />
          </Pressable>
          <Pressable style={styles.addBtn} onPress={() => { closeAll(); navigation.navigate('AddProgram', {}); }}>
            <Text style={styles.addBtnText}>+ Add</Text>
          </Pressable>
        </View>
      </View>

      {programs.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>No programs yet.</Text>
          <Text style={[typography.caption, { marginTop: 4 }]}>Tap + Add to create one.</Text>
        </View>
      ) : (
        <FlatList
          data={programs}
          keyExtractor={p => p.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <Swipeable
              ref={r => { if (r) swipeRefs.current.set(item.id, r); else swipeRefs.current.delete(item.id); }}
              renderRightActions={() => renderActions(item)}
              friction={2}
              rightThreshold={40}
            >
              <Pressable
                style={[styles.card, item.id === selectedId && styles.cardSelected]}
                onPress={() => selectProgram(item.id)}
              >
                <Text style={styles.cardName}>{item.name}</Text>
                <Text style={styles.cardMeta}>{item.goal.join(', ')} · {item.level.join(', ')} · {item.lengthWeeks}w</Text>
                <Text style={styles.cardMeta}>{item.equipment}</Text>
              </Pressable>
            </Swipeable>
          )}
        />
      )}

      <Pressable
        style={[styles.startBtn, !selectedProgram && styles.startBtnDisabled, { marginBottom: bottom || 16 }]}
        disabled={!selectedProgram}
        onPress={() => selectedProgram && navigation.navigate('ActiveWorkout', { program: selectedProgram })}
      >
        <Text style={styles.startBtnText}>
          {selectedProgram ? `Start — ${selectedProgram.name}` : 'Select a program'}
        </Text>
      </Pressable>

      <Modal visible={!!pendingDelete} transparent animationType="fade" onRequestClose={() => setPendingDelete(null)}>
        <Pressable style={styles.overlay} onPress={() => setPendingDelete(null)}>
          <View style={[styles.dialog, { marginBottom: bottom + 32 }]}>
            <Text style={styles.dialogTitle}>Delete program?</Text>
            <Text style={styles.dialogBody}>"{pendingDelete?.name}" will be permanently removed.</Text>
            <View style={styles.dialogActions}>
              <Pressable style={styles.dialogCancel} onPress={() => setPendingDelete(null)}>
                <Text style={styles.dialogCancelText}>Cancel</Text>
              </Pressable>
              <Pressable style={styles.dialogDelete} onPress={() => deleteProgram(pendingDelete!.id)}>
                <Text style={styles.dialogDeleteText}>Delete</Text>
              </Pressable>
            </View>
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  title: { ...typography.heading, fontSize: 26 },
  headerActions: { flexDirection: 'row' as const, alignItems: 'center', gap: 12 },
  addBtn: { paddingVertical: 6, paddingHorizontal: 14, borderRadius: 8, borderWidth: 1, borderColor: colors.border },
  addBtnText: { color: colors.primary, fontSize: 14 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  emptyText: { ...typography.body },
  list: { padding: 16, gap: 12 },
  card: {
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: 16,
    borderWidth: 1,
    borderColor: colors.border,
  },
  cardSelected: { borderColor: colors.accent },
  cardName: { ...typography.body, fontWeight: '600', marginBottom: 4 },
  cardMeta: { ...typography.caption },
  actions: { flexDirection: 'row', alignItems: 'center', gap: 4, marginLeft: 8 },
  action: { width: 52, justifyContent: 'center', alignItems: 'center', height: '100%' },
  startBtn: {
    margin: 16,
    padding: 18,
    backgroundColor: colors.accent,
    borderRadius: 12,
    alignItems: 'center',
  },
  startBtnDisabled: { backgroundColor: colors.surface },
  startBtnText: { fontSize: 16, fontWeight: '700', color: colors.background },
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', paddingHorizontal: 32 },
  dialog: {
    backgroundColor: colors.surface,
    borderRadius: 14,
    padding: 24,
    gap: 8,
  },
  dialogTitle: { ...typography.body, fontWeight: '700' },
  dialogBody: { ...typography.caption, lineHeight: 20 },
  dialogActions: { flexDirection: 'row', gap: 10, marginTop: 8 },
  dialogCancel: { flex: 1, padding: 14, borderRadius: 10, borderWidth: 1, borderColor: colors.border, alignItems: 'center' },
  dialogCancelText: { color: colors.primary, fontSize: 15 },
  dialogDelete: { flex: 1, padding: 14, borderRadius: 10, backgroundColor: '#ff4444', alignItems: 'center' },
  dialogDeleteText: { color: '#fff', fontSize: 15, fontWeight: '700' },
});
