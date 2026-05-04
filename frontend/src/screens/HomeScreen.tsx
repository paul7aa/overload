import { useCallback, useRef, useState } from 'react';
import { FlatList, Modal, Pressable, Text, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { Swipeable } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../theme';
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
    <View className="flex-row items-center gap-1 ml-2">
      <Pressable
        className="w-[52px] justify-center items-center h-full"
        onPress={() => { closeAll(); navigation.navigate('AddProgram', { program: item }); }}
      >
        <Ionicons name="pencil" size={20} color={colors.primary} />
      </Pressable>
      <Pressable
        className="w-[52px] justify-center items-center h-full"
        onPress={() => { closeAll(); setPendingDelete(item); }}
      >
        <Ionicons name="trash" size={20} color={colors.danger} />
      </Pressable>
    </View>
  );

  return (
    <View className="flex-1 bg-background">
      <View
        className="flex-row justify-between items-center px-5 pb-4"
        style={{ paddingTop: top + 16 }}
      >
        <Text className="text-26 font-outfit-bold text-accent">overload</Text>
        <View className="flex-row items-center gap-3">
          <Pressable onPress={() => navigation.navigate('WorkoutHistory')}>
            <Ionicons name="time-outline" size={24} color={colors.primary} />
          </Pressable>
          <Pressable
            className="py-1.5 px-3.5 rounded-lg border border-border"
            onPress={() => { closeAll(); navigation.navigate('AddProgram', {}); }}
          >
            <Text className="text-primary text-sm">+ Add</Text>
          </Pressable>
        </View>
      </View>

      {programs.length === 0 ? (
        <View className="flex-1 items-center justify-center">
          <Text className="text-base font-outfit text-primary">No programs yet.</Text>
          <Text className="text-13 font-outfit text-secondary mt-1">Tap + Add to create one.</Text>
        </View>
      ) : (
        <FlatList
          data={programs}
          keyExtractor={p => p.id}
          contentContainerStyle={{ padding: 16, gap: 12 }}
          renderItem={({ item }) => (
            <Swipeable
              ref={r => { if (r) swipeRefs.current.set(item.id, r); else swipeRefs.current.delete(item.id); }}
              renderRightActions={() => renderActions(item)}
              friction={2}
              rightThreshold={40}
            >
              <Pressable
                className={`bg-surface rounded-[10px] p-4 border ${item.id === selectedId ? 'border-accent' : 'border-border'}`}
                onPress={() => selectProgram(item.id)}
              >
                <Text className="text-base font-outfit text-primary font-semibold mb-1">{item.name}</Text>
                <Text className="text-13 font-outfit text-secondary">{item.goal.join(', ')} · {item.level.join(', ')} · {item.lengthWeeks}w</Text>
                <Text className="text-13 font-outfit text-secondary">{item.equipment}</Text>
              </Pressable>
            </Swipeable>
          )}
        />
      )}

      {/* Dynamic marginBottom from safe area inset stays as style prop */}
      <Pressable
        className={`mx-4 mt-4 p-[18px] rounded-xl items-center ${!selectedProgram ? 'bg-surface' : 'bg-accent'}`}
        style={{ marginBottom: bottom || 16 }}
        disabled={!selectedProgram}
        onPress={() => selectedProgram && navigation.navigate('ActiveWorkout', { program: selectedProgram })}
      >
        <Text className="text-base font-bold text-background">
          {selectedProgram ? `Start — ${selectedProgram.name}` : 'Select a program'}
        </Text>
      </Pressable>

      <Modal visible={!!pendingDelete} transparent animationType="fade" onRequestClose={() => setPendingDelete(null)}>
        <Pressable className="flex-1 bg-black/60 justify-center px-8" onPress={() => setPendingDelete(null)}>
          <View
            className="bg-surface rounded-[14px] p-6 gap-2"
            style={{ marginBottom: bottom + 32 }}
          >
            <Text className="text-base font-outfit font-bold text-primary">Delete program?</Text>
            <Text className="text-13 font-outfit text-secondary leading-5">"{pendingDelete?.name}" will be permanently removed.</Text>
            <View className="flex-row gap-2.5 mt-2">
              <Pressable
                className="flex-1 p-3.5 rounded-[10px] border border-border items-center"
                onPress={() => setPendingDelete(null)}
              >
                <Text className="text-primary text-15">Cancel</Text>
              </Pressable>
              <Pressable
                className="flex-1 p-3.5 rounded-[10px] bg-danger items-center"
                onPress={() => deleteProgram(pendingDelete!.id)}
              >
                <Text className="text-white text-15 font-bold">Delete</Text>
              </Pressable>
            </View>
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}
