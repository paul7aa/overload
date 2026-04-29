import { StyleSheet, Text, View } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { colors, typography } from '../theme';
import { RootStackParamList } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'WorkoutComplete'>;

export default function WorkoutCompleteScreen({ route }: Props) {
  const { logs } = route.params;
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Workout Complete</Text>
      <Text style={typography.caption}>{logs.length} exercises logged</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, alignItems: 'center', justifyContent: 'center', gap: 8 },
  title: { ...typography.heading },
});
