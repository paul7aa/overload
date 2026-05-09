import './global.css';
import { DefaultTheme, NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { useFonts, Outfit_400Regular, Outfit_600SemiBold, Outfit_700Bold } from '@expo-google-fonts/outfit';
import { useEffect, useState } from 'react';
import { View } from 'react-native';
import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import Constants from 'expo-constants';
import { colors } from './src/theme';

const BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';
const API_KEY = process.env.EXPO_PUBLIC_API_KEY ?? '';

async function registerPushToken() {
  if (!Device.isDevice) {
    console.log('[push-token] skipped — not a physical device');
    return;
  }

  const { status } = await Notifications.getPermissionsAsync();
  if (status !== 'granted') {
    console.log('[push-token] skipped — permission not granted:', status);
    return;
  }

  const projectId = Constants.expoConfig?.extra?.eas?.projectId;
  console.log('[push-token] fetching token, projectId:', projectId);

  try {
    const { data: token } = await Notifications.getExpoPushTokenAsync({ projectId });
    console.log('[push-token] got token:', token);

    const resp = await fetch(`${BASE}/register-push-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
      body: JSON.stringify({ token }),
    });
    console.log('[push-token] registered:', resp.status);
  } catch (err) {
    console.warn('[push-token] failed:', err);
  }
}

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
  }),
});
import { RootStackParamList } from './src/types';
import { seedDevData } from './src/dev/seed';
import HomeScreen from './src/screens/HomeScreen';
import ActiveWorkoutScreen from './src/screens/ActiveWorkoutScreen';
import WorkoutCompleteScreen from './src/screens/WorkoutCompleteScreen';
import AddProgramModal from './src/screens/AddProgramModal';
import WorkoutHistoryScreen from './src/screens/WorkoutHistoryScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

const navTheme = { ...DefaultTheme, colors: { ...DefaultTheme.colors, background: colors.background } };

export default function App() {
  const [fontsLoaded] = useFonts({ Outfit_400Regular, Outfit_600SemiBold, Outfit_700Bold });
  const [devReady, setDevReady] = useState(!__DEV__);

  useEffect(() => {
    if (__DEV__) seedDevData().then(() => setDevReady(true));
    Notifications.requestPermissionsAsync().then(registerPushToken);
    Notifications.setNotificationChannelAsync('workout', {
      name: 'Workout',
      importance: Notifications.AndroidImportance.HIGH,
    });
  }, []);

  if (!fontsLoaded || !devReady) return <View style={{ flex: 1, backgroundColor: colors.background }} />;

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
    <NavigationContainer theme={navTheme}>
      <StatusBar style="light" />
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: { backgroundColor: colors.background },
          headerTintColor: colors.accent,
          headerTitleStyle: { fontFamily: 'Outfit_700Bold' },
          contentStyle: { backgroundColor: colors.background },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
        <Stack.Screen name="ActiveWorkout" component={ActiveWorkoutScreen} options={{ title: 'Workout', headerBackTitle: '' }} />
        <Stack.Screen name="WorkoutComplete" component={WorkoutCompleteScreen} options={{ title: 'Complete', headerBackVisible: false }} />
        <Stack.Screen name="WorkoutHistory" component={WorkoutHistoryScreen} options={{ title: 'History' }} />
        <Stack.Screen name="AddProgram" component={AddProgramModal} options={{ presentation: 'transparentModal', title: 'New Program' }} />
      </Stack.Navigator>
    </NavigationContainer>
    </GestureHandlerRootView>
  );
}
