module.exports = {
  expo: {
    name: "Overload",
    slug: "overload",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "dark",
    backgroundColor: "#0f0f0f",
    newArchEnabled: true,
    splash: {
      image: "./assets/splash-icon.png",
      resizeMode: "contain",
      backgroundColor: "#0f0f0f"
    },
    ios: {
      supportsTablet: true
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#0f0f0f"
      },
      googleServicesFile: process.env.GOOGLE_SERVICES_JSON || "./android/app/google-services.json",
      edgeToEdgeEnabled: true,
      package: "com.paul.overload"
    },
    web: {
      favicon: "./assets/favicon.png"
    },
    plugins: [
      "expo-font",
      "expo-notifications"
    ],
    extra: {
      eas: {
        projectId: "419352c2-1ef9-41ef-9240-ce9b5c0e8230"
      }
    },
    owner: "paul7aa"
  }
};