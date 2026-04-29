export const colors = {
  background: '#0f0f0f',
  surface: '#1c1c1c',
  border: '#2a2a2a',
  primary: '#e0e0e0',
  secondary: '#888888',
  accent: '#ffffff',
};

export const typography = {
  heading: { fontSize: 22, fontWeight: '700' as const, color: colors.accent },
  body: { fontSize: 16, fontWeight: '400' as const, color: colors.primary },
  caption: { fontSize: 13, fontWeight: '400' as const, color: colors.secondary },
};
