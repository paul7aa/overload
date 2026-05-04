/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './App.{js,jsx,ts,tsx}',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        background: '#0f0f0f',
        surface: '#1c1c1c',
        border: '#2a2a2a',
        primary: '#e0e0e0',
        secondary: '#888888',
        accent: '#ffffff',
        danger: '#ff4444',
        navy: '#1a1a2e',
      },
      fontFamily: {
        outfit: ['Outfit_400Regular'],
        'outfit-semibold': ['Outfit_600SemiBold'],
        'outfit-bold': ['Outfit_700Bold'],
      },
      fontSize: {
        '11': ['11px', { lineHeight: '14px' }],
        '13': ['13px', { lineHeight: '18px' }],
        '15': ['15px', { lineHeight: '20px' }],
        '17': ['17px', { lineHeight: '22px' }],
        '22': ['22px', { lineHeight: '28px' }],
        '26': ['26px', { lineHeight: '32px' }],
        '28': ['28px', { lineHeight: '36px' }],
      },
    },
  },
  plugins: [],
};
