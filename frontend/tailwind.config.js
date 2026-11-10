/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        warm: {
          50: '#FAF9F6',   // Main background
          100: '#F3F0E9',  // Warm secondary background / surface
          200: '#E8E5DE',  // Primary border color
          300: '#D5D1C8',  // Stronger border
          400: '#B8B4AA',
          500: '#9C988E',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          warm: '#F3F0E9',
          muted: '#F7F6F2',
        },
        ink: {
          DEFAULT: '#292D2B', // Primary text
          secondary: '#777B75', // Secondary text
          muted: '#9CA09A',
          faint: '#D1D3CE',
        },
        accent: {
          green: '#71836B',   // Primary accent: muted natural green
          greenDark: '#566651',
          greenLight: '#E8ECE6',
          sand: '#C5A982',    // Secondary accent: warm sand
          sandDark: '#9F845F',
          sandLight: '#F5EFE6',
          terracotta: '#D47348', // GIS polygon outline accent
        },
        status: {
          success: '#5B7B61',
          warning: '#C98A2C',
          error: '#BD4B4B',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        serif: ['Newsreader', 'Georgia', 'serif'],
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(41, 45, 43, 0.05), 0 1px 2px -1px rgba(41, 45, 43, 0.05)',
        'elevated': '0 4px 12px -2px rgba(41, 45, 43, 0.08), 0 2px 6px -2px rgba(41, 45, 43, 0.04)',
        'floating': '0 12px 28px -4px rgba(41, 45, 43, 0.12), 0 4px 10px -2px rgba(41, 45, 43, 0.06)',
      }
    },
  },
  plugins: [],
}
