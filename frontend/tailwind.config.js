/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Navy/Dark backgrounds for dashboard theme
        navy: {
          50: '#e6e9f0',
          100: '#ccd3e1',
          200: '#99a7c3',
          300: '#667ba5',
          400: '#334f87',
          500: '#002369', // Primary navy
          600: '#001c54',
          700: '#00153f',
          800: '#000e2a',
          900: '#000715',
        },
        // Risk level color coding
        risk: {
          safe: '#10b981',      // Green - SAFE
          caution: '#f59e0b',   // Amber - CAUTION
          warning: '#f97316',   // Orange - WARNING
          critical: '#ef4444',  // Red - CRITICAL
        },
      },
      fontFamily: {
        // Custom font families
        heading: ['Orbitron', 'sans-serif'],  // For headings
        body: ['Inter', 'sans-serif'],        // For body text
        mono: ['JetBrains Mono', 'monospace'], // For monospace
      },
      screens: {
        // Minimum viewport width of 1024px (tablet landscape)
        'tablet': '1024px',
        'laptop': '1280px',
        'desktop': '1536px',
      },
    },
  },
  plugins: [],
}
