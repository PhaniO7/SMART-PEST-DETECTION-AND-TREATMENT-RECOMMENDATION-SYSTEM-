/**
 * ============================================================================
 * VITE CONFIGURATION
 * Project: Smart Pest Detection & Treatment System
 * Author: Phani Nandan M S
           Vedhesha Parthiban
           Shivani Ballani
           Khadga Lohith K
 * ============================================================================
 * 
 * Purpose:
 * This file configures how Vite builds and runs our React frontend.
 * 
 * Key Settings:
 * - base ('./') : Uses relative paths so the app can easily be hosted anywhere 
 *                 (Vercel, GitHub Pages, etc.) without broken links.
 * - alias ('@') : Creates a shortcut to the 'src' folder for cleaner file 
 *                 imports (e.g., import X from '@/components/X').
 * ============================================================================
 */

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';

export default defineConfig(() => {
  return {
    // Step 1: Use relative paths so static hosting works in any folder or domain
    base: './',

    // Step 2: Enable React JSX transformation and Tailwind CSS v4 styling
    plugins: [react(), tailwindcss()],

    // Step 3: Map '@' alias to the project root directory
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },

    // Step 4: Configure development server reload settings
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
