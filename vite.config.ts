/**
 * ============================================================================
 * VITE BUNDLER & DEVELOPMENT SERVER CONFIGURATION
 * Project: Smart Pest Detection and Treatment Recommendation System
 * Author : Phani Nandan M S (5th-Semester B.Tech AI & DS)
 * ============================================================================
 *
 * FILE SUMMARY:
 * Configures Vite to bundle our React + TypeScript + Tailwind CSS frontend.
 * - Uses relative asset paths (base: './') so the built website works on any
 *   static host (Vercel, GitHub Pages, Netlify, or local preview).
 * - Configures the '@' path alias for clean imports across the project.
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
