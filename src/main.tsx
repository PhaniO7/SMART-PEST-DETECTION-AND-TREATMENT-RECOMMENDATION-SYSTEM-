/**
 * ============================================================================
 * REACT APPLICATION ENTRY POINT (main.tsx)
 * Project: Smart Pest Detection and Treatment Recommendation System
 * Author : Phani Nandan M S (5th-Semester B.Tech AI & DS)
 * ============================================================================
 *
 * FILE SUMMARY:
 * Connects our React application (<App />) to the HTML <div id="root"> element
 * inside index.html and loads our global Tailwind CSS stylesheet.
 * ============================================================================
 */

import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Step 1: Locate the root DOM container in index.html and mount the React UI
const rootElement = document.getElementById('root')!;
createRoot(rootElement).render(<App />);
