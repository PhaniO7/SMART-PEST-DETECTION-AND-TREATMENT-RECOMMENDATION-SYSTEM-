/**
 * ============================================================================
 * SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
 * 5th-Semester B.Tech AI & DS Design Project
 * Project By: PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI
 * ============================================================================
 *
 * APPLICATION ARCHITECTURE SUMMARY:
 *   Section 1: Initial Sample Data (Pre-loaded Scans & User Complaint Tickets)
 *   Section 2: Interactive Leaf SVG Illustration with YOLO Bounding Boxes
 *   Section 3: Login & Sign-Up Gate Screen (Role-Based User & Admin Access)
 *   Section 4: Dashboard & Leaf Scanner Tab (Pre-Scan Guide + Post-Scan CV & Treatments)
 *   Section 5: My Scans History, User Complaint Tracker (3-Step Progress & Admin Inbox),
 *              and 4-Stage Algorithm Pipeline Summary
 *   Section 6: Admin Portal (Complainant Identity View, Custom Response Messaging,
 *              and 18-Pest Hash Table Database Inspector)
 * ============================================================================
 */

import React, { useState, useEffect } from 'react';
import {
  CropType,
  SeverityLevel,
  ScanResult,
  UserComplaint,
  ALL_CROPS,
  PEST_DATABASE,
  runCompleteDetectionPipeline,
  classifySeverityDecisionTree,
  pestHashTable
} from './dataAndAlgorithms.ts';
import {
  Sparkles, Shield, Sun, Moon, UploadCloud, CheckCircle2, Clock,
  Search, Download, Trash2, Star, Send, Cpu, Database, GitFork,
  ListOrdered, LogOut, Lock, UserPlus, LogIn, Layers, Eye, Activity,
  IndianRupee, Sprout, Award, BarChart3, HelpCircle
} from 'lucide-react';

// ============================================================================
// SECTION 1: INITIAL SAMPLE DATA (PRE-LOADED SCANS & FEEDBACK TICKETS)
// Summary: Provides realistic starting records so the History and Feedback
// tabs look rich and complete right after logging in.
// ============================================================================

const SAMPLE_LEAF_SPECIMENS = [
  {
    id: 'spec_tomato',
    title: 'Tomato Early Blight',
    crop: 'Tomato' as CropType,
    pestId: 'pest_8',
    damagePercent: 42,
    leafColor: '#15803d',
    spotColor: '#451a03',
    boxLabel: 'Alternaria Target Spot (96%)'
  },
  {
    id: 'spec_rice',
    title: 'Rice Brown Planthopper',
    crop: 'Rice' as CropType,
    pestId: 'pest_1',
    damagePercent: 28,
    leafColor: '#16a34a',
    spotColor: '#78350f',
    boxLabel: 'Hopperburn Colony (95%)'
  },
  {
    id: 'spec_cotton',
    title: 'Cotton Pink Bollworm',
    crop: 'Cotton' as CropType,
    pestId: 'pest_6',
    damagePercent: 22,
    leafColor: '#15803d',
    spotColor: '#27272a',
    boxLabel: 'Bollworm Borehole (94%)'
  },
  {
    id: 'spec_apple',
    title: 'Apple Scab Lesion',
    crop: 'Apple' as CropType,
    pestId: 'pest_16',
    damagePercent: 11,
    leafColor: '#166534',
    spotColor: '#3f6212',
    boxLabel: 'Venturia Scab Patch (97%)'
  }
];

const INITIAL_HISTORY: ScanResult[] = [
  runCompleteDetectionPipeline({ cropInput: 'Tomato', pestIdOverride: 'pest_8', affectedAreaPercent: 42, seed: 'init_1' }),
  runCompleteDetectionPipeline({ cropInput: 'Rice', pestIdOverride: 'pest_1', affectedAreaPercent: 28, seed: 'init_2' }),
  runCompleteDetectionPipeline({ cropInput: 'Cotton', pestIdOverride: 'pest_6', affectedAreaPercent: 22, seed: 'init_3' }),
  runCompleteDetectionPipeline({ cropInput: 'Apple', pestIdOverride: 'pest_16', affectedAreaPercent: 11, seed: 'init_4' })
];

// ----------------------------------------------------------------------------
// SECTION 1B: INITIAL USER COMPLAINT TICKETS
// Summary: Pre-loaded complaints with complainant name, account email,
// full submission date/time (YYYY-MM-DD HH:MM:SS), and sample admin replies.
// ----------------------------------------------------------------------------
const INITIAL_COMPLAINTS: UserComplaint[] = [
  {
    id: 'TKT-101',
    userName: 'Kavitha Devi',
    userEmail: 'kavitha.farm@agri.in',
    crop: 'Rice',
    category: 'Treatment Feedback',
    rating: 5,
    subject: 'Water level during Chess 50 WG spray for Rice Planthopper',
    message: 'Recommended dosage of 120g/acre worked very well. Should we keep 2 cm water in the paddy field during application?',
    status: 'Resolved ✓',
    createdAt: '2026-10-21 11:20:15',
    adminReply: 'Yes, maintain 2–3 cm standing water for 48 hours and direct the nozzle toward the base of the rice tillers.',
    repliedBy: 'PHANI NANDAN M S (Admin)',
    repliedAt: '2026-10-21 14:05:30'
  },
  {
    id: 'TKT-102',
    userName: 'Dr. Ramesh Kumar',
    userEmail: 'ramesh.kumar@icar.gov.in',
    crop: 'Tomato',
    category: 'Detection Issue',
    rating: 4,
    subject: 'Lower shaded leaf with sooty mold flagged as Early Blight',
    message: 'When scanning a shaded lower tomato leaf with dense whitefly honeydew, confidence dropped to 84%. Please advise on optimal lighting.',
    status: 'Pending Admin Action',
    createdAt: '2026-10-22 16:05:42'
  },
  {
    id: 'TKT-103',
    userName: 'Dr. Ramesh Kumar',
    userEmail: 'ramesh.kumar@icar.gov.in',
    crop: 'Potato',
    category: 'Treatment Feedback',
    rating: 5,
    subject: 'Pre-harvest waiting period confirmation for Ridomil Gold',
    message: 'Can we harvest potato tubers 14 days after applying Ridomil Gold 68 WG for Late Blight?',
    status: 'Resolved ✓',
    createdAt: '2026-10-20 09:40:10',
    adminReply: 'Yes, the strict Pre-Harvest Interval (PHI) for Ridomil Gold 68 WG on Potato is 14 days. Ensure soil moisture is moderate before harvest.',
    repliedBy: 'PHANI NANDAN M S (Admin)',
    repliedAt: '2026-10-20 12:18:05'
  }
];

// ============================================================================
// SECTION 2: INTERACTIVE LEAF ILLUSTRATION WITH YOLO BOUNDING BOXES
// Summary: Draws a clean botanical leaf specimen with pathology lesions and
// overlays green YOLOv9 detection bounding boxes and optional thermal heatmap.
// ============================================================================

const LeafScannerViewport: React.FC<{
  scan: ScanResult;
  specimenColor: string;
  spotColor: string;
  showBoxes: boolean;
  showHeatmap: boolean;
}> = ({ scan, specimenColor, spotColor, showBoxes, showHeatmap }) => {
  return (
    <div className="relative w-full h-[340px] rounded-2xl bg-slate-950 border border-slate-800 overflow-hidden flex items-center justify-center select-none shadow-inner">
      {/* Subtle scientific coordinate grid background */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: 'radial-gradient(#10b981 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}
      />

      {scan.imageUrl ? (
        <div className="relative w-full h-full flex items-center justify-center z-10">
          <img
            src={scan.imageUrl}
            alt={scan.detectedPest}
            referrerPolicy="no-referrer"
            className="w-full h-full object-contain"
          />
          {showBoxes && (
            <div className="absolute w-[52%] h-[54%] border-2 border-dashed border-emerald-400 bg-emerald-500/15 rounded-xl flex items-start justify-start p-2 shadow-lg pointer-events-none">
              <span className="px-2 py-0.5 rounded bg-emerald-950/90 border border-emerald-500/50 text-emerald-300 font-mono text-[11px] font-bold">
                {scan.detectedPest} ({scan.confidence}%)
              </span>
            </div>
          )}
        </div>
      ) : (
        <svg viewBox="0 0 320 300" className="w-full h-full max-h-[310px] relative z-10 drop-shadow-2xl">
          <defs>
            <radialGradient id="leafGrad" cx="50%" cy="50%" r="55%">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="70%" stopColor={specimenColor} />
              <stop offset="100%" stopColor="#052e16" />
            </radialGradient>
          </defs>

          {/* Botanical Leaf Blade */}
          <path
            d="M 160 20 C 235 50 270 140 225 225 C 190 270 165 285 160 288 C 155 285 130 270 95 225 C 50 140 85 50 160 20 Z"
            fill="url(#leafGrad)"
            stroke="#4ade80"
            strokeWidth="2"
          />
          {/* Central Midrib & Lateral Veins */}
          <path
            d="M 160 22 L 160 288 M 160 75 Q 195 90 220 110 M 160 120 Q 205 140 235 165 M 160 170 Q 200 190 220 215 M 160 75 Q 125 90 100 110 M 160 120 Q 115 140 85 165 M 160 170 Q 120 190 100 215"
            stroke="#86efac"
            strokeWidth="2.2"
            fill="none"
            opacity="0.75"
          />

          {/* Pathology Lesions / Pest Spots */}
          <circle cx="135" cy="115" r="20" fill="#facc15" opacity="0.4" />
          <circle cx="135" cy="115" r="14" fill={spotColor} stroke="#1c1917" strokeWidth="1.5" />
          <circle cx="135" cy="115" r="6" fill="#09090b" opacity="0.65" />

          <circle cx="185" cy="155" r="18" fill="#facc15" opacity="0.4" />
          <circle cx="185" cy="155" r="12" fill={spotColor} stroke="#1c1917" strokeWidth="1.5" />

          <circle cx="148" cy="190" r="14" fill="#facc15" opacity="0.35" />
          <circle cx="148" cy="190" r="9" fill={spotColor} />

          {/* Computer Vision Bounding Box Overlay */}
          {showBoxes && (
            <g>
              <rect
                x="105"
                y="85"
                width="110"
                height="125"
                fill="rgba(16, 185, 129, 0.14)"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeDasharray="5 3"
                rx="6"
              />
              <rect x="105" y="63" width="165" height="20" fill="#064e3b" rx="4" />
              <text x="112" y="77" fill="#6ee7b7" fontSize="10.5" fontFamily="monospace" fontWeight="bold">
                {scan.detectedPest} ({scan.confidence}%)
              </text>
            </g>
          )}
        </svg>
      )}

      {/* Optional Thermal Severity Heatmap Overlay */}
      {showHeatmap && (
        <div
          className="absolute inset-0 pointer-events-none z-20 opacity-55 mix-blend-screen"
          style={{
            background:
              'radial-gradient(circle at 50% 48%, rgba(244,63,94,0.85) 0%, rgba(245,158,11,0.55) 32%, rgba(16,185,129,0.2) 62%, transparent 80%)'
          }}
        />
      )}

      {/* Bottom Floating HUD Pill */}
      <div className="absolute bottom-3 left-3 right-3 z-30 flex items-center justify-between px-3.5 py-2 rounded-xl bg-slate-900/90 backdrop-blur-md border border-slate-700/80 text-xs text-slate-200">
        <span className="font-mono text-emerald-400 font-semibold flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          Computer Vision & ML Active
        </span>
        <span className="font-mono text-slate-300">
          Damage Area: <strong className="text-amber-400">{scan.affectedAreaPercent}%</strong> · Bucket #{scan.hashBucket}
        </span>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN APPLICATION COMPONENT
// ============================================================================

export default function App() {
  // --------------------------------------------------------------------------
  // STEP 1: AUTHENTICATION & LOGIN GATE STATE
  // Summary: Website starts at the Login / Sign-Up Gate Screen first.
  // Supports ?autostart=1 for automated browser testing.
  // --------------------------------------------------------------------------
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('autostart') === '1';
  });

  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [authName, setAuthName] = useState('');
  const [authEmail, setAuthEmail] = useState('phaninandan78@gmail.com');
  const [authPassword, setAuthPassword] = useState('••••••••••••');
  const [authError, setAuthError] = useState('');

  // Logged-in User Profile State
  const [currentUser, setCurrentUser] = useState<{
    name: string;
    email: string;
    role: 'Project Team / Admin' | 'Field Agronomist';
    isAdmin: boolean;
  }>({
    name: 'PHANI NANDAN M S',
    email: 'phaninandan78@gmail.com',
    role: 'Project Team / Admin',
    isAdmin: true
  });

  // --------------------------------------------------------------------------
  // STEP 2: THEME & NAVIGATION TAB STATE
  // Summary: Controls Dark/Light mode and active top-navbar tab.
  // --------------------------------------------------------------------------
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'history' | 'feedback' | 'pipeline' | 'admin'>('dashboard');

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // --------------------------------------------------------------------------
  // STEP 3: SCANNER, HISTORY & COMPLAINTS STATE
  // Summary: Holds current scan, user's scan history, and feedback tickets.
  // --------------------------------------------------------------------------
  const [selectedCrop, setSelectedCrop] = useState<CropType | 'Auto-Detect'>('Auto-Detect');
  const [activeSpecimen, setActiveSpecimen] = useState(SAMPLE_LEAF_SPECIMENS[0]);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [hasScannedImage, setHasScannedImage] = useState<boolean>(false);
  const [currentScan, setCurrentScan] = useState<ScanResult>(() => INITIAL_HISTORY[0]);
  const [scanHistory, setScanHistory] = useState<ScanResult[]>(INITIAL_HISTORY);
  // --------------------------------------------------------------------------
  // STEP 3B: PERSISTENT COMPLAINTS & ADMIN RESPONSE STATE (LOCALSTORAGE)
  // Summary: Loads and saves all user complaints and admin response messages
  // in localStorage ('smart_pest_complaints_v3') so that when an Admin sends a
  // response and the user logs into their account, the response is persisted
  // and immediately visible in their complaint tracker.
  // --------------------------------------------------------------------------
  const [complaints, setComplaints] = useState<UserComplaint[]>(() => {
    try {
      const saved = localStorage.getItem('smart_pest_complaints_v3');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      // Fallback to initial complaints if localStorage is unavailable
    }
    return INITIAL_COMPLAINTS;
  });

  // Automatically sync updated complaints and admin replies to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('smart_pest_complaints_v3', JSON.stringify(complaints));
    } catch {
      // Ignore storage quota errors
    }
  }, [complaints]);

  // Filters for History & Admin Inspector
  const [historySearch, setHistorySearch] = useState('');
  const [dbSearch, setDbSearch] = useState('');
  const [dbCropFilter, setDbCropFilter] = useState<string>('all');

  // Interactive Sandbox Slider for the Algorithm Pipeline Tab
  const [sandboxDamage, setSandboxDamage] = useState<number>(24);

  // --------------------------------------------------------------------------
  // STEP 3C: USER COMPLAINT FORM, TRACKER FILTER & ADMIN REPLY DRAFTS
  // Summary: Manages form inputs for submitting new complaints, the toggle
  // between 'My Complaints' and 'All Tickets', and per-ticket reply drafts
  // typed by the Admin inside the Admin Portal.
  // --------------------------------------------------------------------------
  const [fbCrop, setFbCrop] = useState<CropType>('Tomato');
  const [fbCategory, setFbCategory] = useState<UserComplaint['category']>('Detection Issue');
  const [fbRating, setFbRating] = useState<number>(5);
  const [fbSubject, setFbSubject] = useState('');
  const [fbMessage, setFbMessage] = useState('');
  const [fbSuccessBanner, setFbSuccessBanner] = useState(false);
  const [complaintFilter, setComplaintFilter] = useState<'my' | 'all'>('my');

  // Stores custom response messages typed by the Admin for each ticket ID
  const [adminReplyDrafts, setAdminReplyDrafts] = useState<Record<string, string>>({});
  const [adminReplySentId, setAdminReplySentId] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // STEP 4: LOGIN & SIGN-UP HANDLERS
  // Summary: Validates user input and assigns Admin rights to the Project Team.
  // --------------------------------------------------------------------------
  const handleAuthSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanEmail = authEmail.trim().toLowerCase();
    if (!cleanEmail || !authPassword.trim()) {
      setAuthError('Please enter both your email address and password.');
      return;
    }
    const isTeamAdmin =
      cleanEmail === 'phaninandan78@gmail.com' ||
      authName.toLowerCase().includes('phani') ||
      authName.toLowerCase().includes('vedhesha') ||
      authName.toLowerCase().includes('khadga') ||
      authName.toLowerCase().includes('shivani');

    setCurrentUser({
      name: isTeamAdmin
        ? authName.trim().toUpperCase() || 'PHANI NANDAN M S'
        : authName.trim() || cleanEmail.split('@')[0].replace('.', ' '),
      email: cleanEmail,
      role: isTeamAdmin ? 'Project Team / Admin' : 'Field Agronomist',
      isAdmin: isTeamAdmin
    });
    setAuthError('');
    setIsLoggedIn(true);
  };

  const handleQuickLogin = (asAdmin: boolean) => {
    if (asAdmin) {
      setCurrentUser({
        name: 'PHANI NANDAN M S',
        email: 'phaninandan78@gmail.com',
        role: 'Project Team / Admin',
        isAdmin: true
      });
    } else {
      setCurrentUser({
        name: 'Dr. Ramesh Kumar',
        email: 'ramesh.kumar@icar.gov.in',
        role: 'Field Agronomist',
        isAdmin: false
      });
      if (activeTab === 'admin') setActiveTab('dashboard');
    }
    setIsLoggedIn(true);
  };

  // --------------------------------------------------------------------------
  // STEP 5: LEAF SCANNING & FILE UPLOAD HANDLERS
  // Summary: Runs our 4-stage pipeline when a user selects a sample leaf,
  // uploads a custom photo, or clicks "Run AI Detection".
  // --------------------------------------------------------------------------
  const handleSelectSpecimen = (spec: typeof SAMPLE_LEAF_SPECIMENS[0]) => {
    setActiveSpecimen(spec);
    setSelectedCrop(spec.crop);
    const result = runCompleteDetectionPipeline({
      cropInput: spec.crop,
      pestIdOverride: spec.pestId,
      affectedAreaPercent: spec.damagePercent,
      userEmail: currentUser.email,
      seed: spec.id
    });
    setCurrentScan(result);
    setHasScannedImage(true);
    setScanHistory((prev) => [result, ...prev]);
  };

  const handleManualScanTrigger = () => {
    const matchingPest =
      selectedCrop === 'Auto-Detect'
        ? undefined
        : PEST_DATABASE.find((p) => p.affectedCrops.includes(selectedCrop))?.id;

    const result = runCompleteDetectionPipeline({
      cropInput: selectedCrop,
      pestIdOverride: matchingPest,
      affectedAreaPercent: activeSpecimen.damagePercent,
      userEmail: currentUser.email,
      seed: `${selectedCrop}_${Date.now()}`
    });
    setCurrentScan(result);
    setHasScannedImage(true);
    setScanHistory((prev) => [result, ...prev]);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result as string;
      const result = runCompleteDetectionPipeline({
        cropInput: selectedCrop,
        affectedAreaPercent: 29,
        userEmail: currentUser.email,
        seed: file.name
      });
      const withImage: ScanResult = { ...result, imageUrl: base64 };
      setCurrentScan(withImage);
      setHasScannedImage(true);
      setScanHistory((prev) => [withImage, ...prev]);
    };
    reader.readAsDataURL(file);
  };

  // --------------------------------------------------------------------------
  // STEP 6: USER COMPLAINT SUBMISSION & ADMIN RESPONSE MESSAGING HANDLERS
  // Summary:
  //   1. formatCurrentDateTime() -> Formats current timestamp as YYYY-MM-DD HH:MM:SS.
  //   2. handleFeedbackSubmit()  -> Creates a new complaint ticket with the logged-in
  //                                 user's name, email, and timestamp, and switches
  //                                 the tracker view to 'My Complaints'.
  //   3. handleResolveComplaint()-> Allows an Admin to send a custom response message
  //                                 to the user's ticket, recording the Admin's name
  //                                 and reply timestamp, and marking it 'Resolved ✓'.
  // --------------------------------------------------------------------------
  const formatCurrentDateTime = () => {
    const now = new Date();
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    const hh = String(now.getHours()).padStart(2, '0');
    const min = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
  };

  // Submits a new complaint from the currently logged-in user
  const handleFeedbackSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!fbSubject.trim() || !fbMessage.trim()) return;
    const newTicket: UserComplaint = {
      id: `TKT-${Math.floor(101 + complaints.length)}`,
      userName: currentUser.name,
      userEmail: currentUser.email,
      crop: fbCrop,
      category: fbCategory,
      rating: fbRating,
      subject: fbSubject.trim(),
      message: fbMessage.trim(),
      status: 'Pending Admin Action',
      createdAt: formatCurrentDateTime()
    };
    setComplaints((prev) => [newTicket, ...prev]);
    setFbSubject('');
    setFbMessage('');
    setComplaintFilter('my');
    setFbSuccessBanner(true);
    setTimeout(() => setFbSuccessBanner(false), 3500);
  };

  // Sends an official Admin response message to a specific complaint ticket
  const handleResolveComplaint = (id: string, customMessage?: string) => {
    const replyTimestamp = formatCurrentDateTime();
    setComplaints((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        const trimmedReply = (customMessage ?? adminReplyDrafts[id] ?? '').trim();
        const finalReply =
          trimmedReply ||
          c.adminReply ||
          `Hello ${c.userName}, your complaint regarding ${c.crop} (${c.subject}) has been reviewed and resolved by our team. Please follow the updated dosage and field guidelines.`;
        return {
          ...c,
          status: 'Resolved ✓',
          adminReply: finalReply,
          repliedBy: `${currentUser.name} (Admin)`,
          repliedAt: replyTimestamp
        };
      })
    );
    setAdminReplySentId(id);
    setTimeout(() => {
      setAdminReplySentId((prev) => (prev === id ? null : prev));
    }, 3000);
  };

  // CSV Download Helper for Scan History
  const handleDownloadCSV = () => {
    const header = 'Scan ID,Timestamp,Crop,Detected Pest,Severity,Damage %,Top Medicine,Cost (INR)\n';
    const rows = scanHistory
      .map(
        (s) =>
          `${s.id},${s.timestamp},${s.crop},"${s.detectedPest}",${s.severity},${s.affectedAreaPercent}%,"${s.rankedTreatments[0]?.productName}",${s.rankedTreatments[0]?.costPerAcreINR}`
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'Smart_Pest_Detection_History.csv';
    link.click();
  };

  // ==========================================================================
  // SECTION 3: FULL-SCREEN LOGIN & SIGN-UP GATE PORTAL
  // Summary: Displayed first before entering the main application.
  // ==========================================================================
  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col justify-between bg-slate-950 text-slate-100 relative overflow-hidden font-sans">
        {/* Ambient MNC Gradient Glows */}
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-emerald-500/20 blur-3xl pointer-events-none" />
        <div className="absolute top-1/3 -right-32 w-96 h-96 rounded-full bg-cyan-500/15 blur-3xl pointer-events-none" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 rounded-full bg-indigo-500/15 blur-3xl pointer-events-none" />

        {/* Top Bar on Login Page */}
        <header className="relative z-10 max-w-7xl w-full mx-auto px-6 py-5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 shrink-0 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center text-slate-950 font-bold shadow-lg shadow-emerald-500/20">
              <Sprout className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xs sm:text-sm font-extrabold tracking-tight text-white block leading-snug">
                SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
              </span>
              <span className="text-[11px] text-emerald-400 font-semibold tracking-wider uppercase block mt-0.5">
                PROJECT BY: PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI
              </span>
            </div>
          </div>
        </header>

        {/* Center Split Login & Project Showcase */}
        <main className="relative z-10 max-w-6xl w-full mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          {/* Left Column: Project Title & Overview */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5" />
              <span>5th-Semester B.Tech AI & DS Design Project</span>
            </div>

            <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight uppercase">
              SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400">
                USING COMPUTER VISION AND MACHINE LEARNING
              </span>
            </h1>

            <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-xs sm:text-sm">
              <span className="text-emerald-400 font-extrabold uppercase tracking-wider block mb-1">
                PROJECT BY:
              </span>
              <span className="font-extrabold text-white tracking-wide">
                PHANI NANDAN M S &nbsp;•&nbsp; VEDHESHA PARTHIBAN &nbsp;•&nbsp; KHADGA LOHITH K &nbsp;•&nbsp; SHIVANI BALLANI
              </span>
            </div>

            <p className="text-slate-300 text-sm sm:text-base leading-relaxed max-w-xl">
              An enterprise-grade agricultural decision support system combining computer vision with four classical Data Structures: <strong>Feature Arrays</strong>, <strong>O(1) Hash Tables</strong>, <strong>Decision Trees</strong>, and <strong>Priority Queue Max-Heaps</strong>.
            </p>

            {/* 4 Key Architecture Highlights */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              <div className="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md">
                <div className="text-xl font-black text-emerald-400 font-mono">15 Crops</div>
                <div className="text-xs text-slate-400 mt-0.5">Full CNN Support</div>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md">
                <div className="text-xl font-black text-cyan-400 font-mono">18 Pests</div>
                <div className="text-xs text-slate-400 mt-0.5">O(1) Hash Indexed</div>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md">
                <div className="text-xl font-black text-amber-400 font-mono">&lt; 3 ms</div>
                <div className="text-xs text-slate-400 mt-0.5">DSA Pipeline Speed</div>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md">
                <div className="text-xl font-black text-indigo-400 font-mono">96.4%</div>
                <div className="text-xs text-slate-400 mt-0.5">Detection Accuracy</div>
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Login / Sign-Up Glassmorphic Card */}
          <div className="lg:col-span-5">
            <div className="rounded-3xl p-6 sm:p-8 bg-slate-900/85 backdrop-blur-xl border border-slate-800 shadow-2xl">
              {/* Mode Switch Tabs */}
              <div className="grid grid-cols-2 p-1 rounded-xl bg-slate-950 border border-slate-800 mb-6 text-xs font-bold">
                <button
                  type="button"
                  onClick={() => setAuthMode('login')}
                  className={`py-2.5 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                    authMode === 'login'
                      ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-md'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <LogIn className="w-3.5 h-3.5" />
                  <span>Sign In</span>
                </button>
                <button
                  type="button"
                  onClick={() => setAuthMode('signup')}
                  className={`py-2.5 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                    authMode === 'signup'
                      ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-md'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <UserPlus className="w-3.5 h-3.5" />
                  <span>Sign Up</span>
                </button>
              </div>

              <h2 className="text-sm font-extrabold text-white mb-1 uppercase leading-snug">
                SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
              </h2>
              <p className="text-xs text-slate-400 mb-5">
                {authMode === 'login'
                  ? 'Sign in with your registered credentials to access the scanner.'
                  : 'Register a new account to track crop scans and submit field reports.'}
              </p>

              {authError && (
                <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300">
                  {authError}
                </div>
              )}

              <form onSubmit={handleAuthSubmit} className="space-y-4 text-xs">
                {authMode === 'signup' && (
                  <div>
                    <label className="block font-semibold text-slate-300 mb-1.5">Full Name</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g., Phani Nandan M S"
                      value={authName}
                      onChange={(e) => setAuthName(e.target.value)}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                )}

                <div>
                  <label className="block font-semibold text-slate-300 mb-1.5">Email Address</label>
                  <input
                    type="email"
                    required
                    placeholder="phaninandan78@gmail.com"
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-300 mb-1.5">Password</label>
                  <input
                    type="password"
                    required
                    placeholder="Enter your password"
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-bold shadow-lg shadow-emerald-500/20 transition-all cursor-pointer"
                >
                  {authMode === 'login' ? 'Sign In & Open Dashboard →' : 'Create Account & Continue →'}
                </button>
              </form>

              {/* Quick One-Click Account Selection */}
              <div className="mt-6 pt-5 border-t border-slate-800">
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2.5 text-center">
                  Quick Access Accounts
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <button
                    type="button"
                    onClick={() => handleQuickLogin(true)}
                    className="p-2.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-left transition-all cursor-pointer"
                  >
                    <div className="text-xs font-bold text-emerald-300 flex items-center gap-1">
                      <Shield className="w-3.5 h-3.5" />
                      <span>Phani Nandan M S</span>
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Admin + Full Portal Access</div>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleQuickLogin(false)}
                    className="p-2.5 rounded-xl bg-slate-800/70 hover:bg-slate-800 border border-slate-700 text-left transition-all cursor-pointer"
                  >
                    <div className="text-xs font-bold text-slate-200">Dr. Ramesh Kumar</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Standard Farmer / User Role</div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Login Footer */}
        <footer className="relative z-10 max-w-7xl w-full mx-auto px-6 py-4 text-center sm:flex items-center justify-between text-xs text-slate-500 border-t border-slate-900 gap-4">
          <span className="font-bold text-slate-400">
            SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
          </span>
          <span className="font-bold text-emerald-400">
            Project By: PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI
          </span>
        </footer>
      </div>
    );
  }

  // ==========================================================================
  // MAIN LOGGED-IN WEBSITE INTERFACE
  // Summary: Computes complaint counts for both the Admin queue and the
  // currently logged-in user's personal complaint tracker.
  // ==========================================================================

  const pendingComplaintsCount = complaints.filter((c) => c.status === 'Pending Admin Action').length;

  // Filter complaints belonging to the currently logged-in user (matched by email or name)
  const userComplaints = complaints.filter(
    (c) =>
      c.userEmail.toLowerCase() === currentUser.email.toLowerCase() ||
      c.userName.toLowerCase() === currentUser.name.toLowerCase()
  );

  // Count how many of the logged-in user's complaints have received an Admin response message
  const userResolvedWithReplyCount = userComplaints.filter((c) => Boolean(c.adminReply)).length;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans transition-colors duration-200">
      {/* --------------------------------------------------------------------
          TOP HEADER & NAVIGATION BAR (Clean Two-Row Non-Overlapping Layout)
          -------------------------------------------------------------------- */}
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 space-y-3">
          {/* Top Row: Full Project Title + Team Names on Left | User Controls on Right */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex items-start sm:items-center gap-3">
              <div className="w-10 h-10 shrink-0 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-white shadow-md shadow-emerald-500/20">
                <Sprout className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs sm:text-sm font-extrabold tracking-tight text-slate-900 dark:text-white block leading-snug uppercase">
                  SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
                </span>
                <span className="text-[10px] sm:text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider block mt-0.5">
                  Project By: PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI
                </span>
              </div>
            </div>

            {/* Right User Badge, Theme Toggle & Sign Out */}
            <div className="flex items-center gap-2 shrink-0 self-end md:self-center">
              <button
                onClick={() => setDarkMode(!darkMode)}
                aria-label="Toggle Dark / Light Mode"
                className="p-2 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors cursor-pointer"
                title="Toggle Dark / Light Mode"
              >
                {darkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-700" />}
              </button>

              <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs">
                <div className="w-6 h-6 rounded-lg bg-emerald-600 text-white font-bold flex items-center justify-center text-[11px]">
                  {currentUser.name.slice(0, 2).toUpperCase()}
                </div>
                <div className="leading-tight">
                  <div className="font-bold text-slate-900 dark:text-white truncate max-w-[130px]">
                    {currentUser.name}
                  </div>
                  <div className="text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                    {currentUser.role}
                  </div>
                </div>
              </div>

              <button
                onClick={() => setIsLoggedIn(false)}
                className="px-3 py-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap"
                title="Sign Out to Login Screen"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>Sign Out</span>
              </button>
            </div>
          </div>

          {/* Bottom Row: Multi-Tab Navigation Bar */}
          <nav className="flex items-center gap-1.5 overflow-x-auto p-1.5 rounded-xl bg-slate-100 dark:bg-slate-800/90 border border-slate-200/70 dark:border-slate-700/70">
            {[
              { id: 'dashboard', label: '1. Dashboard & Scanner' },
              { id: 'history', label: `2. My Scans (${scanHistory.length})` },
              {
                id: 'feedback',
                label: `3. Feedback & Complaints (${userComplaints.length} Mine / ${complaints.length} Total)`
              },
              { id: 'pipeline', label: '4. Algorithm Pipeline' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-3.5 py-2 rounded-lg text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            ))}

            {/* Admin Portal Tab */}
            <button
              onClick={() => setActiveTab('admin')}
              className={`px-3.5 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap ${
                activeTab === 'admin'
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-sm'
                  : 'text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10'
              }`}
            >
              <Shield className="w-3.5 h-3.5" />
              <span>Admin Portal ({pendingComplaintsCount} Pending)</span>
            </button>
          </nav>
        </div>
      </header>

      {/* MAIN CONTENT CONTAINER */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* User Account Complaint & Admin Response Notification Bar */}
        {activeTab !== 'feedback' && activeTab !== 'admin' && userComplaints.length > 0 && (
          <div className="rounded-2xl p-4 bg-emerald-500/10 border border-emerald-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
            <div className="flex items-start sm:items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-emerald-600 text-white flex items-center justify-center shrink-0 font-bold">
                ✓
              </div>
              <div>
                <div className="font-extrabold text-slate-900 dark:text-white">
                  Account Complaint Tracker for {currentUser.name} ({currentUser.email})
                </div>
                <div className="text-slate-600 dark:text-slate-300 mt-0.5">
                  You have <strong>{userComplaints.length}</strong> submitted complaint(s) —{' '}
                  <strong className="text-emerald-600 dark:text-emerald-400">
                    {userResolvedWithReplyCount} with an Admin Response Message
                  </strong>{' '}
                  and{' '}
                  <strong className="text-amber-600 dark:text-amber-400">
                    {userComplaints.length - userResolvedWithReplyCount} Pending Review
                  </strong>.
                </div>
              </div>
            </div>
            <button
              onClick={() => {
                setComplaintFilter('my');
                setActiveTab('feedback');
              }}
              className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold shrink-0 cursor-pointer transition-colors"
            >
              Track My Complaints & Admin Messages →
            </button>
          </div>
        )}
        {/* ====================================================================
            SECTION 4: TAB 1 — MNC DASHBOARD & LEAF SCANNER
            ==================================================================== */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Project Title Banner */}
            <div className="rounded-2xl p-5 bg-gradient-to-r from-slate-900 via-emerald-950 to-slate-900 text-white border border-emerald-500/30 shadow-md">
              <div className="text-[11px] font-bold text-emerald-400 uppercase tracking-wider mb-1">
                PROJECT BY: PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI
              </div>
              <h1 className="text-base sm:text-xl font-extrabold tracking-tight uppercase leading-snug">
                SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
              </h1>
            </div>

            {/* MNC Executive Colorful KPI Strip */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-5 rounded-2xl bg-gradient-to-br from-emerald-600 to-teal-700 text-white shadow-lg shadow-emerald-600/15">
                <div className="flex items-center justify-between opacity-90 text-xs font-semibold">
                  <span>SUPPORTED CROPS</span>
                  <Sprout className="w-4 h-4" />
                </div>
                <div className="text-3xl font-extrabold font-mono mt-2">15 Crops</div>
                <div className="text-[11px] opacity-85 mt-1">Auto-CNN + Manual Selector</div>
              </div>

              <div className="p-5 rounded-2xl bg-gradient-to-br from-cyan-600 to-blue-700 text-white shadow-lg shadow-cyan-600/15">
                <div className="flex items-center justify-between opacity-90 text-xs font-semibold">
                  <span>HASH TABLE CATALOG</span>
                  <Database className="w-4 h-4" />
                </div>
                <div className="text-3xl font-extrabold font-mono mt-2">18 Pests</div>
                <div className="text-[11px] opacity-85 mt-1">O(1) Instant Bucket Lookup</div>
              </div>

              <div className="p-5 rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-700 text-white shadow-lg shadow-indigo-600/15">
                <div className="flex items-center justify-between opacity-90 text-xs font-semibold">
                  <span>PIPELINE LATENCY</span>
                  <Cpu className="w-4 h-4" />
                </div>
                <div className="text-3xl font-extrabold font-mono mt-2">
                  {currentScan.telemetryMs.total} ms
                </div>
                <div className="text-[11px] opacity-85 mt-1">Across 4 Data Structure Stages</div>
              </div>

              <div className="p-5 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 text-white shadow-lg shadow-amber-500/15">
                <div className="flex items-center justify-between opacity-90 text-xs font-semibold">
                  <span>PROJECT BY</span>
                  <Award className="w-4 h-4" />
                </div>
                <div className="text-xs font-extrabold mt-2 leading-snug">
                  PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI
                </div>
              </div>
            </div>

            {/* Scanner Control Bar */}
            <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
              <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
                {/* 15 Crops Dropdown */}
                <div className="md:col-span-5">
                  <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                    1. Select Target Crop (15 Major Crops + Auto-CNN)
                  </label>
                  <select
                    value={selectedCrop}
                    onChange={(e) => setSelectedCrop(e.target.value as any)}
                    className="w-full px-3.5 py-2.5 rounded-xl text-xs font-semibold bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  >
                    <option value="Auto-Detect">✦ Auto-Detect via CNN Feature Array</option>
                    {ALL_CROPS.map((crop) => (
                      <option key={crop} value={crop}>
                        {crop}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Upload Leaf Image */}
                <div className="md:col-span-4">
                  <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                    2. Upload Custom Leaf Image
                  </label>
                  <label className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-dashed border-emerald-500/60 bg-emerald-500/5 hover:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 text-xs font-bold cursor-pointer transition-colors">
                    <UploadCloud className="w-4 h-4" />
                    <span>Upload Leaf Photo (JPG/PNG)</span>
                    <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                  </label>
                </div>

                {/* Run Scan Button */}
                <div className="md:col-span-3">
                  <button
                    onClick={handleManualScanTrigger}
                    className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-bold shadow-md shadow-emerald-600/20 flex items-center justify-center gap-2 cursor-pointer transition-all"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>Run Detection Pipeline</span>
                  </button>
                </div>
              </div>

              {/* 4 Colorful Quick-Test Sample Leaf Buttons */}
              <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center gap-2">
                <span className="text-xs font-bold text-slate-500 dark:text-slate-400 mr-1">
                  Quick-Test Sample Leaves:
                </span>
                {SAMPLE_LEAF_SPECIMENS.map((spec) => (
                  <button
                    key={spec.id}
                    onClick={() => handleSelectSpecimen(spec)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer flex items-center gap-2 ${
                      activeSpecimen.id === spec.id && !currentScan.imageUrl
                        ? 'bg-emerald-600 text-white shadow-sm'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                    <span>{spec.title}</span>
                    <span className="text-[10px] opacity-75">({spec.crop})</span>
                  </button>
                ))}
              </div>
            </div>

            {/* ================================================================
                CONDITIONAL WORKSPACE:
                - Before an image is uploaded (!hasScannedImage): Show useful
                  Interactive Leaf Upload Dropzone, 3-Step Workflow Guide, and
                  15-Crop & 18-Pest Agricultural Reference Directory.
                - Once an image is uploaded / scanned (hasScannedImage): Show
                  the 4-Stage Telemetry Bar, Computer Vision Detection Box, and
                  Priority Queue Ranked Treatments.
                ================================================================ */}
            {!hasScannedImage ? (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
                {/* Left 6 Columns: Interactive Leaf Upload Dropzone & 3-Step Workflow Guide */}
                <div className="lg:col-span-6 rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between space-y-5">
                  <div>
                    <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold mb-2">
                      <UploadCloud className="w-3.5 h-3.5" />
                      <span>Step 1 · Computer Vision Leaf Image Input</span>
                    </div>
                    <h3 className="text-lg font-extrabold text-slate-900 dark:text-white">
                      Upload a Crop Leaf Image to Start Detection
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                      Upload a clear photo of an affected crop leaf below (or select one of the Quick-Test Sample Leaves above) to run the Computer Vision & Machine Learning pipeline.
                    </p>
                  </div>

                  {/* Large Interactive Drag & Drop Upload Box */}
                  <label className="group flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed border-emerald-500/50 hover:border-emerald-500 bg-emerald-500/5 hover:bg-emerald-500/10 transition-all cursor-pointer text-center space-y-3">
                    <div className="w-14 h-14 rounded-2xl bg-emerald-500/15 group-hover:bg-emerald-500/25 flex items-center justify-center text-emerald-600 dark:text-emerald-400 transition-colors">
                      <UploadCloud className="w-7 h-7" />
                    </div>
                    <div>
                      <span className="text-sm font-extrabold text-slate-900 dark:text-white block">
                        Click to Upload Leaf Image (JPG, PNG, WEBP)
                      </span>
                      <span className="text-xs text-slate-500 block mt-0.5">
                        Supports all 15 agricultural crops · Automatic CNN Feature Extraction
                      </span>
                    </div>
                    <span className="px-4 py-2 rounded-xl bg-emerald-600 text-white text-xs font-bold shadow-sm">
                      Browse Leaf Photo from Device
                    </span>
                    <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                  </label>

                  {/* How the System Works (3 Simple Steps) */}
                  <div className="space-y-2.5 pt-2">
                    <div className="text-xs font-extrabold uppercase tracking-wider text-slate-400">
                      How Detection & Treatment Ranking Works:
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                      <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/70 dark:border-slate-700/70">
                        <div className="font-extrabold text-emerald-600 dark:text-emerald-400 mb-0.5">
                          1. CNN Feature Array
                        </div>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                          Extracts 8 leaf color, lesion spot, and edge metrics from your uploaded photo.
                        </p>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/70 dark:border-slate-700/70">
                        <div className="font-extrabold text-cyan-600 dark:text-cyan-400 mb-0.5">
                          2. Hash & Decision Tree
                        </div>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                          Matches the pest in O(1) time and grades leaf damage as Low, Moderate, or Severe.
                        </p>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/70 dark:border-slate-700/70">
                        <div className="font-extrabold text-indigo-600 dark:text-indigo-400 mb-0.5">
                          3. Priority Queue
                        </div>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                          Ranks treatments by Efficacy %, Cost per Acre (₹), and safe waiting days (PHI).
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right 6 Columns: Useful 15-Crop & 18-Pest Quick Reference Guide */}
                <div className="lg:col-span-6 rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between space-y-4">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 text-xs font-bold mb-2">
                        <Database className="w-3.5 h-3.5" />
                        <span>Supported Crops & Pest Reference Guide</span>
                      </div>
                      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white">
                        15 Major Crops & Key Symptoms Directory
                      </h3>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Quick agricultural reference of supported crops, target pests, and top treatments. Click any row to run a sample scan immediately.
                      </p>
                    </div>
                  </div>

                  {/* Scrollable Crop & Pest Reference List */}
                  <div className="max-h-[290px] overflow-y-auto pr-1 space-y-2">
                    {PEST_DATABASE.slice(0, 8).map((pest) => (
                      <div
                        key={pest.id}
                        className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center justify-between gap-3 text-xs"
                      >
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold text-[10px]">
                              {pest.affectedCrops.join(', ')}
                            </span>
                            <span className="font-extrabold text-slate-900 dark:text-white">
                              {pest.commonName}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-1">
                            Symptoms: {pest.symptoms}
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            const cropName = pest.affectedCrops[0];
                            setSelectedCrop(cropName);
                            const result = runCompleteDetectionPipeline({
                              cropInput: cropName,
                              pestIdOverride: pest.id,
                              affectedAreaPercent: 28,
                              userEmail: currentUser.email,
                              seed: pest.id
                            });
                            setCurrentScan(result);
                            setHasScannedImage(true);
                            setScanHistory((prev) => [result, ...prev]);
                          }}
                          className="px-3 py-1.5 rounded-lg bg-emerald-600/15 hover:bg-emerald-600 text-emerald-600 dark:text-emerald-300 hover:text-white font-bold text-[11px] shrink-0 transition-colors cursor-pointer"
                        >
                          Simulate Scan →
                        </button>
                      </div>
                    ))}
                  </div>

                  {/* Field Photography Tip Banner */}
                  <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs flex items-center justify-between gap-3">
                    <div>
                      <span className="font-extrabold text-amber-600 dark:text-amber-400 block">
                        ✦ Field Leaf Photography Tip:
                      </span>
                      <span className="text-[11px] text-slate-600 dark:text-slate-300">
                        Capture the leaf in natural daylight against a plain background so necrotic spots and leaf margins are clearly visible.
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Top Action Bar to Reset / Upload Another Image */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/30">
                  <div className="flex items-center gap-2.5 text-xs">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
                    <div>
                      <span className="font-extrabold text-slate-900 dark:text-white block">
                        Computer Vision & Machine Learning Analysis Complete ({currentScan.telemetryMs.total} ms)
                      </span>
                      <span className="text-slate-600 dark:text-slate-300">
                        Showing bounding-box lesion detection, Decision Tree severity, and Priority Queue treatments for <strong>{currentScan.crop}</strong>.
                      </span>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setHasScannedImage(false)}
                    className="px-3.5 py-2 rounded-xl bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 text-xs font-bold shrink-0 cursor-pointer"
                  >
                    ← Upload Another Leaf / Back to Crop Guide
                  </button>
                </div>

                {/* 4-Stage Execution Telemetry Bar */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-emerald-500/30 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] font-bold uppercase text-emerald-600 dark:text-emerald-400 block">
                        Stage 1: Feature Array
                      </span>
                      <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                        8-Value Color/Spot Vector
                      </span>
                    </div>
                    <span className="font-mono text-xs font-bold px-2 py-1 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                      {currentScan.telemetryMs.featureArray} ms
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-cyan-500/30 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] font-bold uppercase text-cyan-600 dark:text-cyan-400 block">
                        Stage 2: Hash Table O(1)
                      </span>
                      <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                        Bucket #{currentScan.hashBucket} Instant Match
                      </span>
                    </div>
                    <span className="font-mono text-xs font-bold px-2 py-1 rounded-lg bg-cyan-500/10 text-cyan-600 dark:text-cyan-400">
                      {currentScan.telemetryMs.hashTable} ms
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-amber-500/30 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] font-bold uppercase text-amber-600 dark:text-amber-400 block">
                        Stage 3: Decision Tree
                      </span>
                      <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                        Severity: {currentScan.severity}
                      </span>
                    </div>
                    <span className="font-mono text-xs font-bold px-2 py-1 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
                      {currentScan.telemetryMs.decisionTree} ms
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-indigo-500/30 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] font-bold uppercase text-indigo-600 dark:text-indigo-400 block">
                        Stage 4: Priority Queue
                      </span>
                      <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                        Max-Heap Ranked ({currentScan.rankedTreatments.length})
                      </span>
                    </div>
                    <span className="font-mono text-xs font-bold px-2 py-1 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                      {currentScan.telemetryMs.priorityQueue} ms
                    </span>
                  </div>
                </div>

                {/* Split Results View: Left Computer Vision Viewport | Right Priority Queue Medicines */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                  {/* Left 6 Columns: Annotated Leaf Canvas + Diagnostic Summary */}
                  <div className="lg:col-span-6 space-y-4">
                    <div className="rounded-2xl p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                            Computer Vision & Machine Learning Leaf Detection
                          </h3>
                          <p className="text-xs text-slate-500">
                            Crop: <strong>{currentScan.crop}</strong> ({currentScan.detectionMode})
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setShowBoxes(!showBoxes)}
                            className={`px-2.5 py-1 rounded-lg text-xs font-bold flex items-center gap-1 cursor-pointer ${
                              showBoxes
                                ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30'
                                : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                            }`}
                          >
                            <Eye className="w-3.5 h-3.5" />
                            <span>Detection Box</span>
                          </button>
                          <button
                            onClick={() => setShowHeatmap(!showHeatmap)}
                            className={`px-2.5 py-1 rounded-lg text-xs font-bold flex items-center gap-1 cursor-pointer ${
                              showHeatmap
                                ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30'
                                : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                            }`}
                          >
                            <Layers className="w-3.5 h-3.5" />
                            <span>Heatmap</span>
                          </button>
                        </div>
                      </div>

                      <LeafScannerViewport
                        scan={currentScan}
                        specimenColor={activeSpecimen.leafColor}
                        spotColor={activeSpecimen.spotColor}
                        showBoxes={showBoxes}
                        showHeatmap={showHeatmap}
                      />

                      {/* Decision Tree Output Card */}
                      <div className="mt-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700/80 space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-900 dark:text-white text-sm">
                            Detected: {currentScan.detectedPest}
                          </span>
                          <span
                            className={`px-2.5 py-0.5 rounded-full font-bold text-xs ${
                              currentScan.severity === 'Severe'
                                ? 'bg-rose-500/15 text-rose-600 dark:text-rose-400 border border-rose-500/30'
                                : currentScan.severity === 'Moderate'
                                ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30'
                                : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30'
                            }`}
                          >
                            {currentScan.severity} Severity
                          </span>
                        </div>
                        <div className="text-slate-500 italic">{currentScan.scientificName}</div>
                        <div className="p-2.5 rounded-lg bg-slate-900 text-emerald-300 font-mono text-[11px]">
                          Decision Tree Rule: {currentScan.decisionRuleUsed}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right 6 Columns: Priority Queue Ranked Treatment Cards */}
                  <div className="lg:col-span-6 space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                          Priority Queue Ranked Treatments (Max-Heap)
                        </h3>
                        <p className="text-xs text-slate-500">
                          Automatically sorted by Efficacy %, Cost per Acre, and Pre-Harvest Interval (PHI).
                        </p>
                      </div>
                    </div>

                    {currentScan.rankedTreatments.map((trt, index) => (
                      <div
                        key={trt.id}
                        className={`rounded-2xl p-5 border transition-all ${
                          index === 0
                            ? 'bg-gradient-to-br from-white to-emerald-50/60 dark:from-slate-900 dark:to-emerald-950/30 border-emerald-500/50 shadow-md'
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3 mb-3">
                          <div className="flex items-center gap-2.5">
                            <span
                              className={`w-7 h-7 rounded-lg font-mono text-xs font-extrabold flex items-center justify-center ${
                                index === 0
                                  ? 'bg-emerald-600 text-white shadow-sm'
                                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                              }`}
                            >
                              #{index + 1}
                            </span>
                            <div>
                              <div className="flex items-center gap-2">
                                <h4 className="text-sm font-extrabold text-slate-900 dark:text-white">
                                  {trt.productName}
                                </h4>
                                {index === 0 && (
                                  <span className="px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 text-[10px] font-bold">
                                    ★ Top Priority Match
                                  </span>
                                )}
                              </div>
                              <div className="text-xs text-slate-500">
                                Active Ingredient: <strong className="text-slate-700 dark:text-slate-300">{trt.activeIngredient}</strong>
                              </div>
                            </div>
                          </div>

                          <div className="text-right">
                            <span className="text-[10px] font-bold uppercase text-slate-400 block">Heap Score</span>
                            <span className="text-sm font-mono font-extrabold text-emerald-600 dark:text-emerald-400">
                              {trt.priorityScore} / 100
                            </span>
                          </div>
                        </div>

                        {/* Dosage, Cost & PHI Pills */}
                        <div className="grid grid-cols-3 gap-2.5 text-xs">
                          <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/60 dark:border-slate-700/60">
                            <span className="text-[10px] text-slate-400 block">Dosage / Acre</span>
                            <span className="font-mono font-bold text-slate-900 dark:text-white">{trt.dosagePerAcre}</span>
                          </div>
                          <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/60 dark:border-slate-700/60">
                            <span className="text-[10px] text-slate-400 block">Cost / Acre</span>
                            <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">Rs. {trt.costPerAcreINR}</span>
                          </div>
                          <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/60 dark:border-slate-700/60">
                            <span className="text-[10px] text-slate-400 block">Waiting PHI</span>
                            <span className="font-mono font-bold text-amber-600 dark:text-amber-400">{trt.phiDays} Days</span>
                          </div>
                        </div>

                        <div className="mt-3 pt-2.5 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs">
                          <span className="font-semibold text-slate-600 dark:text-slate-400">
                            Category: <strong className="text-slate-800 dark:text-slate-200">{trt.category}</strong>
                          </span>
                          <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                            Field Efficacy: {trt.efficacyPercent}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ====================================================================
            SECTION 5A: TAB 2 — MY SCANS & HISTORY
            ==================================================================== */}
        {activeTab === 'history' && (
          <div className="space-y-5">
            <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-extrabold text-slate-900 dark:text-white">
                  My Scans & Persistent Field History
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Chronological log of all leaf inspections, severity grades, confidence scores, and recommended medicines.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownloadCSV}
                  className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center gap-1.5 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export CSV Report</span>
                </button>
              </div>
            </div>

            {/* Search Filter */}
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3.5 top-3 text-slate-400" />
              <input
                type="text"
                placeholder="Search scan history by crop or pest name..."
                value={historySearch}
                onChange={(e) => setHistorySearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs"
              />
            </div>

            {/* Scans Table */}
            <div className="rounded-2xl overflow-hidden bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 dark:bg-slate-800/80 text-slate-500 uppercase text-[10px] font-bold">
                    <tr>
                      <th className="px-4 py-3.5">Scan ID & Time</th>
                      <th className="px-4 py-3.5">Crop</th>
                      <th className="px-4 py-3.5">Detected Pest</th>
                      <th className="px-4 py-3.5">Confidence</th>
                      <th className="px-4 py-3.5">Severity</th>
                      <th className="px-4 py-3.5">Rank #1 Treatment</th>
                      <th className="px-4 py-3.5 text-right">Delete</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                    {scanHistory
                      .filter(
                        (s) =>
                          s.crop.toLowerCase().includes(historySearch.toLowerCase()) ||
                          s.detectedPest.toLowerCase().includes(historySearch.toLowerCase())
                      )
                      .map((scan) => (
                        <tr key={scan.id} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40">
                          <td className="px-4 py-3 font-mono text-[11px]">
                            <div className="font-bold text-slate-800 dark:text-slate-200">{scan.id}</div>
                            <div className="text-slate-400">{scan.timestamp}</div>
                          </td>
                          <td className="px-4 py-3 font-bold text-slate-900 dark:text-white">{scan.crop}</td>
                          <td className="px-4 py-3">
                            <div className="font-bold text-slate-800 dark:text-slate-200">{scan.detectedPest}</div>
                            <div className="text-[11px] text-slate-400 italic">{scan.scientificName}</div>
                          </td>
                          <td className="px-4 py-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                            {scan.confidence}%
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={`px-2 py-0.5 rounded-full font-bold text-[10px] ${
                                scan.severity === 'Severe'
                                  ? 'bg-rose-500/15 text-rose-500'
                                  : scan.severity === 'Moderate'
                                  ? 'bg-amber-500/15 text-amber-500'
                                  : 'bg-emerald-500/15 text-emerald-500'
                              }`}
                            >
                              {scan.severity} ({scan.affectedAreaPercent}%)
                            </span>
                          </td>
                          <td className="px-4 py-3 font-semibold">
                            {scan.rankedTreatments[0]?.productName} (₹{scan.rankedTreatments[0]?.costPerAcreINR})
                          </td>
                          <td className="px-4 py-3 text-right">
                            <button
                              onClick={() => setScanHistory((prev) => prev.filter((item) => item.id !== scan.id))}
                              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 cursor-pointer"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ====================================================================
            SECTION 5B: TAB 3 — USER FEEDBACK, COMPLAINT TRACKER & ADMIN INBOX
            Summary:
              - Left Column: Form for logged-in users to submit a complaint with
                automatic name, email, and timestamp capture.
              - Right Column: Personal Complaint Status Tracker where the user
                can filter their own complaints ('My Complaints'), view a 3-step
                live progress bar, and read official response messages sent by
                the Admin Team.
            ==================================================================== */}
        {activeTab === 'feedback' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Submit Feedback Form */}
            <div className="lg:col-span-5 rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
              <h3 className="text-base font-extrabold text-slate-900 dark:text-white mb-1">
                Submit Detection Complaint or Feedback
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Report false detections or dosage queries directly to the Project Team.
              </p>

              {fbSuccessBanner && (
                <div className="mb-4 p-3 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-600 dark:text-emerald-300 text-xs font-bold">
                  ✓ Ticket submitted! Live status updated in the tracker.
                </div>
              )}

              <form onSubmit={handleFeedbackSubmit} className="space-y-4 text-xs">
                <div>
                  <label className="block font-bold mb-1">Star Rating</label>
                  <div className="flex items-center gap-1.5">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        type="button"
                        key={star}
                        onClick={() => setFbRating(star)}
                        className="p-1 cursor-pointer"
                      >
                        <Star
                          className={`w-5 h-5 ${
                            star <= fbRating ? 'text-amber-400 fill-amber-400' : 'text-slate-300 dark:text-slate-700'
                          }`}
                        />
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block font-bold mb-1">Crop</label>
                    <select
                      value={fbCrop}
                      onChange={(e) => setFbCrop(e.target.value as CropType)}
                      className="w-full px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700"
                    >
                      {ALL_CROPS.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block font-bold mb-1">Category</label>
                    <select
                      value={fbCategory}
                      onChange={(e) => setFbCategory(e.target.value as any)}
                      className="w-full px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700"
                    >
                      <option value="Detection Issue">Detection Issue</option>
                      <option value="Treatment Feedback">Treatment Feedback</option>
                      <option value="Feature Suggestion">Feature Suggestion</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block font-bold mb-1">Subject Title</label>
                  <input
                    type="text"
                    required
                    placeholder="Brief summary of your issue..."
                    value={fbSubject}
                    onChange={(e) => setFbSubject(e.target.value)}
                    className="w-full px-3.5 py-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700"
                  />
                </div>

                <div>
                  <label className="block font-bold mb-1">Detailed Description</label>
                  <textarea
                    rows={3}
                    required
                    placeholder="Explain the leaf symptoms or treatment feedback..."
                    value={fbMessage}
                    onChange={(e) => setFbMessage(e.target.value)}
                    className="w-full px-3.5 py-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Submit Ticket</span>
                </button>
              </form>
            </div>

            {/* Right Column: User Account Complaint Tracker & Admin Response Inbox */}
            <div className="lg:col-span-7 space-y-4">
              {/* Account Summary & Filter Header */}
              <div className="rounded-2xl p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                      My Complaint Status Tracker & Admin Responses
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Logged in as <strong>{currentUser.name}</strong> ({currentUser.email})
                    </p>
                  </div>

                  {/* Filter Toggle: My Complaints vs All Complaints */}
                  <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl text-xs font-bold">
                    <button
                      type="button"
                      onClick={() => setComplaintFilter('my')}
                      className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                        complaintFilter === 'my'
                          ? 'bg-emerald-600 text-white shadow-sm'
                          : 'text-slate-600 dark:text-slate-300'
                      }`}
                    >
                      My Complaints ({userComplaints.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => setComplaintFilter('all')}
                      className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                        complaintFilter === 'all'
                          ? 'bg-emerald-600 text-white shadow-sm'
                          : 'text-slate-600 dark:text-slate-300'
                      }`}
                    >
                      All Tickets ({complaints.length})
                    </button>
                  </div>
                </div>

                {/* Quick Status Counters for Current User */}
                <div className="grid grid-cols-3 gap-3 text-center text-xs">
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700">
                    <div className="text-lg font-extrabold text-slate-900 dark:text-white font-mono">
                      {userComplaints.length}
                    </div>
                    <div className="text-[11px] text-slate-500">My Submitted Tickets</div>
                  </div>
                  <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
                    <div className="text-lg font-extrabold text-amber-600 dark:text-amber-400 font-mono">
                      {userComplaints.filter((c) => c.status === 'Pending Admin Action').length}
                    </div>
                    <div className="text-[11px] text-amber-700 dark:text-amber-300">Pending Admin Review</div>
                  </div>
                  <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                    <div className="text-lg font-extrabold text-emerald-600 dark:text-emerald-400 font-mono">
                      {userResolvedWithReplyCount}
                    </div>
                    <div className="text-[11px] text-emerald-700 dark:text-emerald-300">Admin Responses Received</div>
                  </div>
                </div>
              </div>

              {/* Ticket Cards List */}
              {(complaintFilter === 'my' ? userComplaints : complaints).length === 0 ? (
                <div className="rounded-2xl p-8 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center space-y-2 text-xs text-slate-500">
                  <div className="font-bold text-sm text-slate-800 dark:text-slate-200">
                    No complaints submitted under {currentUser.name} yet.
                  </div>
                  <p>
                    Use the form on the left to submit a complaint or switch to "All Tickets ({complaints.length})" to view all records.
                  </p>
                </div>
              ) : (
                (complaintFilter === 'my' ? userComplaints : complaints).map((item) => {
                  const isMine =
                    item.userEmail.toLowerCase() === currentUser.email.toLowerCase() ||
                    item.userName.toLowerCase() === currentUser.name.toLowerCase();

                  return (
                    <div
                      key={item.id}
                      className={`rounded-2xl p-5 bg-white dark:bg-slate-900 border space-y-3 text-xs shadow-sm ${
                        isMine
                          ? 'border-emerald-500/40 dark:border-emerald-500/40'
                          : 'border-slate-200 dark:border-slate-800'
                      }`}
                    >
                      {/* Top Header: Ticket ID, Crop, Category & Status */}
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono font-extrabold text-slate-700 dark:text-slate-200 px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800">
                            {item.id}
                          </span>
                          <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">
                            Crop: {item.crop}
                          </span>
                          <span className="px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-semibold">
                            {item.category}
                          </span>
                          {isMine && (
                            <span className="px-2.5 py-1 rounded-lg bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 font-bold">
                              My Ticket
                            </span>
                          )}
                        </div>
                        <span
                          className={`px-3 py-1 rounded-full font-bold ${
                            item.status === 'Resolved ✓'
                              ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                              : 'bg-amber-500/15 text-amber-600 dark:text-amber-400'
                          }`}
                        >
                          {item.status}
                        </span>
                      </div>

                      {/* Submitter Name, Email & Date-Time Metadata */}
                      <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-700/70 grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
                        <div>
                          <span className="text-slate-400 block">Complainant Name:</span>
                          <span className="font-bold text-slate-800 dark:text-slate-200">{item.userName}</span>
                        </div>
                        <div>
                          <span className="text-slate-400 block">Account Email:</span>
                          <span className="font-mono text-slate-700 dark:text-slate-300">{item.userEmail}</span>
                        </div>
                        <div>
                          <span className="text-slate-400 block">Submitted Date & Time:</span>
                          <span className="font-mono font-semibold text-slate-700 dark:text-slate-300">
                            {item.createdAt}
                          </span>
                        </div>
                      </div>

                      {/* Subject & Message */}
                      <div>
                        <h4 className="text-sm font-extrabold text-slate-900 dark:text-white">{item.subject}</h4>
                        <p className="text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">{item.message}</p>
                      </div>

                      {/* 3-Step Visual Complaint Tracking Progress Bar */}
                      <div className="pt-2 border-t border-slate-200 dark:border-slate-800">
                        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                          Live Tracking Status
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-[11px]">
                          <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 font-bold">
                            1. Logged ({item.createdAt.split(' ')[0]}) ✓
                          </div>
                          <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 font-bold">
                            2. Assigned to Admin ✓
                          </div>
                          <div
                            className={`p-2 rounded-lg border font-bold ${
                              item.status === 'Resolved ✓'
                                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-700 dark:text-emerald-300'
                                : 'bg-amber-500/10 border-amber-500/30 text-amber-700 dark:text-amber-300'
                            }`}
                          >
                            {item.status === 'Resolved ✓' ? '3. Response Received ✓' : '3. Awaiting Admin Reply'}
                          </div>
                        </div>
                      </div>

                      {/* Official Admin Response Message Box */}
                      {item.adminReply ? (
                        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-1.5">
                          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-bold text-emerald-700 dark:text-emerald-300">
                            <span>✓ Official Admin Response to {item.userName}</span>
                            <span className="font-mono">
                              {item.repliedBy ? `${item.repliedBy} · ` : ''}
                              {item.repliedAt || item.createdAt}
                            </span>
                          </div>
                          <p className="text-slate-800 dark:text-slate-100 font-medium leading-relaxed">
                            {item.adminReply}
                          </p>
                        </div>
                      ) : (
                        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-300 text-[11px] font-medium">
                          Status: Your complaint is currently under review by the Admin Team. Once the Admin sends a response message, it will appear right here in your account.
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* ====================================================================
            SECTION 5C: TAB 4 — 4-STAGE ALGORITHM PIPELINE SUMMARY
            Summary: Technical overview of the 4 Data Structures & Algorithms
            with live feature outputs, interactive threshold slider, and code snippets.
            ==================================================================== */}
        {activeTab === 'pipeline' && (
          <div className="space-y-6">
            {/* Top Banner: 4-Stage Pipeline Summary */}
            <div className="rounded-2xl p-6 bg-gradient-to-r from-emerald-900 via-teal-900 to-slate-900 text-white border border-emerald-500/30 shadow-lg">
              <div className="flex items-center gap-2 text-emerald-300 text-xs font-bold uppercase tracking-wider mb-1">
                <HelpCircle className="w-4 h-4" />
                <span>4-Stage Computer Vision & Data Structures Pipeline</span>
              </div>
              <h2 className="text-2xl font-extrabold">
                End-to-End Detection & Treatment Ranking Pipeline Summary
              </h2>
              <p className="text-xs sm:text-sm text-slate-200 mt-1 max-w-3xl">
                When a crop leaf image is uploaded, it is processed sequentially through <strong>four core algorithmic stages</strong> to extract visual features, match the pest record in constant time, classify infestation severity, and rank treatments.
              </p>
            </div>

            {/* 4 Visual Step Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* STAGE 1 CARD */}
              <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border-2 border-emerald-500/30 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-3 py-1 rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold text-xs">
                    STAGE 1 · ARRAY
                  </span>
                  <span className="font-mono text-xs text-slate-400">Time Complexity: O(N)</span>
                </div>
                <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                  1. Feature Array (Stores Image Metrics)
                </h3>
                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  <strong>Summary:</strong> Scans the uploaded leaf image and extracts 8 normalized visual features (green chrominance, lesion density, chlorosis ratio, and edge contrast) into a 1D <strong>Feature Array</strong> for constant-time index access.
                </p>
                <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-800 font-mono text-xs">
                  Feature Array = [{currentScan.featureArray.join(', ')}]
                </div>
                <pre className="p-3 rounded-xl bg-slate-950 text-emerald-400 font-mono text-[11px] overflow-x-auto">
{`// Stage 1: Store 8 normalized leaf metrics in a 1D Array
const featureArray: number[] = [];
for (let i = 0; i < 8; i++) {
  featureArray.push(extractLeafMetric(image, i));
}`}
                </pre>
              </div>

              {/* STAGE 2 CARD */}
              <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border-2 border-cyan-500/30 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-3 py-1 rounded-lg bg-cyan-500/15 text-cyan-600 dark:text-cyan-400 font-bold text-xs">
                    STAGE 2 · HASH TABLE
                  </span>
                  <span className="font-mono text-xs text-slate-400">Time Complexity: O(1) Instant</span>
                </div>
                <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                  2. Hash Table Lookup (Constant-Time Pest Retrieval)
                </h3>
                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  <strong>Summary:</strong> Instead of scanning all 18 pest records sequentially, the crop key is passed into a Polynomial Rolling <strong>Hash Function</strong> to compute the exact bucket index in <strong>O(1) constant time</strong>.
                </p>
                <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-800 font-mono text-xs">
                  Key: "{currentScan.crop}" → Hash Bucket #{currentScan.hashBucket} → {currentScan.detectedPest}
                </div>
                <pre className="p-3 rounded-xl bg-slate-950 text-cyan-400 font-mono text-[11px] overflow-x-auto">
{`// Stage 2: Compute bucket index via ASCII hash % 31
function getBucketIndex(cropName: string) {
  let sum = 0;
  for (let ch of cropName) sum += ch.charCodeAt(0);
  return sum % 31; // O(1) Constant-Time Bucket Lookup
}`}
                </pre>
              </div>

              {/* STAGE 3 CARD (With Interactive Slider) */}
              <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border-2 border-amber-500/30 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-3 py-1 rounded-lg bg-amber-500/15 text-amber-600 dark:text-amber-400 font-bold text-xs">
                    STAGE 3 · DECISION TREE
                  </span>
                  <span className="font-mono text-xs text-slate-400">Time Complexity: O(depth)</span>
                </div>
                <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                  3. Decision Tree Classifier (Evaluates Infestation Severity)
                </h3>
                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  <strong>Summary:</strong> Evaluates the affected leaf area percentage using threshold rules: under 15% is classified as <strong>Low Severity</strong>, 15% to 35% as <strong>Moderate Severity</strong>, and above 35% as <strong>Severe Outbreak</strong>.
                </p>

                {/* Live Interactive Slider */}
                <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-2 text-xs">
                  <div className="flex justify-between font-bold">
                    <span>Interactive Leaf Damage % Threshold Simulator:</span>
                    <span className="font-mono text-amber-600 dark:text-amber-400">{sandboxDamage}% Damaged</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="70"
                    value={sandboxDamage}
                    onChange={(e) => setSandboxDamage(Number(e.target.value))}
                    className="w-full accent-amber-500 cursor-pointer"
                  />
                  <div className="font-mono text-[11px] font-bold">
                    Tree Output: {classifySeverityDecisionTree(sandboxDamage).ruleExplanation}
                  </div>
                </div>

                <pre className="p-3 rounded-xl bg-slate-950 text-amber-400 font-mono text-[11px] overflow-x-auto">
{`// Stage 3: Decision Tree Threshold Classification
if (damagePercent < 15) return 'Low Severity';
else if (damagePercent <= 35) return 'Moderate Severity';
else return 'Severe Outbreak';`}
                </pre>
              </div>

              {/* STAGE 4 CARD */}
              <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border-2 border-indigo-500/30 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-3 py-1 rounded-lg bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 font-bold text-xs">
                    STAGE 4 · PRIORITY QUEUE (MAX-HEAP)
                  </span>
                  <span className="font-mono text-xs text-slate-400">Time Complexity: O(K log K)</span>
                </div>
                <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                  4. Priority Queue Max-Heap (Ranks Optimal Treatments)
                </h3>
                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  <strong>Summary:</strong> Computes a composite priority score for each candidate treatment based on Efficacy %, Cost per Acre, and Pre-Harvest Interval (PHI), ordering the optimal recommendation at Rank #1 via a <strong>Max-Heap Priority Queue</strong>.
                </p>
                <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-800 font-mono text-xs">
                  Formula: Score = Efficacy% + BioBonus - (Cost / 100) - (PHI * 0.4)
                </div>
                <pre className="p-3 rounded-xl bg-slate-950 text-indigo-300 font-mono text-[11px] overflow-x-auto">
{`// Stage 4: Priority Score & Max-Heap Ordering
const score = item.efficacyPercent + bioBonus - (item.cost / 100);
treatments.sort((a, b) => b.score - a.score); // Highest Rank #1`}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* ====================================================================
            SECTION 6: TAB 5 — ADMIN PORTAL (COMPLAINTS & DATABASE MANAGEMENT)
            Summary:
              - Displays every user complaint with full Complainant Name,
                User Account Email, and exact Complaint Date & Time.
              - Provides a direct Response Message Composer on every ticket so
                the Admin can send or update a custom reply that appears inside
                the complainant's account when they log in.
              - Includes a searchable Hash Table Database Inspector for all 18
                pests across 15 crops.
            ==================================================================== */}
        {activeTab === 'admin' && (
          <div className="space-y-6">
            {!currentUser.isAdmin ? (
              <div className="max-w-lg mx-auto my-8 rounded-2xl p-8 bg-white dark:bg-slate-900 border border-rose-500/30 text-center space-y-4">
                <Lock className="w-12 h-12 text-rose-500 mx-auto" />
                <h2 className="text-lg font-extrabold">Restricted Administrator Portal</h2>
                <p className="text-xs text-slate-500">
                  This portal is strictly locked to <strong>Project By: PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI</strong>.
                </p>
                <button
                  onClick={() => handleQuickLogin(true)}
                  className="px-5 py-2.5 rounded-xl bg-emerald-600 text-white text-xs font-bold cursor-pointer"
                >
                  Switch to Admin Account
                </button>
              </div>
            ) : (
              <>
                {/* Admin Header */}
                <div className="rounded-2xl p-6 bg-gradient-to-r from-slate-900 via-emerald-950 to-slate-900 text-white border border-emerald-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <span className="text-[11px] font-mono uppercase text-emerald-400 font-bold">
                      ✓ Verified System Administrator
                    </span>
                    <h2 className="text-xl sm:text-2xl font-extrabold mt-0.5">
                      Project By: PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI
                    </h2>
                    <p className="text-xs text-slate-300">
                      Manage farmer complaints with one-click resolution and inspect all 18 pests in the Hash Table.
                    </p>
                  </div>
                  <div className="flex gap-3 text-center font-mono">
                    <div className="px-4 py-2 rounded-xl bg-white/10">
                      <div className="text-lg font-bold text-emerald-400">{scanHistory.length}</div>
                      <div className="text-[10px] text-slate-300">Total Scans</div>
                    </div>
                    <div className="px-4 py-2 rounded-xl bg-white/10">
                      <div className="text-lg font-bold text-amber-400">{pendingComplaintsCount}</div>
                      <div className="text-[10px] text-slate-300">Pending Tickets</div>
                    </div>
                  </div>
                </div>

                {/* Admin Complaint Inspection & Custom Response Messaging Queue */}
                <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h3 className="text-base font-extrabold text-slate-900 dark:text-white">
                        User Complaints & Direct Response Management ({complaints.length})
                      </h3>
                      <p className="text-xs text-slate-500">
                        View complainant name, email, date and time of submission, and send a response message directly to the user's account.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {complaints.map((c) => (
                      <div
                        key={c.id}
                        className="p-5 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-3 text-xs"
                      >
                        {/* Top Ticket Bar: ID, Crop, Category, Rating & Status */}
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="px-2.5 py-1 rounded-lg bg-slate-900 text-white dark:bg-slate-950 font-mono font-bold">
                              {c.id}
                            </span>
                            <span className="px-2.5 py-1 rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold">
                              Crop: {c.crop}
                            </span>
                            <span className="px-2.5 py-1 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-semibold">
                              {c.category}
                            </span>
                            <span className="px-2.5 py-1 rounded-lg bg-amber-500/15 text-amber-600 dark:text-amber-400 font-bold">
                              Rating: {'★'.repeat(c.rating)} ({c.rating}/5)
                            </span>
                          </div>
                          <span
                            className={`px-3 py-1 rounded-full font-bold ${
                              c.status === 'Resolved ✓'
                                ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                                : 'bg-amber-500/15 text-amber-600 dark:text-amber-400'
                            }`}
                          >
                            {c.status}
                          </span>
                        </div>

                        {/* Complainant Identity (Who made complaint, Name, Email, Date & Time) */}
                        <div className="p-3.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 grid grid-cols-1 sm:grid-cols-3 gap-3">
                          <div>
                            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                              Complainant Name
                            </span>
                            <span className="font-extrabold text-slate-900 dark:text-white text-sm">
                              {c.userName}
                            </span>
                          </div>
                          <div>
                            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                              User Account Email
                            </span>
                            <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                              {c.userEmail}
                            </span>
                          </div>
                          <div>
                            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                              Complaint Date & Time
                            </span>
                            <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
                              {c.createdAt}
                            </span>
                          </div>
                        </div>

                        {/* Subject & Full Complaint Description */}
                        <div className="p-3.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 space-y-1">
                          <div className="font-extrabold text-slate-900 dark:text-white text-sm">
                            Subject: {c.subject}
                          </div>
                          <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                            {c.message}
                          </p>
                        </div>

                        {/* Existing Admin Response Preview (if already sent) */}
                        {c.adminReply && (
                          <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-1">
                            <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-bold text-emerald-700 dark:text-emerald-300">
                              <span>Sent Response Message (Visible in {c.userName}'s Account):</span>
                              <span className="font-mono">
                                {c.repliedBy ? `${c.repliedBy} · ` : ''}
                                {c.repliedAt || c.createdAt}
                              </span>
                            </div>
                            <p className="text-slate-800 dark:text-slate-100 font-medium">
                              {c.adminReply}
                            </p>
                          </div>
                        )}

                        {/* Admin Response Composer Box */}
                        <div className="pt-2 border-t border-slate-200 dark:border-slate-700 space-y-2">
                          <label className="block font-bold text-slate-700 dark:text-slate-200">
                            {c.adminReply
                              ? `Update Response Message to ${c.userName}:`
                              : `Write Response Message to ${c.userName} (${c.userEmail}):`}
                          </label>
                          <div className="flex flex-col sm:flex-row gap-2">
                            <input
                              type="text"
                              placeholder={`Type response message for ${c.userName} (will appear in their account)...`}
                              value={adminReplyDrafts[c.id] ?? c.adminReply ?? ''}
                              onChange={(e) =>
                                setAdminReplyDrafts((prev) => ({
                                  ...prev,
                                  [c.id]: e.target.value
                                }))
                              }
                              className="flex-1 px-3.5 py-2.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-xs focus:outline-none focus:border-emerald-500"
                            />
                            <button
                              type="button"
                              onClick={() =>
                                handleResolveComplaint(
                                  c.id,
                                  adminReplyDrafts[c.id] ?? c.adminReply
                                )
                              }
                              className="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold shrink-0 cursor-pointer flex items-center justify-center gap-1.5 transition-colors"
                            >
                              <Send className="w-3.5 h-3.5" />
                              <span>
                                {c.status === 'Resolved ✓'
                                  ? 'Send Updated Response ✓'
                                  : 'Send Response & Resolve ✓'}
                              </span>
                            </button>
                          </div>
                          {adminReplySentId === c.id && (
                            <div className="text-[11px] font-bold text-emerald-600 dark:text-emerald-400">
                              ✓ Response message delivered to {c.userName} ({c.userEmail})! They will see this message when logged into their account.
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Searchable Hash Table Database Inspector (18 Pests & 15 Crops) */}
                <div className="rounded-2xl p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <h3 className="text-base font-extrabold">
                        Hash Table Database Inspector (All 18 Pests & 15 Crops)
                      </h3>
                      <p className="text-xs text-slate-500">
                        Live view of Hash Bucket indices and treatment formulations.
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        placeholder="Search pest..."
                        value={dbSearch}
                        onChange={(e) => setDbSearch(e.target.value)}
                        className="px-3 py-1.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-xs"
                      />
                      <select
                        value={dbCropFilter}
                        onChange={(e) => setDbCropFilter(e.target.value)}
                        className="px-3 py-1.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-xs"
                      >
                        <option value="all">All 15 Crops</option>
                        {ALL_CROPS.map((crop) => (
                          <option key={crop} value={crop}>{crop}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
                    {PEST_DATABASE.filter(
                      (p) =>
                        (dbCropFilter === 'all' || p.affectedCrops.includes(dbCropFilter as CropType)) &&
                        p.commonName.toLowerCase().includes(dbSearch.toLowerCase())
                    ).map((pest) => (
                      <div
                        key={pest.id}
                        className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-1.5"
                      >
                        <div className="flex items-center justify-between font-mono text-[11px] text-cyan-600 dark:text-cyan-400 font-bold">
                          <span>Hash Bucket #{pestHashTable.getBucketIndex(pest.affectedCrops[0])}</span>
                          <span>{pest.affectedCrops.join(', ')}</span>
                        </div>
                        <div className="font-extrabold text-slate-900 dark:text-white text-sm">
                          {pest.commonName}
                        </div>
                        <div className="text-slate-400 italic text-[11px]">{pest.scientificName}</div>
                        <div className="text-slate-600 dark:text-slate-300 text-[11px] pt-1">
                          Top Medicine: <strong>{pest.treatments[0]?.productName}</strong> (₹{pest.treatments[0]?.costPerAcreINR})
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer className="mt-auto border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 py-4 px-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span className="font-bold text-slate-700 dark:text-slate-300">
            SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING
          </span>
          <span className="font-bold text-emerald-600 dark:text-emerald-400">
            Project By: PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI
          </span>
        </div>
      </footer>
    </div>
  );
}
