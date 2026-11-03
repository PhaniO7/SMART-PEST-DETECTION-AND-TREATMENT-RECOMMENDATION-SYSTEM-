# SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING

**5th-Semester B.Tech AI & DS Design Project**  
**Project By:** PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI

---

## 1. Project Overview
Farmers often struggle to identify crop leaf pests early and choose the right, cost-effective treatment with a safe Pre-Harvest Interval (PHI). This web application combines computer vision feature extraction with **four core Data Structures & Algorithms (DSA)** to identify pests across **15 major crops** and rank treatments in milliseconds.

---

## 2. Core 4-Stage Data Structures & Algorithms Pipeline Summary

All four data structures are implemented cleanly inside `src/dataAndAlgorithms.ts`:

1. **Stage 1: Feature Array (`number[8]`)**
   - Extracts and stores 8 normalized visual metrics from the uploaded leaf image (spot density, color change, edge contrast, texture, etc.) in a fixed-size array for $O(1)$ index access.
2. **Stage 2: Hash Table ($O(1)$ Constant-Time Lookup)**
   - Uses a custom Polynomial Rolling Hash function (`hashKey`) across 31 buckets to map the crop and pest name directly to its agricultural database record without scanning a list one by one.
3. **Stage 3: Decision Tree Classifier ($O(H)$ Severity Evaluation)**
   - Evaluates the percentage of damaged leaf area using a clear 3-level decision tree:
     - `< 15% leaf damage` $\rightarrow$ **Low Severity** (Preventive Organic care)
     - `15% – 35% leaf damage` $\rightarrow$ **Moderate Severity** (Targeted IPM intervention)
     - `> 35% leaf damage` $\rightarrow$ **Severe Infestation** (Immediate high-efficacy control)
4. **Stage 4: Priority Queue Max-Heap ($O(N \log N)$ Treatment Ranking)**
   - Calculates a composite Priority Score for each candidate medicine based on **Efficacy %**, **Safety (Pre-Harvest Interval days)**, and **Cost per Acre (₹)**, returning the best recommendation at Rank `#1`.

---

## 3. User Complaint Tracking & Admin Response Workflow Summary

1. **User Complaint Submission (`Tab 3: Feedback & Complaints`)**:
   - Logged-in users can submit detection or treatment complaints. The system automatically records the complainant's **Name** (`userName`), **Account Email** (`userEmail`), and exact **Date & Time** (`createdAt` in `YYYY-MM-DD HH:MM:SS` format).
2. **Admin Inspection & Direct Response Messaging (`Tab 5: Admin Portal`)**:
   - Administrators can view who submitted each complaint (Name, Email, Date & Time, Crop, Category, Rating, and Description) and type a custom **Response Message** (`adminReply`).
   - Sending a response records the Admin's name (`repliedBy`) and reply timestamp (`repliedAt`), updates the ticket status to `Resolved ✓`, and persists the record in `localStorage`.
3. **User Account Complaint Tracker & Inbox**:
   - When the user logs into their account, an account notification banner and the **My Complaint Status Tracker** display their submitted tickets, a **3-Step Live Progress Bar** (`1. Logged` $\rightarrow$ `2. Assigned to Admin` $\rightarrow$ `3. Response Received`), and the official response message sent by the Admin.

---

## 4. Simplified Folder & File Structure

The codebase is intentionally organized into a flat, easy-to-read structure:

```text
├── index.html                    # Main HTML page and Google Fonts setup
├── package.json                  # Project dependencies and build scripts
├── vite.config.ts                # Vite bundler configuration
├── server.ts                     # Express backend & Vision API route
└── src/
    ├── main.tsx                  # Mounts React application into index.html
    ├── index.css                 # Tailwind CSS & Dark/Light mode styling
    ├── dataAndAlgorithms.ts      # 15 Crops, 18 Pests Database + 4 Data Structures
    ├── dataAndAlgorithms.test.ts # Automated unit tests for all 4 DSA stages
    └── App.tsx                   # Complete Frontend UI (Login, Scanner, History, Feedback, DSA Guide, Admin Portal)
```

---

## 4. How to Run Locally

```bash
# Step 1: Install dependencies
npm install --legacy-peer-deps

# Step 2: Run automated unit tests for the 4 Data Structures
npm test

# Step 3: Start the development server on http://localhost:3000
npm run dev

# Step 4: Build production bundle into /dist
npm run build
```
