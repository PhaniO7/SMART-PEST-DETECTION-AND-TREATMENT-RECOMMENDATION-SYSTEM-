# Smart Pest Detection and Treatment Recommendation System using Computer Vision and Machine Learning

**B.Tech AI & DS Design Project (100% Python)**  
**Team:** PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI  
**Scope:** **8 crops and 20 pests** from the **IP102** Insect Pest Dataset (`NUM_CROPS = len(CROPS)` and `NUM_PESTS = len(PEST_SHORTLIST)` in `config.py`)

---

## 1. Project Files (13 Files Only)

| File | Purpose |
| :--- | :--- |
| `config.py` | All settings, constants, thresholds, weights, the 8 crops list (`CROPS`), and the 20-pest list (`PEST_SHORTLIST`). |
| `pest_data.py` | Pest and treatment database as pure Python dictionaries (Organic, Biological, and Chemical options with efficacy %, PHI days, and INR cost/acre). |
| `dsa.py` | The 4 Data Structure & Algorithm stages written from scratch (`# [ARRAY]`, `# [HASH_TABLE]`, `# [DECISION_TREE]`, `# [PRIORITY_QUEUE]`) with Big-O complexity comments. |
| `detect.py` | Real PyTorch `EfficientNet-B0` (`models/cnn_pest.pt`) + Ultralytics `YOLOv8n` (`models/yolo_pest.pt`) inference, fusion, and `time.perf_counter()` DSA speed measurement. |
| `database.py` | SQLite (`sqlite3`) database for users, login (`hashlib.pbkdf2_hmac`), scan history, user complaints + admin replies, and real test-set evaluation metrics. |
| `app.py` | Complete Streamlit website: Login/Register, Leaf Scanner, Scan History, Feedback & Complaint Tracker, Admin Dashboard, and DSA Guide. |
| `prepare_data.py` | Matches `PEST_SHORTLIST` against IP102 `classes.txt`, builds CNN folders, converts Pascal VOC XML boxes to YOLO format, and writes `data.yaml`. |
| `train_cnn.py` | PyTorch `EfficientNet-B0` transfer learning with mixed precision, class-weighted loss, data augmentation, and `--quick-test` (2 epochs) option. |
| `train_yolo.py` | Ultralytics `YOLOv8n` training with mixed precision, RTX 4050 (6 GB VRAM) settings, and `--quick-test` (2 epochs) option. |
| `evaluate.py` | Evaluates trained models on the IP102 test split, saves `models/confusion_matrix.png`, and writes real test accuracy, macro F1, and YOLO mAP50 to SQLite. |
| `test_all.py` | Automated `pytest` / `unittest` suite testing all 4 DSA stages, dataset matching, and SQLite live metrics. |
| `requirements.txt` | Python dependencies required to run the project locally in VS Code. |
| `README.md` | Project documentation and step-by-step run order. |

---

## 2. Supported Scope: 8 Crops and 20 Pests (IP102)

Defined in `config.py` (`CROPS` and `PEST_SHORTLIST`):
1. **Rice:** brown plant hopper, asiatic rice borer, rice leaf roller
2. **Corn:** corn borer, armyworm, aphids
3. **Wheat:** english grain aphid, red spider, wheat blossom midge
4. **Beet:** beet army worm, beet weevil
5. **Alfalfa:** alfalfa weevil, tarnished plant bug, blister beetle
6. **Vitis (Grape):** ampelophaga, brevipoalpus lewisi mcgregor
7. **Citrus:** panonychus citri (citrus red mite), phyllocnistis citrella (citrus leafminer)
8. **Mango:** dacus dorsalis (mango fruit fly), cicadellidae (mango leafhopper)

> **Agronomic Verification Note:** All treatment efficacy percentages, Pre-Harvest Interval (PHI) days, dosages, and costs in INR inside `pest_data.py` are educational reference estimates and must be verified against official **ICAR** or **TNAU Agritech Portal** guidelines before field use.

---

## 3. Step-by-Step Run Order in VS Code

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Set Environment Variables (Optional, Recommended)
Set your admin email(s) and password hashing salt in your terminal before running:

**Windows PowerShell:**
```powershell
$env:PEST_ADMIN_EMAILS="your_admin_email@example.com"
$env:PEST_PASSWORD_SALT="your_secret_random_salt"
```

**Linux / macOS:**
```bash
export PEST_ADMIN_EMAILS="your_admin_email@example.com"
export PEST_PASSWORD_SALT="your_secret_random_salt"
```

### Step 3 — Run Automated Tests
```bash
pytest test_all.py
```

### Step 4 — Prepare the IP102 Dataset (8 Crops, 20 Pests)
Download the **IP102** dataset into `ip102_v1.1/` (or update `IP102_RAW_DIR` in `config.py`) so it contains `classes.txt`, `train.txt`, `val.txt`, `test.txt`, `images/`, and `Annotations/`. Then run:
```bash
python prepare_data.py
```
- `prepare_data.py` removes spaces, hyphens, and brackets and lowercases both sides when matching `PEST_SHORTLIST` against `classes.txt`.
- If any pest name matches zero classes or more than one class (such as `"aphids"`), it stops and prints the candidate class names from `classes.txt` so you can update `match_keyword` in `config.py`.

### Step 5 — Train the CNN and YOLOv8n Models (RTX 4050 6 GB VRAM Ready)
```bash
# Quick 2-epoch test run:
python train_cnn.py --quick-test
python train_yolo.py --quick-test

# Full training run:
python train_cnn.py
python train_yolo.py
```

### Step 6 — Evaluate on Test Split & Save Real Accuracy to SQLite
```bash
python evaluate.py
```
This saves `models/confusion_matrix.png` and records the real test-set accuracy, macro F1, YOLO mAP50, and timestamp in `pest_system.db`.

### Step 7 — Launch the Streamlit Website
```bash
streamlit run app.py
```

---

## 4. Searchable Code Tags (`Ctrl+F` in VS Code)

Use `Ctrl+F` (or `Ctrl+Shift+F` across all files) to jump directly to any concept during your viva or code review:
- `# [ARRAY]` — Stage 1 Fixed-Size 8-Element OpenCV Feature Array
- `# [HASH_TABLE]` — Stage 2 Custom 31-Bucket Polynomial Rolling Hash Table with Separate Chaining
- `# [DECISION_TREE]` — Stage 3 Severity Decision Tree (`< 15%` Low, `15-35%` Moderate, `> 35%` Severe)
- `# [PRIORITY_QUEUE]` — Stage 4 Manual Binary Max-Heap Priority Queue for Treatment Ranking
- `# [CNN]` — PyTorch `EfficientNet-B0` Transfer Learning & Softmax Top-3 Inference
- `# [YOLO]` — Ultralytics `YOLOv8n` Bounding-Box Detection & Affected Area % Calculation
- `# [PREPROCESS]` — OpenCV & Torchvision Image Resizing, Augmentation, and Normalization
- `# [DATASET]` — IP102 Dataset Matching, Class Filtering, and VOC XML to YOLO Conversion
- `# [DATABASE]` — SQLite Tables (`users`, `scans`, `feedback`, `evaluation_metrics`)
- `# [LOGIN]` — PBKDF2-HMAC-SHA256 Password Hashing, Registration, Sign-In, and `DEMO_MODE`
- `# [ADMIN]` — Administrator Role Checks and Complaint Response Management
- `# [FEEDBACK]` — User Complaint Submission, Status Tracking, and Admin Replies
- `# [CONFIG]` / `# [THRESHOLD]` / `# [SEVERITY]` / `# [TREATMENT]` / `# [CHANGE_HERE]` — Tunable Parameters
