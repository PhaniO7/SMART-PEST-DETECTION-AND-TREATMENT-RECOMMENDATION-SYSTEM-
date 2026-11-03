# SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING

**5th-Semester B.Tech AI & DS Design Project (100% Python Implementation)**  
**Project By:** PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI

---

## 1. Project Overview
This project is a **100% Python** agricultural decision-support system combining:
1. **Transfer Learning CNN (`PyTorch` + `torchvision`)**: Classifies insect pests on the **IP102 Benchmark Dataset** (Top 20 classes across all 8 crops: Rice, Corn, Wheat, Beet, Alfalfa, Vitis, Citrus, Mango) and outputs Softmax Top-3 probabilities.
2. **Object Detection (`Ultralytics YOLOv8s`)**: Localizes pests with bounding boxes and computes `affected_area_percent = (union_box_area / image_area) * 100`.
3. **4-Stage Data Structures & Algorithms (DSA) Pipeline (`dsa_pipeline.py`)**:
   - **Stage 1 — Fixed-Size 8-Element Feature Array (`# [ARRAY]`)**: Extracts 8 normalized OpenCV visual features in $O(H \times W)$ time.
   - **Stage 2 — 31-Bucket Hash Table (`# [HASH_TABLE]`)**: Built from scratch using a Polynomial Rolling Hash ($\text{base } 31, \text{modulo } 31$) and singly linked list Separate Chaining for $O(1)$ average lookup of `"crop|pest"` records.
   - **Stage 3 — Binary Decision Tree (`# [DECISION_TREE]`)**: Classifies damage severity into **Low** ($<15\%$), **Moderate** ($15\text{--}35\%$), or **Severe** ($>35\%$).
   - **Stage 4 — Manual Binary Max-Heap Priority Queue (`# [PRIORITY_QUEUE]`)**: Built from scratch without `heapq` to rank organic, biological, and chemical treatments by Efficacy %, Pre-Harvest Interval (PHI) safety, and Cost per Acre in INR ($\text{₹}$).
4. **Flask Web Application + SQLite (`app.py`, `database.py`, `auth.py`)**: Role-based authentication (`user` and `admin`), Leaf Scanner, Scan History with CSV export, Interactive 4-Stage DSA Guide, and Complaint Tracker with direct Admin responses.

---

## 2. Quick Start (Local Execution)

```bash
# 1. Install all Python dependencies
pip install -r requirements.txt

# 2. Run the automated pytest suite (tests all 4 DSA stages + Flask health route)
pytest

# 3. Start the Flask web application on http://127.0.0.1:5000
python app.py
```

---

## 3. Training Real Models on IP102 (Google Colab or Local GPU)

See `dataset_setup.md` for full IP102 download links and directory verification steps.

```bash
# Step 1: Select top 20 IP102 classes across all 8 crops, build CNN folders & YOLO labels
python prepare_dataset.py --ip102-root /path/to/ip102_v1.1 --top-n 20

# Step 2: Train 2-phase Transfer Learning CNN (saves models/cnn_pest.pt)
python train_cnn.py

# Step 3: Train Ultralytics YOLOv8s detector (saves models/yolo_pest.pt)
python train_yolo.py

# Step 4: Generate Confusion Matrix PNG, Classification Report CSV, and YOLO mAP metrics
python evaluate.py
```
