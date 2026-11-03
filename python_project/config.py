"""
================================================================================
FILE: config.py
PURPOSE:
    Central configuration file for the entire Python project:
    "SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING
     COMPUTER VISION AND MACHINE LEARNING"
    Project By: PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI

CONCEPTS USED:
    - Centralised Configuration Pattern (# [CONFIG]): Keeps every tunable number,
      file path, model hyperparameter, and scoring weight in ONE place so there
      are zero "magic numbers" scattered across other files.

HOW IT CONNECTS TO OTHER FILES:
    - Imported by `prepare_dataset.py` for IP102 dataset paths and TOP_N_CLASSES.
    - Imported by `train_cnn.py`, `train_yolo.py`, and `evaluate.py` for training
      hyperparameters and model weight paths.
    - Imported by `dsa_pipeline.py` for Hash Table bucket size, Decision Tree
      severity thresholds, and Priority Queue scoring weights.
    - Imported by `models_loader.py`, `inference.py`, `database.py`, `auth.py`,
      and `app.py` for Flask settings, confidence cutoffs, and admin accounts.
================================================================================
"""

# Import os module to build cross-platform file and folder paths safely
import os

# ==============================================================================
# 1. PROJECT METADATA & BRANDING CONSTANTS
# ==============================================================================

# # [CONFIG] Official project title displayed on every page header and login card
PROJECT_TITLE = (
    "SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM "
    "USING COMPUTER VISION AND MACHINE LEARNING"
)

# # [CONFIG] # [CHANGE_HERE] Edit team member names if needed for your report
PROJECT_BY = "PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI"

# # [CONFIG] Base directory of the project (automatically resolves to this folder)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ==============================================================================
# 2. FLASK & SQLITE DATABASE CONFIGURATION
# ==============================================================================

# # [CONFIG] # [CHANGE_HERE] Secret key used by Flask to sign session cookies
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "btech-5th-sem-smart-pest-secret-key-2026")

# # [CONFIG] # [DATABASE] Path to the local SQLite database file
SQLITE_DB_PATH = os.path.join(BASE_DIR, "pest_system.db")

# # [CONFIG] # [DATABASE] SQLAlchemy connection URI pointing to SQLite
SQLALCHEMY_DATABASE_URI = f"sqlite:///{SQLITE_DB_PATH}"

# # [CONFIG] Disable SQLAlchemy modification tracking overhead
SQLALCHEMY_TRACK_MODIFICATIONS = False

# # [CONFIG] Folder where user-uploaded leaf photos and YOLO annotated images are saved
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# # [CONFIG] Allowed image file extensions for leaf photo uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# # [CONFIG] # [CHANGE_HERE] Port number for running the local Flask web server
PORT = int(os.environ.get("PORT", 5000))

# # [CONFIG] # [ADMIN] # [CHANGE_HERE] Default Admin email addresses that automatically get 'admin' role
ADMIN_EMAILS = {
    "phaninandan78@gmail.com",
    "vedhesha@pestproject.in",
    "khadga@pestproject.in",
    "shivani@pestproject.in",
}

# ==============================================================================
# 3. IP102 DATASET & PREPARATION CONFIGURATION
# ==============================================================================

# # [CONFIG] # [DATASET] # [CHANGE_HERE] Path to your downloaded raw IP102 dataset folder
RAW_IP102_DIR = os.environ.get("RAW_IP102_DIR", os.path.join(BASE_DIR, "raw_ip102"))

# # [CONFIG] # [DATASET] Output folder for the filtered 20-class CNN classification dataset
CNN_DATASET_DIR = os.path.join(BASE_DIR, "dataset_cnn")

# # [CONFIG] # [DATASET] Output folder for the filtered 20-class YOLOv8 detection dataset
YOLO_DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolo")

# # [CONFIG] # [DATASET] Path to the generated YOLOv8 data.yaml configuration file
YOLO_DATA_YAML_PATH = os.path.join(YOLO_DATASET_DIR, "data.yaml")

# # [CONFIG] # [DATASET] # [CHANGE_HERE] Number of top IP102 classes to select (must be between 15 and 20)
TOP_N_CLASSES = 20

# # [CONFIG] # [DATASET] Minimum and maximum allowed values for TOP_N_CLASSES
MIN_CLASSES_ALLOWED = 15
MAX_CLASSES_ALLOWED = 20

# # [CONFIG] # [DATASET] Minimum number of distinct crops that must be represented in selected classes
MIN_CROPS_REPRESENTED = 2

# ==============================================================================
# 4. REAL DEEP LEARNING MODELS (CNN & YOLOv8) CONFIGURATION
# ==============================================================================

# # [CONFIG] Directory where trained model weights (.pt) and class_map.json are stored
MODELS_DIR = os.path.join(BASE_DIR, "models")

# # [CONFIG] # [CNN] Path to the trained PyTorch CNN weights file
CNN_WEIGHTS_PATH = os.path.join(MODELS_DIR, "cnn_pest.pt")

# # [CONFIG] # [YOLO] Path to the trained Ultralytics YOLOv8 weights file
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolo_pest.pt")

# # [CONFIG] # [DATASET] Path to the JSON mapping between class indices, IP102 IDs, pest names, and crops
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_map.json")

# # [CONFIG] # [CNN] # [CHANGE_HERE] Backbone architecture for transfer learning ('efficientnet_b0' or 'resnet50')
CNN_BACKBONE_NAME = "efficientnet_b0"

# # [CONFIG] # [PREPROCESS] Input image width and height (in pixels) required by the CNN
CNN_IMAGE_SIZE = 224

# # [CONFIG] # [PREPROCESS] Standard ImageNet RGB mean values used to normalise CNN input tensors
IMAGENET_MEAN = [0.485, 0.456, 0.406]

# # [CONFIG] # [PREPROCESS] Standard ImageNet RGB standard deviation values for CNN input tensors
IMAGENET_STD = [0.229, 0.224, 0.225]

# # [CONFIG] # [YOLO] # [CHANGE_HERE] Base pretrained YOLOv8 checkpoint used when starting training
YOLO_BASE_MODEL = "yolov8s.pt"

# # [CONFIG] # [YOLO] Input image resolution for YOLOv8 training and inference
YOLO_IMAGE_SIZE = 640

# # [CONFIG] # [THRESHOLD] # [CHANGE_HERE] Minimum confidence score (0.0 to 1.0) required to accept a prediction.
# Below this cutoff, the app displays "Uncertain, please retake the photo" instead of guessing.
CONFIDENCE_CUTOFF = 0.45

# # [CONFIG] # [YOLO] # [THRESHOLD] Minimum confidence threshold for drawing a YOLO bounding box
YOLO_BOX_CONF_THRESHOLD = 0.25

# # [CONFIG] # [YOLO] # [THRESHOLD] Intersection-over-Union (IoU) threshold for Non-Maximum Suppression (NMS)
YOLO_IOU_THRESHOLD = 0.45

# # [CONFIG] # [CNN] Number of top predictions returned by the CNN softmax classifier
CNN_TOP_K = 3

# ==============================================================================
# 5. TRAINING HYPERPARAMETERS (FOR GOOGLE COLAB GPU SCRIPTS)
# ==============================================================================

# # [CONFIG] # [CNN] # [CHANGE_HERE] Batch size for CNN training and validation data loaders
CNN_BATCH_SIZE = 32

# # [CONFIG] # [CNN] # [CHANGE_HERE] Number of epochs for Phase 1 (backbone frozen, train classifier head only)
CNN_EPOCHS_PHASE1_FROZEN = 5

# # [CONFIG] # [CNN] # [CHANGE_HERE] Number of epochs for Phase 2 (unfreeze backbone, fine-tune entire network)
CNN_EPOCHS_PHASE2_FINETUNE = 15

# # [CONFIG] # [CNN] Learning rate for Phase 1 (classifier head training)
CNN_LR_PHASE1 = 1e-3

# # [CONFIG] # [CNN] Lower learning rate for Phase 2 (full network fine-tuning)
CNN_LR_PHASE2 = 1e-4

# # [CONFIG] # [CNN] Weight decay (L2 regularisation) to prevent overfitting
CNN_WEIGHT_DECAY = 1e-4

# # [CONFIG] # [CNN] Early stopping patience (stop if validation F1 does not improve for N epochs)
EARLY_STOPPING_PATIENCE = 4

# # [CONFIG] # [YOLO] # [CHANGE_HERE] Number of epochs for YOLOv8 training
YOLO_EPOCHS = 30

# # [CONFIG] # [YOLO] Batch size for YOLOv8 training
YOLO_BATCH_SIZE = 16

# ==============================================================================
# 6. 4-STAGE DATA STRUCTURES & ALGORITHMS (DSA) CONFIGURATION
# ==============================================================================

# --- Stage 1: Feature Array (# [ARRAY]) ---
# # [CONFIG] # [ARRAY] Fixed number of visual features extracted from each leaf image via OpenCV
FEATURE_ARRAY_SIZE = 8

# # [CONFIG] # [ARRAY] Human-readable names of the 8 OpenCV visual features for the DSA Guide page
FEATURE_NAMES = [
    "0: Green Chrominance Ratio",
    "1: Lesion Spot Density",
    "2: Yellow Chlorosis Ratio",
    "3: Canny Edge Sharpness",
    "4: HSV Saturation Mean",
    "5: Necrotic Dark Ratio",
    "6: Laplacian Texture Variance",
    "7: Normalized Brightness Mean",
]

# --- Stage 2: Hash Table (# [HASH_TABLE]) ---
# # [CONFIG] # [HASH_TABLE] # [CHANGE_HERE] Prime number of buckets in our custom Hash Table
HASH_TABLE_BUCKETS = 31

# # [CONFIG] # [HASH_TABLE] Prime multiplier (base P) used in the Polynomial Rolling Hash function
HASH_POLYNOMIAL_BASE = 31

# --- Stage 3: Decision Tree Severity Thresholds (# [DECISION_TREE] # [SEVERITY]) ---
# # [CONFIG] # [DECISION_TREE] # [THRESHOLD] # [CHANGE_HERE] Affected leaf area % below this is 'Low' severity
SEVERITY_LOW_MAX_PERCENT = 15.0

# # [CONFIG] # [DECISION_TREE] # [THRESHOLD] # [CHANGE_HERE] Affected leaf area % up to this is 'Moderate'; above is 'Severe'
SEVERITY_MODERATE_MAX_PERCENT = 35.0

# --- Stage 4: Priority Queue Max-Heap Scoring Weights (# [PRIORITY_QUEUE] # [TREATMENT]) ---
# # [CONFIG] # [PRIORITY_QUEUE] # [CHANGE_HERE] Bonus points added to Organic/Biological treatments when severity is Low
ORGANIC_BONUS_LOW_SEVERITY = 18.0

# # [CONFIG] # [PRIORITY_QUEUE] # [CHANGE_HERE] Bonus points added to Organic/Biological treatments when severity is Moderate
ORGANIC_BONUS_MODERATE_SEVERITY = 6.0

# # [CONFIG] # [PRIORITY_QUEUE] # [CHANGE_HERE] Bonus points added to Chemical treatments when severity is Severe
CHEMICAL_BONUS_SEVERE = 8.0

# # [CONFIG] # [PRIORITY_QUEUE] # [CHANGE_HERE] Penalty divisor for Cost per Acre in INR (cost_inr / 100.0)
COST_PENALTY_DIVISOR = 100.0

# # [CONFIG] # [PRIORITY_QUEUE] # [CHANGE_HERE] Penalty multiplier per waiting day in Pre-Harvest Interval (PHI)
PHI_PENALTY_MULTIPLIER = 0.4

# # [CONFIG] # [PRIORITY_QUEUE] Minimum and maximum clamped bounds for the final Priority Score
PRIORITY_SCORE_MIN = 50.0
PRIORITY_SCORE_MAX = 99.5
