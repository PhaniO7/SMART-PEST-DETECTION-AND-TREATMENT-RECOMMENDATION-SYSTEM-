"""
================================================================================
FILE: config.py
PURPOSE:
    Stores ALL tunable settings, file paths, model hyperparameters, thresholds,
    scoring weights, the 8 crops list (`CROPS`), and the 20-pest shortlist
    (`PEST_SHORTLIST`) in one place so there are zero magic numbers elsewhere.

HOW IT CONNECTS TO OTHER FILES:
    - Imported by `prepare_data.py` to match the 20 pests in IP102 `classes.txt`.
    - Imported by `pest_data.py` and `dsa.py` for hash table buckets, decision
      tree severity thresholds, and priority queue scoring weights.
    - Imported by `train_cnn.py`, `train_yolo.py`, `evaluate.py`, and `detect.py`
      for RTX 4050 (6 GB VRAM) training settings, model paths, and cutoffs.
    - Imported by `database.py` and `app.py` for SQLite settings, admin emails,
      and dynamic `{NUM_CROPS} crops and {NUM_PESTS} pests` labels.
================================================================================
"""

# Import os so we can build file paths that work on Windows, Mac, and Linux
import os

# ==============================================================================
# 1. PROJECT METADATA & ADMIN ACCOUNTS
# ==============================================================================

# # [CONFIG] Official project title displayed at the top of the Streamlit app
PROJECT_TITLE = (
    "Smart Pest Detection and Treatment Recommendation System "
    "using Computer Vision and Machine Learning"
)  # # [CHANGE_HERE] Change project title if needed

# # [CONFIG] Team member names displayed in the Streamlit header and sidebar
PROJECT_TEAM = (
    "PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI"
)  # # [CHANGE_HERE] Edit team member names here

# # [CONFIG] # [LOGIN] Quick-access demo accounts (including the admin account)
# may ONLY appear when DEMO_MODE = True.
# IMPORTANT: DEMO_MODE must be False when the project is shown or shared!
DEMO_MODE = False  # # [CHANGE_HERE] Keep False when showing or sharing the project

# # [CONFIG] # [ADMIN] # [LOGIN] Read comma-separated admin emails from the
# environment variable PEST_ADMIN_EMAILS (set in terminal before running:
#   Windows PowerShell: $env:PEST_ADMIN_EMAILS="your_admin@domain.com"
#   Linux / macOS:      export PEST_ADMIN_EMAILS="your_admin@domain.com"
# ). Falls back to a safe placeholder default if not set.
ADMIN_EMAILS = [
    email.strip().lower()
    for email in os.environ.get("PEST_ADMIN_EMAILS", "admin@example.com").split(",")
    if email.strip()
]  # # [CHANGE_HERE] Set PEST_ADMIN_EMAILS in your environment variables

# # [CONFIG] # [LOGIN] Read the password hashing salt from the environment
# variable PEST_PASSWORD_SALT (set in terminal before running:
#   Windows PowerShell: $env:PEST_PASSWORD_SALT="your_random_secret_salt"
#   Linux / macOS:      export PEST_PASSWORD_SALT="your_random_secret_salt"
# ). Falls back to a safe placeholder default if not set.
PASSWORD_HASH_SALT = os.environ.get(
    "PEST_PASSWORD_SALT", "change_this_placeholder_salt_in_env"
)  # # [CHANGE_HERE] Set PEST_PASSWORD_SALT in your environment variables

# # [CONFIG] # [LOGIN] Safe demo account credentials used ONLY when DEMO_MODE = True
DEMO_ADMIN_EMAIL = "admin@example.com"  # # [CHANGE_HERE] Placeholder demo admin email
DEMO_USER_EMAIL = "farmer@example.com"  # # [CHANGE_HERE] Placeholder demo farmer email
DEMO_DEFAULT_PASSWORD = os.environ.get(
    "PEST_DEMO_PASSWORD", "DemoPass#2026"
)  # # [CHANGE_HERE] Used only when DEMO_MODE = True

# # [CONFIG] # [LOGIN] Number of PBKDF2-HMAC-SHA256 iterations for password hashing
PASSWORD_HASH_ITERATIONS = 100_000  # # [CHANGE_HERE] Increase or decrease hash iterations


# ==============================================================================
# 2. DIRECTORY & FILE PATHS
# ==============================================================================

# # [CONFIG] Base folder where this Python project lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # [CONFIG] # [DATASET] Folder where the raw IP102 dataset is downloaded
IP102_RAW_DIR = os.path.join(
    BASE_DIR, "ip102_v1.1"
)  # # [CHANGE_HERE] Point this to your downloaded IP102 folder path

# # [CONFIG] # [DATASET] Folder where `prepare_data.py` builds CNN and YOLO datasets
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "datasets", "ip102_20pests")

# # [CONFIG] # [DATASET] Subfolders for CNN classification and YOLO detection data
CNN_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "cnn")
YOLO_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "yolo")
YOLO_DATA_YAML = os.path.join(YOLO_DATA_DIR, "data.yaml")

# # [CONFIG] # [CNN] # [YOLO] Folder where trained model weights are saved
MODELS_DIR = os.path.join(BASE_DIR, "models")
CNN_WEIGHTS_PATH = os.path.join(
    MODELS_DIR, "cnn_pest.pt"
)  # # [CHANGE_HERE] Path to trained PyTorch EfficientNet-B0 weights
YOLO_WEIGHTS_PATH = os.path.join(
    MODELS_DIR, "yolo_pest.pt"
)  # # [CHANGE_HERE] Path to trained Ultralytics YOLOv8n weights
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

# # [CONFIG] # [DATABASE] SQLite database file path and uploaded scan images folder
DATABASE_PATH = os.path.join(
    BASE_DIR, "pest_system.db"
)  # # [CHANGE_HERE] Change SQLite database filename if needed
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")


# ==============================================================================
# 3. SCOPE: 8 CROPS AND 20 PESTS (IP102 SHORTLIST)
# ==============================================================================

# # [CONFIG] # [DATASET] List of the 8 crops covered by the IP102 dataset
CROPS = [
    "Rice",
    "Corn",
    "Wheat",
    "Beet",
    "Alfalfa",
    "Vitis (Grape)",
    "Citrus",
    "Mango",
]  # # [CHANGE_HERE] List of supported host crops

# # [CONFIG] # [DATASET] Ordered list of 20 pests (2-3 per crop).
# IMPORTANT: Each item's index (0 to 19) IS its class ID across `prepare_data.py`,
# `train_cnn.py`, `train_yolo.py`, `pest_data.py`, `detect.py`, and `app.py`.
PEST_SHORTLIST = [
    # --- Crop 1: Rice (3 pests, indices 0..2) ---
    {"class_id": 0, "crop": "Rice", "name": "brown plant hopper", "match_keyword": "brown plant hopper"},
    {"class_id": 1, "crop": "Rice", "name": "asiatic rice borer", "match_keyword": "asiatic rice borer"},
    {"class_id": 2, "crop": "Rice", "name": "rice leaf roller", "match_keyword": "rice leaf roller"},
    # --- Crop 2: Corn (3 pests, indices 3..5) ---
    {"class_id": 3, "crop": "Corn", "name": "corn borer", "match_keyword": "corn borer"},
    {"class_id": 4, "crop": "Corn", "name": "armyworm", "match_keyword": "armyworm"},
    {"class_id": 5, "crop": "Corn", "name": "aphids", "match_keyword": "aphids"},
    # --- Crop 3: Wheat (3 pests, indices 6..8) ---
    {"class_id": 6, "crop": "Wheat", "name": "english grain aphid", "match_keyword": "english grain aphid"},
    {"class_id": 7, "crop": "Wheat", "name": "red spider", "match_keyword": "red spider"},
    {"class_id": 8, "crop": "Wheat", "name": "wheat blossom midge", "match_keyword": "wheat blossom midge"},
    # --- Crop 4: Beet (2 pests, indices 9..10) ---
    {"class_id": 9, "crop": "Beet", "name": "beet army worm", "match_keyword": "beet army worm"},
    {"class_id": 10, "crop": "Beet", "name": "beet weevil", "match_keyword": "beet weevil"},
    # --- Crop 5: Alfalfa (3 pests, indices 11..13) ---
    {"class_id": 11, "crop": "Alfalfa", "name": "alfalfa weevil", "match_keyword": "alfalfa weevil"},
    {"class_id": 12, "crop": "Alfalfa", "name": "tarnished plant bug", "match_keyword": "tarnished plant bug"},
    {"class_id": 13, "crop": "Alfalfa", "name": "blister beetle", "match_keyword": "blister beetle"},
    # --- Crop 6: Vitis (Grape) (2 major IP102 grape pests, indices 14..15) ---
    {"class_id": 14, "crop": "Vitis (Grape)", "name": "ampelophaga", "match_keyword": "ampelophaga"},
    {"class_id": 15, "crop": "Vitis (Grape)", "name": "brevipoalpus lewisi mcgregor", "match_keyword": "brevipoalpus lewisi"},
    # --- Crop 7: Citrus (2 pests, indices 16..17) ---
    {"class_id": 16, "crop": "Citrus", "name": "panonychus citri (citrus red mite)", "match_keyword": "panonychus citri"},
    {"class_id": 17, "crop": "Citrus", "name": "phyllocnistis citrella (citrus leafminer)", "match_keyword": "phyllocnistis citrella"},
    # --- Crop 8: Mango (2 pests, indices 18..19) ---
    {"class_id": 18, "crop": "Mango", "name": "dacus dorsalis (mango fruit fly)", "match_keyword": "dacus dorsalis"},
    {"class_id": 19, "crop": "Mango", "name": "cicadellidae (mango leafhopper)", "match_keyword": "cicadellidae"},
]  # # [CHANGE_HERE] Edit pest names or match_keyword if your IP102 classes.txt uses different spelling

# # [CONFIG] # [DATASET] Dynamic counts computed with len() so the UI is never out of date
NUM_CROPS = len(CROPS)
NUM_PESTS = len(PEST_SHORTLIST)

# # [CONFIG] # [DATASET] Minimum recommended images per class before `prepare_data.py` prints a warning
MIN_IMAGES_WARNING_THRESHOLD = 300  # # [CHANGE_HERE] Warn if any class has fewer than this many images


# ==============================================================================
# 4. MODEL & RTX 4050 (6 GB VRAM) TRAINING SETTINGS
# ==============================================================================

# # [CONFIG] # [CNN] # [PREPROCESS] Input image width/height for EfficientNet-B0
CNN_IMAGE_SIZE = 224  # # [CHANGE_HERE] 224x224 is standard for EfficientNet-B0

# # [CONFIG] # [CNN] Pretrained Torchvision backbone name
CNN_MODEL_NAME = "efficientnet_b0"  # # [CHANGE_HERE] Kept as EfficientNet-B0 for 6 GB VRAM efficiency

# # [CONFIG] # [CNN] Small batch size tailored for a laptop RTX 4050 with 6 GB VRAM
CNN_BATCH_SIZE = 16  # # [CHANGE_HERE] Reduce to 8 if VRAM is full, or increase to 32 on larger GPUs

# # [CONFIG] # [CNN] Number of Windows/Linux DataLoader worker processes (2-4 recommended)
DATALOADER_WORKERS = 2  # # [CHANGE_HERE] Set between 2 and 4 for laptop CPU cores

# # [CONFIG] # [CNN] Training epochs for full run vs quick-test mode
CNN_EPOCHS = 15  # # [CHANGE_HERE] Number of full training epochs for CNN
QUICK_TEST_EPOCHS = 2  # # [CHANGE_HERE] Number of epochs when `--quick-test` flag is used

# # [CONFIG] # [CNN] Learning rate for AdamW optimizer
CNN_LEARNING_RATE = 1e-3  # # [CHANGE_HERE] Initial learning rate for CNN transfer learning
CNN_WEIGHT_DECAY = 1e-4  # # [CHANGE_HERE] Weight decay regularization for AdamW

# # [CONFIG] # [PREPROCESS] Standard ImageNet RGB mean and standard deviation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# # [CONFIG] # [CNN] Number of top predictions returned by CNN Softmax
CNN_TOP_K = 3  # # [CHANGE_HERE] Top-3 CNN predictions shown on the scanner page

# # [CONFIG] # [YOLO] Ultralytics YOLOv8 nano model weight name (lightweight for 6 GB VRAM)
YOLO_BASE_WEIGHTS = "yolov8n.pt"  # # [CHANGE_HERE] Base YOLOv8n pretrained checkpoint
YOLO_IMAGE_SIZE = 512  # # [CHANGE_HERE] 512x512 keeps VRAM usage under 6 GB on RTX 4050
YOLO_BATCH_SIZE = 8  # # [CHANGE_HERE] Small batch size for 6 GB VRAM laptop GPU
YOLO_EPOCHS = 25  # # [CHANGE_HERE] Full training epochs for YOLOv8n

# # [CONFIG] # [YOLO] # [THRESHOLD] Minimum confidence to keep a YOLO bounding box
YOLO_CONF_THRESHOLD = 0.25  # # [CHANGE_HERE] Lower to detect more boxes, raise for stricter boxes
YOLO_IOU_THRESHOLD = 0.45  # # [CHANGE_HERE] Non-Maximum Suppression IoU threshold

# # [CONFIG] # [THRESHOLD] Confidence cutoff below which the app shows "Uncertain, please retake the photo"
CONFIDENCE_CUTOFF = 0.45  # # [CHANGE_HERE] Predictions below 45% confidence trigger the retake warning


# ==============================================================================
# 5. FOUR DATA STRUCTURE STAGES (DSA) CONSTANTS
# ==============================================================================

# --- Stage 1: Fixed-Size 8-Element Feature Array (# [ARRAY]) ---
# # [CONFIG] # [ARRAY] Number of normalized visual features extracted by OpenCV
FEATURE_ARRAY_SIZE = 8
FEATURE_NAMES = [
    "Green Healthy Leaf Ratio",
    "Brown Necrotic Lesion Ratio",
    "Yellow Chlorosis Ratio",
    "Canny Edge Density",
    "Grayscale Texture Entropy",
    "Dark Spot Ratio",
    "HSV Saturation Spread",
    "Lab Colour Contrast",
]

# # [CONFIG] # [PREPROCESS] Resize dimension used by OpenCV when computing the 8 visual features
OPENCV_FEATURE_SIZE = 256  # # [CHANGE_HERE] Image size for OpenCV Stage 1 feature extraction

# --- Stage 2: 31-Bucket Hash Table (# [HASH_TABLE]) ---
# # [CONFIG] # [HASH_TABLE] Prime number of buckets and polynomial rolling hash base
HASH_TABLE_BUCKETS = 31  # # [CHANGE_HERE] Number of hash table buckets
HASH_POLYNOMIAL_BASE = 31  # # [CHANGE_HERE] Prime multiplier for polynomial rolling hash

# --- Stage 3: Decision Tree Severity Thresholds (# [DECISION_TREE] # [SEVERITY]) ---
# # [CONFIG] # [DECISION_TREE] # [SEVERITY] Affected area % cutoffs: <15 Low, 15-35 Moderate, >35 Severe
SEVERITY_LOW_MAX = 15.0  # # [CHANGE_HERE] Under 15% affected area = Low severity
SEVERITY_MODERATE_MAX = 35.0  # # [CHANGE_HERE] 15% to 35% = Moderate; over 35% = Severe

# --- Stage 4: Priority Queue Max-Heap Scoring Weights (# [PRIORITY_QUEUE] # [TREATMENT]) ---
# # [CONFIG] # [PRIORITY_QUEUE] # [TREATMENT] Composite score weights (sum to 1.0)
WEIGHT_EFFICACY = 0.50  # # [CHANGE_HERE] 50% weight for treatment efficacy %
WEIGHT_PHI_SAFETY = 0.30  # # [CHANGE_HERE] 30% weight for short Pre-Harvest Interval (PHI days)
WEIGHT_COST = 0.20  # # [CHANGE_HERE] 20% weight for lower cost per acre in INR

# # [CONFIG] # [PRIORITY_QUEUE] Normalization ceilings for PHI days and Cost in INR
MAX_PHI_DAYS_CAP = 30.0  # # [CHANGE_HERE] PHI days above 30 get 0 PHI safety score
MAX_COST_INR_CAP = 2500.0  # # [CHANGE_HERE] Cost/acre above Rs 2500 gets 0 cost score
