"""
================================================================================
FILE: train_yolo.py
PURPOSE:
    Trains a real Ultralytics YOLOv8 object detection model (`yolov8s.pt`) on
    the 20-class IP102 bounding-box dataset (`dataset_yolo/data.yaml`) prepared
    by `prepare_dataset.py`. Saves the best trained checkpoint directly to
    `models/yolo_pest.pt`.

CONCEPTS USED:
    - Real-Time Single-Stage Object Detection (# [YOLO])
    - Bounding-Box Regression & Non-Maximum Suppression (# [THRESHOLD])

HOW IT CONNECTS TO OTHER FILES:
    - Reads `YOLO_DATA_YAML_PATH`, `YOLO_BASE_MODEL`, `YOLO_EPOCHS`, and
      `YOLO_WEIGHTS_PATH` from `config.py`.
    - Saves `models/yolo_pest.pt`, which is loaded by `models_loader.py` for
      real leaf pest localization and affected area % computation.
================================================================================
"""

# Import standard library modules for file path checking and copying
import os
import shutil

# Import Ultralytics YOLO class for training YOLOv8
from ultralytics import YOLO

# Import central configuration constants
import config


# # [YOLO]
def train_yolov8_detector() -> None:
    """
    Fine-tunes `yolov8s.pt` on the IP102 20-class detection split using
    `dataset_yolo/data.yaml` and copies `best.pt` to `models/yolo_pest.pt`.
    """
    # Verify that data.yaml exists before starting training
    if not os.path.isfile(config.YOLO_DATA_YAML_PATH):
        print(
            f"[ERROR] YOLOv8 data.yaml not found at '{config.YOLO_DATA_YAML_PATH}'. "
            "Please run `python prepare_dataset.py` first!"
        )
        return

    print("=" * 78)
    print(
        f"[YOLOv8 TRAINING] Starting Ultralytics {config.YOLO_BASE_MODEL} training "
        f"for {config.YOLO_EPOCHS} epochs at {config.YOLO_IMAGE_SIZE}x{config.YOLO_IMAGE_SIZE}"
    )
    print("=" * 78)

    # Load pretrained YOLOv8s checkpoint (automatically downloads yolov8s.pt if not cached)
    yolo_model = YOLO(config.YOLO_BASE_MODEL)

    # Run training with data augmentation (mosaic, hsv jitter, horizontal flip, scaling)
    results = yolo_model.train(
        data=config.YOLO_DATA_YAML_PATH,
        epochs=config.YOLO_EPOCHS,
        imgsz=config.YOLO_IMAGE_SIZE,
        batch=config.YOLO_BATCH_SIZE,
        patience=7,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        project=os.path.join(config.BASE_DIR, "runs_yolo"),
        name="ip102_yolov8s",
        exist_ok=True,
    )

    # Locate the best.pt file saved by Ultralytics and copy it to models/yolo_pest.pt
    best_pt_candidate = os.path.join(
        config.BASE_DIR, "runs_yolo", "ip102_yolov8s", "weights", "best.pt"
    )
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    if os.path.isfile(best_pt_candidate):
        shutil.copy2(best_pt_candidate, config.YOLO_WEIGHTS_PATH)
        print(f"\n[SUCCESS] Best YOLOv8 weights copied to: {config.YOLO_WEIGHTS_PATH}")
    else:
        print(
            f"[WARNING] Could not find '{best_pt_candidate}'. Check Ultralytics output folder."
        )


if __name__ == "__main__":
    train_yolov8_detector()
