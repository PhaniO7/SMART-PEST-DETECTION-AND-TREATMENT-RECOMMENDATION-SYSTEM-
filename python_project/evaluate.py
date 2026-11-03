"""
================================================================================
FILE: evaluate.py
PURPOSE:
    Evaluates our trained PyTorch CNN (`models/cnn_pest.pt`) and Ultralytics
    YOLOv8 (`models/yolo_pest.pt`) models on the held-out IP102 test split:
      1. Runs the CNN on `dataset_cnn/test`, computes per-class Precision,
         Recall, F1-score, and Accuracy, saves `models/classification_report.csv`,
         and plots a high-resolution `models/confusion_matrix.png`.
      2. Runs YOLOv8 validation on the test split in `dataset_yolo/data.yaml`
         and prints `mAP50` and `mAP50-95`.

CONCEPTS USED:
    - Confusion Matrix & Classification Report (# [CNN], # [DATASET])
    - Mean Average Precision mAP@0.50 and mAP@0.50:0.95 (# [YOLO])

HOW IT CONNECTS TO OTHER FILES:
    - Uses paths from `config.py` and `build_cnn_model` from `train_cnn.py`.
    - Saves evaluation artifacts (`confusion_matrix.png`, `classification_report.csv`)
      inside `models/`.
================================================================================
"""

# Import standard library modules
import json
import os
from typing import List

# Import Matplotlib, NumPy, Pandas, and scikit-learn for metrics and plotting
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend so PNG saves work on headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Import PyTorch and Torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import Ultralytics YOLO for mAP evaluation
from ultralytics import YOLO

# Import central configuration and CNN builder
import config
from train_cnn import build_cnn_model


# # [CNN]
def evaluate_cnn_on_test_set() -> None:
    """
    Evaluates `models/cnn_pest.pt` on `dataset_cnn/test`, saving:
      - `models/classification_report.csv`
      - `models/confusion_matrix.png`
    """
    test_dir = os.path.join(config.CNN_DATASET_DIR, "test")
    if not os.path.isfile(config.CNN_WEIGHTS_PATH) or not os.path.isdir(test_dir):
        print(
            "[SKIP CNN EVAL] Either `models/cnn_pest.pt` or `dataset_cnn/test` is missing. "
            "Run `prepare_dataset.py` and `train_cnn.py` first."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names from models/class_map.json
    class_labels: List[str] = []
    if os.path.isfile(config.CLASS_MAP_PATH):
        with open(config.CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            cmap = json.load(f)
            for idx in range(len(cmap)):
                entry = cmap.get(str(idx), {})
                class_labels.append(entry.get("pest_name", f"class_{idx}"))

    test_transform = transforms.Compose(
        [
            transforms.Resize((config.CNN_IMAGE_SIZE + 32, config.CNN_IMAGE_SIZE + 32)),
            transforms.CenterCrop(config.CNN_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=False, num_workers=2
    )

    checkpoint = torch.load(config.CNN_WEIGHTS_PATH, map_location=device)
    num_classes = checkpoint.get("num_classes", len(test_dataset.classes))
    backbone_name = checkpoint.get("backbone_name", config.CNN_BACKBONE_NAME)

    model = build_cnn_model(
        num_classes=num_classes, backbone_name=backbone_name, pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    target_names = (
        class_labels[:num_classes]
        if len(class_labels) >= num_classes
        else [f"class_{i}" for i in range(num_classes)]
    )

    # 1. Generate and save Classification Report CSV
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(config.MODELS_DIR, "classification_report.csv")
    report_df.to_csv(csv_path, index=True)
    print(f"[SAVED] CNN Classification Report CSV -> {csv_path}")
    print(report_df.round(3))

    # 2. Generate and save Confusion Matrix PNG
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.colorbar(im, ax=ax)
    ax.set_title("IP102 Top-20 Pest Classification Confusion Matrix (CNN)")
    ax.set_xlabel("Predicted Pest Class")
    ax.set_ylabel("True Pest Class")
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(target_names, fontsize=8)
    plt.tight_layout()

    cm_png_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] CNN Confusion Matrix PNG -> {cm_png_path}")


# # [YOLO]
def evaluate_yolo_on_test_set() -> None:
    """
    Evaluates `models/yolo_pest.pt` on the IP102 detection test split and prints
    `mAP50` and `mAP50-95`.
    """
    if not os.path.isfile(config.YOLO_WEIGHTS_PATH) or not os.path.isfile(
        config.YOLO_DATA_YAML_PATH
    ):
        print(
            "[SKIP YOLO EVAL] Either `models/yolo_pest.pt` or `dataset_yolo/data.yaml` is missing. "
            "Run `prepare_dataset.py` and `train_yolo.py` first."
        )
        return

    yolo_model = YOLO(config.YOLO_WEIGHTS_PATH)
    metrics = yolo_model.val(data=config.YOLO_DATA_YAML_PATH, split="test")

    map50 = float(metrics.box.map50) * 100.0
    map50_95 = float(metrics.box.map) * 100.0

    print("\n" + "=" * 78)
    print("[YOLOv8 TEST SET EVALUATION METRICS]")
    print(f"  mAP@0.50       : {map50:.2f}%")
    print(f"  mAP@0.50:0.95  : {map50_95:.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    evaluate_cnn_on_test_set()
    evaluate_yolo_on_test_set()
