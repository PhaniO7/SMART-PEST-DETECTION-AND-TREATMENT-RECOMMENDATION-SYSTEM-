"""
================================================================================
FILE: evaluate.py
PURPOSE:
    Evaluates the trained CNN (`models/cnn_pest.pt`) and YOLOv8n
    (`models/yolo_pest.pt`) models on the real IP102 test split:
      1. Computes real test-set accuracy (%), macro F1 (%), and per-class
         precision, recall, and F1-score using `scikit-learn`.
      2. Saves the confusion matrix plot PNG to `config.CONFUSION_MATRIX_PATH`
         (`models/confusion_matrix.png`).
      3. Evaluates YOLOv8n (`models/yolo_pest.pt`) to obtain real `mAP50` (%).
      4. Saves the real measured results (`cnn_test_accuracy`, `cnn_macro_f1`,
         `yolo_map50`, and evaluation date) to the SQLite database via
         `database.save_evaluation_metrics(...)` so `app.py` displays the real
         accuracy on the website instead of any hardcoded number.

HOW IT CONNECTS TO OTHER FILES:
    - Loads `config.CNN_WEIGHTS_PATH` and `config.YOLO_WEIGHTS_PATH`.
    - Saves evaluation metrics to SQLite (`database.py`) read by `app.py`.
================================================================================
"""

# Import standard library modules
import os
import sys
from typing import List, Optional

# Import central configuration, SQLite database helper, and CNN builder
import config
import database
from detect import build_efficientnet_b0
from pest_data import get_pest_by_class_id


# # [CNN] # [YOLO] # [DATABASE]
def evaluate_models() -> None:
    """
    Runs test-set evaluation for CNN and YOLOv8n, saves the confusion matrix PNG,
    prints per-class precision/recall/F1, and writes the real evaluation results
    into the SQLite database (`evaluation_metrics` table).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    database.init_db()

    if not os.path.isfile(config.CNN_WEIGHTS_PATH):
        print(
            f"[ERROR] Trained CNN weights not found at: {config.CNN_WEIGHTS_PATH}\n"
            f"Run `python train_cnn.py` first!"
        )
        sys.exit(1)

    test_dir = os.path.join(config.CNN_DATA_DIR, "test")
    if not os.path.isdir(test_dir):
        print(
            f"[ERROR] CNN test split directory not found at: {test_dir}\n"
            f"Run `python prepare_data.py` first!"
        )
        sys.exit(1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_transform = transforms.Compose(
        [
            transforms.Resize((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD
            ),
        ]
    )

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER_WORKERS,
    )

    model = build_efficientnet_b0(
        num_classes=config.NUM_PESTS, pretrained=False
    )
    checkpoint = torch.load(
        config.CNN_WEIGHTS_PATH, map_location=device, weights_only=False
    )
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cnn_acc_pct = float(accuracy_score(all_labels, all_preds)) * 100.0
    cnn_f1_pct = (
        float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        * 100.0
    )

    target_names = [
        f"{i:02d}_{get_pest_by_class_id(i)['common_name']}"
        for i in range(config.NUM_PESTS)
    ]
    print("\n" + "=" * 78)
    print(
        f"CNN TEST-SET CLASSIFICATION REPORT ({config.NUM_CROPS} crops, {config.NUM_PESTS} pests)"
    )
    print("=" * 78)
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(config.NUM_PESTS)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print(f"Real CNN Test Accuracy : {cnn_acc_pct:.2f}%")
    print(f"Real CNN Macro F1      : {cnn_f1_pct:.2f}%")

    # Save Confusion Matrix PNG
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    cm = confusion_matrix(
        all_labels, all_preds, labels=list(range(config.NUM_PESTS))
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_title(
        f"IP102 Confusion Matrix ({config.NUM_CROPS} Crops, {config.NUM_PESTS} Pests)"
    )
    ax.set_xlabel("Predicted Class ID")
    ax.set_ylabel("True Class ID")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix PNG to: {config.CONFUSION_MATRIX_PATH}")

    # Evaluate YOLOv8n mAP50 if weights exist
    yolo_map50_pct: Optional[float] = None
    if os.path.isfile(config.YOLO_WEIGHTS_PATH) and os.path.isfile(
        config.YOLO_DATA_YAML
    ):
        from ultralytics import YOLO

        yolo_model = YOLO(config.YOLO_WEIGHTS_PATH)
        val_metrics = yolo_model.val(
            data=config.YOLO_DATA_YAML,
            imgsz=config.YOLO_IMAGE_SIZE,
            batch=config.YOLO_BATCH_SIZE,
            verbose=False,
        )
        if hasattr(val_metrics, "box") and hasattr(val_metrics.box, "map50"):
            yolo_map50_pct = float(val_metrics.box.map50) * 100.0
            print(f"Real YOLOv8n mAP@50    : {yolo_map50_pct:.2f}%")

    # Save real evaluation metrics to the SQLite database for the website to display
    metric_id = database.save_evaluation_metrics(
        cnn_test_accuracy=cnn_acc_pct,
        cnn_macro_f1=cnn_f1_pct,
        yolo_map50=yolo_map50_pct,
    )
    print(
        f"\nSaved real evaluation metrics (record #{metric_id}) to SQLite database "
        f"({config.DATABASE_PATH}). The Streamlit website will now display these "
        f"real numbers!"
    )


if __name__ == "__main__":
    evaluate_models()
