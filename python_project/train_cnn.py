"""
================================================================================
FILE: train_cnn.py
PURPOSE:
    Trains a real PyTorch Convolutional Neural Network (CNN) using Transfer
    Learning (EfficientNet-B0 or ResNet50) on the 20-class IP102 dataset prepared
    by `prepare_dataset.py`. Designed to run on Google Colab (free T4 GPU) or a
    local CUDA/MPS/CPU machine.

    KEY TRAINING FEATURES:
      1. Handles IP102 Class Imbalance using BOTH inverse-frequency class weights
         in `nn.CrossEntropyLoss` AND a `WeightedRandomSampler` on the training set.
      2. Strong Data Augmentation: RandomResizedCrop, RandomHorizontalFlip,
         RandomVerticalFlip, RandomRotation(25°), and ColorJitter.
      3. Two-Phase Transfer Learning:
           - Phase 1: Freeze convolutional backbone; train classifier head only.
           - Phase 2: Unfreeze all layers; fine-tune full network at a lower LR.
      4. ReduceLROnPlateau learning rate scheduler + Early Stopping based on
         validation Macro F1-score.
      5. Saves best weights to `models/cnn_pest.pt` and prints Accuracy,
         Macro Precision, Macro Recall, and Macro F1-score.

CONCEPTS USED:
    - Transfer Learning & 2-Phase Fine-Tuning (# [CNN], # [PREPROCESS])
    - Class-Imbalance Weighting (# [DATASET])

HOW IT CONNECTS TO OTHER FILES:
    - Reads dataset paths and hyperparameters from `config.py`.
    - Saves `models/cnn_pest.pt`, which is loaded by `models_loader.py` and
      evaluated by `evaluate.py`.
================================================================================
"""

# Import standard library modules
import copy
import json
import os
from typing import Dict, List, Tuple

# Import NumPy and scikit-learn metrics for Precision, Recall, F1, and Accuracy
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import PyTorch and Torchvision for deep learning model, transforms, and loaders
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# Import our central configuration constants
import config


# ==============================================================================
# 1. BUILD CNN ARCHITECTURE (EFFICIENTNET-B0 OR RESNET50)
# ==============================================================================

# # [CNN]
def build_cnn_model(
    num_classes: int,
    backbone_name: str = config.CNN_BACKBONE_NAME,
    pretrained: bool = True,
) -> nn.Module:
    """
    Constructs a Transfer Learning CNN (`efficientnet_b0` or `resnet50`) and
    replaces the final classification layer with a custom Dropout + Linear head
    matching `num_classes` (default 20).
    """
    # # [CNN] # [CHANGE_HERE] Switch between 'efficientnet_b0' and 'resnet50' in config.py
    if backbone_name.lower() == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        # Replace final fully connected layer with Dropout + Linear(in_features -> num_classes)
        net.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        # Replace EfficientNet classifier head with Dropout + Linear(in_features -> num_classes)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )

    return net


# # [CNN]
def set_backbone_trainable(
    model: nn.Module, backbone_name: str, trainable: bool
) -> None:
    """
    Freezes (`trainable=False` for Phase 1) or unfreezes (`trainable=True` for Phase 2)
    the convolutional feature extractor while keeping the classifier head trainable.
    """
    # First set requires_grad on all parameters
    for param in model.parameters():
        param.requires_grad = trainable

    # Always keep the final classifier head trainable in both Phase 1 and Phase 2
    head = model.fc if backbone_name.lower() == "resnet50" else model.classifier
    for param in head.parameters():
        param.requires_grad = True


# ==============================================================================
# 2. DATA AUGMENTATION & CLASS-IMBALANCE WEIGHTED DATALOADERS
# ==============================================================================

# # [DATASET] # [PREPROCESS]
def create_dataloaders(
    dataset_dir: str, batch_size: int = config.CNN_BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, torch.Tensor, int]:
    """
    Builds PyTorch DataLoaders for `train` and `val` splits with strong data
    augmentation and inverse-frequency class weighting to handle IP102 imbalance.
    """
    # Strong augmentation pipeline for the training split
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(config.CNN_IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    # Deterministic resize + center crop for validation split
    val_transform = transforms.Compose(
        [
            transforms.Resize((config.CNN_IMAGE_SIZE + 32, config.CNN_IMAGE_SIZE + 32)),
            transforms.CenterCrop(config.CNN_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)

    # Compute per-class sample counts to handle IP102 class imbalance
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)

    # Inverse-frequency class weights normalized so their mean is 1.0
    inv_weights = 1.0 / class_counts
    norm_class_weights = inv_weights / np.mean(inv_weights)

    # Per-sample weights for WeightedRandomSampler
    sample_weights = [norm_class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    class_weights_tensor = torch.tensor(norm_class_weights, dtype=torch.float32)
    return train_loader, val_loader, class_weights_tensor, num_classes


# ==============================================================================
# 3. ONE EPOCH TRAIN & VALIDATION FUNCTIONS
# ==============================================================================

# # [CNN]
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool = True,
) -> Dict[str, float]:
    """Runs one training or validation epoch and returns Loss, Accuracy, P, R, F1."""
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())

    total_samples = max(1, len(all_targets))
    epoch_loss = running_loss / total_samples
    acc = float(accuracy_score(all_targets, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return {
        "loss": round(epoch_loss, 4),
        "accuracy": round(acc * 100.0, 2),
        "precision": round(float(precision) * 100.0, 2),
        "recall": round(float(recall) * 100.0, 2),
        "f1": round(float(f1) * 100.0, 2),
    }


# ==============================================================================
# 4. TWO-PHASE TRAINING PIPELINE WITH EARLY STOPPING
# ==============================================================================

# # [CNN]
def train_two_phase_cnn() -> None:
    """
    Executes Phase 1 (frozen backbone) followed by Phase 2 (full fine-tuning),
    saving the best checkpoint to `models/cnn_pest.pt`.
    """
    if not os.path.isdir(os.path.join(config.CNN_DATASET_DIR, "train")):
        print(
            f"[ERROR] CNN dataset folder '{config.CNN_DATASET_DIR}/train' not found. "
            "Please run `python prepare_dataset.py` first!"
        )
        return

    # Select GPU (CUDA or Apple MPS) if available, otherwise CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[CNN TRAINING] Using compute device: {device}")

    train_loader, val_loader, class_weights, num_classes = create_dataloaders(
        config.CNN_DATASET_DIR, batch_size=config.CNN_BATCH_SIZE
    )

    model = build_cnn_model(
        num_classes=num_classes,
        backbone_name=config.CNN_BACKBONE_NAME,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    best_weights = copy.deepcopy(model.state_dict())

    # --------------------------------------------------------------------------
    # PHASE 1: FREEZE BACKBONE, TRAIN CLASSIFIER HEAD ONLY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(
        f"PHASE 1: Training Classifier Head Only ({config.CNN_EPOCHS_PHASE1_FROZEN} Epochs, "
        f"LR={config.CNN_LR_PHASE1})"
    )
    print("=" * 78)

    set_backbone_trainable(model, config.CNN_BACKBONE_NAME, trainable=False)
    optimizer_p1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CNN_LR_PHASE1,
        weight_decay=config.CNN_WEIGHT_DECAY,
    )

    for epoch in range(1, config.CNN_EPOCHS_PHASE1_FROZEN + 1):
        tr = run_one_epoch(model, train_loader, criterion, optimizer_p1, device, True)
        va = run_one_epoch(model, val_loader, criterion, optimizer_p1, device, False)
        print(
            f"[Phase 1 | Epoch {epoch:02d}/{config.CNN_EPOCHS_PHASE1_FROZEN:02d}] "
            f"Train Loss: {tr['loss']:.4f} Acc: {tr['accuracy']:.2f}% | "
            f"Val Loss: {va['loss']:.4f} Acc: {va['accuracy']:.2f}% F1: {va['f1']:.2f}%"
        )
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_metrics = va
            best_weights = copy.deepcopy(model.state_dict())

    # --------------------------------------------------------------------------
    # PHASE 2: UNFREEZE ALL LAYERS, FINE-TUNE ENTIRE NETWORK WITH LR SCHEDULER
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(
        f"PHASE 2: Fine-Tuning Full CNN ({config.CNN_EPOCHS_PHASE2_FINETUNE} Epochs, "
        f"LR={config.CNN_LR_PHASE2})"
    )
    print("=" * 78)

    set_backbone_trainable(model, config.CNN_BACKBONE_NAME, trainable=True)
    optimizer_p2 = AdamW(
        model.parameters(),
        lr=config.CNN_LR_PHASE2,
        weight_decay=config.CNN_WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer_p2, mode="max", factor=0.5, patience=2
    )

    epochs_without_improvement = 0

    for epoch in range(1, config.CNN_EPOCHS_PHASE2_FINETUNE + 1):
        tr = run_one_epoch(model, train_loader, criterion, optimizer_p2, device, True)
        va = run_one_epoch(model, val_loader, criterion, optimizer_p2, device, False)
        scheduler.step(va["f1"])

        print(
            f"[Phase 2 | Epoch {epoch:02d}/{config.CNN_EPOCHS_PHASE2_FINETUNE:02d}] "
            f"Train Loss: {tr['loss']:.4f} Acc: {tr['accuracy']:.2f}% | "
            f"Val Loss: {va['loss']:.4f} Acc: {va['accuracy']:.2f}% "
            f"P: {va['precision']:.2f}% R: {va['recall']:.2f}% F1: {va['f1']:.2f}%"
        )

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_metrics = va
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            # Save best checkpoint immediately to models/cnn_pest.pt
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            torch.save(
                {
                    "backbone_name": config.CNN_BACKBONE_NAME,
                    "num_classes": num_classes,
                    "state_dict": best_weights,
                    "best_val_metrics": best_metrics,
                },
                config.CNN_WEIGHTS_PATH,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"[EARLY STOPPING] No validation F1 improvement for "
                    f"{config.EARLY_STOPPING_PATIENCE} epochs. Stopping early."
                )
                break

    print("\n" + "=" * 78)
    print(f"[TRAINING COMPLETE] Best CNN Weights Saved To: {config.CNN_WEIGHTS_PATH}")
    print(
        f"Best Validation Metrics -> Accuracy: {best_metrics.get('accuracy', 0.0):.2f}% | "
        f"Precision: {best_metrics.get('precision', 0.0):.2f}% | "
        f"Recall: {best_metrics.get('recall', 0.0):.2f}% | "
        f"Macro F1: {best_metrics.get('f1', 0.0):.2f}%"
    )
    print("=" * 78)


if __name__ == "__main__":
    train_two_phase_cnn()
