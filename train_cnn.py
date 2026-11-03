"""
================================================================================
FILE: train_cnn.py
PURPOSE:
    Trains the PyTorch `EfficientNet-B0` transfer-learning classifier on the
    `config.NUM_PESTS` (20) IP102 pest classes prepared by `prepare_data.py`.

    Tailored for a Windows laptop with an NVIDIA RTX 4050 (6 GB VRAM):
      - Checks `torch.cuda.is_available()` and prints GPU device name & VRAM.
      - Uses Automatic Mixed Precision (`torch.cuda.amp.autocast`) to halve VRAM.
      - Uses class-weighted `CrossEntropyLoss` to counter IP102 class imbalance.
      - Uses strong data augmentation (flip, rotation, color jitter, crop).
      - Uses small batch size (`config.CNN_BATCH_SIZE = 16`) and
        `config.DATALOADER_WORKERS = 2`.
      - Supports `--quick-test` (`config.QUICK_TEST_EPOCHS = 2` epochs) to verify
        the training loop quickly.
      - Saves the best validation checkpoint to `config.CNN_WEIGHTS_PATH`
        (`models/cnn_pest.pt`).

HOW IT CONNECTS TO OTHER FILES:
    - Reads dataset folders from `config.CNN_DATA_DIR` created by `prepare_data.py`.
    - Uses `build_efficientnet_b0` from `detect.py`.
    - Saves `models/cnn_pest.pt` loaded by `evaluate.py` and `detect.py`.
================================================================================
"""

# Import standard library modules for CLI arguments, paths, and timing
import argparse
from collections import Counter
import os
import sys
import time
from typing import List

# Import central configuration and model builder
import config
from detect import build_efficientnet_b0


# # [CNN] # [DATASET] # [PREPROCESS]
def train_cnn(quick_test: bool = False) -> None:
    """
    Runs 2-phase transfer learning on `EfficientNet-B0` and saves the best
    weights to `config.CNN_WEIGHTS_PATH` (`models/cnn_pest.pt`).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    train_dir = os.path.join(config.CNN_DATA_DIR, "train")
    val_dir = os.path.join(config.CNN_DATA_DIR, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(
            f"[ERROR] CNN dataset folders not found at: {config.CNN_DATA_DIR}\n"
            f"Run `python prepare_data.py` first!"
        )
        sys.exit(1)

    # Check CUDA GPU availability (designed for RTX 4050 6 GB VRAM)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Using CUDA device: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[CPU] CUDA not available; training on CPU.")

    # Strong data augmentation for training split (# [PREPROCESS])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.CNN_IMAGE_SIZE, scale=(0.75, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Compute inverse-frequency class weights for imbalanced IP102 classes
    label_counts = Counter(train_dataset.targets)
    total_samples = float(len(train_dataset))
    class_weights_list: List[float] = []
    for cls_idx in range(config.NUM_PESTS):
        cnt = float(max(1, label_counts.get(cls_idx, 1)))
        class_weights_list.append(total_samples / (config.NUM_PESTS * cnt))

    class_weights = torch.tensor(
        class_weights_list, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATALOADER_WORKERS,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER_WORKERS,
        pin_memory=use_cuda,
    )

    model = build_efficientnet_b0(
        num_classes=config.NUM_PESTS, pretrained=True
    ).to(device)

    total_epochs = (
        config.QUICK_TEST_EPOCHS if quick_test else config.CNN_EPOCHS
    )
    freeze_epochs = max(1, total_epochs // 3)

    # Phase 1: Freeze feature backbone for initial warm-up epochs
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CNN_LEARNING_RATE,
        weight_decay=config.CNN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    best_val_acc = 0.0

    print(
        f"Starting EfficientNet-B0 training for {total_epochs} epochs "
        f"({config.NUM_CROPS} crops, {config.NUM_PESTS} pests)..."
    )

    for epoch in range(1, total_epochs + 1):
        t0 = time.perf_counter()

        # Phase 2: Unfreeze entire backbone after warm-up epochs
        if epoch == freeze_epochs + 1:
            print("[Phase 2] Unfreezing EfficientNet-B0 backbone for fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.CNN_LEARNING_RATE * 0.2,
                weight_decay=config.CNN_WEIGHT_DECAY,
            )

        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item()) * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += int((preds == labels).sum().item())
            train_total += int(images.size(0))

        scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(images.size(0))

        train_acc = (train_correct / max(1, train_total)) * 100.0
        val_acc = (val_correct / max(1, val_total)) * 100.0
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] "
            f"Loss: {train_loss / max(1, train_total):.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": best_val_acc,
                    "num_classes": config.NUM_PESTS,
                },
                config.CNN_WEIGHTS_PATH,
            )
            print(
                f"  -> Saved new best CNN weights to {config.CNN_WEIGHTS_PATH} "
                f"(Val Acc: {best_val_acc:.2f}%)"
            )

    print(
        f"\nCNN training finished! Best validation accuracy: {best_val_acc:.2f}%\n"
        f"Next step: run `python evaluate.py` to evaluate on the test split and "
        f"record accuracy in the SQLite database."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PyTorch EfficientNet-B0 on IP102 shortlist."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help=f"Run {config.QUICK_TEST_EPOCHS} epochs to test the pipeline quickly.",
    )
    args = parser.parse_args()
    train_cnn(quick_test=args.quick_test)
