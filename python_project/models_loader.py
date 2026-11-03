"""
================================================================================
FILE: models_loader.py
PURPOSE:
    Loads the REAL trained PyTorch CNN (`models/cnn_pest.pt`), Ultralytics
    YOLOv8 (`models/yolo_pest.pt`), and class mapping (`models/class_map.json`)
    into memory once at startup or on demand.

    STRICT REAL-MODEL POLICY (NO FAKE OUTPUT):
    - Never generates random numbers or hardcoded confidences.
    - If `models/cnn_pest.pt` is missing, `cnn_model` is `None` and `status_message`
      explicitly reports: "Model not trained yet, run train_cnn.py".
    - If `models/yolo_pest.pt` is missing, `yolo_model` is `None` and `yolo_status`
      explicitly reports: "YOLO model not trained yet, run train_yolo.py".

CONCEPTS USED:
    - Singleton Model Registry (# [CNN], # [YOLO], # [CONFIG])
    - Safe Checkpoint Deserialization on CPU/GPU

HOW IT CONNECTS TO OTHER FILES:
    - Uses paths from `config.py` and `build_cnn_model()` from `train_cnn.py`.
    - Called by `inference.py` and `/api/health` in `app.py`.
================================================================================
"""

# Import standard library modules
import json
import os
from typing import Any, Dict, Optional

# Import PyTorch for loading CNN state_dict
import torch
import torch.nn as nn

# Import central configuration constants
import config
from train_cnn import build_cnn_model

# Try importing Ultralytics YOLO safely
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]


# # [CNN] # [YOLO]
class RealModelRegistry:
    """
    Holds the loaded PyTorch CNN model, Ultralytics YOLOv8 model, device, and
    class_map dictionary. Never fakes predictions if weight files are absent.
    """

    def __init__(self) -> None:
        # Select GPU if available, otherwise CPU
        self.device: torch.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        # Loaded PyTorch CNN module (or None if models/cnn_pest.pt does not exist yet)
        self.cnn_model: Optional[nn.Module] = None
        # Loaded Ultralytics YOLOv8 object (or None if models/yolo_pest.pt does not exist yet)
        self.yolo_model: Optional[Any] = None
        # Dictionary mapping string class index ("0".."19") -> pest metadata dict
        self.class_map: Dict[str, Dict[str, Any]] = {}
        # Human-readable status strings for UI banners and /api/health
        self.cnn_status: str = "Model not trained yet, run train_cnn.py"
        self.yolo_status: str = "YOLO model not trained yet, run train_yolo.py"

        # Load class_map.json and attempt to load trained model weights
        self.reload_all()

    def load_class_map(self) -> None:
        """Loads `models/class_map.json` mapping class indices 0..19 to IP102 pests."""
        if os.path.isfile(config.CLASS_MAP_PATH):
            with open(config.CLASS_MAP_PATH, "r", encoding="utf-8") as f:
                self.class_map = json.load(f)
        else:
            self.class_map = {}

    def load_cnn_weights(self) -> bool:
        """
        Loads `models/cnn_pest.pt` into `self.cnn_model` if the file exists.
        Returns True if loaded successfully, or False if weights are not trained yet.
        """
        if not os.path.isfile(config.CNN_WEIGHTS_PATH):
            self.cnn_model = None
            self.cnn_status = "Model not trained yet, run train_cnn.py"
            return False

        try:
            checkpoint = torch.load(config.CNN_WEIGHTS_PATH, map_location=self.device)
            # Support both wrapped dict checkpoints and raw state_dicts
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                num_classes = int(
                    checkpoint.get("num_classes", max(1, len(self.class_map) or config.TOP_N_CLASSES))
                )
                backbone_name = str(
                    checkpoint.get("backbone_name", config.CNN_BACKBONE_NAME)
                )
                state_dict = checkpoint["state_dict"]
            else:
                num_classes = max(1, len(self.class_map) or config.TOP_N_CLASSES)
                backbone_name = config.CNN_BACKBONE_NAME
                state_dict = checkpoint

            model = build_cnn_model(
                num_classes=num_classes,
                backbone_name=backbone_name,
                pretrained=False,
            ).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self.cnn_model = model
            self.cnn_status = f"Ready ({backbone_name}, {num_classes} classes on {self.device})"
            return True
        except Exception as exc:
            self.cnn_model = None
            self.cnn_status = f"Error loading CNN weights ({exc}); run train_cnn.py"
            return False

    def load_yolo_weights(self) -> bool:
        """
        Loads `models/yolo_pest.pt` into `self.yolo_model` if the file exists.
        Returns True if loaded successfully, or False if weights are not trained yet.
        """
        if YOLO is None:
            self.yolo_model = None
            self.yolo_status = "Ultralytics package not installed; run pip install -r requirements.txt"
            return False

        if not os.path.isfile(config.YOLO_WEIGHTS_PATH):
            self.yolo_model = None
            self.yolo_status = "YOLO model not trained yet, run train_yolo.py"
            return False

        try:
            self.yolo_model = YOLO(config.YOLO_WEIGHTS_PATH)
            self.yolo_status = f"Ready (YOLOv8 weights loaded from {os.path.basename(config.YOLO_WEIGHTS_PATH)})"
            return True
        except Exception as exc:
            self.yolo_model = None
            self.yolo_status = f"Error loading YOLO weights ({exc}); run train_yolo.py"
            return False

    def reload_all(self) -> Dict[str, Any]:
        """Reloads `class_map.json`, CNN weights, and YOLOv8 weights."""
        self.load_class_map()
        cnn_ok = self.load_cnn_weights()
        yolo_ok = self.load_yolo_weights()
        return {
            "cnn_ready": cnn_ok,
            "yolo_ready": yolo_ok,
            "cnn_status": self.cnn_status,
            "yolo_status": self.yolo_status,
            "num_classes_mapped": len(self.class_map),
            "device": str(self.device),
        }


# Instantiate global singleton model registry
MODEL_REGISTRY = RealModelRegistry()
