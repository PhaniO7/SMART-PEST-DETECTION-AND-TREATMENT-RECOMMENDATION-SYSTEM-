"""
================================================================================
FILE: detect.py
PURPOSE:
    Performs real Computer Vision inference using:
      1. PyTorch `EfficientNet-B0` (`models/cnn_pest.pt`) for top-3 softmax
         pest classification across `config.NUM_PESTS` (20) classes (`# [CNN]`).
      2. Ultralytics `YOLOv8n` (`models/yolo_pest.pt`) for pest bounding-box
         localization and affected leaf area % calculation (`# [YOLO]`).
      3. Fuses CNN + YOLO predictions, checks `config.CONFIDENCE_CUTOFF`
         (`"Uncertain, please retake the photo"`), and runs the 4-stage DSA
         pipeline (`dsa.run_dsa_pipeline`) while timing its exact execution
         speed in milliseconds with `time.perf_counter()`.

    IMPORTANT:
      - NO random numbers, NO hardcoded confidences, NO fake outputs, and NO
        LLM calls are used anywhere in this file.
      - If `models/cnn_pest.pt` is missing, returns
        `status_message = "Model not trained yet, run train_cnn.py"`.

HOW IT CONNECTS TO OTHER FILES:
    - Uses paths and thresholds from `config.py`.
    - Uses `PEST_DATABASE` and `get_pest_by_class_id` from `pest_data.py`.
    - Calls `run_dsa_pipeline` from `dsa.py`.
    - Called by `app.py` when a user scans a leaf image.
================================================================================
"""

# Import standard library modules for file checks and high-resolution timing
import os
import time
from typing import Any, Dict, List, Optional

# Import central configuration, pest database, and the 4-stage DSA pipeline
import config
import dsa
from pest_data import get_pest_by_class_id

# Optional imports so the project can still be imported/tested before pip install
try:
    import cv2
    import numpy as np
    from PIL import Image
    HAS_VISION_LIBS = True
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    Image = None  # type: ignore
    HAS_VISION_LIBS = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None  # type: ignore
    HAS_ULTRALYTICS = False


# ==============================================================================
# 1. PYTORCH EFFICIENTNET-B0 ARCHITECTURE BUILDER (# [CNN])
# ==============================================================================

# # [CNN]
def build_efficientnet_b0(
    num_classes: int = config.NUM_PESTS,
    pretrained: bool = False,
) -> Any:
    """
    Builds the PyTorch `EfficientNet-B0` transfer-learning model with its final
    classifier linear layer replaced to output `num_classes` (`config.NUM_PESTS` = 20)
    logits.
    """
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch / torchvision is not installed. Run: pip install -r requirements.txt"
        )

    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# # [PREPROCESS] # [CNN]
def preprocess_for_cnn(pil_image: Any) -> Any:
    """
    Resizes an RGB PIL Image to `config.CNN_IMAGE_SIZE` (224x224), converts to a
    PyTorch float tensor in [0, 1], and normalizes with `config.IMAGENET_MEAN`
    and `config.IMAGENET_STD`.
    Returns a 4D batch tensor of shape `(1, 3, 224, 224)`.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD,
            ),
        ]
    )
    rgb_img = pil_image.convert("RGB")
    return transform(rgb_img).unsqueeze(0)


# ==============================================================================
# 2. REAL CNN & YOLOv8 INFERENCE FUNCTIONS (# [CNN] # [YOLO])
# ==============================================================================

# # [CNN]
def run_cnn_inference(
    pil_image: Any,
    weights_path: str = config.CNN_WEIGHTS_PATH,
) -> Dict[str, Any]:
    """
    Loads trained weights from `models/cnn_pest.pt` and runs a forward pass
    to obtain softmax probabilities and the top-3 predicted pest classes.
    If weights are missing, returns `trained = False` with message
    `"Model not trained yet, run train_cnn.py"`.
    """
    if not os.path.isfile(weights_path):
        return {
            "trained": False,
            "message": "Model not trained yet, run train_cnn.py",
            "top_predictions": [],
        }

    if not HAS_TORCH:
        return {
            "trained": False,
            "message": "PyTorch not installed. Run: pip install -r requirements.txt",
            "top_predictions": [],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_b0(num_classes=config.NUM_PESTS, pretrained=False)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_tensor = preprocess_for_cnn(pil_image).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        k = min(config.CNN_TOP_K, config.NUM_PESTS)
        top_probs, top_indices = torch.topk(probs, k=k)

    top_predictions: List[Dict[str, Any]] = []
    for prob_val, cls_idx in zip(top_probs.tolist(), top_indices.tolist()):
        pest_rec = get_pest_by_class_id(int(cls_idx))
        top_predictions.append(
            {
                "class_id": int(cls_idx),
                "pest_name": pest_rec["common_name"],
                "scientific_name": pest_rec["scientific_name"],
                "crop": pest_rec["crop"],
                "confidence": round(float(prob_val), 4),
            }
        )

    return {
        "trained": True,
        "message": "CNN inference complete.",
        "top_predictions": top_predictions,
    }


# # [YOLO] # [SEVERITY]
def run_yolo_inference(
    image_bgr: Any,
    weights_path: str = config.YOLO_WEIGHTS_PATH,
) -> Dict[str, Any]:
    """
    Loads trained Ultralytics YOLOv8n weights from `models/yolo_pest.pt`,
    detects pest bounding boxes, draws boxes on an annotated image copy, and
    computes `affected_area_pct` = `(bounding box pixel area / total image area) * 100`.
    If weights are missing, returns `trained = False` with message
    `"Model not trained yet, run train_yolo.py"`.
    """
    if not os.path.isfile(weights_path):
        return {
            "trained": False,
            "message": "Model not trained yet, run train_yolo.py",
            "boxes": [],
            "affected_area_pct": 0.0,
            "max_confidence": 0.0,
            "annotated_bgr": image_bgr,
        }

    if not HAS_ULTRALYTICS or not HAS_VISION_LIBS:
        return {
            "trained": False,
            "message": "Ultralytics / OpenCV not installed. Run: pip install -r requirements.txt",
            "boxes": [],
            "affected_area_pct": 0.0,
            "max_confidence": 0.0,
            "annotated_bgr": image_bgr,
        }

    img_h, img_w = image_bgr.shape[:2]
    total_area = float(max(1, img_h * img_w))
    annotated = image_bgr.copy()
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    yolo_model = YOLO(weights_path)
    results = yolo_model.predict(
        source=image_bgr,
        conf=config.YOLO_CONF_THRESHOLD,
        iou=config.YOLO_IOU_THRESHOLD,
        imgsz=config.YOLO_IMAGE_SIZE,
        verbose=False,
    )

    boxes_out: List[Dict[str, Any]] = []
    max_conf = 0.0

    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            pest_info = get_pest_by_class_id(cls_id)

            x1 = max(0, min(img_w, int(xyxy[0])))
            y1 = max(0, min(img_h, int(xyxy[1])))
            x2 = max(0, min(img_w, int(xyxy[2])))
            y2 = max(0, min(img_h, int(xyxy[3])))

            # Mark union mask so overlapping boxes are not double-counted
            mask[y1:y2, x1:x2] = 1
            if conf > max_conf:
                max_conf = conf

            boxes_out.append(
                {
                    "class_id": cls_id,
                    "pest_name": pest_info["common_name"],
                    "crop": pest_info["crop"],
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

            # Draw bounding box and label on annotated image
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
            label = f"{pest_info['common_name']} ({conf * 100:.1f}%)"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (34, 197, 94),
                2,
            )

    affected_pixels = float(np.count_nonzero(mask))
    affected_area_pct = round((affected_pixels / total_area) * 100.0, 2)

    return {
        "trained": True,
        "message": "YOLO inference complete.",
        "boxes": boxes_out,
        "affected_area_pct": affected_area_pct,
        "max_confidence": round(max_conf, 4),
        "annotated_bgr": annotated,
    }


# ==============================================================================
# 3. FUSION OF CNN + YOLO + TIMED 4-STAGE DSA PIPELINE (# [CNN] # [YOLO] # [THRESHOLD])
# ==============================================================================

# # [CNN] # [YOLO] # [THRESHOLD]
def analyze_leaf_image(
    image_bgr: Any,
    pil_image: Optional[Any] = None,
    selected_crop: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the complete end-to-end pipeline on an uploaded leaf image:
      1. Runs `EfficientNet-B0` CNN inference (`models/cnn_pest.pt`).
      2. Runs `YOLOv8n` object detection (`models/yolo_pest.pt`).
      3. If `models/cnn_pest.pt` is missing, sets `models_trained = False` and
         `status_message = "Model not trained yet, run train_cnn.py"`.
      4. Fuses CNN + YOLO confidence and checks against `config.CONFIDENCE_CUTOFF`.
         If below cutoff, sets `uncertain = True` and
         `warning_message = "Uncertain, please retake the photo"`.
      5. Measures the exact execution time (`dsa_speed_ms`) of `dsa.run_dsa_pipeline`
         using `time.perf_counter()`.
    """
    cnn_res = (
        run_cnn_inference(pil_image)
        if pil_image is not None
        else {
            "trained": False,
            "message": "Model not trained yet, run train_cnn.py",
            "top_predictions": [],
        }
    )
    yolo_res = run_yolo_inference(image_bgr)

    # Check if the CNN model has been trained yet
    if not cnn_res["trained"]:
        # Still measure Stage 1 feature extraction speed with time.perf_counter()
        t0 = time.perf_counter()
        feature_array = dsa.extract_feature_array(image_bgr)
        dsa_speed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "models_trained": False,
            "status_message": "Model not trained yet, run train_cnn.py",
            "yolo_message": yolo_res["message"],
            "uncertain": False,
            "warning_message": "",
            "cnn_top3": [],
            "yolo_boxes": [],
            "affected_area_pct": 0.0,
            "dsa_speed_ms": dsa_speed_ms,
            "feature_array": feature_array,
            "dsa_result": None,
            "annotated_bgr": image_bgr,
        }

    # Fuse CNN top-1 prediction with YOLO detections
    top1 = cnn_res["top_predictions"][0]
    cnn_conf = float(top1["confidence"])
    yolo_conf = float(yolo_res["max_confidence"])
    yolo_boxes = yolo_res["boxes"]

    # If YOLO also detected the same class, combine confidence; else use CNN confidence
    if yolo_res["trained"] and yolo_boxes:
        matching_yolo = [
            b for b in yolo_boxes if int(b["class_id"]) == int(top1["class_id"])
        ]
        if matching_yolo:
            fused_conf = round(0.65 * cnn_conf + 0.35 * yolo_conf, 4)
        else:
            fused_conf = round(0.75 * cnn_conf + 0.25 * yolo_conf, 4)
        affected_area_pct = float(yolo_res["affected_area_pct"])
    else:
        fused_conf = cnn_conf
        affected_area_pct = 0.0

    # Check confidence cutoff (# [THRESHOLD])
    uncertain = fused_conf < config.CONFIDENCE_CUTOFF
    warning_message = (
        "Uncertain, please retake the photo" if uncertain else ""
    )

    # Time the 4-stage DSA pipeline using time.perf_counter()
    crop_for_lookup = top1["crop"]
    pest_for_lookup = top1["pest_name"]

    t_start = time.perf_counter()
    dsa_result = dsa.run_dsa_pipeline(
        image_bgr=image_bgr,
        crop=crop_for_lookup,
        pest_name=pest_for_lookup,
        affected_area_pct=affected_area_pct,
    )
    dsa_speed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    # Check if user selected a different crop in the dropdown than the detected pest's host crop
    crop_mismatch_note = ""
    if (
        selected_crop
        and selected_crop.strip().lower() != crop_for_lookup.strip().lower()
    ):
        crop_mismatch_note = (
            f"Note: Selected crop was '{selected_crop}', while the highest-scoring "
            f"IP102 class belongs to host crop '{crop_for_lookup}'."
        )

    return {
        "models_trained": True,
        "status_message": "Inference complete.",
        "yolo_message": yolo_res["message"],
        "uncertain": uncertain,
        "warning_message": warning_message,
        "crop_mismatch_note": crop_mismatch_note,
        "predicted_class_id": top1["class_id"],
        "predicted_pest": top1["pest_name"],
        "scientific_name": top1["scientific_name"],
        "predicted_crop": crop_for_lookup,
        "cnn_confidence": cnn_conf,
        "yolo_confidence": yolo_conf,
        "fused_confidence": fused_conf,
        "cnn_top3": cnn_res["top_predictions"],
        "yolo_boxes": yolo_boxes,
        "affected_area_pct": affected_area_pct,
        "dsa_speed_ms": dsa_speed_ms,
        "feature_array": dsa_result["feature_array"],
        "dsa_result": dsa_result,
        "annotated_bgr": yolo_res["annotated_bgr"],
    }
