"""
================================================================================
FILE: inference.py
PURPOSE:
    Executes the complete end-to-end Computer Vision, Deep Learning (CNN + YOLOv8),
    and 4-Stage Data Structures & Algorithms (DSA) inference pipeline on a
    real uploaded leaf/pest photograph:
      1. Preprocessing (# [PREPROCESS]): Loads the image with OpenCV & Pillow,
         validates dimensions, and normalizes tensors for PyTorch.
      2. YOLOv8 Localization (# [YOLO]): Detects pest bounding boxes, draws green
         boxes with confidence labels onto an annotated output image, and computes
         `affected_area_percent = (union_box_area / image_area) * 100.0`.
      3. CNN Classification (# [CNN]): Runs a forward pass through the trained
         EfficientNet-B0/ResNet50 model, applies Softmax, and extracts Top-3
         predictions with their exact probabilities.
      4. Prediction Fusion: If CNN and YOLOv8 disagree on the top class, selects
         the higher-confidence prediction as primary and returns BOTH predictions
         so the user sees complete transparency.
      5. Confidence Cutoff (# [THRESHOLD]): If the winning confidence is below
         `config.CONFIDENCE_CUTOFF`, flags `is_uncertain = True` and returns
         "Uncertain, please retake the photo" rather than a wrong diagnosis.
      6. 4-Stage DSA Pipeline: Runs Stage 1 (8-Element Feature Array), Stage 2
         (31-Bucket Hash Table O(1) Lookup), Stage 3 (Decision Tree Severity),
         and Stage 4 (Manual Max-Heap Priority Queue Treatment Ranking).

CONCEPTS USED:
    - Softmax Top-K Classification & Bounding-Box Area Ratio (# [CNN], # [YOLO])
    - Multi-Model Decision Fusion & Confidence Thresholding (# [THRESHOLD])
    - End-to-End Integration with 4 DSA Stages (# [ARRAY], # [HASH_TABLE],
      # [DECISION_TREE], # [PRIORITY_QUEUE])

HOW IT CONNECTS TO OTHER FILES:
    - Uses `MODEL_REGISTRY` from `models_loader.py`.
    - Uses `run_four_stage_dsa_pipeline()` from `dsa_pipeline.py`.
    - Uses `GLOBAL_PEST_HASH_TABLE` from `data/pest_database.py`.
    - Called by `app.py` when a user uploads a leaf image on the Scanner page.
================================================================================
"""

# Import standard library modules
import os
from typing import Any, Dict, List, Optional, Tuple

# Import OpenCV, NumPy, and Pillow for image preprocessing and bounding-box drawing
import cv2
import numpy as np
from PIL import Image

# Import PyTorch and Torchvision transforms for CNN tensor preprocessing
import torch
from torchvision import transforms

# Import our central project configuration, DSA pipeline, pest hash table, and model loader
import config
from data.pest_database import GLOBAL_PEST_HASH_TABLE, IP102_TOP20_PEST_DATABASE
from dsa_pipeline import extract_feature_array, run_four_stage_dsa_pipeline
from models_loader import MODEL_REGISTRY


# ==============================================================================
# 1. IMAGE PREPROCESSING FOR PYTORCH CNN
# ==============================================================================

# # [PREPROCESS] # [CNN]
CNN_INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((config.CNN_IMAGE_SIZE + 32, config.CNN_IMAGE_SIZE + 32)),
        transforms.CenterCrop(config.CNN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)


def resolve_class_metadata(class_index: int) -> Dict[str, Any]:
    """
    Looks up `class_index` (0..19) in `MODEL_REGISTRY.class_map` or falls back
    to `IP102_TOP20_PEST_DATABASE`.
    """
    str_idx = str(class_index)
    if str_idx in MODEL_REGISTRY.class_map:
        entry = MODEL_REGISTRY.class_map[str_idx]
        return {
            "class_index": class_index,
            "ip102_id": entry.get("ip102_id", class_index + 1),
            "pest_name": entry.get("pest_name", f"class_{class_index}"),
            "display_name": entry.get("display_name", entry.get("pest_name", "").title()),
            "crop": entry.get("crop", "Rice"),
        }

    # Fallback lookup in IP102_TOP20_PEST_DATABASE list
    for item in IP102_TOP20_PEST_DATABASE:
        if item["class_index"] == class_index:
            return {
                "class_index": class_index,
                "ip102_id": item["ip102_id"],
                "pest_name": item["common_name"],
                "display_name": item["display_name"],
                "crop": item["crop"],
            }

    return {
        "class_index": class_index,
        "ip102_id": class_index + 1,
        "pest_name": f"class_{class_index}",
        "display_name": f"Pest Class #{class_index}",
        "crop": "Unknown",
    }


# ==============================================================================
# 2. REAL PYTORCH CNN TOP-3 SOFTMAX CLASSIFICATION
# ==============================================================================

# # [CNN]
def run_cnn_classification(pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Runs the loaded PyTorch CNN (`MODEL_REGISTRY.cnn_model`) on `pil_image`,
    applies `torch.softmax`, and returns the Top-3 predictions sorted by
    descending probability.
    """
    if MODEL_REGISTRY.cnn_model is None:
        return []

    # Transform PIL RGB image to normalized tensor of shape (1, 3, 224, 224)
    input_tensor = CNN_INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(MODEL_REGISTRY.device)

    with torch.no_grad():
        logits = MODEL_REGISTRY.cnn_model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = min(config.CNN_TOP_K, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=top_k)

    top_predictions: List[Dict[str, Any]] = []
    for prob_val, idx_val in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        meta = resolve_class_metadata(int(idx_val))
        top_predictions.append(
            {
                "class_index": int(idx_val),
                "ip102_id": meta["ip102_id"],
                "pest_name": meta["pest_name"],
                "display_name": meta["display_name"],
                "crop": meta["crop"],
                "confidence": round(float(prob_val), 4),
                "confidence_percent": round(float(prob_val) * 100.0, 2),
            }
        )

    return top_predictions


# ==============================================================================
# 3. REAL ULTRALYTICS YOLOv8 LOCALIZATION & AFFECTED LEAF AREA % CALCULATION
# ==============================================================================

# # [YOLO]
def run_yolo_detection_and_draw_boxes(
    image_bgr: np.ndarray, annotated_save_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Runs the loaded Ultralytics YOLOv8 model (`MODEL_REGISTRY.yolo_model`) on
    `image_bgr`, draws bounding boxes onto an annotated copy (saved to
    `annotated_save_path`), and calculates `affected_area_percent` from the
    union of detected bounding-box pixels divided by total image pixels.
    """
    if MODEL_REGISTRY.yolo_model is None or image_bgr is None or image_bgr.size == 0:
        return [], 0.0

    img_h, img_w = image_bgr.shape[:2]
    total_image_area = float(max(1, img_h * img_w))

    # Run YOLOv8 inference using configured confidence and IoU thresholds
    results = MODEL_REGISTRY.yolo_model.predict(
        source=image_bgr,
        imgsz=config.YOLO_IMAGE_SIZE,
        conf=config.YOLO_BOX_CONF_THRESHOLD,
        iou=config.YOLO_IOU_THRESHOLD,
        verbose=False,
    )

    boxes_list: List[Dict[str, Any]] = []
    # Create a binary pixel mask to compute exact union area of overlapping boxes
    damage_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    annotated_bgr = image_bgr.copy()

    if results and len(results) > 0:
        det = results[0]
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy_arr = det.boxes.xyxy.cpu().numpy()
            conf_arr = det.boxes.conf.cpu().numpy()
            cls_arr = det.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf_val, cls_idx in zip(xyxy_arr, conf_arr, cls_arr):
                ix1 = max(0, min(img_w - 1, int(round(x1))))
                iy1 = max(0, min(img_h - 1, int(round(y1))))
                ix2 = max(0, min(img_w, int(round(x2))))
                iy2 = max(0, min(img_h, int(round(y2))))

                # Mark box region in binary mask so overlapping boxes are not double-counted
                damage_mask[iy1:iy2, ix1:ix2] = 1

                box_area = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
                box_area_pct = round((box_area / total_image_area) * 100.0, 2)
                meta = resolve_class_metadata(int(cls_idx))

                boxes_list.append(
                    {
                        "class_index": int(cls_idx),
                        "pest_name": meta["pest_name"],
                        "display_name": meta["display_name"],
                        "crop": meta["crop"],
                        "confidence": round(float(conf_val), 4),
                        "confidence_percent": round(float(conf_val) * 100.0, 2),
                        "xmin": ix1,
                        "ymin": iy1,
                        "xmax": ix2,
                        "ymax": iy2,
                        "box_area_percent": box_area_pct,
                    }
                )

                # Draw vivid green bounding box and label onto annotated_bgr
                cv2.rectangle(annotated_bgr, (ix1, iy1), (ix2, iy2), (34, 197, 94), 2)
                label_text = f"{meta['display_name']} ({conf_val * 100:.1f}%)"
                cv2.putText(
                    annotated_bgr,
                    label_text,
                    (ix1, max(18, iy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (34, 197, 94),
                    2,
                )

    # Sort detected boxes by confidence descending
    boxes_list.sort(key=lambda b: b["confidence"], reverse=True)

    # Calculate total affected leaf area percentage from union mask
    affected_area_percent = round(
        (float(np.sum(damage_mask)) / total_image_area) * 100.0, 2
    )

    # Save annotated image if requested
    if annotated_save_path:
        os.makedirs(os.path.dirname(annotated_save_path), exist_ok=True)
        cv2.imwrite(annotated_save_path, annotated_bgr)

    return boxes_list, affected_area_percent


# ==============================================================================
# 4. PREDICTION FUSION (CNN + YOLOv8) & CONFIDENCE CUTOFF CHECK
# ==============================================================================

# # [CNN] # [YOLO] # [THRESHOLD]
def fuse_cnn_and_yolo_predictions(
    cnn_top3: List[Dict[str, Any]],
    yolo_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combines the CNN Top-1 classification and the highest-confidence YOLOv8
    bounding-box detection:
      - If both models are present and predict the same class, `models_agreed = True`
        and we take the higher confidence of the two.
      - If CNN and YOLOv8 disagree, `models_agreed = False`, we pick whichever
        model has the higher confidence as the primary prediction, and we include
        both predictions in `fusion_explanation` so the UI shows both.
      - If only one model produced a prediction, we use that model's prediction.
    """
    cnn_top1 = cnn_top3[0] if cnn_top3 else None
    yolo_top1 = yolo_boxes[0] if yolo_boxes else None

    if cnn_top1 and yolo_top1:
        if cnn_top1["class_index"] == yolo_top1["class_index"]:
            best = cnn_top1 if cnn_top1["confidence"] >= yolo_top1["confidence"] else yolo_top1
            return {
                "chosen": best,
                "cnn_top1": cnn_top1,
                "yolo_top1": yolo_top1,
                "models_agreed": True,
                "winning_source": "CNN + YOLOv8 Consensus",
                "fusion_explanation": (
                    f"Both CNN ({cnn_top1['confidence_percent']}%) and YOLOv8 "
                    f"({yolo_top1['confidence_percent']}%) agreed on {best['display_name']}."
                ),
            }

        # Models disagreed: choose the higher-confidence prediction and show both
        if yolo_top1["confidence"] > cnn_top1["confidence"]:
            chosen = yolo_top1
            winner = "YOLOv8 Detector (Higher Confidence)"
        else:
            chosen = cnn_top1
            winner = "CNN Classifier (Higher Confidence)"

        return {
            "chosen": chosen,
            "cnn_top1": cnn_top1,
            "yolo_top1": yolo_top1,
            "models_agreed": False,
            "winning_source": winner,
            "fusion_explanation": (
                f"CNN predicted {cnn_top1['display_name']} ({cnn_top1['confidence_percent']}%) "
                f"while YOLOv8 detected {yolo_top1['display_name']} ({yolo_top1['confidence_percent']}%). "
                f"Selected {chosen['display_name']} based on higher confidence."
            ),
        }

    if cnn_top1:
        return {
            "chosen": cnn_top1,
            "cnn_top1": cnn_top1,
            "yolo_top1": None,
            "models_agreed": True,
            "winning_source": "CNN Classifier",
            "fusion_explanation": (
                f"Classified by CNN as {cnn_top1['display_name']} ({cnn_top1['confidence_percent']}%)."
            ),
        }

    if yolo_top1:
        return {
            "chosen": yolo_top1,
            "cnn_top1": None,
            "yolo_top1": yolo_top1,
            "models_agreed": True,
            "winning_source": "YOLOv8 Detector",
            "fusion_explanation": (
                f"Localized by YOLOv8 as {yolo_top1['display_name']} ({yolo_top1['confidence_percent']}%)."
            ),
        }

    return {
        "chosen": None,
        "cnn_top1": None,
        "yolo_top1": None,
        "models_agreed": False,
        "winning_source": "None",
        "fusion_explanation": "No prediction available.",
    }


# ==============================================================================
# 5. COMPLETE END-TO-END LEAF SCANNER PIPELINE
# ==============================================================================

def analyze_leaf_image(
    image_path: str,
    annotated_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the complete pipeline on `image_path`:
      - Verifies that real trained model weights (`models/cnn_pest.pt` or
        `models/yolo_pest.pt`) are loaded. If neither is trained yet, returns
        `status = "models_not_trained"` with `"Model not trained yet, run train_cnn.py"`.
      - Runs CNN Top-3 + YOLOv8 bounding-box localization + affected area %.
      - Checks `config.CONFIDENCE_CUTOFF`; if below cutoff, flags `is_uncertain = True`
        with `"Uncertain, please retake the photo"`.
      - Executes the 4-stage DSA pipeline (`dsa_pipeline.py`) and returns all results.
    """
    if not os.path.isfile(image_path):
        return {
            "status": "error",
            "error_message": f"Image file not found: {image_path}",
        }

    # Read BGR image with OpenCV and RGB image with Pillow
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {
            "status": "error",
            "error_message": "Could not decode uploaded image file. Please upload a valid JPG/PNG.",
        }

    pil_image = Image.open(image_path).convert("RGB")

    # Stage 1 OpenCV visual feature extraction always works on real image pixels
    feature_array = extract_feature_array(image_bgr)

    # STRICT CHECK: Do not fake output if neither CNN nor YOLO weights are trained yet
    if MODEL_REGISTRY.cnn_model is None and MODEL_REGISTRY.yolo_model is None:
        return {
            "status": "models_not_trained",
            "error_message": "Model not trained yet, run train_cnn.py",
            "cnn_status": MODEL_REGISTRY.cnn_status,
            "yolo_status": MODEL_REGISTRY.yolo_status,
            "feature_array": feature_array,
            "feature_names": config.FEATURE_NAMES,
        }

    # Run CNN Top-3 classification (if CNN weights are loaded)
    cnn_top3 = run_cnn_classification(pil_image)

    # Run YOLOv8 bounding-box detection & affected area % calculation (if YOLO weights are loaded)
    yolo_boxes, affected_area_percent = run_yolo_detection_and_draw_boxes(
        image_bgr, annotated_save_path=annotated_output_path
    )

    # If YOLO found no boxes (or only CNN is loaded), estimate affected area % from
    # Stage 1 OpenCV lesion + chlorosis + necrotic feature ratios on the real image
    if affected_area_percent <= 0.0:
        lesion_plus_chlorosis = (
            feature_array[1] + feature_array[2] + (feature_array[5] * 0.5)
        ) * 100.0
        affected_area_percent = round(max(1.0, min(95.0, lesion_plus_chlorosis)), 2)

    # Fuse CNN and YOLOv8 predictions
    fusion = fuse_cnn_and_yolo_predictions(cnn_top3, yolo_boxes)
    chosen = fusion["chosen"]

    if chosen is None:
        return {
            "status": "uncertain",
            "is_uncertain": True,
            "uncertain_message": "Uncertain, please retake the photo",
            "cnn_top3": cnn_top3,
            "yolo_boxes": yolo_boxes,
            "affected_area_percent": affected_area_percent,
            "feature_array": feature_array,
            "feature_names": config.FEATURE_NAMES,
        }

    fused_confidence = float(chosen["confidence"])
    is_uncertain = fused_confidence < config.CONFIDENCE_CUTOFF

    # Execute all 4 DSA stages using the fused crop, pest name, and affected area %
    dsa_result = run_four_stage_dsa_pipeline(
        image_bgr=image_bgr,
        crop_name=chosen["crop"],
        pest_name=chosen["pest_name"],
        affected_area_percent=affected_area_percent,
        hash_table=GLOBAL_PEST_HASH_TABLE,
    )

    pest_record = dsa_result["pest_record"] or {}

    return {
        "status": "uncertain" if is_uncertain else "ok",
        "is_uncertain": is_uncertain,
        "uncertain_message": (
            f"Uncertain, please retake the photo (Confidence {chosen['confidence_percent']}% "
            f"is below {config.CONFIDENCE_CUTOFF * 100:.0f}% cutoff)."
            if is_uncertain
            else ""
        ),
        "crop": chosen["crop"],
        "pest_name": chosen["pest_name"],
        "display_name": pest_record.get("display_name", chosen["display_name"]),
        "scientific_name": pest_record.get("scientific_name", "IP102 Insect Pest"),
        "life_stage": pest_record.get("life_stage", "Larva / Adult"),
        "symptoms": pest_record.get("symptoms", ""),
        "fused_confidence": fused_confidence,
        "fused_confidence_percent": chosen["confidence_percent"],
        "models_agreed": fusion["models_agreed"],
        "winning_source": fusion["winning_source"],
        "fusion_explanation": fusion["fusion_explanation"],
        "cnn_top3": cnn_top3,
        "yolo_boxes": yolo_boxes,
        "affected_area_percent": affected_area_percent,
        "severity": dsa_result["severity"],
        "severity_strategy": dsa_result["severity_strategy"],
        "decision_rule": dsa_result["decision_rule"],
        "decision_path": dsa_result["decision_path"],
        "feature_array": dsa_result["feature_array"],
        "feature_names": dsa_result["feature_names"],
        "hash_key_used": dsa_result["hash_key_used"],
        "hash_bucket_index": dsa_result["hash_bucket_index"],
        "hash_chain_steps": dsa_result["hash_chain_steps"],
        "ranked_treatments": [] if is_uncertain else dsa_result["ranked_treatments"],
        "telemetry_ms": dsa_result["telemetry_ms"],
    }
