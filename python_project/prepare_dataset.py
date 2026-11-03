"""
================================================================================
FILE: prepare_dataset.py
PURPOSE:
    Prepares the IP102 dataset for training both our PyTorch CNN Classifier
    and Ultralytics YOLOv8 Detector without training on all 102 classes:
      1. Counts images per class in the IP102 classification split (`train.txt`,
         `val.txt`, `test.txt`).
      2. Selects the TOP N classes (default N=20 from `config.TOP_N_CLASSES`,
         configurable in range 15..20), ensuring at least 2 distinct host crops
         are represented.
      3. Saves `models/class_map.json` mapping `0..(N-1)` -> `(ip102_id, pest_name, crop)`.
      4. Builds the PyTorch `ImageFolder` directory tree (`dataset_cnn/train`,
         `dataset_cnn/val`, `dataset_cnn/test`) for only the selected N classes.
      5. Parses Pascal VOC XML bounding-box annotations from the IP102 detection
         split, filters boxes belonging to the selected N classes, converts them
         to normalized YOLO `.txt` format (`class_idx x_center y_center w h`),
         and writes `dataset_yolo/data.yaml`.
      6. Prints a clean summary table showing Class ID, Pest Name, Host Crop,
         and Train / Val / Test image counts.

CONCEPTS USED:
    - Dataset Filtering & Class Remapping (# [DATASET], # [CONFIG])
    - Pascal VOC XML to Normalized YOLO Bounding-Box Conversion (# [YOLO])

HOW IT CONNECTS TO OTHER FILES:
    - Reads default paths and `TOP_N_CLASSES` from `config.py`.
    - Uses `get_crop_for_ip102_class_id()` from `data/pest_database.py`.
    - Produces `models/class_map.json`, `dataset_cnn/`, and `dataset_yolo/data.yaml`
      consumed by `train_cnn.py`, `train_yolo.py`, `evaluate.py`, and `inference.py`.
================================================================================
"""

# Import standard library modules for CLI args, file copying, JSON, and XML parsing
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Import PyYAML to write the Ultralytics YOLOv8 data.yaml file cleanly
import yaml

# Import our central project configuration
import config

# Import IP102 crop mapping helper from our pest database module
from data.pest_database import get_crop_for_ip102_class_id


# ==============================================================================
# 1. HELPER: PARSE IP102 CLASSES.TXT AND TRAIN/VAL/TEST SPLIT LISTS
# ==============================================================================

# # [DATASET]
def load_ip102_class_names(classes_txt_path: str) -> Dict[int, str]:
    """
    Reads `classes.txt` from the IP102 classification folder and returns a dictionary
    mapping 0-based IP102 class ID (0..101) -> pest name string.
    """
    class_names: Dict[int, str] = {}
    if not os.path.isfile(classes_txt_path):
        raise FileNotFoundError(
            f"Could not find classes.txt at '{classes_txt_path}'. "
            "Please check dataset_setup.md and verify your downloaded IP102 folder names."
        )

    with open(classes_txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            # IP102 classes.txt lines are formatted like: "1 rice leaf roller"
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                zero_based_id = int(parts[0]) - 1
                class_names[zero_based_id] = parts[1].strip().lower()
            else:
                class_names[idx] = stripped.lower()

    return class_names


# # [DATASET]
def parse_split_file(split_txt_path: str) -> List[Tuple[str, int]]:
    """
    Reads an IP102 split file (`train.txt`, `val.txt`, or `test.txt`) where each
    line is `<image_filename.jpg> <0_based_class_id>` and returns a list of tuples.
    """
    entries: List[Tuple[str, int]] = []
    if not os.path.isfile(split_txt_path):
        raise FileNotFoundError(
            f"Missing split file: '{split_txt_path}'. Please verify folder names in raw_ip102/."
        )

    with open(split_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                label_id = int(parts[1])
                entries.append((img_name, label_id))

    return entries


# ==============================================================================
# 2. SELECT TOP N CLASSES WITH MOST IMAGES (KEEPING MULTI-CROP REPRESENTATION)
# ==============================================================================

# # [DATASET] # [CONFIG]
def select_top_n_classes(
    train_entries: List[Tuple[str, int]],
    val_entries: List[Tuple[str, int]],
    test_entries: List[Tuple[str, int]],
    class_names: Dict[int, str],
    top_n: int = config.TOP_N_CLASSES,
) -> Tuple[List[int], Dict[int, Dict[str, int]]]:
    """
    Counts images per class across train/val/test and selects the TOP N classes
    (validated between 15 and 20) with the highest total image counts while
    ensuring at least `config.MIN_CROPS_REPRESENTED` crops are included.
    """
    # Validate that top_n is within the allowed range [15, 20]
    if not (config.MIN_CLASSES_ALLOWED <= top_n <= config.MAX_CLASSES_ALLOWED):
        raise ValueError(
            f"top_n must be between {config.MIN_CLASSES_ALLOWED} and "
            f"{config.MAX_CLASSES_ALLOWED}, got {top_n}."
        )

    # Count train, val, and test images per 0-based IP102 class ID
    counts_by_class: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0, "total": 0}
    )

    for _, cid in train_entries:
        counts_by_class[cid]["train"] += 1
        counts_by_class[cid]["total"] += 1
    for _, cid in val_entries:
        counts_by_class[cid]["val"] += 1
        counts_by_class[cid]["total"] += 1
    for _, cid in test_entries:
        counts_by_class[cid]["test"] += 1
        counts_by_class[cid]["total"] += 1

    # Sort all IP102 class IDs descending by total image count
    sorted_classes = sorted(
        counts_by_class.keys(),
        key=lambda cid: counts_by_class[cid]["total"],
        reverse=True,
    )

    # Take the top_n classes with the most images
    selected_ids: List[int] = sorted_classes[:top_n]

    # Verify that at least MIN_CROPS_REPRESENTED crops are present in selected_ids
    crops_present = {get_crop_for_ip102_class_id(cid + 1) for cid in selected_ids}
    if len(crops_present) < config.MIN_CROPS_REPRESENTED and len(sorted_classes) > top_n:
        # Swap the last selected class with the highest-count class from a different crop
        for candidate_cid in sorted_classes[top_n:]:
            candidate_crop = get_crop_for_ip102_class_id(candidate_cid + 1)
            if candidate_crop not in crops_present:
                selected_ids[-1] = candidate_cid
                break

    return selected_ids, counts_by_class


# ==============================================================================
# 3. BUILD CNN FOLDER DATASET (TRAIN / VAL / TEST) FOR SELECTED CLASSES
# ==============================================================================

# # [DATASET] # [CNN]
def build_cnn_folder_dataset(
    raw_images_dir: str,
    output_cnn_dir: str,
    splits: Dict[str, List[Tuple[str, int]]],
    old_to_new_idx: Dict[int, int],
) -> None:
    """
    Copies images belonging to the selected top N classes into PyTorch ImageFolder
    format: `dataset_cnn/{train,val,test}/<new_class_idx:02d>/image.jpg`.
    """
    for split_name, entries in splits.items():
        for img_filename, old_cid in entries:
            # Skip images that do not belong to our selected top N classes
            if old_cid not in old_to_new_idx:
                continue

            new_idx = old_to_new_idx[old_cid]
            # Use zero-padded 2-digit folder name ("00", "01", ..., "19") so alphabetical
            # order in torchvision.datasets.ImageFolder matches integer class_index!
            class_folder = os.path.join(output_cnn_dir, split_name, f"{new_idx:02d}")
            os.makedirs(class_folder, exist_ok=True)

            src_path = os.path.join(raw_images_dir, img_filename)
            dst_path = os.path.join(class_folder, os.path.basename(img_filename))
            if os.path.isfile(src_path) and not os.path.isfile(dst_path):
                shutil.copy2(src_path, dst_path)


# ==============================================================================
# 4. CONVERT PASCAL VOC XML BOUNDING BOXES TO YOLOv8 TXT FORMAT
# ==============================================================================

# # [DATASET] # [YOLO]
def convert_voc_bbox_to_yolo(
    img_width: float,
    img_height: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Tuple[float, float, float, float]:
    """
    Converts Pascal VOC pixel coordinates `(xmin, ymin, xmax, ymax)` into
    normalized YOLO format `(x_center, y_center, width, height)` in [0.0, 1.0].
    """
    dw = 1.0 / max(1.0, img_width)
    dh = 1.0 / max(1.0, img_height)

    x_center = ((xmin + xmax) / 2.0) * dw
    y_center = ((ymin + ymax) / 2.0) * dh
    box_w = (xmax - xmin) * dw
    box_h = (ymax - ymin) * dh

    # Clamp normalized coordinates to [0.0, 1.0]
    return (
        round(max(0.0, min(1.0, x_center)), 6),
        round(max(0.0, min(1.0, y_center)), 6),
        round(max(0.0, min(1.0, box_w)), 6),
        round(max(0.0, min(1.0, box_h)), 6),
    )


# # [DATASET] # [YOLO]
def build_yolo_detection_dataset(
    raw_detection_dir: str,
    output_yolo_dir: str,
    old_to_new_idx: Dict[int, int],
    name_to_old_idx: Dict[str, int],
    class_map: Dict[str, Any],
) -> int:
    """
    Parses Pascal VOC XML files in `raw_ip102/detection/Annotations/`, converts
    bounding boxes belonging to the selected top N classes into YOLO `.txt` labels,
    copies matching images, and generates `dataset_yolo/data.yaml`.
    """
    annotations_dir = os.path.join(raw_detection_dir, "Annotations")
    jpeg_dir = os.path.join(raw_detection_dir, "JPEGImages")
    imagesets_dir = os.path.join(raw_detection_dir, "ImageSets", "Main")

    if not os.path.isdir(annotations_dir) or not os.path.isdir(jpeg_dir):
        print(
            f"[INFO] Detection folder '{raw_detection_dir}' not found yet. "
            "Writing data.yaml template; re-run after extracting Detection archive."
        )
        write_yolo_data_yaml(output_yolo_dir, class_map)
        return 0

    # Read trainval and test image stems
    trainval_txt = os.path.join(imagesets_dir, "trainval.txt")
    test_txt = os.path.join(imagesets_dir, "test.txt")

    trainval_stems: List[str] = []
    if os.path.isfile(trainval_txt):
        with open(trainval_txt, "r", encoding="utf-8") as f:
            trainval_stems = [line.strip().split()[0] for line in f if line.strip()]

    test_stems: List[str] = []
    if os.path.isfile(test_txt):
        with open(test_txt, "r", encoding="utf-8") as f:
            test_stems = [line.strip().split()[0] for line in f if line.strip()]

    # Split trainval_stems 85% train / 15% val
    val_cutoff = int(len(trainval_stems) * 0.85)
    split_stems = {
        "train": trainval_stems[:val_cutoff],
        "val": trainval_stems[val_cutoff:],
        "test": test_stems,
    }

    converted_count = 0

    for split_name, stems in split_stems.items():
        img_out_dir = os.path.join(output_yolo_dir, "images", split_name)
        lbl_out_dir = os.path.join(output_yolo_dir, "labels", split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for stem in stems:
            xml_path = os.path.join(annotations_dir, f"{stem}.xml")
            jpg_path = os.path.join(jpeg_dir, f"{stem}.jpg")
            if not os.path.isfile(xml_path) or not os.path.isfile(jpg_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            size_elem = root.find("size")
            if size_elem is None:
                continue

            img_w = float(size_elem.findtext("width", default="640"))
            img_h = float(size_elem.findtext("height", default="640"))

            yolo_lines: List[str] = []
            for obj in root.findall("object"):
                raw_label = (obj.findtext("name") or "").strip().lower()
                # IP102 VOC XML `<name>` can be 0-based/1-based integer ID or class name
                old_cid = None
                if raw_label.isdigit():
                    val = int(raw_label)
                    old_cid = val if val in old_to_new_idx else (val - 1)
                elif raw_label in name_to_old_idx:
                    old_cid = name_to_old_idx[raw_label]

                if old_cid not in old_to_new_idx:
                    continue

                new_idx = old_to_new_idx[old_cid]
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = float(bndbox.findtext("xmin", default="0"))
                ymin = float(bndbox.findtext("ymin", default="0"))
                xmax = float(bndbox.findtext("xmax", default="0"))
                ymax = float(bndbox.findtext("ymax", default="0"))

                xc, yc, bw, bh = convert_voc_bbox_to_yolo(
                    img_w, img_h, xmin, ymin, xmax, ymax
                )
                yolo_lines.append(f"{new_idx} {xc} {yc} {bw} {bh}")

            # Only copy image and write label if at least one selected pest box exists
            if yolo_lines:
                shutil.copy2(jpg_path, os.path.join(img_out_dir, f"{stem}.jpg"))
                with open(
                    os.path.join(lbl_out_dir, f"{stem}.txt"), "w", encoding="utf-8"
                ) as lf:
                    lf.write("\n".join(yolo_lines) + "\n")
                converted_count += 1

    write_yolo_data_yaml(output_yolo_dir, class_map)
    return converted_count


# # [DATASET] # [YOLO]
def write_yolo_data_yaml(output_yolo_dir: str, class_map: Dict[str, Any]) -> None:
    """Writes `dataset_yolo/data.yaml` required by Ultralytics YOLOv8."""
    os.makedirs(output_yolo_dir, exist_ok=True)
    names_dict = {
        int(k): v["pest_name"]
        for k, v in sorted(class_map.items(), key=lambda x: int(x[0]))
    }
    data_yaml_content = {
        "path": os.path.abspath(output_yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names_dict),
        "names": names_dict,
    }
    with open(config.YOLO_DATA_YAML_PATH, "w", encoding="utf-8") as yf:
        yaml.safe_dump(data_yaml_content, yf, sort_keys=False)


# ==============================================================================
# 5. MAIN CLI ENTRYPOINT & SUMMARY TABLE PRINTER
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare top-N IP102 classes for CNN & YOLOv8 training."
    )
    # # [CHANGE_HERE] Override --raw-dir or --top-n from the command line if needed
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=config.RAW_IP102_DIR,
        help="Path to extracted raw_ip102 directory.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=config.TOP_N_CLASSES,
        help="Number of top IP102 classes to select (15 to 20).",
    )
    args = parser.parse_args()

    classification_dir = os.path.join(args.raw_dir, "classification")
    detection_dir = os.path.join(args.raw_dir, "detection")

    classes_txt = os.path.join(classification_dir, "classes.txt")
    train_txt = os.path.join(classification_dir, "train.txt")
    val_txt = os.path.join(classification_dir, "val.txt")
    test_txt = os.path.join(classification_dir, "test.txt")
    images_dir = os.path.join(classification_dir, "images")

    print("=" * 88)
    print(f"PREPARING IP102 DATASET (SELECTING TOP {args.top_n} CLASSES)")
    print("=" * 88)

    class_names = load_ip102_class_names(classes_txt)
    train_entries = parse_split_file(train_txt)
    val_entries = parse_split_file(val_txt)
    test_entries = parse_split_file(test_txt)

    selected_old_ids, counts_by_class = select_top_n_classes(
        train_entries, val_entries, test_entries, class_names, top_n=args.top_n
    )

    # Build old_to_new_idx and class_map dictionary
    old_to_new_idx: Dict[int, int] = {}
    name_to_old_idx: Dict[str, int] = {name: cid for cid, name in class_names.items()}
    class_map: Dict[str, Any] = {}

    print(
        f"{'NewID':<6} | {'IP102_ID':<8} | {'Pest Name':<32} | {'Crop':<8} | "
        f"{'Train':<6} | {'Val':<5} | {'Test':<6} | {'Total':<6}"
    )
    print("-" * 88)

    for new_idx, old_cid in enumerate(selected_old_ids):
        old_to_new_idx[old_cid] = new_idx
        ip102_1based = old_cid + 1
        pest_name = class_names.get(old_cid, f"pest_{ip102_1based}")
        crop_name = get_crop_for_ip102_class_id(ip102_1based)
        c_stats = counts_by_class[old_cid]

        class_map[str(new_idx)] = {
            "class_index": new_idx,
            "ip102_id": ip102_1based,
            "pest_name": pest_name,
            "display_name": pest_name.title(),
            "crop": crop_name,
        }

        print(
            f"{new_idx:<6} | {ip102_1based:<8} | {pest_name[:32]:<32} | {crop_name:<8} | "
            f"{c_stats['train']:<6} | {c_stats['val']:<5} | {c_stats['test']:<6} | {c_stats['total']:<6}"
        )

    print("-" * 88)

    # Save class_map.json to models/class_map.json
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.CLASS_MAP_PATH, "w", encoding="utf-8") as cmf:
        json.dump(class_map, cmf, indent=2)
    print(f"[SAVED] Class map written to: {config.CLASS_MAP_PATH}")

    # Build CNN folder dataset
    splits = {"train": train_entries, "val": val_entries, "test": test_entries}
    build_cnn_folder_dataset(images_dir, config.CNN_DATASET_DIR, splits, old_to_new_idx)
    print(f"[SAVED] CNN ImageFolder dataset built at: {config.CNN_DATASET_DIR}")

    # Convert Pascal VOC XML detection boxes to YOLO format and write data.yaml
    yolo_count = build_yolo_detection_dataset(
        detection_dir,
        config.YOLO_DATASET_DIR,
        old_to_new_idx,
        name_to_old_idx,
        class_map,
    )
    print(
        f"[SAVED] YOLOv8 dataset ({yolo_count} annotated images) & data.yaml at: "
        f"{config.YOLO_DATA_YAML_PATH}"
    )


if __name__ == "__main__":
    main()
