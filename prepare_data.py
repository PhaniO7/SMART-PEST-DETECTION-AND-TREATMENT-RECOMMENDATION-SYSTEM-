"""
================================================================================
FILE: prepare_data.py
PURPOSE:
    Prepares both the CNN classification dataset (`ImageFolder` structure) and
    the YOLOv8 object detection dataset (normalized `.txt` bounding boxes +
    `data.yaml`) from the official IP102 dataset for the `config.NUM_PESTS`
    (20) pests across `config.NUM_CROPS` (8) crops defined in `config.py`.

    KEY MATCHING RULES:
      1. Normalizes both `config.PEST_SHORTLIST` (`match_keyword`) and every line
         in IP102's `classes.txt` by removing spaces, hyphens, and brackets
         `()[]{}` and lowercasing both sides (so `"armyworm"` matches
         `"army worm"`).
      2. If any shortlist entry matches ZERO classes or MORE THAN ONE class
         (for example `"aphids"`), it STOPS immediately and prints the candidate
         or closest class names from `classes.txt` so you can pick the exact one
         and edit `match_keyword` in `config.py`.
      3. Prints a summary table of Class ID, Pest Name, Crop, and Train/Val/Test
         image counts, and warns for every class under
         `config.MIN_IMAGES_WARNING_THRESHOLD` (300) images.

HOW IT CONNECTS TO OTHER FILES:
    - Reads `IP102_RAW_DIR`, `PEST_SHORTLIST`, `MIN_IMAGES_WARNING_THRESHOLD`,
      `CNN_DATA_DIR`, `YOLO_DATA_DIR`, and `YOLO_DATA_YAML` from `config.py`.
    - Produces the folders required by `train_cnn.py`, `train_yolo.py`, and
      `evaluate.py`.
    - Tested by `test_all.py`.
================================================================================
"""

# Import standard library modules for file handling, regex, similarity, and XML parsing
import difflib
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# Import central project configuration
import config


# ==============================================================================
# 1. NAME NORMALIZATION & IP102 CLASS MATCHING (# [DATASET])
# ==============================================================================

# # [DATASET]
def normalize_pest_name(text: str) -> str:
    """
    Removes spaces, hyphens, underscores, and brackets `()[]{}` and converts
    `text` to lowercase so `"armyworm"` and `"army worm"` normalize to the
    exact same string (`"armyworm"`).
    """
    # Strip any leading integer label if present (e.g. "1 rice leaf roller" -> "rice leaf roller")
    cleaned = re.sub(r"^\s*\d+\s+", "", text.strip().lower())
    # Remove spaces, hyphens, underscores, and round/square/curly brackets
    cleaned = re.sub(r"[\s\-_\(\)\[\]\{\}]+", "", cleaned)
    return cleaned


# # [DATASET]
def parse_classes_txt(classes_txt_path: str) -> List[Dict[str, Any]]:
    """
    Reads IP102 `classes.txt` and returns a list of dictionaries:
      `{"ip102_id": 0..101, "raw_name": str, "norm_name": str}`
    Note: IP102 `classes.txt` lines are 1-indexed in display ("1 rice leaf roller")
    while `train.txt` / `val.txt` / `test.txt` use 0-indexed class IDs (0..101).
    """
    if not os.path.isfile(classes_txt_path):
        raise FileNotFoundError(
            f"IP102 classes.txt not found at: {classes_txt_path}\n"
            f"Please check IP102_RAW_DIR in config.py."
        )

    entries: List[Dict[str, Any]] = []
    with open(classes_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(lines):
        # Remove leading 1-based number if present
        raw_name = re.sub(r"^\d+\s+", "", line).strip()
        entries.append(
            {
                "ip102_id": idx,
                "raw_name": raw_name,
                "norm_name": normalize_pest_name(raw_name),
            }
        )
    return entries


# # [DATASET]
def match_shortlist_to_ip102(
    shortlist: List[Dict[str, Any]],
    ip102_classes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Matches each entry in `shortlist` (`config.PEST_SHORTLIST`) against
    `ip102_classes` after removing spaces, hyphens, and brackets and lowercasing
    both sides.

    If any shortlist entry matches ZERO classes or MORE THAN ONE class, stops
    immediately (`ValueError`) and prints the candidate/closest class names so
    the user can update `match_keyword` in `config.py`.
    """
    matched_results: List[Dict[str, Any]] = []
    all_raw_names = [c["raw_name"] for c in ip102_classes]

    for item in shortlist:
        new_class_id = int(item["class_id"])
        crop = str(item["crop"])
        display_name = str(item["name"])
        keyword = str(item.get("match_keyword", display_name))
        norm_query = normalize_pest_name(keyword)

        # 1. Check for exact normalized match first
        exact_matches = [
            c for c in ip102_classes if c["norm_name"] == norm_query
        ]

        # 2. If no exact normalized match, check partial normalized substring match
        if len(exact_matches) == 1:
            candidates = exact_matches
        else:
            candidates = [
                c
                for c in ip102_classes
                if norm_query in c["norm_name"] or c["norm_name"] in norm_query
            ]

        # STOP if ZERO classes matched
        if len(candidates) == 0:
            norm_map = {c["norm_name"]: c["raw_name"] for c in ip102_classes}
            closest_norms = difflib.get_close_matches(
                norm_query, list(norm_map.keys()), n=5, cutoff=0.35
            )
            closest_names = [norm_map[n] for n in closest_norms]
            if not closest_names:
                closest_names = difflib.get_close_matches(
                    keyword.lower(), all_raw_names, n=5, cutoff=0.25
                )

            msg = (
                f"\n[STOP] Shortlist class_id={new_class_id} ('{display_name}', "
                f"match_keyword='{keyword}') matched ZERO classes in IP102 classes.txt!\n"
                f"Closest candidate class names in classes.txt:\n"
                + "\n".join(f"  - {name}" for name in closest_names)
                + "\nPlease pick the right one and edit 'match_keyword' in config.py."
            )
            print(msg)
            raise ValueError(msg)

        # STOP if MORE THAN ONE class matched (e.g. "aphids")
        if len(candidates) > 1:
            candidate_list_str = "\n".join(
                f"  - [IP102 id={c['ip102_id']}] {c['raw_name']}"
                for c in candidates
            )
            msg = (
                f"\n[STOP] Shortlist class_id={new_class_id} ('{display_name}', "
                f"match_keyword='{keyword}') is ambiguous and matched "
                f"{len(candidates)} classes in IP102 classes.txt:\n"
                f"{candidate_list_str}\n"
                f"Please pick the exact class name above and edit 'match_keyword' in config.py."
            )
            print(msg)
            raise ValueError(msg)

        # Exactly 1 unique match found
        chosen = candidates[0]
        matched_results.append(
            {
                "new_class_id": new_class_id,
                "crop": crop,
                "name": display_name,
                "ip102_id": chosen["ip102_id"],
                "ip102_raw_name": chosen["raw_name"],
            }
        )

    return matched_results


# ==============================================================================
# 2. PASCAL VOC XML -> YOLO BOUNDING BOX CONVERSION (# [DATASET] # [YOLO])
# ==============================================================================

# # [DATASET] # [YOLO]
def convert_voc_xml_to_yolo(
    xml_path: str,
    ip102_to_new_id: Dict[int, int],
) -> List[str]:
    """
    Parses a Pascal VOC XML annotation file from IP102's `Annotations/` folder
    and converts bounding boxes belonging to our selected pests into normalized
    YOLO format lines:
        `<new_class_id> <x_center> <y_center> <width> <height>`
    where all coordinates are floats in [0.0, 1.0].
    """
    if not os.path.isfile(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        return []

    img_w = float(size_node.findtext("width", default="0"))
    img_h = float(size_node.findtext("height", default="0"))
    if img_w <= 0 or img_h <= 0:
        return []

    yolo_lines: List[str] = []
    for obj in root.findall("object"):
        raw_cls = obj.findtext("name", default="").strip()
        # In IP102 Detection VOC XML, <name> is the integer class ID (0..101)
        if not raw_cls.isdigit():
            continue
        old_id = int(raw_cls)
        if old_id not in ip102_to_new_id:
            continue

        new_id = ip102_to_new_id[old_id]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = max(0.0, float(bndbox.findtext("xmin", default="0")))
        ymin = max(0.0, float(bndbox.findtext("ymin", default="0")))
        xmax = min(img_w, float(bndbox.findtext("xmax", default="0")))
        ymax = min(img_h, float(bndbox.findtext("ymax", default="0")))

        box_w = max(0.0, xmax - xmin)
        box_h = max(0.0, ymax - ymin)
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h

        yolo_lines.append(
            f"{new_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    return yolo_lines


# ==============================================================================
# 3. SUMMARY TABLE & DATASET BUILDER (# [DATASET])
# ==============================================================================

# # [DATASET]
def print_dataset_summary_table(
    matched_classes: List[Dict[str, Any]],
    split_counts: Dict[int, Dict[str, int]],
) -> List[str]:
    """
    Prints a formatted summary table of:
      Class ID | Pest Name | IP102 Class | Crop | Train | Val | Test | Total
    and prints a warning for every class with fewer than
    `config.MIN_IMAGES_WARNING_THRESHOLD` (300) total images.
    Returns the list of warning messages.
    """
    header = (
        f"{'ID':<4} | {'Crop':<14} | {'Shortlist Pest Name':<36} | "
        f"{'Train':>6} | {'Val':>5} | {'Test':>6} | {'Total':>6}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(
        f"IP102 DATASET SUMMARY ({config.NUM_CROPS} crops and {config.NUM_PESTS} pests)"
    )
    print(sep)
    print(header)
    print(sep)

    warnings_list: List[str] = []
    for item in matched_classes:
        cid = int(item["new_class_id"])
        counts = split_counts.get(cid, {"train": 0, "val": 0, "test": 0})
        tr = counts.get("train", 0)
        va = counts.get("val", 0)
        te = counts.get("test", 0)
        tot = tr + va + te
        print(
            f"{cid:<4} | {item['crop']:<14} | {item['name'][:36]:<36} | "
            f"{tr:>6} | {va:>5} | {te:>6} | {tot:>6}"
        )
        if tot < config.MIN_IMAGES_WARNING_THRESHOLD:
            warn_msg = (
                f"[WARNING] Class {cid} ('{item['name']}', crop={item['crop']}) "
                f"has only {tot} total images (< {config.MIN_IMAGES_WARNING_THRESHOLD} "
                f"recommended in config.MIN_IMAGES_WARNING_THRESHOLD)."
            )
            warnings_list.append(warn_msg)

    print(sep)
    for w in warnings_list:
        print(w)
    if not warnings_list:
        print(
            f"All {config.NUM_PESTS} classes meet the minimum threshold of "
            f"{config.MIN_IMAGES_WARNING_THRESHOLD} images."
        )
    print()
    return warnings_list


# # [DATASET]
def write_yolo_data_yaml(
    yaml_path: str,
    yolo_root_dir: str,
    matched_classes: List[Dict[str, Any]],
) -> None:
    """
    Writes the Ultralytics YOLOv8 `data.yaml` file mapping class indices
    `0 .. config.NUM_PESTS - 1` to their pest names.
    """
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    lines = [
        f"# Ultralytics YOLOv8 dataset configuration ({config.NUM_CROPS} crops, {config.NUM_PESTS} pests)",
        f"path: {os.path.abspath(yolo_root_dir)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(matched_classes)}",
        "names:",
    ]
    for item in sorted(matched_classes, key=lambda x: int(x["new_class_id"])):
        lines.append(f"  {int(item['new_class_id'])}: \"{item['name']}\"")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# # [DATASET]
def build_datasets(raw_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Main dataset preparation workflow:
      1. Reads `classes.txt` from `raw_dir` (defaults to `config.IP102_RAW_DIR`).
      2. Matches all `config.NUM_PESTS` pests from `config.PEST_SHORTLIST`.
      3. Reads `train.txt`, `val.txt`, and `test.txt` splits and copies images
         into `config.CNN_DATA_DIR/{train,val,test}/{class_id:02d}/`.
      4. Converts available Pascal VOC XML annotations into `config.YOLO_DATA_DIR`
         and creates `config.YOLO_DATA_YAML`.
      5. Prints the image count table and warnings for classes under
         `config.MIN_IMAGES_WARNING_THRESHOLD`.
    """
    ip102_dir = raw_dir if raw_dir is not None else config.IP102_RAW_DIR
    classes_txt = os.path.join(ip102_dir, "classes.txt")

    print(f"Checking IP102 dataset folder: {ip102_dir}")
    ip102_classes = parse_classes_txt(classes_txt)
    matched = match_shortlist_to_ip102(config.PEST_SHORTLIST, ip102_classes)

    ip102_to_new: Dict[int, int] = {
        int(m["ip102_id"]): int(m["new_class_id"]) for m in matched
    }

    # Prepare split counters
    split_counts: Dict[int, Dict[str, int]] = {
        int(m["new_class_id"]): {"train": 0, "val": 0, "test": 0}
        for m in matched
    }

    images_src_dir = os.path.join(ip102_dir, "images")
    voc_ann_dir = os.path.join(ip102_dir, "Annotations")

    for split_name in ("train", "val", "test"):
        split_file = os.path.join(ip102_dir, f"{split_name}.txt")
        if not os.path.isfile(split_file):
            continue

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_filename, old_cls_str = parts[0], parts[1]
                if not old_cls_str.isdigit():
                    continue
                old_cls = int(old_cls_str)
                if old_cls not in ip102_to_new:
                    continue

                new_cls = ip102_to_new[old_cls]
                split_counts[new_cls][split_name] += 1

                # Copy image into CNN ImageFolder structure if source image exists
                src_img = os.path.join(images_src_dir, img_filename)
                if os.path.isfile(src_img):
                    cnn_cls_dir = os.path.join(
                        config.CNN_DATA_DIR, split_name, f"{new_cls:02d}"
                    )
                    os.makedirs(cnn_cls_dir, exist_ok=True)
                    shutil.copy2(src_img, os.path.join(cnn_cls_dir, img_filename))

                    # Check if a Pascal VOC XML box exists for YOLO detection subset
                    stem = os.path.splitext(img_filename)[0]
                    xml_file = os.path.join(voc_ann_dir, f"{stem}.xml")
                    yolo_lines = convert_voc_xml_to_yolo(xml_file, ip102_to_new)
                    if yolo_lines:
                        yolo_img_dir = os.path.join(
                            config.YOLO_DATA_DIR, "images", split_name
                        )
                        yolo_lbl_dir = os.path.join(
                            config.YOLO_DATA_DIR, "labels", split_name
                        )
                        os.makedirs(yolo_img_dir, exist_ok=True)
                        os.makedirs(yolo_lbl_dir, exist_ok=True)
                        shutil.copy2(
                            src_img, os.path.join(yolo_img_dir, img_filename)
                        )
                        with open(
                            os.path.join(yolo_lbl_dir, f"{stem}.txt"),
                            "w",
                            encoding="utf-8",
                        ) as lf:
                            lf.write("\n".join(yolo_lines) + "\n")

    write_yolo_data_yaml(config.YOLO_DATA_YAML, config.YOLO_DATA_DIR, matched)
    warnings_list = print_dataset_summary_table(matched, split_counts)

    return {
        "matched": matched,
        "split_counts": split_counts,
        "warnings": warnings_list,
    }


if __name__ == "__main__":
    try:
        build_datasets()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nprepare_data.py stopped: {exc}")
        sys.exit(1)
