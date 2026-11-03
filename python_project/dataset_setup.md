# IP102 Dataset Setup & Preparation Guide (`dataset_setup.md`)

**Project Title:** SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING  
**Project By:** PHANI NANDAN M S , VEDHESHA PARTHIBAN , KHADGA LOHITH K , SHIVANI BALLANI

---

## 1. Where to Download the Official IP102 Dataset

The **IP102 (Insect Pest 102)** benchmark dataset was published at CVPR 2019 (*"IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition"*). It contains **75,222 images** for classification and **18,981 annotated images** (with Pascal VOC XML bounding boxes) across **102 pest classes** affecting **8 agricultural crops** (`Rice`, `Corn`, `Wheat`, `Beet`, `Alfalfa`, `Vitis/Grapes`, `Citrus`, `Mango`).

Download the two official archives from the IP102 GitHub repository (`https://github.com/xpwu95/IP102`):

1. **Classification Split Archive (`ip102_v1.1.tar` / `Classification.zip`)**:
   - Contains `images/` (all pest photos) and text split files: `train.txt`, `val.txt`, `test.txt`, and `classes.txt`.
2. **Detection Subset Archive (`Detection.tar` / `VOC2007.zip`)**:
   - Contains Pascal VOC format bounding box annotations (`Annotations/*.xml`, `JPEGImages/*.jpg`, and `ImageSets/Main/{trainval.txt, test.txt}`).

---

## 2. Exact Folder Layout Expected by `prepare_dataset.py`

Create a folder named `raw_ip102/` inside your project root (or set the `RAW_IP102_DIR` path in `config.py`) and extract both archives so the directory looks like this:

```text
raw_ip102/
├── classification/
│   ├── classes.txt                 # 102 lines: "1 rice leaf roller", "2 rice leaf caterpillar", ...
│   ├── train.txt                   # Lines: "<image_filename.jpg> <0_based_class_id>"
│   ├── val.txt                     # Lines: "<image_filename.jpg> <0_based_class_id>"
│   ├── test.txt                    # Lines: "<image_filename.jpg> <0_based_class_id>"
│   └── images/                     # Folder containing 00001.jpg, 00002.jpg, ...
│       ├── 00001.jpg
│       └── ...
└── detection/
    ├── Annotations/                # Folder containing Pascal VOC XML bounding-box files
    │   ├── IP000000001.xml
    │   └── ...
    ├── JPEGImages/                 # Folder containing detection images
    │   ├── IP000000001.jpg
    │   └── ...
    └── ImageSets/
        └── Main/
            ├── trainval.txt        # List of image stems (without .jpg) for training/validation
            └── test.txt            # List of image stems (without .jpg) for testing
```

> **IMPORTANT — VERIFY FOLDER NAMES AGAINST YOUR DOWNLOADED COPY:**  
> Different mirrors of IP102 (Google Drive, Baidu Pan, Kaggle) sometimes name the inner folders `ip102_v1.1/images` instead of `classification/images`, or `Detection/VOC2007/Annotations` instead of `detection/Annotations`.  
> **Before running `prepare_dataset.py`, open your downloaded IP102 folder and verify the exact folder names!** If your folder names differ, either rename them to match the tree above or edit the `# [CHANGE_HERE]` path constants at the top of `prepare_dataset.py`.

---

## 3. How to Run `prepare_dataset.py`

Once your `raw_ip102/` folder is in place, run:

```bash
# Default: selects the TOP 20 classes with the most images (from config.py)
python prepare_dataset.py

# Optional: override the number of classes (between 15 and 20) or raw dataset path
python prepare_dataset.py --raw-dir ./raw_ip102 --top-n 20
```

### What `prepare_dataset.py` Produces Automatically
1. **`models/class_map.json`**: Maps `new_class_index (0..19)` $\rightarrow$ `original_ip102_id`, `pest_name`, and `crop`.
2. **`dataset_cnn/{train,val,test}/<class_idx>_<pest_name>/`**: Image folders ready for PyTorch `ImageFolder` in `train_cnn.py`.
3. **`dataset_yolo/images/{train,val,test}/` & `dataset_yolo/labels/{train,val,test}/`**: Converts Pascal VOC `<xmin, ymin, xmax, ymax>` XML boxes into normalized YOLO `<class_idx> <x_center> <y_center> <width> <height>` `.txt` files.
4. **`dataset_yolo/data.yaml`**: Ready-to-train Ultralytics YOLOv8 dataset configuration file.
5. **Console Summary Table**: Prints the exact `Class Index | IP102 ID | Pest Name | Crop | Train | Val | Test` counts.
