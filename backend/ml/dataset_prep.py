"""
VisDrone Dataset Preparation for CrowdGuard.

Converts VisDrone2019-DET annotations (CSV format) to YOLO format
(normalized xywh) and generates datasets/VisDrone.yaml.

VisDrone annotation row format:
  bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

Category mapping:
  0  -> ignored region  (SKIP)
  1  -> pedestrian      -> class 0
  2  -> people          -> class 1
  3  -> bicycle         -> class 2
  4  -> car             -> class 3
  5  -> van             -> class 4
  6  -> truck           -> class 5
  7  -> tricycle        -> class 6
  8  -> awning-tricycle -> class 7
  9  -> bus             -> class 8
  10 -> motor           -> class 9
  11 -> others          (SKIP)
  12 -> crowd           (SKIP)

Run: python backend/ml/dataset_prep.py
"""

import sys
import cv2
from pathlib import Path

# Allow running as standalone script from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"
YAML_OUT = PROJECT_ROOT / "datasets" / "VisDrone.yaml"

SPLITS = {
    "train":    "VisDrone2019-DET-train",
    "val":      "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}

SKIP_CATEGORIES = {0, 11, 12}   # ignored region, others, crowd

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


# -----------------------------------------------------------------------
# Core conversion helpers
# -----------------------------------------------------------------------

def get_image_wh(img_path: Path):
    """Return (width, height) using OpenCV. Raises if image cannot be read."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"cv2.imread returned None for: {img_path}")
    h, w = img.shape[:2]
    return w, h


def convert_annotation_file(ann_path: Path, img_w: int, img_h: int):
    """
    Parse one VisDrone annotation file and return YOLO-format lines.

    Returns:
        (yolo_lines, skipped_count)
    """
    yolo_lines = []
    skipped = 0

    with open(ann_path, "r") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            parts = raw_line.split(",")
            if len(parts) < 6:
                skipped += 1
                continue

            try:
                bbox_left   = int(parts[0])
                bbox_top    = int(parts[1])
                bbox_width  = int(parts[2])
                bbox_height = int(parts[3])
                category    = int(parts[5])
            except ValueError:
                skipped += 1
                continue

            # Skip ignored / others / crowd
            if category in SKIP_CATEGORIES:
                skipped += 1
                continue

            # Skip degenerate boxes
            if bbox_width <= 0 or bbox_height <= 0:
                skipped += 1
                continue

            # YOLO class id is 0-indexed (category 1 -> class 0)
            class_id = category - 1

            x_center = (bbox_left + bbox_width  / 2.0) / img_w
            y_center = (bbox_top  + bbox_height / 2.0) / img_h
            w_norm   = bbox_width  / img_w
            h_norm   = bbox_height / img_h

            # Clamp to valid range
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm   = max(0.0, min(1.0, w_norm))
            h_norm   = max(0.0, min(1.0, h_norm))

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

    return yolo_lines, skipped


# -----------------------------------------------------------------------
# Per-split processing
# -----------------------------------------------------------------------

def process_split(split_name: str, split_dir_name: str):
    """
    Convert all annotations for one dataset split.

    Args:
        split_name:     Human-readable name (e.g. "train")
        split_dir_name: Folder name under DATASET_ROOT

    Returns:
        (images_processed, labels_written, total_skipped)
    """
    split_dir  = DATASET_ROOT / split_dir_name
    images_dir = split_dir / "images"
    ann_dir    = split_dir / "annotations"
    labels_dir = split_dir / "labels"

    # Hard-fail if source folders are missing
    if not images_dir.exists():
        raise FileNotFoundError(
            f"[{split_name}] Images folder not found: {images_dir}"
        )
    if not ann_dir.exists():
        raise FileNotFoundError(
            f"[{split_name}] Annotations folder not found: {ann_dir}"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        raise FileNotFoundError(
            f"[{split_name}] No .txt annotation files found in: {ann_dir}"
        )

    images_processed = 0
    labels_written   = 0
    total_skipped    = 0

    for ann_file in ann_files:
        stem = ann_file.stem

        # Find matching image (jpg is standard for VisDrone)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.warning(f"[{split_name}] No image found for annotation: {ann_file.name} — skipping")
            continue

        try:
            img_w, img_h = get_image_wh(img_path)
        except IOError as e:
            logger.warning(f"[{split_name}] {e} — skipping")
            continue

        yolo_lines, skipped = convert_annotation_file(ann_file, img_w, img_h)
        total_skipped += skipped

        # Always write the label file (even if empty — YOLO expects it)
        out_path = labels_dir / (stem + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        images_processed += 1
        if yolo_lines:
            labels_written += 1

    logger.info(
        f"[{split_name}] images processed: {images_processed} | "
        f"label files with annotations: {labels_written} | "
        f"annotations skipped (cat 0/11/12 or invalid): {total_skipped}"
    )
    return images_processed, labels_written, total_skipped


# -----------------------------------------------------------------------
# YAML generation
# -----------------------------------------------------------------------

def generate_yaml():
    """Write datasets/VisDrone.yaml for Ultralytics training."""
    # Use a relative path so the YAML works regardless of OS/user
    content = (
        "# VisDrone2019-DET dataset — generated by dataset_prep.py\n"
        f"path: {DATASET_ROOT}\n"
        f"train: VisDrone2019-DET-train/images\n"
        f"val:   VisDrone2019-DET-val/images\n"
        f"test:  VisDrone2019-DET-test-dev/images\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(YAML_OUT, "w") as f:
        f.write(content)
    logger.info(f"Dataset YAML written to: {YAML_OUT}")


# -----------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------

def verify_labels():
    """Count label files in each split and raise if any split has 0."""
    all_ok = True
    for split_name, split_dir_name in SPLITS.items():
        labels_dir = DATASET_ROOT / split_dir_name / "labels"
        if not labels_dir.exists():
            logger.error(f"[{split_name}] labels/ folder does not exist: {labels_dir}")
            all_ok = False
            continue
        count = len(list(labels_dir.glob("*.txt")))
        logger.info(f"[{split_name}] label files in {labels_dir}: {count}")
        if count == 0:
            logger.error(
                f"[{split_name}] CONVERSION FAILED — 0 label files written to {labels_dir}"
            )
            all_ok = False

    if not all_ok:
        raise RuntimeError(
            "Dataset preparation failed: one or more splits have 0 label files. "
            "Check the logs above for details."
        )
    logger.info("Verification passed — all splits have label files.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Starting VisDrone dataset preparation")
    logger.info(f"Dataset root: {DATASET_ROOT}")
    logger.info("=" * 60)

    for split_name, split_dir_name in SPLITS.items():
        split_path = DATASET_ROOT / split_dir_name
        if not split_path.exists():
            logger.warning(f"Split folder not found, skipping: {split_path}")
            continue
        process_split(split_name, split_dir_name)

    generate_yaml()
    verify_labels()

    logger.info("=" * 60)
    logger.info("Dataset preparation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
