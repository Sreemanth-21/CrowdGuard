"""
Baseline Evaluation: Pretrained YOLOv8n on VisDrone val set.

Loads the stock pretrained YOLOv8n (no fine-tuning) and evaluates it
on the VisDrone validation set, saving metrics to results/baseline_metrics.json.

Run: python backend/ml/evaluate_baseline.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from ultralytics import YOLO
from backend.utils.logger import get_logger

logger = get_logger(__name__)

YAML_PATH = Path("datasets/VisDrone.yaml")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "baseline_metrics.json"


def evaluate_baseline() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating pretrained YOLOv8n baseline on device: {device}")

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {YAML_PATH}. "
            "Run dataset_prep.py first."
        )

    model = YOLO("yolov8n.pt")
    logger.info("Loaded pretrained YOLOv8n weights (no fine-tuning)")

    metrics = model.val(
        data=str(YAML_PATH),
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        split="val",
    )

    results = {
        "model": "pretrained_yolov8n",
        "weights": "yolov8n.pt",
        "dataset": "VisDrone2019-DET-val",
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(
            2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-9)
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Baseline metrics saved to {OUTPUT_FILE}")
    logger.info(
        f"mAP@0.5={results['map50']:.4f}  "
        f"mAP@0.5:0.95={results['map50_95']:.4f}  "
        f"P={results['precision']:.4f}  R={results['recall']:.4f}  "
        f"F1={results['f1']:.4f}"
    )
    return results


if __name__ == "__main__":
    evaluate_baseline()
