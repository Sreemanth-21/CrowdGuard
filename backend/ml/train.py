"""
Fine-tune YOLOv8n on VisDrone dataset for CrowdGuard.

Training config:
  - Base weights: pretrained yolov8n.pt
  - Epochs: 50, imgsz: 640, batch: 16
  - Optimizer: AdamW, patience: 10 (early stopping)
  - Augmentation: mosaic, mixup, flips, HSV jitter
  - Best weights saved to: weights/crowdguard_visdrone.pt
  - Training metrics saved to: results/training_metrics.json

Run: python backend/ml/train.py
"""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from ultralytics import YOLO
from backend.utils.logger import get_logger

logger = get_logger(__name__)

YAML_PATH = Path("datasets/VisDrone.yaml")
WEIGHTS_DIR = Path("weights")
OUTPUT_WEIGHTS = WEIGHTS_DIR / "crowdguard_visdrone.pt"
RESULTS_DIR = Path("results")
OUTPUT_METRICS = RESULTS_DIR / "training_metrics.json"


def train() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting YOLOv8n fine-tuning on device: {device}")

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {YAML_PATH}. "
            "Run dataset_prep.py first."
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    logger.info("Loaded pretrained YOLOv8n as starting point")

    results = model.train(
        data=str(YAML_PATH),
        epochs=50,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        patience=10,
        device=device,
        project="runs/train",
        name="crowdguard_visdrone",
        exist_ok=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
    )

    # Copy best weights to canonical location
    best_pt = Path("runs/train/crowdguard_visdrone/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, OUTPUT_WEIGHTS)
        logger.info(f"Best weights saved to {OUTPUT_WEIGHTS}")
    else:
        logger.warning(f"best.pt not found at {best_pt}; check training output.")

    # Extract and save metrics
    metrics_dict = {
        "model": "finetuned_yolov8n",
        "weights": str(OUTPUT_WEIGHTS),
        "epochs_trained": int(results.epoch) if hasattr(results, "epoch") else 50,
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
    }
    metrics_dict["f1"] = float(
        2 * metrics_dict["precision"] * metrics_dict["recall"]
        / (metrics_dict["precision"] + metrics_dict["recall"] + 1e-9)
    )

    # Also capture per-epoch loss history if available
    if hasattr(results, "results_dict"):
        metrics_dict["final_box_loss"] = float(
            results.results_dict.get("train/box_loss", 0.0)
        )
        metrics_dict["final_cls_loss"] = float(
            results.results_dict.get("train/cls_loss", 0.0)
        )
        metrics_dict["final_dfl_loss"] = float(
            results.results_dict.get("train/dfl_loss", 0.0)
        )

    with open(OUTPUT_METRICS, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Training metrics saved to {OUTPUT_METRICS}")
    logger.info(
        f"mAP@0.5={metrics_dict['map50']:.4f}  "
        f"mAP@0.5:0.95={metrics_dict['map50_95']:.4f}  "
        f"P={metrics_dict['precision']:.4f}  R={metrics_dict['recall']:.4f}  "
        f"F1={metrics_dict['f1']:.4f}"
    )
    return metrics_dict


if __name__ == "__main__":
    train()
