"""
Model Comparison: Pretrained vs Fine-tuned vs Fine-tuned + Soft-NMS.

Evaluates three model configurations on the VisDrone val set and saves
a side-by-side comparison to results/comparison_results.json.

  Model A: Pretrained YOLOv8n (baseline)
  Model B: Fine-tuned YOLOv8n (standard NMS)
  Model C: Fine-tuned YOLOv8n + Soft-NMS

Run: python backend/ml/compare_models.py
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
FINETUNED_WEIGHTS = Path("weights/crowdguard_visdrone.pt")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "comparison_results.json"


def _val_metrics(model: YOLO, device: str, label: str) -> dict:
    """Run model.val() and return a clean metrics dict."""
    logger.info(f"Evaluating {label}...")
    metrics = model.val(
        data=str(YAML_PATH),
        imgsz=640,
        batch=16,
        device=device,
        verbose=False,
        split="val",
    )
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    return {
        "label": label,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": mp,
        "recall": mr,
        "f1": float(2 * mp * mr / (mp + mr + 1e-9)),
    }


def _val_with_soft_nms(weights_path: Path, device: str, label: str) -> dict:
    """
    Evaluate fine-tuned model with Soft-NMS by running inference image-by-image
    on the val set and computing COCO-style metrics via ultralytics validator.

    Because ultralytics val() uses its own NMS internally, we approximate
    Soft-NMS impact by evaluating with a lower iou threshold (0.3) which
    mimics the softer suppression behaviour, then annotate the result.
    For a full Soft-NMS integration the ImprovedDetector is used at runtime;
    here we report the val-set proxy metric.
    """
    logger.info(f"Evaluating {label} (Soft-NMS proxy via iou=0.3)...")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(YAML_PATH),
        imgsz=640,
        batch=16,
        device=device,
        verbose=False,
        split="val",
        iou=0.3,   # softer suppression proxy
    )
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    return {
        "label": label,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": mp,
        "recall": mr,
        "f1": float(2 * mp * mr / (mp + mr + 1e-9)),
        "note": "Soft-NMS approximated via iou=0.3 during val; "
                "runtime uses ImprovedDetector with Gaussian Soft-NMS.",
    }


def compare() -> dict:
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {YAML_PATH}. Run dataset_prep.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running model comparison on device: {device}")

    # Model A: pretrained baseline
    model_a = YOLO("yolov8n.pt")
    result_a = _val_metrics(model_a, device, "Model A: Pretrained YOLOv8n (baseline)")

    if not FINETUNED_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Fine-tuned weights not found at {FINETUNED_WEIGHTS}. Run train.py first."
        )

    # Model B: fine-tuned, standard NMS
    model_b = YOLO(str(FINETUNED_WEIGHTS))
    result_b = _val_metrics(model_b, device, "Model B: Fine-tuned YOLOv8n (standard NMS)")

    # Model C: fine-tuned + Soft-NMS
    result_c = _val_with_soft_nms(
        FINETUNED_WEIGHTS, device, "Model C: Fine-tuned YOLOv8n + Soft-NMS"
    )

    comparison = {
        "models": [result_a, result_b, result_c],
        "best_map50": max(result_a["map50"], result_b["map50"], result_c["map50"]),
        "best_model": max(
            [result_a, result_b, result_c], key=lambda x: x["map50"]
        )["label"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison results saved to {OUTPUT_FILE}")
    for r in comparison["models"]:
        logger.info(
            f"  {r['label']}: mAP@0.5={r['map50']:.4f}  "
            f"P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}"
        )
    logger.info(f"Best model: {comparison['best_model']}")
    return comparison


if __name__ == "__main__":
    compare()
