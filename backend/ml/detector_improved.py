"""
Improved YOLOv8 Detector with Soft-NMS post-processing.

Extends the base Detector with optional Soft-NMS (Gaussian weighting,
sigma=0.5) to reduce duplicate detections in dense crowds without
hard-suppressing overlapping boxes.

Usage:
    from backend.ml.detector_improved import ImprovedDetector
    detector = ImprovedDetector(use_soft_nms=True)
    detections = detector.detect(frame)

The base detector.py is NOT modified.
"""

import numpy as np
import torch
from typing import List
from ultralytics import YOLO

from backend.ml.detector import Detection, Detector
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.25,
) -> np.ndarray:
    """
    Gaussian Soft-NMS (Bodla et al., 2017).

    Instead of hard-suppressing overlapping boxes, decays their scores
    using a Gaussian function of IoU overlap.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        scores: (N,) confidence scores
        sigma: Gaussian decay parameter
        score_threshold: Minimum score to keep a box

    Returns:
        Indices of kept boxes (sorted by descending score after decay).
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    scores = scores.copy()
    N = len(boxes)
    indices = np.arange(N)
    kept = []

    for _ in range(N):
        # Pick highest-scoring remaining box
        remaining = [i for i in indices if scores[i] > score_threshold]
        if not remaining:
            break
        best_idx = remaining[int(np.argmax(scores[remaining]))]
        kept.append(best_idx)

        bx1, by1, bx2, by2 = boxes[best_idx]
        b_area = max(0, bx2 - bx1) * max(0, by2 - by1)

        for i in remaining:
            if i == best_idx:
                continue
            ix1 = max(bx1, boxes[i][0])
            iy1 = max(by1, boxes[i][1])
            ix2 = min(bx2, boxes[i][2])
            iy2 = min(by2, boxes[i][3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a_area = max(0, boxes[i][2] - boxes[i][0]) * max(0, boxes[i][3] - boxes[i][1])
            union = b_area + a_area - inter
            iou = inter / (union + 1e-9)
            # Gaussian decay
            scores[i] *= np.exp(-(iou ** 2) / sigma)

        # Remove best_idx from future consideration
        indices = np.array([i for i in indices if i != best_idx])

    return np.array(kept, dtype=np.int32)


class ImprovedDetector(Detector):
    """
    YOLOv8 detector with optional Soft-NMS post-processing.

    Inherits all behaviour from Detector. When use_soft_nms=True,
    replaces the default hard-NMS output with Soft-NMS filtered results.
    """

    SOFT_NMS_SIGMA = 0.5
    SOFT_NMS_SCORE_THRESHOLD = 0.25

    def __init__(
        self,
        model_variant: str = "nano",
        confidence_threshold: float = 0.5,
        use_soft_nms: bool = True,
    ):
        """
        Args:
            model_variant: YOLOv8 variant ("nano", "small", "medium")
            confidence_threshold: Minimum confidence for initial detections
            use_soft_nms: If True, apply Soft-NMS after YOLO inference
        """
        self.use_soft_nms = use_soft_nms
        super().__init__(model_variant=model_variant, confidence_threshold=confidence_threshold)
        logger.info(f"ImprovedDetector initialized: use_soft_nms={use_soft_nms}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons with optional Soft-NMS post-processing.

        If use_soft_nms=False, delegates entirely to the parent detect().
        If use_soft_nms=True, runs raw inference (no built-in NMS filtering
        on confidence), then applies Soft-NMS manually.
        """
        if not self.use_soft_nms:
            return super().detect(frame)

        if self.model is None:
            logger.error("Model not loaded.")
            return []
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received.")
            return []

        try:
            # Run inference; agnostic_nms=False keeps per-class boxes
            results = self.model(
                frame,
                verbose=False,
                device=self.device,
                conf=self.confidence_threshold,
                iou=0.7,  # loose IoU for initial NMS; Soft-NMS refines further
            )

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return []

            result = results[0]
            boxes_raw = result.boxes.xyxy.cpu().numpy()
            confs_raw = result.boxes.conf.cpu().numpy()
            classes_raw = result.boxes.cls.cpu().numpy()

            # Filter person class only
            person_mask = classes_raw.astype(int) == self.PERSON_CLASS_ID
            boxes_p = boxes_raw[person_mask]
            confs_p = confs_raw[person_mask]

            if len(boxes_p) == 0:
                return []

            # Apply Soft-NMS
            kept_idx = soft_nms(
                boxes_p,
                confs_p,
                sigma=self.SOFT_NMS_SIGMA,
                score_threshold=self.SOFT_NMS_SCORE_THRESHOLD,
            )

            detections = []
            for i in kept_idx:
                x1, y1, x2, y2 = map(int, boxes_p[i])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confs_p[i]),
                        centroid=(cx, cy),
                    )
                )

            return detections

        except Exception as e:
            logger.error(f"ImprovedDetector inference failed: {e}")
            return []
