"""
YOLOv8 Person Detector for CrowdGuard.

This module provides person detection using the YOLOv8 model from Ultralytics.
It handles model loading, automatic weight downloads, GPU/CPU fallback, and
inference with confidence filtering.

**Validates: Requirements 2.1, 2.8, 2.9**
"""

import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """
    Represents a single person detection.
    
    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        confidence: Detection confidence score (0.0 to 1.0)
        centroid: Center point of bounding box (cx, cy)
    """
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float  # 0.0 to 1.0
    centroid: Tuple[int, int]  # (cx, cy)


class Detector:
    """
    YOLOv8-based person detector.
    
    Detects persons in video frames using YOLOv8 model with configurable
    confidence threshold. Automatically downloads model weights if not present
    and falls back to CPU if GPU is unavailable.
    
    **Validates: Requirements 2.1, 2.8, 2.9**
    """
    
    # COCO class ID for 'person'. When fine-tuned on VisDrone,
    # class 0 = pedestrian and class 1 = people — both count as persons.
    PERSON_CLASS_ID = 0

    # VisDrone person class IDs (pedestrian=0, people=1)
    VISDRONE_PERSON_CLASS_IDS = {0, 1}

    # Keywords that identify a person class by name (case-insensitive)
    PERSON_CLASS_KEYWORDS = {"person", "pedestrian", "people"}
    
    # Model variant mapping
    MODEL_VARIANTS = {
        "nano": "yolov8n.pt",
        "small": "yolov8s.pt",
        "medium": "yolov8m.pt"
    }
    
    def __init__(self, model_variant: str = "nano", confidence_threshold: float = 0.20):
        """
        Initialize YOLOv8 detector.
        
        Loads the specified YOLOv8 model variant and configures detection
        parameters. Automatically downloads model weights if not present.
        Uses GPU if CUDA is available, otherwise falls back to CPU.
        
        Args:
            model_variant: YOLOv8 model variant ("nano", "small", "medium")
            confidence_threshold: Minimum confidence score for detections (0.0 to 1.0)
            
        Raises:
            ValueError: If model_variant is not supported
            RuntimeError: If model loading fails
            
        **Validates: Requirements 2.1, 2.8, 2.9**
        """
        if model_variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model variant: {model_variant}. "
                f"Must be one of: {list(self.MODEL_VARIANTS.keys())}"
            )
        
        self.model_variant = model_variant
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = None
        self._using_finetuned = False  # set properly in _load_model
        
        # Determine device (GPU/CPU)
        self._setup_device()
        
        # Load model
        self._load_model()
        
        logger.info(
            f"Detector initialized: variant={model_variant}, "
            f"threshold={confidence_threshold}, device={self.device}"
        )
    
    def _setup_device(self) -> None:
        """
        Set up computation device (GPU or CPU).
        
        Checks for CUDA availability and falls back to CPU if not available.
        
        **Validates: Requirement 2.9**
        """
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Using CPU for inference.")
    
    # Path to fine-tuned CrowdGuard weights (relative to project root)
    FINETUNED_WEIGHTS = "weights/crowdguard_visdrone.pt"

    def _load_model(self) -> None:
        """
        Load YOLOv8 model weights.

        Always loads pretrained YOLOv8n (COCO) for live webcam/video detection.
        The fine-tuned VisDrone model is intentionally NOT used here because it
        was trained on top-down drone footage and performs poorly on webcam input.
        The fine-tuned weights are used only in the offline evaluation pipeline
        (evaluate_baseline.py, compare_models.py).

        Raises:
            RuntimeError: If model loading fails

        **Validates: Requirements 2.1, 2.8**
        """
        try:
            model_name = self.MODEL_VARIANTS[self.model_variant]
            logger.info(f"Loading pretrained {model_name} for live detection (COCO weights)")
            self.model = YOLO(model_name)
            self._using_finetuned = False

            self.model.to(self.device)
            logger.info(f"Model loaded successfully: {model_name}")
            logger.info(f"Model classes: {self.model.names}")

        except Exception as e:
            error_msg = f"Failed to load YOLOv8 model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load YOLOv8 model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _is_person(self, cls_id: int, cls_name: str) -> bool:
        """
        Return True for any detection that represents a person.

        With pretrained COCO weights: class 0 = 'person'.
        Also accepts by name for robustness.
        """
        if cls_id == 0:
            return True
        return cls_name.lower() in self.PERSON_CLASS_KEYWORDS

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a video frame.
        
        Runs YOLOv8 inference on the input frame, filters for person class,
        applies confidence threshold, and computes centroids.
        
        If inference fails, logs the error and returns an empty list to allow
        processing to continue with subsequent frames.
        
        Args:
            frame: Input frame in BGR format (H, W, 3) as numpy array
            
        Returns:
            List of Detection objects for detected persons above confidence threshold.
            Returns empty list if inference fails or frame is invalid.
            
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.7**
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot perform detection.")
            return []
        
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received. Skipping detection.")
            return []
        
        detections = []

        # Use Soft-NMS when fine-tuned weights are loaded
        use_soft_nms = getattr(self, "_using_finetuned", False)

        try:
            # Pass conf threshold directly to YOLO so it pre-filters low-confidence
            # detections before we even see them — avoids processing noise.
            results = self.model(
                frame,
                verbose=False,
                device=self.device,
                conf=self.confidence_threshold,
            )

            if len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_raw = result.boxes.xyxy.cpu().numpy()
                    confidences_raw = result.boxes.conf.cpu().numpy()
                    classes_raw = result.boxes.cls.cpu().numpy().astype(int)

                    # class name map: {int: str} — present on all Ultralytics results
                    class_names: dict = {}
                    if hasattr(result, "names") and result.names:
                        class_names = result.names

                    # Log every raw detection BEFORE filtering
                    for cls_id, conf in zip(classes_raw, confidences_raw):
                        cls_name = class_names.get(int(cls_id), "unknown")
                        logger.info(f"raw: id={int(cls_id)} name='{cls_name}' conf={float(conf):.3f}")

                    # Apply person filter (confidence already pre-filtered by YOLO)
                    person_mask = np.array(
                        [self._is_person(int(c), class_names.get(int(c), "")) for c in classes_raw],
                        dtype=bool,
                    )

                    boxes_p = boxes_raw[person_mask]
                    confs_p = confidences_raw[person_mask]

                    logger.info(
                        f"Raw detections: {len(boxes_raw)}, "
                        f"filtered persons: {len(boxes_p)} "
                        f"(threshold={self.confidence_threshold}, finetuned={use_soft_nms})"
                    )

                    if use_soft_nms and len(boxes_p) > 0:
                        from backend.ml.detector_improved import soft_nms
                        kept = soft_nms(boxes_p, confs_p, sigma=0.5, score_threshold=0.15)
                        boxes_p = boxes_p[kept]
                        confs_p = confs_p[kept]

                    for box, conf in zip(boxes_p, confs_p):
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            centroid=(cx, cy),
                        ))

        except Exception as e:
            logger.error(f"Detection inference failed: {e}")
            return []

        return detections
    
    def update_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
            
        Raises:
            ValueError: If threshold is not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model variant, device, threshold, and weight source.
        """
        using_finetuned = getattr(self, "_using_finetuned", False)
        return {
            "model_variant": self.model_variant,
            "model_name": self.FINETUNED_WEIGHTS if using_finetuned else self.MODEL_VARIANTS[self.model_variant],
            "finetuned": using_finetuned,
            "soft_nms": using_finetuned,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "cuda_available": torch.cuda.is_available(),
        }
