"""
Video Processor for CrowdGuard.

This module orchestrates the complete ML pipeline for video processing.
It integrates all ML components (Detector, Tracker, HeatmapGenerator,
AnomalyEngine, RiskScorer) and manages video capture from webcam or file.

**Validates: Requirements 1.1, 2.1, 33.1**
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from backend.ml.detector import Detector, Detection
from backend.ml.tracker import CentroidTracker, TrackedPerson
from backend.ml.heatmap import HeatmapGenerator, Heatmap
from backend.ml.anomaly_engine import AnomalyEngine, Anomaly
from backend.ml.risk_scorer import RiskScorer, RiskScore
from backend.config import ConfigManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedFrame:
    """
    Represents a fully processed video frame with all analysis results.
    
    Attributes:
        annotated_frame: Frame with bounding boxes and overlays
        person_count: Number of detected persons
        tracked_persons: Dictionary mapping track_id to TrackedPerson
        heatmap: Heatmap object with density visualization
        anomalies: List of detected anomalies
        risk_score: Composite risk score and classification
        timestamp: Frame processing timestamp
    """
    annotated_frame: np.ndarray
    person_count: int
    tracked_persons: Dict[int, TrackedPerson]
    heatmap: Heatmap
    anomalies: list[Anomaly]
    risk_score: RiskScore
    timestamp: datetime


class VideoProcessor:
    """
    Main video processing pipeline orchestrator.
    
    Integrates all ML components to process video frames through the complete
    pipeline: detection → tracking → heatmap generation → anomaly detection
    → risk scoring → frame annotation.
    
    Supports both webcam and video file input sources.
    
    **Validates: Requirements 1.1, 2.1, 33.1**
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize video processor with all ML components.
        
        Loads configuration and initializes:
        - YOLOv8 Detector for person detection
        - CentroidTracker for person tracking
        - HeatmapGenerator for density visualization
        - AnomalyEngine for anomaly detection
        - RiskScorer for risk assessment
        
        Args:
            config_manager: Configuration manager instance
            
        **Validates: Requirements 1.1, 2.1**
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Initialize ML components
        logger.info("Initializing ML components...")
        
        # 1. Detector - YOLOv8 person detection
        self.detector = Detector(
            model_variant=self.config.get("model_variant", "nano"),
            confidence_threshold=self.config.get("confidence_threshold", 0.20)
        )
        logger.info("✓ Detector initialized")
        
        # 2. Tracker - Centroid-based tracking
        self.tracker = CentroidTracker(
            max_distance=80,  # Maximum distance for centroid matching
            max_disappeared=10  # Maximum frames before track removal
        )
        logger.info("✓ Tracker initialized")
        
        # 3. HeatmapGenerator - Density visualization
        self.heatmap_generator = HeatmapGenerator(
            grid_size=(10, 10)
        )
        logger.info("✓ HeatmapGenerator initialized")
        
        # 4. AnomalyEngine - Anomaly detection
        self.anomaly_engine = AnomalyEngine(config=self.config)
        logger.info("✓ AnomalyEngine initialized")
        
        # 5. RiskScorer - Risk assessment
        self.risk_scorer = RiskScorer()
        logger.info("✓ RiskScorer initialized")
        
        # Video capture state
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.source_type: Optional[str] = None  # "webcam" or "upload"
        self.source_name: Optional[str] = None
        self.is_active: bool = False
        
        # Frame processing state
        self.frame_count: int = 0
        self.last_heatmap_update: datetime = datetime.now()
        self.heatmap_update_interval: float = 2.0  # Update heatmap every 2 seconds
        self.cached_heatmap: Optional[Heatmap] = None
        
        logger.info("VideoProcessor initialized successfully")
    
    def start_session(self, source_type: str, source_name: str) -> bool:
        """
        Start video processing session.
        
        Initializes video capture from webcam or file and prepares
        the processor for frame processing.
        
        Args:
            source_type: Type of video source ("webcam" or "upload")
            source_name: Webcam index (e.g., "0") or video file path
            
        Returns:
            True if session started successfully, False otherwise
            
        **Validates: Requirements 1.1, 1.2**
        """
        if self.is_active:
            logger.warning("Session already active. Stop current session first.")
            return False
        
        logger.info(f"Starting session: type={source_type}, source={source_name}")
        
        try:
            # Initialize video capture based on source type
            if source_type == "webcam":
                # Webcam source - convert source_name to integer index
                webcam_index = int(source_name) if source_name.isdigit() else 0
                self.video_capture = cv2.VideoCapture(webcam_index)
                logger.info(f"Opening webcam: index={webcam_index}")
                
            elif source_type == "upload":
                # Video file source
                self.video_capture = cv2.VideoCapture(source_name)
                logger.info(f"Opening video file: {source_name}")
                
            else:
                logger.error(f"Unsupported source type: {source_type}")
                return False
            
            # Verify video capture opened successfully
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open video source: {source_name}")
                self.video_capture = None
                return False
            
            # Set session state
            self.source_type = source_type
            self.source_name = source_name
            self.is_active = True
            self.frame_count = 0
            
            # Reset ML component state
            self.tracker.reset()
            self.anomaly_engine.reset()
            self.cached_heatmap = None
            self.last_heatmap_update = datetime.now()
            
            logger.info("Session started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            return False
    
    def stop_session(self) -> bool:
        """
        Stop active video processing session.
        
        Releases video capture resources and resets processor state.
        
        Returns:
            True if session stopped successfully, False otherwise
        """
        if not self.is_active:
            logger.warning("No active session to stop")
            return False
        
        logger.info("Stopping session...")
        
        try:
            # Release video capture
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            
            # Reset state
            self.is_active = False
            self.source_type = None
            self.source_name = None
            self.frame_count = 0
            self.cached_heatmap = None
            
            logger.info("Session stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping session: {e}")
            return False
    
    def process_frame(self) -> Optional[ProcessedFrame]:
        """
        Process a single frame through the complete ML pipeline.
        
        Pipeline stages:
        1. Capture frame from video source
        2. Run YOLOv8 person detection
        3. Update centroid tracker
        4. Generate heatmap (every 2 seconds)
        5. Detect anomalies
        6. Compute risk score
        7. Annotate frame with bounding boxes and overlays
        
        Returns:
            ProcessedFrame object with all analysis results, or None if
            frame capture fails or session is not active
            
        **Validates: Requirements 2.1, 33.1**
        """
        if not self.is_active or not self.video_capture:
            logger.warning("Cannot process frame: session not active")
            return None
        
        # Capture frame from video source
        ret, frame = self.video_capture.read()
        
        if not ret or frame is None:
            logger.warning("Failed to capture frame from video source")
            return None
        
        timestamp = datetime.now()
        self.frame_count += 1
        
        try:
            # Stage 1: Person Detection
            detections = self.detector.detect(frame)
            logger.debug(f"Frame {self.frame_count}: {len(detections)} persons detected")
            
            # Stage 2: Person Tracking
            tracked_persons = self.tracker.update(detections)
            person_count = len(tracked_persons)
            
            # Compute mean velocity
            velocities = [person.velocity for person in tracked_persons.values()]
            mean_velocity = float(np.mean(velocities)) if velocities else 0.0
            
            # Stage 3: Heatmap Generation (every 2 seconds)
            frame_height, frame_width = frame.shape[:2]
            
            # Check if it's time to update heatmap
            time_since_last_update = (timestamp - self.last_heatmap_update).total_seconds()
            if time_since_last_update >= self.heatmap_update_interval or self.cached_heatmap is None:
                self.cached_heatmap = self.heatmap_generator.create_heatmap_object(
                    tracked_persons=tracked_persons,
                    frame_shape=(frame_height, frame_width)
                )
                self.last_heatmap_update = timestamp
                logger.debug(f"Heatmap updated: density={self.cached_heatmap.density:.3f}")
            
            heatmap = self.cached_heatmap
            
            # Stage 4: Anomaly Detection
            anomalies = self.anomaly_engine.detect_anomalies(
                tracked_persons=tracked_persons,
                density=heatmap.density,
                timestamp=timestamp
            )
            logger.debug(f"Frame {self.frame_count}: {len(anomalies)} anomalies detected")
            
            # Stage 5: Risk Scoring
            risk_score = self.risk_scorer.compute_risk(
                density=heatmap.density,
                mean_velocity=mean_velocity,
                anomalies=anomalies
            )
            logger.debug(
                f"Frame {self.frame_count}: risk_score={risk_score.score:.2f} "
                f"({risk_score.level})"
            )
            
            # Stage 6: Frame Annotation
            annotated_frame = self._annotate_frame(
                frame=frame,
                detections=detections,
                tracked_persons=tracked_persons,
                heatmap=heatmap,
                anomalies=anomalies,
                risk_score=risk_score
            )
            
            # Create ProcessedFrame result
            processed_frame = ProcessedFrame(
                annotated_frame=annotated_frame,
                person_count=person_count,
                tracked_persons=tracked_persons,
                heatmap=heatmap,
                anomalies=anomalies,
                risk_score=risk_score,
                timestamp=timestamp
            )
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            return None
    
    def _annotate_frame(self,
                       frame: np.ndarray,
                       detections: list[Detection],
                       tracked_persons: Dict[int, TrackedPerson],
                       heatmap: Heatmap,
                       anomalies: list[Anomaly],
                       risk_score: RiskScore) -> np.ndarray:
        """
        Annotate frame with bounding boxes, heatmap overlay, and metadata.
        
        Args:
            frame: Original video frame
            detections: List of person detections
            tracked_persons: Dictionary of tracked persons
            heatmap: Heatmap object with overlay
            anomalies: List of detected anomalies
            risk_score: Computed risk score
            
        Returns:
            Annotated frame with all visualizations
        """
        # Create a copy to avoid modifying original
        annotated = frame.copy()
        
        # Apply heatmap overlay with configured opacity
        heatmap_opacity = self.config.get("heatmap_opacity", 0.6)
        if heatmap.overlay is not None:
            annotated = cv2.addWeighted(
                annotated, 
                1.0 - heatmap_opacity,
                heatmap.overlay,
                heatmap_opacity,
                0
            )
        
        # Draw bounding boxes for tracked persons
        for tracked_person in tracked_persons.values():
            x1, y1, x2, y2 = tracked_person.bbox
            
            # Color: green for normal, red for anomaly zones
            # For now, use green (anomaly zone detection would require spatial analysis)
            color = (0, 255, 0)  # Green in BGR
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID:{tracked_person.track_id}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Draw anomaly markers
        for anomaly in anomalies:
            x, y = anomaly.location
            
            # Draw circle at anomaly location
            cv2.circle(annotated, (x, y), 20, (0, 0, 255), 2)  # Red circle
            
            # Draw anomaly type label
            cv2.putText(
                annotated,
                anomaly.type,
                (x - 40, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        return annotated
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update processor configuration at runtime.
        
        Updates configuration and applies changes to ML components.
        
        Args:
            config_updates: Dictionary of configuration parameters to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update configuration
            self.config = self.config_manager.update_config(config_updates)
            
            # Apply updates to ML components
            if "confidence_threshold" in config_updates:
                self.detector.update_threshold(config_updates["confidence_threshold"])
            
            if any(key in config_updates for key in [
                "high_density_threshold", "rapid_movement_threshold",
                "sudden_dispersal_threshold", "crowd_surge_threshold",
                "stationary_crowd_threshold", "stationary_velocity_threshold",
                "stationary_duration_seconds", "fighting_iou_threshold",
                "fighting_velocity_threshold"
            ]):
                self.anomaly_engine.update_config(config_updates)
            
            logger.info(f"Configuration updated: {config_updates}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current processor status.
        
        Returns:
            Dictionary containing processor state and statistics
        """
        return {
            "is_active": self.is_active,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "frame_count": self.frame_count,
            "detector_info": self.detector.get_model_info(),
            "tracker_count": self.tracker.get_track_count(),
            "heatmap_grid": self.heatmap_generator.get_grid_info()
        }
