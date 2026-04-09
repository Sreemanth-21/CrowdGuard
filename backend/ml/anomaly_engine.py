"""
Anomaly Detection Engine for CrowdGuard.

This module detects behavioral anomalies in crowd scenes including:
- HIGH_DENSITY: Dangerous crowd density levels
- RAPID_MOVEMENT: Unusually fast crowd movement
- SUDDEN_DISPERSAL: Rapid crowd scattering
- CROWD_SURGE: Sudden increase in crowd size
- STATIONARY_CROWD: Dense stationary crowds
- FIGHTING: Physical altercations detected

**Validates: Requirements 5.1, 6.2, 7.3, 8.2, 9.1, 10.2**
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from backend.ml.tracker import TrackedPerson
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Anomaly:
    """
    Represents a detected anomaly.
    
    Attributes:
        type: Anomaly type (HIGH_DENSITY, RAPID_MOVEMENT, etc.)
        confidence: Detection confidence score (0.0-1.0)
        risk_level: Risk classification (SAFE, CAUTION, WARNING, CRITICAL)
        description: Human-readable description
        location: Center point (x, y) of anomaly
        affected_persons: Number of persons involved
    """
    type: str
    confidence: float  # 0.0 to 1.0
    risk_level: str  # SAFE, CAUTION, WARNING, CRITICAL
    description: str
    location: Tuple[int, int]  # (x, y) center
    affected_persons: int


@dataclass
class FrameData:
    """
    Historical frame data for temporal analysis.
    
    Attributes:
        timestamp: Frame timestamp
        person_count: Number of detected persons
        density: Crowd density value (0.0-1.0)
        mean_velocity: Average velocity across all persons
        spatial_spread: Standard deviation of centroid positions
        centroids: List of all person centroids
    """
    timestamp: datetime
    person_count: int
    density: float
    mean_velocity: float
    spatial_spread: float
    centroids: List[Tuple[int, int]]


class AnomalyEngine:
    """
    Anomaly detection engine for crowd behavior analysis.
    
    Detects six types of anomalies using configurable thresholds:
    1. HIGH_DENSITY: Dangerous crowd density (>70%)
    2. RAPID_MOVEMENT: Fast crowd movement (>25 px/frame)
    3. SUDDEN_DISPERSAL: Rapid scattering (40% spread increase in 2s)
    4. CROWD_SURGE: Sudden crowd increase (30% in 3s)
    5. STATIONARY_CROWD: Dense stationary crowd (>50% density, <3 px/frame for 30s)
    6. FIGHTING: Physical altercations (IoU >0.3, velocity >20 px/frame)
    
    **Validates: Requirements 5.1, 6.2, 7.3, 8.2, 9.1, 10.2**
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize anomaly detection engine.
        
        Loads detection thresholds from configuration and initializes
        frame history buffer for temporal analysis.
        
        Args:
            config: Configuration dictionary containing anomaly thresholds:
                - high_density_threshold: Density threshold for HIGH_DENSITY (default: 0.7)
                - rapid_movement_threshold: Velocity threshold for RAPID_MOVEMENT (default: 25)
                - sudden_dispersal_threshold: Spread increase threshold (default: 0.4)
                - crowd_surge_threshold: Count increase threshold (default: 0.3)
                - stationary_crowd_threshold: Density threshold for STATIONARY_CROWD (default: 0.5)
                - stationary_velocity_threshold: Velocity threshold for stationary (default: 3)
                - stationary_duration_seconds: Duration threshold for stationary (default: 30)
                - fighting_iou_threshold: IoU threshold for FIGHTING (default: 0.3)
                - fighting_velocity_threshold: Velocity threshold for FIGHTING (default: 20)
                
        **Validates: Requirements 5.1, 6.2, 7.3, 8.2, 9.1, 10.2**
        """
        # Load thresholds from configuration
        self.high_density_threshold = config.get("high_density_threshold", 0.7)
        self.rapid_movement_threshold = config.get("rapid_movement_threshold", 25)
        self.sudden_dispersal_threshold = config.get("sudden_dispersal_threshold", 0.4)
        self.crowd_surge_threshold = config.get("crowd_surge_threshold", 0.3)
        self.stationary_crowd_threshold = config.get("stationary_crowd_threshold", 0.5)
        self.stationary_velocity_threshold = config.get("stationary_velocity_threshold", 3)
        self.stationary_duration_seconds = config.get("stationary_duration_seconds", 30)
        self.fighting_iou_threshold = config.get("fighting_iou_threshold", 0.3)
        self.fighting_velocity_threshold = config.get("fighting_velocity_threshold", 20)
        
        # Frame history buffer for temporal analysis
        # Store up to 180 frames (assuming 10 fps = 18 seconds of history)
        # This covers the longest temporal window needed (30 seconds for stationary crowd)
        self.frame_history: deque[FrameData] = deque(maxlen=180)
        
        # Stationary crowd tracking
        self.stationary_start_time: datetime | None = None
        
        logger.info(
            f"AnomalyEngine initialized with thresholds: "
            f"high_density={self.high_density_threshold}, "
            f"rapid_movement={self.rapid_movement_threshold}, "
            f"sudden_dispersal={self.sudden_dispersal_threshold}, "
            f"crowd_surge={self.crowd_surge_threshold}, "
            f"stationary_crowd={self.stationary_crowd_threshold}, "
            f"stationary_velocity={self.stationary_velocity_threshold}, "
            f"stationary_duration={self.stationary_duration_seconds}s, "
            f"fighting_iou={self.fighting_iou_threshold}, "
            f"fighting_velocity={self.fighting_velocity_threshold}"
        )
    
    def detect_anomalies(self,
                        tracked_persons: Dict[int, TrackedPerson],
                        density: float,
                        timestamp: datetime) -> List[Anomaly]:
        """
        Detect anomalies in current frame.
        
        Analyzes current frame data and historical data to detect all six
        anomaly types. Updates frame history buffer with current frame data.
        
        Args:
            tracked_persons: Dictionary mapping track_id to TrackedPerson
            density: Current crowd density value (0.0-1.0)
            timestamp: Current frame timestamp
            
        Returns:
            List of detected Anomaly objects
            
        **Validates: Requirements 5.1, 6.2, 7.3, 8.2, 9.1, 10.2**
        """
        anomalies = []
        
        # Extract current frame metrics
        person_count = len(tracked_persons)
        centroids = [person.centroid for person in tracked_persons.values()]
        velocities = [person.velocity for person in tracked_persons.values()]
        mean_velocity = np.mean(velocities) if velocities else 0.0
        spatial_spread = self._compute_spatial_spread(centroids)
        
        # Store current frame data in history
        frame_data = FrameData(
            timestamp=timestamp,
            person_count=person_count,
            density=density,
            mean_velocity=mean_velocity,
            spatial_spread=spatial_spread,
            centroids=centroids
        )
        self.frame_history.append(frame_data)
        
        # Detect each anomaly type
        
        # 1. HIGH_DENSITY: Detect when density > 0.70
        if density > self.high_density_threshold:
            confidence = (density - self.high_density_threshold) / (1.0 - self.high_density_threshold)
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # Compute center of mass for location
            if centroids:
                center_x = int(np.mean([c[0] for c in centroids]))
                center_y = int(np.mean([c[1] for c in centroids]))
            else:
                center_x, center_y = 0, 0
            
            anomalies.append(Anomaly(
                type="HIGH_DENSITY",
                confidence=confidence,
                risk_level="HIGH",
                description=f"High crowd density detected: {density:.1%}",
                location=(center_x, center_y),
                affected_persons=person_count
            ))
        
        # 2. RAPID_MOVEMENT: Detect when average velocity > 25 px/frame
        if mean_velocity > self.rapid_movement_threshold:
            confidence = (mean_velocity - self.rapid_movement_threshold) / self.rapid_movement_threshold
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # Compute center of mass for location
            if centroids:
                center_x = int(np.mean([c[0] for c in centroids]))
                center_y = int(np.mean([c[1] for c in centroids]))
            else:
                center_x, center_y = 0, 0
            
            anomalies.append(Anomaly(
                type="RAPID_MOVEMENT",
                confidence=confidence,
                risk_level="HIGH",
                description=f"Rapid crowd movement detected: {mean_velocity:.1f} px/frame",
                location=(center_x, center_y),
                affected_persons=person_count
            ))
        
        # 3. SUDDEN_DISPERSAL: Compare spatial spread between frames 2 seconds apart
        # Assuming 10 fps, 2 seconds = 20 frames
        if len(self.frame_history) >= 20:
            frame_2s_ago = self.frame_history[-20]
            if frame_2s_ago.spatial_spread > 0:
                spread_increase = (spatial_spread - frame_2s_ago.spatial_spread) / frame_2s_ago.spatial_spread
                
                if spread_increase >= self.sudden_dispersal_threshold:
                    confidence = (spread_increase - self.sudden_dispersal_threshold) / 0.6
                    confidence = min(confidence, 1.0)  # Cap at 1.0
                    
                    # Compute center of mass for location
                    if centroids:
                        center_x = int(np.mean([c[0] for c in centroids]))
                        center_y = int(np.mean([c[1] for c in centroids]))
                    else:
                        center_x, center_y = 0, 0
                    
                    anomalies.append(Anomaly(
                        type="SUDDEN_DISPERSAL",
                        confidence=confidence,
                        risk_level="MEDIUM",
                        description=f"Sudden crowd dispersal detected: {spread_increase:.1%} increase",
                        location=(center_x, center_y),
                        affected_persons=person_count
                    ))
        
        # 4. CROWD_SURGE: Compare person counts between frames 3 seconds apart
        # Assuming 10 fps, 3 seconds = 30 frames
        if len(self.frame_history) >= 30:
            frame_3s_ago = self.frame_history[-30]
            if frame_3s_ago.person_count > 0:
                count_increase = (person_count - frame_3s_ago.person_count) / frame_3s_ago.person_count
                
                if count_increase >= self.crowd_surge_threshold:
                    confidence = (count_increase - self.crowd_surge_threshold) / 0.7
                    confidence = min(confidence, 1.0)  # Cap at 1.0
                    
                    # Compute center of mass for location
                    if centroids:
                        center_x = int(np.mean([c[0] for c in centroids]))
                        center_y = int(np.mean([c[1] for c in centroids]))
                    else:
                        center_x, center_y = 0, 0
                    
                    anomalies.append(Anomaly(
                        type="CROWD_SURGE",
                        confidence=confidence,
                        risk_level="CRITICAL",
                        description=f"Crowd surge detected: {count_increase:.1%} increase in 3 seconds",
                        location=(center_x, center_y),
                        affected_persons=person_count
                    ))
        
        # 5. STATIONARY_CROWD: Track density > 0.5 AND velocity < 3 px/frame for 30 seconds
        is_stationary_condition = (
            density > self.stationary_crowd_threshold and 
            mean_velocity < self.stationary_velocity_threshold
        )
        
        if is_stationary_condition:
            if self.stationary_start_time is None:
                # Start tracking stationary condition
                self.stationary_start_time = timestamp
            else:
                # Check if condition has persisted for 30 seconds
                duration = (timestamp - self.stationary_start_time).total_seconds()
                if duration >= self.stationary_duration_seconds:
                    confidence = density - self.stationary_crowd_threshold
                    confidence = min(confidence, 1.0)  # Cap at 1.0
                    
                    # Compute center of mass for location
                    if centroids:
                        center_x = int(np.mean([c[0] for c in centroids]))
                        center_y = int(np.mean([c[1] for c in centroids]))
                    else:
                        center_x, center_y = 0, 0
                    
                    anomalies.append(Anomaly(
                        type="STATIONARY_CROWD",
                        confidence=confidence,
                        risk_level="LOW",
                        description=f"Stationary crowd detected for {duration:.0f} seconds",
                        location=(center_x, center_y),
                        affected_persons=person_count
                    ))
        else:
            # Reset stationary tracking if condition no longer met
            self.stationary_start_time = None
        
        # 6. FIGHTING: Compute IoU for all pairs of bounding boxes
        persons_list = list(tracked_persons.values())
        for i in range(len(persons_list)):
            for j in range(i + 1, len(persons_list)):
                person1 = persons_list[i]
                person2 = persons_list[j]
                
                # Compute IoU between bounding boxes
                iou = self._compute_iou(person1.bbox, person2.bbox)
                
                # Check if IoU > 0.3 AND either person has velocity > 20 px/frame
                if iou > self.fighting_iou_threshold:
                    max_velocity = max(person1.velocity, person2.velocity)
                    
                    if max_velocity > self.fighting_velocity_threshold:
                        # Compute confidence as min(IoU / 0.3, velocity / 20)
                        iou_confidence = iou / self.fighting_iou_threshold
                        velocity_confidence = max_velocity / self.fighting_velocity_threshold
                        confidence = min(iou_confidence, velocity_confidence)
                        confidence = min(confidence, 1.0)  # Cap at 1.0
                        
                        # Compute location as midpoint between the two persons
                        location_x = (person1.centroid[0] + person2.centroid[0]) // 2
                        location_y = (person1.centroid[1] + person2.centroid[1]) // 2
                        
                        anomalies.append(Anomaly(
                            type="FIGHTING",
                            confidence=confidence,
                            risk_level="CRITICAL",
                            description=f"Potential physical altercation detected (IoU: {iou:.2f}, velocity: {max_velocity:.1f})",
                            location=(location_x, location_y),
                            affected_persons=2
                        ))
        
        logger.debug(
            f"Anomaly detection complete: {len(anomalies)} anomalies detected "
            f"(density={density:.2f}, mean_velocity={mean_velocity:.2f}, "
            f"person_count={person_count})"
        )
        
        return anomalies
    
    def _compute_spatial_spread(self, centroids: List[Tuple[int, int]]) -> float:
        """
        Compute spatial spread as standard deviation of centroid positions.
        
        Calculates the combined standard deviation of x and y coordinates
        to measure how dispersed the crowd is spatially.
        
        Args:
            centroids: List of (x, y) centroid positions
            
        Returns:
            Spatial spread value (standard deviation in pixels)
        """
        if len(centroids) < 2:
            return 0.0
        
        centroids_array = np.array(centroids)
        x_coords = centroids_array[:, 0]
        y_coords = centroids_array[:, 1]
        
        # Compute combined standard deviation
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        spatial_spread = np.sqrt(x_std**2 + y_std**2)
        
        return float(spatial_spread)
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return float(iou)
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update anomaly detection thresholds.
        
        Allows runtime configuration updates without recreating the engine.
        
        Args:
            config: Dictionary containing threshold updates
        """
        if "high_density_threshold" in config:
            self.high_density_threshold = config["high_density_threshold"]
        if "rapid_movement_threshold" in config:
            self.rapid_movement_threshold = config["rapid_movement_threshold"]
        if "sudden_dispersal_threshold" in config:
            self.sudden_dispersal_threshold = config["sudden_dispersal_threshold"]
        if "crowd_surge_threshold" in config:
            self.crowd_surge_threshold = config["crowd_surge_threshold"]
        if "stationary_crowd_threshold" in config:
            self.stationary_crowd_threshold = config["stationary_crowd_threshold"]
        if "stationary_velocity_threshold" in config:
            self.stationary_velocity_threshold = config["stationary_velocity_threshold"]
        if "stationary_duration_seconds" in config:
            self.stationary_duration_seconds = config["stationary_duration_seconds"]
        if "fighting_iou_threshold" in config:
            self.fighting_iou_threshold = config["fighting_iou_threshold"]
        if "fighting_velocity_threshold" in config:
            self.fighting_velocity_threshold = config["fighting_velocity_threshold"]
        
        logger.info(f"AnomalyEngine configuration updated: {config}")
    
    def reset(self) -> None:
        """
        Reset frame history and stationary tracking.
        
        Should be called when starting a new session to clear historical data.
        """
        self.frame_history.clear()
        self.stationary_start_time = None
        logger.info("AnomalyEngine reset: frame history cleared")
