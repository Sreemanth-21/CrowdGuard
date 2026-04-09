"""
Risk Scorer for CrowdGuard.

This module computes composite risk scores from crowd metrics including:
- Crowd density (weighted 35%)
- Average velocity (weighted 30%)
- Anomaly confidence (weighted 35%)

The risk scorer normalizes each component to a 0-100 scale and classifies
the overall risk level as SAFE, CAUTION, WARNING, or CRITICAL.

**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 11.11**
"""

from typing import List, Dict
from dataclasses import dataclass
from backend.ml.anomaly_engine import Anomaly
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskScore:
    """
    Represents a computed risk score with classification.
    
    Attributes:
        score: Composite risk score (0-100)
        level: Risk classification (SAFE, CAUTION, WARNING, CRITICAL)
        components: Breakdown of score components (density, velocity, anomaly)
    """
    score: float  # 0 to 100
    level: str  # SAFE, CAUTION, WARNING, CRITICAL
    components: Dict[str, float]  # Breakdown of score components


class RiskScorer:
    """
    Risk scorer for computing composite risk scores from crowd metrics.
    
    Computes a weighted risk score from three components:
    - Density: 35% weight, normalized to 0-100 scale
    - Velocity: 30% weight, normalized to 0-100 scale (capped at 50 px/frame)
    - Anomaly: 35% weight, using highest anomaly confidence
    
    Risk levels are classified as:
    - SAFE: 0-25
    - CAUTION: 26-50
    - WARNING: 51-75
    - CRITICAL: 76-100
    
    **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 11.11**
    """
    
    # Component weights (must sum to 1.0)
    DENSITY_WEIGHT = 0.35
    VELOCITY_WEIGHT = 0.30
    ANOMALY_WEIGHT = 0.35
    
    # Velocity normalization constant (50 px/frame = 100 on normalized scale)
    VELOCITY_NORMALIZATION_FACTOR = 50.0
    
    def __init__(self):
        """
        Initialize risk scorer.
        
        **Validates: Requirements 11.1**
        """
        logger.info(
            f"RiskScorer initialized with weights: "
            f"density={self.DENSITY_WEIGHT}, "
            f"velocity={self.VELOCITY_WEIGHT}, "
            f"anomaly={self.ANOMALY_WEIGHT}"
        )
    
    def compute_risk(self,
                    density: float,
                    mean_velocity: float,
                    anomalies: List[Anomaly]) -> RiskScore:
        """
        Compute composite risk score from crowd metrics.
        
        Normalizes each component to 0-100 scale and computes weighted sum:
        - Density: multiplied by 100 (0.0-1.0 → 0-100)
        - Velocity: (velocity / 50) * 100, capped at 100
        - Anomaly: highest confidence * 100, or 0 if no anomalies
        
        Args:
            density: Current crowd density (0.0-1.0)
            mean_velocity: Average velocity in pixels per frame
            anomalies: List of detected Anomaly objects
            
        Returns:
            RiskScore object with score, level, and component breakdown
            
        **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7**
        """
        # Normalize density to 0-100 scale (Requirement 11.5)
        normalized_density = density * 100.0
        
        # Normalize velocity to 0-100 scale (Requirement 11.6)
        # Formula: (velocity / 50) * 100, capped at 100
        normalized_velocity = min((mean_velocity / self.VELOCITY_NORMALIZATION_FACTOR) * 100.0, 100.0)
        
        # Get highest anomaly confidence and normalize to 0-100 scale (Requirement 11.7)
        if anomalies:
            max_anomaly_confidence = max(anomaly.confidence for anomaly in anomalies)
            normalized_anomaly = max_anomaly_confidence * 100.0
        else:
            normalized_anomaly = 0.0
        
        # Compute weighted composite score (Requirements 11.1, 11.2, 11.3, 11.4)
        composite_score = (
            self.DENSITY_WEIGHT * normalized_density +
            self.VELOCITY_WEIGHT * normalized_velocity +
            self.ANOMALY_WEIGHT * normalized_anomaly
        )
        
        # Ensure score is within bounds [0, 100]
        composite_score = max(0.0, min(100.0, composite_score))
        
        # Classify risk level (Requirements 11.8, 11.9, 11.10, 11.11)
        risk_level = self._classify_risk_level(composite_score)
        
        # Build component breakdown
        components = {
            "density": normalized_density,
            "velocity": normalized_velocity,
            "anomaly": normalized_anomaly
        }
        
        logger.debug(
            f"Risk score computed: {composite_score:.2f} ({risk_level}) - "
            f"density={normalized_density:.2f}, "
            f"velocity={normalized_velocity:.2f}, "
            f"anomaly={normalized_anomaly:.2f}"
        )
        
        return RiskScore(
            score=composite_score,
            level=risk_level,
            components=components
        )
    
    def _classify_risk_level(self, score: float) -> str:
        """
        Classify risk level based on composite score.
        
        Args:
            score: Composite risk score (0-100)
            
        Returns:
            Risk level string (SAFE, CAUTION, WARNING, CRITICAL)
            
        **Validates: Requirements 11.8, 11.9, 11.10, 11.11**
        """
        if score <= 25:
            return "SAFE"  # Requirement 11.8
        elif score <= 50:
            return "CAUTION"  # Requirement 11.9
        elif score <= 75:
            return "WARNING"  # Requirement 11.10
        else:
            return "CRITICAL"  # Requirement 11.11
