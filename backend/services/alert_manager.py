"""
Alert Manager Service for CrowdGuard.

This module manages alert generation, deduplication, and persistence.
Alerts are generated when risk scores exceed thresholds or CRITICAL anomalies
are detected, with cooldown periods to prevent duplicate notifications.

**Validates: Requirements 12.1-12.7, 35.1-35.3**
"""

import os
import uuid
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from backend.models.alert import Alert
from backend.ml.anomaly_engine import Anomaly
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AlertManager:
    """
    Manages alert generation, deduplication, and persistence.
    
    The AlertManager enforces cooldown periods per anomaly type to prevent
    duplicate alerts from overwhelming operators. It persists alerts to the
    database and handles frame snapshot saving.
    
    Attributes:
        db: Database session for alert persistence
        cooldown_seconds: Cooldown period in seconds (default: 10)
        snapshot_dir: Directory for saving frame snapshots
        last_alert_times: Tracking of last alert time per anomaly type
    """
    
    def __init__(
        self,
        db: Session,
        cooldown_seconds: int = 10,
        snapshot_dir: str = "snapshots"
    ):
        """
        Initialize AlertManager.
        
        Args:
            db: Database session for persistence
            cooldown_seconds: Cooldown period between alerts of same type
            snapshot_dir: Directory to save frame snapshots
        """
        self.db = db
        self.cooldown_seconds = cooldown_seconds
        self.snapshot_dir = snapshot_dir
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Create snapshot directory if it doesn't exist
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        logger.info(
            f"AlertManager initialized with {cooldown_seconds}s cooldown, "
            f"snapshot_dir={snapshot_dir}"
        )
    
    def should_generate_alert(
        self,
        anomaly: Anomaly,
        risk_score: float
    ) -> bool:
        """
        Determine if an alert should be generated based on thresholds.
        
        Alert generation criteria:
        - Risk score > 50 OR
        - Anomaly risk level is CRITICAL
        
        Args:
            anomaly: Detected anomaly
            risk_score: Composite risk score (0-100)
            
        Returns:
            True if alert should be generated, False otherwise
            
        **Validates: Requirement 12.1**
        """
        # Check risk score threshold
        if risk_score > 50:
            return True
        
        # Check if anomaly is CRITICAL
        if anomaly.risk_level == "CRITICAL":
            return True
        
        return False
    
    def is_within_cooldown(self, anomaly_type: str) -> bool:
        """
        Check if an anomaly type is within its cooldown period.
        
        Args:
            anomaly_type: Type of anomaly to check
            
        Returns:
            True if within cooldown period, False otherwise
            
        **Validates: Requirements 12.3, 12.4, 35.1**
        """
        if anomaly_type not in self.last_alert_times:
            return False
        
        last_alert_time = self.last_alert_times[anomaly_type]
        time_since_last = datetime.now() - last_alert_time
        
        return time_since_last < timedelta(seconds=self.cooldown_seconds)
    
    def save_frame_snapshot(
        self,
        frame: np.ndarray,
        alert_id: str
    ) -> Optional[str]:
        """
        Save frame snapshot to disk.
        
        Args:
            frame: Frame image as numpy array (BGR format)
            alert_id: Unique alert identifier
            
        Returns:
            Path to saved snapshot, or None if save failed
            
        **Validates: Requirement 12.2**
        """
        try:
            filename = f"alert_{alert_id}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            # Save frame as JPEG
            success = cv2.imwrite(filepath, frame)
            
            if success:
                logger.debug(f"Saved frame snapshot: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save frame snapshot: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving frame snapshot: {e}")
            return None
    
    def generate_alert(
        self,
        session_id: str,
        anomaly: Anomaly,
        risk_score: float,
        risk_level: str,
        frame: Optional[np.ndarray] = None
    ) -> Optional[Alert]:
        """
        Generate and persist an alert for a detected anomaly.
        
        This method:
        1. Checks if alert should be generated (risk score > 50 OR CRITICAL)
        2. Checks cooldown period for anomaly type
        3. Creates Alert object with all required fields
        4. Saves frame snapshot to disk
        5. Persists alert to database (with error handling)
        6. Updates cooldown tracking
        
        Args:
            session_id: Current session identifier
            anomaly: Detected anomaly
            risk_score: Composite risk score (0-100)
            risk_level: Risk level classification
            frame: Optional frame image for snapshot
            
        Returns:
            Alert object if generated, None if suppressed or failed
            
        **Validates: Requirements 12.1-12.7, 35.1-35.3**
        """
        # Check if alert should be generated based on thresholds
        if not self.should_generate_alert(anomaly, risk_score):
            logger.debug(
                f"Alert suppressed: {anomaly.type} - "
                f"risk_score={risk_score:.1f}, risk_level={anomaly.risk_level}"
            )
            return None
        
        # Check cooldown period for this anomaly type
        if self.is_within_cooldown(anomaly.type):
            logger.debug(
                f"Alert suppressed due to cooldown: {anomaly.type} - "
                f"last alert was within {self.cooldown_seconds}s"
            )
            return None
        
        # Generate unique alert ID
        alert_id = str(uuid.uuid4())
        
        # Save frame snapshot if provided
        snapshot_path = None
        if frame is not None:
            snapshot_path = self.save_frame_snapshot(frame, alert_id)
        
        # Create alert object
        alert = Alert(
            alert_id=alert_id,
            session_id=session_id,
            timestamp=datetime.now(),
            anomaly_type=anomaly.type,
            risk_level=risk_level,
            confidence_score=anomaly.confidence,
            description=anomaly.description,
            frame_snapshot_path=snapshot_path,
            affected_persons=anomaly.affected_persons,
            location_x=anomaly.location[0],
            location_y=anomaly.location[1],
            is_dismissed=False,
            dismissed_at=None
        )
        
        # Persist to database with error handling
        try:
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            logger.info(
                f"Alert generated: {alert.anomaly_type} - "
                f"risk_level={alert.risk_level}, "
                f"confidence={alert.confidence_score:.2f}, "
                f"alert_id={alert.alert_id}"
            )
            
        except Exception as e:
            # Log error but continue (Requirement 12.7)
            logger.error(f"Failed to persist alert to database: {e}")
            self.db.rollback()
            # Alert object is still returned for WebSocket transmission
        
        # Update cooldown tracking
        self.last_alert_times[anomaly.type] = datetime.now()
        
        return alert
    
    def generate_alerts(
        self,
        session_id: str,
        anomalies: List[Anomaly],
        risk_score: float,
        risk_level: str,
        frame: Optional[np.ndarray] = None
    ) -> List[Alert]:
        """
        Generate alerts for multiple detected anomalies.
        
        Each anomaly type is processed independently with its own cooldown
        tracking, allowing simultaneous alerts for different anomaly types.
        
        Args:
            session_id: Current session identifier
            anomalies: List of detected anomalies
            risk_score: Composite risk score (0-100)
            risk_level: Risk level classification
            frame: Optional frame image for snapshots
            
        Returns:
            List of generated Alert objects
            
        **Validates: Requirement 35.3**
        """
        generated_alerts = []
        
        for anomaly in anomalies:
            alert = self.generate_alert(
                session_id=session_id,
                anomaly=anomaly,
                risk_score=risk_score,
                risk_level=risk_level,
                frame=frame
            )
            
            if alert is not None:
                generated_alerts.append(alert)
        
        return generated_alerts
    
    def update_cooldown(self, cooldown_seconds: int) -> None:
        """
        Update cooldown period.
        
        Args:
            cooldown_seconds: New cooldown period in seconds
        """
        self.cooldown_seconds = cooldown_seconds
        logger.info(f"Cooldown period updated to {cooldown_seconds}s")
    
    def reset_cooldowns(self) -> None:
        """
        Reset all cooldown tracking.
        
        This clears the last alert times for all anomaly types,
        allowing immediate alert generation.
        """
        self.last_alert_times.clear()
        logger.info("All cooldown tracking reset")
    
    def get_cooldown_status(self) -> Dict[str, dict]:
        """
        Get current cooldown status for all anomaly types.
        
        Returns:
            Dictionary mapping anomaly type to cooldown status
        """
        now = datetime.now()
        status = {}
        
        for anomaly_type, last_time in self.last_alert_times.items():
            time_since = (now - last_time).total_seconds()
            remaining = max(0, self.cooldown_seconds - time_since)
            
            status[anomaly_type] = {
                "last_alert": last_time.isoformat(),
                "time_since_seconds": time_since,
                "cooldown_remaining_seconds": remaining,
                "can_alert": remaining == 0
            }
        
        return status
