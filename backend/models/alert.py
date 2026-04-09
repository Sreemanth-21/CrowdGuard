"""
Alert ORM model for CrowdGuard system.

This module defines the Alert model representing detected anomalies
with their associated metadata and risk information.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, CheckConstraint, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class Alert(Base):
    """
    Alert model representing a detected crowd anomaly.
    
    Alerts are generated when anomalies are detected that exceed configured
    thresholds. Each alert captures the anomaly type, risk level, confidence,
    and associated frame snapshot.
    
    Attributes:
        alert_id: Unique identifier for the alert (UUID)
        session_id: Foreign key to associated session
        timestamp: When the alert was generated
        anomaly_type: Type of detected anomaly
        risk_level: Risk classification (SAFE, CAUTION, WARNING, CRITICAL)
        confidence_score: Anomaly detection confidence (0.0-1.0)
        description: Human-readable alert description
        frame_snapshot_path: Path to saved frame snapshot
        affected_persons: Number of persons involved in anomaly
        location_x: X coordinate of anomaly center
        location_y: Y coordinate of anomaly center
        is_dismissed: Whether alert has been dismissed by operator
        dismissed_at: Timestamp when alert was dismissed
        created_at: Record creation timestamp
    """
    
    __tablename__ = "alerts"
    
    # Primary key
    alert_id = Column(String, primary_key=True, nullable=False)
    
    # Foreign key
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False)
    dismissed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Anomaly information
    anomaly_type = Column(String, nullable=False)
    risk_level = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    description = Column(String, nullable=False)
    
    # Frame and location information
    frame_snapshot_path = Column(String, nullable=True)
    affected_persons = Column(Integer, nullable=False, default=0)
    location_x = Column(Integer, nullable=True)
    location_y = Column(Integer, nullable=True)
    
    # Dismissal tracking
    is_dismissed = Column(Boolean, nullable=False, default=False)
    
    # Relationships
    session = relationship("Session", back_populates="alerts")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "anomaly_type IN ('HIGH_DENSITY', 'RAPID_MOVEMENT', 'SUDDEN_DISPERSAL', "
            "'CROWD_SURGE', 'STATIONARY_CROWD', 'FIGHTING')",
            name="check_anomaly_type"
        ),
        CheckConstraint(
            "risk_level IN ('SAFE', 'CAUTION', 'WARNING', 'CRITICAL')",
            name="check_risk_level"
        ),
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0",
            name="check_confidence_score"
        ),
        Index("idx_alerts_session_id", "session_id"),
        Index("idx_alerts_timestamp", "timestamp"),
        Index("idx_alerts_anomaly_type", "anomaly_type"),
        Index("idx_alerts_risk_level", "risk_level"),
        Index("idx_alerts_is_dismissed", "is_dismissed"),
    )
    
    def __repr__(self) -> str:
        """String representation of Alert."""
        return (
            f"<Alert(alert_id='{self.alert_id}', "
            f"anomaly_type='{self.anomaly_type}', "
            f"risk_level='{self.risk_level}', "
            f"timestamp='{self.timestamp}')>"
        )
    
    def to_dict(self) -> dict:
        """
        Convert alert to dictionary representation.
        
        Returns:
            Dictionary containing alert data
        """
        return {
            "alert_id": self.alert_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "anomaly_type": self.anomaly_type,
            "risk_level": self.risk_level,
            "confidence_score": self.confidence_score,
            "description": self.description,
            "frame_snapshot_path": self.frame_snapshot_path,
            "affected_persons": self.affected_persons,
            "location_x": self.location_x,
            "location_y": self.location_y,
            "is_dismissed": self.is_dismissed,
            "dismissed_at": self.dismissed_at.isoformat() if self.dismissed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
