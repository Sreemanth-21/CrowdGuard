"""
DensityLog ORM model for CrowdGuard system.

This module defines the DensityLog model for storing time-series data
of crowd density and risk metrics during video processing sessions.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, CheckConstraint, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class DensityLog(Base):
    """
    DensityLog model for time-series crowd metrics.
    
    Density logs are sampled periodically during active sessions to capture
    crowd density, risk scores, person counts, and movement velocities for
    historical analysis and trend visualization.
    
    Attributes:
        log_id: Auto-incrementing primary key
        session_id: Foreign key to associated session
        timestamp: When the metrics were recorded
        density: Crowd density value (0.0-1.0)
        risk_score: Composite risk score (0-100)
        person_count: Number of detected persons
        mean_velocity: Average velocity of tracked persons (pixels/frame)
        created_at: Record creation timestamp
    """
    
    __tablename__ = "density_logs"
    
    # Primary key
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Crowd metrics
    density = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    person_count = Column(Integer, nullable=False)
    mean_velocity = Column(Float, nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="density_logs")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "density >= 0.0 AND density <= 1.0",
            name="check_density"
        ),
        CheckConstraint(
            "risk_score >= 0.0 AND risk_score <= 100.0",
            name="check_risk_score"
        ),
        CheckConstraint(
            "person_count >= 0",
            name="check_person_count"
        ),
        CheckConstraint(
            "mean_velocity >= 0.0",
            name="check_mean_velocity"
        ),
        Index("idx_density_logs_session_id", "session_id"),
        Index("idx_density_logs_timestamp", "timestamp"),
    )
    
    def __repr__(self) -> str:
        """String representation of DensityLog."""
        return (
            f"<DensityLog(log_id={self.log_id}, "
            f"session_id='{self.session_id}', "
            f"density={self.density:.2f}, "
            f"risk_score={self.risk_score:.2f}, "
            f"timestamp='{self.timestamp}')>"
        )
    
    def to_dict(self) -> dict:
        """
        Convert density log to dictionary representation.
        
        Returns:
            Dictionary containing density log data
        """
        return {
            "log_id": self.log_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "density": self.density,
            "risk_score": self.risk_score,
            "person_count": self.person_count,
            "mean_velocity": self.mean_velocity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
