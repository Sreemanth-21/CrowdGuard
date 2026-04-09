"""
Session ORM model for CrowdGuard system.

This module defines the Session model representing video processing sessions
with their associated metadata and statistics.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, CheckConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class Session(Base):
    """
    Session model representing a video processing session.
    
    A session tracks a continuous period of video processing from start to stop,
    including statistics about frames processed, alerts generated, and crowd metrics.
    
    Attributes:
        session_id: Unique identifier for the session (UUID)
        start_time: Timestamp when session started
        end_time: Timestamp when session ended (NULL if active)
        video_source_type: Type of video source ('webcam' or 'upload')
        source_name: Name/identifier of the video source
        total_frames: Total number of frames processed
        total_alerts: Total number of alerts generated
        peak_risk_score: Maximum risk score recorded (0-100)
        average_density: Average crowd density (0.0-1.0)
        created_at: Record creation timestamp
    """
    
    __tablename__ = "sessions"
    
    # Primary key
    session_id = Column(String, primary_key=True, nullable=False)
    
    # Timestamps
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Video source information
    video_source_type = Column(String, nullable=False)
    source_name = Column(String, nullable=False)
    
    # Session statistics
    total_frames = Column(Integer, nullable=False, default=0)
    total_alerts = Column(Integer, nullable=False, default=0)
    peak_risk_score = Column(Float, nullable=False, default=0.0)
    average_density = Column(Float, nullable=False, default=0.0)
    
    # Relationships
    alerts = relationship("Alert", back_populates="session", cascade="all, delete-orphan")
    density_logs = relationship("DensityLog", back_populates="session", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "video_source_type IN ('webcam', 'upload')",
            name="check_video_source_type"
        ),
        Index("idx_sessions_start_time", "start_time"),
        Index("idx_sessions_end_time", "end_time"),
    )
    
    def __repr__(self) -> str:
        """String representation of Session."""
        return (
            f"<Session(session_id='{self.session_id}', "
            f"source_type='{self.video_source_type}', "
            f"source_name='{self.source_name}', "
            f"start_time='{self.start_time}')>"
        )
    
    def to_dict(self) -> dict:
        """
        Convert session to dictionary representation.
        
        Returns:
            Dictionary containing session data
        """
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "video_source_type": self.video_source_type,
            "source_name": self.source_name,
            "total_frames": self.total_frames,
            "total_alerts": self.total_alerts,
            "peak_risk_score": self.peak_risk_score,
            "average_density": self.average_density,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
