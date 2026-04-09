"""
Settings ORM model for CrowdGuard system.

This module defines the Settings model for storing system configuration
parameters with key-value pairs.
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class Settings(Base):
    """
    Settings model for system configuration.
    
    Stores configuration parameters as key-value pairs with update tracking.
    Configuration values are stored as JSON-encoded strings to support
    various data types.
    
    Attributes:
        setting_key: Configuration parameter name (primary key)
        setting_value: JSON-encoded configuration value
        updated_at: Timestamp of last update
    """
    
    __tablename__ = "settings"
    
    # Primary key
    setting_key = Column(String, primary_key=True, nullable=False)
    
    # Configuration value
    setting_value = Column(String, nullable=False)
    
    # Timestamp
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self) -> str:
        """String representation of Settings."""
        return (
            f"<Settings(setting_key='{self.setting_key}', "
            f"setting_value='{self.setting_value}', "
            f"updated_at='{self.updated_at}')>"
        )
    
    def to_dict(self) -> dict:
        """
        Convert settings to dictionary representation.
        
        Returns:
            Dictionary containing settings data
        """
        return {
            "setting_key": self.setting_key,
            "setting_value": self.setting_value,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
