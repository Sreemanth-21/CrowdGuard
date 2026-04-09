"""
Settings Router for CrowdGuard.

This module provides REST API endpoints for configuration management,
including retrieving, updating, and resetting system settings.

**Validates: Requirements 24.1-24.8**
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
from pydantic import BaseModel, Field, validator
from backend.database import get_db
from backend.config import ConfigManager, ConfigurationError, VALIDATION_RULES
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


# Pydantic models for request/response validation
class SettingsResponse(BaseModel):
    """Response model for settings retrieval."""
    confidence_threshold: float
    model_variant: str
    high_density_threshold: float
    cooldown_period_seconds: int
    heatmap_opacity: float
    rapid_movement_threshold: float
    crowd_surge_threshold: float
    sudden_dispersal_threshold: float
    stationary_crowd_threshold: float
    stationary_velocity_threshold: float
    stationary_duration_seconds: int
    fighting_iou_threshold: float
    fighting_velocity_threshold: float
    
    class Config:
        schema_extra = {
            "example": {
                "confidence_threshold": 0.5,
                "model_variant": "nano",
                "high_density_threshold": 0.7,
                "cooldown_period_seconds": 10,
                "heatmap_opacity": 0.6,
                "rapid_movement_threshold": 25,
                "crowd_surge_threshold": 0.3,
                "sudden_dispersal_threshold": 0.4,
                "stationary_crowd_threshold": 0.5,
                "stationary_velocity_threshold": 3,
                "stationary_duration_seconds": 30,
                "fighting_iou_threshold": 0.3,
                "fighting_velocity_threshold": 20
            }
        }


class SettingsUpdateRequest(BaseModel):
    """Request model for settings updates (partial updates allowed)."""
    confidence_threshold: float | None = Field(None, ge=0.3, le=0.9)
    model_variant: str | None = Field(None, pattern="^(nano|small|medium)$")
    high_density_threshold: float | None = Field(None, ge=0.5, le=0.9)
    cooldown_period_seconds: int | None = Field(None, ge=5, le=60)
    heatmap_opacity: float | None = Field(None, ge=0.0, le=1.0)
    rapid_movement_threshold: float | None = Field(None, ge=0)
    crowd_surge_threshold: float | None = Field(None, ge=0.0, le=1.0)
    sudden_dispersal_threshold: float | None = Field(None, ge=0.0, le=1.0)
    stationary_crowd_threshold: float | None = Field(None, ge=0.0, le=1.0)
    stationary_velocity_threshold: float | None = Field(None, ge=0)
    stationary_duration_seconds: int | None = Field(None, ge=0)
    fighting_iou_threshold: float | None = Field(None, ge=0.0, le=1.0)
    fighting_velocity_threshold: float | None = Field(None, ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "confidence_threshold": 0.6,
                "heatmap_opacity": 0.7
            }
        }


class SettingsUpdateResponse(BaseModel):
    """Response model for settings update."""
    updated: bool
    settings: SettingsResponse
    applied_to_session: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "updated": True,
                "settings": {
                    "confidence_threshold": 0.6,
                    "model_variant": "nano",
                    "high_density_threshold": 0.7,
                    "cooldown_period_seconds": 10,
                    "heatmap_opacity": 0.7
                },
                "applied_to_session": True
            }
        }


class SettingsResetResponse(BaseModel):
    """Response model for settings reset."""
    reset: bool
    settings: SettingsResponse
    
    class Config:
        schema_extra = {
            "example": {
                "reset": True,
                "settings": {
                    "confidence_threshold": 0.5,
                    "model_variant": "nano",
                    "high_density_threshold": 0.7,
                    "cooldown_period_seconds": 10,
                    "heatmap_opacity": 0.6
                }
            }
        }


# Global reference to video processor for hot-reload
# This will be set by main.py during application startup
_video_processor = None


def set_video_processor(processor):
    """
    Set the global video processor reference for hot-reload.
    
    This function should be called by main.py during application startup
    to enable configuration hot-reload functionality.
    
    Args:
        processor: VideoProcessor instance
    """
    global _video_processor
    _video_processor = processor
    logger.info("Video processor reference set for settings hot-reload")


@router.get("", response_model=SettingsResponse)
async def get_settings(db: Session = Depends(get_db)) -> SettingsResponse:
    """
    Get current system configuration.
    
    Retrieves all configuration parameters from the database or defaults.
    
    Returns:
        SettingsResponse containing all configuration parameters
        
    **Validates: Requirement 24.1**
    """
    try:
        config_manager = ConfigManager(db)
        config = config_manager.get_config()
        
        logger.info("Settings retrieved successfully")
        return SettingsResponse(**config)
        
    except Exception as e:
        logger.error(f"Failed to retrieve settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve settings: {str(e)}"
        )


@router.put("", response_model=SettingsUpdateResponse)
async def update_settings(
    updates: SettingsUpdateRequest,
    db: Session = Depends(get_db)
) -> SettingsUpdateResponse:
    """
    Update system configuration.
    
    Accepts partial updates - only provided fields will be updated.
    Validates all updates against allowed ranges and persists to database.
    If a video processing session is active, applies changes to the active
    session within 1 second (hot-reload).
    
    Args:
        updates: SettingsUpdateRequest with fields to update
        db: Database session
        
    Returns:
        SettingsUpdateResponse with updated configuration
        
    Raises:
        HTTPException 400: If validation fails
        HTTPException 500: If update fails
        
    **Validates: Requirements 24.1-24.7**
    """
    try:
        # Convert request to dictionary, excluding None values
        update_dict = updates.model_dump(exclude_none=True)
        
        if not update_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No settings provided for update"
            )
        
        # Create config manager and update settings
        config_manager = ConfigManager(db)
        
        try:
            updated_config = config_manager.update_config(update_dict)
        except ConfigurationError as e:
            logger.warning(f"Configuration validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Apply hot-reload to active session if video processor is available
        applied_to_session = False
        if _video_processor is not None:
            try:
                success = _video_processor.update_config(update_dict)
                if success:
                    applied_to_session = True
                    logger.info(
                        f"Configuration hot-reloaded to active session: {list(update_dict.keys())}"
                    )
            except Exception as e:
                logger.warning(f"Failed to apply hot-reload: {e}")
                # Don't fail the request if hot-reload fails
        
        logger.info(f"Settings updated: {list(update_dict.keys())}")
        
        return SettingsUpdateResponse(
            updated=True,
            settings=SettingsResponse(**updated_config),
            applied_to_session=applied_to_session
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )


@router.post("/reset", response_model=SettingsResetResponse)
async def reset_settings(db: Session = Depends(get_db)) -> SettingsResetResponse:
    """
    Reset all settings to default values.
    
    Restores all configuration parameters to their system defaults and
    persists to database. If a video processing session is active, applies
    the default configuration to the active session.
    
    Args:
        db: Database session
        
    Returns:
        SettingsResetResponse with default configuration
        
    Raises:
        HTTPException 500: If reset fails
        
    **Validates: Requirement 24.8**
    """
    try:
        config_manager = ConfigManager(db)
        default_config = config_manager.reset_to_defaults()
        
        # Apply hot-reload to active session if video processor is available
        if _video_processor is not None:
            try:
                _video_processor.update_config(default_config)
                logger.info("Default configuration hot-reloaded to active session")
            except Exception as e:
                logger.warning(f"Failed to apply hot-reload after reset: {e}")
                # Don't fail the request if hot-reload fails
        
        logger.info("Settings reset to defaults")
        
        return SettingsResetResponse(
            reset=True,
            settings=SettingsResponse(**default_config)
        )
        
    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset settings: {str(e)}"
        )


@router.get("/validation-rules")
async def get_validation_rules() -> Dict[str, Any]:
    """
    Get validation rules for all configuration parameters.
    
    Returns information about allowed ranges, types, and descriptions
    for each configuration parameter. Useful for frontend validation
    and UI generation.
    
    Returns:
        Dictionary mapping parameter names to validation rules
    """
    # Convert validation rules to JSON-serializable format
    serializable_rules = {}
    for key, rules in VALIDATION_RULES.items():
        rule_copy = rules.copy()
        
        # Convert type to string representation
        if "type" in rule_copy:
            type_val = rule_copy["type"]
            if isinstance(type_val, tuple):
                rule_copy["type"] = [t.__name__ for t in type_val]
            else:
                rule_copy["type"] = type_val.__name__
        
        serializable_rules[key] = rule_copy
    
    return {
        "rules": serializable_rules,
        "description": "Validation rules for configuration parameters"
    }
