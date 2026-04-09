"""
Configuration management system for CrowdGuard.

This module provides configuration loading, validation, and management
for the CrowdGuard system. It handles default settings initialization,
environment variable loading, and runtime configuration updates.

**Validates: Requirements 24.1, 24.2, 24.3, 24.4, 24.5**
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from backend.models.settings import Settings


# Default configuration settings
DEFAULT_CONFIG = {
    "confidence_threshold": 0.20,
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

# Validation rules for configuration parameters
VALIDATION_RULES = {
    "confidence_threshold": {
        "type": float,
        "min": 0.10,
        "max": 0.9,
        "description": "Minimum detection confidence score (0.10 to 0.9)"
    },
    "model_variant": {
        "type": str,
        "allowed_values": ["nano", "small", "medium"],
        "description": "YOLOv8 model variant (nano, small, medium)"
    },
    "high_density_threshold": {
        "type": float,
        "min": 0.5,
        "max": 0.9,
        "description": "High density anomaly threshold (0.5 to 0.9)"
    },
    "cooldown_period_seconds": {
        "type": int,
        "min": 5,
        "max": 60,
        "description": "Alert cooldown period in seconds (5 to 60)"
    },
    "heatmap_opacity": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "Heatmap overlay opacity (0.0 to 1.0)"
    },
    "rapid_movement_threshold": {
        "type": (int, float),
        "min": 0,
        "description": "Rapid movement velocity threshold in pixels per frame"
    },
    "crowd_surge_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "Crowd surge detection threshold (0.0 to 1.0)"
    },
    "sudden_dispersal_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "Sudden dispersal detection threshold (0.0 to 1.0)"
    },
    "stationary_crowd_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "Stationary crowd density threshold (0.0 to 1.0)"
    },
    "stationary_velocity_threshold": {
        "type": (int, float),
        "min": 0,
        "description": "Stationary crowd velocity threshold in pixels per frame"
    },
    "stationary_duration_seconds": {
        "type": int,
        "min": 0,
        "description": "Stationary crowd duration threshold in seconds"
    },
    "fighting_iou_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "Fighting detection IoU threshold (0.0 to 1.0)"
    },
    "fighting_velocity_threshold": {
        "type": (int, float),
        "min": 0,
        "description": "Fighting detection velocity threshold in pixels per frame"
    }
}


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ConfigManager:
    """
    Configuration manager for CrowdGuard system.
    
    Handles loading, validation, persistence, and retrieval of configuration
    settings. Supports both database-backed and in-memory configuration.
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize configuration manager.
        
        Args:
            db: Optional database session for persistent configuration
        """
        self.db = db
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from database or defaults.
        
        Loads configuration from database if available, otherwise uses
        default values. Also checks environment variables for overrides.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        config = DEFAULT_CONFIG.copy()
        
        # Load from database if available
        if self.db:
            try:
                settings = self.db.query(Settings).all()
                for setting in settings:
                    try:
                        config[setting.setting_key] = json.loads(setting.setting_value)
                    except json.JSONDecodeError:
                        # If not JSON, use as string
                        config[setting.setting_key] = setting.setting_value
            except Exception as e:
                # If database read fails, use defaults
                print(f"Warning: Failed to load config from database: {e}")
        
        # Override with environment variables if present
        config = self._load_env_overrides(config)
        
        # Cache the configuration
        self._config_cache = config
        
        return config
    
    def _load_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.
        
        Environment variables should be prefixed with CROWDGUARD_
        and use uppercase with underscores (e.g., CROWDGUARD_CONFIDENCE_THRESHOLD).
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration dictionary with environment overrides applied
        """
        for key in config.keys():
            env_key = f"CROWDGUARD_{key.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Parse environment variable based on expected type
                try:
                    if key in VALIDATION_RULES:
                        expected_type = VALIDATION_RULES[key]["type"]
                        if expected_type == float or expected_type == (int, float):
                            config[key] = float(env_value)
                        elif expected_type == int:
                            config[key] = int(env_value)
                        else:
                            config[key] = env_value
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid environment variable {env_key}: {e}")
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate configuration parameters against validation rules.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, errors_dict) where errors_dict maps
            parameter names to error messages
        """
        errors = {}
        
        for key, value in config.items():
            if key not in VALIDATION_RULES:
                # Unknown parameter, skip validation
                continue
            
            rules = VALIDATION_RULES[key]
            
            # Type validation
            expected_type = rules["type"]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    errors[key] = f"Must be of type {' or '.join(t.__name__ for t in expected_type)}"
                    continue
            else:
                if not isinstance(value, expected_type):
                    errors[key] = f"Must be of type {expected_type.__name__}"
                    continue
            
            # Range validation for numeric types
            if "min" in rules and value < rules["min"]:
                errors[key] = f"Must be at least {rules['min']}"
                continue
            
            if "max" in rules and value > rules["max"]:
                errors[key] = f"Must be at most {rules['max']}"
                continue
            
            # Allowed values validation for strings
            if "allowed_values" in rules and value not in rules["allowed_values"]:
                errors[key] = f"Must be one of: {', '.join(rules['allowed_values'])}"
                continue
        
        return len(errors) == 0, errors
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with new values.
        
        Validates updates and persists to database if available.
        
        Args:
            updates: Dictionary of configuration parameters to update
            
        Returns:
            Updated configuration dictionary
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Load current config
        current_config = self.load_config()
        
        # Apply updates
        updated_config = current_config.copy()
        updated_config.update(updates)
        
        # Validate updated configuration
        is_valid, errors = self.validate_config(updated_config)
        if not is_valid:
            raise ConfigurationError(f"Invalid configuration: {errors}")
        
        # Persist to database if available
        if self.db:
            try:
                for key, value in updates.items():
                    setting = self.db.query(Settings).filter(
                        Settings.setting_key == key
                    ).first()
                    
                    if setting:
                        setting.setting_value = json.dumps(value)
                    else:
                        setting = Settings(
                            setting_key=key,
                            setting_value=json.dumps(value)
                        )
                        self.db.add(setting)
                
                self.db.commit()
            except Exception as e:
                self.db.rollback()
                raise ConfigurationError(f"Failed to persist configuration: {e}")
        
        # Update cache
        self._config_cache = updated_config
        
        return updated_config
    
    def reset_to_defaults(self) -> Dict[str, Any]:
        """
        Reset all configuration to default values.
        
        Returns:
            Default configuration dictionary
        """
        if self.db:
            try:
                # Delete all settings from database
                self.db.query(Settings).delete()
                self.db.commit()
            except Exception as e:
                self.db.rollback()
                raise ConfigurationError(f"Failed to reset configuration: {e}")
        
        # Reset cache
        self._config_cache = DEFAULT_CONFIG.copy()
        
        return self._config_cache
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns cached configuration if available, otherwise loads from database.
        
        Returns:
            Current configuration dictionary
        """
        if not self._config_cache:
            return self.load_config()
        return self._config_cache.copy()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a single configuration setting.
        
        Args:
            key: Configuration parameter name
            default: Default value if setting not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config()
        return config.get(key, default)
    
    def initialize_defaults(self) -> None:
        """
        Initialize database with default configuration if empty.
        
        This should be called on application startup to ensure
        configuration exists in the database.
        """
        if not self.db:
            return
        
        try:
            # Check if any settings exist
            existing_count = self.db.query(Settings).count()
            
            if existing_count == 0:
                # Initialize with defaults
                for key, value in DEFAULT_CONFIG.items():
                    setting = Settings(
                        setting_key=key,
                        setting_value=json.dumps(value)
                    )
                    self.db.add(setting)
                
                self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Warning: Failed to initialize default configuration: {e}")


def get_config_manager(db: Optional[Session] = None) -> ConfigManager:
    """
    Factory function to create ConfigManager instance.
    
    Args:
        db: Optional database session
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(db)
