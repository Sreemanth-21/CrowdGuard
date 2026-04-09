"""
Logging infrastructure for CrowdGuard system.

This module provides structured logging configuration with support for:
- Configurable log levels via environment variables
- Console and optional file output
- Timestamp formatting
- Component/module identification
- ML pipeline error logging (Requirement 2.7)
- Alert manager error logging (Requirement 12.7)
- Cleanup service logging (Requirements 23.4, 23.6)
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color coding."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for subsequent formatters
        record.levelname = levelname
        
        return formatted


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name (typically module name)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, reads from LOG_LEVEL environment variable.
        log_file: Path to log file. If None, reads from LOG_FILE environment variable.
                 If not set, logs only to console.
        enable_colors: Enable colored output for console (default: True)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Failed to process frame", extra={"frame_id": 123})
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Determine log level
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Validate log level
    numeric_level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatters
    # Detailed format with timestamp, level, module, and message
    log_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_colors and sys.stdout.isatty():
        # Use colored formatter for terminal output
        console_formatter = ColoredFormatter(log_format, datefmt=date_format)
    else:
        # Use plain formatter for non-terminal output (e.g., redirected to file)
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is None:
        log_file = os.getenv('LOG_FILE')
    
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        
        # Use plain formatter for file output (no colors)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    This is a convenience function that calls setup_logger with default settings.
    Use this in most cases unless you need custom configuration.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from backend.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return setup_logger(name)


# Component-specific logger helpers
def log_ml_error(logger: logging.Logger, component: str, error: Exception, frame_id: Optional[int] = None):
    """
    Log ML pipeline errors with structured information.
    
    Supports Requirement 2.7: Log errors when inference fails for a frame.
    
    Args:
        logger: Logger instance
        component: ML component name (e.g., "Detector", "Tracker", "AnomalyEngine")
        error: Exception that occurred
        frame_id: Optional frame identifier
    
    Example:
        >>> logger = get_logger(__name__)
        >>> try:
        ...     # ML processing
        ...     pass
        ... except Exception as e:
        ...     log_ml_error(logger, "Detector", e, frame_id=123)
    """
    error_msg = f"ML pipeline error in {component}: {str(error)}"
    if frame_id is not None:
        error_msg += f" (frame_id={frame_id})"
    
    logger.error(error_msg, exc_info=True, extra={
        'component': component,
        'error_type': type(error).__name__,
        'frame_id': frame_id
    })


def log_alert_manager_error(logger: logging.Logger, operation: str, error: Exception, alert_data: Optional[dict] = None):
    """
    Log alert manager errors with structured information.
    
    Supports Requirement 12.7: Log errors when database write fails.
    
    Args:
        logger: Logger instance
        operation: Operation that failed (e.g., "database_write", "websocket_send")
        error: Exception that occurred
        alert_data: Optional alert data dictionary
    
    Example:
        >>> logger = get_logger(__name__)
        >>> try:
        ...     # Save alert to database
        ...     pass
        ... except Exception as e:
        ...     log_alert_manager_error(logger, "database_write", e, alert_data)
    """
    error_msg = f"Alert manager error during {operation}: {str(error)}"
    
    extra_data = {
        'operation': operation,
        'error_type': type(error).__name__
    }
    
    if alert_data:
        extra_data['alert_type'] = alert_data.get('anomaly_type')
        extra_data['alert_id'] = alert_data.get('alert_id')
    
    logger.error(error_msg, exc_info=True, extra=extra_data)


def log_cleanup_summary(logger: logging.Logger, snapshots_deleted: int, density_logs_deleted: int):
    """
    Log cleanup service summary with deletion counts.
    
    Supports Requirements 23.4 and 23.6: Log the number of snapshots and density logs deleted.
    
    Args:
        logger: Logger instance
        snapshots_deleted: Number of alert snapshots deleted
        density_logs_deleted: Number of density log records deleted
    
    Example:
        >>> logger = get_logger(__name__)
        >>> log_cleanup_summary(logger, snapshots_deleted=15, density_logs_deleted=1200)
    """
    logger.info(
        f"Cleanup completed: {snapshots_deleted} snapshots deleted, "
        f"{density_logs_deleted} density logs deleted",
        extra={
            'snapshots_deleted': snapshots_deleted,
            'density_logs_deleted': density_logs_deleted
        }
    )


def log_session_event(logger: logging.Logger, event: str, session_id: str, **kwargs):
    """
    Log session lifecycle events.
    
    Args:
        logger: Logger instance
        event: Event type (e.g., "started", "stopped", "error")
        session_id: Session identifier
        **kwargs: Additional event data
    
    Example:
        >>> logger = get_logger(__name__)
        >>> log_session_event(logger, "started", session_id="abc-123", source="webcam")
    """
    extra_data = {'session_id': session_id, 'event': event}
    extra_data.update(kwargs)
    
    logger.info(f"Session {event}: {session_id}", extra=extra_data)


def log_performance_metric(logger: logging.Logger, metric_name: str, value: float, unit: str = ""):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        metric_name: Name of the metric (e.g., "inference_time", "fps")
        value: Metric value
        unit: Optional unit (e.g., "ms", "fps")
    
    Example:
        >>> logger = get_logger(__name__)
        >>> log_performance_metric(logger, "inference_time", 45.2, "ms")
    """
    unit_str = f" {unit}" if unit else ""
    logger.debug(
        f"Performance: {metric_name} = {value}{unit_str}",
        extra={
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit
        }
    )


# Module-level logger for this module
_module_logger = get_logger(__name__)
_module_logger.info("Logging infrastructure initialized")
