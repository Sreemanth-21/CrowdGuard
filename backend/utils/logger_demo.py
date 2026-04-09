"""
Demonstration script for CrowdGuard logging infrastructure.

This script demonstrates all the logging features including:
- Basic logging at different levels
- Component-specific logging helpers
- Structured logging with extra data
- File and console output
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.logger import (
    get_logger,
    log_ml_error,
    log_alert_manager_error,
    log_cleanup_summary,
    log_session_event,
    log_performance_metric
)


def demo_basic_logging():
    """Demonstrate basic logging at different levels."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Logging")
    print("="*60)
    
    logger = get_logger("demo.basic")
    
    logger.debug("This is a DEBUG message - detailed diagnostic info")
    logger.info("This is an INFO message - general information")
    logger.warning("This is a WARNING message - potential issue")
    logger.error("This is an ERROR message - something went wrong")
    logger.critical("This is a CRITICAL message - serious problem")


def demo_ml_error_logging():
    """Demonstrate ML pipeline error logging (Requirement 2.7)."""
    print("\n" + "="*60)
    print("DEMO 2: ML Pipeline Error Logging (Requirement 2.7)")
    print("="*60)
    
    logger = get_logger("demo.ml_pipeline")
    
    # Simulate ML inference error
    try:
        raise RuntimeError("CUDA out of memory during inference")
    except Exception as e:
        log_ml_error(logger, "Detector", e, frame_id=123)
    
    logger.info("System continues processing despite error")


def demo_alert_manager_error_logging():
    """Demonstrate alert manager error logging (Requirement 12.7)."""
    print("\n" + "="*60)
    print("DEMO 3: Alert Manager Error Logging (Requirement 12.7)")
    print("="*60)
    
    logger = get_logger("demo.alert_manager")
    
    # Simulate database write error
    try:
        raise ConnectionError("Database connection lost")
    except Exception as e:
        alert_data = {
            'alert_id': 'alert-abc-123',
            'anomaly_type': 'HIGH_DENSITY',
            'confidence': 0.85
        }
        log_alert_manager_error(logger, "database_write", e, alert_data)
    
    logger.info("Alert sent via WebSocket despite database error")


def demo_cleanup_logging():
    """Demonstrate cleanup service logging (Requirements 23.4, 23.6)."""
    print("\n" + "="*60)
    print("DEMO 4: Cleanup Service Logging (Requirements 23.4, 23.6)")
    print("="*60)
    
    logger = get_logger("demo.cleanup_service")
    
    # Simulate cleanup operation
    snapshots_deleted = 15
    density_logs_deleted = 1200
    
    log_cleanup_summary(logger, snapshots_deleted, density_logs_deleted)


def demo_session_event_logging():
    """Demonstrate session event logging."""
    print("\n" + "="*60)
    print("DEMO 5: Session Event Logging")
    print("="*60)
    
    logger = get_logger("demo.session_manager")
    
    session_id = "session-xyz-789"
    
    # Session started
    log_session_event(logger, "started", session_id, 
                     source="webcam", fps=10.0, resolution="1920x1080")
    
    # Session running
    log_session_event(logger, "processing", session_id,
                     frames_processed=1000, alerts_generated=5)
    
    # Session stopped
    log_session_event(logger, "stopped", session_id,
                     total_frames=36000, total_alerts=15, duration_seconds=3600)


def demo_performance_metric_logging():
    """Demonstrate performance metric logging."""
    print("\n" + "="*60)
    print("DEMO 6: Performance Metric Logging")
    print("="*60)
    
    logger = get_logger("demo.performance")
    
    # Log various performance metrics
    log_performance_metric(logger, "inference_time", 45.2, "ms")
    log_performance_metric(logger, "fps", 10.5, "fps")
    log_performance_metric(logger, "memory_usage", 512.8, "MB")
    log_performance_metric(logger, "cpu_usage", 65.3, "%")


def demo_structured_logging():
    """Demonstrate structured logging with extra data."""
    print("\n" + "="*60)
    print("DEMO 7: Structured Logging with Extra Data")
    print("="*60)
    
    logger = get_logger("demo.structured")
    
    # Log with structured extra data
    logger.info("Frame processed successfully", extra={
        'frame_id': 456,
        'person_count': 35,
        'risk_score': 45.2,
        'risk_level': 'CAUTION',
        'anomalies_detected': ['HIGH_DENSITY']
    })
    
    logger.warning("High crowd density detected", extra={
        'density': 0.75,
        'threshold': 0.70,
        'location': {'x': 512, 'y': 384}
    })


def demo_exception_logging():
    """Demonstrate exception logging with stack traces."""
    print("\n" + "="*60)
    print("DEMO 8: Exception Logging with Stack Traces")
    print("="*60)
    
    logger = get_logger("demo.exceptions")
    
    def nested_function():
        """Nested function to show stack trace."""
        raise ValueError("Invalid configuration parameter")
    
    try:
        nested_function()
    except Exception as e:
        # Log with full stack trace
        logger.exception("Configuration error occurred")


def main():
    """Run all logging demonstrations."""
    print("\n" + "="*60)
    print("CrowdGuard Logging Infrastructure Demo")
    print("="*60)
    print("\nThis demo shows all logging features of the CrowdGuard system.")
    print("Log level can be controlled via LOG_LEVEL environment variable.")
    print(f"Current log level: {os.getenv('LOG_LEVEL', 'INFO')}")
    
    # Run all demos
    demo_basic_logging()
    demo_ml_error_logging()
    demo_alert_manager_error_logging()
    demo_cleanup_logging()
    demo_session_event_logging()
    demo_performance_metric_logging()
    demo_structured_logging()
    demo_exception_logging()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nTo change log level, set LOG_LEVEL environment variable:")
    print("  export LOG_LEVEL=DEBUG    # Show all messages")
    print("  export LOG_LEVEL=INFO     # Show info and above (default)")
    print("  export LOG_LEVEL=WARNING  # Show warnings and errors only")
    print("\nTo enable file logging, set LOG_FILE environment variable:")
    print("  export LOG_FILE=./logs/crowdguard.log")
    print()


if __name__ == "__main__":
    main()
