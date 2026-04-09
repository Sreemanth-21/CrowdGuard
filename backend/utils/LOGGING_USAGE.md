# Logging Infrastructure Usage Guide

This guide demonstrates how to use the CrowdGuard logging infrastructure in your modules.

## Quick Start

### Basic Usage

```python
from backend.utils.logger import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Log messages at different levels
logger.debug("Detailed debugging information")
logger.info("General information about system operation")
logger.warning("Warning about potential issues")
logger.error("Error that occurred but system continues")
logger.critical("Critical error that may cause system failure")
```

### Configuration via Environment Variables

Set the log level in your `.env` file:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Optional: Enable file logging
LOG_FILE=./logs/crowdguard.log
```

## Component-Specific Logging

### ML Pipeline Error Logging (Requirement 2.7)

Use `log_ml_error()` when ML inference fails:

```python
from backend.utils.logger import get_logger, log_ml_error

logger = get_logger(__name__)

try:
    # YOLOv8 detection
    results = model.predict(frame)
except Exception as e:
    # Log the error and continue processing
    log_ml_error(logger, "Detector", e, frame_id=frame_number)
    continue  # Continue with next frame
```

### Alert Manager Error Logging (Requirement 12.7)

Use `log_alert_manager_error()` when database writes fail:

```python
from backend.utils.logger import get_logger, log_alert_manager_error

logger = get_logger(__name__)

try:
    # Save alert to database
    db.session.add(alert)
    db.session.commit()
except Exception as e:
    # Log the error but continue to send alert via WebSocket
    log_alert_manager_error(logger, "database_write", e, alert_data={
        'alert_id': alert.alert_id,
        'anomaly_type': alert.anomaly_type
    })
    # Continue to send alert via WebSocket
    await websocket.send_alert(alert)
```

### Cleanup Service Logging (Requirements 23.4, 23.6)

Use `log_cleanup_summary()` to log deletion counts:

```python
from backend.utils.logger import get_logger, log_cleanup_summary

logger = get_logger(__name__)

# Perform cleanup
snapshots_deleted = cleanup_old_snapshots()
density_logs_deleted = cleanup_old_density_logs()

# Log the summary
log_cleanup_summary(logger, snapshots_deleted, density_logs_deleted)
```

### Session Event Logging

Use `log_session_event()` for session lifecycle events:

```python
from backend.utils.logger import get_logger, log_session_event

logger = get_logger(__name__)

# Session started
log_session_event(logger, "started", session_id, source="webcam", fps=10.0)

# Session stopped
log_session_event(logger, "stopped", session_id, 
                 total_frames=36000, total_alerts=15)

# Session error
log_session_event(logger, "error", session_id, 
                 error="Webcam disconnected", retry_count=3)
```

### Performance Metric Logging

Use `log_performance_metric()` for performance tracking:

```python
from backend.utils.logger import get_logger, log_performance_metric
import time

logger = get_logger(__name__)

# Measure inference time
start = time.time()
results = model.predict(frame)
inference_time = (time.time() - start) * 1000  # Convert to ms

log_performance_metric(logger, "inference_time", inference_time, "ms")
log_performance_metric(logger, "fps", current_fps, "fps")
```

## Advanced Usage

### Custom Logger Configuration

```python
from backend.utils.logger import setup_logger

# Create a logger with custom settings
logger = setup_logger(
    name="my_module",
    log_level="DEBUG",
    log_file="./logs/my_module.log",
    enable_colors=True
)
```

### Structured Logging with Extra Data

```python
logger = get_logger(__name__)

# Add extra structured data to log messages
logger.info("Processing frame", extra={
    'frame_id': 123,
    'person_count': 35,
    'risk_score': 45.2
})

logger.error("Alert generation failed", extra={
    'alert_type': 'HIGH_DENSITY',
    'session_id': 'abc-123',
    'error_code': 'DB_WRITE_FAILED'
})
```

### Exception Logging with Stack Traces

```python
logger = get_logger(__name__)

try:
    # Some operation
    process_video_frame(frame)
except Exception as e:
    # Log with full stack trace
    logger.error("Frame processing failed", exc_info=True)
    # Or use exception() which automatically includes exc_info
    logger.exception("Frame processing failed")
```

## Log Output Format

### Console Output (with colors)

```
2024-01-15 10:30:00 | INFO     | backend.ml.detector | YOLOv8 model loaded successfully
2024-01-15 10:30:05 | WARNING  | backend.services.alert_manager | Alert cooldown active for HIGH_DENSITY
2024-01-15 10:30:10 | ERROR    | backend.ml.detector | ML pipeline error in Detector: CUDA out of memory
```

### File Output (no colors)

```
2024-01-15 10:30:00 | INFO     | backend.ml.detector | YOLOv8 model loaded successfully
2024-01-15 10:30:05 | WARNING  | backend.services.alert_manager | Alert cooldown active for HIGH_DENSITY
2024-01-15 10:30:10 | ERROR    | backend.ml.detector | ML pipeline error in Detector: CUDA out of memory
```

## Best Practices

1. **Use `get_logger(__name__)`** at the module level to automatically include the module name in logs
2. **Log errors but continue processing** - Use the helper functions to log errors without stopping the system
3. **Include context in log messages** - Use the `extra` parameter to add structured data
4. **Use appropriate log levels**:
   - `DEBUG`: Detailed diagnostic information
   - `INFO`: General informational messages
   - `WARNING`: Warning messages for potentially harmful situations
   - `ERROR`: Error messages for serious problems
   - `CRITICAL`: Critical messages for very serious errors
5. **Don't log sensitive information** - Avoid logging passwords, API keys, or personal data
6. **Use structured logging** - Include extra data for better log analysis

## Example: Complete Module with Logging

```python
"""
Example module demonstrating logging best practices.
"""

from backend.utils.logger import get_logger, log_ml_error, log_performance_metric
import time

# Module-level logger
logger = get_logger(__name__)

class VideoProcessor:
    def __init__(self):
        logger.info("VideoProcessor initialized")
        self.frame_count = 0
    
    def process_frame(self, frame, frame_id):
        """Process a single video frame."""
        start_time = time.time()
        
        try:
            # Process frame
            results = self._detect_persons(frame)
            
            # Log performance
            processing_time = (time.time() - start_time) * 1000
            log_performance_metric(logger, "frame_processing_time", processing_time, "ms")
            
            self.frame_count += 1
            logger.debug(f"Frame {frame_id} processed successfully", extra={
                'frame_id': frame_id,
                'person_count': len(results),
                'processing_time_ms': processing_time
            })
            
            return results
            
        except Exception as e:
            # Log error and continue
            log_ml_error(logger, "VideoProcessor", e, frame_id=frame_id)
            return []
    
    def _detect_persons(self, frame):
        """Detect persons in frame."""
        # Detection logic here
        pass
```

## Troubleshooting

### Logs not appearing

1. Check the `LOG_LEVEL` environment variable
2. Ensure you're using `get_logger(__name__)` or `setup_logger()`
3. Verify the log level of your message matches the configured level

### File logging not working

1. Check the `LOG_FILE` environment variable is set
2. Ensure the directory exists or can be created
3. Verify write permissions for the log file location

### Colors not showing in console

1. Colors are automatically disabled for non-TTY output (e.g., redirected to file)
2. Set `enable_colors=False` in `setup_logger()` to disable colors explicitly
