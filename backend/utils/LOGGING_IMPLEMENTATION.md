# Logging Infrastructure Implementation Summary

## Overview

This document summarizes the logging infrastructure implementation for the CrowdGuard system, completed as part of Task 1.7.

## Requirements Addressed

The logging infrastructure supports the following requirements:

### Requirement 2.7: ML Pipeline Error Logging
- **Acceptance Criteria**: "WHEN inference fails for a Frame, THE Detector SHALL log the error and continue processing subsequent frames"
- **Implementation**: `log_ml_error()` helper function logs ML pipeline errors with component name, error details, and optional frame ID
- **Usage**: Allows the system to log errors and continue processing without stopping

### Requirement 12.7: Alert Manager Error Logging
- **Acceptance Criteria**: "IF Database write fails, THEN THE Alert_Manager SHALL log the error and continue Alert transmission to Dashboard"
- **Implementation**: `log_alert_manager_error()` helper function logs alert manager errors with operation type, error details, and optional alert data
- **Usage**: Enables logging of database failures while continuing to send alerts via WebSocket

### Requirement 23.4: Snapshot Cleanup Logging
- **Acceptance Criteria**: "THE System SHALL log the number of snapshots deleted in each cleanup run"
- **Implementation**: `log_cleanup_summary()` helper function logs the count of deleted snapshots
- **Usage**: Provides visibility into cleanup operations

### Requirement 23.6: Density Log Cleanup Logging
- **Acceptance Criteria**: "THE System SHALL log the number of density_log records deleted in each cleanup run"
- **Implementation**: `log_cleanup_summary()` helper function logs the count of deleted density logs
- **Usage**: Provides visibility into cleanup operations

## Files Created

### 1. `backend/utils/logger.py` (Main Implementation)
**Purpose**: Core logging infrastructure with structured logging support

**Key Features**:
- Configurable log levels via `LOG_LEVEL` environment variable
- Console output with optional color coding
- Optional file output via `LOG_FILE` environment variable
- Timestamp formatting (YYYY-MM-DD HH:MM:SS)
- Component/module identification
- Structured logging with extra data support

**Main Functions**:
- `setup_logger(name, log_level, log_file, enable_colors)`: Configure a logger with custom settings
- `get_logger(name)`: Convenience function to get a logger with default settings
- `log_ml_error(logger, component, error, frame_id)`: Log ML pipeline errors (Req 2.7)
- `log_alert_manager_error(logger, operation, error, alert_data)`: Log alert manager errors (Req 12.7)
- `log_cleanup_summary(logger, snapshots_deleted, density_logs_deleted)`: Log cleanup results (Req 23.4, 23.6)
- `log_session_event(logger, event, session_id, **kwargs)`: Log session lifecycle events
- `log_performance_metric(logger, metric_name, value, unit)`: Log performance metrics

**Classes**:
- `ColoredFormatter`: Custom formatter for colored console output

### 2. `backend/test_logger.py` (Comprehensive Tests)
**Purpose**: Unit tests for all logging functionality

**Test Coverage**:
- Logger setup and configuration (8 tests)
- Colored formatter functionality (2 tests)
- ML error logging (2 tests)
- Alert manager error logging (2 tests)
- Cleanup logging (2 tests)
- Session event logging (2 tests)
- Performance metric logging (2 tests)
- Integration tests (2 tests)

**Total**: 22 tests, all passing ✓

### 3. `backend/utils/LOGGING_USAGE.md` (Documentation)
**Purpose**: Comprehensive usage guide for developers

**Contents**:
- Quick start guide
- Configuration via environment variables
- Component-specific logging examples
- Advanced usage patterns
- Best practices
- Troubleshooting guide

### 4. `backend/utils/logger_demo.py` (Demonstration)
**Purpose**: Interactive demonstration of all logging features

**Demonstrations**:
1. Basic logging at different levels
2. ML pipeline error logging (Req 2.7)
3. Alert manager error logging (Req 12.7)
4. Cleanup service logging (Req 23.4, 23.6)
5. Session event logging
6. Performance metric logging
7. Structured logging with extra data
8. Exception logging with stack traces

### 5. `backend/utils/__init__.py` (Package Exports)
**Purpose**: Make logging functions easily importable

**Exports**:
- `get_logger`
- `setup_logger`
- `log_ml_error`
- `log_alert_manager_error`
- `log_cleanup_summary`
- `log_session_event`
- `log_performance_metric`

## Configuration

### Environment Variables

The logging infrastructure is configured via environment variables in `.env`:

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Optional: Log file path
# If not set, logs only to console
LOG_FILE=./logs/crowdguard.log
```

These variables are already documented in `.env.example`.

## Usage Examples

### Basic Usage
```python
from backend.utils import get_logger

logger = get_logger(__name__)
logger.info("System initialized")
```

### ML Pipeline Error Logging (Requirement 2.7)
```python
from backend.utils import get_logger, log_ml_error

logger = get_logger(__name__)

try:
    results = model.predict(frame)
except Exception as e:
    log_ml_error(logger, "Detector", e, frame_id=123)
    continue  # Continue processing
```

### Alert Manager Error Logging (Requirement 12.7)
```python
from backend.utils import get_logger, log_alert_manager_error

logger = get_logger(__name__)

try:
    db.session.add(alert)
    db.session.commit()
except Exception as e:
    log_alert_manager_error(logger, "database_write", e, alert_data)
    await websocket.send_alert(alert)  # Continue to send via WebSocket
```

### Cleanup Logging (Requirements 23.4, 23.6)
```python
from backend.utils import get_logger, log_cleanup_summary

logger = get_logger(__name__)

snapshots_deleted = cleanup_old_snapshots()
density_logs_deleted = cleanup_old_density_logs()

log_cleanup_summary(logger, snapshots_deleted, density_logs_deleted)
```

## Log Output Format

### Console Output (with colors)
```
2024-01-15 10:30:00 | INFO     | backend.ml.detector | YOLOv8 model loaded successfully
2024-01-15 10:30:05 | WARNING  | backend.services.alert_manager | Alert cooldown active
2024-01-15 10:30:10 | ERROR    | backend.ml.detector | ML pipeline error in Detector
```

### File Output (no colors)
Same format as console but without ANSI color codes.

## Testing

All tests pass successfully:

```bash
$ python -m pytest backend/test_logger.py -v
======================== 22 passed in 0.48s =========================
```

Test coverage includes:
- ✓ Logger setup and configuration
- ✓ Log level handling
- ✓ File and console output
- ✓ Colored formatting
- ✓ Component-specific logging helpers
- ✓ Structured logging with extra data
- ✓ Exception logging with stack traces

## Integration with Existing Code

The logging infrastructure is designed to integrate seamlessly with existing CrowdGuard components:

1. **ML Pipeline** (`backend/ml/`): Use `log_ml_error()` for inference failures
2. **Alert Manager** (`backend/services/alert_manager.py`): Use `log_alert_manager_error()` for database errors
3. **Cleanup Service** (`backend/services/cleanup_service.py`): Use `log_cleanup_summary()` for cleanup results
4. **Session Manager** (`backend/services/session_manager.py`): Use `log_session_event()` for lifecycle events
5. **Video Processor** (`backend/ml/video_processor.py`): Use `log_performance_metric()` for FPS tracking

## Benefits

1. **Consistent Logging**: All components use the same logging infrastructure
2. **Configurable**: Log level and output can be controlled via environment variables
3. **Structured**: Extra data can be attached to log messages for better analysis
4. **Colored Output**: Easy-to-read colored console output for development
5. **File Logging**: Optional file output for production environments
6. **Error Resilience**: Helper functions support error logging without stopping the system
7. **Well Tested**: Comprehensive test suite ensures reliability
8. **Well Documented**: Usage guide and examples for developers

## Next Steps

To use the logging infrastructure in other components:

1. Import the logger: `from backend.utils import get_logger`
2. Create a module-level logger: `logger = get_logger(__name__)`
3. Use appropriate helper functions for specific scenarios
4. Add structured data with the `extra` parameter when needed

## Conclusion

The logging infrastructure is complete, tested, and ready for use throughout the CrowdGuard backend. It provides comprehensive support for all logging requirements (2.7, 12.7, 23.4, 23.6) and offers a flexible, easy-to-use API for developers.
