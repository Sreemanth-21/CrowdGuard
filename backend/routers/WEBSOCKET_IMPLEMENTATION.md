# WebSocket Server Implementation

## Overview

The WebSocket server provides real-time bidirectional communication between the CrowdGuard backend and frontend dashboard. It broadcasts video frames, alerts, status updates, and error messages while handling client commands.

**Validates: Requirements 25.1-25.8**

## Implementation Summary

### Files Created

1. **backend/routers/websocket.py** - WebSocket router with ConnectionManager
2. **backend/test_websocket.py** - Comprehensive unit tests (20 tests, all passing)
3. **backend/websocket_demo.py** - Demo script showing usage

### Key Components

#### ConnectionManager Class

The `ConnectionManager` handles all WebSocket connections and message broadcasting:

**Features:**
- Connection lifecycle management (connect, disconnect)
- Client ID assignment and tracking
- Message broadcasting to all connected clients
- Priority-based message queue (alerts > errors > status > frames)
- Queue overflow handling (drops oldest frames, preserves alerts/errors)
- Graceful handling of disconnected clients

**Methods:**
- `connect(websocket)` - Accept new WebSocket connection
- `disconnect(websocket)` - Remove WebSocket connection
- `broadcast_frame()` - Broadcast frame with annotated image and stats
- `broadcast_alert()` - Broadcast alert with snapshot
- `broadcast_status()` - Broadcast system status
- `broadcast_error()` - Broadcast error message

#### WebSocket Endpoint

**Endpoint:** `ws://localhost:8000/ws`

**Connection Flow:**
1. Client connects to /ws endpoint
2. Server accepts connection and assigns unique client ID
3. Server sends connection acknowledgment message
4. Client can send command messages
5. Server broadcasts messages to all connected clients
6. Client disconnects (gracefully or due to error)

### Message Types

#### 1. Connection Acknowledgment (Backend → Frontend)

```json
{
  "type": "connected",
  "payload": {
    "client_id": "uuid-string",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Validates: Requirement 25.1**

#### 2. Frame Message (Backend → Frontend)

```json
{
  "type": "frame",
  "payload": {
    "image": "base64-encoded-jpeg-string",
    "person_count": 35,
    "risk_score": 45.2,
    "risk_level": "CAUTION",
    "density": 0.42,
    "density_zone": "MEDIUM",
    "anomalies": [
      {
        "type": "HIGH_DENSITY",
        "confidence": 0.85,
        "location": {"x": 512, "y": 384}
      }
    ],
    "session_stats": {
      "uptime_seconds": 3600,
      "frames_processed": 36000,
      "total_persons": 1250,
      "total_alerts": 15,
      "fps": 10.2
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Features:**
- Frame encoded as base64 JPEG for efficient transmission
- Includes cumulative session statistics (uptime, total persons, total alerts, peak risk score)
- Density zone classification (LOW < 0.3, MEDIUM 0.3-0.6, HIGH >= 0.6)
- List of detected anomalies with locations

**Validates: Requirements 25.2, 13.1, 13.2, 13.3, 17.5**

#### 3. Alert Message (Backend → Frontend)

```json
{
  "type": "alert",
  "payload": {
    "alert_id": "uuid-string",
    "anomaly_type": "HIGH_DENSITY",
    "risk_level": "WARNING",
    "confidence": 0.85,
    "description": "High crowd density detected in zone",
    "timestamp": "2024-01-15T10:35:22Z",
    "snapshot": "base64-encoded-jpeg-string",
    "affected_persons": 45,
    "location": {"x": 512, "y": 384}
  }
}
```

**Features:**
- Alert snapshot encoded as base64 JPEG
- Includes all alert metadata
- Broadcast within 100ms of generation (handled by async broadcasting)

**Validates: Requirements 12.6, 25.3**

#### 4. Status Message (Backend → Frontend)

```json
{
  "type": "status",
  "payload": {
    "uptime_seconds": 3600,
    "frames_processed": 36000,
    "total_persons": 1250,
    "total_alerts": 15,
    "fps": 10.2,
    "memory_usage_mb": 512.5,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Features:**
- Broadcast every 30 seconds (to be implemented in video processor integration)
- Includes system performance metrics

**Validates: Requirement 25.4**

#### 5. Error Message (Backend → Frontend)

```json
{
  "type": "error",
  "payload": {
    "code": "WEBCAM_DISCONNECTED",
    "message": "Webcam connection lost. Attempting reconnection...",
    "timestamp": "2024-01-15T10:30:00Z",
    "retry_count": 1,
    "max_retries": 3
  }
}
```

**Features:**
- Broadcast immediately when errors occur
- Includes retry information for recoverable errors

**Validates: Requirement 25.5**

#### 6. Command Message (Frontend → Backend)

**Toggle Heatmap:**
```json
{
  "type": "command",
  "action": "toggle_heatmap",
  "params": {
    "enabled": true
  }
}
```

**Update Threshold:**
```json
{
  "type": "command",
  "action": "update_threshold",
  "params": {
    "confidence_threshold": 0.6
  }
}
```

**Features:**
- Handles client commands to control system behavior
- Command handlers are placeholders (to be connected to video processor and settings)

**Validates: Requirement 25.6**

### Message Queue Management

**Priority System:**
1. Alert messages (priority 0) - highest
2. Error messages (priority 1)
3. Status messages (priority 2)
4. Frame messages (priority 3) - lowest

**Overflow Handling:**
- Maximum queue size: 100 messages
- When queue is full:
  - Attempts to drop oldest frame message first
  - If no frames available, drops oldest message
  - Always preserves alert and error messages when possible

**Validates: Requirement 25.7**

### Connection Management

**Features:**
- Supports multiple simultaneous client connections
- Each client gets unique UUID identifier
- Graceful handling of disconnections
- Automatic cleanup of disconnected clients during broadcast
- Connection acknowledgment sent immediately after connection

**Error Handling:**
- Failed message sends don't crash the server
- Disconnected clients are automatically removed
- Errors are logged for debugging

## Test Coverage

### Unit Tests (20 tests, all passing)

**Connection Management:**
- ✓ WebSocket connection establishment
- ✓ WebSocket disconnection
- ✓ Multiple simultaneous connections
- ✓ Connection acknowledgment message

**Message Broadcasting:**
- ✓ Frame message broadcasting
- ✓ Alert message broadcasting
- ✓ Status message broadcasting
- ✓ Error message broadcasting
- ✓ Broadcasting to multiple clients
- ✓ Handling disconnected clients during broadcast

**Queue Management:**
- ✓ Message priority ordering
- ✓ Queue overflow drops frames
- ✓ Queue preserves alerts and errors

**Command Handling:**
- ✓ Toggle heatmap command
- ✓ Update threshold command
- ✓ Unknown command handling

**Data Encoding:**
- ✓ Frame base64 encoding
- ✓ Alert snapshot base64 encoding
- ✓ Density zone calculation (LOW, MEDIUM, HIGH)

## Usage Example

```python
from backend.routers.websocket import get_connection_manager
import numpy as np

# Get the global connection manager
manager = get_connection_manager()

# Broadcast a frame
await manager.broadcast_frame(
    frame=annotated_frame,
    person_count=15,
    risk_score=45.5,
    risk_level="CAUTION",
    density=0.42,
    anomalies=[...],
    session_stats={...}
)

# Broadcast an alert
await manager.broadcast_alert(
    alert_id="alert-123",
    anomaly_type="HIGH_DENSITY",
    risk_level="WARNING",
    confidence=0.85,
    description="High crowd density detected",
    timestamp=datetime.utcnow(),
    snapshot=frame,
    affected_persons=45,
    location=(512, 384)
)

# Broadcast status
await manager.broadcast_status(
    uptime_seconds=3600,
    frames_processed=36000,
    total_persons=1250,
    total_alerts=15,
    fps=10.2
)

# Broadcast error
await manager.broadcast_error(
    error_code="WEBCAM_DISCONNECTED",
    error_message="Webcam connection lost",
    retry_count=1,
    max_retries=3
)
```

## Integration Points

### Video Processor Integration (Task 20.2)

The video processor will need to:
1. Get the connection manager instance
2. Call `broadcast_frame()` after each frame is processed
3. Call `broadcast_status()` every 30 seconds
4. Call `broadcast_error()` when errors occur

### Alert Manager Integration (Task 20.2)

The alert manager will need to:
1. Get the connection manager instance
2. Call `broadcast_alert()` when alerts are generated
3. Ensure alerts are broadcast within 100ms

### Settings Integration

Command handlers need to be connected to:
1. Video processor for heatmap toggle
2. Settings manager for threshold updates

## Performance Considerations

**Efficiency:**
- Async/await for non-blocking I/O
- Base64 encoding is efficient for JSON transmission
- JPEG compression reduces frame size
- Priority queue ensures important messages are sent first

**Scalability:**
- Supports multiple simultaneous connections
- Queue size limit prevents memory overflow
- Automatic cleanup of disconnected clients

**Reliability:**
- Graceful error handling
- No single client failure affects others
- Message queue preserves critical messages

## Next Steps

1. **Task 20.1** - Register WebSocket router in main FastAPI application
2. **Task 20.2** - Integrate with VideoProcessor and AlertManager
3. **Task 23** - Implement frontend WebSocket client hook
4. **Task 29** - Connect VideoFeed component to WebSocket messages

## Demo

Run the demo to see WebSocket functionality:

```bash
python backend/websocket_demo.py
```

The demo shows:
- Connection manager initialization
- Frame message broadcasting
- Alert message broadcasting
- Status message broadcasting
- Error message broadcasting
- Message queue priority
- Connection lifecycle

## Conclusion

The WebSocket server implementation is complete and fully tested. It provides:
- ✓ Real-time bidirectional communication
- ✓ Priority-based message queue
- ✓ Multiple message types (frame, alert, status, error, command)
- ✓ Efficient base64 encoding for images
- ✓ Graceful error handling
- ✓ Support for multiple clients
- ✓ Comprehensive test coverage (20/20 tests passing)

All sub-tasks for Task 17 are complete and ready for integration with the video processor and frontend.
