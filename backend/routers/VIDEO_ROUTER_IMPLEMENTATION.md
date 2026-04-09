# Video Management Router Implementation

## Overview

The video management router provides REST API endpoints for video upload, session control, and video source management. It integrates with the SessionManager service to handle video processing session lifecycle.

## Implementation Summary

### Task 13.1: Video Upload Endpoint ✅

**Endpoint**: `POST /api/video/upload`

**Features**:
- Validates file size (max 500MB)
- Validates file format (MP4, AVI, MOV, MKV)
- Saves uploaded files to `uploads/` directory
- Extracts video metadata (duration, resolution) using OpenCV
- Returns file metadata in response

**Validation**:
- File size validation: Rejects files > 500MB with HTTP 400
- Format validation: Rejects unsupported formats with HTTP 400
- Error handling: Cleans up partial uploads on failure

**Requirements Validated**: 1.2-1.8

### Task 13.3: Session Control Endpoints ✅

#### Start Session
**Endpoint**: `POST /api/video/start`

**Features**:
- Creates new video processing session
- Validates source_type ('webcam' or 'upload')
- Enforces single active session constraint
- Integrates with SessionManager service
- Returns session ID and metadata

**Requirements Validated**: 18.1, 18.2, 18.6

#### Stop Session
**Endpoint**: `POST /api/video/stop`

**Features**:
- Stops active video processing session
- Computes final session statistics
- Returns session statistics (total_frames, total_alerts, peak_risk_score, average_density)

**Requirements Validated**: 18.3, 18.4, 18.5

#### Get Status
**Endpoint**: `GET /api/video/status`

**Features**:
- Returns current session status
- Includes uptime, frames processed, and current FPS
- Returns `active: false` when no session is running

**Requirements Validated**: 18.3

#### List Sources
**Endpoint**: `GET /api/video/sources`

**Features**:
- Lists available webcam devices (checks indices 0-4)
- Lists uploaded video files in uploads directory
- Returns structured response with webcams and uploaded_files arrays

**Requirements Validated**: 1.1

## API Endpoints

### POST /api/video/upload

**Request**: multipart/form-data
- `file`: Video file (MP4, AVI, MOV, MKV, max 500MB)

**Response 200**:
```json
{
  "filename": "crowd_video.mp4",
  "size": 45678901,
  "duration": 120.5,
  "resolution": [1920, 1080]
}
```

**Response 400** (Invalid format):
```json
{
  "detail": {
    "error": "Unsupported video format",
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv"],
    "uploaded_format": ".txt"
  }
}
```

**Response 400** (File too large):
```json
{
  "detail": {
    "error": "File size exceeds 500MB limit",
    "max_size": 524288000,
    "uploaded_size": 600000000
  }
}
```

### POST /api/video/start

**Request**:
```json
{
  "source_type": "webcam" | "upload",
  "source_name": "Webcam 0" | "filename.mp4",
  "config": {
    "confidence_threshold": 0.5,
    "model_variant": "nano"
  }
}
```

**Response 200**:
```json
{
  "session_id": "uuid-string",
  "started_at": "2024-01-15T10:30:00Z",
  "source_type": "webcam",
  "source_name": "Webcam 0"
}
```

**Response 400** (Session already active):
```json
{
  "detail": {
    "error": "Session already active",
    "active_session_id": "uuid-string"
  }
}
```

### POST /api/video/stop

**Response 200**:
```json
{
  "session_id": "uuid-string",
  "ended_at": "2024-01-15T11:30:00Z",
  "statistics": {
    "total_frames": 36000,
    "total_alerts": 15,
    "peak_risk_score": 78.5,
    "average_density": 0.42
  }
}
```

**Response 400** (No active session):
```json
{
  "detail": {
    "error": "No active session to stop"
  }
}
```

### GET /api/video/status

**Response 200** (Active session):
```json
{
  "active": true,
  "session_id": "uuid-string",
  "uptime_seconds": 3600,
  "frames_processed": 36000,
  "current_fps": 10.2
}
```

**Response 200** (No active session):
```json
{
  "active": false
}
```

### GET /api/video/sources

**Response 200**:
```json
{
  "webcams": [
    {"id": 0, "name": "Webcam 0", "available": true}
  ],
  "uploaded_files": [
    {"filename": "crowd_video.mp4", "size": 45678901}
  ]
}
```

## Integration

### SessionManager Integration

The video router integrates with the SessionManager service through FastAPI dependency injection:

```python
def get_session_manager(db: Session = Depends(get_db)) -> SessionManager:
    return SessionManager(db=db)
```

Each endpoint that requires session management receives a SessionManager instance:
- `start_session()`: Creates new session via `session_manager.create_session()`
- `stop_session()`: Ends session via `session_manager.end_session()`
- `get_status()`: Retrieves session info via `session_manager.get_active_session()`

### Database Integration

The router uses SQLAlchemy ORM through the SessionManager service:
- Session records are persisted to the `sessions` table
- Foreign key constraints ensure data integrity
- Session statistics are computed and stored on session end

## Testing

### Unit Tests

**File**: `backend/test_video_router.py`

**Test Coverage**:
- ✅ Video upload with valid formats (MP4, AVI, MOV, MKV)
- ✅ Video upload with invalid format (rejection)
- ✅ Video upload with oversized file (rejection)
- ✅ Video sources listing
- ✅ Helper function validation

**Test Results**: 6/6 tests passing

### Test Limitations

Session control endpoint tests require full database integration and are better tested through integration tests with a real database instance. The current unit tests focus on:
- Upload validation logic
- Format validation
- Size validation
- Source listing

## Error Handling

### Upload Errors
- **Invalid format**: Returns HTTP 400 with supported formats list
- **File too large**: Returns HTTP 400 with size limits
- **Save failure**: Returns HTTP 500 and cleans up partial files

### Session Errors
- **Invalid source_type**: Returns HTTP 400 with allowed values
- **Session already active**: Returns HTTP 400 with active session ID
- **No active session**: Returns HTTP 400 when trying to stop
- **Database errors**: Returns HTTP 500 with generic error message

## File Structure

```
backend/routers/
├── __init__.py
├── health.py                          # Health check endpoint
└── video.py                           # Video management router (NEW)

backend/test_video_router.py          # Unit tests (NEW)
backend/routers/VIDEO_ROUTER_IMPLEMENTATION.md  # This file (NEW)
```

## Dependencies

- **FastAPI**: Web framework and routing
- **OpenCV (cv2)**: Video metadata extraction
- **SQLAlchemy**: Database ORM (via SessionManager)
- **Pydantic**: Request/response validation
- **SessionManager**: Session lifecycle management

## Next Steps

For full system integration:
1. Create main FastAPI application (`backend/main.py`)
2. Register video router with application
3. Set up database initialization on startup
4. Add WebSocket server for real-time updates
5. Integrate with VideoProcessor for actual video processing

## Notes

- Upload directory (`uploads/`) is created automatically on router initialization
- Webcam detection checks indices 0-4 by default
- Video metadata extraction may fail for corrupted files (returns None values)
- Session control requires active database connection
- All timestamps use UTC timezone
