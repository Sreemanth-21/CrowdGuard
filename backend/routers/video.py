"""
Video Management Router for CrowdGuard.
"""

import os
import cv2
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime

from backend.database import get_db
from backend.services.session_manager import SessionManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])

MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request/Response Models ──────────────────────────────────────────────────

class VideoStartRequest(BaseModel):
    source_type: str = Field(..., description="'webcam' or 'upload'")
    source_name: str = Field(..., description="Webcam index or video file path")
    config: Optional[Dict[str, Any]] = Field(default=None)


class VideoStartResponse(BaseModel):
    session_id: str
    started_at: str
    source_type: str
    source_name: str


class VideoStopResponse(BaseModel):
    session_id: str
    ended_at: str
    statistics: Dict[str, Any]


class VideoStatusResponse(BaseModel):
    active: bool
    session_id: Optional[str] = None
    uptime_seconds: Optional[float] = None
    frames_processed: Optional[int] = None
    current_fps: Optional[float] = None


class VideoUploadResponse(BaseModel):
    filename: str
    size: int
    duration: Optional[float] = None
    resolution: Optional[List[int]] = None


class VideoSourcesResponse(BaseModel):
    webcams: List[Dict[str, Any]]
    uploaded_files: List[Dict[str, Any]]


# ── Helpers ──────────────────────────────────────────────────────────────────

def validate_video_format(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_FORMATS


def get_video_metadata(filepath: str) -> Dict[str, Any]:
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {"duration": None, "resolution": None}
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else None
        resolution = [width, height] if width > 0 and height > 0 else None
        cap.release()
        return {"duration": duration, "resolution": resolution}
    except Exception as e:
        logger.error(f"Failed to extract video metadata: {e}")
        return {"duration": None, "resolution": None}


def list_webcams() -> List[Dict[str, Any]]:
    webcams = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            webcams.append({"id": i, "name": f"Webcam {i}", "available": True})
            cap.release()
    return webcams


def list_uploaded_files() -> List[Dict[str, Any]]:
    uploaded_files = []
    if not os.path.exists(UPLOAD_DIR):
        return uploaded_files
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(filepath) and validate_video_format(filename):
            uploaded_files.append({"filename": filename, "size": os.path.getsize(filepath)})
    return uploaded_files


def get_session_manager(db: Session = Depends(get_db)) -> SessionManager:
    return SessionManager(db=db)


def _get_video_processor():
    """
    Get video processor from settings module.
    
    Uses module reference to avoid stale imports.
    
    Returns:
        VideoProcessor instance or None if not initialized
    """
    import backend.routers.settings as settings_module
    processor = settings_module._video_processor
    
    if processor is None:
        logger.error("Video processor is None - not initialized by main.py")
    else:
        logger.debug(f"Video processor retrieved: active={processor.is_active}")
    
    return processor


# ── Background frame processing loop ─────────────────────────────────────────

async def _frame_processing_loop(session_id: str):
    """
    Background task that continuously captures frames, runs the full ML
    pipeline, and broadcasts results via WebSocket.

    Runs until the video processor is stopped or an unrecoverable error
    occurs. Consecutive frame failures are counted; after 30 in a row the
    loop exits to avoid a busy-spin on a broken source.
    """
    from backend.routers.websocket import broadcast_frame
    from backend.database import SessionLocal

    video_processor = _get_video_processor()
    if video_processor is None:
        logger.error("Video processor not available for frame loop")
        return

    logger.info(f"Frame processing loop started for session {session_id}")

    db = SessionLocal()
    session_manager = SessionManager(db=db)

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30
    total_persons_seen = 0
    total_alerts_seen = 0

    try:
        while video_processor.is_active:
            try:
                # Run ML pipeline in thread pool (blocking OpenCV call)
                processed_frame = await asyncio.get_event_loop().run_in_executor(
                    None, video_processor.process_frame
                )

                if processed_frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(
                            f"[{session_id}] {MAX_CONSECUTIVE_FAILURES} consecutive frame "
                            "failures — stopping loop"
                        )
                        break
                    await asyncio.sleep(0.05)
                    continue

                consecutive_failures = 0  # reset on success

                # Log detection count every frame
                logger.info(
                    f"[{session_id}] frame={video_processor.frame_count} "
                    f"persons={processed_frame.person_count} "
                    f"risk={processed_frame.risk_score.level}({processed_frame.risk_score.score:.1f})"
                )

                # Accumulate session-level counters
                total_persons_seen += processed_frame.person_count
                total_alerts_seen += len(processed_frame.anomalies)

                # Encode annotated frame to JPEG base64
                import base64
                ret, buffer = cv2.imencode(
                    '.jpg', processed_frame.annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not ret:
                    logger.warning("Failed to JPEG-encode frame, skipping")
                    await asyncio.sleep(0.033)
                    continue

                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Uptime from session manager
                active_session = session_manager.get_active_session()
                uptime = active_session.get("uptime_seconds", 0) if active_session else 0
                fps = video_processor.frame_count / uptime if uptime and uptime > 0 else 0.0

                # Flat-key message shape — matches useWebSocket.ts handler
                message = {
                    "type": "frame",
                    "image": frame_b64,
                    "session_id": session_id,
                    "person_count": processed_frame.person_count,
                    "risk_score": round(processed_frame.risk_score.score, 2),
                    "risk_level": processed_frame.risk_score.level,
                    "density": round(processed_frame.heatmap.density, 4),
                    "anomalies": [
                        {
                            "type": a.type,
                            "confidence": round(a.confidence, 3),
                            "description": a.description,
                            "location": {"x": a.location[0], "y": a.location[1]}
                            if hasattr(a, "location") and a.location else None,
                        }
                        for a in processed_frame.anomalies
                    ],
                    "timestamp": processed_frame.timestamp.isoformat(),
                    "session_stats": {
                        "uptime_seconds": uptime,
                        "frames_processed": video_processor.frame_count,
                        "total_persons": total_persons_seen,
                        "total_alerts": total_alerts_seen,
                        "fps": round(fps, 1),
                        "peak_risk_score": round(processed_frame.risk_score.score, 2),
                    },
                }

                await broadcast_frame(message)

            except Exception as frame_err:
                consecutive_failures += 1
                logger.error(
                    f"[{session_id}] Frame loop error (failure {consecutive_failures}): "
                    f"{frame_err}",
                    exc_info=True,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"[{session_id}] Too many consecutive errors — stopping loop"
                    )
                    break
                await asyncio.sleep(0.1)
                continue

            # ~30 FPS cap
            await asyncio.sleep(0.033)

    except Exception as e:
        logger.error(f"[{session_id}] Frame processing loop fatal error: {e}", exc_info=True)
    finally:
        db.close()
        logger.info(f"Frame processing loop ended for session {session_id}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_200_OK)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    """
    Upload a video file for processing.
    
    Accepts multipart/form-data with a file field.
    Validates format and size before saving.
    
    Args:
        file: Video file to upload
        
    Returns:
        VideoUploadResponse with file metadata
        
    Raises:
        HTTPException 400: Invalid format or size
        HTTPException 500: Upload failed
    """
    try:
        # Validate filename exists
        if not file.filename:
            logger.error("Upload rejected: No filename provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "No filename provided"}
            )
        
        logger.info(f"Upload request: filename={file.filename}, content_type={file.content_type}")
        
        # Validate video format
        if not validate_video_format(file.filename):
            logger.warning(f"Upload rejected: Unsupported format for {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Unsupported video format", "supported_formats": list(ALLOWED_FORMATS)}
            )

        # Read file content
        content = await file.read()
        file_size = len(content)
        
        logger.info(f"File read: {file.filename}, size={file_size} bytes")

        # Validate file size
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Upload rejected: File size {file_size} exceeds limit {MAX_FILE_SIZE}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"}
            )

        # Save file
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved: {filepath}")
        
        # Extract metadata
        metadata = get_video_metadata(filepath)
        
        logger.info(f"Upload successful: {file.filename}, duration={metadata['duration']}, resolution={metadata['resolution']}")
        
        return VideoUploadResponse(
            filename=file.filename,
            size=file_size,
            duration=metadata["duration"],
            resolution=metadata["resolution"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        # Clean up file if it was partially saved
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up partial file: {filepath}")
            except:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Upload failed: {str(e)}"}
        )
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail={"error": "Failed to save uploaded file"})


@router.post("/start", response_model=VideoStartResponse, status_code=status.HTTP_200_OK)
async def start_session(
    request: VideoStartRequest,
    session_manager: SessionManager = Depends(get_session_manager)
) -> VideoStartResponse:
    logger.info(f"Start session request: source_type={request.source_type}, source_name={request.source_name}")
    
    if request.source_type not in ("webcam", "upload"):
        logger.error(f"Invalid source_type: {request.source_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid source_type", "allowed_values": ["webcam", "upload"]}
        )

    # Get video processor
    video_processor = _get_video_processor()
    if video_processor is None:
        logger.error("Video processor is None - cannot start session. Check if main.py initialized it via settings.set_video_processor()")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Video processor not initialized. Check server logs."}
        )

    # If a session is already active, stop it first so the new one can start cleanly
    active_session = session_manager.get_active_session()
    if active_session:
        old_id = active_session["session_id"]
        logger.warning(
            f"Session {old_id} already active — auto-stopping before starting new session"
        )
        try:
            if video_processor.is_active:
                await asyncio.get_event_loop().run_in_executor(
                    None, video_processor.stop_session
                )
            session_manager.end_session(old_id)
            logger.info(f"Auto-stopped previous session {old_id}")
        except Exception as stop_err:
            logger.error(f"Failed to auto-stop previous session: {stop_err}")

    try:
        # Start actual video capture
        logger.info(f"Starting video processor: {request.source_type}:{request.source_name}")
        started = await asyncio.get_event_loop().run_in_executor(
            None,
            video_processor.start_session,
            request.source_type,
            request.source_name
        )

        if not started:
            logger.error(f"Video processor failed to open source: {request.source_name}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"Failed to open video source: {request.source_name}"}
            )

        # Create DB session record
        session_id = session_manager.create_session(
            source_type=request.source_type,
            source_name=request.source_name
        )

        started_at = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Session started: {session_id}, source={request.source_type}:{request.source_name}")

        # Start background frame processing loop
        asyncio.create_task(_frame_processing_loop(session_id))
        logger.info(f"Frame processing loop task created for session {session_id}")

        return VideoStartResponse(
            session_id=session_id,
            started_at=started_at,
            source_type=request.source_type,
            source_name=request.source_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start session: {e}", exc_info=True)
        # Clean up video processor if DB session creation failed
        if video_processor.is_active:
            video_processor.stop_session()
        raise HTTPException(status_code=500, detail={"error": "Failed to start session"})


@router.post("/stop", response_model=VideoStopResponse, status_code=status.HTTP_200_OK)
async def stop_session(
    session_manager: SessionManager = Depends(get_session_manager)
) -> VideoStopResponse:
    logger.info("Stop session request received")
    
    active_session = session_manager.get_active_session()

    if not active_session:
        logger.warning("No active session to stop")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No active session to stop"}
        )

    try:
        # Stop video processor
        video_processor = _get_video_processor()
        if video_processor is None:
            logger.error("Video processor is None during stop - cannot release webcam")
        elif video_processor.is_active:
            logger.info("Stopping video processor and releasing webcam")
            await asyncio.get_event_loop().run_in_executor(
                None, video_processor.stop_session
            )
            logger.info("Video processor stopped successfully")
        else:
            logger.warning("Video processor not active during stop")

        # End DB session
        statistics = session_manager.end_session(active_session["session_id"])
        logger.info(f"Session stopped: {statistics['session_id']}, frames={statistics['total_frames']}, alerts={statistics['total_alerts']}")

        return VideoStopResponse(
            session_id=statistics["session_id"],
            ended_at=statistics["end_time"],
            statistics={
                "total_frames": statistics["total_frames"],
                "total_alerts": statistics["total_alerts"],
                "peak_risk_score": statistics["peak_risk_score"],
                "average_density": statistics["average_density"]
            }
        )

    except Exception as e:
        logger.error(f"Failed to stop session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Failed to stop session"})


@router.get("/status", response_model=VideoStatusResponse, status_code=status.HTTP_200_OK)
async def get_status(
    session_manager: SessionManager = Depends(get_session_manager)
) -> VideoStatusResponse:
    video_processor = _get_video_processor()

    if video_processor is None:
        logger.warning("Video processor is None in status check")
        return VideoStatusResponse(active=False)

    # Use video processor as source of truth for active state
    if video_processor.is_active:
        active_session = session_manager.get_active_session()
        session_id = active_session["session_id"] if active_session else None
        uptime = active_session["uptime_seconds"] if active_session else 0
        total_frames = video_processor.frame_count
        current_fps = total_frames / uptime if uptime and uptime > 0 else None

        logger.debug(f"Status check: active=True, session_id={session_id}, frames={total_frames}")
        
        return VideoStatusResponse(
            active=True,
            session_id=session_id,
            uptime_seconds=uptime,
            frames_processed=total_frames,
            current_fps=current_fps
        )

    logger.debug("Status check: active=False")
    return VideoStatusResponse(active=False)


@router.get("/sources", response_model=VideoSourcesResponse, status_code=status.HTTP_200_OK)
async def get_sources() -> VideoSourcesResponse:
    webcams = list_webcams()
    uploaded_files = list_uploaded_files()
    return VideoSourcesResponse(webcams=webcams, uploaded_files=uploaded_files)