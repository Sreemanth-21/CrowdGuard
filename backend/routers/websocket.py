"""
WebSocket Router for CrowdGuard.

This module implements real-time WebSocket communication for streaming
video frames, alerts, status updates, and handling client commands.

**Validates: Requirements 25.1-25.8**
"""

import asyncio
import json
import uuid
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.utils.logger import get_logger
import cv2
import numpy as np

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.
    
    Handles connection lifecycle, message queuing with priority,
    and broadcasting to all connected clients.
    
    Attributes:
        active_connections: List of active WebSocket connections
        message_queue: Priority queue for outgoing messages
        max_queue_size: Maximum queue size (default: 100)
    """
    
    def __init__(self, max_queue_size: int = 100):
        """
        Initialize ConnectionManager.
        
        Args:
            max_queue_size: Maximum number of messages in queue
        """
        self.active_connections: List[WebSocket] = []
        self.message_queue: deque = deque(maxlen=max_queue_size)
        self.max_queue_size = max_queue_size
        self.client_ids: Dict[WebSocket, str] = {}
        
        logger.info(f"ConnectionManager initialized with max_queue_size={max_queue_size}")
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to register
            
        Returns:
            Client ID assigned to this connection
            
        **Validates: Requirement 25.1**
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Generate unique client ID
        client_id = str(uuid.uuid4())
        self.client_ids[websocket] = client_id
        
        logger.info(f"WebSocket connected: client_id={client_id}, total_connections={len(self.active_connections)}")
        
        # Send connection acknowledgment
        await self.send_connection_ack(websocket, client_id)
        
        return client_id
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to remove
            
        **Validates: Requirement 25.1**
        """
        if websocket in self.active_connections:
            client_id = self.client_ids.get(websocket, "unknown")
            self.active_connections.remove(websocket)
            if websocket in self.client_ids:
                del self.client_ids[websocket]
            
            logger.info(f"WebSocket disconnected: client_id={client_id}, remaining_connections={len(self.active_connections)}")
    
    async def send_connection_ack(self, websocket: WebSocket, client_id: str):
        """
        Send connection acknowledgment message.
        
        Args:
            websocket: WebSocket connection
            client_id: Assigned client ID
            
        **Validates: Requirement 25.1**
        """
        message = {
            "type": "connected",
            "payload": {
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        try:
            await websocket.send_json(message)
            logger.debug(f"Sent connection acknowledgment to client {client_id}")
        except Exception as e:
            logger.error(f"Failed to send connection acknowledgment: {e}")
    
    def _get_message_priority(self, message_type: str) -> int:
        """
        Get priority value for message type (lower = higher priority).
        
        Priority order: alerts > errors > status > frames
        
        Args:
            message_type: Type of message
            
        Returns:
            Priority value (0-3)
            
        **Validates: Requirement 25.7**
        """
        priority_map = {
            "alert": 0,
            "error": 1,
            "status": 2,
            "frame": 3
        }
        return priority_map.get(message_type, 4)
    
    def _add_to_queue(self, message: Dict[str, Any]):
        """
        Add message to priority queue with overflow handling.
        
        If queue is full, drops oldest frame messages while preserving
        alerts and errors.
        
        Args:
            message: Message to add to queue
            
        **Validates: Requirement 25.7**
        """
        message_type = message.get("type", "unknown")
        priority = self._get_message_priority(message_type)
        
        # If queue is at max capacity, handle overflow
        if len(self.message_queue) >= self.max_queue_size:
            # Try to find and remove oldest frame message
            removed = False
            for i in range(len(self.message_queue)):
                if self.message_queue[i][1].get("type") == "frame":
                    del self.message_queue[i]
                    removed = True
                    logger.debug("Dropped oldest frame message due to queue overflow")
                    break
            
            # If no frame found and queue still full, drop oldest message
            if not removed and len(self.message_queue) >= self.max_queue_size:
                dropped = self.message_queue.popleft()
                logger.warning(f"Dropped message of type {dropped[1].get('type')} due to queue overflow")
        
        # Add message with priority
        self.message_queue.append((priority, message))
        
        # Sort queue by priority (stable sort maintains order within same priority)
        self.message_queue = deque(sorted(self.message_queue, key=lambda x: x[0]))
    
    async def broadcast_frame(
        self,
        frame: np.ndarray,
        person_count: int,
        risk_score: float,
        risk_level: str,
        density: float,
        anomalies: List[Dict[str, Any]],
        session_stats: Dict[str, Any]
    ):
        """
        Broadcast frame message to all connected clients.
        
        Encodes frame as base64 JPEG and includes all frame metadata.
        
        Args:
            frame: Annotated frame image (BGR format)
            person_count: Number of detected persons
            risk_score: Composite risk score (0-100)
            risk_level: Risk level classification
            density: Crowd density (0.0-1.0)
            anomalies: List of detected anomalies
            session_stats: Cumulative session statistics
            
        **Validates: Requirements 25.2, 13.1, 13.2, 13.3, 17.5**
        """
        try:
            # Encode frame as base64 JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Determine density zone
            if density < 0.3:
                density_zone = "LOW"
            elif density < 0.6:
                density_zone = "MEDIUM"
            else:
                density_zone = "HIGH"
            
            message = {
                "type": "frame",
                "payload": {
                    "image": image_base64,
                    "person_count": person_count,
                    "risk_score": round(risk_score, 2),
                    "risk_level": risk_level,
                    "density": round(density, 3),
                    "density_zone": density_zone,
                    "anomalies": anomalies,
                    "session_stats": {
                        "uptime_seconds": session_stats.get("uptime_seconds", 0),
                        "frames_processed": session_stats.get("frames_processed", 0),
                        "total_persons": session_stats.get("total_persons", 0),
                        "total_alerts": session_stats.get("total_alerts", 0),
                        "fps": round(session_stats.get("fps", 0.0), 1)
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
            
            await self._broadcast_message(message)
            
        except Exception as e:
            logger.error(f"Failed to broadcast frame: {e}")
    
    async def broadcast_alert(
        self,
        alert_id: str,
        anomaly_type: str,
        risk_level: str,
        confidence: float,
        description: str,
        timestamp: datetime,
        snapshot: Optional[np.ndarray],
        affected_persons: int,
        location: tuple
    ):
        """
        Broadcast alert message to all connected clients.
        
        Includes alert snapshot as base64 image if available.
        
        Args:
            alert_id: Unique alert identifier
            anomaly_type: Type of detected anomaly
            risk_level: Risk level classification
            confidence: Anomaly confidence score
            description: Human-readable description
            timestamp: Alert generation timestamp
            snapshot: Optional frame snapshot
            affected_persons: Number of persons involved
            location: (x, y) coordinates of anomaly
            
        **Validates: Requirements 12.6, 25.3**
        """
        try:
            # Encode snapshot if provided
            snapshot_base64 = None
            if snapshot is not None:
                _, buffer = cv2.imencode('.jpg', snapshot)
                snapshot_base64 = base64.b64encode(buffer).decode('utf-8')
            
            message = {
                "type": "alert",
                "payload": {
                    "alert_id": alert_id,
                    "anomaly_type": anomaly_type,
                    "risk_level": risk_level,
                    "confidence": round(confidence, 3),
                    "description": description,
                    "timestamp": timestamp.isoformat() + "Z" if isinstance(timestamp, datetime) else timestamp,
                    "snapshot": snapshot_base64,
                    "affected_persons": affected_persons,
                    "location": {"x": location[0], "y": location[1]}
                }
            }
            
            await self._broadcast_message(message)
            logger.info(f"Broadcasted alert: {anomaly_type} (alert_id={alert_id})")
            
        except Exception as e:
            logger.error(f"Failed to broadcast alert: {e}")
    
    async def broadcast_status(
        self,
        uptime_seconds: int,
        frames_processed: int,
        total_persons: int,
        total_alerts: int,
        fps: float,
        memory_usage_mb: Optional[float] = None
    ):
        """
        Broadcast status message to all connected clients.
        
        Args:
            uptime_seconds: Session uptime in seconds
            frames_processed: Total frames processed
            total_persons: Total persons detected
            total_alerts: Total alerts generated
            fps: Current frames per second
            memory_usage_mb: Optional memory usage in MB
            
        **Validates: Requirement 25.4**
        """
        try:
            message = {
                "type": "status",
                "payload": {
                    "uptime_seconds": uptime_seconds,
                    "frames_processed": frames_processed,
                    "total_persons": total_persons,
                    "total_alerts": total_alerts,
                    "fps": round(fps, 1),
                    "memory_usage_mb": round(memory_usage_mb, 1) if memory_usage_mb else None,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
            
            await self._broadcast_message(message)
            logger.debug(f"Broadcasted status: fps={fps:.1f}, frames={frames_processed}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast status: {e}")
    
    async def broadcast_error(
        self,
        error_code: str,
        error_message: str,
        retry_count: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """
        Broadcast error message to all connected clients.
        
        Args:
            error_code: Error code identifier
            error_message: Human-readable error message
            retry_count: Current retry attempt
            max_retries: Maximum retry attempts
            
        **Validates: Requirement 25.5**
        """
        try:
            message = {
                "type": "error",
                "payload": {
                    "code": error_code,
                    "message": error_message,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "retry_count": retry_count,
                    "max_retries": max_retries
                }
            }
            
            await self._broadcast_message(message)
            logger.warning(f"Broadcasted error: {error_code} - {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast error: {e}")
    
    async def _broadcast_message(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients.
        
        Handles disconnected clients gracefully.
        
        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return
        
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()


async def broadcast_frame(message: Dict[str, Any]):
    """
    Broadcast frame message to all connected WebSocket clients.
    Called by video.py on every processed frame.
    """
    person_count = message.get("person_count", message.get("data", {}).get("person_count", "?"))
    logger.info(f"WS sending frame, persons: {person_count}, clients: {len(manager.active_connections)}")
    await manager._broadcast_message(message)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.

    Keeps the connection alive with periodic pings while also handling
    any command messages the client sends. Exceptions are caught so a
    single bad message never kills the connection.

    **Validates: Requirements 25.1, 25.6**
    """
    client_id = await manager.connect(websocket)

    async def _receive_loop():
        """Process incoming client messages until disconnect."""
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "command":
                        await handle_command(websocket, message)
                    else:
                        logger.debug(f"Client {client_id} sent: {message.get('type')}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
            except WebSocketDisconnect:
                raise  # propagate so outer handler can clean up
            except Exception as e:
                logger.error(f"Receive error for client {client_id}: {e}")
                raise

    async def _ping_loop():
        """Send a ping every 20 s to keep the connection alive through proxies."""
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break  # connection gone; let _receive_loop handle cleanup

    try:
        # Run receive and ping loops concurrently; cancel both on first exit
        receive_task = asyncio.create_task(_receive_loop())
        ping_task = asyncio.create_task(_ping_loop())
        done, pending = await asyncio.wait(
            [receive_task, ping_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Re-raise any exception from the receive task
        for task in done:
            if not task.cancelled() and task.exception():
                exc = task.exception()
                if not isinstance(exc, WebSocketDisconnect):
                    logger.error(f"WebSocket task error for {client_id}: {exc}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        manager.disconnect(websocket)


async def handle_command(websocket: WebSocket, message: Dict[str, Any]):
    """
    Handle command messages from clients.
    
    Supported commands:
    - toggle_heatmap: Enable/disable heatmap overlay
    - update_threshold: Update detection threshold
    
    Args:
        websocket: Client WebSocket connection
        message: Command message
        
    **Validates: Requirement 25.6**
    """
    action = message.get("action")
    params = message.get("params", {})
    
    client_id = manager.client_ids.get(websocket, "unknown")
    
    if action == "toggle_heatmap":
        enabled = params.get("enabled", False)
        logger.info(f"Client {client_id} toggled heatmap: enabled={enabled}")
        # TODO: Implement heatmap toggle logic (will be connected to video processor)
        
    elif action == "update_threshold":
        threshold = params.get("confidence_threshold")
        if threshold is not None:
            logger.info(f"Client {client_id} requested threshold update: {threshold}")
            # TODO: Implement threshold update logic (will be connected to settings)
        else:
            logger.warning(f"Client {client_id} sent update_threshold without threshold value")
    
    else:
        logger.warning(f"Unknown command from client {client_id}: {action}")


def get_connection_manager() -> ConnectionManager:
    """
    Get the global ConnectionManager instance.
    
    Returns:
        Global ConnectionManager instance
    """
    return manager
