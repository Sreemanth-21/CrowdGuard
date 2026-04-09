"""
Alerts Router for CrowdGuard.

This module provides REST API endpoints for alert querying, filtering,
dismissal, summary statistics, and CSV export functionality.

**Validates: Requirements 21.1-21.7, 15.1-15.4**
"""

import os
import csv
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from backend.database import get_db
from backend.models.alert import Alert
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/alerts", tags=["alerts"])

# Constants
SNAPSHOT_DIR = "snapshots"


# Request/Response Models
class AlertResponse(BaseModel):
    """Response model for a single alert."""
    alert_id: str
    session_id: str
    timestamp: str
    anomaly_type: str
    risk_level: str
    confidence_score: float
    description: str
    frame_snapshot_path: Optional[str]
    affected_persons: int
    location_x: Optional[int]
    location_y: Optional[int]
    is_dismissed: bool
    dismissed_at: Optional[str]


class AlertsQueryResponse(BaseModel):
    """Response model for paginated alerts query."""
    alerts: List[AlertResponse]
    pagination: Dict[str, Any]


class DismissResponse(BaseModel):
    """Response model for alert dismissal."""
    alert_id: str
    is_dismissed: bool
    dismissed_at: str


class BulkDismissRequest(BaseModel):
    """Request model for bulk dismissal."""
    alert_ids: List[str] = Field(..., description="List of alert IDs to dismiss")


class BulkDismissResponse(BaseModel):
    """Response model for bulk dismissal."""
    dismissed_count: int
    dismissed_at: str


class AlertSummaryResponse(BaseModel):
    """Response model for alert summary statistics."""
    total_alerts: int
    active_alerts: int
    by_type: Dict[str, int]
    by_risk_level: Dict[str, int]


# Helper Functions
def alert_to_response(alert: Alert) -> AlertResponse:
    """
    Convert Alert model to AlertResponse.
    
    Args:
        alert: Alert ORM model
        
    Returns:
        AlertResponse object
    """
    return AlertResponse(
        alert_id=alert.alert_id,
        session_id=alert.session_id,
        timestamp=alert.timestamp.isoformat() + "Z" if alert.timestamp else None,
        anomaly_type=alert.anomaly_type,
        risk_level=alert.risk_level,
        confidence_score=alert.confidence_score,
        description=alert.description,
        frame_snapshot_path=alert.frame_snapshot_path,
        affected_persons=alert.affected_persons,
        location_x=alert.location_x,
        location_y=alert.location_y,
        is_dismissed=alert.is_dismissed,
        dismissed_at=alert.dismissed_at.isoformat() + "Z" if alert.dismissed_at else None
    )


# API Endpoints

@router.get("", response_model=AlertsQueryResponse, status_code=status.HTTP_200_OK)
async def query_alerts(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records to return"),
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type (comma-separated)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (comma-separated)"),
    start_date: Optional[str] = Query(None, description="Filter start date (ISO 8601)"),
    end_date: Optional[str] = Query(None, description="Filter end date (ISO 8601)"),
    dismissed: Optional[bool] = Query(None, description="Filter by dismissed status"),
    db: Session = Depends(get_db)
) -> AlertsQueryResponse:
    """
    Query alerts with pagination and filtering.
    
    Supports filtering by:
    - anomaly_type: Comma-separated list of anomaly types
    - risk_level: Comma-separated list of risk levels
    - start_date/end_date: Date range filter
    - dismissed: Include/exclude dismissed alerts
    
    Returns alerts in reverse chronological order (newest first).
    
    Args:
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        anomaly_type: Filter by anomaly type(s)
        risk_level: Filter by risk level(s)
        start_date: Filter start date
        end_date: Filter end date
        dismissed: Filter by dismissed status
        db: Database session
        
    Returns:
        AlertsQueryResponse with paginated alerts
        
    **Validates: Requirements 21.1, 21.2, 21.3, 21.4, 21.5**
    """
    try:
        # Build query with filters
        query = db.query(Alert)
        
        # Filter by anomaly type
        if anomaly_type:
            types = [t.strip() for t in anomaly_type.split(",")]
            query = query.filter(Alert.anomaly_type.in_(types))
        
        # Filter by risk level
        if risk_level:
            levels = [l.strip() for l in risk_level.split(",")]
            query = query.filter(Alert.risk_level.in_(levels))
        
        # Filter by date range
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                query = query.filter(Alert.timestamp >= start_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Invalid start_date format. Use ISO 8601 format."}
                )
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                query = query.filter(Alert.timestamp <= end_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Invalid end_date format. Use ISO 8601 format."}
                )
        
        # Filter by dismissed status
        if dismissed is not None:
            query = query.filter(Alert.is_dismissed == dismissed)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply ordering (newest first) and pagination
        alerts = query.order_by(Alert.timestamp.desc()).offset(skip).limit(limit).all()
        
        # Convert to response models
        alert_responses = [alert_to_response(alert) for alert in alerts]
        
        # Calculate pagination info
        pages = (total + limit - 1) // limit if limit > 0 else 0
        current_page = (skip // limit) + 1 if limit > 0 else 1
        
        logger.debug(
            f"Query alerts: {len(alert_responses)} results, "
            f"page {current_page}/{pages}, total={total}"
        )
        
        return AlertsQueryResponse(
            alerts=alert_responses,
            pagination={
                "page": current_page,
                "per_page": limit,
                "total": total,
                "pages": pages
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to query alerts"}
        )


@router.get("/summary", response_model=AlertSummaryResponse, status_code=status.HTTP_200_OK)
async def get_alert_summary(
    db: Session = Depends(get_db)
) -> AlertSummaryResponse:
    """
    Get alert summary statistics.
    
    Returns:
    - Total alert count
    - Active (not dismissed) alert count
    - Counts by anomaly type
    - Counts by risk level
    
    Args:
        db: Database session
        
    Returns:
        AlertSummaryResponse with summary statistics
        
    **Validates: Requirement 21.6**
    """
    try:
        # Get total alerts
        total_alerts = db.query(Alert).count()
        
        # Get active alerts (not dismissed)
        active_alerts = db.query(Alert).filter(Alert.is_dismissed == False).count()
        
        # Get counts by anomaly type
        type_counts = db.query(
            Alert.anomaly_type,
            func.count(Alert.alert_id).label("count")
        ).group_by(Alert.anomaly_type).all()
        
        by_type = {row.anomaly_type: row.count for row in type_counts}
        
        # Get counts by risk level
        level_counts = db.query(
            Alert.risk_level,
            func.count(Alert.alert_id).label("count")
        ).group_by(Alert.risk_level).all()
        
        by_risk_level = {row.risk_level: row.count for row in level_counts}
        
        logger.debug(
            f"Alert summary: total={total_alerts}, active={active_alerts}, "
            f"types={len(by_type)}, levels={len(by_risk_level)}"
        )
        
        return AlertSummaryResponse(
            total_alerts=total_alerts,
            active_alerts=active_alerts,
            by_type=by_type,
            by_risk_level=by_risk_level
        )
        
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get alert summary"}
        )


@router.get("/export", status_code=status.HTTP_200_OK)
async def export_alerts_csv(
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type (comma-separated)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (comma-separated)"),
    start_date: Optional[str] = Query(None, description="Filter start date (ISO 8601)"),
    end_date: Optional[str] = Query(None, description="Filter end date (ISO 8601)"),
    dismissed: Optional[bool] = Query(None, description="Filter by dismissed status"),
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Export alerts to CSV file.
    
    Supports same filtering options as GET /api/alerts.
    Returns CSV file with all alert fields.
    
    Args:
        anomaly_type: Filter by anomaly type(s)
        risk_level: Filter by risk level(s)
        start_date: Filter start date
        end_date: Filter end date
        dismissed: Filter by dismissed status
        db: Database session
        
    Returns:
        StreamingResponse with CSV file
        
    **Validates: Requirement 21.7**
    """
    try:
        # Build query with same filters as query_alerts
        query = db.query(Alert)
        
        # Filter by anomaly type
        if anomaly_type:
            types = [t.strip() for t in anomaly_type.split(",")]
            query = query.filter(Alert.anomaly_type.in_(types))
        
        # Filter by risk level
        if risk_level:
            levels = [l.strip() for l in risk_level.split(",")]
            query = query.filter(Alert.risk_level.in_(levels))
        
        # Filter by date range
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                query = query.filter(Alert.timestamp >= start_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Invalid start_date format. Use ISO 8601 format."}
                )
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                query = query.filter(Alert.timestamp <= end_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Invalid end_date format. Use ISO 8601 format."}
                )
        
        # Filter by dismissed status
        if dismissed is not None:
            query = query.filter(Alert.is_dismissed == dismissed)
        
        # Get all alerts (ordered by timestamp)
        alerts = query.order_by(Alert.timestamp.desc()).all()
        
        # Generate CSV content
        def generate_csv():
            """Generator function for CSV content."""
            # CSV header
            yield "timestamp,anomaly_type,risk_level,confidence_score,session_id,description,affected_persons,location_x,location_y,is_dismissed,dismissed_at\n"
            
            # CSV rows
            for alert in alerts:
                row = [
                    alert.timestamp.isoformat() if alert.timestamp else "",
                    alert.anomaly_type,
                    alert.risk_level,
                    str(alert.confidence_score),
                    alert.session_id,
                    f'"{alert.description}"',  # Quote description to handle commas
                    str(alert.affected_persons),
                    str(alert.location_x) if alert.location_x is not None else "",
                    str(alert.location_y) if alert.location_y is not None else "",
                    str(alert.is_dismissed),
                    alert.dismissed_at.isoformat() if alert.dismissed_at else ""
                ]
                yield ",".join(row) + "\n"
        
        logger.info(f"Exporting {len(alerts)} alerts to CSV")
        
        return StreamingResponse(
            generate_csv(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=alerts_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export alerts to CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to export alerts"}
        )


@router.get("/{alert_id}", response_model=AlertResponse, status_code=status.HTTP_200_OK)
async def get_alert(
    alert_id: str,
    db: Session = Depends(get_db)
) -> AlertResponse:
    """
    Get single alert details by ID.
    
    Args:
        alert_id: Alert identifier
        db: Database session
        
    Returns:
        AlertResponse with alert details
        
    Raises:
        HTTPException 404: If alert not found
        
    **Validates: Requirement 15.1**
    """
    try:
        alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Alert not found"}
            )
        
        logger.debug(f"Retrieved alert: {alert_id}")
        
        return alert_to_response(alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve alert"}
        )


@router.get("/{alert_id}/snapshot", status_code=status.HTTP_200_OK)
async def get_alert_snapshot(
    alert_id: str,
    db: Session = Depends(get_db)
) -> FileResponse:
    """
    Serve alert snapshot image file.
    
    Args:
        alert_id: Alert identifier
        db: Database session
        
    Returns:
        FileResponse with snapshot image
        
    Raises:
        HTTPException 404: If alert or snapshot not found
        
    **Validates: Requirement 21.5**
    """
    try:
        # Get alert from database
        alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Alert not found"}
            )
        
        # Check if snapshot path exists
        if not alert.frame_snapshot_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Snapshot not available for this alert"}
            )
        
        # Check if file exists on disk
        if not os.path.exists(alert.frame_snapshot_path):
            logger.warning(
                f"Snapshot file not found on disk: {alert.frame_snapshot_path}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Snapshot file not found"}
            )
        
        logger.debug(f"Serving snapshot for alert {alert_id}: {alert.frame_snapshot_path}")
        
        return FileResponse(
            path=alert.frame_snapshot_path,
            media_type="image/jpeg",
            filename=f"alert_{alert_id}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve snapshot for alert {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to serve snapshot"}
        )


@router.put("/{alert_id}/dismiss", response_model=DismissResponse, status_code=status.HTTP_200_OK)
async def dismiss_alert(
    alert_id: str,
    db: Session = Depends(get_db)
) -> DismissResponse:
    """
    Dismiss a single alert.
    
    Sets dismissed=True and records dismissal timestamp.
    
    Args:
        alert_id: Alert identifier
        db: Database session
        
    Returns:
        DismissResponse with updated alert status
        
    Raises:
        HTTPException 404: If alert not found
        
    **Validates: Requirements 15.1, 15.2, 15.4**
    """
    try:
        # Get alert from database
        alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Alert not found"}
            )
        
        # Update dismissal status
        dismissed_at = datetime.utcnow()
        alert.is_dismissed = True
        alert.dismissed_at = dismissed_at
        
        db.commit()
        db.refresh(alert)
        
        logger.info(f"Alert dismissed: {alert_id}")
        
        return DismissResponse(
            alert_id=alert.alert_id,
            is_dismissed=alert.is_dismissed,
            dismissed_at=dismissed_at.isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to dismiss alert {alert_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to dismiss alert"}
        )


@router.post("/bulk-dismiss", response_model=BulkDismissResponse, status_code=status.HTTP_200_OK)
async def bulk_dismiss_alerts(
    request: BulkDismissRequest,
    db: Session = Depends(get_db)
) -> BulkDismissResponse:
    """
    Dismiss multiple alerts at once.
    
    Args:
        request: Bulk dismiss request with list of alert IDs
        db: Database session
        
    Returns:
        BulkDismissResponse with count of dismissed alerts
        
    **Validates: Requirement 15.4**
    """
    try:
        dismissed_at = datetime.utcnow()
        
        # Update all specified alerts
        result = db.query(Alert).filter(
            Alert.alert_id.in_(request.alert_ids)
        ).update(
            {
                Alert.is_dismissed: True,
                Alert.dismissed_at: dismissed_at
            },
            synchronize_session=False
        )
        
        db.commit()
        
        logger.info(f"Bulk dismissed {result} alerts")
        
        return BulkDismissResponse(
            dismissed_count=result,
            dismissed_at=dismissed_at.isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Failed to bulk dismiss alerts: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to bulk dismiss alerts"}
        )

