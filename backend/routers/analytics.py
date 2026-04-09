"""
Analytics Router for CrowdGuard.

This module provides REST API endpoints for analytics data retrieval,
including time-series data, session statistics, and KPI metrics.

**Validates: Requirements 19.1-19.6, 20.1-20.4**
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from backend.database import get_db
from backend.models.density_log import DensityLog
from backend.models.alert import Alert
from backend.models.session import Session as SessionModel
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# Response Models
class TimeSeriesDataPoint(BaseModel):
    """Single data point in time series."""
    timestamp: str
    value: float
    person_count: Optional[int] = None


class DensityTimeSeriesResponse(BaseModel):
    """Response model for density time series."""
    data: List[TimeSeriesDataPoint]
    start_time: str
    end_time: str
    count: int


class RiskTimeSeriesResponse(BaseModel):
    """Response model for risk time series."""
    data: List[TimeSeriesDataPoint]
    start_time: str
    end_time: str
    count: int


class AlertFrequencyBucket(BaseModel):
    """Alert frequency bucket."""
    time_bucket: str
    count: int


class AlertFrequencyResponse(BaseModel):
    """Response model for alert frequency analysis."""
    buckets: List[AlertFrequencyBucket]
    interval_minutes: int
    total_alerts: int


class SessionStatsResponse(BaseModel):
    """Response model for session statistics."""
    session_id: str
    total_frames: int
    total_alerts: int
    peak_risk_score: float
    average_density: float
    start_time: str
    end_time: Optional[str]
    duration_minutes: float


class KPIResponse(BaseModel):
    """Response model for key performance indicators."""
    average_density: float
    total_alerts: int
    peak_risk_score: float
    session_duration_minutes: float
    session_id: Optional[str]


# API Endpoints

@router.get("/density-timeseries", response_model=DensityTimeSeriesResponse, status_code=status.HTTP_200_OK)
async def get_density_timeseries(
    minutes: int = Query(60, ge=1, le=10080, description="Time window in minutes (default 60)"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    db: Session = Depends(get_db)
) -> DensityTimeSeriesResponse:
    try:
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(minutes=minutes)

        query = db.query(DensityLog).filter(
            and_(DensityLog.timestamp >= start_time, DensityLog.timestamp <= end_time)
        )
        if session_id:
            query = query.filter(DensityLog.session_id == session_id)

        logs = query.order_by(DensityLog.timestamp.asc()).all()
        data = [
            TimeSeriesDataPoint(
                timestamp=log.timestamp.isoformat() + "Z",
                value=log.density,
                person_count=log.person_count,
            )
            for log in logs
        ]
        logger.debug(f"Density timeseries: {len(data)} points for last {minutes} min")
        return DensityTimeSeriesResponse(
            data=data,
            start_time=start_time.isoformat() + "Z",
            end_time=end_time.isoformat() + "Z",
            count=len(data),
        )
    except Exception as e:
        logger.error(f"Failed to retrieve density time series: {e}")
        raise HTTPException(status_code=500, detail={"error": "Failed to retrieve density time series"})


@router.get("/risk-timeseries", response_model=RiskTimeSeriesResponse, status_code=status.HTTP_200_OK)
async def get_risk_timeseries(
    minutes: int = Query(60, ge=1, le=10080, description="Time window in minutes (default 60)"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    db: Session = Depends(get_db)
) -> RiskTimeSeriesResponse:
    try:
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(minutes=minutes)

        query = db.query(DensityLog).filter(
            and_(DensityLog.timestamp >= start_time, DensityLog.timestamp <= end_time)
        )
        if session_id:
            query = query.filter(DensityLog.session_id == session_id)

        logs = query.order_by(DensityLog.timestamp.asc()).all()
        data = [
            TimeSeriesDataPoint(
                timestamp=log.timestamp.isoformat() + "Z",
                value=log.risk_score,
            )
            for log in logs
        ]
        logger.debug(f"Risk timeseries: {len(data)} points for last {minutes} min")
        return RiskTimeSeriesResponse(
            data=data,
            start_time=start_time.isoformat() + "Z",
            end_time=end_time.isoformat() + "Z",
            count=len(data),
        )
    except Exception as e:
        logger.error(f"Failed to retrieve risk time series: {e}")
        raise HTTPException(status_code=500, detail={"error": "Failed to retrieve risk time series"})


@router.get("/alert-frequency", response_model=AlertFrequencyResponse, status_code=status.HTTP_200_OK)
async def get_alert_frequency(
    interval_minutes: int = Query(5, ge=1, le=60, description="Time bucket interval in minutes"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    db: Session = Depends(get_db)
) -> AlertFrequencyResponse:
    """
    Get alert frequency grouped by time intervals.
    
    Retrieves alerts from the past 60 minutes and groups them into
    time buckets of the specified interval size.
    
    Args:
        interval_minutes: Size of time buckets in minutes (default: 5)
        session_id: Optional session ID filter
        db: Database session
        
    Returns:
        AlertFrequencyResponse with alert counts per time bucket
        
    **Validates: Requirements 19.4, 19.5, 19.6**
    """
    try:
        # Calculate time window (past 60 minutes)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=60)
        
        # Build query
        query = db.query(Alert).filter(
            and_(
                Alert.timestamp >= start_time,
                Alert.timestamp <= end_time
            )
        )
        
        # Filter by session if provided
        if session_id:
            query = query.filter(Alert.session_id == session_id)
        
        # Get all alerts in time window
        alerts = query.order_by(Alert.timestamp.asc()).all()
        
        # Group alerts into time buckets
        buckets: Dict[str, int] = {}
        interval_seconds = interval_minutes * 60
        
        for alert in alerts:
            # Calculate bucket start time
            seconds_since_start = (alert.timestamp - start_time).total_seconds()
            bucket_index = int(seconds_since_start // interval_seconds)
            bucket_start = start_time + timedelta(seconds=bucket_index * interval_seconds)
            bucket_key = bucket_start.isoformat() + "Z"
            
            # Increment bucket count
            buckets[bucket_key] = buckets.get(bucket_key, 0) + 1
        
        # Convert to response format
        bucket_list = [
            AlertFrequencyBucket(time_bucket=key, count=count)
            for key, count in sorted(buckets.items())
        ]
        
        logger.debug(
            f"Retrieved alert frequency: {len(alerts)} alerts in "
            f"{len(bucket_list)} buckets ({interval_minutes}min intervals)"
        )
        
        return AlertFrequencyResponse(
            buckets=bucket_list,
            interval_minutes=interval_minutes,
            total_alerts=len(alerts)
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve alert frequency: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve alert frequency"}
        )


@router.get("/session-stats", response_model=SessionStatsResponse, status_code=status.HTTP_200_OK)
async def get_session_stats(
    session_id: Optional[str] = Query(None, description="Session ID (defaults to most recent)"),
    db: Session = Depends(get_db)
) -> SessionStatsResponse:
    """
    Get statistics for a specific session.
    
    Returns comprehensive statistics including total frames, alerts,
    peak risk score, and average density. If no session_id is provided,
    returns stats for the most recent session.
    
    Args:
        session_id: Optional session ID (defaults to most recent)
        db: Database session
        
    Returns:
        SessionStatsResponse with session statistics
        
    Raises:
        HTTPException 404: If session not found
        
    **Validates: Requirements 20.1, 20.2, 20.3, 20.4**
    """
    try:
        # Get session
        if session_id:
            session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
        else:
            # Get most recent session
            session = db.query(SessionModel).order_by(
                SessionModel.start_time.desc()
            ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Session not found"}
            )
        
        # Calculate duration
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds() / 60
        else:
            # Session is still active
            duration = (datetime.utcnow() - session.start_time).total_seconds() / 60
        
        logger.debug(f"Retrieved session stats for session {session.session_id}")
        
        return SessionStatsResponse(
            session_id=session.session_id,
            total_frames=session.total_frames,
            total_alerts=session.total_alerts,
            peak_risk_score=session.peak_risk_score,
            average_density=session.average_density,
            start_time=session.start_time.isoformat() + "Z",
            end_time=session.end_time.isoformat() + "Z" if session.end_time else None,
            duration_minutes=round(duration, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve session stats"}
        )


@router.get("/kpis", response_model=KPIResponse, status_code=status.HTTP_200_OK)
async def get_kpis(
    minutes: int = Query(60, ge=1, le=10080, description="Time window in minutes (default 60)"),
    session_id: Optional[str] = Query(None, description="Session ID filter (optional)"),
    db: Session = Depends(get_db)
) -> KPIResponse:
    """
    Get KPIs aggregated over the specified time window.
    Changing `minutes` changes all four numbers.
    """
    try:
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(minutes=minutes)

        if session_id:
            session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            if not session:
                raise HTTPException(status_code=404, detail={"error": "Session not found"})
            duration = (
                (session.end_time - session.start_time).total_seconds() / 60
                if session.end_time
                else (datetime.utcnow() - session.start_time).total_seconds() / 60
            )
            return KPIResponse(
                average_density=session.average_density,
                total_alerts=session.total_alerts,
                peak_risk_score=session.peak_risk_score,
                session_duration_minutes=round(duration, 2),
                session_id=session.session_id,
            )

        # Aggregate density logs within the time window
        density_agg = db.query(
            func.avg(DensityLog.density).label("avg_density"),
            func.max(DensityLog.risk_score).label("peak_risk"),
        ).filter(
            and_(DensityLog.timestamp >= start_time, DensityLog.timestamp <= end_time)
        ).first()

        # Count alerts within the time window
        alert_count = db.query(func.count(Alert.alert_id)).filter(
            and_(Alert.timestamp >= start_time, Alert.timestamp <= end_time)
        ).scalar() or 0

        # Total session time that overlaps with the window
        sessions_in_window = db.query(SessionModel).filter(
            SessionModel.start_time <= end_time,
            func.coalesce(SessionModel.end_time, end_time) >= start_time,
        ).all()
        total_minutes = 0.0
        for s in sessions_in_window:
            s_start = max(s.start_time, start_time)
            s_end   = min(s.end_time or end_time, end_time)
            total_minutes += max(0, (s_end - s_start).total_seconds() / 60)

        avg_density = float(density_agg.avg_density or 0.0)
        peak_risk   = float(density_agg.peak_risk   or 0.0)

        logger.debug(
            f"KPIs [{minutes}min]: density={avg_density:.3f}, "
            f"alerts={alert_count}, peak={peak_risk:.1f}, duration={total_minutes:.1f}min"
        )

        return KPIResponse(
            average_density=round(avg_density, 3),
            total_alerts=int(alert_count),
            peak_risk_score=round(peak_risk, 1),
            session_duration_minutes=round(total_minutes, 1),
            session_id=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve KPIs: {e}")
        raise HTTPException(status_code=500, detail={"error": "Failed to retrieve KPIs"})
