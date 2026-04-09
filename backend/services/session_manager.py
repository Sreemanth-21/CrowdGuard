"""
Session Manager Service for CrowdGuard.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.models.session import Session as SessionModel
from backend.models.density_log import DensityLog
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    def __init__(self, db: Session, density_sample_interval: int = 5):
        self.db = db
        self.density_sample_interval = density_sample_interval
        self.last_density_sample_time: Optional[datetime] = None
        self.session_stats: Dict[str, Any] = {
            "total_frames": 0,
            "total_alerts": 0,
            "peak_risk_score": 0.0,
            "density_sum": 0.0,
            "density_count": 0
        }
        logger.info(f"SessionManager initialized with {density_sample_interval}s density sampling interval")

    def _get_active_session_record(self) -> Optional[SessionModel]:
        """Query DB for the currently active (not ended) session."""
        try:
            return self.db.query(SessionModel).filter(
                SessionModel.end_time == None
            ).order_by(SessionModel.start_time.desc()).first()
        except Exception as e:
            logger.error(f"Failed to query active session: {e}")
            return None

    @property
    def active_session_id(self) -> Optional[str]:
        """Derive active session ID from DB — never stale."""
        record = self._get_active_session_record()
        return record.session_id if record else None

    def create_session(self, source_type: str, source_name: str) -> str:
        # Check for existing active session in DB
        existing = self._get_active_session_record()
        if existing:
            raise ValueError(f"Session already active: {existing.session_id}")

        if source_type not in ("webcam", "upload"):
            raise ValueError(f"Invalid source_type: {source_type}")

        session_id = str(uuid.uuid4())
        session = SessionModel(
            session_id=session_id,
            start_time=datetime.utcnow(),
            end_time=None,
            video_source_type=source_type,
            source_name=source_name,
            total_frames=0,
            total_alerts=0,
            peak_risk_score=0.0,
            average_density=0.0
        )

        try:
            self.db.add(session)
            self.db.commit()
            self.last_density_sample_time = None
            self.session_stats = {
                "total_frames": 0,
                "total_alerts": 0,
                "peak_risk_score": 0.0,
                "density_sum": 0.0,
                "density_count": 0
            }
            logger.info(f"Session created: {session_id}")
            return session_id
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create session: {e}")
            raise

    def end_session(self, session_id: str) -> Dict[str, Any]:
        try:
            session = self.db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()

            if not session:
                raise ValueError(f"Session {session_id} not found in database")

            if session.end_time is not None:
                raise ValueError(f"Session {session_id} is already ended")

            end_time = datetime.utcnow()

            average_density = 0.0
            if self.session_stats["density_count"] > 0:
                average_density = self.session_stats["density_sum"] / self.session_stats["density_count"]

            session.end_time = end_time
            session.total_frames = self.session_stats["total_frames"]
            session.total_alerts = self.session_stats["total_alerts"]
            session.peak_risk_score = self.session_stats["peak_risk_score"]
            session.average_density = average_density

            self.db.commit()

            statistics = {
                "session_id": session_id,
                "start_time": session.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - session.start_time).total_seconds(),
                "total_frames": session.total_frames,
                "total_alerts": session.total_alerts,
                "peak_risk_score": session.peak_risk_score,
                "average_density": average_density
            }

            logger.info(f"Session ended: {session_id}")
            return statistics

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to end session {session_id}: {e}")
            raise

    def get_active_session(self) -> Optional[Dict[str, Any]]:
        """Get active session info — always queries DB."""
        session = self._get_active_session_record()
        if not session:
            return None

        try:
            average_density = 0.0
            if self.session_stats["density_count"] > 0:
                average_density = self.session_stats["density_sum"] / self.session_stats["density_count"]

            uptime_seconds = (datetime.utcnow() - session.start_time).total_seconds()

            return {
                "session_id": session.session_id,
                "source_type": session.video_source_type,
                "source_name": session.source_name,
                "start_time": session.start_time.isoformat(),
                "uptime_seconds": uptime_seconds,
                "total_frames": self.session_stats["total_frames"],
                "total_alerts": self.session_stats["total_alerts"],
                "peak_risk_score": self.session_stats["peak_risk_score"],
                "average_density": average_density
            }
        except Exception as e:
            logger.error(f"Failed to get active session info: {e}")
            return None

    def update_frame_stats(self, density: float, risk_score: float) -> None:
        self.session_stats["total_frames"] += 1
        self.session_stats["density_sum"] += density
        self.session_stats["density_count"] += 1
        if risk_score > self.session_stats["peak_risk_score"]:
            self.session_stats["peak_risk_score"] = risk_score

    def increment_alert_count(self) -> None:
        self.session_stats["total_alerts"] += 1

    def should_sample_density(self) -> bool:
        if self.active_session_id is None:
            return False
        if self.last_density_sample_time is None:
            return True
        elapsed = (datetime.utcnow() - self.last_density_sample_time).total_seconds()
        return elapsed >= self.density_sample_interval

    def sample_density_log(self, density: float, risk_score: float, person_count: int, mean_velocity: float) -> None:
        session_id = self.active_session_id
        if session_id is None:
            return
        if not self.should_sample_density():
            return

        try:
            log_entry = DensityLog(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                density=density,
                risk_score=risk_score,
                person_count=person_count,
                mean_velocity=mean_velocity
            )
            self.db.add(log_entry)
            self.db.commit()
            self.last_density_sample_time = datetime.utcnow()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to sample density log: {e}")

    def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = self.db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()

            if not session:
                return None

            duration_seconds = (
                (session.end_time - session.start_time).total_seconds()
                if session.end_time
                else (datetime.utcnow() - session.start_time).total_seconds()
            )

            return {
                "session_id": session.session_id,
                "source_type": session.video_source_type,
                "source_name": session.source_name,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "duration_seconds": duration_seconds,
                "total_frames": session.total_frames,
                "total_alerts": session.total_alerts,
                "peak_risk_score": session.peak_risk_score,
                "average_density": session.average_density,
                "is_active": session.end_time is None
            }
        except Exception as e:
            logger.error(f"Failed to get session statistics for {session_id}: {e}")
            return None