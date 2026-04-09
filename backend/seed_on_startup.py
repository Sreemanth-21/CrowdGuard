"""
Auto-seeder — called once at backend startup.

Checks if density logs and alerts are fresh (within last 2 hours).
If stale or missing, regenerates everything anchored to datetime.utcnow().
This means the analytics dashboard always has data regardless of when
the server is started.
"""

import uuid
import random
import math
from datetime import datetime, timedelta

from backend.database import SessionLocal
from backend.models.session import Session as SM
from backend.models.alert import Alert
from backend.models.density_log import DensityLog
from backend.utils.logger import get_logger

logger = get_logger(__name__)

ANOMALY_TYPES = [
    "HIGH_DENSITY", "RAPID_MOVEMENT", "SUDDEN_DISPERSAL",
    "CROWD_SURGE", "STATIONARY_CROWD", "FIGHTING",
]
RISK_LEVELS  = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
RISK_WEIGHTS = [0.10, 0.25, 0.35, 0.30]
DESCRIPTIONS = {
    "HIGH_DENSITY":     "High crowd density detected — area approaching capacity",
    "RAPID_MOVEMENT":   "Rapid crowd movement detected — possible panic or rush",
    "SUDDEN_DISPERSAL": "Sudden crowd dispersal detected — possible incident",
    "CROWD_SURGE":      "Crowd surge detected — directional pressure building",
    "STATIONARY_CROWD": "Large stationary crowd detected — potential bottleneck",
    "FIGHTING":         "Aggressive movement pattern detected — possible altercation",
}
SEED_SOURCE_NAME = "__auto_seed__"


def _is_data_fresh(db, now: datetime) -> bool:
    """Return True if we already have density logs within the last 2 hours."""
    cutoff = now - timedelta(hours=2)
    count = (
        db.query(DensityLog)
        .filter(DensityLog.timestamp >= cutoff)
        .count()
    )
    return count >= 10  # at least 10 recent points = data is fresh


def _make_density(t: datetime, idx: int) -> tuple:
    """Return (density, risk_score, person_count, mean_velocity) for timestamp t."""
    hour  = t.hour + t.minute / 60
    base  = 0.15 + 0.65 * math.exp(-0.5 * ((hour - 13) / 4) ** 2)
    wave  = 0.08 * math.sin(2 * math.pi * idx / 200)
    noise = random.gauss(0, 0.03)
    density      = max(0.05, min(0.95, base + wave + noise))
    risk_score   = min(100.0, max(0.0, density * 110 + random.gauss(0, 4)))
    person_count = max(0, int(density * 80 + random.gauss(0, 3)))
    velocity     = max(0.0, round(random.gauss(12, 5), 2))
    return round(density, 4), round(risk_score, 2), person_count, velocity


def run_auto_seed():
    """
    Main entry point — called from backend/main.py lifespan startup.
    Idempotent: does nothing if data is already fresh.
    """
    db  = SessionLocal()
    now = datetime.utcnow()

    try:
        if _is_data_fresh(db, now):
            logger.info("Auto-seed: data is fresh, skipping.")
            return

        logger.info("Auto-seed: data is stale — regenerating with current timestamp...")

        # ── Remove old auto-seed data ─────────────────────────────────────────
        old = db.query(SM).filter(SM.source_name == SEED_SOURCE_NAME).all()
        for s in old:
            db.query(Alert).filter(Alert.session_id == s.session_id).delete()
            db.query(DensityLog).filter(DensityLog.session_id == s.session_id).delete()
            db.delete(s)
        db.commit()

        # ── Create one seed session (active, covers last 24h) ─────────────────
        session = SM(
            session_id=str(uuid.uuid4()),
            start_time=now - timedelta(hours=24),
            end_time=None,                      # still "active"
            video_source_type="webcam",
            source_name=SEED_SOURCE_NAME,
            total_frames=0,
            total_alerts=0,
            peak_risk_score=0.0,
            average_density=0.45,
        )
        db.add(session)
        db.flush()
        sid = session.session_id

        # ── Density logs ──────────────────────────────────────────────────────
        # Last 2 hours  : 30-second intervals  (~240 points)
        # 2h – 24h ago  : 5-minute intervals   (~264 points)
        density_count = 0
        idx = 0

        t = now - timedelta(hours=2)
        while t <= now:
            d, r, p, v = _make_density(t, idx)
            db.add(DensityLog(
                session_id=sid, timestamp=t,
                density=d, risk_score=r, person_count=p, mean_velocity=v,
            ))
            t += timedelta(seconds=30)
            idx += 1
            density_count += 1

        t = now - timedelta(hours=24)
        while t < now - timedelta(hours=2):
            d, r, p, v = _make_density(t, idx)
            db.add(DensityLog(
                session_id=sid, timestamp=t,
                density=d, risk_score=r, person_count=p, mean_velocity=v,
            ))
            t += timedelta(minutes=5)
            idx += 1
            density_count += 1

        # ── Alerts spread across all time windows ─────────────────────────────
        # (start_mins_ago, end_mins_ago, count)
        buckets = [
            (0,    15,    8),
            (15,   30,    8),
            (30,   60,   12),
            (60,   180,  20),
            (180,  360,  20),
            (360,  720,  30),
            (720,  1440, 40),
        ]
        alert_count = 0
        for (start_ago, end_ago, count) in buckets:
            for _ in range(count):
                mins_ago = random.randint(start_ago, end_ago)
                ts       = now - timedelta(minutes=mins_ago)
                atype    = random.choice(ANOMALY_TYPES)
                level    = random.choices(RISK_LEVELS, weights=RISK_WEIGHTS, k=1)[0]
                db.add(Alert(
                    alert_id=str(uuid.uuid4()),
                    session_id=sid,
                    timestamp=ts,
                    anomaly_type=atype,
                    risk_level=level,
                    confidence_score=round(random.uniform(0.55, 0.99), 3),
                    description=DESCRIPTIONS[atype],
                    affected_persons=random.randint(3, 80),
                    location_x=random.randint(50, 1230),
                    location_y=random.randint(50, 670),
                    is_dismissed=random.random() < 0.12,
                ))
                alert_count += 1

        # Update session stats
        session.total_alerts    = alert_count
        session.peak_risk_score = 85.0
        session.total_frames    = density_count * 15   # rough estimate

        db.commit()
        logger.info(
            f"Auto-seed complete: {density_count} density logs, "
            f"{alert_count} alerts anchored to {now.strftime('%Y-%m-%d %H:%M')} UTC"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Auto-seed failed: {e}", exc_info=True)
    finally:
        db.close()
