"""
Seed demo data: sessions + alerts spread across all time windows.

Alert distribution (so every time-range selector shows different counts):
  Last 15 min  :  8 alerts
  15–30 min    :  8 alerts
  30–60 min    : 12 alerts
  1–3 hr       : 20 alerts
  3–6 hr       : 20 alerts
  6–12 hr      : 30 alerts
  12–24 hr     : 40 alerts
  Older        : 30 alerts (historical sessions)

Run: python seed_dummy_data.py
"""
import sys, uuid, random
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from backend.database import SessionLocal, init_db
from backend.models.session import Session as SM
from backend.models.alert import Alert
from backend.models.density_log import DensityLog

init_db()
db = SessionLocal()

TYPES  = ["HIGH_DENSITY","RAPID_MOVEMENT","SUDDEN_DISPERSAL","CROWD_SURGE","STATIONARY_CROWD","FIGHTING"]
LEVELS = ["SAFE","CAUTION","WARNING","CRITICAL"]
WEIGHTS = [0.10, 0.25, 0.35, 0.30]
DESCS  = {
    "HIGH_DENSITY":     "High crowd density detected — area approaching capacity",
    "RAPID_MOVEMENT":   "Rapid crowd movement detected — possible panic or rush",
    "SUDDEN_DISPERSAL": "Sudden crowd dispersal detected — possible incident",
    "CROWD_SURGE":      "Crowd surge detected — directional pressure building",
    "STATIONARY_CROWD": "Large stationary crowd detected — potential bottleneck",
    "FIGHTING":         "Aggressive movement pattern detected — possible altercation",
}
SOURCES = [("webcam","0"), ("upload","entrance_cam.mp4"), ("upload","food_court.mp4")]

NOW = datetime.utcnow()
random.seed(42)

# ── Clear old seed data ───────────────────────────────────────────────────────
# Only delete sessions that look like seed sessions (source_name in known list)
seed_names = {"0","entrance_cam.mp4","food_court.mp4","parking_lot.mp4"}
old_sessions = db.query(SM).filter(SM.source_name.in_(seed_names)).all()
for s in old_sessions:
    db.query(Alert).filter(Alert.session_id == s.session_id).delete()
    db.query(DensityLog).filter(DensityLog.session_id == s.session_id).delete()
    db.delete(s)
db.commit()
print(f"Cleared {len(old_sessions)} old seed sessions")

# ── Create 3 sessions ─────────────────────────────────────────────────────────
sessions = []
offsets = [(0, 120), (1440, 60), (2880, 45)]   # (start_mins_ago, duration_mins)
for start_ago, dur in offsets:
    start = NOW - timedelta(minutes=start_ago + dur)
    end   = NOW - timedelta(minutes=start_ago) if start_ago > 0 else NOW
    src   = random.choice(SOURCES)
    s = SM(
        session_id=str(uuid.uuid4()),
        start_time=start,
        end_time=end if start_ago > 0 else None,
        video_source_type=src[0],
        source_name=src[1],
        total_frames=random.randint(500, 5000),
        total_alerts=0,
        peak_risk_score=0.0,
        average_density=round(random.uniform(0.25, 0.75), 3),
    )
    db.add(s)
    sessions.append(s)
db.flush()

# ── Create alerts spread across time windows ──────────────────────────────────
# (window_start_mins_ago, window_end_mins_ago, count)
buckets = [
    (0,   15,   8),
    (15,  30,   8),
    (30,  60,  12),
    (60,  180, 20),
    (180, 360, 20),
    (360, 720, 30),
    (720, 1440, 40),
    (1440, 4320, 30),
]

alerts_added = 0
for (start_ago, end_ago, count) in buckets:
    # Pick a session that was active during this window
    sess = sessions[0] if start_ago < 200 else random.choice(sessions)
    for _ in range(count):
        mins_ago = random.randint(start_ago, end_ago)
        ts = NOW - timedelta(minutes=mins_ago)
        atype = random.choice(TYPES)
        level = random.choices(LEVELS, weights=WEIGHTS, k=1)[0]
        db.add(Alert(
            alert_id=str(uuid.uuid4()),
            session_id=sess.session_id,
            timestamp=ts,
            anomaly_type=atype,
            risk_level=level,
            confidence_score=round(random.uniform(0.55, 0.99), 3),
            description=DESCS[atype],
            affected_persons=random.randint(3, 80),
            location_x=random.randint(50, 1230),
            location_y=random.randint(50, 670),
            is_dismissed=random.random() < 0.12,
        ))
        alerts_added += 1

# Update session stats
for s in sessions:
    s.total_alerts = db.query(Alert).filter(Alert.session_id == s.session_id).count()
    from sqlalchemy import func
    peak = db.query(func.max(Alert.confidence_score)).filter(Alert.session_id == s.session_id).scalar()
    s.peak_risk_score = round((peak or 0) * 100, 1)

db.commit()
db.close()

print(f"Done — {len(sessions)} sessions, {alerts_added} alerts")
print("Alert counts per window:")
print("  15min: ~8  |  30min: ~16  |  1hr: ~28  |  3hr: ~48")
print("  6hr: ~68   |  12hr: ~98   |  24hr: ~138 |  all: ~168")
