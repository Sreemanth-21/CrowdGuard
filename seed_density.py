"""
Seed density logs with two resolutions:
  - Last 2 hours  : one point every 30 seconds  (~240 points, always covers 15/30/60/180 min windows)
  - 2h–24h ago    : one point every 5 minutes   (~264 points, covers 6/12/24 hr windows)

Total: ~504 points.  Run any time to refresh data to NOW.
Run: python seed_density.py
"""
import sys, random, math
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from backend.database import SessionLocal, init_db
from backend.models.density_log import DensityLog
from backend.models.session import Session as SM

init_db()
db = SessionLocal()

# Use the session with the most alerts
sess = db.query(SM).order_by(SM.total_alerts.desc()).first()
if not sess:
    print("No sessions found — run seed_dummy_data.py first")
    db.close()
    sys.exit(1)

print(f"Seeding density logs for session {sess.session_id[:8]}...")

# Clear existing density logs for this session to avoid duplicates
deleted = db.query(DensityLog).filter(DensityLog.session_id == sess.session_id).delete()
db.commit()
print(f"Cleared {deleted} old density logs")

NOW = datetime.utcnow()
random.seed(int(NOW.timestamp()) // 3600)  # changes every hour → fresh but stable within an hour

def make_point(t: datetime, i: int) -> DensityLog:
    """Generate a realistic density log entry for timestamp t."""
    hour = t.hour + t.minute / 60
    # Bell curve centred at 13:00, low at night
    base = 0.15 + 0.65 * math.exp(-0.5 * ((hour - 13) / 4) ** 2)
    wave  = 0.08 * math.sin(2 * math.pi * i / 200)
    noise = random.gauss(0, 0.03)
    density    = max(0.05, min(0.95, base + wave + noise))
    risk_score = min(100.0, max(0.0, density * 110 + random.gauss(0, 4)))
    person_count = max(0, int(density * 80 + random.gauss(0, 3)))
    return DensityLog(
        session_id=sess.session_id,
        timestamp=t,
        density=round(density, 4),
        risk_score=round(risk_score, 2),
        person_count=person_count,
        mean_velocity=round(max(0.0, random.gauss(12, 5)), 2),
    )

added = 0

# ── Dense zone: last 2 hours at 30-second intervals ──────────────────────────
t = NOW - timedelta(hours=2)
i = 0
while t <= NOW:
    db.add(make_point(t, i))
    t += timedelta(seconds=30)
    i += 1
    added += 1

# ── Sparse zone: 2h–24h ago at 5-minute intervals ────────────────────────────
t = NOW - timedelta(hours=24)
while t < NOW - timedelta(hours=2):
    db.add(make_point(t, i))
    t += timedelta(minutes=5)
    i += 1
    added += 1

db.commit()
db.close()
print(f"Done — added {added} density logs")
print(f"  Last 2h  : 30-second intervals (~240 points)")
print(f"  2h–24h   : 5-minute intervals  (~264 points)")
