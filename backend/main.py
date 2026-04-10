"""
Main FastAPI application for CrowdGuard backend

Integrates all routers and services for the crowd anomaly detection system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.database import init_db, SessionLocal
from backend.routers import health, video, alerts, analytics, settings, websocket, federated
from backend.routers.settings import set_video_processor
from backend.services.cleanup_service import CleanupService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Global cleanup service instance
cleanup_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cleanup_service

    # Startup
    logger.info("Starting CrowdGuard backend application")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Auto-seed demo data anchored to current time (idempotent — skips if fresh)
    from backend.seed_on_startup import run_auto_seed
    run_auto_seed()

    # Initialize cleanup service
    cleanup_service = CleanupService(
        db=SessionLocal(),
        snapshot_dir="uploads/snapshots",
        cleanup_interval_seconds=86400,
        snapshot_retention_days=30,
        density_log_retention_days=7
    )
    cleanup_service.start()
    logger.info("Cleanup service started")

    # Set video processor for settings hot-reload
    from backend.ml.video_processor import VideoProcessor
    from backend.config import ConfigManager
    config_manager = ConfigManager(db=SessionLocal())
    video_processor = VideoProcessor(config_manager)
    set_video_processor(video_processor)
    logger.info("Video processor configured")

    yield

    # Shutdown
    logger.info("Shutting down CrowdGuard backend application")
    if cleanup_service:
        cleanup_service.stop()
        logger.info("Cleanup service stopped")


# Create FastAPI application
app = FastAPI(
    title="CrowdGuard API",
    description="AI-powered crowd anomaly detection system",
    version="1.0.0",
    lifespan=lifespan,
)

# No body size limit — video uploads can be up to 500 MB
# Uvicorn default is unlimited; this makes it explicit
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    # No size-limiting middleware added — intentional
except ImportError:
    pass

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(video.router)
app.include_router(alerts.router)
app.include_router(analytics.router)
app.include_router(settings.router)
app.include_router(websocket.router)
app.include_router(federated.router)

logger.info("All routers registered")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CrowdGuard API",
        "version": "1.0.0",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )