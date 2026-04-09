"""
Cleanup service for CrowdGuard system.

This module provides background cleanup tasks for managing storage:
- Delete alert snapshots older than 30 days
- Delete density log records older than 7 days
- Run cleanup tasks on a daily schedule

**Validates: Requirements 23.1, 23.2, 23.3, 23.4, 23.5, 23.6**
"""

import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from backend.models.alert import Alert
from backend.models.density_log import DensityLog
from backend.utils.logger import get_logger, log_cleanup_summary


logger = get_logger(__name__)


class CleanupService:
    """
    Background cleanup service for managing storage.
    
    Runs daily cleanup tasks to:
    - Delete alert snapshots older than 30 days (Requirement 23.1)
    - Delete density log records older than 7 days (Requirement 23.5)
    - Preserve alert metadata in database (Requirement 23.3)
    - Log deletion counts (Requirements 23.4, 23.6)
    
    Attributes:
        db: Database session for cleanup operations
        snapshot_dir: Directory containing alert snapshots
        cleanup_interval_seconds: Interval between cleanup runs (default: 24 hours)
        snapshot_retention_days: Days to retain snapshots (default: 30)
        density_log_retention_days: Days to retain density logs (default: 7)
        _cleanup_thread: Background thread running cleanup tasks
        _stop_event: Event to signal thread shutdown
        _running: Flag indicating if service is running
    """
    
    def __init__(
        self,
        db: Session,
        snapshot_dir: str = "snapshots",
        cleanup_interval_seconds: int = 86400,  # 24 hours
        snapshot_retention_days: int = 30,
        density_log_retention_days: int = 7
    ):
        """
        Initialize cleanup service.
        
        Args:
            db: Database session for cleanup operations
            snapshot_dir: Directory containing alert snapshots
            cleanup_interval_seconds: Interval between cleanup runs in seconds (default: 86400 = 24 hours)
            snapshot_retention_days: Days to retain snapshots (default: 30)
            density_log_retention_days: Days to retain density logs (default: 7)
        """
        self.db = db
        self.snapshot_dir = snapshot_dir
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.snapshot_retention_days = snapshot_retention_days
        self.density_log_retention_days = density_log_retention_days
        
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        logger.info(
            f"CleanupService initialized: snapshot_dir={snapshot_dir}, "
            f"cleanup_interval={cleanup_interval_seconds}s, "
            f"snapshot_retention={snapshot_retention_days} days, "
            f"density_log_retention={density_log_retention_days} days"
        )
    
    def start(self) -> None:
        """
        Start the cleanup service background thread.
        
        Starts a daemon thread that runs cleanup tasks at the configured interval.
        The thread will automatically stop when the main program exits.
        """
        if self._running:
            logger.warning("CleanupService already running")
            return
        
        self._stop_event.clear()
        self._running = True
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CleanupService"
        )
        self._cleanup_thread.start()
        
        logger.info("CleanupService started")
    
    def stop(self) -> None:
        """
        Stop the cleanup service background thread.
        
        Signals the background thread to stop and waits for it to complete.
        """
        if not self._running:
            logger.warning("CleanupService not running")
            return
        
        logger.info("Stopping CleanupService...")
        self._stop_event.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        self._running = False
        logger.info("CleanupService stopped")
    
    def _cleanup_loop(self) -> None:
        """
        Background loop that runs cleanup tasks at configured intervals.
        
        This method runs in a separate thread and performs cleanup tasks
        every cleanup_interval_seconds. It can be stopped by calling stop().
        """
        logger.info("CleanupService background loop started")
        
        while not self._stop_event.is_set():
            try:
                # Run cleanup tasks
                self.run_cleanup()
                
                # Wait for next cleanup interval or stop signal
                self._stop_event.wait(timeout=self.cleanup_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                # Continue running even if cleanup fails
                self._stop_event.wait(timeout=60)  # Wait 1 minute before retry
        
        logger.info("CleanupService background loop stopped")
    
    def run_cleanup(self) -> tuple[int, int]:
        """
        Run all cleanup tasks.
        
        Performs:
        1. Snapshot cleanup (delete files older than 30 days)
        2. Density log cleanup (delete records older than 7 days)
        
        Returns:
            Tuple of (snapshots_deleted, density_logs_deleted)
        """
        logger.info("Starting cleanup tasks...")
        
        snapshots_deleted = self.cleanup_snapshots()
        density_logs_deleted = self.cleanup_density_logs()
        
        # Log summary (Requirements 23.4, 23.6)
        log_cleanup_summary(logger, snapshots_deleted, density_logs_deleted)
        
        return snapshots_deleted, density_logs_deleted
    
    def cleanup_snapshots(self) -> int:
        """
        Delete alert snapshots older than retention period.
        
        Deletes snapshot files from disk while preserving alert metadata
        in the database (Requirement 23.3).
        
        Returns:
            Number of snapshots deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.snapshot_retention_days)
        deleted_count = 0
        
        logger.info(
            f"Cleaning up snapshots older than {self.snapshot_retention_days} days "
            f"(before {cutoff_date.isoformat()})"
        )
        
        try:
            # Query alerts with snapshots older than cutoff date
            old_alerts = self.db.query(Alert).filter(
                Alert.timestamp < cutoff_date,
                Alert.frame_snapshot_path.isnot(None)
            ).all()
            
            for alert in old_alerts:
                if alert.frame_snapshot_path and os.path.exists(alert.frame_snapshot_path):
                    try:
                        # Delete the snapshot file
                        os.remove(alert.frame_snapshot_path)
                        deleted_count += 1
                        
                        logger.debug(
                            f"Deleted snapshot: {alert.frame_snapshot_path} "
                            f"(alert_id={alert.alert_id})"
                        )
                        
                        # Clear the snapshot path in database but keep alert metadata
                        alert.frame_snapshot_path = None
                        
                    except OSError as e:
                        logger.error(
                            f"Failed to delete snapshot {alert.frame_snapshot_path}: {e}"
                        )
            
            # Commit database changes
            self.db.commit()
            
            logger.info(f"Snapshot cleanup completed: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"Error during snapshot cleanup: {e}", exc_info=True)
            self.db.rollback()
        
        return deleted_count
    
    def cleanup_density_logs(self) -> int:
        """
        Delete density log records older than retention period.
        
        Deletes old density log records from the database to manage storage
        (Requirement 23.5).
        
        Returns:
            Number of density log records deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.density_log_retention_days)
        deleted_count = 0
        
        logger.info(
            f"Cleaning up density logs older than {self.density_log_retention_days} days "
            f"(before {cutoff_date.isoformat()})"
        )
        
        try:
            # Delete density logs older than cutoff date
            result = self.db.query(DensityLog).filter(
                DensityLog.timestamp < cutoff_date
            ).delete(synchronize_session=False)
            
            deleted_count = result
            
            # Commit database changes
            self.db.commit()
            
            logger.info(f"Density log cleanup completed: {deleted_count} records deleted")
            
        except Exception as e:
            logger.error(f"Error during density log cleanup: {e}", exc_info=True)
            self.db.rollback()
        
        return deleted_count
    
    def is_running(self) -> bool:
        """
        Check if cleanup service is running.
        
        Returns:
            True if service is running, False otherwise
        """
        return self._running
