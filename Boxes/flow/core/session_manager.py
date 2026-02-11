"""Session manager for handling multiple concurrent detection sessions."""

import asyncio
import logging
import threading
from typing import Dict, Optional, Union

from core.mqtt_client import MqttClient
from core.session_worker import SessionWorker

logger = logging.getLogger(__name__)


class SessionManager:
    """Singleton for managing active detection sessions with thread-safe operations."""

    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.sessions: Dict[str, SessionWorker] = {}
        self.camera_locks: Dict[str, str] = {}  # camera_source -> report_id
        self.sessions_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_session(
        self,
        report_id: str,
        production_line: str,
        camera_source: Union[str, int],
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        mqtt_client: MqttClient,
        loop: asyncio.AbstractEventLoop,
    ) -> SessionWorker:
        """
        Create and start a new headless detection session.

        Args:
            report_id: Unique session identifier
            production_line: Production line identifier
            camera_source: Camera device path or index
            box_cfg: Box detector configuration
            defect_cfg: Defect detector configuration
            stream_cfg: Stream configuration (source overridden by camera_source)
            mqtt_client: MQTT client for publishing insights
            loop: Main asyncio event loop for broadcasting

        Returns:
            SessionWorker instance

        Raises:
            ValueError: If session exists or camera is in use
        """
        camera_key = str(camera_source)
        with self.sessions_lock:
            if report_id in self.sessions:
                raise ValueError(f"Session {report_id} already exists")
            if camera_key in self.camera_locks:
                locked_by = self.camera_locks[camera_key]
                raise ValueError(
                    f"Camera {camera_source} in use by session {locked_by}"
                )

            worker = SessionWorker(
                report_id=report_id,
                production_line=production_line,
                camera_source=camera_source,
                box_cfg=box_cfg,
                defect_cfg=defect_cfg,
                stream_cfg=stream_cfg,
                mqtt_client=mqtt_client,
                loop=loop,
            )
            self.sessions[report_id] = worker
            self.camera_locks[camera_key] = report_id
            worker.start()
            logger.info("Created session %s with camera %s", report_id, camera_source)
            return worker

    def close_session(
        self,
        report_id: str,
        camera_source: Optional[Union[str, int]] = None,
    ) -> bool:
        """
        Close and cleanup a session.

        Args:
            report_id: Session identifier
            camera_source: Optional; close only if matches

        Returns:
            True if closed, False if not found or mismatch
        """
        with self.sessions_lock:
            worker = self.sessions.get(report_id)
            if worker is None:
                logger.warning("Session %s not found", report_id)
                return False
            if (
                camera_source is not None
                and str(worker.camera_source) != str(camera_source)
            ):
                logger.warning(
                    "Session %s camera mismatch: expected %s, got %s",
                    report_id,
                    camera_source,
                    worker.camera_source,
                )
                return False

            worker.stop()
            camera_key = str(worker.camera_source)
            if camera_key in self.camera_locks:
                del self.camera_locks[camera_key]
            del self.sessions[report_id]
            logger.info("Closed session %s", report_id)
            return True

    def get_session(self, report_id: str) -> Optional[SessionWorker]:
        """Get session by report_id."""
        with self.sessions_lock:
            return self.sessions.get(report_id)

    def is_camera_in_use(self, camera_source: Union[str, int]) -> bool:
        """Check if camera is currently in use by any session."""
        camera_key = str(camera_source)
        with self.sessions_lock:
            return camera_key in self.camera_locks

    def list_active_sessions(self) -> list:
        """Get list of all active sessions."""
        with self.sessions_lock:
            return [worker.get_info() for worker in self.sessions.values()]
