"""Session manager for handling multiple concurrent detection sessions."""

import asyncio
import logging
import threading
from typing import Dict, Optional, Union

from core.session_worker import SessionWorker

logger = logging.getLogger(__name__)


class SessionManager:
    """Singleton for managing active detection sessions. No static factory/line/camera data."""

    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.sessions: Dict[str, SessionWorker] = {}  # session_id -> worker
        self.camera_locks: Dict[str, str] = {}  # camera_source -> session_id
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
        session_id: str,
        camera_source: Union[str, int],
        factory_id: str,
        factory_name: str,
        production_line_id: str,
        production_line_name: str,
        camera_id: str,
        station_id: str,
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        loop: asyncio.AbstractEventLoop,
    ) -> SessionWorker:
        """
        Create and start a new headless detection session.
        All metadata is provided by the caller (backend/client); nothing is hardcoded.
        """
        camera_key = str(camera_source)
        with self.sessions_lock:
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
            if camera_key in self.camera_locks:
                locked_by = self.camera_locks[camera_key]
                raise ValueError(
                    f"Camera {camera_source} in use by session {locked_by}"
                )

            worker = SessionWorker(
                session_id=session_id,
                camera_source=camera_source,
                factory_id=factory_id,
                factory_name=factory_name,
                production_line_id=production_line_id,
                production_line_name=production_line_name,
                camera_id=camera_id,
                station_id=station_id,
                box_cfg=box_cfg,
                defect_cfg=defect_cfg,
                stream_cfg=stream_cfg,
                loop=loop,
            )
            self.sessions[session_id] = worker
            self.camera_locks[camera_key] = session_id
            worker.start()
            logger.info("Session started: session_id=%s camera=%s", session_id, camera_source)
            return worker

    def close_session(
        self,
        session_id: str,
        camera_source: Optional[Union[str, int]] = None,
    ) -> bool:
        """Close a session by session_id. Optionally validate camera_source."""
        with self.sessions_lock:
            worker = self.sessions.get(session_id)
            if worker is None:
                logger.warning("Session %s not found", session_id)
                return False
            if (
                camera_source is not None
                and str(worker.camera_source) != str(camera_source)
            ):
                logger.warning(
                    "Session %s camera mismatch: expected %s, got %s",
                    session_id, camera_source, worker.camera_source,
                )
                return False

            worker.stop()
            camera_key = str(worker.camera_source)
            if camera_key in self.camera_locks:
                del self.camera_locks[camera_key]
            del self.sessions[session_id]
            logger.info("Session stopped: session_id=%s", session_id)
            return True

    def get_session(self, session_id: str) -> Optional[SessionWorker]:
        """Get session by session_id."""
        with self.sessions_lock:
            return self.sessions.get(session_id)

    def is_camera_in_use(self, camera_source: Union[str, int]) -> bool:
        """Check if camera is currently in use by any session."""
        camera_key = str(camera_source)
        with self.sessions_lock:
            return camera_key in self.camera_locks

    def list_active_sessions(self) -> list:
        """Return list of session metadata for GET /api/sessions."""
        with self.sessions_lock:
            return [w.get_info() for w in self.sessions.values()]
