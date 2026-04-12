"""Session manager for handling multiple concurrent detection sessions."""

import asyncio
import logging
import threading
from typing import Dict, Optional, Tuple, Union

from core.session_worker import SessionWorker

logger = logging.getLogger(__name__)


class SessionManager:
    """Singleton for managing active detection sessions. No static factory/line/camera data."""

    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.sessions: Dict[str, SessionWorker] = {}  # report_id -> worker
        self.camera_locks: Dict[str, str] = {}  # camera_source -> report_id
        self.production_line_to_report: Dict[str, str] = {}  # production_line_id -> report_id
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
        camera_source: Union[str, int],
        production_line_id: str,
        target_speed: int,
        max_temp: int,
        max_amps: int,
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        app_cfg: dict,
        loop: asyncio.AbstractEventLoop,
    ) -> SessionWorker:
        """
        Create and start a new headless detection session.

        Raises ValueError if report_id, production_line_id, or camera_source
        is already used by an active session.
        """
        camera_key = str(camera_source)
        with self.sessions_lock:
            if report_id in self.sessions:
                raise ValueError(f"Session {report_id} already exists")

            if production_line_id in self.production_line_to_report:
                existing_report_id = self.production_line_to_report[production_line_id]
                if existing_report_id in self.sessions:
                    raise ValueError(
                        f"Production line {production_line_id} already has an active session {existing_report_id}"
                    )

            if camera_key in self.camera_locks:
                locked_by = self.camera_locks[camera_key]
                raise ValueError(
                    f"Camera {camera_source} in use by session {locked_by}"
                )

            worker = SessionWorker(
                report_id=report_id,
                camera_source=camera_source,
                production_line_id=production_line_id,
                target_speed=target_speed,
                max_temp=max_temp,
                max_amps=max_amps,
                box_cfg=box_cfg,
                defect_cfg=defect_cfg,
                stream_cfg=stream_cfg,
                app_cfg=app_cfg,
                loop=loop,
            )
            self.sessions[report_id] = worker
            self.camera_locks[camera_key] = report_id
            self.production_line_to_report[production_line_id] = report_id
            worker.start()
            logger.info("[Session] opened: %s", report_id)
            return worker

    def close_session(self, report_id: str) -> Tuple[bool, bool]:
        """
        Close a session by report_id.

        Returns:
            (closed, already_closed): closed is True if we closed the session;
            already_closed is True if the session was not found (already closed).
        """
        with self.sessions_lock:
            worker = self.sessions.get(report_id)
            if worker is None:
                logger.debug("Session %s not found (already closed)", report_id)
                return False, True

            worker.stop()
            camera_key = str(worker.camera_source)
            production_line_id = worker.production_line_id
            if camera_key in self.camera_locks:
                del self.camera_locks[camera_key]
            if production_line_id in self.production_line_to_report:
                del self.production_line_to_report[production_line_id]
            del self.sessions[report_id]
            logger.info("[Session] closed: %s", report_id)
            return True, False

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
        """Return list of active report summaries: report_id, viewers_count."""
        with self.sessions_lock:
            return [w.get_info() for w in self.sessions.values()]
