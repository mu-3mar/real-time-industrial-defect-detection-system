import threading
import logging
from typing import Dict, Optional, Union
from core.session_worker import SessionWorker
from core.backend_client import BackendClient

logger = logging.getLogger(__name__)

class SessionManager:
    """Singleton class for managing active detection sessions."""
    
    _instance: Optional['SessionManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.sessions: Dict[str, SessionWorker] = {}
        self.camera_locks: Dict[str, str] = {}  # camera_source -> report_id
        self.sessions_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'SessionManager':
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
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        backend_client: BackendClient,
    ) -> SessionWorker:
        """
        Create and start a new headless detection session.

        Args:
            report_id: Unique session identifier
            camera_source: Camera device path or index
            box_cfg: Box detector configuration
            defect_cfg: Defect detector configuration
            stream_cfg: Stream configuration (source will be overridden by camera_source)
            backend_client: Backend HTTP client

        Returns:
            SessionWorker instance

        Raises:
            ValueError: If session already exists or camera is in use
        """
        camera_key = str(camera_source)
        with self.sessions_lock:
            if report_id in self.sessions:
                raise ValueError(f"Session {report_id} already exists")
            if camera_key in self.camera_locks:
                locked_by = self.camera_locks[camera_key]
                raise ValueError(f"Camera {camera_source} is already in use by session {locked_by}")

            worker = SessionWorker(
                report_id=report_id,
                camera_source=camera_source,
                box_cfg=box_cfg,
                defect_cfg=defect_cfg,
                stream_cfg=stream_cfg,
                backend_client=backend_client,
                headless=True,
            )
            self.sessions[report_id] = worker
            self.camera_locks[camera_key] = report_id
            worker.start()
            logger.info("Created session %s with camera %s", report_id, camera_source)
            return worker

    def close_session(
        self, report_id: str, camera_source: Optional[Union[str, int]] = None
    ) -> bool:
        """
        Close and cleanup a session.

        Args:
            report_id: Session identifier
            camera_source: Optional; if provided, session is closed only when it matches

        Returns:
            True if session was closed, False if not found or camera mismatch
        """
        with self.sessions_lock:
            worker = self.sessions.get(report_id)
            if worker is None:
                logger.warning("Session %s not found", report_id)
                return False
            if camera_source is not None and str(worker.camera_source) != str(camera_source):
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

    def get_session_by_report_and_camera(
        self, report_id: str, camera_source: Union[str, int]
    ) -> Optional[SessionWorker]:
        """Get session by report_id and camera_source; returns None if not found or mismatch."""
        worker = self.get_session(report_id)
        if worker is None:
            return None
        if str(worker.camera_source) != str(camera_source):
            return None
        return worker

    def attach_viewer(self, report_id: str, camera_source: Union[str, int]) -> bool:
        """
        Attach a viewer window to a running session (report_id, camera_source).
        Returns True if viewer was attached, False if session not found or not headless.
        """
        worker = self.get_session_by_report_and_camera(report_id, camera_source)
        if worker is None:
            return False
        return worker.start_viewer()
    
    def is_camera_in_use(self, camera_source: Union[str, int]) -> bool:
        """Check if camera is currently in use."""
        camera_key = str(camera_source)
        with self.sessions_lock:
            return camera_key in self.camera_locks
    
    def list_active_sessions(self) -> list:
        """Get list of all active sessions."""
        with self.sessions_lock:
            return [worker.get_info() for worker in self.sessions.values()]
