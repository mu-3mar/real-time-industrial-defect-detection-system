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
        headless: bool = True
    ) -> SessionWorker:
        """
        Create and start a new session.
        
        Args:
            report_id: Unique session identifier
            camera_source: Camera device path or index
            box_cfg: Box detector configuration
            defect_cfg: Defect detector configuration
            stream_cfg: Stream configuration
            backend_client: Backend HTTP client
        
        Returns:
            SessionWorker instance
        
        Raises:
            ValueError: If session already exists or camera is in use
        """
        camera_key = str(camera_source)
        
        with self.sessions_lock:
            # Check if session already exists
            if report_id in self.sessions:
                raise ValueError(f"Session {report_id} already exists")
            
            # Check if camera is already in use
            if camera_key in self.camera_locks:
                locked_by = self.camera_locks[camera_key]
                raise ValueError(f"Camera {camera_source} is already in use by session {locked_by}")
            
            # Create worker
            worker = SessionWorker(
                report_id=report_id,
                camera_source=camera_source,
                box_cfg=box_cfg,
                defect_cfg=defect_cfg,
                stream_cfg=stream_cfg,
                backend_client=backend_client,
                headless=headless
            )
            
            # Register session and lock camera
            self.sessions[report_id] = worker
            self.camera_locks[camera_key] = report_id
            
            # Start worker
            worker.start()
            
            logger.info(f"Created session {report_id} with camera {camera_source}")
            return worker
    
    def close_session(self, report_id: str) -> bool:
        """
        Close and cleanup a session.
        
        Args:
            report_id: Session identifier
        
        Returns:
            True if session was closed, False if not found
        """
        with self.sessions_lock:
            worker = self.sessions.get(report_id)
            
            if worker is None:
                logger.warning(f"Session {report_id} not found")
                return False
            
            # Stop worker
            worker.stop()
            
            # Release camera lock
            camera_key = str(worker.camera_source)
            if camera_key in self.camera_locks:
                del self.camera_locks[camera_key]
            
            # Remove session
            del self.sessions[report_id]
            
            logger.info(f"Closed session {report_id}")
            return True
    
    def get_session(self, report_id: str) -> Optional[SessionWorker]:
        """Get session by report_id."""
        with self.sessions_lock:
            return self.sessions.get(report_id)
    
    def is_camera_in_use(self, camera_source: Union[str, int]) -> bool:
        """Check if camera is currently in use."""
        camera_key = str(camera_source)
        with self.sessions_lock:
            return camera_key in self.camera_locks
    
    def list_active_sessions(self) -> list:
        """Get list of all active sessions."""
        with self.sessions_lock:
            return [worker.get_info() for worker in self.sessions.values()]
