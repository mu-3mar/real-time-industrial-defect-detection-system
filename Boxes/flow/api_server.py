"""
QC-SCM Detection Service API
Multi-session quality control detection service with REST API endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import logging
import yaml
from pathlib import Path
from core.session_manager import SessionManager
from core.backend_client import BackendClient
from core.model_loader import ModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QC-SCM Detection Service",
    description="Multi-session quality control detection service",
    version="1.0.0"
)

# Global instances
session_manager = SessionManager.get_instance()
backend_client = None
configs = {}

# Request/Response Models
class OpenSessionRequest(BaseModel):
    """Request model for opening a new detection session."""
    report_id: str
    camera_source: Union[str, int]

class CloseSessionRequest(BaseModel):
    """Request model for closing an active session."""
    report_id: str

class SessionResponse(BaseModel):
    """Response model for session operations."""
    status: str
    report_id: str
    message: str

class SessionListResponse(BaseModel):
    """Response model for listing active sessions."""
    sessions: list

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    active_sessions: int

@app.on_event("startup")
async def startup_event():
    """Load models and configurations on startup."""
    global backend_client, configs
    
    logger.info("Starting QC-SCM Detection Service...")
    
    BASE = Path(__file__).resolve().parent
    
    try:
        # Load configurations
        with open(BASE / "configs/box_detector.yaml") as f:
            configs["box"] = yaml.safe_load(f)
        
        with open(BASE / "configs/defect_detector.yaml") as f:
            configs["defect"] = yaml.safe_load(f)
        
        with open(BASE / "configs/stream.yaml") as f:
            configs["stream"] = yaml.safe_load(f)
        
        with open(BASE / "configs/backend.yaml") as f:
            backend_cfg = yaml.safe_load(f)
        
        # Inject stability config if missing
        if "stability" not in configs["defect"]:
            configs["defect"]["stability"] = {
                "min_frames": 3,
                "max_missed": 5,
                "vote_window": 7,
                "vote_threshold": 4
            }
        
        # Initialize backend client
        backend_client = BackendClient(
            base_url=backend_cfg["base_url"],
            result_endpoint=backend_cfg["result_endpoint"],
            timeout=backend_cfg.get("timeout", 5),
            max_retries=backend_cfg.get("max_retries", 3)
        )
        
        # Load models globally
        model_loader = ModelLoader.get_instance()
        model_loader.load_models(
            configs["box"]["model_path"],
            configs["defect"]["model_path"]
        )
        
        logger.info("Service startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

@app.post("/api/sessions/open", response_model=SessionResponse)
async def open_session(request: OpenSessionRequest):
    """
    Open a new detection session (headless mode - no GUI).
    
    Args:
        request: Contains report_id and camera_source
    
    Returns:
        Session creation status
    """
    try:
        logger.info(f"Opening headless session: {request.report_id} with camera {request.camera_source}")
        
        session_manager.create_session(
            report_id=request.report_id,
            camera_source=request.camera_source,
            box_cfg=configs["box"],
            defect_cfg=configs["defect"],
            stream_cfg=configs["stream"],
            backend_client=backend_client,
            headless=True
        )
        
        return SessionResponse(
            status="success",
            report_id=request.report_id,
            message=f"Headless session started with camera {request.camera_source}"
        )
        
    except ValueError as e:
        logger.error(f"Failed to open session {request.report_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error opening session {request.report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/sessions/open_gui", response_model=SessionResponse)
async def open_gui_session(request: OpenSessionRequest):
    """
    Open a new detection session with GUI window.
    
    Args:
        request: Contains report_id and camera_source
    
    Returns:
        Session creation status
    """
    try:
        logger.info(f"Opening GUI session: {request.report_id} with camera {request.camera_source}")
        
        session_manager.create_session(
            report_id=request.report_id,
            camera_source=request.camera_source,
            box_cfg=configs["box"],
            defect_cfg=configs["defect"],
            stream_cfg=configs["stream"],
            backend_client=backend_client,
            headless=False
        )
        
        return SessionResponse(
            status="success",
            report_id=request.report_id,
            message=f"GUI session started with camera {request.camera_source}"
        )
        
    except ValueError as e:
        logger.error(f"Failed to open GUI session {request.report_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error opening GUI session {request.report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/sessions/close", response_model=SessionResponse)
async def close_session(request: CloseSessionRequest):
    """
    Close an active detection session.
    
    Args:
        request: Contains report_id
    
    Returns:
        Session closure status
    """
    try:
        logger.info(f"Closing session: {request.report_id}")
        
        success = session_manager.close_session(request.report_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Session {request.report_id} not found")
        
        return SessionResponse(
            status="success",
            report_id=request.report_id,
            message="Session closed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing session {request.report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    """
    List all active sessions.
    
    Returns:
        List of active session information
    """
    try:
        sessions = session_manager.list_active_sessions()
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    try:
        active_count = len(session_manager.list_active_sessions())
        return HealthResponse(
            status="healthy",
            active_sessions=active_count
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
