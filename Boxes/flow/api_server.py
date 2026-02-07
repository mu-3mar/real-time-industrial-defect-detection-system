"""QC-SCM Detection Service API with multi-session support."""

import logging
from pathlib import Path
from typing import Optional, Union

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.backend_client import BackendClient
from core.model_loader import ModelLoader
from core.session_manager import SessionManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="QC-SCM Detection Service",
    description="Multi-session quality control detection with defect analysis",
    version="1.0.0",
)

# Global state
session_manager = SessionManager.get_instance()
backend_client: Optional[BackendClient] = None
configs: dict = {}


# -----------------------------------------------------------------------------
# Schema models
class SessionIdentifiers(BaseModel):
    """Request body for session operations."""

    report_id: str
    camera_source: Union[str, int]


class SessionResponse(BaseModel):
    """Response for session mutation endpoints."""

    status: str
    report_id: str
    message: str


class SessionListResponse(BaseModel):
    """Response for list sessions endpoint."""

    sessions: list


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    active_sessions: int


# -----------------------------------------------------------------------------
# Configuration loading
def _load_configs(base: Path) -> None:
    """Load YAML configs and set defaults for missing sections."""
    config_path = base / "configs"

    with open(config_path / "box_detector.yaml") as f:
        configs["box"] = yaml.safe_load(f)
    with open(config_path / "defect_detector.yaml") as f:
        configs["defect"] = yaml.safe_load(f)
    with open(config_path / "stream.yaml") as f:
        configs["stream"] = yaml.safe_load(f)
    with open(config_path / "backend.yaml") as f:
        backend_cfg = yaml.safe_load(f)

    # Ensure stability config
    if "stability" not in configs["defect"]:
        configs["defect"]["stability"] = {
            "min_frames": 4,
            "max_missed": 6,
            "vote_window": 9,
            "vote_threshold": 5,
            "early_detection_frames": 3,
            "track_grace_frames": 3,
            "recent_track_max_age": 15,
            "recovery_iou_threshold": 0.4,
        }
    # Ensure tracking config
    if "tracking" not in configs["defect"]:
        configs["defect"]["tracking"] = {
            "iou_threshold": 0.35,
            "bbox_smooth_alpha": 0.6,
        }
    # Ensure rendering config
    if "rendering" not in configs["defect"]:
        configs["defect"]["rendering"] = {
            "visibility_threshold": 0.2,
        }

    global backend_client
    backend_client = BackendClient(
        base_url=backend_cfg["base_url"],
        result_endpoint=backend_cfg["result_endpoint"],
        timeout=backend_cfg.get("timeout", 5),
        max_retries=backend_cfg.get("max_retries", 3),
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize configs, models, and backend client."""
    base = Path(__file__).resolve().parent
    logger.info("Starting QC-SCM Detection Service...")
    try:
        _load_configs(base)
        model_loader = ModelLoader.get_instance()
        model_loader.load_models(
            configs["box"]["model_path"],
            configs["defect"]["model_path"],
        )
        logger.info("Service startup complete")
    except Exception as e:
        logger.error("Startup error: %s", e, exc_info=True)
        raise


# -----------------------------------------------------------------------------
# API Endpoints
@app.post("/api/sessions/open", response_model=SessionResponse)
async def open_session(body: SessionIdentifiers) -> SessionResponse:
    """Open a new headless detection session."""
    try:
        logger.info(
            "Opening session: report_id=%s camera_source=%s",
            body.report_id,
            body.camera_source,
        )
        session_manager.create_session(
            report_id=body.report_id,
            camera_source=body.camera_source,
            box_cfg=configs["box"],
            defect_cfg=configs["defect"],
            stream_cfg=configs["stream"],
            backend_client=backend_client,
        )
        return SessionResponse(
            status="success",
            report_id=body.report_id,
            message=f"Session started with camera {body.camera_source}",
        )
    except ValueError as e:
        logger.warning("Open session failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Open session error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/sessions/close", response_model=SessionResponse)
async def close_session(body: SessionIdentifiers) -> SessionResponse:
    """Close an active detection session."""
    try:
        logger.info(
            "Closing session: report_id=%s camera_source=%s",
            body.report_id,
            body.camera_source,
        )
        ok = session_manager.close_session(
            report_id=body.report_id,
            camera_source=body.camera_source,
        )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found or camera mismatch: report_id={body.report_id}",
            )
        return SessionResponse(
            status="success",
            report_id=body.report_id,
            message="Session closed successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Close session error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """List all active detection sessions."""
    try:
        sessions = session_manager.list_active_sessions()
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        logger.exception("List sessions error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/sessions/view", response_model=SessionResponse)
async def view_running_session(body: SessionIdentifiers) -> SessionResponse:
    """
    Open a viewer window for an already-running session.
    Shows the live feed with annotations (boxes, defects, stats) if the session is running.
    """
    try:
        logger.info("View session: report_id=%s camera_source=%s", body.report_id, body.camera_source)
        attached = session_manager.attach_viewer(
            report_id=body.report_id,
            camera_source=body.camera_source,
        )
        if not attached:
            raise HTTPException(
                status_code=404,
                detail=f"No running session for report_id={body.report_id} with camera_source={body.camera_source}",
            )
        return SessionResponse(
            status="success",
            report_id=body.report_id,
            message=f"Viewer opened for session (camera {body.camera_source})",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("View session error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/sessions/view/close", response_model=SessionResponse)
async def close_view(body: SessionIdentifiers) -> SessionResponse:
    """
    Close the viewer window for a running session.
    The session continues running in headless mode.
    """
    try:
        logger.info("Closing viewer: report_id=%s camera_source=%s", body.report_id, body.camera_source)
        closed = session_manager.detach_viewer(
            report_id=body.report_id,
            camera_source=body.camera_source,
        )
        if not closed:
            raise HTTPException(
                status_code=404,
                detail=f"No running session or viewer for report_id={body.report_id} with camera_source={body.camera_source}",
            )
        return SessionResponse(
            status="success",
            report_id=body.report_id,
            message="Viewer closed successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Close view error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health and active session count."""
    try:
        active = len(session_manager.list_active_sessions())
        return HealthResponse(status="healthy", active_sessions=active)
    except Exception as e:
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail="Internal server error")
