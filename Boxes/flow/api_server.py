"""
QC-SCM Detection Service API
Multi-session quality control detection with four core endpoints.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.backend_client import BackendClient
from core.model_loader import ModelLoader
from core.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QC-SCM Detection Service",
    description="Multi-session quality control detection service",
    version="1.0.0",
)

session_manager = SessionManager.get_instance()
backend_client: Optional[BackendClient] = None
configs: dict = {}


# -----------------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------------


class SessionIdentifiers(BaseModel):
    """Report and camera identifying a session."""

    report_id: str
    camera_source: Union[str, int]


class SessionResponse(BaseModel):
    """Standard response for session mutations."""

    status: str
    report_id: str
    message: str


class SessionListResponse(BaseModel):
    """Response for listing active sessions."""

    sessions: list


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    active_sessions: int


# -----------------------------------------------------------------------------
# Config and startup
# -----------------------------------------------------------------------------


def _load_configs(base: Path) -> None:
    """Load YAML configs and inject defaults where needed."""
    with open(base / "configs/box_detector.yaml") as f:
        configs["box"] = yaml.safe_load(f)
    with open(base / "configs/defect_detector.yaml") as f:
        configs["defect"] = yaml.safe_load(f)
    with open(base / "configs/stream.yaml") as f:
        configs["stream"] = yaml.safe_load(f)
    with open(base / "configs/backend.yaml") as f:
        backend_cfg = yaml.safe_load(f)

        if "stability" not in configs["defect"]:
            configs["defect"]["stability"] = {
                "min_frames": 4,
                "max_missed": 6,
                "vote_window": 9,
                "vote_threshold": 5,
            }
        if "tracking" not in configs["defect"]:
            configs["defect"]["tracking"] = {
                "iou_threshold": 0.35,
                "bbox_smooth_alpha": 0.6,
            }
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
    """Load configs, backend client, and models on startup."""
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
        logger.error("Error during startup: %s", e, exc_info=True)
        raise


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.post("/api/sessions/open", response_model=SessionResponse)
async def open_session(body: SessionIdentifiers) -> SessionResponse:
    """
    Open a new detection session (headless).
    Uses the given report_id and camera_source.
    """
    try:
        logger.info("Opening session: report_id=%s camera_source=%s", body.report_id, body.camera_source)
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
    """
    Close an active detection session.
    Report id and camera source must match the running session.
    """
    try:
        logger.info("Closing session: report_id=%s camera_source=%s", body.report_id, body.camera_source)
        ok = session_manager.close_session(
            report_id=body.report_id,
            camera_source=body.camera_source,
        )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found or camera mismatch: report_id={body.report_id}, camera_source={body.camera_source}",
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
    """List all active sessions."""
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


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health and active session count."""
    try:
        active = len(session_manager.list_active_sessions())
        return HealthResponse(status="healthy", active_sessions=active)
    except Exception as e:
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail="Internal server error")
