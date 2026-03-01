"""QC-SCM Detection Service API with multi-session support."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)

from core.model_loader import ModelLoader
from core.mqtt_client import MqttClient
from core.session_manager import SessionManager
from core.webrtc_track import VideoTransformTrack

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_cors_config() -> tuple:
    """
    Load CORS config. Returns (allow_origins, allow_credentials).
    - If config/app.yaml has cors_origins (non-empty list): use those with credentials=True.
    - Else: use ["*"] with credentials=False (works for localhost, LAN, remote out-of-the-box).
    """
    base = Path(__file__).resolve().parent
    app_cfg_path = base / "config" / "app.yaml"
    if app_cfg_path.exists():
        try:
            with open(app_cfg_path) as f:
                app_cfg = yaml.safe_load(f) or {}
            origins = app_cfg.get("cors_origins")
            if origins and isinstance(origins, list) and len(origins) > 0:
                cleaned = [str(o).rstrip("/") for o in origins if o]
                if cleaned:
                    return cleaned, True
        except Exception:  # noqa: S110
            pass
    return ["*"], False


# FastAPI app
app = FastAPI(
    title="QC-SCM Detection Service",
    description="Multi-session quality control detection with defect analysis",
    version="1.0.0",
)

_cors_origins, _cors_credentials = _get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global state
session_manager = SessionManager.get_instance()
mqtt_client: Optional[MqttClient] = None
configs: Dict[str, Any] = {}

# WebRTC Peer Connections
pcs: Set[RTCPeerConnection] = set()


# -----------------------------------------------------------------------------
# Schema models
class SessionIdentifiers(BaseModel):
    """Request body for session operations."""

    report_id: str
    production_line: str
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


class OfferRequest(BaseModel):
    """WebRTC Offer request."""
    sdp: str
    type: str
    report_id: str


class OfferResponse(BaseModel):
    """WebRTC Answer response."""
    sdp: str
    type: str


# -----------------------------------------------------------------------------
# Configuration loading
def _load_configs(base: Path) -> None:
    """Load all YAML configs from the single config/ directory."""
    cfg = base / "config"

    # Environment-dependent config (optional, with defaults)
    # Note: api.yaml is loaded by main.py only; webrtc used here for ICE
    for key, filename in [("app", "app.yaml"), ("webrtc", "webrtc.yaml")]:
        path = cfg / filename
        if path.exists():
            with open(path) as f:
                configs[key] = yaml.safe_load(f) or {}
        else:
            configs[key] = {}
    # Defaults for optional central config
    if "webrtc" not in configs or not configs["webrtc"]:
        configs["webrtc"] = {
            "stun": {"urls": "stun:20.51.117.96:3478"},
            "turn": {"urls": "turn:20.51.117.96:3478", "username": "turnuser", "credential": "Sup3r$tr0ngP@ssw0rd"},
        }

    # Service configs (now co-located in config/ alongside environment configs)
    with open(cfg / "box_detector.yaml") as f:
        configs["box"] = yaml.safe_load(f)
    with open(cfg / "defect_detector.yaml") as f:
        configs["defect"] = yaml.safe_load(f)
    with open(cfg / "stream.yaml") as f:
        configs["stream"] = yaml.safe_load(f)
    with open(cfg / "mqtt.yaml") as f:
        configs["mqtt"] = yaml.safe_load(f)

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

    # Initialize MQTT client
    global mqtt_client
    mqtt_cfg = configs["mqtt"]
    mqtt_client = MqttClient.initialize(
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        username=mqtt_cfg["broker"]["username"],
        password=mqtt_cfg["broker"]["password"],
        client_id_prefix=mqtt_cfg["client"]["client_id_prefix"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        clean_session=mqtt_cfg["client"]["clean_session"],
        topic_pattern=mqtt_cfg["topics"]["insights_pattern"],
    )
    mqtt_client.connect()


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize configs, models, and MQTT client."""
    base = Path(__file__).resolve().parent
    logger.info("Starting QC-SCM Detection Service...")
    try:
        _load_configs(base)
        model_loader = ModelLoader.get_instance()
        model_loader.load_models(
            configs["box"]["model_path"],
            configs["defect"]["model_path"],
        )
        device = str(configs.get("box", {}).get("device", "0"))
        model_loader.warmup(device=device)
        logger.info("Service startup complete")
    except Exception as e:
        logger.error("Startup error: %s", e, exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up WebRTC connections and MQTT client."""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
    # Disconnect MQTT client
    if mqtt_client:
        mqtt_client.disconnect()


# -----------------------------------------------------------------------------
# API Endpoints
@app.post("/api/sessions/open", response_model=SessionResponse)
async def open_session(body: SessionIdentifiers) -> SessionResponse:
    """Open a new headless detection session."""
    try:
        logger.info(
            "Opening session: report_id=%s production_line=%s camera_source=%s",
            body.report_id,
            body.production_line,
            body.camera_source,
        )
        # Capture the current event loop to pass to the worker
        loop = asyncio.get_running_loop()

        session_manager.create_session(
            report_id=body.report_id,
            production_line=body.production_line,
            camera_source=body.camera_source,
            box_cfg=configs["box"],
            defect_cfg=configs["defect"],
            stream_cfg=configs["stream"],
            mqtt_client=mqtt_client,
            loop=loop,
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


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health and active session count."""
    try:
        active = len(session_manager.list_active_sessions())
        return HealthResponse(status="healthy", active_sessions=active)
    except Exception as e:
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Optional metrics for observability. Does not affect API contracts."""
    try:
        sessions = session_manager.list_active_sessions()
        active_viewers = sum(s.get("active_viewers", 0) for s in sessions)
        return {
            "active_sessions": len(sessions),
            "active_viewers": active_viewers,
            "webrtc_connections": len(pcs),
        }
    except Exception as e:
        logger.exception("Metrics error")
        raise HTTPException(status_code=500, detail="Internal server error")


def _client_webrtc_config() -> Dict[str, Any]:
    """Build client-safe WebRTC config. Returns only STUN; TURN credentials never exposed."""
    w = configs.get("webrtc") or {}
    stun = w.get("stun") or {}
    ice_servers = []
    if stun.get("urls"):
        ice_servers.append({"urls": stun["urls"] if isinstance(stun["urls"], str) else stun["urls"]})
    return {
        "webrtc": {
            "iceServers": ice_servers,
        }
    }


@app.get("/api/config")
async def get_client_config() -> Dict[str, Any]:
    """Client configuration (WebRTC STUN servers). Consumed by external frontends."""
    return _client_webrtc_config()


@app.post("/webrtc/offer", response_model=OfferResponse)
async def webrtc_offer(params: OfferRequest) -> OfferResponse:
    """
    Handle WebRTC SDP offer and establish connection.
    Attach the detection stream as a video track.
    """
    worker = session_manager.get_session(params.report_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Session not found")

    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    # Single ICE config from config/webrtc.yaml
    w = configs.get("webrtc") or {}
    stun = w.get("stun") or {}
    turn = w.get("turn") or {}
    ice_servers = []
    if stun.get("urls"):
        ice_servers.append(RTCIceServer(urls=stun["urls"]))
    if turn.get("urls"):
        ice_servers.append(RTCIceServer(
            urls=turn["urls"],
            username=turn.get("username"),
            credential=turn.get("credential"),
        ))
    config = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    def _log_ice_candidates() -> None:
        """Log gathered ICE candidates with type (host, srflx, relay) for debugging."""
        try:
            ice_transport = None
            if getattr(pc, "sctp", None) and getattr(pc.sctp, "transport", None):
                dtls = pc.sctp.transport
                ice_transport = getattr(dtls, "transport", None)
            if ice_transport is None:
                for transceiver in pc.getTransceivers():
                    sender = getattr(transceiver, "sender", None)
                    dtls = getattr(sender, "transport", None) if sender else None
                    if dtls and getattr(dtls, "transport", None):
                        ice_transport = dtls.transport
                        break
            if ice_transport and hasattr(ice_transport, "iceGatherer"):
                gatherer = ice_transport.iceGatherer
                if hasattr(gatherer, "getLocalCandidates"):
                    for c in gatherer.getLocalCandidates():
                        ctype = getattr(c, "type", "unknown")
                        logger.info("ICE candidate gathered: type=%s", ctype)
        except Exception as e:
            logger.debug("ICE candidate logging skipped: %s", e)

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        if pc.iceGatheringState == "complete":
            _log_ice_candidates()

    async def _log_selected_transport() -> None:
        """
        Log a single clean line indicating how WebRTC connected:
        - direct (host/srflx)
        - turn-relay (relay)

        Primary method uses aiortc's selected ICE candidate pair from the
        underlying iceTransport. Falls back to 'unknown' on error.
        """
        transport = "unknown"
        try:
            ice_transport = None

            # Prefer SCTP transport (if datachannel present)
            if getattr(pc, "sctp", None) and getattr(pc.sctp, "transport", None):
                dtls = pc.sctp.transport
                ice_transport = getattr(dtls, "transport", None)

            # Fallback: inspect sender transports if SCTP is not available
            if ice_transport is None:
                for transceiver in pc.getTransceivers():
                    sender = getattr(transceiver, "sender", None)
                    dtls = getattr(sender, "transport", None) if sender else None
                    if dtls and getattr(dtls, "transport", None):
                        ice_transport = dtls.transport
                        break

            if ice_transport and hasattr(ice_transport, "getSelectedCandidatePair"):
                pair = ice_transport.getSelectedCandidatePair()
                if pair:
                    local_type = getattr(pair.local, "type", None)
                    remote_type = getattr(pair.remote, "type", None)
                    if local_type == "relay" or remote_type == "relay":
                        transport = "turn-relay"
                    else:
                        transport = "direct"

        except Exception as e:
            logger.info("WebRTC connected; transport detection failed: %s", e)

        logger.info(
            "WebRTC Connected via: %s (connectionState=%s, iceConnectionState=%s)",
            transport,
            pc.connectionState,
            pc.iceConnectionState,
        )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        if state == "connected":
            await _log_selected_transport()
        elif state in {"failed", "disconnected", "closed"}:
            logger.info(
                "WebRTC state=%s, ICE=%s", state, pc.iceConnectionState
            )
        if state == "failed":
            await pc.close()
            pcs.discard(pc)
        elif state == "closed":
            pcs.discard(pc)

    # Create video track from session
    video_track = VideoTransformTrack()
    worker.add_track(video_track)

    @pc.on("track")
    def on_track(track):
        # We don't handle incoming tracks (audio/video from client)
        pass
    
    # We add track to PC to send video to client
    pc.addTrack(video_track)

    # Handle cleanup when PC closes
    # We can't easily hook into pc.close() here, but connectionstatechange helps.
    # More robustly, we should remove the track from worker when PC is closed.
    # But for now, we rely on connectionstatechange.
    orig_close = pc.close
    async def new_close():
        worker.remove_track(video_track)
        await orig_close()
    pc.close = new_close

    try:
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return OfferResponse(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type
        )
    except Exception as e:
        logger.error("WebRTC offer failed: %s", e)
        # Cleanup
        worker.remove_track(video_track)
        await pc.close()
        pcs.discard(pc)
        raise HTTPException(status_code=500, detail=str(e))
