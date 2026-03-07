"""Firebase Realtime Database client for publishing detection events."""

import logging
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, db

logger = logging.getLogger(__name__)

_initialized: bool = False


def initialize(credentials_path: str, database_url: str) -> bool:
    """
    Initialize Firebase Admin SDK with Realtime Database and return success.
    Idempotent: if already initialized, returns True.

    Args:
        credentials_path: Path to service account JSON file.
        database_url: Firebase Realtime Database URL (e.g. from FIREBASE_DATABASE_URL).

    Returns:
        True if initialized successfully.
    """
    global _initialized
    if _initialized:
        return True

    path = Path(credentials_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Firebase credentials not found: {path}")

    if not database_url or not database_url.strip():
        raise ValueError("Firebase Realtime Database URL is required (e.g. FIREBASE_DATABASE_URL)")

    cred = credentials.Certificate(str(path))
    firebase_admin.initialize_app(cred, {"databaseURL": database_url.strip()})
    _initialized = True
    logger.debug("Firebase initialized; Realtime Database client ready")
    return True


def get_initialized() -> bool:
    """Return True if Firebase has been initialized."""
    return _initialized


def publish_detection(
    factory_id: str,
    production_line_id: str,
    session_id: str,
    timestamp: str,
    defect: bool,
    camera_id: str,
    station_id: str,
    factory_name: str,
    production_line_name: str,
    model_version: str,
    confidence: float,
) -> bool:
    """
    Write a detection event to Firebase Realtime Database. Each call pushes a new
    record under factories/{factory_id}/production_lines/{production_line_id}/sessions/{session_id}/insights/
    with an auto-generated key.

    Args:
        factory_id: Factory identifier.
        production_line_id: Production line identifier.
        session_id: Session identifier.
        timestamp: ISO 8601 UTC timestamp string.
        defect: True if defect detected, False otherwise.
        camera_id: Camera identifier.
        station_id: Station identifier.
        factory_name: Human-readable factory name.
        production_line_name: Human-readable production line name.
        model_version: Detection model version string.
        confidence: Detection confidence (0.0–1.0).

    Returns:
        True if write succeeded, False otherwise. Does not raise; logs errors.
    """
    if not _initialized:
        logger.error("Firebase not initialized; cannot publish detection")
        return False

    path = (
        f"factories/{factory_id}/production_lines/{production_line_id}"
        f"/sessions/{session_id}/insights"
    )
    result = {
        "timestamp": timestamp,
        "defect": defect,
        "camera_id": camera_id,
        "station_id": station_id,
        "factory_name": factory_name,
        "production_line_name": production_line_name,
        "model_version": model_version,
        "confidence": confidence,
    }

    try:
        ref = db.reference(path)
        ref.push(result)
        logger.info(
            "Detection sent → factory=%s line=%s session=%s defect=%s",
            factory_id, production_line_id, session_id, defect,
        )
        return True
    except Exception as e:
        logger.error(
            "Realtime Database write failed: factory=%s line=%s session=%s error=%s",
            factory_id, production_line_id, session_id, e,
            exc_info=True,
        )
        return False
