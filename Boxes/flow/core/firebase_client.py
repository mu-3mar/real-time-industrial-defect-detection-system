"""Firebase Firestore client for publishing detection events."""

import logging
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

_db: Optional[firestore.Client] = None


def initialize(credentials_path: str) -> firestore.Client:
    """
    Initialize Firebase Admin SDK and return Firestore client.
    Idempotent: if already initialized, returns existing client.

    Args:
        credentials_path: Path to service account JSON file.

    Returns:
        Firestore client instance.
    """
    global _db
    if _db is not None:
        return _db

    path = Path(credentials_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Firebase credentials not found: {path}")

    cred = credentials.Certificate(str(path))
    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    logger.debug("Firebase initialized; Firestore client ready")
    return _db


def get_client() -> Optional[firestore.Client]:
    """Return the Firestore client if initialized, else None."""
    return _db


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
    Write a detection event to Firestore. Each call creates a new document
    under factories/{factory_id}/production_lines/{production_line_id}/sessions/{session_id}/insights/{event_id}.
    Parent documents are auto-created by Firestore if they do not exist.

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
    client = get_client()
    if client is None:
        logger.error("Firestore not initialized; cannot publish detection")
        return False

    try:
        db = client
        ref = (
            db.collection("factories")
            .document(factory_id)
            .collection("production_lines")
            .document(production_line_id)
            .collection("sessions")
            .document(session_id)
            .collection("insights")
        )
        ref.add({
            "timestamp": timestamp,
            "defect": defect,
            "camera_id": camera_id,
            "station_id": station_id,
            "factory_name": factory_name,
            "production_line_name": production_line_name,
            "model_version": model_version,
            "confidence": confidence,
        })
        logger.info(
            "Detection sent → factory=%s line=%s session=%s defect=%s",
            factory_id, production_line_id, session_id, defect,
        )
        return True
    except Exception as e:
        logger.error(
            "Firestore write failed: factory=%s line=%s session=%s error=%s",
            factory_id, production_line_id, session_id, e,
            exc_info=True,
        )
        return False
