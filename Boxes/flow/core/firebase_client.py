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
    logger.info("Firebase initialized; Firestore client ready")
    return _db


def get_client() -> Optional[firestore.Client]:
    """Return the Firestore client if initialized, else None."""
    return _db


def publish_detection(
    factory_id: str,
    line_id: str,
    session_id: str,
    timestamp: str,
    defect: bool,
) -> bool:
    """
    Write a detection event to Firestore. Each call creates a new document
    under factories/{factory_id}/production_lines/{line_id}/sessions/{session_id}/insights/.
    Parent documents are auto-created by Firestore if they do not exist.

    Args:
        factory_id: Factory identifier.
        line_id: Production line identifier.
        session_id: Session identifier (e.g. report_id).
        timestamp: ISO timestamp string for the detection.
        defect: True if defect detected, False otherwise.

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
            .document(line_id)
            .collection("sessions")
            .document(session_id)
            .collection("insights")
        )
        ref.add({
            "timestamp": timestamp,
            "defect": defect,
        })
        logger.info(
            "Detection sent → factory=%s line=%s session=%s defect=%s",
            factory_id, line_id, session_id, defect,
        )
        return True
    except Exception as e:
        logger.error(
            "Firestore write failed: factory=%s line=%s session=%s error=%s",
            factory_id, line_id, session_id, e,
            exc_info=True,
        )
        return False
