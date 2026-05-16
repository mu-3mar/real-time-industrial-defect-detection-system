"""Firebase Realtime Database client for publishing detection events."""

import logging
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, db

logger = logging.getLogger(__name__)

_initialized: bool = False


def _teardown_existing_app() -> None:
    """Delete any previously initialized Firebase app so we can start fresh."""
    try:
        app = firebase_admin.get_app()
        firebase_admin.delete_app(app)
        logger.debug("Deleted stale Firebase app before re-init")
    except ValueError:
        pass  # no app exists yet


def initialize(credentials_path: str, database_url: str) -> bool:
    """
    Initialize Firebase Admin SDK with Realtime Database.

    Always tears down any existing Firebase app first so credentials
    are never stale across restarts or hot-reloads.

    Args:
        credentials_path: Path to service account JSON file.
        database_url: Firebase Realtime Database URL.

    Returns:
        True if initialized successfully.
    """
    global _initialized

    path = Path(credentials_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Firebase credentials not found: {path}")

    if not database_url or not database_url.strip():
        raise ValueError(
            "Firebase Realtime Database URL is required "
            "(set database_url in firebase.yaml or FIREBASE_DATABASE_URL env)"
        )

    _teardown_existing_app()

    cred = credentials.Certificate(str(path))
    firebase_admin.initialize_app(cred, {"databaseURL": database_url.strip()})
    _initialized = True
    logger.info("[Service] Firebase initialized")
    return True


def get_initialized() -> bool:
    """Return True if Firebase has been initialized."""
    return _initialized


def publish_detection(report_id: str, timestamp: str, defect: bool) -> Optional[str]:
    """
    Push a detection event to Firebase Realtime Database.

    Uses Firebase ``push()`` so the child key is auto-generated (same as POST to
    ``/{report_id}/detections.json`` with a JSON body).

    Structure:
        {report_id}
           └── detections
                └── {-FirebasePushId-}
                     ├── timestamp: "2026-03-09T14:21:00Z"
                     └── defect: true|false

    Args:
        report_id: Report identifier (from /api/reports/open), e.g. ``chainly``.
        timestamp: ISO 8601 UTC timestamp string.
        defect: True if defect detected, False otherwise.

    Returns:
        The Firebase push key on success, None on failure.
    """
    if not _initialized:
        logger.error("[Error] Firebase not initialized")
        return None

    payload = {
        "timestamp": timestamp,
        "defect": defect,
    }

    try:
        new_ref = db.reference(f"{report_id}/detections").push(payload)
        push_key = new_ref.key
        logger.debug(
            "Detection pushed → report_id=%s key=%s defect=%s",
            report_id,
            push_key,
            defect,
        )
        return push_key
    except Exception as e:
        logger.error("[Error] Firebase write failed: %s", e)
        return None


def publish_session_info(report_id: str, session_info: dict) -> bool:
    """
    Safely write session_info under the report_id node without overwriting
    sibling nodes (e.g. detections).

    Args:
        report_id: Report identifier.
        session_info: Dictionary containing telemetry, control, and config metadata.

    Returns:
        True if write succeeded, False otherwise.
    """
    if not _initialized:
        logger.error("[Error] Firebase not initialized")
        return False

    logger.info("Writing session_info to Firebase for report_id: %s...", report_id)
    try:
        # We use set on the specific child node to avoid any root update issues
        # and ensure it doesn't overwrite sibling nodes like detections
        ref = db.reference(f"{report_id}/session_info")
        ref.set(session_info)
        logger.info("Success: session_info written to Firebase for report_id: %s", report_id)
        return True
    except Exception as e:
        logger.error("[Error] Failed to write session_info to Firebase for report_id: %s. Reason: %s", report_id, e)
        return False
