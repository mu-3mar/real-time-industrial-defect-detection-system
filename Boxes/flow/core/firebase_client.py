"""Firebase Realtime Database client for publishing detection events."""

import logging
from pathlib import Path

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


def publish_detection(report_id: str, detection_id: str, timestamp: str, defect: bool) -> bool:
    """
    Push a detection event to Firebase Realtime Database.

    Structure:
        {report_id}
           ├── defect
           │    └── {detection_id}
           │         └── timestamp: "2026-03-09T14:21:00Z"
           └── non_defect
                └── {detection_id}
                     └── timestamp: "2026-03-09T14:21:00Z"

    Args:
        report_id: Report identifier (from /api/reports/open).
        detection_id: Unique identifier for this detection.
        timestamp: ISO 8601 UTC timestamp string.
        defect: True if defect detected, False otherwise.

    Returns:
        True if write succeeded, False otherwise.
    """
    if not _initialized:
        logger.error("[Error] Firebase not initialized")
        return False

    group = "defect" if defect else "non_defect"
    payload = {
        "timestamp": timestamp,
    }

    try:
        # report_id / defect|non_defect / detection_id
        ref = db.reference(f"{report_id}/{group}/{detection_id}")
        ref.set(payload)
        logger.debug("Detection sent → report_id=%s group=%s det_id=%s", report_id, group, detection_id)
        return True
    except Exception as e:
        logger.error("[Error] Firebase write failed: %s", e)
        return False
