"""Temporary, session-scoped storage for local detection results."""

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).resolve().parent.parent / ".session_data"
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _session_path(report_id: str) -> Path:
    digest = hashlib.sha256(report_id.encode("utf-8")).hexdigest()[:16]
    return _STORE_DIR / f"session_{digest}.json"


def _lock_for(report_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(report_id, threading.Lock())


def _write(report_id: str, payload: dict[str, Any]) -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    target = _session_path(report_id)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def create_session(report_id: str, session_info: dict[str, Any]) -> None:
    """Create a fresh local result file for a detection session."""
    payload = {
        report_id: {
            "session_info": session_info,
            "detections": {},
        }
    }
    with _lock_for(report_id):
        _write(report_id, payload)


def publish_detection(report_id: str, timestamp: str, defect: bool) -> Optional[str]:
    """Append a detection using a local UUID key, matching Firebase's shape."""
    with _lock_for(report_id):
        target = _session_path(report_id)
        if not target.exists():
            logger.warning("Local session file does not exist: %s", report_id)
            return None
        try:
            with target.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            node = payload[report_id]
            key = uuid.uuid4().hex
            node.setdefault("detections", {})[key] = {
                "timestamp": timestamp,
                "defect": defect,
            }
            _write(report_id, payload)
            return key
        except (OSError, KeyError, TypeError, ValueError):
            logger.exception("Could not write local detection: %s", report_id)
            return None


def delete_session(report_id: str) -> None:
    """Delete all temporary data belonging to a closed session."""
    with _lock_for(report_id):
        try:
            _session_path(report_id).unlink()
        except FileNotFoundError:
            return
        except OSError:
            logger.exception("Could not delete local session file: %s", report_id)

    with _locks_guard:
        _locks.pop(report_id, None)


def session_file(report_id: str) -> Path:
    """Return the deterministic path used for a session's temporary file."""
    return _session_path(report_id)
