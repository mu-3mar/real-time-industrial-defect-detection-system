"""Background worker thread for running detection sessions."""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Optional, Union, Set

import numpy as np

from core.firebase_client import publish_detection
from core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SessionWorker(threading.Thread):
    """
    Runs a detection session in a background thread.
    All factory/line/camera metadata is provided at creation; no static mappings.
    """

    def __init__(
        self,
        session_id: str,
        camera_source: Union[str, int],
        factory_id: str,
        factory_name: str,
        production_line_id: str,
        production_line_name: str,
        camera_id: str,
        station_id: str,
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.camera_source = camera_source
        self.factory_id = factory_id
        self.factory_name = factory_name
        self.production_line_id = production_line_id
        self.production_line_name = production_line_name
        self.camera_id = camera_id
        self.station_id = station_id
        self._box_cfg = box_cfg
        self._defect_cfg = defect_cfg
        self._stream_cfg = {**stream_cfg, "source": camera_source}
        self._loop = loop

        self._stop_event = threading.Event()
        self._started_at: Optional[datetime] = None
        self._pipeline_ref: Optional[Pipeline] = None
        self._tracks: Set[Any] = set()
        self._tracks_lock = threading.Lock()

    def _on_result(self, is_defect: bool) -> None:
        """Callback when box exits; publish result to Firebase Realtime Database."""
        logger.info(
            "Publishing detection → session=%s line=%s defect=%s",
            self.session_id, self.production_line_id, is_defect,
        )
        timestamp = datetime.utcnow().isoformat() + "Z"
        model_version = str(self._defect_cfg.get("model_version", "1.0"))
        confidence = 1.0 if is_defect else 0.0
        publish_detection(
            factory_id=self.factory_id,
            production_line_id=self.production_line_id,
            session_id=self.session_id,
            timestamp=timestamp,
            defect=is_defect,
            camera_id=self.camera_id,
            station_id=self.station_id,
            factory_name=self.factory_name,
            production_line_name=self.production_line_name,
            model_version=model_version,
            confidence=confidence,
        )

    def _on_frame(self, frame: np.ndarray) -> None:
        """Callback from pipeline; update all active WebRTC tracks."""
        with self._tracks_lock:
            tracks = list(self._tracks)
        for track in tracks:
            try:
                track.update_frame(frame)
            except Exception as e:
                logger.error("Error updating track for session %s: %s", self.session_id, e)

    def run(self) -> None:
        """Entry point for worker thread."""
        self._started_at = datetime.utcnow()
        try:
            self._pipeline_ref = Pipeline(
                box_cfg=self._box_cfg,
                defect_cfg=self._defect_cfg,
                stream_cfg=self._stream_cfg,
                headless=True,
                on_result_callback=self._on_result,
                on_frame_callback=self._on_frame,
            )
            self._pipeline_ref.run(stop_event=self._stop_event)
        except Exception as e:
            logger.exception("Session %s pipeline error: %s", self.session_id, e)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def get_info(self) -> dict:
        """Return session metadata for GET /api/sessions."""
        return {
            "session_id": self.session_id,
            "factory_id": self.factory_id,
            "factory_name": self.factory_name,
            "production_line_id": self.production_line_id,
            "production_line_name": self.production_line_name,
            "camera_id": self.camera_id,
            "camera_source": str(self.camera_source),
            "status": "active",
        }

    def add_track(self, track: Any) -> None:
        """Add a WebRTC video track to receive frames."""
        with self._tracks_lock:
            self._tracks.add(track)

    def remove_track(self, track: Any) -> None:
        """Remove a WebRTC video track."""
        with self._tracks_lock:
            self._tracks.discard(track)
