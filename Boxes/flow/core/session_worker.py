"""Background worker thread for running detection sessions."""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Optional, Union, List, Set

import numpy as np

from core.mqtt_client import MqttClient
from core.pipeline import Pipeline
# We avoid direct import of VideoTransformTrack here to avoid circular dependencies
# or issues if it's not needed by the worker logic directly other than typing.
# But actually we need to Type hint it? 
# For now, use Any or specific type if possible. 
# Better to not strictly type hint with the class if it causes issues, but it should be fine.

logger = logging.getLogger(__name__)


class SessionWorker(threading.Thread):
    """
    Runs a detection session in a background thread.
    Manages WebRTC video tracks for streaming.
    """

    def __init__(
        self,
        report_id: str,
        production_line: str,
        camera_source: Union[str, int],
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        mqtt_client: MqttClient,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(daemon=True)
        self.report_id = report_id
        self.production_line = production_line
        self.camera_source = camera_source
        self._box_cfg = box_cfg
        self._defect_cfg = defect_cfg
        # Override source in stream cfg
        self._stream_cfg = {**stream_cfg, "source": camera_source}
        self._mqtt_client = mqtt_client
        self._loop = loop  # Main event loop (kept for compatibility, though less used now)

        self._stop_event = threading.Event()
        self._started_at: Optional[datetime] = None
        self._pipeline_ref: Optional[Pipeline] = None
        
        # WebRTC Tracks
        self._tracks: Set[Any] = set() # Set of VideoTransformTrack
        self._tracks_lock = threading.Lock()

    def _on_result(self, is_defect: bool) -> None:
        """Callback when box exits; publish result to MQTT."""
        self._mqtt_client.publish_insight(
            production_line=self.production_line,
            report_id=self.report_id,
            defect=is_defect,
        )

    def _on_frame(self, frame: np.ndarray) -> None:
        """
        Callback from pipeline with new annotated frame.
        Update all active WebRTC tracks.
        """
        with self._tracks_lock:
            # Create a localized copy of tracks for iteration
            tracks = list(self._tracks)
        
        for track in tracks:
            try:
                track.update_frame(frame)
            except Exception as e:
                logger.error("Error updating track for session %s: %s", self.report_id, e)

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
            logger.exception("Session %s pipeline error: %s", self.report_id, e)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def get_info(self) -> dict:
        """Return session info for list endpoint."""
        with self._tracks_lock:
            viewer_count = len(self._tracks)
            
        info = {
            "report_id": self.report_id,
            "camera_source": str(self.camera_source),
            "status": "running",
            "started_at": (
                self._started_at.isoformat() + "Z" if self._started_at else None
            ),
            "active_viewers": viewer_count,
        }
        
        # Inject metrics from pipeline if running
        if self._pipeline_ref is not None:
            info["pipeline_fps"] = self._pipeline_ref.pipeline_fps
            info["camera_fps_estimate"] = self._pipeline_ref.camera_fps_estimate
            info["queue_latency_ms"] = self._pipeline_ref.queue_latency_ms
            
        return info

    def add_track(self, track: Any) -> None:
        """Add a WebRTC video track to receive frames."""
        with self._tracks_lock:
            self._tracks.add(track)

    def remove_track(self, track: Any) -> None:
        """Remove a WebRTC video track."""
        with self._tracks_lock:
            self._tracks.discard(track)