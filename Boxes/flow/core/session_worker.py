"""Background worker thread for running detection sessions."""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional, Union, Set

import numpy as np

from core.pipeline import Pipeline
from core.pipeline_diagnostics import get_diagnostics
from core.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class SessionWorker(threading.Thread):
    """
    Runs a detection session: starts camera, feeds frames to the single
    inference thread via PipelineManager, and registers pipeline/tracks/Firebase meta.
    Firebase and WebRTC updates are handled by manager workers; inference never waits on them.
    """

    def __init__(
        self,
        report_id: str,
        camera_source: Union[str, int],
        production_line_id: str,
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(daemon=True)
        self.report_id = report_id
        self.camera_source = camera_source
        self.production_line_id = production_line_id
        self._box_cfg = box_cfg
        self._defect_cfg = defect_cfg
        self._stream_cfg = {**stream_cfg, "source": camera_source}
        self._loop = loop

        self._stop_event = threading.Event()
        self._started_at: Optional[datetime] = None
        self._pipeline_ref: Optional[Pipeline] = None
        self._tracks: Set[Any] = set()
        self._tracks_lock = threading.Lock()
        self._feeder_thread: Optional[threading.Thread] = None
        self._manager = PipelineManager.get_instance()

    def _camera_feeder_loop(self) -> None:
        """Feed frames from this session's stream into the shared frame queue (non-blocking for inference)."""
        stream = self._pipeline_ref.stream if self._pipeline_ref else None
        if stream is None:
            return
        diag = get_diagnostics()
        while not self._stop_event.is_set():
            ret, frame = stream.get_latest_frame()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            enqueue_time = stream.last_enqueue_time
            camera_fps = stream.camera_fps
            diag.set_camera_capture_fps(camera_fps)
            ok = self._manager.put_frame(self.report_id, frame, enqueue_time, camera_fps)
            diag.record_frame_enqueue(ok)
            if not ok:
                time.sleep(0.001)
            # Single attempt only; no retry loop

    def run(self) -> None:
        """Start camera, register with PipelineManager, run camera feeder until stop."""
        self._started_at = datetime.utcnow()
        try:
            self._pipeline_ref = Pipeline(
                box_cfg=self._box_cfg,
                defect_cfg=self._defect_cfg,
                stream_cfg=self._stream_cfg,
                headless=True,
                on_result_callback=None,
                on_frame_callback=None,
            )
            self._pipeline_ref.stream.start()

            firebase_meta = {"report_id": self.report_id}
            self._manager.register_session(
                self.report_id,
                self._pipeline_ref,
                (self._tracks, self._tracks_lock),
                firebase_meta,
            )

            self._feeder_thread = threading.Thread(
                target=self._camera_feeder_loop,
                name=f"CameraFeeder-{self.report_id}",
                daemon=True,
            )
            self._feeder_thread.start()

            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)
        except Exception as e:
            logger.error("[Error] session %s: %s", self.report_id, e)
        finally:
            self._manager.unregister_session(self.report_id)
            logger.debug("Session %s worker ended", self.report_id)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def get_info(self) -> dict:
        """Return report summary for GET /api/reports: report_id, viewers_count."""
        with self._tracks_lock:
            viewers_count = len(self._tracks)
        return {
            "report_id": self.report_id,
            "viewers_count": viewers_count,
        }

    def add_track(self, track: Any) -> None:
        """Add a WebRTC video track to receive frames."""
        with self._tracks_lock:
            self._tracks.add(track)

    def remove_track(self, track: Any) -> None:
        """Remove a WebRTC video track."""
        with self._tracks_lock:
            self._tracks.discard(track)
