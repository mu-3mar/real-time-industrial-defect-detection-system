"""Background worker thread for running detection sessions."""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Union

import cv2

from core.backend_client import BackendClient
from core.pipeline import Pipeline

logger = logging.getLogger(__name__)

VIEWER_WINDOW_PREFIX = "Session"


def _viewer_loop(
    report_id: str,
    canvas_ref: Dict[str, Any],
    stop_event: threading.Event,
) -> None:
    """Display loop for viewer window."""
    window_name = f"{VIEWER_WINDOW_PREFIX} | {report_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while not stop_event.is_set():
            canvas = canvas_ref.get("canvas")
            if canvas is not None:
                cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


class SessionWorker(threading.Thread):
    """
    Runs a detection session in a background thread.
    Supports headless mode with optional viewer attachment.
    """

    def __init__(
        self,
        report_id: str,
        camera_source: Union[str, int],
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        backend_client: BackendClient,
        headless: bool = True,
    ):
        super().__init__(daemon=True)
        self.report_id = report_id
        self.camera_source = camera_source
        self._box_cfg = box_cfg
        self._defect_cfg = defect_cfg
        self._stream_cfg = {**stream_cfg, "source": camera_source}
        self._backend_client = backend_client
        self._headless = headless

        self._stop_event = threading.Event()
        self._viewer_canvas_ref: Dict[str, Any] = {}
        self._viewer_thread: Optional[threading.Thread] = None
        self._viewer_stop_event = threading.Event()
        self._started_at: Optional[datetime] = None

    def _on_result(self, is_defect: bool) -> None:
        """Callback when box exits; send result to backend."""
        self._backend_client.send_result(self.report_id, is_defect)

    def run(self) -> None:
        """Entry point for worker thread."""
        self._started_at = datetime.utcnow()
        try:
            pipeline = Pipeline(
                box_cfg=self._box_cfg,
                defect_cfg=self._defect_cfg,
                stream_cfg=self._stream_cfg,
                headless=self._headless,
                on_result_callback=self._on_result,
                viewer_canvas_ref=self._viewer_canvas_ref if self._headless else None,
            )
            pipeline.run(stop_event=self._stop_event)
        except Exception as e:
            logger.exception("Session %s pipeline error: %s", self.report_id, e)
        finally:
            self._viewer_stop_event.set()

    def stop(self) -> None:
        """Signal the worker and viewer to stop."""
        self._stop_event.set()
        self._viewer_stop_event.set()
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            self._viewer_thread.join(timeout=2.0)

    def get_info(self) -> dict:
        """Return session info for list endpoint."""
        return {
            "report_id": self.report_id,
            "camera_source": str(self.camera_source),
            "status": "running",
            "started_at": (
                self._started_at.isoformat() + "Z" if self._started_at else None
            ),
            "viewer_attached": (
                self._viewer_thread is not None
                and self._viewer_thread.is_alive()
            ),
        }

    def start_viewer(self) -> bool:
        """
        Attach a viewer to this session.

        Returns:
            True if viewer started, False if already attached
        """
        if not self._headless:
            return False
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            return True
        self._viewer_stop_event.clear()
        self._viewer_thread = threading.Thread(
            target=_viewer_loop,
            args=(self.report_id, self._viewer_canvas_ref, self._viewer_stop_event),
            daemon=True,
        )
        self._viewer_thread.start()
        logger.info("Viewer attached to session %s", self.report_id)
        return True

    def stop_viewer(self) -> bool:
        """
        Detach the viewer window. Session continues running.

        Returns:
            True if viewer was stopped, False if no viewer attached
        """
        if self._viewer_thread is None or not self._viewer_thread.is_alive():
            return False
        self._viewer_stop_event.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=2.0)
        logger.info("Viewer detached from session %s", self.report_id)
        return True