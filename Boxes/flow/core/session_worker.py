"""Background worker thread for running detection sessions."""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np

from core.firebase_client import publish_session_info
from core.pipeline import Pipeline
from core.pipeline_diagnostics import get_diagnostics
from core.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class SessionWorker(threading.Thread):
    """
    Runs a detection session: starts camera, feeds frames to the single
    inference thread via PipelineManager.
    """

    def __init__(
        self,
        report_id: str,
        camera_source: Union[str, int],
        production_line_id: str,
        box_cfg: dict,
        defect_cfg: dict,
        stream_cfg: dict,
        app_cfg: dict,
        loop: asyncio.AbstractEventLoop,
        target_speed: Optional[int] = None,
        max_temp: Optional[int] = None,
        max_amps: Optional[int] = None,
        command_state: Optional[str] = None,
        emergency_state: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.report_id = report_id
        self.camera_source = camera_source
        self.production_line_id = production_line_id
        self._box_cfg = box_cfg
        self._defect_cfg = defect_cfg
        self._stream_cfg = {**stream_cfg, "source": camera_source}
        self._app_cfg = app_cfg
        self._loop = loop

        # Initialize structured session info with default and provided values
        defaults = self._app_cfg.get("session_defaults", {})
        tel_def = defaults.get("telemetry", {})
        ctrl_def = defaults.get("control", {})
        config_def = defaults.get("config", {})

        self.session_info = {
            "telemetry": {
                "rpm_actual": tel_def.get("rpm_actual", 0),
                "predicted_temp": tel_def.get("predicted_temp", 0),
                "torque": tel_def.get("torque", 0),
            },
            "control": {
                "target_speed": target_speed if target_speed is not None else ctrl_def.get("target_speed", 100),
                "machine_status": ctrl_def.get("machine_status", "idle"),
                "command_state": command_state if command_state is not None else ctrl_def.get("command_state", "off"),
                "emergency_state": emergency_state if emergency_state is not None else ctrl_def.get("emergency_state", "normal"),
            },
            "config": {
                "max_temp": max_temp if max_temp is not None else config_def.get("max_temp", 80),
                "max_amps": max_amps if max_amps is not None else config_def.get("max_amps", 10),
            },
        }

        self._stop_event = threading.Event()
        self._started_at: Optional[datetime] = None
        self._pipeline_ref: Optional[Pipeline] = None
        self._feeder_thread: Optional[threading.Thread] = None
        self._manager = PipelineManager.get_instance()

    def _camera_feeder_loop(self) -> None:
        """Feed frames from this session's stream into the shared frame queue."""
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
            # Publish session_info immediately on startup
            publish_session_info(self.report_id, self.session_info)

            self._pipeline_ref = Pipeline(
                box_cfg=self._box_cfg,
                defect_cfg=self._defect_cfg,
                stream_cfg=self._stream_cfg,
                headless=True,
                on_result_callback=None,
                on_frame_callback=None,
            )
            self._pipeline_ref.stream.start()

            firebase_meta = {
                "report_id": self.report_id,
                "session_info": self.session_info,
            }
            self._manager.register_session(
                self.report_id,
                self._pipeline_ref,
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
        """Return report summary for GET /api/reports: report_id."""
        return {
            "report_id": self.report_id,
            "viewers_count": 0,
        }
