"""
Lightweight runtime diagnostics for the threaded pipeline.
Used to identify FPS / latency bottlenecks without changing architecture.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Interval between diagnostic logs (seconds)
DIAG_LOG_INTERVAL = 10.0


class PipelineDiagnostics:
    """Thread-safe counters and latency samples for pipeline stages."""

    def __init__(self):
        self._lock = threading.Lock()
        # Camera / frame enqueue
        self.camera_capture_fps: float = 0.0  # Set by camera feeder from stream
        self.frame_enqueue_count: int = 0
        self.frame_enqueue_drops: int = 0
        self._frame_enqueue_window_count: int = 0
        self._frame_enqueue_window_start: float = time.time()
        # Inference
        self.inference_count: int = 0
        self.inference_latency_sum: float = 0.0
        self.inference_latency_max: float = 0.0
        self._inference_window_count: int = 0
        self._inference_window_start: float = time.time()
        self.inference_fps: float = 0.0
        # Result queue
        self.result_queue_drops: int = 0
        # WebRTC
        self.webrtc_time_sum: float = 0.0
        self.webrtc_frame_count: int = 0
        # Last full log time
        self._last_log_time: float = time.time()

    def record_frame_enqueue(self, success: bool) -> None:
        with self._lock:
            if success:
                self.frame_enqueue_count += 1
                self._frame_enqueue_window_count += 1
            else:
                self.frame_enqueue_drops += 1

    def set_camera_capture_fps(self, fps: float) -> None:
        with self._lock:
            self.camera_capture_fps = fps

    def record_inference(self, latency_sec: float) -> None:
        with self._lock:
            self.inference_count += 1
            self.inference_latency_sum += latency_sec
            if latency_sec > self.inference_latency_max:
                self.inference_latency_max = latency_sec
            self._inference_window_count += 1

    def record_result_queue_drop(self) -> None:
        with self._lock:
            self.result_queue_drops += 1

    def record_webrtc_update(self, elapsed_sec: float) -> None:
        with self._lock:
            self.webrtc_time_sum += elapsed_sec
            self.webrtc_frame_count += 1

    def get_snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                self.inference_latency_sum / self.inference_count
                if self.inference_count > 0 else 0.0
            )
            webrtc_avg_ms = (
                (self.webrtc_time_sum / self.webrtc_frame_count) * 1000.0
                if self.webrtc_frame_count > 0 else 0.0
            )
            return {
                "camera_capture_fps": self.camera_capture_fps,
                "frame_enqueue_count": self.frame_enqueue_count,
                "frame_enqueue_drops": self.frame_enqueue_drops,
                "inference_count": self.inference_count,
                "inference_fps": self.inference_fps,
                "inference_latency_avg_ms": avg_latency * 1000.0,
                "inference_latency_max_ms": self.inference_latency_max * 1000.0,
                "result_queue_drops": self.result_queue_drops,
                "webrtc_avg_ms": webrtc_avg_ms,
                "webrtc_frame_count": self.webrtc_frame_count,
            }

    def maybe_log(self, frame_queue_size: int, result_queue_size: int, firebase_queue_size: int) -> bool:
        """If interval elapsed, log diagnostics and return True."""
        now = time.time()
        with self._lock:
            if now - self._last_log_time < DIAG_LOG_INTERVAL:
                return False
            self._last_log_time = now
            # Window rates (then reset windows)
            enqueue_elapsed = now - self._frame_enqueue_window_start
            inference_elapsed = now - self._inference_window_start
            enqueue_rate = self._frame_enqueue_window_count / enqueue_elapsed if enqueue_elapsed > 0 else 0.0
            inf_fps = self._inference_window_count / inference_elapsed if inference_elapsed > 0 else 0.0
            self._frame_enqueue_window_count = 0
            self._frame_enqueue_window_start = now
            self._inference_window_count = 0
            self._inference_window_start = now
            self.inference_fps = inf_fps
            avg_latency = (
                self.inference_latency_sum / self.inference_count
                if self.inference_count > 0 else 0.0
            )
            webrtc_avg_ms = (
                (self.webrtc_time_sum / self.webrtc_frame_count) * 1000.0
                if self.webrtc_frame_count > 0 else 0.0
            )
            log_camera_fps = self.camera_capture_fps
            log_drops = self.frame_enqueue_drops
            log_result_drops = self.result_queue_drops
            log_latency_avg_ms = avg_latency * 1000.0
            log_latency_max_ms = self.inference_latency_max * 1000.0
        logger.debug(
            "diagnostics: cam_fps=%.1f enq_rate=%.1f drops=%d inf_fps=%.1f "
            "lat_avg=%.1fms lat_max=%.1fms q=%d/%d/%d",
            log_camera_fps,
            enqueue_rate,
            log_drops,
            inf_fps,
            log_latency_avg_ms,
            log_latency_max_ms,
            frame_queue_size,
            result_queue_size,
            firebase_queue_size,
        )
        return True


# Singleton used by pipeline_manager and session_worker
_diagnostics: Optional[PipelineDiagnostics] = None
_diagnostics_lock = threading.Lock()


def get_diagnostics() -> PipelineDiagnostics:
    global _diagnostics
    with _diagnostics_lock:
        if _diagnostics is None:
            _diagnostics = PipelineDiagnostics()
        return _diagnostics
