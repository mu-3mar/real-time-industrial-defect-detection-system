"""
Video stream reader with OpenCV.

Architecture (Tasks 1, 2, 4):
==============================
CamStream contains a dedicated capture thread (producer) that reads frames from
the camera independently of the detection pipeline. The latest frame is stored in
a collections.deque(maxlen=1), which acts as an atomic single-slot buffer with no
explicit locking required for reads (the deque handles concurrent access safely at
the single-element level).

  Camera HW → _capture_loop() [thread]
                    ↓
              _frame_queue (deque maxlen=1)   ← newest frame only; old frames dropped
                    ↓
         get_latest_frame()  ← called by Pipeline; never blocks on camera IO

This eliminates the main bottleneck where the pipeline loop was blocked waiting
for camera device I/O (which can stall during YUYV decode or USB bus contention).
"""

import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CamStream:
    """
    OpenCV camera reader with a dedicated background capture thread.

    Task 4 — Camera Performance Enforcement:
      - Forces V4L2 backend (Linux; avoids generic fallback).
      - Forces MJPG fourcc (avoids YUYV which caps many webcams at ~10 FPS).
      - Applies configured resolution.
      - Logs actual backend, format, resolution, and estimated FPS after open.

    Task 1+2 — Decoupled Capture / Thread-Safe Frame Queue:
      - A background thread continuously reads frames at hardware FPS.
      - Frames are stored in deque(maxlen=1) — old frames are dropped automatically.
      - get_latest_frame() is non-blocking; returns the most recent frame or (False, None).
    """

    def __init__(self, source, width: int, height: int):
        """
        Initialize camera and configure hardware parameters.

        Args:
            source: Camera device index or file path.
            width:  Deprecated. Kept for backwards compatibility; capture always
                    uses the camera’s native resolution.
            height: Deprecated. Kept for backwards compatibility; capture always
                    uses the camera’s native resolution.
        """
        # ── Task 4: Force V4L2 backend + MJPG for maximum Linux camera FPS ──
        # NOTE (Resolution Policy):
        #   We deliberately DO NOT set CAP_PROP_FRAME_WIDTH / HEIGHT here.
        #   The camera is opened at its native resolution and all downstream
        #   processing (detection + rendering + WebRTC) operates on the
        #   full‑resolution frames. Any inference-time resizing must be done
        #   on a copy inside the pipeline, never by downscaling the capture.
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        if not self.cap.isOpened():
            logger.error("Failed to open video source: %s", source)
            raise RuntimeError(f"Could not open video source: {source}")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.error("Failed to read initial frame from source: %s", source)
            raise RuntimeError(f"Could not read initial frame from source: {source}")

        print("SHAPE:", frame.shape)
        print("FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
        print("FOURCC:", fourcc_str)
        print("BACKEND:", self.cap.getBackendName())
        print("CONVERT_RGB:", self.cap.get(cv2.CAP_PROP_CONVERT_RGB))

        # Read back actual properties (camera may not honour all requests exactly)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])

        logger.debug(
            "CamStream opened: backend=V4L2, fourcc=%s, res=%dx%d, estimated_fps=%.1f",
            fourcc_str, actual_w, actual_h, actual_fps,
        )

        # ── Task 1+2: Single-slot frame queue — newest frame only ──
        # deque(maxlen=1) provides atomic single-element replacement;
        # appending a new frame automatically evicts the old one.
        self._frame_queue: deque = deque(maxlen=1)

        # Capture thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Metrics: camera-side FPS estimate (updated by capture thread)
        self._camera_fps: float = actual_fps if actual_fps > 0 else 0.0
        self._capture_frame_count: int = 0
        self._capture_last_time: float = time.time()

        # Track when a frame was enqueued (for latency measurement by consumer)
        self._last_enqueue_time: float = 0.0

        # Diagnostics: periodic log of camera_capture_fps
        self._diag_last_log: float = 0.0
        self._diag_log_interval: float = 10.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "CamStream":
        """Start the background capture thread. Call once before get_latest_frame()."""
        if self._thread is not None and self._thread.is_alive():
            return self  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CamCaptureThread",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Camera capture thread started")
        return self

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the most recent frame produced by the capture thread.

        Non-blocking — if no frame is available yet, returns (False, None).
        The pipeline should handle the empty case by skipping the iteration.

        Returns:
            (True, frame_ndarray)  when a frame is available.
            (False, None)          when the queue is empty (camera not ready yet).
        """
        if self._frame_queue:
            return True, self._frame_queue[-1]
        return False, None

    @property
    def camera_fps(self) -> float:
        """Estimated camera-side capture FPS (measured by capture thread)."""
        return self._camera_fps

    @property
    def last_enqueue_time(self) -> float:
        """Timestamp (time.time()) of the last frame enqueued. Used for latency calc."""
        return self._last_enqueue_time

    def release(self) -> None:
        """Stop the capture thread and release the VideoCapture resource."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logger.debug("CamStream released")

    # ── Legacy compatibility: direct read() for callers that haven't been migrated ──

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Legacy blocking read. Retained for backward compatibility only.
        New code should use start() + get_latest_frame() instead.
        """
        return self.cap.read()

    # ── Private ──────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Background thread: read frames from camera at hardware speed.

        This loop is intentionally lightweight — no inference, no callbacks,
        just read → enqueue → repeat. The deque(maxlen=1) ensures that if the
        consumer (pipeline) is slower than the camera, old frames are silently
        dropped and only the newest is kept.
        """
        logger.debug("Camera capture loop running (produces to deque maxlen=1)")
        fps_count = 0
        fps_last_time = time.time()
        _read_fail_logged = False

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if not _read_fail_logged:
                    logger.warning("Camera read failed (ret=False)")
                    _read_fail_logged = True
                time.sleep(0.05)
                continue
            _read_fail_logged = False

            # Enqueue — automatically evicts old frame if consumer is slow
            self._frame_queue.append(frame)
            self._last_enqueue_time = time.time()

            # ── Measure actual capture FPS every 30 frames ──
            fps_count += 1
            self._capture_frame_count += 1
            if fps_count >= 30:
                now = time.time()
                elapsed = now - fps_last_time
                if elapsed > 0:
                    self._camera_fps = fps_count / elapsed
                fps_last_time = now
                fps_count = 0

            # ── Diagnostics: log camera_capture_fps periodically ──
            now = time.time()
            if now - self._diag_last_log >= self._diag_log_interval:
                self._diag_last_log = now
                logger.debug(
                    "camera_capture_fps=%.1f (%d frames)",
                    self._camera_fps, self._capture_frame_count,
                )

        logger.debug(
            "Camera capture loop exited. Total frames captured: %d",
            self._capture_frame_count,
        )
