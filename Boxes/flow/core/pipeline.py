import time
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple, Union

import cv2
import numpy as np
import threading

from core.state import AppState
from core.stream import CamStream
from core.model_loader import ModelLoader
from detectors.detector import Detector
from utils.visualizer import Visualizer
from utils.geometry import box_iou, smooth_bbox

logger = logging.getLogger(__name__)

# Layout constants (shared by GUI and viewer canvas)
INFO_WIDTH = 300
ROI_WIDTH = 400
ROI_CENTER_OFFSET = 420


class Pipeline:
    def __init__(
        self,
        box_cfg,
        defect_cfg,
        stream_cfg,
        headless: bool = True,
        on_result_callback: Optional[Callable[[bool], None]] = None,
        on_frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        Initialize detection pipeline.

        Args:
            box_cfg:             Box detector configuration
            defect_cfg:          Defect detector configuration
            stream_cfg:          Stream configuration (includes throttle settings)
            headless:            If True, skip own GUI window (always True now)
            on_result_callback:  Optional callback when box exits with decision
            on_frame_callback:   Optional callback to receive the annotated frame
        """
        self.headless = headless
        self.on_result_callback = on_result_callback
        self.on_frame_callback = on_frame_callback

        # 1. Setup Stream (producer thread, decouple camera IO from inference)
        #    Camera runs independently at hardware FPS; pipeline reads latest frame only.
        self.strict_debug_mode = bool(stream_cfg.get("strict_debug_mode", False))
        self.strict_current_frame_mode = bool(stream_cfg.get("strict_current_frame_mode", False))
        self.strict_crop_padding_ratio = float(stream_cfg.get("strict_crop_padding_ratio", 0.08))
        self.strict_frame_queue_size = int(stream_cfg.get("strict_frame_queue_size", 120))
        self.stream = CamStream(
            stream_cfg["source"],
            stream_cfg["width"],
            stream_cfg["height"],
            strict_per_frame=self.strict_debug_mode,
            frame_queue_size=self.strict_frame_queue_size,
        )

        # 2. Setup Detectors with pre-loaded models
        model_loader = ModelLoader.get_instance()
        self.box_detector = Detector(
            model_loader.get_box_model(),
            box_cfg["conf_thres"],
            box_cfg["iou_thres"],
            box_cfg["device"],
        )
        self.defect_detector = Detector(
            model_loader.get_defect_model(),
            defect_cfg["conf_thres"],
            defect_cfg["iou_thres"],
            defect_cfg["device"],
        )

        # 3. Setup State (single-track, smooth voting)
        self.state = AppState(defect_cfg.get("stability", {}))

        # Tracking: IoU match threshold and bbox smoothing
        tracking = defect_cfg.get("tracking", {})
        self.iou_match_threshold = float(tracking.get("iou_threshold", 0.35))
        self.bbox_smooth_alpha = float(tracking.get("bbox_smooth_alpha", 0.6))
        self._current_track: Optional[np.ndarray] = None  # smoothed bbox in ROI coords

        # Rendering: defect visibility when box partially exits frame (draw only, not tracking)
        rendering = defect_cfg.get("rendering", {})
        self.defect_visibility_threshold = float(rendering.get("visibility_threshold", 0.2))

        # 4. Visualizer: always initialized for streaming
        self.LEFT_X = INFO_WIDTH + ROI_CENTER_OFFSET - ROI_WIDTH // 2
        self.RIGHT_X = INFO_WIDTH + ROI_CENTER_OFFSET + ROI_WIDTH // 2
        self.visualizer = Visualizer(
            stream_cfg["width"],
            stream_cfg["height"],
            INFO_WIDTH,
            ROI_WIDTH,
        )

        # ── Task 3: Adaptive Detection Throttle ──────────────────────────────
        # Read throttle values from stream_cfg (set via stream.yaml).
        # box_detect_every_n_frames=1  → detect on every frame (no skip)
        # defect_detect_every_n_frames=3 → detect every 3rd frame (2 skipped)
        # Both values default to safe settings if not present in config.
        self.box_detect_every_n: int = int(
            stream_cfg.get("box_detect_every_n_frames", 1)
        )
        self.defect_detect_every_n: int = int(
            stream_cfg.get("defect_detect_every_n_frames", 3)
        )
        if self.strict_debug_mode:
            self.box_detect_every_n = 1
            self.defect_detect_every_n = 1
        if self.strict_current_frame_mode:
            self.box_detect_every_n = 1
            self.defect_detect_every_n = 1
            self.state.max_missed = 0
            self.state.track_grace_frames = 0
        logger.debug(
            "Detection throttle: box every %d frames, defect every %d frames",
            self.box_detect_every_n,
            self.defect_detect_every_n,
        )

        # Cached box detection result (reused between throttled frames)
        self._last_boxes_roi: np.ndarray = np.zeros((0, 4))

        # Reusable canvas double-buffer to avoid per-frame allocation (same thread, no copy on return)
        self._canvas_bufs: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._canvas_buf_index: int = 0

        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0

        # ── Task 6: Runtime metrics (read by api_server /api/metrics) ──
        self.pipeline_fps: float = 0.0
        self.camera_fps_estimate: float = 0.0
        self.queue_latency_ms: float = 0.0
        self._last_diag_log_time: float = time.time()
        self._diag_log_interval: float = 30.0

    def run_step(
        self,
        frame: np.ndarray,
        enqueue_time: Optional[float] = None,
        camera_fps: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[bool, str]]]:
        """
        Process one frame: detection, tracking, drawing. Used by the single
        inference thread in a producer-consumer setup. Does NOT call
        on_result_callback or on_frame_callback; caller routes results via queues.

        Args:
            frame: BGR frame from camera.
            enqueue_time: Optional time.time() when frame was enqueued (for latency metrics).
            camera_fps: Optional camera FPS estimate.

        Returns:
            (canvas, exit_event): canvas is the annotated frame; exit_event is None
            normally, or (is_defect, detection_id) when a box just exited.
        """
        if enqueue_time is not None and enqueue_time > 0:
            self.queue_latency_ms = (time.time() - enqueue_time) * 1000.0
        if camera_fps is not None:
            self.camera_fps_estimate = camera_fps

        try:
            self.state.tick_recent_lost_track()
        except Exception:
            logger.debug("tick_recent_lost_track() encountered an error", exc_info=False)

        self.frame_count += 1
        self.update_fps()
        now = time.time()
        if now - self._last_diag_log_time >= self._diag_log_interval:
            logger.debug(
                "pipeline fps=%.1f camera_fps=%.1f latency=%.1fms",
                self.pipeline_fps, self.camera_fps_estimate, self.queue_latency_ms,
            )
            self._last_diag_log_time = now

        h, w = frame.shape[:2]
        info_width = self.visualizer.info_width
        expected_shape = (h, w + info_width, 3)
        if self._canvas_bufs is None or self._canvas_bufs[0].shape != expected_shape:
            self._canvas_bufs = (
                np.zeros((h, w + info_width, 3), dtype=np.uint8),
                np.zeros((h, w + info_width, 3), dtype=np.uint8),
            )
            self._canvas_buf_index = 0
        buf = self._canvas_bufs[self._canvas_buf_index]
        self._canvas_buf_index = 1 - self._canvas_buf_index
        buf[:, :info_width, :] = 235
        buf[:, info_width:, :] = frame
        canvas = buf
        self.visualizer.draw_layout(canvas)
        self.visualizer.draw_stats(canvas, self.state, self.fps)
        box_input = frame

        current_boxes = np.zeros((0, 4))
        if self.frame_count % self.box_detect_every_n == 0:
            box_result = self.box_detector.detect(box_input)
            current_boxes = (
                box_result.boxes.xyxy.cpu().numpy()
                if box_result.boxes is not None
                else np.zeros((0, 4))
            )
            self._last_boxes_roi = current_boxes
        boxes_roi = current_boxes if self.strict_current_frame_mode else self._last_boxes_roi

        matched_box, detected = self._match_track(boxes_roi)
        label, color, final_code = "—", (128, 128, 128), "OK"
        defect_boxes: List[np.ndarray] = []

        if detected and matched_box is not None:
            x1, y1, x2, y2 = matched_box[0], matched_box[1], matched_box[2], matched_box[3]
            if self.strict_debug_mode:
                # Debug mode: expand crop around full-frame detection box.
                box_w = max(1.0, float(x2 - x1))
                box_h = max(1.0, float(y2 - y1))
                pad_x = int(box_w * self.strict_crop_padding_ratio)
                pad_y = int(box_h * self.strict_crop_padding_ratio)
                crop_x1 = max(0, int(x1) - pad_x)
                crop_y1 = max(0, int(y1) - pad_y)
                crop_x2 = min(frame.shape[1], int(x2) + pad_x)
                crop_y2 = min(frame.shape[0], int(y2) + pad_y)
            else:
                crop_x1 = int(x1)
                crop_y1 = int(y1)
                crop_x2 = int(x2)
                crop_y2 = int(y2)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            is_defect, defect_boxes = self._check_defect_track(crop)
            self.state.update_history(is_defect)
            if defect_boxes:
                boxes_relative = [
                    (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
                    for d in defect_boxes
                ]
                self.state.add_defect_boxes_relative(boxes_relative)
            label, color, final_code = self.state.get_status()
            box_for_draw = matched_box.astype(np.float32)
            accumulated = self.state.get_accumulated_defect_boxes()
            is_early_phase = self.state.is_early_detection_phase()
            if self.strict_debug_mode:
                draw_x1, draw_y1, draw_x2, draw_y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(
                    canvas,
                    (self.visualizer.info_width + draw_x1, draw_y1),
                    (self.visualizer.info_width + draw_x2, draw_y2),
                    color,
                    2,
                )
                cv2.putText(
                    canvas,
                    label,
                    (self.visualizer.info_width + draw_x1, max(0, draw_y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                if accumulated and is_early_phase:
                    for d_box in accumulated:
                        dx1, dy1, dx2, dy2 = map(int, d_box)
                        abs_dx1 = self.visualizer.info_width + crop_x1 + dx1
                        abs_dy1 = crop_y1 + dy1
                        abs_dx2 = self.visualizer.info_width + crop_x1 + dx2
                        abs_dy2 = crop_y1 + dy2
                        cv2.rectangle(canvas, (abs_dx1, abs_dy1), (abs_dx2, abs_dy2), (0, 0, 255), 1)
            else:
                self.visualizer.draw_box(canvas, box_for_draw, label, color)
                if accumulated and is_early_phase:
                    self.visualizer.draw_defects(
                        canvas,
                        (int(x1), int(y1)),
                        accumulated,
                        is_early_phase=is_early_phase,
                        visibility_threshold=self.defect_visibility_threshold,
                    )

        if self.strict_current_frame_mode and not detected:
            self._current_track = None
            self.state.set_last_defect_result(False, [])

        self.state.increment_defect_lock_frame()
        just_exited, final_decision = self.state.process_entry_exit(detected)
        exit_event: Optional[Tuple[bool, str]] = None
        if just_exited:
            self._current_track = None
            is_defect = final_decision == "DEFECT"
            # detection_id = det_{total_count} (e.g. det_001)
            detection_id = f"det_{self.state.total_count:03d}"
            label_text = "DEFECT" if is_defect else "OK"
            logger.info("[Detection] %s (id=%s total=%d)", label_text, detection_id, self.state.total_count)
            exit_event = (is_defect, detection_id)

        return canvas, exit_event

    def run(self, stop_event: Optional[threading.Event] = None):
        """
        Run detection pipeline.

        Architecture (Tasks 1, 2):
        --------------------------
        The camera capture thread is started first and fills a deque(maxlen=1).
        The main pipeline loop reads the latest frame non-blocking from that
        queue, so inference latency never blocks camera capture.

              CamStream._capture_loop()  [background thread]
                        ↓
              deque(maxlen=1)            [newest frame only]
                        ↓
              Pipeline.run() loop        [this method, consumer]

        Args:
            stop_event: Optional threading event to signal stop
        """
        logger.debug("Pipeline started (headless=%s)", self.headless)

        # Start the camera capture thread (producer)
        self.stream.start()

        try:
            while True:
                if stop_event and stop_event.is_set():
                    logger.debug("Pipeline stopped")
                    break

                # ── Non-blocking frame read from camera queue (Task 1+2) ──
                ret, frame = self.stream.get_latest_frame()
                if not ret or frame is None:
                    # Queue empty: camera thread hasn't produced a frame yet.
                    # Tiny sleep avoids a busy-spin and keeps CPU usage negligible
                    # during the startup transition until the first frame arrives.
                    time.sleep(0.001)
                    continue

                # ── Queue latency: time since producer enqueued this frame (Task 6) ──
                enqueue_ts = self.stream.last_enqueue_time
                if enqueue_ts > 0:
                    self.queue_latency_ms = (time.time() - enqueue_ts) * 1000.0

                # Advance recent lost track age to ensure recovery window expires
                try:
                    self.state.tick_recent_lost_track()
                except Exception:
                    logger.debug("tick_recent_lost_track() encountered an error", exc_info=False)

                self.frame_count += 1
                self.update_fps()
                now = time.time()
                if now - self._last_diag_log_time >= self._diag_log_interval:
                    logger.debug(
                        "pipeline fps=%.1f camera_fps=%.1f latency=%.1fms",
                        self.pipeline_fps, self.camera_fps_estimate, self.queue_latency_ms,
                    )
                    self._last_diag_log_time = now
                h, w = frame.shape[:2]

                # Prepare Canvas
                canvas = cv2.copyMakeBorder(
                    frame, 0, 0, self.visualizer.info_width, 0,
                    cv2.BORDER_CONSTANT, value=(235, 235, 235),
                )
                self.visualizer.draw_layout(canvas)
                self.visualizer.draw_stats(canvas, self.state, self.fps)
                roi_offset = self.visualizer.info_width

                # ── STAGE 1: Box Detection (with throttle, Task 3) ──
                # Run the box detector every box_detect_every_n frames.
                # Between runs, reuse the cached result so tracking stays continuous.
                if self.frame_count % self.box_detect_every_n == 0:
                    box_result = self.box_detector.detect(frame)
                    self._last_boxes_roi = (
                        box_result.boxes.xyxy.cpu().numpy()
                        if box_result.boxes is not None
                        else np.zeros((0, 4))
                    )

                boxes_roi = self._last_boxes_roi

                # ── Track single box by IoU (smooth, stable identity) ──
                matched_box, detected = self._match_track(boxes_roi)
                label, color, final_code = "—", (128, 128, 128), "OK"
                defect_boxes: List[np.ndarray] = []

                if detected and matched_box is not None:
                    x1, y1, x2, y2 = matched_box[0], matched_box[1], matched_box[2], matched_box[3]
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    is_defect, defect_boxes = self._check_defect_track(crop)
                    self.state.update_history(is_defect)
                    # Store defects in box-relative coords (crop = box); IoU dedup in that space
                    if defect_boxes:
                        boxes_relative = [
                            (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
                            for d in defect_boxes
                        ]
                        self.state.add_defect_boxes_relative(boxes_relative)
                    label, color, final_code = self.state.get_status()
                    box_for_draw = matched_box.astype(np.float32)

                    self.visualizer.draw_box(canvas, box_for_draw, label, color)
                    # Convert box-relative defects to frame using current box position
                    # Only draw defects during early detection phase (first N frames after lock)
                    accumulated = self.state.get_accumulated_defect_boxes()
                    is_early_phase = self.state.is_early_detection_phase()
                    if accumulated and is_early_phase:
                        self.visualizer.draw_defects(
                            canvas,
                            (int(x1), int(y1)),
                            accumulated,
                            is_early_phase=is_early_phase,
                            visibility_threshold=self.defect_visibility_threshold,
                        )

                # Increment frame counter for early detection phase tracking
                self.state.increment_defect_lock_frame()

                just_exited, final_decision = self.state.process_entry_exit(detected)
                if just_exited:
                    self._current_track = None
                    is_defect = final_decision == "DEFECT"
                    # detection_id = det_{total_count} (e.g. det_001)
                    detection_id = f"det_{self.state.total_count:03d}"
                    label_text = "DEFECT" if is_defect else "OK"
                    logger.info("[Detection] %s (id=%s total=%d)", label_text, detection_id, self.state.total_count)
                    if self.on_result_callback:
                        # Modified callback to pass detection_id if supported, 
                        # but for backward compatibility with existing callbacks 
                        # that might only expect one arg, we check or just pass both.
                        try:
                            self.on_result_callback(is_defect, detection_id)
                        except TypeError:
                            self.on_result_callback(is_defect)
                    else:
                        logger.warning("No on_result_callback registered; result not published")

                # Update camera FPS estimate from stream for metrics
                self.camera_fps_estimate = self.stream.camera_fps

                # Send frame to callback if registered (for broadcasting)
                if self.on_frame_callback:
                    self.on_frame_callback(canvas)

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            self.cleanup()
            logger.debug("Pipeline ended (frames=%d)", self.frame_count)

    def _match_track(self, boxes_roi: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Match current track to detections by IoU; update smoothed bbox.
        Includes track grace period and recovery from brief loss (e.g., camera jitter).
        Returns (matched_bbox_in_roi_or_None, detected: bool).
        """
        if boxes_roi is None or len(boxes_roi) == 0:
            return None, False

        if self._current_track is not None:
            best_iou = -1.0
            best_idx = -1
            for i, box in enumerate(boxes_roi):
                iou = box_iou(self._current_track, box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= self.iou_match_threshold:
                matched = boxes_roi[best_idx]
                self._current_track = smooth_bbox(
                    self._current_track, matched, self.bbox_smooth_alpha
                )
                logger.debug("Track matched (IoU=%.3f)", best_iou)
                return self._current_track.copy(), True

            # No match with current track; store it for potential recovery
            logger.debug("Track lost, storing for recovery attempt")
            self.state.set_recent_lost_track(tuple(self._current_track))
            self._current_track = None
            return None, False

        # No current track: check for recent track recovery first
        if boxes_roi is not None and len(boxes_roi) > 0:
            # Try to recover with largest box (most likely the one to track)
            areas = (boxes_roi[:, 2] - boxes_roi[:, 0]) * (boxes_roi[:, 3] - boxes_roi[:, 1])
            best_idx = int(np.argmax(areas))
            new_box = boxes_roi[best_idx]

            # Check if this matches recently lost track (e.g., from camera jitter)
            if self.state.try_recover_recent_track(tuple(new_box)):
                # Recovered! Restart tracking with recovered box
                self._current_track = smooth_bbox(None, new_box.astype(np.float64), self.bbox_smooth_alpha)
                return self._current_track.copy(), True

            # No recovery; start new track
            new_box_float = new_box.astype(np.float64)
            self._current_track = smooth_bbox(None, new_box_float, self.bbox_smooth_alpha)
            logger.debug("New track started")
            return self._current_track.copy(), True

        return None, False

    def _check_defect_track(self, crop: np.ndarray) -> Tuple[bool, List]:
        """
        Run defect detection on the tracked box crop.

        Task 3: Defect detector is throttled to run every `defect_detect_every_n`
        frames. Between runs the last result is returned from cache in AppState,
        preserving tracking correctness without forcing inference every frame.

        Returns (is_defect, defect_boxes).
        """
        # defect_detect_every_n=3 → run on frames 0, 3, 6, 9 … (every 3rd)
        run_this_frame = (
            True if self.strict_current_frame_mode
            else (self.frame_count % self.defect_detect_every_n == 0)
        )
        if self.strict_current_frame_mode and crop.size == 0:
            return False, []
        if run_this_frame and crop.size > 0:
            d_res = self.defect_detector.detect(crop)
            hole_detected = False
            defect_boxes = []
            if d_res.boxes is not None:
                defects = d_res.boxes.data.cpu().numpy()
                for d in defects:
                    cls = int(d[5])
                    if cls == 0:
                        hole_detected = True
                        defect_boxes.append(d[:4])
            if not self.strict_current_frame_mode:
                self.state.set_last_defect_result(hole_detected, defect_boxes)
            return hole_detected, defect_boxes
        if self.strict_current_frame_mode:
            return False, []
        return self.state.get_last_defect_result()

    def update_fps(self):
        """Measure pipeline FPS every 10 frames and store for metrics."""
        if self.frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            if elapsed > 0:
                self.fps = 10 / elapsed
                self.pipeline_fps = self.fps  # expose for /api/metrics
            self.last_time = current_time

    def cleanup(self):
        """Release resources."""
        self.stream.release()
        logger.debug("Pipeline resources cleaned up")
