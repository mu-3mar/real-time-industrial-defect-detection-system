import cv2
import time
from datetime import datetime
import numpy as np
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple

from core.state import AppState
from core.stream import CamStream
from core.model_loader import ModelLoader
from detectors.detector import Detector
from utils.visualizer import Visualizer
from utils.geometry import box_iou, smooth_bbox

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
        headless: bool = False,
        on_result_callback: Optional[Callable[[bool], None]] = None,
        viewer_canvas_ref: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize detection pipeline.

        Args:
            box_cfg: Box detector configuration
            defect_cfg: Defect detector configuration
            stream_cfg: Stream configuration
            headless: If True, skip own GUI window
            on_result_callback: Optional callback when box exits with decision
            viewer_canvas_ref: If set (headless), build annotated canvas and set
                ref["canvas"] each frame for an external viewer
        """
        self.headless = headless
        self.on_result_callback = on_result_callback
        self.viewer_canvas_ref = viewer_canvas_ref
        self.produce_viewer_canvas = headless and (viewer_canvas_ref is not None)

        # 1. Setup Stream
        self.stream = CamStream(
            stream_cfg["source"],
            stream_cfg["width"],
            stream_cfg["height"],
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

        # 4. Visualizer: used for GUI, or for viewer canvas in headless
        self.LEFT_X = INFO_WIDTH + ROI_CENTER_OFFSET - ROI_WIDTH // 2
        self.RIGHT_X = INFO_WIDTH + ROI_CENTER_OFFSET + ROI_WIDTH // 2
        if not self.headless or self.produce_viewer_canvas:
            self.visualizer = Visualizer(
                stream_cfg["width"],
                stream_cfg["height"],
                INFO_WIDTH,
                ROI_WIDTH,
            )
        else:
            self.visualizer = None

        self.SKIP_DEFECT_FRAMES = 2
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0

    def run(self, stop_event: Optional[threading.Event] = None):
        """
        Run detection pipeline.

        Args:
            stop_event: Optional threading event to signal stop (headless mode)
        """
        need_canvas = not self.headless or self.produce_viewer_canvas
        if not self.headless:
            cv2.namedWindow("Box Inspection System", cv2.WINDOW_NORMAL)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                ret, frame = self.stream.read()
                if not ret:
                    break

                self.frame_count += 1
                self.update_fps()
                h, w = frame.shape[:2]

                if need_canvas:
                    canvas = cv2.copyMakeBorder(
                        frame, 0, 0, self.visualizer.info_width, 0,
                        cv2.BORDER_CONSTANT, value=(235, 235, 235),
                    )
                    self.visualizer.draw_layout(canvas)
                    self.visualizer.draw_stats(canvas, self.state, self.fps)
                    roi_offset = self.visualizer.info_width
                else:
                    roi_offset = 0

                roi = frame[:, self.LEFT_X - roi_offset : self.RIGHT_X - roi_offset]

                # --- STAGE 1: Box Detection ---
                box_result = self.box_detector.detect(roi)
                boxes_roi = (
                    box_result.boxes.xyxy.cpu().numpy()
                    if box_result.boxes is not None
                    else np.zeros((0, 4))
                )

                # --- Track single box by IoU (smooth, stable identity) ---
                matched_box, detected = self._match_track(boxes_roi)
                label, color, final_code = "—", (128, 128, 128), "OK"
                defect_boxes: List[np.ndarray] = []

                if detected and matched_box is not None:
                    x1, y1, x2, y2 = matched_box[0], matched_box[1], matched_box[2], matched_box[3]
                    frame_x1 = int(x1) + (self.LEFT_X - roi_offset)
                    frame_x2 = int(x2) + (self.LEFT_X - roi_offset)
                    crop = frame[int(y1):int(y2), frame_x1:frame_x2]
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

                    if need_canvas:
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
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Box Processed: {final_decision} | Total: {self.state.total_count}"
                    )
                    if self.on_result_callback:
                        self.on_result_callback(is_defect)

                if self.produce_viewer_canvas and self.viewer_canvas_ref is not None:
                    self.viewer_canvas_ref["canvas"] = canvas.copy()

                if not self.headless:
                    cv2.imshow("Box Inspection System", canvas)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        finally:
            self.cleanup()

    def _match_track(self, boxes_roi: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Match current track to detections by IoU; update smoothed bbox.
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
                return self._current_track.copy(), True
            return None, False

        # No current track: start one with largest box (most likely the one to track)
        areas = (boxes_roi[:, 2] - boxes_roi[:, 0]) * (boxes_roi[:, 3] - boxes_roi[:, 1])
        best_idx = int(np.argmax(areas))
        new_box = boxes_roi[best_idx].astype(np.float64)
        self._current_track = smooth_bbox(None, new_box, self.bbox_smooth_alpha)
        return self._current_track.copy(), True

    def _check_defect_track(self, crop: np.ndarray) -> Tuple[bool, List]:
        """
        Run defect detection on the tracked box crop; cache and skip frames for smoothness.
        Returns (is_defect, defect_boxes).
        """
        run_this_frame = (
            self.frame_count % (self.SKIP_DEFECT_FRAMES + 1) == 0
        )
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
            self.state.set_last_defect_result(hole_detected, defect_boxes)
        return self.state.get_last_defect_result()

    def update_fps(self):
        if self.frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            if elapsed > 0:
                self.fps = 10 / elapsed
            self.last_time = current_time
    
    def cleanup(self):
        """Release resources."""
        self.stream.release()
        # Only destroy pipeline's own window; viewer thread has its own
        if not self.headless:
            cv2.destroyAllWindows()
