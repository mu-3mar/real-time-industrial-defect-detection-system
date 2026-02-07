"""
Single-track state: one box in the ROI at a time, with smooth defect voting,
defect lock (once defective, stays red), and accumulated defect annotations.
"""

from collections import deque
from typing import List, Tuple

from utils.geometry import box_iou


class AppState:
    """
    Tracks one logical box through the ROI.
    - Defect lock: once defect votes reach threshold, box is permanently DEFECT (red) for this track.
    - Accumulated defect boxes: stored in box-relative coords; at render time converted using current box position so they move with the box. IoU dedup in box-relative space.
    - Entry/exit and voting as before.
    """

    def __init__(self, stability_config: dict):
        self.min_frames = stability_config.get("min_frames", 4)
        self.max_missed = stability_config.get("max_missed", 6)
        self.vote_window = stability_config.get("vote_window", 9)
        self.vote_threshold = stability_config.get("vote_threshold", 5)

        self._vote_history: deque = deque(maxlen=self.vote_window)
        self._last_defect_result = (False, [])
        # Once True, this track is permanently DEFECT (bbox stays red)
        self._defect_locked = False
        # Defect boxes in box-relative coords (dx1, dy1, dx2, dy2); origin = box top-left
        self._accumulated_defect_boxes: List[Tuple[float, float, float, float]] = []
        self._defect_dedup_iou = 0.35  # treat as same defect if IoU above this (in box-relative space)

        self.frames_inside = 0
        self.missed_frames = 0
        self.inside = False
        self.final_decision = None  # "DEFECT" | "OK"

        self.total_count = 0
        self.defect_count = 0
        self.ok_count = 0

    def update_history(self, is_defect: bool) -> None:
        """Append one defect vote for the current track."""
        self._vote_history.append(is_defect)

    def _lock_defect(self) -> None:
        """Permanently mark this track as defective (bbox stays red until exit)."""
        self._defect_locked = True

    def get_status(self) -> tuple:
        """
        Returns (label_str, (B,G,R), decision_code).
        If defect_locked, always DEFECT/red. Else use vote threshold.
        """
        if self._defect_locked:
            self.final_decision = "DEFECT"
            return "Defect Box", (0, 0, 255), "DEFECT"

        n = len(self._vote_history)
        if n == 0:
            self.final_decision = "OK"
            return "Tracking...", (128, 128, 128), "OK"
        votes = sum(1 for v in self._vote_history if v)
        if votes >= self.vote_threshold:
            self._lock_defect()
            self.final_decision = "DEFECT"
            return "Defect Box", (0, 0, 255), "DEFECT"
        self.final_decision = "OK"
        return "Non-Defect Box", (0, 170, 0), "OK"

    def add_defect_boxes_relative(
        self,
        boxes_relative: List[Tuple[float, float, float, float]],
        iou_threshold: float = None,
    ) -> None:
        """
        Add defect boxes in box-relative coordinates (origin = box top-left).
        IoU deduplication is applied in box-relative space; only new non-overlapping
        defects are added. At render time these are converted to frame coords using
        current box position so they move with the box.
        """
        th = iou_threshold if iou_threshold is not None else self._defect_dedup_iou
        for b in boxes_relative:
            if len(b) != 4:
                continue
            b_tuple = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            overlaps = False
            for existing in self._accumulated_defect_boxes:
                if box_iou(b_tuple, existing) >= th:
                    overlaps = True
                    break
            if not overlaps:
                self._accumulated_defect_boxes.append(b_tuple)

    def get_accumulated_defect_boxes(self) -> List[Tuple[float, float, float, float]]:
        """Return list of defect boxes in box-relative coords (dx1, dy1, dx2, dy2)."""
        return list(self._accumulated_defect_boxes)

    def process_entry_exit(self, detected: bool) -> tuple:
        """
        Call each frame: detected = we matched the current track this frame.
        Returns (just_exited: bool, final_decision: str | None).
        """
        just_exited = False
        decision = None

        if detected:
            self.frames_inside += 1
            self.missed_frames = 0
            if self.frames_inside >= self.min_frames:
                self.inside = True
        else:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed and self.inside:
                # Use current vote-based decision before reset
                _label, _color, decision = self.get_status()
                self.frame_exit()
                just_exited = True

        return just_exited, decision

    def frame_exit(self) -> None:
        """On exit: update counters and reset track state."""
        self.total_count += 1
        if self.final_decision == "DEFECT":
            self.defect_count += 1
        else:
            self.ok_count += 1

        self.inside = False
        self.frames_inside = 0
        self.final_decision = None
        self._defect_locked = False
        self._accumulated_defect_boxes.clear()
        self._vote_history.clear()
        self._last_defect_result = (False, [])

    def set_last_defect_result(self, is_defect: bool, defect_boxes: list) -> None:
        """Store last defect run for current track (used by pipeline cache)."""
        self._last_defect_result = (is_defect, defect_boxes)

    def get_last_defect_result(self) -> tuple:
        """(is_defect, defect_boxes)."""
        return self._last_defect_result
