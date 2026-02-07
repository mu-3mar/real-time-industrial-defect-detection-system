"""
Single-track state: one box in the ROI at a time, with smooth defect voting,
defect lock (once defective, stays red), and accumulated defect annotations.
Includes track grace period and recovery from small camera movements.
"""

from collections import deque
from typing import List, Tuple, Optional
import logging

from utils.geometry import box_iou

logger = logging.getLogger(__name__)


class AppState:
    """
    Tracks one logical box through the ROI.
    - Defect lock: once defect votes reach threshold, box is permanently DEFECT (red) for this track.
    - Accumulated defect boxes: stored in box-relative coords; at render time converted using current box position so they move with the box. IoU dedup in box-relative space.
    - Entry/exit and voting as before.
    - Track grace period: allows brief loss (e.g., from camera jitter) before declaring track as lost.
    - Recent track recovery: if detection reappears with high IoU with recently lost track, reuse it.
    """

    def __init__(self, stability_config: dict):
        self.min_frames = stability_config.get("min_frames", 4)
        self.max_missed = stability_config.get("max_missed", 6)
        self.vote_window = stability_config.get("vote_window", 9)
        self.vote_threshold = stability_config.get("vote_threshold", 5)
        self.early_detection_frames = stability_config.get("early_detection_frames", 3)
        
        # Track grace period: allow brief loss before declaring track as lost
        self.track_grace_frames = stability_config.get("track_grace_frames", 3)
        # How long to keep recently lost track for recovery attempt
        self.recent_track_max_age = stability_config.get("recent_track_max_age", 15)
        # IoU threshold for recovery (may be higher than matching threshold)
        self.recovery_iou_threshold = stability_config.get("recovery_iou_threshold", 0.4)

        self._vote_history: deque = deque(maxlen=self.vote_window)
        self._last_defect_result = (False, [])
        # Once True, this track is permanently DEFECT (bbox stays red)
        self._defect_locked = False
        # Track frames since defect lock for early detection phase visualization
        self._frames_since_lock = 0
        # Defect boxes in box-relative coords (dx1, dy1, dx2, dy2); origin = box top-left
        self._accumulated_defect_boxes: List[Tuple[float, float, float, float]] = []
        self._defect_dedup_iou = 0.35  # treat as same defect if IoU above this (in box-relative space)
        
        # Recent track for recovery: (bbox, frames_since_lost)
        self._recent_lost_track: Optional[Tuple[Tuple[float, float, float, float], int]] = None

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
        self._frames_since_lock = 0  # Reset counter when defect is locked

    def is_early_detection_phase(self) -> bool:
        """
        Returns True if in early detection phase (first N frames after defect lock).
        Only show defect annotations during this phase.
        """
        return self._defect_locked and self._frames_since_lock < self.early_detection_frames

    def increment_defect_lock_frame(self) -> None:
        """Increment frame counter for early detection phase tracking."""
        if self._defect_locked:
            self._frames_since_lock += 1

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
        
        Track grace period allows brief loss (e.g., from camera jitter).
        Recent track recovery: if detection reappears with high IoU with recently lost track,
        reuse it instead of counting as exit+entry.
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
            # Grace period: allow brief loss before declaring track as lost
            if self.missed_frames > self.max_missed + self.track_grace_frames and self.inside:
                # Use current vote-based decision before reset
                _label, _color, decision = self.get_status()
                self.frame_exit()
                just_exited = True

        return just_exited, decision
    
    def set_recent_lost_track(self, bbox: Tuple[float, float, float, float]) -> None:
        """
        Store the track bbox when it's lost, for potential recovery.
        Allows reuse if detection reappears with high IoU.
        """
        self._recent_lost_track = (bbox, 0)
        logger.debug("Stored recent lost track for recovery")
    
    def try_recover_recent_track(self, new_bbox: Tuple[float, float, float, float]) -> bool:
        """
        Check if new detection matches recently lost track by IoU.
        If yes, consider this a recovery (not a new box).
        Returns True if recovered, False if no match.
        """
        if self._recent_lost_track is None:
            return False
        
        stored_bbox, frames_since_lost = self._recent_lost_track
        
        # Increment age of stored track
        self._recent_lost_track = (stored_bbox, frames_since_lost + 1)
        
        # If track is too old, forget it
        if frames_since_lost >= self.recent_track_max_age:
            self._recent_lost_track = None
            return False
        
        # Check IoU with recently lost track
        iou = box_iou(new_bbox, stored_bbox)
        if iou >= self.recovery_iou_threshold:
            logger.info("Recovered lost track (IoU=%.3f), preventing duplicate count", iou)
            self._recent_lost_track = None
            return True
        
        return False
    
    def clear_recent_lost_track(self) -> None:
        """Clear recent track cache (e.g., when frame_exit is called)."""
        self._recent_lost_track = None

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
        self._frames_since_lock = 0
        self._accumulated_defect_boxes.clear()
        self._vote_history.clear()
        self._last_defect_result = (False, [])
        self._recent_lost_track = None  # Clear recovery candidate on exit

    def set_last_defect_result(self, is_defect: bool, defect_boxes: list) -> None:
        """Store last defect run for current track (used by pipeline cache)."""
        self._last_defect_result = (is_defect, defect_boxes)

    def tick_recent_lost_track(self) -> None:
        """
        Advance age of the recent lost track each frame. If it becomes too old,
        forget it to avoid spurious recovery attempts far in the past.
        """
        if self._recent_lost_track is None:
            return
        stored_bbox, age = self._recent_lost_track
        age += 1
        if age >= self.recent_track_max_age:
            logger.debug("Recent lost track expired after %d frames", age)
            self._recent_lost_track = None
        else:
            self._recent_lost_track = (stored_bbox, age)

    def get_last_defect_result(self) -> tuple:
        """(is_defect, defect_boxes)."""
        return self._last_defect_result
