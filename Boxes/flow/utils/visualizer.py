import cv2
from datetime import datetime

class Visualizer:
    def __init__(self, width, height, info_width, roi_width):
        self.width = width
        self.height = height
        self.info_width = info_width
        self.roi_left = info_width + 420 - roi_width // 2  # 420 is center offset from original code
        self.roi_right = info_width + 420 + roi_width // 2
        
    def draw_layout(self, canvas):
        """Draws static UI elements (ROI lines, background)."""
        h, w = canvas.shape[:2]
        # ROI Lines
        cv2.line(canvas, (self.roi_left, 0), (self.roi_left, h), (0, 0, 0), 2)
        cv2.line(canvas, (self.roi_right, 0), (self.roi_right, h), (0, 0, 0), 2)

    def draw_box(self, canvas, box, label, color):
        """Draws a bounding box and label."""
        x1, y1, x2, y2 = map(int, box)
        # Shift x coordinates because we are drawing on the full canvas (Info + Frame)
        # The detection passed in is physically on the ROI, but we map it to canvas if needed.
        # Actually, in the original code, x1 is relative to the ROI? 
        # No, in original: `x1, y1, x2, y2 = map(int, box)` comes from `box_detector.detect(roi)`.
        # So x1 is 0-indexed relative to ROI.
        
        abs_x1 = x1 + self.roi_left
        abs_x2 = x2 + self.roi_left
        
        cv2.rectangle(canvas, (abs_x1, y1), (abs_x2, y2), color, 2)
        cv2.putText(canvas, label, (abs_x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_stats(self, canvas, state, fps):
        """Draws the side panel stats."""
        px, py = 30, 70
        
        # TOTAL
        cv2.putText(canvas, "TOTAL", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
        py += 45
        cv2.putText(canvas, str(state.total_count), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

        # DEFECT / OK
        py += 60
        cv2.putText(canvas, f"DEFECT: {state.defect_count}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        py += 35
        cv2.putText(canvas, f"OK: {state.ok_count}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 0), 2)

        # Date
        py += 40
        cv2.putText(canvas, datetime.now().strftime("%d / %m / %Y"), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # FPS
        py += 50
        cv2.line(canvas, (px, py), (self.info_width - 30, py), (180, 180, 180), 1)
        py += 30
        cv2.putText(canvas, f"FPS: {fps:.1f}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
