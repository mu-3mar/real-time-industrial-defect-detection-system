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
        """Draws a bounding box (corners only) and label."""
        x1, y1, x2, y2 = map(int, box)
        
        abs_x1 = x1 + self.roi_left
        abs_x2 = x2 + self.roi_left
        
        # Draw Fancy Corners
        line_len = min(abs_x2 - abs_x1, y2 - y1) // 4
        thickness = 2
        
        # Top-Left
        cv2.line(canvas, (abs_x1, y1), (abs_x1 + line_len, y1), color, thickness)
        cv2.line(canvas, (abs_x1, y1), (abs_x1, y1 + line_len), color, thickness)
        
        # Top-Right
        cv2.line(canvas, (abs_x2, y1), (abs_x2 - line_len, y1), color, thickness)
        cv2.line(canvas, (abs_x2, y1), (abs_x2, y1 + line_len), color, thickness)
        
        # Bottom-Left
        cv2.line(canvas, (abs_x1, y2), (abs_x1 + line_len, y2), color, thickness)
        cv2.line(canvas, (abs_x1, y2), (abs_x1, y2 - line_len), color, thickness)
        
        # Bottom-Right
        cv2.line(canvas, (abs_x2, y2), (abs_x2 - line_len, y2), color, thickness)
        cv2.line(canvas, (abs_x2, y2), (abs_x2, y2 - line_len), color, thickness)

        cv2.putText(canvas, label, (abs_x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_defects(self, canvas, box_origin, defects):
        """Draws specific defect areas."""
        box_x1, box_y1 = box_origin
        
        # box_x1 is relative to ROI, so we add ROI offset
        base_x = box_x1 + self.roi_left
        base_y = box_y1
        
        for d_box in defects:
            dx1, dy1, dx2, dy2 = map(int, d_box)
            
            abs_dx1 = base_x + dx1
            abs_dy1 = base_y + dy1
            abs_dx2 = base_x + dx2
            abs_dy2 = base_y + dy2
            
            # Semi-transparent red fill for defect
            overlay = canvas.copy()
            cv2.rectangle(overlay, (abs_dx1, abs_dy1), (abs_dx2, abs_dy2), (0, 0, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
            
            # Solid border
            cv2.rectangle(canvas, (abs_dx1, abs_dy1), (abs_dx2, abs_dy2), (0, 0, 255), 1)

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
