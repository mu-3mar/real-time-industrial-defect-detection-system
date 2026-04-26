import cv2
from datetime import datetime


def _rect_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """Intersection area of two axis-aligned rectangles. Coords are (x1, y1, x2, y2)."""
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


class Visualizer:
    def __init__(self, width, height, info_width, roi_width):
        self.width = width
        self.height = height
        self.info_width = info_width
        self.roi_left = info_width + 420 - roi_width // 2  # 420 is center offset from original code
        self.roi_right = info_width + 420 + roi_width // 2
        
    def draw_layout(self, canvas):
        """Draws static UI elements (background only)."""
        # Guide lines intentionally disabled.
        return

    def draw_box(self, canvas, box, label, color):
        """Draws a bounding box (corners only) and label."""
        x1, y1, x2, y2 = map(int, box)
        
        abs_x1 = x1 + self.info_width
        abs_x2 = x2 + self.info_width
        
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

    def draw_defects(
        self,
        canvas,
        box_origin,
        defects,
        is_early_phase=True,
        visibility_threshold=0.2,
    ):
        """
        Draws defect areas in box-relative coordinates, with visibility filtering.
        Only draws defects if in early detection phase.

        - Converts each defect to frame/canvas coordinates using current box position.
        - Skips defects fully outside the frame.
        - Skips defects whose visible fraction (intersection with frame / defect area)
          is below visibility_threshold (default 0.2 = 20%). Use 0 to draw any
          partially visible defect.
        - If not in early_phase, skips drawing entirely (but keeps tracking).

        Does not affect tracking or defect state; only rendering is filtered.
        """
        # Only draw defects during early detection phase
        if not is_early_phase:
            return

        box_x1, box_y1 = box_origin
        base_x = box_x1 + self.info_width
        base_y = box_y1
        h, w = canvas.shape[:2]
        # Frame = visible camera image on canvas (right of info panel)
        frame_x1, frame_y1 = self.info_width, 0
        frame_x2, frame_y2 = w, h

        for d_box in defects:
            dx1, dy1, dx2, dy2 = map(int, d_box)
            abs_dx1 = base_x + dx1
            abs_dy1 = base_y + dy1
            abs_dx2 = base_x + dx2
            abs_dy2 = base_y + dy2

            # Fully outside frame -> do not draw
            if (
                abs_dx2 <= frame_x1
                or abs_dx1 >= frame_x2
                or abs_dy2 <= frame_y1
                or abs_dy1 >= frame_y2
            ):
                continue

            # Visibility ratio: intersection area / defect area
            defect_area = (abs_dx2 - abs_dx1) * (abs_dy2 - abs_dy1)
            if defect_area <= 0:
                continue
            inter_area = _rect_intersection_area(
                abs_dx1, abs_dy1, abs_dx2, abs_dy2,
                frame_x1, frame_y1, frame_x2, frame_y2,
            )
            if inter_area / defect_area < visibility_threshold:
                continue

            overlay = canvas.copy()
            cv2.rectangle(overlay, (abs_dx1, abs_dy1), (abs_dx2, abs_dy2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
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
