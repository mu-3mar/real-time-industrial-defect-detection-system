import cv2
import numpy as np
from typing import Tuple

qr_detector = cv2.QRCodeDetector()


def clamp(v: int, a: int, b: int) -> int:
    """Clamp value v between a and b."""
    return max(a, min(b, v))


def expand_box(box: Tuple[int, int, int, int], img_w: int, img_h: int, expand_ratio: float) -> Tuple[int, int, int, int]:
    """Expand box by ratio, clamped to image bounds."""
    x1, y1, x2, y2 = map(int, box)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)
    ex1 = clamp(x1 - pad_x, 0, img_w - 1)
    ey1 = clamp(y1 - pad_y, 0, img_h - 1)
    ex2 = clamp(x2 + pad_x, 0, img_w - 1)
    ey2 = clamp(y2 + pad_y, 0, img_h - 1)
    return ex1, ey1, ex2, ey2


def draw_box(img: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], label: str) -> None:
    """Draw rectangle and label on image."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def highlight_box(img: np.ndarray, box: Tuple[int, int, int, int], alpha: float = 0.35) -> None:
    """Highlight region with red overlay."""
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def read_qr(crop: np.ndarray) -> str | None:
    """Detect and decode QR code from image crop."""
    try:
        data, pts, _ = qr_detector.detectAndDecode(crop)
        return data.strip() if data and data.strip() else None
    except Exception:
        return None


def highlight_defect_circle(img: np.ndarray, box: Tuple[int, int, int, int], limit_box: Tuple[int, int, int, int] = None, alpha: float = 0.4) -> None:
    """Highlight region with a red circle overlay, clipped to limit_box."""
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    radius = max(w, h) // 2

    # Define the clipping area
    lx1, ly1, lx2, ly2 = map(int, limit_box) if limit_box else (0, 0, img.shape[1], img.shape[0])
    lx1 = max(0, lx1)
    ly1 = max(0, ly1)
    lx2 = min(img.shape[1], lx2)
    ly2 = min(img.shape[0], ly2)

    if lx2 <= lx1 or ly2 <= ly1:
        return

    # Create masks
    # 1. Circle mask
    circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

    # 2. Limit mask (rectangle)
    rect_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(rect_mask, (lx1, ly1), (lx2, ly2), 255, -1)

    # 3. Intersection
    combined_mask = cv2.bitwise_and(circle_mask, rect_mask)

    # Apply Red Overlay
    indices = np.where(combined_mask > 0)
    if len(indices[0]) > 0:
        roi = img[indices]
        red = np.array([0, 0, 255], dtype=np.float32)
        blended = roi.astype(np.float32) * (1 - alpha) + red * alpha
        img[indices] = blended.astype(np.uint8)

    # Draw Outline (clipped)
    outline_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(outline_mask, (cx, cy), radius, 255, 2)
    combined_outline_mask = cv2.bitwise_and(outline_mask, rect_mask)
    
    outline_indices = np.where(combined_outline_mask > 0)
    if len(outline_indices[0]) > 0:
        img[outline_indices] = (255, 255, 255)
