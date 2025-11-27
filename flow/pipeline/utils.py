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
