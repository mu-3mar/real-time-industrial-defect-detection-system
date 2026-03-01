import numpy as np


def _box_area(box):
    """Area of box [x1, y1, x2, y2]; 0 if invalid."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = max(0, float(x2) - float(x1))
    h = max(0, float(y2) - float(y1))
    return w * h


def box_iou(box_a, box_b):
    """
    IoU of two boxes, each [x1, y1, x2, y2].
    Returns value in [0, 1]. Uses numpy for clarity.
    """
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def smooth_bbox(prev_box, curr_box, alpha=0.6):
    """
    Exponential smoothing of bbox: result = alpha * curr + (1 - alpha) * prev.
    prev_box can be None (then returns curr_box).
    Boxes are [x1, y1, x2, y2].
    """
    if prev_box is None:
        return np.array(curr_box, dtype=np.float64)
    prev = np.asarray(prev_box, dtype=np.float64)
    curr = np.asarray(curr_box, dtype=np.float64)
    return alpha * curr + (1.0 - alpha) * prev
