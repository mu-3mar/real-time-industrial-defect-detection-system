# two_stage_cam_qr_tracking_no_smooth.py
"""
Two-stage real-time inference with QR-based tracking (no smoothing):
- ONNX INT8 models (carton + defect)
- Expanded crop to defect model
- Only process crops that contain a QR code (QR becomes product_id)
- Per-product (per-QR) tracking + vote-based final label with conservative rule:
    final = "defect" if max_defects > 0 else locked_status or last_vote or "ok"
- Each product_id logged ONCE to CSV when it leaves view
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import csv
import uuid

# ====== CONFIG defaults ======
DEFAULT_CARTON_MODEL = "fine-tuning/combine-fine-tuning/box-YOLO/runs/train/detect_boxs/weights/best_box_detector_int8.onnx"
DEFAULT_DEFECT_MODEL = "fine-tuning/combine-fine-tuning/defect-YOLO/runs/train/defect/weights/best_defect_detector_int8.onnx"
DEFAULT_CAMERA_INDEX = 0

CARTON_CONF = 0.65
DEFECT_CONF = 0.15

CSV_PATH = "products_log_qr.csv"
# ============================

# tracking params based on QR
MAX_DISAPPEAR = 12    # frames to wait before considering product left & logging
MIN_LOCK_FRAMES = 5   # frames needed to lock a label (optional)
# ============================

session_id = str(uuid.uuid4())
qr_detector = cv2.QRCodeDetector()

# tracks keyed by QR string/product_id
# each track: {
#   "box": [x1,y1,x2,y2],
#   "last_seen": frame_index,
#   "first_seen": frame_index,
#   "frames_seen": int,
#   "votes": ["defect"/"ok",...],
#   "max_defects": int,
#   "locked_status": str|None,
#   "logged": bool
# }
tracks = {}
logged_qrs = set()  # to avoid duplicate CSV entries per session

def init_csv():
    if not Path(CSV_PATH).exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "session_id",
                "product_id",
                "final_status",
                "max_defects",
                "first_frame",
                "last_frame",
                "frames_seen",
                "timestamp",
            ])

def clamp(v, a, b):
    return max(a, min(b, v))

def expand_box(box, img_w, img_h, expand_ratio=0.2, expand_pixels=0):
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if expand_pixels and expand_pixels > 0:
        pad_x = expand_pixels
        pad_y = expand_pixels
    else:
        pad_x = int(bw * expand_ratio)
        pad_y = int(bh * expand_ratio)
    ex1 = clamp(x1 - pad_x, 0, img_w - 1)
    ey1 = clamp(y1 - pad_y, 0, img_h - 1)
    ex2 = clamp(x2 + pad_x, 0, img_w - 1)
    ey2 = clamp(y2 + pad_y, 0, img_h - 1)
    return (ex1, ey1, ex2, ey2)

def draw_box(img, box, color, label=None, score=None, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        text = label
        if score is not None:
            text = f"{label} {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def highlight_box(img, box, color=(0, 0, 255), alpha=0.35):
    x1, y1, x2, y2 = [int(v) for v in box]
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def read_qr_from_crop(crop):
    try:
        data, points, _ = qr_detector.detectAndDecode(crop)
        if isinstance(data, str) and data.strip() != "":
            return data.strip()
    except Exception:
        pass
    return None

def log_product_and_remove(qr, info):
    """
    Log the product once per session, using the conservative rule:
      final_status = "defect" if max_defects > 0
                    else locked_status or last_vote or "ok"
    """
    if qr in logged_qrs:
        # already logged in this session -> just remove track
        if qr in tracks:
            del tracks[qr]
        return

    init_csv()
    # conservative rule
    if info.get("max_defects", 0) > 0:
        final_status = "defect"
    else:
        final_status = info.get("locked_status")
        if final_status is None:
            # use last vote if exists
            final_status = info["votes"][-1] if info.get("votes") else "ok"

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            session_id,
            qr,
            final_status,
            info.get("max_defects", 0),
            info.get("first_seen"),
            info.get("last_seen"),
            info.get("frames_seen", 0),
            datetime.now().isoformat(),
        ])

    logged_qrs.add(qr)
    if qr in tracks:
        del tracks[qr]

def process_frame(frame, carton_model, defect_model,
                  frame_index,
                  carton_conf=0.5, defect_conf=0.4,
                  expand_ratio=0.2, expand_pixels=0):
    """
    Process a frame:
    - detect cartons
    - crop expanded area, read QR
    - if QR found => run defect model, update track
    - if QR not found => skip box (no annotation, no tracking)
    - log product when it disappears (once per session) using conservative rule
    """
    img = frame
    h, w = img.shape[:2]
    annotated = img.copy()

    # detect cartons
    res_carton = carton_model(img, conf=carton_conf, verbose=False)[0]
    if not hasattr(res_carton, "boxes") or len(res_carton.boxes) == 0:
        # check disappeared tracks
        for qr, info in list(tracks.items()):
            if frame_index - info["last_seen"] > MAX_DISAPPEAR and not info.get("logged", False):
                log_product_and_remove(qr, info)
        return annotated

    boxes = res_carton.boxes.xyxy.cpu().numpy()
    scores = res_carton.boxes.conf.cpu().numpy()

    for box, cscore in zip(boxes, scores):
        if cscore < carton_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), w, h, expand_ratio, expand_pixels)
        crop = img[ey1:ey2, ex1:ex2].copy()
        if crop.size == 0:
            continue

        qr = read_qr_from_crop(crop)
        if qr is None:
            # skip boxes with no QR
            continue

        # create track if new
        if qr not in tracks:
            tracks[qr] = {
                "box": np.array([x1, y1, x2, y2], dtype=float),
                "first_seen": frame_index,
                "last_seen": frame_index,
                "frames_seen": 0,
                "votes": [],
                "max_defects": 0,
                "locked_status": None,
                "logged": False,
            }
        info = tracks[qr]

        # update box (no smoothing) and meta
        info["box"] = np.array([x1, y1, x2, y2], dtype=float)
        info["last_seen"] = frame_index
        info["frames_seen"] += 1

        # run defect model on crop
        res_def = defect_model(crop, conf=defect_conf, verbose=False)[0]
        defect_boxes_global = []
        defect_count = 0
        if hasattr(res_def, "boxes") and len(res_def.boxes) > 0:
            dboxes = res_def.boxes.xyxy.cpu().numpy()
            dscores = res_def.boxes.conf.cpu().numpy()
            for dbox, dscore in zip(dboxes, dscores):
                if dscore < defect_conf:
                    continue
                dx1, dy1, dx2, dy2 = [int(v) for v in dbox]
                gx1 = ex1 + dx1; gy1 = ey1 + dy1
                gx2 = ex1 + dx2; gy2 = ey1 + dy2
                defect_boxes_global.append(((gx1, gy1, gx2, gy2), float(dscore)))
                defect_count += 1

        status = "defect" if defect_count > 0 else "ok"
        info["votes"].append(status)
        if defect_count > info.get("max_defects", 0):
            info["max_defects"] = defect_count

        # lock status optionally
        if info.get("locked_status") is None and info["frames_seen"] >= MIN_LOCK_FRAMES:
            info["locked_status"] = "defect" if info["votes"].count("defect") > info["votes"].count("ok") else "ok"

        # draw carton box + label (using latest box no smoothing)
        disp_box = info["box"].astype(int)
        display_status = info.get("locked_status") if info.get("locked_status") is not None else status
        color = (0,0,255) if display_status == "defect" else (0,255,0)
        draw_box(annotated, disp_box, color, label=f"{qr} {display_status}", score=float(cscore))

        # highlight defect boxes
        for (gx1, gy1, gx2, gy2), dscore in defect_boxes_global:
            highlight_box(annotated, (gx1, gy1, gx2, gy2), color=(0,0,255), alpha=0.35)

    # after processing: log tracks that disappeared
    for qr, info in list(tracks.items()):
        if frame_index - info["last_seen"] > MAX_DISAPPEAR and not info.get("logged", False):
            info["logged"] = True
            log_product_and_remove(qr, info)

    return annotated

def main(args):
    print("[INFO] Loading carton model:", args.carton_model)
    carton_model = YOLO(str(args.carton_model))
    print("[INFO] Loading defect model:", args.defect_model)
    defect_model = YOLO(str(args.defect_model))

    init_csv()

    cap = cv2.VideoCapture(int(args.cam))
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera", args.cam)

    print("[INFO] Press 'q' to quit.")
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        annotated = process_frame(
            frame,
            carton_model,
            defect_model,
            frame_index=frame_index,
            carton_conf=args.carton_conf,
            defect_conf=args.defect_conf,
            expand_ratio=args.expand_ratio,
            expand_pixels=args.expand_pixels,
        )

        cv2.imshow("Two-stage QR Tracking (ONNX, no-smooth)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--carton-model", default=DEFAULT_CARTON_MODEL)
    parser.add_argument("--defect-model", default=DEFAULT_DEFECT_MODEL)
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--carton-conf", type=float, default=CARTON_CONF)
    parser.add_argument("--defect-conf", type=float, default=DEFECT_CONF)
    parser.add_argument("--expand-ratio", type=float, default=0.2)
    parser.add_argument("--expand-pixels", type=int, default=0)
    args = parser.parse_args()
    main(args)
