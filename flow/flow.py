import cv2
import uuid
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import requests

# ======================================================
#                   CONFIGURATION
# ======================================================

DEFAULT_CARTON_MODEL = (
    "fine-tuning/combine-fine-tuning/box-YOLO/runs/train/detect_boxs/weights/"
    "best_box_detector_int8.onnx"
)
DEFAULT_DEFECT_MODEL = (
    "fine-tuning/combine-fine-tuning/defect-YOLO/runs/train/defect/weights/"
    "best_defect_detector_int8.onnx"
)

DEFAULT_CAMERA_INDEX = 0

CARTON_CONF = 0.5
DEFECT_CONF = 0.25

MAX_DISAPPEAR = 12
EXPAND_RATIO = 0.1  # small expand

# --- API CONFIG ---
API_URL = "https://chainly.azurewebsites.net/api/ProductionLines/sessions"
PRODUCTION_LINE_ID = 1
COMPANY_ID = 90

qr_detector = cv2.QRCodeDetector()
session_id = str(uuid.uuid4())

# ======================================================
#                 FINAL STATUS LOGIC
# ======================================================

def compute_final_status_for_db(info: dict) -> str:
    """
    Lightweight but more robust than:
      - first defect = defect
      - pure majority

    Uses:
      frames_seen, defect_frames, max_defects
    """
    frames = info.get("frames_seen", 0)
    df = info.get("defect_frames", 0)
    max_defects = info.get("max_defects", 0)

    # Very short track: 1-3 frames
    if frames <= 3:
        return "defect" if df > 0 else "ok"

    if frames == 0:
        return "ok"

    ratio = df / frames

    # If a significant portion of frames had defects -> defect
    if ratio >= 0.3:
        return "defect"

    # If never saw a defect -> ok
    if df == 0:
        return "ok"

    # If in any frame we saw multiple defects -> defect
    if max_defects >= 2:
        return "defect"

    # Small noise: treat as ok
    return "ok"

# ======================================================
#                  SEND TO EXTERNAL API
# ======================================================

def send_product_to_api(product_id: str, session_id: str, info: dict, final_status: str):
    """
    Send product info to external API instead of saving to DB.
    Fields to send:

        product_id
        session_id
        status          (note: NOT 'final_status' key)
        max_defects
        timestamp
        productionline_id (always 1)
        companyId         (always 90)
    """
    payload = {
        "product_id": product_id,
        "session_id": session_id,
        "status": final_status,
        "max_defects": int(info.get("max_defects", 0)),
        "timestamp": datetime.now().isoformat(),
        "productionline_id": PRODUCTION_LINE_ID,
        "companyId": COMPANY_ID,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code in (200, 201):
            print(f"[API] OK for product {product_id} -> {final_status}")
        else:
            print(f"[API] Error {response.status_code} for {product_id}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[API] Connection error for {product_id}: {e}")

# ======================================================
#                  QR-BASED TRACKING
# ======================================================

tracks = {}
# tracks[qr] = {
#   "box": np.array([x1,y1,x2,y2]),
#   "first_seen": int,
#   "last_seen": int,
#   "frames_seen": int,
#   "defect_frames": int,
#   "max_defects": int,
# }

def clamp(v, a, b):
    return max(a, min(b, v))

def expand_box(box, img_w, img_h, expand_ratio=EXPAND_RATIO):
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


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(10, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

def highlight_box(img, box, alpha=0.35):
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def read_qr(crop):
    try:
        data, pts, _ = qr_detector.detectAndDecode(crop)
        return data.strip() if data and data.strip() else None
    except Exception:
        return None

# ======================================================
#                  PROCESS FRAME
# ======================================================

def process_frame(frame, carton_model, defect_model, frame_index):
    global tracks

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Detect cartons
    res_carton = carton_model(frame, conf=CARTON_CONF, verbose=False)[0]

    # If no cartons: check disappeared products
    if len(res_carton.boxes) == 0:
        for qr, info in list(tracks.items()):
            if frame_index - info["last_seen"] > MAX_DISAPPEAR:
                final_status = compute_final_status_for_db(info)
                send_product_to_api(qr, session_id, info, final_status)
                del tracks[qr]
        return annotated

    boxes = res_carton.boxes.xyxy.cpu().numpy()
    scores = res_carton.boxes.conf.cpu().numpy()

    for box, score in zip(boxes, scores):
        if float(score) < CARTON_CONF:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), w, h)
        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue

        # ------------------ DEFECT FIRST ------------------
        res_def = defect_model(crop, conf=DEFECT_CONF, verbose=False)[0]
        defect_count = 0
        defect_boxes_global = []

        if hasattr(res_def, "boxes") and len(res_def.boxes) > 0:
            dboxes = res_def.boxes.xyxy.cpu().numpy()
            dscores = res_def.boxes.conf.cpu().numpy()
            for dbox, dscore in zip(dboxes, dscores):
                if float(dscore) < DEFECT_CONF:
                    continue
                dx1, dy1, dx2, dy2 = [int(v) for v in dbox]
                gx1 = ex1 + dx1
                gy1 = ey1 + dy1
                gx2 = ex1 + dx2
                gy2 = ey1 + dy2
                defect_boxes_global.append((gx1, gy1, gx2, gy2))
                defect_count += 1

        status_now = "defect" if defect_count > 0 else "ok"

        # ------------------ QR ------------------
        qr = read_qr(crop)
        if qr is None:
            # No QR: no API (no product_id), no tracking
            # (You can draw generic box if you want)
            continue

        # ------------------ TRACK UPDATE ------------------
        if qr not in tracks:
            tracks[qr] = {
                "box": np.array([x1, y1, x2, y2], dtype=float),
                "first_seen": frame_index,
                "last_seen": frame_index,
                "frames_seen": 1,
                "defect_frames": 1 if status_now == "defect" else 0,
                "max_defects": defect_count,
            }
        else:
            info = tracks[qr]
            info["box"] = np.array([x1, y1, x2, y2], dtype=float)
            info["last_seen"] = frame_index
            info["frames_seen"] += 1

            if status_now == "defect":
                info["defect_frames"] += 1
            if defect_count > info.get("max_defects", 0):
                info["max_defects"] = defect_count

        # ------------------ DISPLAY (ANNOTATION) ------------------
        info = tracks[qr]
        final_display = compute_final_status_for_db(info)
        color = (0, 0, 255) if final_display == "defect" else (0, 255, 0)
        draw_box(annotated, info["box"], color, f"{qr} {final_display}")

        if final_display == "defect":
            for (gx1, gy1, gx2, gy2) in defect_boxes_global:
                gx1 = max(0, min(gx1, w - 1))
                gy1 = max(0, min(gy1, h - 1))
                gx2 = max(0, min(gx2, w - 1))
                gy2 = max(0, min(gy2, h - 1))
                if gx2 > gx1 and gy2 > gy1:
                    highlight_box(annotated, (gx1, gy1, gx2, gy2))

    # ------------------ HANDLE DISAPPEAR ------------------
    for qr, info in list(tracks.items()):
        if frame_index - info["last_seen"] > MAX_DISAPPEAR:
            final_status = compute_final_status_for_db(info)
            send_product_to_api(qr, session_id, info, final_status)
            del tracks[qr]

    return annotated

# ======================================================
#                          MAIN
# ======================================================

def main(args):
    carton_model = YOLO(args.carton_model)
    defect_model = YOLO(args.defect_model)

    cap = cv2.VideoCapture(args.cam)
    frame_idx = 0

    print("[INFO] Session ID:", session_id)
    print("[INFO] Press Q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, carton_model, defect_model, frame_idx)

        cv2.imshow("Realtime QR Two-Stage (API Mode)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ======= FINAL FLUSH: send all remaining tracks even if not disappeared =======
    for qr, info in list(tracks.items()):
        final_status = compute_final_status_for_db(info)
        send_product_to_api(qr, session_id, info, final_status)
        del tracks[qr]

    cap.release()
    cv2.destroyAllWindows()

# RUN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--carton-model", default=DEFAULT_CARTON_MODEL)
    parser.add_argument("--defect-model", default=DEFAULT_DEFECT_MODEL)
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX)
    args = parser.parse_args()
    main(args)
