import cv2
import sqlite3
import uuid
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ======================================================
#                   CONFIGURATION
# ======================================================

DEFAULT_CARTON_MODEL = "fine-tuning/combine-fine-tuning/box-YOLO/runs/train/detect_boxs/weights/best_box_detector_int8.onnx"
DEFAULT_DEFECT_MODEL = "fine-tuning/combine-fine-tuning/defect-YOLO/runs/train/defect/weights/best_defect_detector_int8.onnx"

DEFAULT_CAMERA_INDEX = 0

CARTON_CONF = 0.5
DEFECT_CONF = 0.8

MAX_DISAPPEAR = 12
EXPAND_RATIO = 0.1     # small expand

DB_PATH = "fine-tuning/combine-fine-tuning/products.db"
SNAPSHOT_DIR = "fine-tuning/combine-fine-tuning/defect_snapshots"

qr_detector = cv2.QRCodeDetector()
session_id = str(uuid.uuid4())

# ======================================================
#                   DATABASE SETUP
# ======================================================

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            session_id TEXT,
            final_status TEXT,
            max_defects INTEGER,
            first_frame INTEGER,
            last_frame INTEGER,
            frames_seen INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def compute_final_status_for_db(info):
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


def save_or_update_product(product_id, session_id, info, final_status):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT product_id FROM products WHERE product_id = ?", (product_id,))
    exists = cur.fetchone()

    if exists:
        cur.execute("""
            UPDATE products SET 
                final_status=?,
                max_defects=?,
                first_frame=?,
                last_frame=?,
                frames_seen=?,
                timestamp=?
            WHERE product_id=?
        """, (
            final_status,
            info.get("max_defects", 0),
            info.get("first_seen"),
            info.get("last_seen"),
            info.get("frames_seen", 0),
            datetime.now().isoformat(),
            product_id
        ))
    else:
        cur.execute("""
            INSERT INTO products (
                product_id, session_id, final_status, max_defects, 
                first_frame, last_frame, frames_seen, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product_id,
            session_id,
            final_status,
            info.get("max_defects", 0),
            info.get("first_seen"),
            info.get("last_seen"),
            info.get("frames_seen", 0),
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()

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
#   "snapshot_frame": np.ndarray or None,
#   "snapshot_box": (x1,y1,x2,y2) or None,
#   "snapshot_defect_boxes": list of (x1,y1,x2,y2),
#   "snapshot_defect_count": int   # best defect_count seen in snapshot frame
# }

def clamp(v,a,b):
    return max(a,min(b,v))

def expand_box(box, img_w, img_h, expand_ratio=EXPAND_RATIO):
    x1,y1,x2,y2 = map(int, box)
    bw = max(1, x2-x1)
    bh = max(1, y2-y1)

    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)

    ex1 = clamp(x1-pad_x, 0, img_w-1)
    ey1 = clamp(y1-pad_y, 0, img_h-1)
    ex2 = clamp(x2+pad_x, 0, img_w-1)
    ey2 = clamp(y2+pad_y, 0, img_h-1)

    return ex1,ey1,ex2,ey2


def draw_box(img, box, color, label):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(10,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def highlight_box(img, box, alpha=0.35):
    x1,y1,x2,y2 = map(int, box)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def read_qr(crop):
    try:
        data, pts, _ = qr_detector.detectAndDecode(crop)
        return data.strip() if data and data.strip() else None
    except:
        return None

# ======================================================
#              SNAPSHOT SAVE (DEFECT ONLY)
# ======================================================

def save_defect_snapshot(product_id, info, final_status):
    """
    Save exactly ONE annotated image per product_id (only for defect).
    Uses snapshot_frame + snapshot_box + snapshot_defect_boxes.
    """
    frame = info.get("snapshot_frame", None)
    box = info.get("snapshot_box", None)
    defect_boxes = info.get("snapshot_defect_boxes", [])

    if frame is None or box is None:
        return  # nothing to save

    out_dir = Path(SNAPSHOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = frame.copy()

    # carton box
    color = (0,0,255) if final_status == "defect" else (0,255,0)
    label = f"{product_id} {final_status}"
    draw_box(img, box, color, label)

    # defect boxes (only if defect)
    if final_status == "defect":
        for db in defect_boxes:
            highlight_box(img, db)

    out_path = out_dir / f"{product_id}.jpg"
    cv2.imwrite(str(out_path), img)

# ======================================================
#                  PROCESS FRAME
# ======================================================

def process_frame(frame, carton_model, defect_model, frame_index):
    global tracks

    h, w = frame.shape[:2]
    annotated = frame.copy()

    res_carton = carton_model(frame, conf=CARTON_CONF, verbose=False)[0]

    if len(res_carton.boxes) == 0:
        # check disappeared products
        for qr, info in list(tracks.items()):
            if frame_index - info["last_seen"] > MAX_DISAPPEAR:
                final_status = compute_final_status_for_db(info)
                save_or_update_product(qr, session_id, info, final_status)
                if final_status == "defect":
                    save_defect_snapshot(qr, info, final_status)
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
            # No QR: no DB, no snapshot (you can draw if you want)
            continue

        # ------------------ TRACK UPDATE ------------------
        if qr not in tracks:
            tracks[qr] = {
                "box": np.array([x1,y1,x2,y2], dtype=float),
                "first_seen": frame_index,
                "last_seen": frame_index,
                "frames_seen": 1,
                "defect_frames": 1 if status_now == "defect" else 0,
                "max_defects": defect_count,
                "snapshot_frame": frame.copy() if defect_count > 0 else None,
                "snapshot_box": (x1,y1,x2,y2) if defect_count > 0 else None,
                "snapshot_defect_boxes": defect_boxes_global[:] if defect_count > 0 else [],
                "snapshot_defect_count": defect_count if defect_count > 0 else 0
            }
        else:
            info = tracks[qr]
            info["box"] = np.array([x1,y1,x2,y2], dtype=float)
            info["last_seen"] = frame_index
            info["frames_seen"] += 1

            if status_now == "defect":
                info["defect_frames"] += 1
            if defect_count > info.get("max_defects", 0):
                info["max_defects"] = defect_count

            # update snapshot only if this frame has more defects than previous snapshot
            if defect_count > 0 and defect_count > info.get("snapshot_defect_count", 0):
                info["snapshot_frame"] = frame.copy()
                info["snapshot_box"] = (x1,y1,x2,y2)
                info["snapshot_defect_boxes"] = defect_boxes_global[:]
                info["snapshot_defect_count"] = defect_count

        # ------------------ DISPLAY (OPTIONAL, SIMPLE) ------------------
        info = tracks[qr]
        final_display = compute_final_status_for_db(info)
        color = (0,0,255) if final_display == "defect" else (0,255,0)
        draw_box(annotated, info["box"], color, f"{qr} {final_display}")

        if final_display == "defect":
            for (gx1,gy1,gx2,gy2) in defect_boxes_global:
                gx1 = max(0, min(gx1, w-1)); gy1 = max(0, min(gy1, h-1))
                gx2 = max(0, min(gx2, w-1)); gy2 = max(0, min(gy2, h-1))
                if gx2 > gx1 and gy2 > gy1:
                    highlight_box(annotated, (gx1,gy1,gx2,gy2))

    # ------------------ HANDLE DISAPPEAR ------------------
    for qr, info in list(tracks.items()):
        if frame_index - info["last_seen"] > MAX_DISAPPEAR:
            final_status = compute_final_status_for_db(info)
            save_or_update_product(qr, session_id, info, final_status)
            if final_status == "defect":
                save_defect_snapshot(qr, info, final_status)
            del tracks[qr]

    return annotated

# ======================================================
#                          MAIN
# ======================================================

def main(args):
    init_database()

    carton_model = YOLO(args.carton_model)
    defect_model = YOLO(args.defect_model)

    cap = cv2.VideoCapture(args.cam)
    frame_idx = 0

    print("[INFO] Press Q to exit.")
    while True:
        ret,frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, carton_model, defect_model, frame_idx)

        cv2.imshow("Realtime QR Two-Stage (Lite + DB-focused)", annotated)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    # ======= FINAL FLUSH: save all remaining tracks even if not disappeared =======
    for qr, info in list(tracks.items()):
        final_status = compute_final_status_for_db(info)
        save_or_update_product(qr, session_id, info, final_status)
        if final_status == "defect":
            save_defect_snapshot(qr, info, final_status)
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
