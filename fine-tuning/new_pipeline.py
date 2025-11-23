# improved_pipeline.py
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import csv
import uuid
from collections import deque
import time

# ====== CONFIG ======
DEFAULT_CARTON_MODEL = "fine-tuning/combine-fine-tuning/box-YOLO/runs/train/detect_boxs/weights/best_box_detector_int8.onnx"
DEFAULT_DEFECT_MODEL = "fine-tuning/combine-fine-tuning/defect-YOLO/runs/train/defect/weights/best_defect_detector_int8.onnx"
DEFAULT_CAMERA_INDEX = 0
CSV_PATH = "products_log_enhanced.csv"

# Confidence thresholds
CARTON_CONF = 0.65
DEFECT_CONF = 0.15

# Tracking params
MAX_DISAPPEAR = 15        # frames to wait
MIN_DECISION_FRAMES = 3   # minimum frames to make decision
OVERLAP_THRESHOLD = 0.5   # IoU threshold for tracking

# Box sizing
MIN_BOX_AREA = 5000      # minimum pixels
MAX_BOX_AREA = 500000    # maximum pixels
DYNAMIC_EXPAND = True     # use dynamic expansion based on box size

# Offline mode
OFFLINE_MODE = False      # set True to skip QR reading (use box ID instead)
# ===================

session_id = str(uuid.uuid4())
qr_detector = cv2.QRCodeDetector()

# Enhanced tracking structure
class ProductTracker:
    def __init__(self, max_history=10):
        self.tracks = {}
        self.next_box_id = 1
        self.max_history = max_history
        self.logged_ids = set()
        
    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def match_detection_to_track(self, detection_box):
        best_match = None
        best_iou = OVERLAP_THRESHOLD
        
        for track_id, track_info in self.tracks.items():
            if track_info.get("logged", False):
                continue
            iou = self.compute_iou(detection_box, track_info["box"])
            if iou > best_iou:
                best_iou = iou
                best_match = track_id
                
        return best_match
    
    def create_track(self, box, qr_code, frame_idx):
        if OFFLINE_MODE and qr_code is None:
            track_id = f"BOX_{self.next_box_id:04d}"
            self.next_box_id += 1
        else:
            track_id = qr_code if qr_code else f"BOX_{self.next_box_id:04d}"
            if not qr_code:
                self.next_box_id += 1
        
        self.tracks[track_id] = {
            "box": np.array(box, dtype=float),
            "first_seen": frame_idx,
            "last_seen": frame_idx,
            "frames_seen": 1,
            "defect_history": deque(maxlen=self.max_history),  # sliding window
            "defect_count_history": deque(maxlen=self.max_history),
            "max_defects": 0,
            "final_status": None,
            "logged": False,
            "qr_code": qr_code,
        }
        return track_id
    
    def update_track(self, track_id, box, defect_count, frame_idx):
        """تحديث track موجود"""
        info = self.tracks[track_id]
        info["box"] = np.array(box, dtype=float)
        info["last_seen"] = frame_idx
        info["frames_seen"] += 1
        
        # Update history
        status = "defect" if defect_count > 0 else "ok"
        info["defect_history"].append(status)
        info["defect_count_history"].append(defect_count)
        
        if defect_count > info["max_defects"]:
            info["max_defects"] = defect_count
        
        # Decision logic with sliding window
        if info["final_status"] is None and info["frames_seen"] >= MIN_DECISION_FRAMES:
            defect_votes = sum(1 for s in info["defect_history"] if s == "defect")
            total_votes = len(info["defect_history"])
            
            if defect_votes / total_votes >= 0.4:
                info["final_status"] = "defect"
            elif info["frames_seen"] >= 10:
                info["final_status"] = "ok"
    
    def get_disappeared_tracks(self, current_frame):
        disappeared = []
        for track_id, info in self.tracks.items():
            if (current_frame - info["last_seen"] > MAX_DISAPPEAR and 
                not info.get("logged", False)):
                disappeared.append(track_id)
        return disappeared

tracker = ProductTracker()

def init_csv():
    """Initialize CSV with enhanced columns"""
    if not Path(CSV_PATH).exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "session_id",
                "product_id",
                "qr_code",
                "final_status",
                "max_defects",
                "avg_defects",
                "defect_frames_ratio",
                "first_frame",
                "last_frame",
                "frames_seen",
                "box_area_avg",
                "timestamp",
            ])

def compute_dynamic_expand(box, img_w, img_h):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h
    
    # Dynamic ratio based on size
    if box_area < 20000:
        ratio = 0.35  # small boxes need more context
    elif box_area < 50000:
        ratio = 0.25
    elif box_area < 100000:
        ratio = 0.20
    else:
        ratio = 0.15  # large boxes need less expansion
    
    pad_x = int(box_w * ratio)
    pad_y = int(box_h * ratio)
    
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(img_w, x2 + pad_x)
    ey2 = min(img_h, y2 + pad_y)
    
    return (ex1, ey1, ex2, ey2)

def read_qr_optimized(crop):
    if OFFLINE_MODE:
        return None
        
    try:
        # Convert to grayscale for faster QR detection
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
            
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        data, points, _ = qr_detector.detectAndDecode(gray)
        if isinstance(data, str) and data.strip():
            return data.strip()
    except Exception:
        pass
    return None

def log_product(track_id, info):
    if track_id in tracker.logged_ids:
        return
    
    init_csv()
    
    # Calculate statistics
    defect_counts = list(info.get("defect_count_history", []))
    avg_defects = np.mean(defect_counts) if defect_counts else 0
    defect_frames = sum(1 for s in info.get("defect_history", []) if s == "defect")
    total_frames = len(info.get("defect_history", []))
    defect_ratio = defect_frames / total_frames if total_frames > 0 else 0
    
    # Conservative final decision
    if info.get("max_defects", 0) > 0:
        final_status = "defect"
    else:
        final_status = info.get("final_status", "ok")
    
    # Calculate average box area
    box = info.get("box", [0, 0, 0, 0])
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            session_id,
            track_id,
            info.get("qr_code", "N/A"),
            final_status,
            info.get("max_defects", 0),
            f"{avg_defects:.2f}",
            f"{defect_ratio:.2%}",
            info.get("first_seen", 0),
            info.get("last_seen", 0),
            info.get("frames_seen", 0),
            int(box_area),
            datetime.now().isoformat(),
        ])
    
    tracker.logged_ids.add(track_id)
    info["logged"] = True

def process_frame_enhanced(frame, carton_model, defect_model, frame_idx):
    """
    Pipeline محسّن:
    1. Detect boxes (filter by size)
    2. Match to existing tracks (IoU)
    3. For new boxes: try QR (fast), then defect
    4. Update tracks with sliding window
    """
    img = frame
    h, w = img.shape[:2]
    annotated = img.copy()
    
    # Stage 1: Detect cartons
    res_carton = carton_model(img, conf=CARTON_CONF, verbose=False)[0]
    
    if not hasattr(res_carton, "boxes") or len(res_carton.boxes) == 0:
        # Check for disappeared tracks
        for track_id in tracker.get_disappeared_tracks(frame_idx):
            log_product(track_id, tracker.tracks[track_id])
        return annotated
    
    boxes = res_carton.boxes.xyxy.cpu().numpy()
    scores = res_carton.boxes.conf.cpu().numpy()
    
    # Filter by size
    valid_detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        area = (x2 - x1) * (y2 - y1)
        if MIN_BOX_AREA <= area <= MAX_BOX_AREA:
            valid_detections.append((box, score))
    
    # Stage 2: Match detections to tracks
    matched_tracks = set()
    new_detections = []
    
    for box, score in valid_detections:
        track_id = tracker.match_detection_to_track(box)
        
        if track_id:
            matched_tracks.add(track_id)
            # Process existing track
            ex1, ey1, ex2, ey2 = compute_dynamic_expand(box, w, h)
            crop = img[ey1:ey2, ex1:ex2].copy()
            
            if crop.size > 0:
                # Defect detection
                res_def = defect_model(crop, conf=DEFECT_CONF, verbose=False)[0]
                defect_count = 0
                defect_boxes = []
                
                if hasattr(res_def, "boxes") and len(res_def.boxes) > 0:
                    dboxes = res_def.boxes.xyxy.cpu().numpy()
                    dscores = res_def.boxes.conf.cpu().numpy()
                    
                    for dbox, dscore in zip(dboxes, dscores):
                        if dscore >= DEFECT_CONF:
                            dx1, dy1, dx2, dy2 = [int(v) for v in dbox]
                            gx1 = ex1 + dx1; gy1 = ey1 + dy1
                            gx2 = ex1 + dx2; gy2 = ey1 + dy2
                            defect_boxes.append((gx1, gy1, gx2, gy2))
                            defect_count += 1
                
                tracker.update_track(track_id, box, defect_count, frame_idx)
                
                # Draw
                info = tracker.tracks[track_id]
                status = info.get("final_status") or ("defect" if defect_count > 0 else "ok")
                color = (0, 0, 255) if status == "defect" else (0, 255, 0)
                
                draw_box(annotated, box, color, 
                        label=f"{track_id} {status} [{info['frames_seen']}f]",
                        score=float(score))
                
                for dbox in defect_boxes:
                    highlight_box(annotated, dbox, (0, 0, 255), 0.3)
        else:
            new_detections.append((box, score))
    
    # Stage 3: Process new detections
    for box, score in new_detections:
        ex1, ey1, ex2, ey2 = compute_dynamic_expand(box, w, h)
        crop = img[ey1:ey2, ex1:ex2].copy()
        
        if crop.size == 0:
            continue
        
        # Quick QR check (with timeout simulation via smaller crop)
        qr_code = read_qr_optimized(crop)
        
        # Create track
        track_id = tracker.create_track(box, qr_code, frame_idx)
        
        # Initial defect check
        res_def = defect_model(crop, conf=DEFECT_CONF, verbose=False)[0]
        defect_count = 0
        defect_boxes = []
        
        if hasattr(res_def, "boxes") and len(res_def.boxes) > 0:
            dboxes = res_def.boxes.xyxy.cpu().numpy()
            dscores = res_def.boxes.conf.cpu().numpy()
            
            for dbox, dscore in zip(dboxes, dscores):
                if dscore >= DEFECT_CONF:
                    dx1, dy1, dx2, dy2 = [int(v) for v in dbox]
                    gx1 = ex1 + dx1; gy1 = ey1 + dy1
                    gx2 = ex1 + dx2; gy2 = ey1 + dy2
                    defect_boxes.append((gx1, gy1, gx2, gy2))
                    defect_count += 1
        
        tracker.tracks[track_id]["defect_history"].append("defect" if defect_count > 0 else "ok")
        tracker.tracks[track_id]["defect_count_history"].append(defect_count)
        tracker.tracks[track_id]["max_defects"] = defect_count
        
        # Draw
        status = "defect" if defect_count > 0 else "ok"
        color = (0, 0, 255) if status == "defect" else (0, 255, 0)
        draw_box(annotated, box, color, label=f"{track_id} {status} [NEW]", score=float(score))
        
        for dbox in defect_boxes:
            highlight_box(annotated, dbox, (0, 0, 255), 0.3)
    
    # Stage 4: Log disappeared
    for track_id in tracker.get_disappeared_tracks(frame_idx):
        log_product(track_id, tracker.tracks[track_id])
    
    return annotated

def draw_box(img, box, color, label=None, score=None, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        text = f"{label} {score:.2f}" if score else label
        cv2.putText(img, text, (x1, max(y1 - 6, 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def highlight_box(img, box, color=(0, 0, 255), alpha=0.35):
    x1, y1, x2, y2 = [int(v) for v in box]
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def main(args):
    global OFFLINE_MODE, CARTON_CONF, DEFECT_CONF
    
    OFFLINE_MODE = args.offline
    CARTON_CONF = args.carton_conf
    DEFECT_CONF = args.defect_conf
    
    print(f"[INFO] Loading models...")
    print(f"  Carton: {args.carton_model}")
    print(f"  Defect: {args.defect_model}")
    print(f"[INFO] Offline mode: {OFFLINE_MODE}")
    
    carton_model = YOLO(str(args.carton_model))
    defect_model = YOLO(str(args.defect_model))
    
    init_csv()
    
    cap = cv2.VideoCapture(int(args.cam))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam}")
    
    print("[INFO] Press 'q' to quit, 's' to save frame")
    frame_idx = 0
    fps_history = deque(maxlen=30)
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        annotated = process_frame_enhanced(
            frame, carton_model, defect_model, frame_idx
        )
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
        avg_fps = np.mean(fps_history)
        
        # Add FPS and info overlay
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {frame_idx}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Active tracks: {len([t for t in tracker.tracks.values() if not t.get('logged')])}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced QC System", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"frame_{frame_idx}.jpg", annotated)
            print(f"[INFO] Saved frame_{frame_idx}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session complete. Logged {len(tracker.logged_ids)} products")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced QC System")
    parser.add_argument("--carton-model", default=DEFAULT_CARTON_MODEL)
    parser.add_argument("--defect-model", default=DEFAULT_DEFECT_MODEL)
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--carton-conf", type=float, default=CARTON_CONF)
    parser.add_argument("--defect-conf", type=float, default=DEFECT_CONF)
    parser.add_argument("--offline", action="store_true", 
                       help="Skip QR reading, use auto box IDs")
    args = parser.parse_args()
    main(args)