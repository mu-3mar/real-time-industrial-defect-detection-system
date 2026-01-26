import cv2
import os

# Suppress Logs
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["ORT_LOGGING_LEVEL"] = "3"

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    pass

import yaml
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from detector.yolo_detector import YOLODetector

# ================== PATHS ==================
BASE = Path(__file__).resolve().parent
det_cfg = yaml.safe_load(open(BASE / "config/detector.yaml"))
info_cfg = yaml.safe_load(open(BASE / "config/info_detector.yaml"))
str_cfg = yaml.safe_load(open(BASE / "config/stream.yaml"))

# ================== UI CONFIG ==================
INFO_WIDTH = 300

ROI_WIDTH = 260
ROI_CENTER_OFFSET = 420

LEFT_X = INFO_WIDTH + ROI_CENTER_OFFSET - ROI_WIDTH // 2
RIGHT_X = INFO_WIDTH + ROI_CENTER_OFFSET + ROI_WIDTH // 2

MIN_FRAMES = 3      # confirmation frames
MAX_MISSED = 5      # grace frames for missing detection

WINDOW_NAME = "Bottle Inspection System"

# ================== OPTIMIZATION CONFIG ==================
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame (50% reduction)

# ================== VIDEO ==================
cap = cv2.VideoCapture(str_cfg["source"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, str_cfg["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, str_cfg["height"])

detector = YOLODetector(
    det_cfg["model_path"],
    det_cfg["conf_thres"],
    det_cfg["iou_thres"],
    det_cfg["device"]
)

info_detector = YOLODetector(
    info_cfg["model_path"],
    info_cfg["conf_thres"],
    info_cfg["iou_thres"],
    info_cfg["device"]
)

# ================== STATE ==================
frames_inside = 0
missed_frames = 0
inside = False
total_count = 0
ok_count = 0
defect_count = 0
session_cap = False
session_label = False
cap_frames_count = 0
label_frames_count = 0
MIN_CONFIRM_FRAMES = 3  # How many frames cap/label must be seen to counting it

frame_count = 0
last_detected = False
last_detected = False
last_boxes = []  # Store last detection results
last_info_boxes = [] # Store cap/label annotations (x1, y1, x2, y2, label)

# Performance tracking
frame_times = deque(maxlen=30)
last_time = time.time()

# ================== WINDOW (MAXIMIZED, NOT FULLSCREEN) ==================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_AUTOSIZE,
    cv2.WINDOW_NORMAL
)

# ================== LOOP ==================
while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_count += 1

    canvas = cv2.copyMakeBorder(
        frame, 0, 0, INFO_WIDTH, 0,
        cv2.BORDER_CONSTANT, value=(235, 235, 235)
    )

    roi = canvas[:, LEFT_X:RIGHT_X]
    
    # ========== OPTIMIZED DETECTION (Every N Frames) ==========
    detected = last_detected  # Use previous state as default
    current_boxes = last_boxes  # Use previous boxes by default
    
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        result = detector.detect(roi)
        detected = False
        detected = False
        current_boxes = []
        current_info_boxes = []
        cap_seen = False
        label_seen = False
        
        if result.boxes is not None and len(result.boxes) > 0:
            detected = True
            for b in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                current_boxes.append((x1, y1, x2, y2))
                
                # --- STAGE 2: INFO DETECTION ---
                # Crop the bottle from the ROI
                # Coordinates (x1,y1,x2,y2) are relative to ROI
                bottle_crop = roi[y1:y2, x1:x2]
                
                if bottle_crop.size > 0:
                    info_res = info_detector.detect(bottle_crop)
                    if info_res.boxes is not None:
                        for i, cls_id in enumerate(info_res.boxes.cls.cpu().numpy()):
                            cls_name = info_detector.model.names[int(cls_id)].lower()
                            if "cap" in cls_name:
                                cap_seen = True
                            if "label" in cls_name:
                                label_seen = True

                            # Save info box for drawing
                            # box coordinates are relative to crop, need to shift to canvas
                            bx1, by1, bx2, by2 = map(int, info_res.boxes.xyxy[i].cpu().numpy())
                            
                            # Info box relative to ROI:
                            #   crop x1, y1 are relative to ROI
                            #   so: ib_x = bx + crop_x
                            ib_x1 = bx1 + x1
                            ib_y1 = by1 + y1
                            ib_x2 = bx2 + x1
                            ib_y2 = by2 + y1
                            
                            current_info_boxes.append((ib_x1, ib_y1, ib_x2, ib_y2, cls_name))
        
        # --- Update Session Logic with Persistence ---
        if cap_seen:
            cap_frames_count += 1
            if cap_frames_count >= MIN_CONFIRM_FRAMES:
                session_cap = True
        
        if label_seen:
            label_frames_count += 1
            if label_frames_count >= MIN_CONFIRM_FRAMES:
                session_label = True

        last_detected = detected
        last_boxes = current_boxes
        last_info_boxes = current_info_boxes
    
    # ========== DRAW BOXES (from cache on skipped frames) ==========
    for x1, y1, x2, y2 in current_boxes:
        abs_x1 = x1 + LEFT_X
        abs_x2 = x2 + LEFT_X

        cv2.rectangle(
            canvas,
            (abs_x1, y1),
            (abs_x2, y2),
            (0, 170, 0),
            2
        )

        cv2.putText(
            canvas,
            "Bottle",
            (abs_x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 170, 0),
            2
        )
    
    # ========== DRAW INFO BOXES (Cap/Label) ==========
    current_info_boxes = last_info_boxes # Use cached info boxes
    for x1, y1, x2, y2, label_name in current_info_boxes:
        abs_x1 = x1 + LEFT_X
        abs_x2 = x2 + LEFT_X
        
        color = (255, 100, 0) # Blue-ish for info
        if "cap" in label_name:
            color = (0, 200, 200) # Yellow-ish for cap
        elif "label" in label_name:
             color = (200, 0, 200) # Purple-ish for label

        cv2.rectangle(canvas, (abs_x1, y1), (abs_x2, y2), color, 2)
        cv2.putText(canvas, label_name, (abs_x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ================== MIN + MAX LOGIC ==================
    if detected:
        frames_inside += 1
        missed_frames = 0
    else:
        missed_frames += 1
        if missed_frames > MAX_MISSED:
            if inside:
                # Bottle has exited, classify it
                if session_cap and session_label:
                    ok_count += 1
                else:
                    defect_count += 1
                
                # Reset session flags
                # Reset session flags
                session_cap = False
                session_label = False
                cap_frames_count = 0
                label_frames_count = 0

            frames_inside = 0
            inside = False

    if frames_inside >= MIN_FRAMES and not inside:
        total_count += 1
        inside = True

    # ================== ROI LINES ==================
    cv2.line(canvas, (LEFT_X, 0), (LEFT_X, h), (0, 180, 0), 2)
    cv2.line(canvas, (RIGHT_X, 0), (RIGHT_X, h), (0, 180, 0), 2)

    # ================== LEFT INFO PANEL ==================
    px = 30
    py = 70

    cv2.putText(canvas, "TOTAL COUNT", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)

    py += 55
    cv2.putText(canvas, str(total_count), (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 4)

    py += 40
    py += 40
    # Checkbox Logic
    # Cap Checkbox
    cap_color = (0, 200, 0) if session_cap else (200, 200, 200)
    cap_fill = -1 if session_cap else 2
    cv2.rectangle(canvas, (px, py), (px + 20, py + 20), cap_color, 2)
    if session_cap:
        cv2.rectangle(canvas, (px + 3, py + 3), (px + 17, py + 17), cap_color, -1)
    
    cv2.putText(canvas, "Cap", (px + 30, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    
    py += 35
    # Label Checkbox
    lbl_color = (0, 200, 0) if session_label else (200, 200, 200)
    cv2.rectangle(canvas, (px, py), (px + 20, py + 20), lbl_color, 2)
    if session_label:
        cv2.rectangle(canvas, (px + 3, py + 3), (px + 17, py + 17), lbl_color, -1)
    
    cv2.putText(canvas, "Label", (px + 30, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    py += 50
    # Percentages
    total_final = ok_count + defect_count
    ok_pct = 0.0
    def_pct = 0.0
    if total_final > 0:
        ok_pct = (ok_count / total_final) * 100
        def_pct = (defect_count / total_final) * 100
    
    cv2.putText(canvas, f"OK: {ok_pct:.1f}%", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
    py += 25
    cv2.putText(canvas, f"Defect: {def_pct:.1f}%", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    py += 90
    cv2.line(canvas, (px, py), (INFO_WIDTH - 30, py), (180, 180, 180), 2)

    py += 50
    cv2.putText(canvas, "DATE", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)

    py += 40
    cv2.putText(canvas,
                datetime.now().strftime("%d / %m / %Y"),
                (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # ================== PERFORMANCE STATS ==================
    py += 50
    cv2.line(canvas, (px, py), (INFO_WIDTH - 30, py), (180, 180, 180), 1)
    py += 30
    
    # Calculate FPS
    frame_time = time.time() - start_time
    frame_times.append(frame_time)
    avg_time = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    cv2.putText(canvas, f"FPS: {fps:.1f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    py += 20
    cv2.putText(canvas, f"Frame: {avg_time*1000:.1f}ms", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # ================== SHOW ==================
    cv2.imshow(WINDOW_NAME, canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print final stats
print("\n" + "="*50)
print("BOTTLE INSPECTION - FINAL STATISTICS")
print("="*50)
print(f"Total Frames: {frame_count}")
print(f"Average FPS: {fps:.1f}")
print(f"Total Bottles Counted: {total_count}")
print(f"OK Bottles: {ok_count}")
print(f"Defect Bottles: {defect_count}")
print("="*50)
