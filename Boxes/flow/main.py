import cv2
import yaml
import time
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict
from detector.yolo_detector import YOLODetector

# ================== PATHS ==================
BASE = Path(__file__).resolve().parent
box_cfg = yaml.safe_load(open(BASE / "config/box_detector.yaml"))
defect_cfg = yaml.safe_load(open(BASE / "config/defect_detector.yaml"))
str_cfg = yaml.safe_load(open(BASE / "config/stream.yaml"))

# ================== UI ==================
INFO_WIDTH = 300
ROI_WIDTH = 400
ROI_CENTER_OFFSET = 420

LEFT_X = INFO_WIDTH + ROI_CENTER_OFFSET - ROI_WIDTH // 2
RIGHT_X = INFO_WIDTH + ROI_CENTER_OFFSET + ROI_WIDTH // 2

WINDOW_NAME = "Box Inspection System"

# ================== STABILITY ==================
MIN_FRAMES = 3          # frames to consider "entered"
MAX_MISSED = 5          # frames to consider "exited"
VOTE_WINDOW = 7         # temporal smoothing window
VOTE_THRESHOLD = 4      # >= => DEFECT

# Optimization
SKIP_DEFECT_FRAMES = 2  # Run defect detector every 3rd frame (0, 3, 6...)

# ================== VIDEO ==================
cap = cv2.VideoCapture(str_cfg["source"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, str_cfg["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, str_cfg["height"])

# ================== MODELS ==================
box_detector = YOLODetector(
    box_cfg["model_path"],
    box_cfg["conf_thres"],
    box_cfg["iou_thres"],
    box_cfg["device"]
)

defect_detector = YOLODetector(
    defect_cfg["model_path"],
    defect_cfg["conf_thres"],
    defect_cfg["iou_thres"],
    defect_cfg["device"]
)

# ================== STATE ==================
box_histories = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
last_defect_results = {} # Cache for optimization: box_id -> bool

frames_inside = 0
missed_frames = 0
inside = False

final_decision = None   # "DEFECT" | "OK"

total_count = 0
defect_count = 0
ok_count = 0

# Performance
frame_count = 0
start_time = time.time()
last_time = time.time()
fps = 0

# ================== WINDOW ==================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # FPS Calculation
    if frame_count % 10 == 0:
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed > 0:
            fps = 10 / elapsed
        last_time = current_time

    h, w, _ = frame.shape

    canvas = cv2.copyMakeBorder(
        frame, 0, 0, INFO_WIDTH, 0,
        cv2.BORDER_CONSTANT, value=(235, 235, 235)
    )

    roi = frame[:, LEFT_X - INFO_WIDTH:RIGHT_X - INFO_WIDTH]
    result = box_detector.detect(roi)

    detected = False

    # ========== DETECTION ==========
    if result.boxes is not None:
        for box in result.boxes.xyxy.cpu().numpy():
            detected = True
            x1, y1, x2, y2 = map(int, box)

            abs_x1 = x1 + LEFT_X
            abs_x2 = x2 + LEFT_X

            # ---------- CROP ----------
            crop = frame[
                y1:y2,
                x1 + (LEFT_X - INFO_WIDTH):x2 + (LEFT_X - INFO_WIDTH)
            ]

            # ---------- DEFECT DETECTION (OPTIMIZED) ----------
            box_id = (x1 // 20, y1 // 20)  # spatially stable ID
            
            hole_detected = False
            
            # Determine if we should run defect detection
            # Run if: 1. New box (not in cache) OR 2. Interval met
            should_run = (box_id not in last_defect_results) or (frame_count % (SKIP_DEFECT_FRAMES + 1) == 0)
            
            if should_run:
                if crop.size > 0:
                    d_res = defect_detector.detect(crop)
                    if d_res.boxes is not None:
                        for cls in d_res.boxes.cls.cpu().numpy():
                            if int(cls) == 0:  # Hole ONLY
                                hole_detected = True
                                break
                last_defect_results[box_id] = hole_detected
            else:
                # Use cached result
                hole_detected = last_defect_results.get(box_id, False)

            # ---------- TEMPORAL VOTING ----------
            box_histories[box_id].append(hole_detected)

            votes = sum(box_histories[box_id])
            if votes >= VOTE_THRESHOLD:
                status = "Defect Box"
                color = (0, 0, 255)
                final_decision = "DEFECT"
            else:
                status = "Non-Defect Box"
                color = (0, 170, 0)
                final_decision = "OK"

            # ---------- DRAW ----------
            cv2.rectangle(canvas, (abs_x1, y1), (abs_x2, y2), color, 2)
            cv2.putText(
                canvas, status,
                (abs_x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    # ================== ENTRY / EXIT LOGIC ==================
    if detected:
        frames_inside += 1
        missed_frames = 0

        if frames_inside >= MIN_FRAMES:
            inside = True

    else:
        missed_frames += 1

        # ===== EXIT → COUNT HERE ONLY =====
        if missed_frames > MAX_MISSED and inside:
            total_count += 1

            if final_decision == "DEFECT":
                defect_count += 1
            else:
                ok_count += 1
            
            # Log output every ~ second (per box exit technically, but close enough for logic)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Box Processed: {final_decision} | Total: {total_count} (D:{defect_count} OK:{ok_count})")

            # reset state
            inside = False
            frames_inside = 0
            final_decision = None
            box_histories.clear()
            last_defect_results.clear() # Clear cache on exit (simplified)

    # ================== ROI LINES ==================
    cv2.line(canvas, (LEFT_X, 0), (LEFT_X, h), (0, 0, 0), 2)
    cv2.line(canvas, (RIGHT_X, 0), (RIGHT_X, h), (0, 0, 0), 2)

    # ================== INFO PANEL ==================
    px, py = 30, 70
    cv2.putText(canvas, "TOTAL", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    py += 45
    cv2.putText(canvas, str(total_count), (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

    py += 60
    cv2.putText(canvas, f"DEFECT: {defect_count}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    py += 35
    cv2.putText(canvas, f"OK: {ok_count}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 0), 2)

    py += 40
    cv2.putText(canvas,
                datetime.now().strftime("%d / %m / %Y"),
                (px, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2)
    
    # FPS
    py += 50
    cv2.line(canvas, (px, py), (INFO_WIDTH - 30, py), (180, 180, 180), 1)
    py += 30
    cv2.putText(canvas, f"FPS: {fps:.1f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # ================== SHOW ==================
    cv2.imshow(WINDOW_NAME, canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
