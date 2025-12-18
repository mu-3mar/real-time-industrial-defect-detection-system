import cv2
import yaml
from pathlib import Path
from datetime import datetime
from detector.yolo_detector import YOLODetector

# ================== PATHS ==================
BASE = Path(__file__).resolve().parent
det_cfg = yaml.safe_load(open(BASE / "config/detector.yaml"))
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

# ================== STATE ==================
frames_inside = 0
missed_frames = 0
inside = False
total_count = 0

# ================== WINDOW (MAXIMIZED, NOT FULLSCREEN) ==================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_AUTOSIZE,
    cv2.WINDOW_NORMAL
)

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    canvas = cv2.copyMakeBorder(
        frame, 0, 0, INFO_WIDTH, 0,
        cv2.BORDER_CONSTANT, value=(235, 235, 235)
    )

    roi = canvas[:, LEFT_X:RIGHT_X]
    result = detector.detect(roi)

    detected = False

    if result.boxes is not None and len(result.boxes) > 0:
        detected = True
        for b in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b)

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

    # ================== MIN + MAX LOGIC ==================
    if detected:
        frames_inside += 1
        missed_frames = 0
    else:
        missed_frames += 1
        if missed_frames > MAX_MISSED:
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

    # ================== SHOW ==================
    cv2.imshow(WINDOW_NAME, canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
