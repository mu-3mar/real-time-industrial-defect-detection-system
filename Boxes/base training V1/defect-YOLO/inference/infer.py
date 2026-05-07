import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import PROJECT_DIR, PROJECT_NAME

# ============ CONFIG ============
# Note: Update this path to the correct location of your box detection model
BOX_MODEL_PATH = "/home/muhammad-ammar/GraduationProject/QC-SCM/Boxes/fine-tuning/combine-fine-tuning/box-YOLO/runs/train/boxs/weights/best.pt"

# Defect model path (from current project training)
DEFECT_MODEL_PATH = PROJECT_DIR / PROJECT_NAME / "weights" / "best.pt"

BOX_CONF    = 0.8
DEFECT_CONF = 0.3
IOU         = 0.6

COLORS = {
    "ok":     (0, 120, 0),   # dark green (BGR)
    "defect": (0, 0, 150)    # dark red
}
# ===============================

def draw_rounded_corners(img, x1, y1, x2, y2, color, r=12, t=2):
    """Draws a rectangle with rounded corners."""
    # Top-left
    cv2.line(img, (x1 + r, y1), (x1 + 2 * r, y1), color, t)
    cv2.line(img, (x1, y1 + r), (x1, y1 + 2 * r), color, t)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t)

    # Top-right
    cv2.line(img, (x2 - 2 * r, y1), (x2 - r, y1), color, t)
    cv2.line(img, (x2, y1 + r), (x2, y1 + 2 * r), color, t)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t)

    # Bottom-left
    cv2.line(img, (x1 + r, y2), (x1 + 2 * r, y2), color, t)
    cv2.line(img, (x1, y2 - 2 * r), (x1, y2 - r), color, t)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t)

    # Bottom-right
    cv2.line(img, (x2 - 2 * r, y2), (x2 - r, y2), color, t)
    cv2.line(img, (x2, y2 - 2 * r), (x2, y2 - r), color, t)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t)


def draw_defect_box(img, x1, y1, x2, y2):
    """Draws a simple rectangle around the defect inside the crop."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 150), 2)


def main():
    print(f"Loading Box Model: {BOX_MODEL_PATH}")
    print(f"Loading Defect Model: {DEFECT_MODEL_PATH}")
    
    try:
        box_model = YOLO(BOX_MODEL_PATH)
        defect_model = YOLO(str(DEFECT_MODEL_PATH))
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Open camera (index 4 as per original script, change if needed)
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting inference... Press 'q' or 'Esc' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # -------- Box Detection --------
        box_results = box_model(
            frame,
            conf=BOX_CONF,
            iou=IOU,
            verbose=False
        )

        for r in box_results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                bx1, by1, bx2, by2 = map(int, b.xyxy[0])

                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)

                crop = frame[by1:by2, bx1:bx2]
                if crop.size == 0:
                    continue

                # -------- Defect Detection --------
                defect_results = defect_model(
                    crop,
                    conf=DEFECT_CONF,
                    iou=IOU,
                    verbose=False
                )

                status = "ok"
                defect_boxes = []

                for dr in defect_results:
                    if dr.boxes is None:
                        continue

                    for db in dr.boxes:
                        status = "defect"
                        dx1, dy1, dx2, dy2 = map(int, db.xyxy[0])
                        defect_boxes.append((dx1, dy1, dx2, dy2))

                color = COLORS[status]

                # -------- Draw Defect Boxes (inside crop) --------
                if status == "defect":
                    for dx1, dy1, dx2, dy2 in defect_boxes:
                        draw_defect_box(
                            frame[by1:by2, bx1:bx2],
                            dx1, dy1, dx2, dy2
                        )

                # -------- Draw Outer Box --------
                draw_rounded_corners(frame, bx1, by1, bx2, by2, color)

                cv2.putText(
                    frame,
                    status.upper(),
                    (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("Defect Inspection", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
