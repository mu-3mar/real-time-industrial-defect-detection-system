import cv2
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "runs/train/boxs/weights/best.pt"   # حط مسار الموديل هنا
CONF_THRES = 0.6
IOU_THRES  = 0.6
CAM_ID = 0               # 0 = webcam
# =========================================

# Load model
model = YOLO(MODEL_PATH)

# Open camera
cap = cv2.VideoCapture(CAM_ID)

if not cap.isOpened():
    raise RuntimeError("❌ Camera not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(
        frame,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device="cpu",      # 0 = GPU, cpu = CPU
        verbose=False
    )

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            label = f"box {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
