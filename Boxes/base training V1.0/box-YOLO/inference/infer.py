import sys
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

# ================= CONFIG =================

DEVICE = "cpu"     # "cpu" او "cuda"
MODE   = "camera"  # "camera" او "folder"

FOLDER_PATH = "/home/muhammad-ammar/GraduationProject/tamp train/test"  
CAM_INDEX = 4

MODEL_PATH = "models/exported/best.pt"

CONF = 0.75
IOU  = 0.6

COLOR = (0,120,0)

# ==========================================

def draw_rounded_corners(img, x1, y1, x2, y2, color, r=12, t=2):

    cv2.line(img,(x1+r,y1),(x1+2*r,y1),color,t)
    cv2.line(img,(x1,y1+r),(x1,y1+2*r),color,t)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,t)

    cv2.line(img,(x2-2*r,y1),(x2-r,y1),color,t)
    cv2.line(img,(x2,y1+r),(x2,y1+2*r),color,t)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,t)

    cv2.line(img,(x1+r,y2),(x1+2*r,y2),color,t)
    cv2.line(img,(x1,y2-2*r),(x1,y2-r),color,t)
    cv2.ellipse(img,(x1+r,y2-r),(r,r),90,0,90,color,t)

    cv2.line(img,(x2-2*r,y2),(x2-r,y2),color,t)
    cv2.line(img,(x2,y2-2*r),(x2,y2-r),color,t)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),0,0,90,color,t)


# -------- Basic inference (CPU/GPU) --------
def detect(model, frame):

    results = model(
        frame,
        imgsz=640,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        half=(DEVICE=="cuda"),
        verbose=False
    )

    boxes=[]

    for r in results:
        if r.boxes is None:
            continue

        for b in r.boxes:
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            boxes.append((x1,y1,x2,y2))

    return boxes

# -------- Camera Mode --------
def camera_mode(model):

    cap = cv2.VideoCapture(CAM_INDEX)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect(model, frame)

        for x1,y1,x2,y2 in boxes:
            draw_rounded_corners(frame,x1,y1,x2,y2,COLOR)

        cv2.imshow("REALTIME DETECTION", frame)

        key = cv2.waitKey(1)

        if key in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- Folder Mode --------
def folder_mode(model):
    IMAGE_EXTENSIONS = [
        "*.jpg", "*.jpeg", "*.png",
        "*.bmp", "*.tiff", "*.tif",
        "*.webp", "*.JPG", "*.PNG"
    ]

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(f"{FOLDER_PATH}/{ext}"))

    images = sorted(images)

    index = 0

    while True:

        img = cv2.imread(images[index])
        frame = img.copy()

        boxes = detect(model, frame)

        for x1,y1,x2,y2 in boxes:
            draw_rounded_corners(frame,x1,y1,x2,y2,COLOR)

        cv2.imshow("IMAGE BROWSER", frame)

        key = cv2.waitKey(0)

        # right arrow
        if key == 83:
            index = min(index+1, len(images)-1)

        # left arrow
        elif key == 81:
            index = max(index-1, 0)

        elif key in [27, ord('q')]:
            break

    cv2.destroyAllWindows()


# ================= MAIN =================

def main():

    print(f"🔥 DEVICE: {DEVICE}")
    print(f"🔥 MODE: {MODE}")

    model = YOLO(MODEL_PATH)

    if MODE == "camera":
        camera_mode(model)

    elif MODE == "folder":
        folder_mode(model)

if __name__ == "__main__":
    main()
