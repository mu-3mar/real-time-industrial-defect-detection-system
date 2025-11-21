from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

IMG = "test.jpg"          # غيّر إلى أى صورة عندك
CONF = 0.80

# ---------- إنشاء/تحميل النماذج ----------
pt_model   = YOLO("output/best.pt")                       # PyTorch FP32
onnx_model = YOLO("output/best.onnx")                     # ONNX FP32
# لو الـ INT8 ملف مختلف غيّر الاسم
int8_model = YOLO("output/best_int8.onnx")                # ONNX INT8

models = {
    "PyTorch (FP32)": pt_model,
    "ONNX (FP32)":    onnx_model,
    "ONNX (INT8)":    int8_model,
}

# ---------- دالة المقارنة ----------
def compare_models(models, image_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, model) in zip(axes, models.items()):
        results = model(image_path, save=False, conf=CONF)[0]
        annotated = results.plot()[:, :, ::-1]  # BGR → RGB
        ax.imshow(annotated)
        ax.set_title(name); ax.axis('off')

        # الأرقام
        if results.boxes is not None:
            b = results.boxes.xyxy.cpu().numpy()
            s = results.boxes.conf.cpu().numpy()
            c = results.boxes.cls.cpu().numpy()
            print(f"\n--- {name} ---")
            print("Boxes :\n", b)
            print("Scores:\n", s)
            print("Classes:\n", c)
        else:
            print(f"\n--- {name} ---\nNo detections")
    plt.tight_layout(); plt.show()

# ---------- تشغيل ----------
compare_models(models, IMG)