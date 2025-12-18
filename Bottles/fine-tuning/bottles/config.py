from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROJECT = ROOT / "runs" / "train"
NAME = "bottles"
BASE_MODEL = "yolov8s.pt"

DATA_YAML = ROOT / "data" / "data.yaml"

EPOCHS = 80
BATCH_SIZE = 8
IMG_SIZE = 640
DEVICE = "0"

ONNX_NAME = "bottles.onnx"
ONNX_INT8_NAME = "bottles_int8.onnx"
OPSET = 12

FINAL_METRICS_DIR = ROOT / "runs" / "metrics"
