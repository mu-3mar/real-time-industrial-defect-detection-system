from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROJECT = ROOT / "runs" / "train"
NAME = "bottle_info"
BASE_MODEL = "yolov8n.pt"

DATA_YAML = ROOT / "data" / "data.yaml"

EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "0"

ONNX_NAME = "bottle_info.onnx"
ONNX_INT8_NAME = "bottle_info_int8.onnx"
OPSET = 12

FINAL_METRICS_DIR = ROOT / "runs" / "metrics"
