from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "train"
NAME = "boxs"

WEIGHTS_DIR = PROJECT / NAME / "weights"
RAW_METRICS_DIRNAME = "metrics"
FINAL_METRICS_DIR = ROOT / "runs" / "metrics"

DATA_YAML = ROOT / "data" / "data.yaml"
BASE_MODEL = "yolov8s.pt"

EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640
DEVICE = "0"

ONNX_NAME = "best_box_detector.onnx"
ONNX_INT8_NAME = "best_box_detector_int8.onnx"
TFLITE_NAME = "best_box_detector.tflite"
TFLITE_FLOAT16_NAME = "best_box_detector_float16.tflite"

OPSET = 12
