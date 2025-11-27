from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "train"
NAME = "detect_defects"

WEIGHTS_DIR = PROJECT / NAME / "weights"
RAW_METRICS_DIRNAME = "metrics"
FINAL_METRICS_DIR = ROOT / "runs" / "metrics"

DATA_YAML = ROOT / "data" / "data.yaml"
BASE_MODEL = "yolov8n.pt"

EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "0"

ONNX_NAME = "best_defect_detector.onnx"
ONNX_INT8_NAME = "best_defect_detector_int8.onnx"
TFLITE_NAME = "best_defect_detector.tflite"
TFLITE_FLOAT16_NAME = "best_defect_detector_float16.tflite"

OPSET = 12