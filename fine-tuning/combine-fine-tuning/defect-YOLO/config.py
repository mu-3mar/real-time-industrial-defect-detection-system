from pathlib import Path

# -------- PROJECT ROOT (defect-YOLO) --------
# This file should be at: /home/.../defect-YOLO/config.py
ROOT = Path(__file__).resolve().parent

# Runs / project layout (all INSIDE ROOT)
PROJECT = ROOT / "runs" / "train"    # base folder where runs are created
NAME = "detect_defects"                 # run name used by ultralytics

# Weights & metrics locations (derived)
WEIGHTS_DIR = PROJECT / NAME / "weights"
RAW_METRICS_DIRNAME = "metrics"      # inside each run, raw ultralytics metrics go here
FINAL_METRICS_DIR = ROOT / "runs" / "metrics"  # flattened final metrics

# Data & model
DATA_YAML = ROOT / "data" / "data.yaml"
BASE_MODEL = "yolov8n.pt"             # path or model name

# Training hyperparams
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "0"   # "cpu" or GPU id like "0"

# Export filenames (inside weights/)
ONNX_NAME = "best_defect_detector.onnx"
ONNX_INT8_NAME = "best_defect_detector_int8.onnx"
TFLITE_NAME = "best_defect_detector.tflite"
TFLITE_FLOAT16_NAME = "best_defect_detector_float16.tflite"

# Misc
OPSET = 12