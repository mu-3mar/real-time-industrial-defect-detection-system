from pathlib import Path

# Project root directory (one level up from this file)
ROOT = Path(__file__).resolve().parent.parent

# Paths
MODELS_DIR = ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
EXPORTED_DIR = MODELS_DIR / "exported"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

# Training Configuration
PROJECT_NAME = "detect_box"
PROJECT_DIR = RUNS_DIR / "train"
BASE_MODEL = PRETRAINED_DIR / "yolo26n.pt"
DATA_YAML = DATA_DIR / "data" / "data.yaml"

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640
DEVICE = "auto"  # auto | cuda | mps | cpu; override with QC_SCM_TRAIN_DEVICE

# Export Configuration
ONNX_NAME = "detect_box.onnx"
ONNX_INT8_NAME = "detect_box_int8.onnx"
OPSET = 12

# Metrics
FINAL_METRICS_DIR = RUNS_DIR / "metrics"
