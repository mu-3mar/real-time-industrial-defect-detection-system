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
PROJECT_NAME = "defect_detector"
PROJECT_DIR = RUNS_DIR / "train"
BASE_MODEL = PRETRAINED_DIR / "yolo26n.pt"
DATA_YAML = DATA_DIR / "data.yaml"

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "auto"  # auto | cuda | mps | cpu; override with QC_SCM_TRAIN_DEVICE

# Export Configuration
ONNX_NAME = "defect_detector.onnx"
ONNX_INT8_NAME = "defect_detector_int8.onnx"
TENSORRT_NAME = "defect_detector.engine"
OPSET = 12

# Metrics
FINAL_METRICS_DIR = RUNS_DIR / "metrics"
