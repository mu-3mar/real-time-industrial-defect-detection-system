from pathlib import Path
import os

# Project root directory (one level up from this file)
ROOT = Path(__file__).resolve().parent.parent

# Paths
MODELS_DIR = ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
EXPORTED_DIR = MODELS_DIR / "exported"
PROJECT_ROOT = ROOT.parent
DATA_PREPARATION_DIR = PROJECT_ROOT / "data_preparation"
DEFECT_DATASET_DIR = DATA_PREPARATION_DIR / "datasets" / "defect_dataset"
RUNS_DIR = ROOT / "runs"

# Training Configuration
PROJECT_NAME = "detect_defect"
PROJECT_DIR = RUNS_DIR / "train"
BASE_MODEL = Path(
    os.getenv(
        "DEFECT_TRAIN_BASE_MODEL",
        str(PRETRAINED_DIR / "defect_detector.pt"),
    )
).resolve()
DATA_YAML = Path(
    os.getenv(
        "DEFECT_TRAIN_DATA_YAML",
        str(DEFECT_DATASET_DIR / "data.yaml"),
    )
).resolve()

# Hyperparameters
EPOCHS = int(os.getenv("DEFECT_TRAIN_EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("DEFECT_TRAIN_BATCH_SIZE", "8"))
IMG_SIZE = int(os.getenv("DEFECT_TRAIN_IMG_SIZE", "640"))
DEVICE = os.getenv("DEFECT_TRAIN_DEVICE", "auto")  # auto | cuda | mps | cpu

# Export Configuration
ONNX_NAME = "detect_defect.onnx"
ONNX_INT8_NAME = "detect_defect_int8.onnx"
ENGINE_NAME = "detect_defect.engine"
OPSET = 12
ENGINE_DEVICE = os.getenv("DEFECT_TRAIN_ENGINE_DEVICE", DEVICE)
ENGINE_HALF = os.getenv("DEFECT_TRAIN_ENGINE_HALF", "true").lower() == "true"

# Metrics
FINAL_METRICS_DIR = RUNS_DIR / "metrics"
