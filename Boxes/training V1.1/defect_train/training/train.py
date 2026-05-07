import sys
import os
import logging
from pathlib import Path
from ultralytics import YOLO

# Add project root to path to allow imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from configs.config import (
    ROOT, PROJECT_DIR, DATA_YAML, BASE_MODEL,
    EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE, PROJECT_NAME
)
from utils.utils import ensure_dirs, prune_weights, check_and_download_model

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def resolve_training_device() -> str:
    """
    Resolve device through optional external selector with safe fallback.
    """
    try:
        _boxes = current_file.parent.parent.parent.parent
        _flow = _boxes / "flow"
        if str(_flow) not in sys.path:
            sys.path.insert(0, str(_flow))
        from core.device_manager import select_device  # type: ignore
        return select_device(DEVICE, context="training")
    except Exception:
        logger.info("External device_manager not found, using configured device.")
        return DEVICE


def train():
    """
    Executes the training pipeline.
    """
    run_dir = PROJECT_DIR / PROJECT_NAME
    weights_dir = run_dir / "weights"
    last_pt = weights_dir / "last.pt"

    ensure_dirs(run_dir, weights_dir)
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}")

    if last_pt.exists():
        logger.info(f"Resuming training from {last_pt}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        logger.info(f"Starting fresh training from {BASE_MODEL}")
        # Ensure model exists at specific path before initializing YOLO
        # This prevents YOLO from auto-downloading to root dir
        check_and_download_model(BASE_MODEL)
        model = YOLO(str(BASE_MODEL))
        resume = False

    device = resolve_training_device()
    logger.info(f"Training with model={BASE_MODEL} data={DATA_YAML} device={device}")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=str(PROJECT_DIR),
        name=PROJECT_NAME,
        exist_ok=True,
        resume=resume,
        patience=15,
        cache=True,
        cos_lr=True,
        mosaic=0.8,
        save=True,
        save_period=0
    )

    prune_weights(weights_dir)

if __name__ == "__main__":
    train()
