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

try:
    # Optional external helper used in some environments.
    from core.device_manager import select_device  # type: ignore
except ImportError:
    def select_device(device, context=None):
        """Fallback device selector when external helper is unavailable."""
        try:
            import torch

            requested = str(device).lower()
            if requested in ("cpu", "mps"):
                return requested
            if requested == "auto":
                # Ultralytics does not always accept "auto"; provide explicit value.
                return "0" if torch.cuda.device_count() > 0 else "cpu"
            if requested == "cuda":
                return "0" if torch.cuda.device_count() > 0 else "cpu"
            if torch.cuda.device_count() > 0:
                return str(device)
        except Exception:
            pass
        return "cpu"

from configs.config import (
    ROOT, PROJECT_DIR, DATA_YAML, BASE_MODEL, 
    EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE, PROJECT_NAME
)
from utils.utils import ensure_dirs, prune_weights, check_and_download_model

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _latest_checkpoint():
    """
    Return latest checkpoint path from runs/train, preferring last.pt then best.pt.
    """
    candidates = list(PROJECT_DIR.glob("**/weights/last.pt"))
    if not candidates:
        candidates = list(PROJECT_DIR.glob("**/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def train():
    """
    Executes the training pipeline.
    """
    run_dir = PROJECT_DIR / PROJECT_NAME
    weights_dir = run_dir / "weights"
    ensure_dirs(run_dir, weights_dir)

    checkpoint = _latest_checkpoint()
    resume_training = checkpoint is not None

    if resume_training:
        logger.info(f"Resuming training from checkpoint: {checkpoint}")
        model = YOLO(str(checkpoint))
    else:
        logger.info(f"Starting fresh training from {BASE_MODEL}")
        # Ensure model exists at specific path before initializing YOLO
        # This prevents YOLO from auto-downloading to root dir
        check_and_download_model(BASE_MODEL)
        model = YOLO(str(BASE_MODEL))

    device = select_device(DEVICE, context="training")
    logger.info(f"Using training device: {device}")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=str(PROJECT_DIR),
        name=PROJECT_NAME,
        exist_ok=True,
        resume=resume_training,
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
