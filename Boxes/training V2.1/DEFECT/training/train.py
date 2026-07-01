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


# Realistic industrial conveyor belt augmentation parameters
INDUSTRIAL_AUGMENTATION_PARAMS = {
    'hsv_h': 0.015,       # Subtle hue variation
    'hsv_s': 0.45,        # Moderate saturation variation
    'hsv_v': 0.30,        # Mild value variation
    'degrees': 2.0,       # Very small rotation (±2 degrees)
    'translate': 0.04,    # Small translation (±4%)
    'scale': 0.20,        # Moderate scaling (±20%)
    'shear': 1.0,         # Tiny shear (±1 degree)
    'perspective': 0.0,   # No perspective distortion
    'flipud': 0.0,        # No vertical flip (unrealistic for conveyor)
    'fliplr': 0.5,        # 50% horizontal flip (acceptable for symmetrical objects)
    'mosaic': 0.6,        # Mosaic with 60% probability
    'mixup': 0.10,        # MixUp with 10% probability
    'copy_paste': 0.0     # No copy-paste
}


def _latest_checkpoint():
    """Return latest checkpoint path from runs/train, preferring last.pt then best.pt."""
    candidates = list(PROJECT_DIR.glob("**/weights/last.pt"))
    if not candidates:
        candidates = list(PROJECT_DIR.glob("**/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def train():
    """Execute training pipeline for industrial conveyor belt quality control."""
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
        check_and_download_model(BASE_MODEL)
        model = YOLO(str(BASE_MODEL))

    device = select_device(DEVICE, context="training")
    logger.info(f"Using training device: {device}")
    logger.info("Using industrial conveyor belt optimized augmentations")

    # Train the model
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
        save=True,
        save_period=0,
        # Industrial augmentation parameters
        hsv_h=INDUSTRIAL_AUGMENTATION_PARAMS['hsv_h'],
        hsv_s=INDUSTRIAL_AUGMENTATION_PARAMS['hsv_s'],
        hsv_v=INDUSTRIAL_AUGMENTATION_PARAMS['hsv_v'],
        degrees=INDUSTRIAL_AUGMENTATION_PARAMS['degrees'],
        translate=INDUSTRIAL_AUGMENTATION_PARAMS['translate'],
        scale=INDUSTRIAL_AUGMENTATION_PARAMS['scale'],
        shear=INDUSTRIAL_AUGMENTATION_PARAMS['shear'],
        perspective=INDUSTRIAL_AUGMENTATION_PARAMS['perspective'],
        flipud=INDUSTRIAL_AUGMENTATION_PARAMS['flipud'],
        fliplr=INDUSTRIAL_AUGMENTATION_PARAMS['fliplr'],
        mosaic=INDUSTRIAL_AUGMENTATION_PARAMS['mosaic'],
        mixup=INDUSTRIAL_AUGMENTATION_PARAMS['mixup'],
        copy_paste=INDUSTRIAL_AUGMENTATION_PARAMS['copy_paste']
    )

    prune_weights(weights_dir)


if __name__ == "__main__":
    train()

