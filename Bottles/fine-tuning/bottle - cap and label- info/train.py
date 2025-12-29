import os
import logging
from ultralytics import YOLO
from pathlib import Path

from config import ROOT, PROJECT, DATA_YAML, BASE_MODEL, EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE, NAME
from utils import ensure_dirs, prune_weights

os.chdir(str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def train():
    run_dir = PROJECT / NAME
    weights_dir = run_dir / "weights"
    last_pt = weights_dir / "last.pt"

    ensure_dirs(run_dir, weights_dir)

    if last_pt.exists():
        logger.info("Resuming training from last.pt")
        model = YOLO(str(last_pt))
        resume = True
    else:
        logger.info("Starting fresh training from base model")
        model = YOLO(BASE_MODEL)
        resume = False

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT),
        name=NAME,
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
