import os
import re
import shutil
import logging
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
from config import ROOT, PROJECT, DATA_YAML, BASE_MODEL, EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE, NAME
from utils import ensure_dirs, get_latest_run, copy_raw_metrics_to_run

PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_epoch_from_name(name: str) -> Optional[int]:
    """Extract epoch number from filename."""
    nums = re.findall(r'(\d+)', name)
    return max(int(n) for n in nums) if nums else None


def get_max_epoch_in_weights(run_dir: Path) -> int:
    """Get the maximum epoch number from existing weights."""
    weights = run_dir / "weights"
    if not weights.exists():
        return 0
    pt_files = list(weights.glob("*.pt"))
    if not pt_files:
        return 0
    epochs = [_extract_epoch_from_name(f.name) for f in pt_files]
    epochs = [e for e in epochs if e is not None]
    return max(epochs) if epochs else 0


def _remove_run_dir(run_dir: Path):
    """Delete run directory completely."""
    if run_dir.exists():
        logger.info("Removing existing run directory: %s", run_dir)
        shutil.rmtree(run_dir)


def _prune_epoch_files(weights_dir: Path):
    """Keep only last.pt and best.pt, remove all epoch*.pt files."""
    if not weights_dir.exists():
        return
    for f in weights_dir.glob("epoch*.pt"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in weights_dir.glob("*epoch*.pt"):
        name = f.name.lower()
        if "last" not in name and "best" not in name:
            try:
                f.unlink()
            except Exception:
                pass


def train(desired_total_epochs: Optional[int] = None,
          resume_if_exists: bool = True,
          force_clean: bool = False) -> None:
    """Train YOLO model, keeping only last.pt and best.pt."""
    if desired_total_epochs is None:
        desired_total_epochs = EPOCHS

    ensure_dirs(PROJECT)
    run_dir = PROJECT / NAME
    weights_dir = run_dir / "weights"

    if force_clean:
        _remove_run_dir(run_dir)

    ensure_dirs(run_dir, weights_dir)

    last_epoch = 0
    if run_dir.exists():
        last_epoch = get_max_epoch_in_weights(run_dir)
        logger.info("Found existing run '%s'. Last epoch = %s", NAME, last_epoch)
    else:
        logger.info("Starting fresh run '%s'.", NAME)

    if last_epoch >= desired_total_epochs:
        logger.info("Training already complete (epoch %s >= %s).", last_epoch, desired_total_epochs)
        return

    model = YOLO(BASE_MODEL)
    train_args = dict(
        data=str(DATA_YAML),
        epochs=desired_total_epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        save=True,
        save_period=1,
    )

    if weights_dir.exists():
        logger.info("Pruning epoch files before resume.")
        _prune_epoch_files(weights_dir)

    if resume_if_exists and last_epoch > 0 and not force_clean:
        logger.info("Attempting resume from epoch %s to %s.", last_epoch, desired_total_epochs)
        try:
            model.train(resume=True, **train_args)
            logger.info("Resume training finished.")
            run_dir_post = get_latest_run(str(PROJECT))
            if run_dir_post:
                _prune_epoch_files(run_dir_post / "weights")
                copy_raw_metrics_to_run(run_dir_post)
            return
        except (AssertionError, Exception) as e:
            logger.warning("Resume failed: %s. Starting fresh.", str(e))

    if run_dir.exists():
        _remove_run_dir(run_dir)
    ensure_dirs(run_dir, weights_dir)

    logger.info("Starting fresh training (epochs=%s).", desired_total_epochs)
    model.train(**train_args)

    run_dir_post = get_latest_run(str(PROJECT))
    if run_dir_post:
        _prune_epoch_files(run_dir_post / "weights")
        copy_raw_metrics_to_run(run_dir_post)
    logger.info("Training complete for '%s'.", NAME)


if __name__ == "__main__":
    train()
