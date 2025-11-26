# train.py -- clean-first / resume-in-place policy (no retrain folders)
# ========== WORKING DIRECTORY GUARD ==========
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# ============================================

import re
import time
import shutil
import logging
from typing import Optional
from ultralytics import YOLO
from config import DATA_YAML, BASE_MODEL, EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE, NAME
from utils import ensure_dirs, get_latest_run, copy_raw_metrics_to_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_epoch_from_name(name: str) -> Optional[int]:
    nums = re.findall(r'(\d+)', name)
    if not nums:
        return None
    try:
        return max(int(n) for n in nums)
    except Exception:
        return None


def get_max_epoch_in_weights(run_dir: Path) -> int:
    weights = run_dir / "weights"
    if not weights.exists():
        return 0
    pt_files = list(weights.glob("*.pt"))
    if not pt_files:
        return 0
    epochs = []
    for f in pt_files:
        e = _extract_epoch_from_name(f.name)
        if e is not None:
            epochs.append(e)
    return max(epochs) if epochs else 0


def _remove_run_dir(run_dir: Path):
    """Delete run_dir completely (used for clean start)."""
    if run_dir.exists():
        logger.info("Removing existing run directory: %s", run_dir)
        shutil.rmtree(run_dir)


def train(desired_total_epochs: Optional[int] = None,
          resume_if_exists: bool = True,
          force_clean: bool = False) -> None:
    """
    Policy:
      - force_clean=True : delete runs/train/NAME entirely before training (clean-first).
      - resume_if_exists=True : attempt resume from checkpoints in runs/train/NAME.
      - If resume fails, do a fresh run on the SAME NAME (delete old run and start fresh).
    """

    # default epochs
    if desired_total_epochs is None:
        desired_total_epochs = EPOCHS

    # ensure base project path exists
    ensure_dirs(PROJECT)

    run_dir = PROJECT / NAME

    # if user asked to clean, remove the existing run dir entirely
    if force_clean:
        _remove_run_dir(run_dir)

    # ensure run dir exists (ultralytics will create as needed)
    ensure_dirs(run_dir, run_dir / "weights")

    # inspect current run weights for resume info
    last_epoch = 0
    if run_dir.exists():
        last_epoch = get_max_epoch_in_weights(run_dir)
        logger.info("Found existing run '%s'. Inferred last epoch = %s", NAME, last_epoch)
    else:
        logger.info("No existing run '%s' found; starting fresh.", NAME)

    # if already done
    if last_epoch >= desired_total_epochs:
        logger.info("Last epoch (%s) >= desired_total_epochs (%s). Skipping training.", last_epoch, desired_total_epochs)
        return

    model = YOLO(BASE_MODEL)

    # common train args
    train_args = dict(
        data=str(DATA_YAML),
        epochs=desired_total_epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,   # allow writing into same folder
        save=True,
        save_period=1,   # save every epoch (helps resume)
    )

    # If we have partial work and resume is allowed => attempt resume
    if resume_if_exists and last_epoch > 0 and not force_clean:
        logger.info("Attempting resume for run '%s' from epoch %s to %s", NAME, last_epoch, desired_total_epochs)
        try:
            model.train(resume=True, **train_args)
            logger.info("Resume training call finished (check logs for progress).")
            # copy raw metrics into run/metrics/
            copy_raw_metrics_to_run(get_latest_run(str(PROJECT)))
            return
        except AssertionError as e:
            logger.warning("Resume AssertionError: %s", e)
            logger.info("Will fallback to starting fresh on the SAME run name (deleting old run first).")
        except Exception as e:
            logger.warning("Resume raised exception: %s", e)
            logger.info("Will fallback to starting fresh on the SAME run name (deleting old run first).")

    # Fallback: start fresh on the SAME run name (delete old run folder to avoid retrain folder)
    if run_dir.exists():
        _remove_run_dir(run_dir)
    ensure_dirs(run_dir, run_dir / "weights")

    logger.info("Starting fresh training on SAME run name '%s' (epochs=%s).", NAME, desired_total_epochs)
    model.train(**train_args)

    # after training copy raw metrics
    copy_raw_metrics_to_run(get_latest_run(str(PROJECT)))
    logger.info("Training finished for run '%s'.", NAME)


if __name__ == "__main__":
    # default: no force_clean, will try resume
    train()
