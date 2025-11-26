# train.py -- keep only last.pt & best.pt, resume from last.pt
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


def _prune_epoch_files(weights_dir: Path):
    """
    Remove epochN.pt files but keep last.pt and best.pt.
    This ensures resume uses last.pt only and prevents many epoch files stacking.
    """
    if not weights_dir.exists():
        return
    for f in weights_dir.glob("epoch*.pt"):
        try:
            f.unlink()
        except Exception as e:
            logger.debug("Could not delete %s: %s", f, e)
    # also remove any other files that look like 'yolov8n_epoch_123.pt' etc but not last/best
    for f in weights_dir.glob("*epoch*.pt"):
        name = f.name.lower()
        if "last" in name or "best" in name:
            continue
        try:
            f.unlink()
        except Exception:
            pass


def train(desired_total_epochs: Optional[int] = None,
          resume_if_exists: bool = True,
          force_clean: bool = False) -> None:
    """
    - Keep only last.pt & best.pt in weights/.
    - Before resume: prune epoch*.pt so resume will use last.pt cleanly.
    - After training: prune epoch files, leaving only last.pt & best.pt.
    - If resume fails: delete old run and start fresh on the SAME run name.
    """

    # default epochs
    if desired_total_epochs is None:
        desired_total_epochs = EPOCHS

    # ensure base project path exists
    ensure_dirs(PROJECT)

    run_dir = PROJECT / NAME
    weights_dir = run_dir / "weights"

    # if user asked to clean, remove the existing run dir entirely
    if force_clean:
        _remove_run_dir(run_dir)

    # ensure run dir exists (ultralytics will create as needed)
    ensure_dirs(run_dir, weights_dir)

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
        exist_ok=True,
        save=True,
        save_period=1,   # save every epoch (helps resume and produces last.pt)
    )

    # PREP: prune epoch files so resume reads only last.pt & best.pt
    if weights_dir.exists():
        logger.info("Pruning epoch files before attempting resume (keep last.pt & best.pt).")
        _prune_epoch_files(weights_dir)

    # If we have partial work and resume is allowed => attempt resume
    if resume_if_exists and last_epoch > 0 and not force_clean:
        logger.info("Attempting resume for run '%s' from epoch %s to %s", NAME, last_epoch, desired_total_epochs)
        try:
            model.train(resume=True, **train_args)
            logger.info("Resume training call finished (check logs for progress).")
            # After training prune epoch files and copy metrics
            run_dir_post = get_latest_run(str(PROJECT))
            if run_dir_post:
                _prune_epoch_files(run_dir_post / "weights")
                copy_raw_metrics_to_run(run_dir_post)
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
    ensure_dirs(run_dir, weights_dir)

    logger.info("Starting fresh training on SAME run name '%s' (epochs=%s).", NAME, desired_total_epochs)
    model.train(**train_args)

    # After training prune epoch files and copy raw metrics
    run_dir_post = get_latest_run(str(PROJECT))
    if run_dir_post:
        _prune_epoch_files(run_dir_post / "weights")
        copy_raw_metrics_to_run(run_dir_post)
    logger.info("Training finished for run '%s'. Remaining weights: %s", NAME,
                [p.name for p in (run_dir_post / "weights").glob("*") if p.exists()] if run_dir_post else "none")


if __name__ == "__main__":
    # default: no force_clean, will try resume
    train()
