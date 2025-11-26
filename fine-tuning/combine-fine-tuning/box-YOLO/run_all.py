# run_all.py
# ========== WORKING DIRECTORY GUARD ==========
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# ============================================

import logging
import sys
from train import train
from export_onnx import export_onnx
from quantize_onnx import quantize
from export_tflite import export_tflite
from utils import get_latest_run, collect_final_metrics
from config import FINAL_METRICS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # CLI parsing simple:
    # usage:
    #   python run_all.py            -> try resume if possible
    #   python run_all.py clean      -> force clean start (delete old run)
    #   python run_all.py 100        -> desired epochs = 100 (resume if possible)
    #   python run_all.py 100 clean  -> desired epochs=100 and force clean
    args = sys.argv[1:]
    desired_epochs = None
    force_clean = False

    for a in args:
        if a.isdigit():
            desired_epochs = int(a)
        if a.lower() in ("clean", "force", "fresh"):
            force_clean = True

    logger.info("Starting pipeline. desired_epochs=%s, force_clean=%s", desired_epochs, force_clean)

    # call train with policy
    train(desired_total_epochs=desired_epochs, resume_if_exists=not force_clean, force_clean=force_clean)

    run_dir = get_latest_run(str(PROJECT))
    if not run_dir:
        logger.error("No run found after training. Aborting exports.")
        return

    logger.info("Run dir: %s", run_dir)

    try:
        export_onnx()
    except Exception as e:
        logger.warning("export_onnx failed: %s", e)

    try:
        quantize()
    except Exception as e:
        logger.warning("quantize_onnx failed: %s", e)

    try:
        export_tflite()
    except Exception as e:
        logger.warning("export_tflite failed: %s", e)

    try:
        collect_final_metrics(run_dir, FINAL_METRICS_DIR)
    except Exception as e:
        logger.warning("collect_final_metrics failed: %s", e)


if __name__ == "__main__":
    main()
