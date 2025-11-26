# run_all.py
# ====== WORKING DIRECTORY GUARD ======
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# =====================================

import logging
from train import train
from export_onnx import export_onnx
from quantize_onnx import quantize
from export_tflite import export_tflite
from utils import get_latest_run, collect_final_metrics
from config import FINAL_METRICS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1) train (train.py handles resume/clean logic)
    logger.info("1) Training step")
    # call with defaults (change args in CLI if you want)
    train()

    # 2) find run dir
    run_dir = get_latest_run(str(PROJECT))
    if not run_dir:
        logger.error("No run directory found after training. Aborting exports.")
        return
    logger.info("Using run dir: %s", run_dir)

    # 3) export ONNX (float32)
    try:
        logger.info("2) Exporting ONNX (float32)")
        onnx_path = export_onnx(run_dir=run_dir)
    except Exception as e:
        logger.exception("export_onnx failed: %s", e)
        onnx_path = None

    # 4) quantize ONNX -> INT8
    if onnx_path is not None:
        try:
            logger.info("3) Quantizing ONNX -> INT8")
            quant_path = quantize(run_dir=run_dir, onnx_path=onnx_path)
        except Exception as e:
            logger.exception("quantize failed: %s", e)

    # 5) export TFLite float32 and float16
    try:
        logger.info("4) Exporting TFLite (float32 + optional float16)")
        export_tflite(run_dir=run_dir)
    except Exception as e:
        logger.exception("export_tflite failed: %s", e)

    # 6) collect final metrics
    try:
        collect_final_metrics(run_dir, FINAL_METRICS_DIR)
        logger.info("Collected final metrics into %s", FINAL_METRICS_DIR)
    except Exception as e:
        logger.exception("collect_final_metrics failed: %s", e)


if __name__ == "__main__":
    main()
