# export_tflite.py
# ====== WORKING DIRECTORY GUARD ======
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# =====================================

import logging
from ultralytics import YOLO
from utils import get_latest_run, find_best_pt
from config import TFLITE_NAME, TFLITE_FLOAT16_NAME, NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def export_tflite(run_dir: Path = None):
    if run_dir is None:
        run_dir = get_latest_run(str(PROJECT))
    if run_dir is None:
        raise RuntimeError("No run directory found")

    weights = run_dir / "weights"
    # prefer best.pt, fallback to last.pt
    best = weights / "best.pt"
    if not best.exists():
        last = weights / "last.pt"
        if last.exists():
            best = last
            logger.info("best.pt not found; falling back to last.pt for TFLite export.")
        else:
            raise RuntimeError("No best.pt or last.pt found for TFLite export")

    logger.info("Exporting %s -> tflite", best)
    model = YOLO(str(best))
    # export float32 TFLite (ultralytics will produce best.tflite in weights)
    model.export(format="tflite", project=str(PROJECT), name=NAME)

    # normalize filename
    default = weights / "best.tflite"
    out32 = weights / TFLITE_NAME
    if default.exists() and default != out32:
        default.rename(out32)
    else:
        # check any tflite candidate
        cand = list(weights.glob("*.tflite"))
        if cand:
            cand[0].rename(out32)

    if not out32.exists():
        raise RuntimeError("TFLite export failed: no .tflite produced")

    logger.info("TFLite (float32) saved at %s", out32)

    # Try float16 conversion using TensorFlow if available and if saved_model exists
    try:
        import tensorflow as tf  # type: ignore
        saved_model_dir = weights / "saved_model"
        if saved_model_dir.exists():
            logger.info("Converting SavedModel -> float16 TFLite")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_fp16 = converter.convert()
            out16 = weights / TFLITE_FLOAT16_NAME
            with open(out16, "wb") as f:
                f.write(tflite_fp16)
            logger.info("TFLite (float16) saved at %s", out16)
        else:
            # fallback: try to convert float32 tflite to float16 via TFLite converter (if TF supports)
            try:
                logger.info("No SavedModel; attempting TFLite float32 -> float16 conversion using TF")
                # convert tflite buffer to float16 by re-exporting from saved model isn't always possible
                # so we try to load the model via TF and reconvert (best-effort)
                logger.info("Skipping additional float16 conversion (no saved_model).")
            except Exception as e:
                logger.warning("Float16 conversion attempt failed: %s", e)
    except ImportError:
        logger.info("TensorFlow not installed; skipping float16 conversion.")

    return out32

if __name__ == "__main__":
    export_tflite()
