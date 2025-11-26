# ========== WORKING DIRECTORY GUARD ==========
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# ============================================

import logging
from ultralytics import YOLO
from config import TFLITE_NAME, TFLITE_FLOAT16_NAME, NAME
from utils import get_latest_run, find_best_pt, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_tflite():
    run_dir = get_latest_run(str(PROJECT))
    if run_dir is None:
        raise RuntimeError("No run directory found")

    best = find_best_pt(run_dir)
    if best is None:
        raise RuntimeError("best.pt not found in run")

    weights = run_dir / "weights"
    ensure_dirs(weights)

    model = YOLO(str(best))
    logger.info("Exporting to TFLite (float32)")
    # pass project/name to keep export inside project area
    model.export(format="tflite", project=str(PROJECT), name=NAME)

    default = weights / "best.tflite"
    out_path = weights / TFLITE_NAME
    if default.exists() and default != out_path:
        default.rename(out_path)
        logger.info("TFLite saved at %s", out_path)
    else:
        # try other tflite candidates
        candidates = list(weights.glob("*.tflite"))
        if candidates:
            candidates[0].rename(out_path)
            logger.info("TFLite saved at %s", out_path)
        else:
            raise RuntimeError("No tflite file produced by ultralytics export")

    # Optional: try float16 if SavedModel exists and TF is available
    try:
        import tensorflow as tf  # type: ignore
        saved_model_dir = weights / "saved_model"
        if saved_model_dir.exists():
            logger.info("Converting SavedModel to float16 TFLite")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_float16 = converter.convert()
            out16 = weights / TFLITE_FLOAT16_NAME
            with open(out16, "wb") as f:
                f.write(tflite_float16)
            logger.info("Float16 TFLite saved at %s", out16)
        else:
            logger.info("No SavedModel folder found; skipped float16 conversion.")
    except ImportError:
        logger.info("TensorFlow not installed; skipped float16 conversion.")
    except Exception as e:
        logger.warning("Float16 conversion failed: %s", e)

    return out_path


if __name__ == "__main__":
    export_tflite()
