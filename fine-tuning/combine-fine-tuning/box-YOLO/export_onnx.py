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
from config import ONNX_NAME, OPSET, NAME
from utils import get_latest_run, find_best_pt, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_onnx(opset: int = OPSET, dynamic: bool = True):
    run_dir = get_latest_run(str(PROJECT))
    if run_dir is None:
        raise RuntimeError("No run directory found under PROJECT: " + str(PROJECT))

    best = find_best_pt(run_dir)
    if best is None:
        raise RuntimeError("best.pt not found in weights for run: " + str(run_dir))

    weights = run_dir / "weights"
    ensure_dirs(weights)
    out_path = weights / ONNX_NAME

    model = YOLO(str(best))
    logger.info("Exporting %s -> ONNX (opset=%s)", best, opset)
    # Pass project/name to keep export inside project structure where possible
    model.export(format="onnx", opset=opset, dynamic=dynamic, project=str(PROJECT), name=NAME)

    default = weights / "best.onnx"
    if default.exists() and default != out_path:
        default.rename(out_path)

    logger.info("ONNX exported to %s", out_path)
    return out_path


if __name__ == "__main__":
    export_onnx()
