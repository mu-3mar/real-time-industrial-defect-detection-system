# export_onnx.py
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
from utils import get_latest_run
from pathlib import Path
from config import ONNX_NAME, NAME, OPSET

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def export_onnx(run_dir: Path = None, opset: int = OPSET, dynamic: bool = True) -> Path:
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
            logger.info("best.pt not found; falling back to last.pt for ONNX export.")
        else:
            raise RuntimeError("No best.pt or last.pt found for ONNX export")

    out_path = weights / ONNX_NAME
    logger.info("Exporting model %s to ONNX -> %s", best, out_path)

    model = YOLO(str(best))
    # pass project/name to keep outputs inside run, ultralytics will write best.onnx into weights
    model.export(format="onnx", opset=opset, dynamic=dynamic, project=str(PROJECT), name=NAME)

    # normalize filename: rename default best.onnx -> desired name
    default = weights / "best.onnx"
    if default.exists() and default != out_path:
        default.rename(out_path)
    else:
        # if ultralytics produced another onnx file, pick first
        cand = list(weights.glob("*.onnx"))
        if cand:
            cand[0].rename(out_path)

    if not out_path.exists():
        raise RuntimeError("ONNX export did not produce expected file: " + str(out_path))

    logger.info("ONNX exported to %s", out_path)
    return out_path

if __name__ == "__main__":
    export_onnx()
