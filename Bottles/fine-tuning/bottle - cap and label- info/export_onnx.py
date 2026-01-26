import os
import logging
from ultralytics import YOLO
from pathlib import Path
from config import ROOT, PROJECT, NAME, ONNX_NAME, OPSET

os.chdir(str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def export_onnx(run_dir: Path) -> Path:
    weights = run_dir / "weights"
    best = weights / "best.pt"
    model = YOLO(str(best))
    model.export(
        format="onnx",
        opset=OPSET,
        dynamic=True,
        project=str(PROJECT),
        name=NAME
    )
    default = weights / "best.onnx"
    out = weights / ONNX_NAME
    if default.exists():
        default.rename(out)
    return out

if __name__ == "__main__":
    export_onnx()
