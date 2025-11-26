# quantize_onnx.py
# ====== WORKING DIRECTORY GUARD ======
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# =====================================

import logging
from onnxruntime.quantization import quantize_dynamic, QuantType
from utils import get_latest_run
from config import ONNX_NAME, ONNX_INT8_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def quantize(run_dir: Path = None, onnx_path: Path = None) -> Path:
    if run_dir is None:
        run_dir = get_latest_run(str(PROJECT))
    if run_dir is None:
        raise RuntimeError("No run_dir found")

    weights = run_dir / "weights"
    if onnx_path is None:
        onnx_path = weights / ONNX_NAME

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    out_path = weights / ONNX_INT8_NAME
    logger.info("Quantizing %s -> %s", onnx_path, out_path)

    quantize_dynamic(model_input=str(onnx_path), model_output=str(out_path), weight_type=QuantType.QUInt8)
    if not out_path.exists():
        raise RuntimeError("Quantization failed: output not found")

    logger.info("Quantized ONNX saved at %s", out_path)
    return out_path

if __name__ == "__main__":
    quantize()
