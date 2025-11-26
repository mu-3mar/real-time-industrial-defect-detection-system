# ========== WORKING DIRECTORY GUARD ==========
import os
from pathlib import Path
from config import ROOT, PROJECT
PROJECT = Path(PROJECT).resolve()
os.makedirs(PROJECT, exist_ok=True)
os.chdir(str(ROOT))
# ============================================

import logging
from onnxruntime.quantization import quantize_dynamic, QuantType
from config import ONNX_NAME, ONNX_INT8_NAME
from utils import get_latest_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def quantize():
    run_dir = get_latest_run(str(PROJECT))
    if run_dir is None:
        raise RuntimeError("No run directory found")

    src = run_dir / "weights" / ONNX_NAME
    dst = run_dir / "weights" / ONNX_INT8_NAME

    if not src.exists():
        raise FileNotFoundError(f"ONNX file not found at {src}")

    logger.info("Quantizing ONNX %s -> %s", src, dst)
    quantize_dynamic(model_input=str(src), model_output=str(dst), weight_type=QuantType.QUInt8)
    logger.info("Quantized ONNX saved at %s", dst)
    return dst


if __name__ == "__main__":
    quantize()
