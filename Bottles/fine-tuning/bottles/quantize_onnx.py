import os
import logging
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from config import ROOT, ONNX_NAME, ONNX_INT8_NAME

os.chdir(str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def quantize(run_dir: Path, onnx_path: Path = None) -> Path:
    weights = run_dir / "weights"
    src = onnx_path or (weights / ONNX_NAME)
    dst = weights / ONNX_INT8_NAME
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return dst

if __name__ == "__main__":
    quantize()
