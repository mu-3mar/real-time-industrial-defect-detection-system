import sys
import os
import logging
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import ROOT, ONNX_NAME, ONNX_INT8_NAME, EXPORTED_DIR

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def quantize(run_dir: Path, onnx_path: Path = None) -> Path:
    """
    Quantizes an ONNX model to INT8 dynamic quantization.
    """
    weights = run_dir / "weights"
    src = onnx_path or (EXPORTED_DIR / ONNX_NAME)
    dst = EXPORTED_DIR / ONNX_INT8_NAME
    
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        logger.error(f"Source ONNX file not found: {src}")
        raise FileNotFoundError(f"Source ONNX file not found: {src}")

    logger.info(f"Quantizing model: {src} -> {dst}")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    
    logger.info("Quantization complete.")
    return dst

if __name__ == "__main__":
    from configs.config import PROJECT_DIR, PROJECT_NAME
    run_dir = PROJECT_DIR / PROJECT_NAME
    try:
        quantize(run_dir)
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
