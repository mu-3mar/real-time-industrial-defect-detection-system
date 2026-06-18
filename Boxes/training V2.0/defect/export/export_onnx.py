import sys
import os
import logging
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import ROOT, PROJECT_DIR, PROJECT_NAME, ONNX_NAME, OPSET, EXPORTED_DIR

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def export_onnx(run_dir: Path) -> Path:
    """
    Exports the best trained PyTorch model to ONNX format.
    """
    weights = run_dir / "weights"
    best = weights / "best.pt"
    
    if not best.exists():
        logger.error(f"Model file not found: {best}")
        raise FileNotFoundError(f"Model file not found: {best}")

    # Copy best.pt to exported directory
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    target_pt = EXPORTED_DIR / "best.pt"
    try:
        shutil.copy(best, target_pt)
        logger.info(f"Copied best model to {target_pt}")
    except Exception as e:
        logger.error(f"Failed to copy best model: {e}")

    logger.info(f"Exporting model: {best}")
    model = YOLO(str(best))
    model.export(
        format="onnx",
        opset=OPSET,
        dynamic=True,
        project=str(PROJECT_DIR),
        name=PROJECT_NAME
    )
    
    default_onnx = weights / "best.onnx"
    
    if default_onnx.exists():
        target_onnx = EXPORTED_DIR / ONNX_NAME
        EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
        default_onnx.rename(target_onnx)
        logger.info(f"Renamed {default_onnx} to {target_onnx}")
    else:
        logger.warning(f"Expected exported file {default_onnx} not found")
        target_onnx = default_onnx # Fallback if rename failed / file absent
        
    return target_onnx

if __name__ == "__main__":
    # If run directly, assume default run directory
    run_dir = PROJECT_DIR / PROJECT_NAME
    try:
        export_onnx(run_dir)
    except Exception as e:
        logger.error(f"Export failed: {e}")
