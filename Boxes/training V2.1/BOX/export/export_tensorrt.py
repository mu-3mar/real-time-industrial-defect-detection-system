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

from configs.config import ROOT, PROJECT_DIR, PROJECT_NAME, TENSORRT_NAME, EXPORTED_DIR, IMG_SIZE

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_tensorrt(onnx_path: Path = None, run_dir: Path = None) -> Path:
    """
    Exports the trained model to TensorRT format (.engine).
    Can use either an ONNX file path or a training run directory.
    """
    if onnx_path is None and run_dir is None:
        raise ValueError("Either onnx_path or run_dir must be provided")

    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which model to use: prefer ONNX path, then best.pt from run dir
    if onnx_path and onnx_path.exists():
        logger.info(f"Using ONNX model for TensorRT export: {onnx_path}")
        model = YOLO(str(onnx_path))
    else:
        weights = run_dir / "weights"
        best = weights / "best.pt"
        if not best.exists():
            logger.error(f"Model file not found: {best}")
            raise FileNotFoundError(f"Model file not found: {best}")
        logger.info(f"Using PyTorch model for TensorRT export: {best}")
        model = YOLO(str(best))

    logger.info("Exporting to TensorRT...")
    # Export to TensorRT format
    model.export(
        format="engine",
        imgsz=IMG_SIZE,
        dynamic=True,  # Allow dynamic batch size
        simplify=True,  # Simplify ONNX before TensorRT conversion
        workspace=4,  # Workspace size in GB (adjust as needed)
        project=str(PROJECT_DIR),
        name=PROJECT_NAME
    )

    # Find the exported .engine file
    if onnx_path and onnx_path.exists():
        default_engine = onnx_path.parent / f"{onnx_path.stem}.engine"
    else:
        weights = run_dir / "weights"
        default_engine = weights / "best.engine"

    target_engine = EXPORTED_DIR / TENSORRT_NAME

    if default_engine.exists():
        import shutil
        shutil.move(str(default_engine), str(target_engine))
        logger.info(f"Renamed and moved TensorRT engine to {target_engine}")
    else:
        # Check if it was exported directly to project dir
        candidates = list(PROJECT_DIR.glob(f"**/*.engine"))
        if candidates:
            default_engine = max(candidates, key=lambda p: p.stat().st_mtime)
            import shutil
            shutil.move(str(default_engine), str(target_engine))
            logger.info(f"Renamed and moved TensorRT engine to {target_engine}")
        else:
            logger.warning(f"Expected exported TensorRT file not found!")
            target_engine = default_engine  # Fallback

    return target_engine


if __name__ == "__main__":
    from configs.config import PROJECT_DIR, PROJECT_NAME
    run_dir = PROJECT_DIR / PROJECT_NAME
    try:
        export_tensorrt(run_dir=run_dir)
    except Exception as e:
        logger.error(f"TensorRT export failed: {e}")
