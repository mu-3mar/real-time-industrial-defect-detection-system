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

from configs.config import (
    ROOT,
    PROJECT_DIR,
    PROJECT_NAME,
    EXPORTED_DIR,
    ENGINE_NAME,
    ENGINE_DEVICE,
    ENGINE_HALF,
)

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_engine(run_dir: Path) -> Path:
    """
    Exports the best trained PyTorch model to TensorRT engine.
    """
    weights = run_dir / "weights"
    best = weights / "best.pt"

    if not best.exists():
        raise FileNotFoundError(f"Model file not found: {best}")

    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Exporting TensorRT engine from {best} with device={ENGINE_DEVICE}, half={ENGINE_HALF}"
    )

    model = YOLO(str(best))
    model.export(
        format="engine",
        device=ENGINE_DEVICE,
        half=ENGINE_HALF,
        project=str(PROJECT_DIR),
        name=PROJECT_NAME,
    )

    default_engine = weights / "best.engine"
    target_engine = EXPORTED_DIR / ENGINE_NAME

    if default_engine.exists():
        if target_engine.exists():
            target_engine.unlink()
        default_engine.rename(target_engine)
        logger.info(f"Saved engine to {target_engine}")
        return target_engine

    logger.warning(f"Expected exported engine file not found at {default_engine}")
    return default_engine


if __name__ == "__main__":
    run_dir = PROJECT_DIR / PROJECT_NAME
    try:
        export_engine(run_dir)
    except Exception as e:
        logger.error(f"TensorRT export failed: {e}")
