import sys
import os
import logging
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import ROOT, PROJECT_DIR, PROJECT_NAME, FINAL_METRICS_DIR
from training.train import train
from export.export_onnx import export_onnx
from export.quantize_onnx import quantize
from utils.utils import collect_final_metrics

# Ensure we are working from root
os.chdir(str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting full pipeline...")
    
    # 1. Train
    logger.info("Step 1: Training")
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Proceeding to export best model found so far...")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Decide if we want to proceed or exit. Usually if train errors out, we might not have a model.
        # But if it's just an interruption, we proceed.
        # For other errors, let's re-raise or return.
        # For now, let's try to proceed if best.pt exists.
        pass
    
    run_dir = PROJECT_DIR / PROJECT_NAME
    
    # Check if we have anything to export
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    
    if not best_pt.exists():
        logger.error(f"No 'best.pt' found in {weights_dir}. Cannot proceed with export.")
        return

    # 2. Export
    logger.info("Step 2: Exporting to ONNX")
    onnx_path = export_onnx(run_dir)
    
    # 3. Quantize
    logger.info("Step 3: Quantizing")
    quantize(run_dir, onnx_path)
    
    # 4. Collect Metrics
    logger.info("Step 4: Collecting Metrics")
    collect_final_metrics(run_dir, FINAL_METRICS_DIR)
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
