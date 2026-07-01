import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def ensure_dirs(*paths: Path):
    """
    Ensure that the specified directories exist.
    Creates them if they do not.
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def prune_weights(weights_dir: Path):
    """
    Remove all .pt weight files except 'best.pt' and 'last.pt' to save space.
    """
    for f in weights_dir.glob("*.pt"):
        if f.name not in ("best.pt", "last.pt"):
            f.unlink(missing_ok=True)
            logger.info(f"Removed unused weight file: {f.name}")

def collect_final_metrics(run_dir: Path, dest_dir: Path):
    """
    Copy important training metrics and plots to a destination directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "results.png",
        "confusion_matrix*.png",
        "*curve*.png",
        "results.csv"
    ]
    for pat in patterns:
        for f in run_dir.glob(pat):
            dst = dest_dir / f.name
            if not dst.exists():
                shutil.copy(f, dst)
                logger.info(f"Copied metric: {f.name} -> {dest_dir}")

def check_and_download_model(model_path: Path):
    """
    Checks if the model exists at the given path.
    If not, attempts to download it from Ultralytics assets.
    """
    if model_path.exists():
        return

    logger.info(f"Model not found at {model_path}. Attempting download...")
    
    # Ensure parent directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct URL for standard YOLOv8 models
    # Note: This works for standard YOLOv8 models hosted by Ultralytics.
    # Filename should be like 'yolov8n.pt' or 'yolo26n.pt'.
    model_name = model_path.name
    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}"
    
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {model_name} to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name} to {model_path}. Error: {e}")
        logger.info("Ultralytics YOLO() class might check internal logic as fallback, but output path won't be guaranteed.")

