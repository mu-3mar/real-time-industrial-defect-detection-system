import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ========== PATHS ==========
# Assuming this script is run from project root, or we use relative paths
# If run from scripts/ folder, we need to adjust or use absolute paths via ROOT
# For simplicity, we assume running from root or we find root relative to this file.

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Input Datasets
DATASET1 = project_root / "data" / "data 1"
DATASET2 = project_root / "data" / "data 2"

# Output Dataset
OUT = project_root / "data" / "data"

SPLITS = ["train", "valid", "test"]

def prepare_dirs():
    """Creates directory structure for merged dataset."""
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

def copy_split(src_root, split, prefix):
    """
    Copies images and labels from source to destination with a prefix 
    to avoid filename collisions.
    """
    img_src = src_root / split / "images"
    lbl_src = src_root / split / "labels"

    if not img_src.exists():
        logger.warning(f"Source image directory not found: {img_src}")
        return

    logger.info(f"Processing {src_root.name}/{split}...")
    
    count = 0
    for img in img_src.iterdir():
        if img.name.startswith("."): continue # Skip hidden files
        
        new_name = f"{prefix}_{img.name}"
        shutil.copy(img, OUT / split / "images" / new_name)

        # Copy corresponding label if it exists
        label_name = img.stem + ".txt"
        label = lbl_src / label_name
        if label.exists():
            shutil.copy(label, OUT / split / "labels" / f"{prefix}_{label_name}")
        
        count += 1
    logger.info(f"Copied {count} files from {src_root.name}/{split}")

def main():
    if not DATASET1.exists() or not DATASET2.exists():
        logger.error(f"One or both datasets not found:\n{DATASET1}\n{DATASET2}")
        return

    logger.info("Preparing output directories...")
    prepare_dirs()

    # dataset1 -> train, valid, test
    logger.info("Merging Dataset 1...")
    for s in ["train", "valid", "test"]:
        copy_split(DATASET1, s, "d1")

    # dataset2 -> train, valid only (as per original logic)
    logger.info("Merging Dataset 2...")
    for s in ["train", "valid", "test"]:  # Original code had test in loop but comment said "only train, valid"
        # Checking original code logic:
        # # dataset2 -> train, valid فقط
        # for s in ["train", "valid", "test"]:
        #     copy_split(DATASET2, s, "d2")
        # It actually looped over test too, despite the comment? 
        # Or maybe the folder didn't exist for test.
        # I will keep the loop as is to preserve original behavior, just catching errors if dir doesn't exist.
        copy_split(DATASET2, s, "d2")

    logger.info("Merge completed successfully.")

if __name__ == "__main__":
    main()
