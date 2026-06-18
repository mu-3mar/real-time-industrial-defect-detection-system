import logging
import shutil
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import DATA_DIR, DATA_YAML
from utils.utils import count_images, write_data_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Prepared dataset from the collect pipeline
SOURCE_DATASET = Path("/home/mu-3mar/projects/collect/data/DEFECT")
SPLITS = ("train", "val")


def _copy_split(split: str) -> tuple[int, int]:
    copied_images = 0
    copied_labels = 0

    for kind in ("images", "labels"):
        src = SOURCE_DATASET / split / kind
        dst = DATA_DIR / split / kind
        dst.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            logger.warning("Missing source folder: %s", src)
            continue

        for file_path in src.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                shutil.copy2(file_path, dst / file_path.name)
                if kind == "images":
                    copied_images += 1
                else:
                    copied_labels += 1

    return copied_images, copied_labels


def main():
    if not SOURCE_DATASET.exists():
        logger.error("Source dataset not found: %s", SOURCE_DATASET)
        return 1

    logger.info("Copying dataset from %s -> %s", SOURCE_DATASET, DATA_DIR)

    total_images = 0
    total_labels = 0
    for split in SPLITS:
        images, labels = _copy_split(split)
        total_images += images
        total_labels += labels
        logger.info("%s: %d images, %d labels", split, images, labels)

    write_data_yaml(DATA_DIR, DATA_YAML)
    logger.info("Updated %s", DATA_YAML)

    train_count = count_images(DATA_DIR / "train" / "images")
    val_count = count_images(DATA_DIR / "val" / "images")

    if train_count == 0 or val_count == 0:
        logger.error(
            "Still no images in defect/data. Put source images in %s/images "
            "with matching labels, then run:\n"
            "  python %s/prepare_dataset.py\n"
            "  python scripts/prepare_data.py",
            SOURCE_DATASET,
            SOURCE_DATASET,
        )
        return 1

    logger.info("Dataset ready: train=%d, val=%d", train_count, val_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
