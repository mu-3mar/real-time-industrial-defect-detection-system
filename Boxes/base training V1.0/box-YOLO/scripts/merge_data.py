import os
import shutil
from pathlib import Path

# ============ CONFIG ============
DATASETS = [
    "data/data 1",
    "data/data 2",
    "data/data 3",
]

OUTPUT_DIR = "data/data"
SPLITS = ["train", "valid", "test"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]
CLASS_ID = 0  # unify class

# ============ CREATE STRUCTURE ============
for split in SPLITS:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

# ============ MERGE ============
img_counter = 0

for dataset in DATASETS:
    for split in SPLITS:
        img_dir = Path(dataset) / split / "images"
        lbl_dir = Path(dataset) / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            new_img_name = f"img_{img_counter}{img_path.suffix}"
            new_lbl_name = f"img_{img_counter}.txt"

            shutil.copy(
                img_path,
                Path(OUTPUT_DIR) / split / "images" / new_img_name
            )

            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    lines = f.readlines()

                with open(Path(OUTPUT_DIR) / split / "labels" / new_lbl_name, "w") as f:
                    for line in lines:
                        parts = line.strip().split()
                        parts[0] = str(CLASS_ID)  # force class = 0
                        f.write(" ".join(parts) + "\n")

            img_counter += 1

print(f"✅ Merge finished — total images: {img_counter}")
