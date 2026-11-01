import os
import cv2
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_YAML = PROJECT_ROOT / "data/data/data.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "data/data_blur"

# نسبة الصور اللي هتتعملها motion blur
AUGMENT_PROBABILITY = 0.6

# blur أقوى وأوضح
KERNEL_SIZES = [15, 21, 25, 31]


def horizontal_motion_blur(image, kernel_size):
    """
    Strong horizontal motion blur
    Simulates fast conveyor movement
    """
    kernel = np.zeros((kernel_size, kernel_size))

    # horizontal motion line
    kernel[kernel_size // 2, :] = 1

    # normalize
    kernel = kernel / kernel.sum()

    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def process_split(split_name, image_dir_rel):
    image_dir = (INPUT_YAML.parent / image_dir_rel).resolve()
    label_dir = image_dir.parent.parent / split_name / "labels"

    output_image_dir = OUTPUT_ROOT / split_name / "images"
    output_label_dir = OUTPUT_ROOT / split_name / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.*"))

    print(f"\nProcessing {split_name}: {len(images)} images")

    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"

        # Copy original image
        shutil.copy2(
            img_path,
            output_image_dir / img_path.name
        )

        # Copy original label
        if label_path.exists():
            shutil.copy2(
                label_path,
                output_label_dir / label_path.name
            )

        # Create blurred version
        if random.random() < AUGMENT_PROBABILITY:
            kernel_size = random.choice(KERNEL_SIZES)

            blurred_img = horizontal_motion_blur(
                img,
                kernel_size
            )

            new_img_name = f"{img_path.stem}_motionblur.jpg"
            new_label_name = f"{img_path.stem}_motionblur.txt"

            cv2.imwrite(
                str(output_image_dir / new_img_name),
                blurred_img
            )

            if label_path.exists():
                shutil.copy2(
                    label_path,
                    output_label_dir / new_label_name
                )

    print(f"{split_name} completed.")


def main():
    print("Loading dataset config...")

    with open(INPUT_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    process_split("train", data_cfg["train"])
    process_split("valid", data_cfg["val"])
    process_split("test", data_cfg["test"])

    # Create new YAML for augmented dataset
    new_yaml = {
        "train": "./train/images",
        "val": "./valid/images",
        "test": "./test/images",
        "nc": data_cfg["nc"],
        "names": data_cfg["names"]
    }

    with open(OUTPUT_ROOT / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f)

    print("\nDone ✅")
    print(f"Augmented dataset saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()