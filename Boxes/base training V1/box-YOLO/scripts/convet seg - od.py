from pathlib import Path

DATA_PATH = "data/data 1-1"  # ← غير ده بس
SPLITS = ["train", "valid", "test"]

def convert_seg_to_box(label_file):
    new_lines = []

    with open(label_file, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            coords = parts[1:]

            # لو أصلاً box (5 values) سيبه
            if len(coords) == 4:
                new_lines.append(line)
                continue

            # segmentation → box
            if len(coords) >= 6:
                xs = coords[0::2]
                ys = coords[1::2]

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                x_c = (x_min + x_max) / 2
                y_c = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min

                new_lines.append(
                    f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                )

    return new_lines


# ============ RUN ============
total_files = 0

for split in SPLITS:
    label_dir = Path(DATA_PATH) / split / "labels"
    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):
        new_content = convert_seg_to_box(label_file)
        if new_content:
            with open(label_file, "w") as f:
                f.writelines(new_content)
            total_files += 1

print(f"✅ Done. Processed {total_files} label files")
