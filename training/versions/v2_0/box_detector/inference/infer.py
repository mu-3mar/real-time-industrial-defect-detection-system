import argparse
import os
import sys
import cv2
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
from ultralytics import YOLO

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import PROJECT_DIR, PROJECT_NAME

# ============ CONFIG ============
# Use the trained model from the runs directory by default
DEFAULT_MODEL_PATH = PROJECT_DIR / PROJECT_NAME / "weights" / "best.pt"

BOX_CONF    = 0.4
IOU         = 0.6

# Create output directory for saved frames
OUTPUT_DIR = project_root / "output_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Single-stage inference: detect 'box' class (live or save mode)"
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to detection model (.pt)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image (if not using camera)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Live source: camera index (0, 1, ...) or path to a video file. Default: auto-detect camera",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames to disk instead of showing a live window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames to save when --save-frames is set (default=100)",
    )
    return parser.parse_args()


def _open_camera(preferred_index=None):
    """Open first working V4L index, or a specific index if given."""
    order = []
    if preferred_index is not None:
        order.append(preferred_index)
    raw = os.environ.get("CAMERA_INDEX", "").strip()
    if raw:
        try:
            env_idx = int(raw)
            if env_idx not in order:
                order.append(env_idx)
        except ValueError:
            pass
    order.extend(i for i in range(10) if i not in order)

    for idx in order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, idx
            cap.release()
    return None, None


def _open_source(source):
    """Open camera (index), video file, or auto-detect camera."""
    if source is None:
        return _open_camera()

    try:
        cam_idx = int(source)
    except (TypeError, ValueError):
        cam_idx = None

    if cam_idx is not None:
        cap, idx = _open_camera(cam_idx)
        if cap is not None:
            return cap, f"camera {idx}"
        return None, None

    path = Path(source)
    if not path.exists():
        print(f"Error: Source not found: {path}")
        return None, None

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap, str(path)
        cap.release()

    print(f"Error: Could not open source: {path}")
    return None, None


class TkLiveViewer:
    """Live camera preview using tkinter (works without OpenCV GUI/GTK)."""

    def __init__(self, title="Live Detection"):
        self._quit = False
        self._photo = None
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.bind("q", lambda _e: self.close())
        self.label = tk.Label(self.root)
        self.label.pack()

    def show(self, frame_bgr):
        if self._quit:
            return False
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self._photo)
        self.root.update()
        return True

    def close(self, *_args):
        self._quit = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    @property
    def running(self):
        return not self._quit


def draw_rounded_corners(img, x1, y1, x2, y2, color, r=12, t=2):
    """Draws a rectangle with rounded corners."""
    # Top-left
    cv2.line(img, (x1 + r, y1), (x1 + 2 * r, y1), color, t)
    cv2.line(img, (x1, y1 + r), (x1, y1 + 2 * r), color, t)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t)

    # Top-right
    cv2.line(img, (x2 - 2 * r, y1), (x2 - r, y1), color, t)
    cv2.line(img, (x2, y1 + r), (x2, y1 + 2 * r), color, t)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t)

    # Bottom-left
    cv2.line(img, (x1 + r, y2), (x1 + 2 * r, y2), color, t)
    cv2.line(img, (x1, y2 - 2 * r), (x1, y2 - r), color, t)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t)

    # Bottom-right
    cv2.line(img, (x2 - 2 * r, y2), (x2 - r, y2), color, t)
    cv2.line(img, (x2, y2 - 2 * r), (x2, y2 - r), color, t)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t)


def main():
    args = _parse_args()
    model_path = Path(args.model)

    print(f"Loading Model: {model_path}")
    
    try:
        model = YOLO(str(model_path), task="detect")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.image:
        # Process a single image
        print(f"Processing image: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not read image {args.image}")
            return
        
        results = model(
            frame,
            conf=BOX_CONF,
            iou=IOU,
            verbose=False
        )
        
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    color = (0, 255, 0)
                    draw_rounded_corners(frame, x1, y1, x2, y2, color)
                    cv2.putText(
                        frame,
                        "BOX",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
        
        output_path = OUTPUT_DIR / "output_image.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Output saved to: {output_path}")
        return

    # Use camera or video
    cap, source_label = _open_source(args.source)
    if cap is None:
        print("Error: Could not open any camera (tried indices 0–9, or CAMERA_INDEX if set)")
        return
    print(f"Using source: {source_label}")

    viewer = None
    if not args.save_frames:
        viewer = TkLiveViewer()
        print("Starting LIVE inference... Press 'q' or 'Esc' to exit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(
            frame,
            conf=BOX_CONF,
            iou=IOU,
            verbose=False
        )

        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    color = (0, 255, 0)
                    draw_rounded_corners(frame, x1, y1, x2, y2, color)
                    cv2.putText(
                        frame,
                        "BOX",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        if viewer is not None:
            if not viewer.show(frame):
                break
        else:
            if frame_count >= args.max_frames:
                break
            output_frame_path = OUTPUT_DIR / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_frame_path), frame)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Saved {frame_count}/{args.max_frames} frames")

    cap.release()
    if viewer is not None:
        viewer.close()
        print("Live session ended.")
    else:
        print(f"Done! Saved {frame_count} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
