import cv2
import argparse
from .config import cfg
from .detector import Detector
from .utils import expand_box, read_qr, draw_box, highlight_box
from .tracker import create_new_track, update_track, finalize_disappeared, finalize_all_and_send, tracks
from .helpers import compute_final_status_for_db


def process_frame(frame, carton_detector: Detector, defect_detector: Detector, frame_index: int):
    """Process single frame: detect cartons, detect defects, track, and visualize."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    boxes, scores, classes = carton_detector.infer(frame, conf=cfg.CARTON_CONF)
    if boxes.size == 0:
        finalize_disappeared(frame_index)
        return annotated

    for box, score in zip(boxes, scores):
        if float(score) < cfg.CARTON_CONF:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), w, h, cfg.EXPAND_RATIO)
        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue

        dboxes, dscores, dclasses = defect_detector.infer(crop, conf=cfg.DEFECT_CONF)
        defect_count = 0
        defect_boxes_global = []
        if dboxes.size != 0:
            for dbox, dscore in zip(dboxes, dscores):
                if float(dscore) < cfg.DEFECT_CONF:
                    continue
                dx1, dy1, dx2, dy2 = [int(v) for v in dbox]
                gx1 = ex1 + dx1
                gy1 = ey1 + dy1
                gx2 = ex1 + dx2
                gy2 = ey1 + dy2
                defect_boxes_global.append((gx1, gy1, gx2, gy2))
                defect_count += 1

        status_now = "defect" if defect_count > 0 else "ok"

        qr = read_qr(crop)
        if qr is None:
            continue

        if qr not in tracks:
            create_new_track(qr, (x1, y1, x2, y2), frame_index, status_now, defect_count)
        else:
            update_track(qr, (x1, y1, x2, y2), frame_index, status_now, defect_count)

        info = tracks[qr]
        final_display = compute_final_status_for_db(info)
        color = (0, 0, 255) if final_display == "defect" else (0, 255, 0)
        draw_box(annotated, info["box"], color, f"{qr} {final_display}")

        if final_display == "defect":
            for gx1, gy1, gx2, gy2 in defect_boxes_global:
                gx1 = max(0, min(gx1, w - 1))
                gy1 = max(0, min(gy1, h - 1))
                gx2 = max(0, min(gx2, w - 1))
                gy2 = max(0, min(gy2, h - 1))
                if gx2 > gx1 and gy2 > gy1:
                    highlight_box(annotated, (gx1, gy1, gx2, gy2))

    finalize_disappeared(frame_index)
    return annotated

def main():
    """Run real-time QC pipeline with video capture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--carton-model", default=cfg.CARTON_MODEL)
    parser.add_argument("--defect-model", default=cfg.DEFECT_MODEL)
    parser.add_argument("--cam", type=int, default=cfg.CAMERA_INDEX)
    args = parser.parse_args()

    carton_detector = Detector(args.carton_model)
    defect_detector = Detector(args.defect_model)

    cap = cv2.VideoCapture(args.cam)
    frame_idx = 0

    print("[INFO] Session ID:", cfg.SESSION_ID)
    print("[INFO] Press Q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, carton_detector, defect_detector, frame_idx)

        cv2.imshow("QC Pipeline", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    finalize_all_and_send()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
