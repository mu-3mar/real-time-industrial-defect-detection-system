# pipeline/tracker.py
from typing import Dict
import numpy as np
from .config import cfg
from .api_client import send_product_to_api
from .helpers import compute_final_status_for_db

# in-memory QR-keyed tracks
tracks: Dict[str, Dict] = {}
# track fields:
# "box", "first_seen", "last_seen", "frames_seen", "defect_frames", "max_defects"

def create_new_track(qr: str, box: tuple, frame_index: int, status_now: str, defect_count: int):
    tracks[qr] = {
        "box": np.array(box, dtype=float),
        "first_seen": frame_index,
        "last_seen": frame_index,
        "frames_seen": 1,
        "defect_frames": 1 if status_now == "defect" else 0,
        "max_defects": defect_count,
    }

def update_track(qr: str, box: tuple, frame_index: int, status_now: str, defect_count: int):
    info = tracks[qr]
    info["box"] = np.array(box, dtype=float)
    info["last_seen"] = frame_index
    info["frames_seen"] += 1
    if status_now == "defect":
        info["defect_frames"] += 1
    if defect_count > info.get("max_defects", 0):
        info["max_defects"] = defect_count

def finalize_disappeared(frame_index: int):
    """
    For any track not seen for > MAX_DISAPPEAR frames -> compute final status and send to API,
    then delete the track.
    """
    for qr, info in list(tracks.items()):
        if frame_index - info["last_seen"] > cfg.MAX_DISAPPEAR:
            final_status = compute_final_status_for_db(info)
            send_product_to_api(qr, info, final_status)
            del tracks[qr]

def finalize_all_and_send():
    """
    Send all remaining tracks (used at program exit).
    """
    for qr, info in list(tracks.items()):
        final_status = compute_final_status_for_db(info)
        send_product_to_api(qr, info, final_status)
        del tracks[qr]
