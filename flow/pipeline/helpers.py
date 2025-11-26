# pipeline/helpers.py
# keeps the final-status decision logic (shared)
from datetime import datetime

def compute_final_status_for_db(info: dict) -> str:
    """
    Decision rule used to decide final 'defect' or 'ok' for stored product.
    Keeps the same logic as original single-file script.
    """
    frames = info.get("frames_seen", 0)
    df = info.get("defect_frames", 0)
    max_defects = info.get("max_defects", 0)

    # Very short track: 1-3 frames
    if frames <= 3:
        return "defect" if df > 0 else "ok"

    if frames == 0:
        return "ok"

    ratio = df / frames

    # If a significant portion of frames had defects -> defect
    if ratio >= 0.3:
        return "defect"

    # If never saw a defect -> ok
    if df == 0:
        return "ok"

    # If in any frame we saw multiple defects -> defect
    if max_defects >= 2:
        return "defect"

    # Small noise: treat as ok
    return "ok"
