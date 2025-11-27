def compute_final_status_for_db(info: dict) -> str:
    """Compute final status (ok/defect) based on track statistics."""
    frames = info.get("frames_seen", 0)
    df = info.get("defect_frames", 0)
    max_defects = info.get("max_defects", 0)

    if frames <= 3:
        return "defect" if df > 0 else "ok"

    if frames == 0:
        return "ok"

    ratio = df / frames

    if ratio >= 0.3:
        return "defect"

    if df == 0:
        return "ok"

    if max_defects >= 2:
        return "defect"

    return "ok"
