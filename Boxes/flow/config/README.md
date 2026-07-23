# QC-SCM Configuration Reference

This directory contains the YAML configuration files for tuning the QC-SCM system.

## Files

| File | Purpose |
| :--- | :--- |
| `api.yaml` | Server binding (host, port, log level) and library log suppression. |
| `app.yaml` | CORS origins and default values for session telemetry/control fields. |
| `box_detector.yaml` | Box detection model path, confidence threshold, IoU threshold, and device. |
| `defect_detector.yaml` | Defect model path, tracking thresholds, stability (voting window), and rendering. |
| `stream.yaml` | Camera resolution, ROI gate geometry, and detection frame-skip throttle. |
| `firebase.yaml` | Firebase service account path and Realtime Database URL. **Never commit this file.** |
| `firebase.example.yaml` | Template for `firebase.yaml`. Copy and fill in your own values. |
| `firebase_config.json.example` | Alternative secrets format (JSON). |
| `firebase-service-account.example.json` | Example service account JSON structure for reference. |

## Adapting to a New Production Line

The most common adjustments when deploying to a new camera or conveyor setup:

1. **`stream.yaml`** — set `roi_width`, `roi_center_offset`, and `roi_top_y` to position the detection gate over the conveyor.
2. **`box_detector.yaml`** / **`defect_detector.yaml`** — update `model_path` to point to your trained TensorRT `.engine` files.
3. **`defect_detector.yaml`** — tune `stability.min_frames` and `stability.vote_threshold` for your conveyor speed and lighting conditions.

## Firebase Setup

```bash
# 1. Copy the example
cp firebase.example.yaml firebase.yaml

# 2. Place your service account JSON in this directory (gitignored)
cp /path/to/your-service-account.json ./your-service-account.json

# 3. Edit firebase.yaml
#    service_account_path: "your-service-account.json"
#    database_url: "https://your-project-default-rtdb.region.firebasedatabase.app"
```
