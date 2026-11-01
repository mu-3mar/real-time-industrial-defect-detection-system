# QC-SCM Configuration Guide

This directory contains the YAML configuration files required to tune the QC-SCM system for different industrial environments.

## ⚙️ Configuration Files

- **box_detector.yaml**: Box model path and confidence/IoU thresholds.
- **defect_detector.yaml**: Defect thresholds, stability voting, and tracking.
- **stream.yaml**: Camera settings, ROI gate dimensions, and frame throttling.
- **app.yaml**: Global application settings, CORS origins, and session defaults.

Detection results are written to a temporary JSON file per active session in
the runtime `.session_data/` directory and removed when that session closes.
Firebase was a previous integration and is not part of the current runtime.

## 🛠️ Customization
To adapt the system to a new production line, primarily modify `stream.yaml` to set the correct `roi_width` and `roi_center_offset` for your camera view.
