# QC-SCM Configuration Guide

This directory contains the YAML configuration files required to tune the QC-SCM system for different industrial environments.

## ⚙️ Configuration Files

- **[box_detector.yaml](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/config/box_detector.yaml)**: Paths to the box detection model and its confidence/IoU thresholds.
- **[defect_detector.yaml](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/config/defect_detector.yaml)**: Tuning parameters for defect detection, including **stability (voting window)** and **tracking thresholds**.
- **[stream.yaml](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/config/stream.yaml)**: Camera settings, ROI gate dimensions, and frame skip (throttling) parameters.
- **[firebase.yaml](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/config/firebase.yaml)**: Firebase Realtime Database URL and service account path.
- **[app.yaml](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/config/app.yaml)**: Global application settings, CORS origins, and session defaults.

## 🛠️ Customization
To adapt the system to a new production line, primarily modify `stream.yaml` to set the correct `roi_width` and `roi_center_offset` for your camera view.
