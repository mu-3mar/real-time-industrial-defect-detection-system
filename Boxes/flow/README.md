# QC-SCM Flow: Runtime Detection Service

The `flow` directory contains the high-performance inference engine and API service for the QC-SCM system. It is designed to handle multiple real-time camera streams, perform dual-stage AI detection, and synchronize results with a cloud dashboard.

## 🏗️ Architecture Overview

The runtime service is built on a **Multi-Threaded Producer-Consumer** architecture:

1.  **Session Workers**: Independent threads for each camera stream.
2.  **Pipeline Manager**: Orchestrates the frame queue and inference dispatch.
3.  **Inference Thread**: A single, high-speed thread dedicated to GPU-bound YOLO processing.
4.  **Reporting Workers**: Asynchronous threads for Firebase cloud publishing and MJPEG streaming.

## 📂 Directory Structure

- **[api/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/api/)**: FastAPI server and REST endpoints.
- **[config/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/config/)**: YAML configuration files for detectors, stream, and Firebase.
- **[core/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/)**: Core logic including the Pipeline, State Management, and Model Loading.
- **[detectors/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/detectors/)**: Wrapper classes for YOLO models.
- **[requirements/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/requirements/)**: Dependency files.
- **[scripts/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/scripts/)**: Utility scripts for development and deployment.
- **[utils/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/utils/)**: Geometry helpers and visualization tools.

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements/requirements.txt
```

### 2. Configuration
-   Edit `config/firebase.yaml` with your database URL and credentials path.
-   Tune `config/box_detector.yaml` and `config/defect_detector.yaml` for your specific environment.

### 3. Execution
```bash
python main.py
```

## 📡 API Endpoints

-   `POST /api/reports/open`: Start a new detection session.
-   `POST /api/reports/close`: End an active session.
-   `GET /api/reports`: List active sessions.
-   `GET /api/session/{report_id}/stream`: Real-time MJPEG annotated stream.

---
For more details, see the **[System Architecture Documentation](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/System_Architecture_Documentation.md)**.
