# QC-SCM Runtime Detection Service

The `runtime` directory contains the high-performance inference engine and API
service for the QC-SCM system. It handles multiple camera streams, performs
dual-stage detection, and synchronizes results with the dashboard.

## 🏗️ Architecture Overview

The runtime service is built on a **Multi-Threaded Producer-Consumer** architecture:

1.  **Session Workers**: Independent threads for each camera stream.
2.  **Pipeline Manager**: Orchestrates the frame queue and inference dispatch.
3.  **Inference Thread**: A single, high-speed thread dedicated to GPU-bound YOLO processing.
4.  **Reporting Workers**: Asynchronous local result persistence and MJPEG streaming.

## 📂 Directory Structure

- **api/**: FastAPI server and REST endpoints.
- **config/**: YAML configuration files for detectors and stream.
- **core/**: Pipeline, state management, model loading, and sessions.
- **detectors/**: YOLO detector wrappers.
- **requirements/**: Runtime dependency files.
- **scripts/**: Development and deployment helpers.
- **utils/**: Geometry and visualization helpers.

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements/requirements.txt
```

### 2. Configuration
-   Tune `config/box_detector.yaml` and `config/defect_detector.yaml` for your specific environment.

Detection results are stored temporarily per active session in
`.session_data/` and are removed when the session ends. Firebase was the
previous cloud reporting integration and is not required by the current
runtime.

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
For project-wide usage, see the repository [README](../../../README.md).
