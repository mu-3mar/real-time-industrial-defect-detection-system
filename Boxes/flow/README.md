# QC-SCM Flow: Runtime Detection Service

The `flow` directory contains the high-performance inference engine and REST API for the QC-SCM system. It handles multiple real-time camera streams, performs dual-stage AI detection, and synchronizes results with Firebase.

## 🏗️ Architecture

The runtime service is built on a **Multi-Threaded Producer-Consumer** architecture:

1. **Session Workers** — one thread per camera stream, feeds frames into the shared queue.
2. **Pipeline Manager** — orchestrates the frame queue and dispatches to the single inference thread.
3. **Inference Thread** — GPU-bound; runs `Pipeline.run_step()` on one frame at a time.
4. **Result Consumer** — stores latest annotated frames for MJPEG streaming.
5. **Firebase Worker** — publishes detection events to Realtime Database without blocking inference.

## 📂 Directory Structure

```
flow/
├── main.py                  # Entry point: reads api.yaml, starts uvicorn
├── api/
│   ├── api_server.py        # FastAPI app, endpoints, lifespan hooks
│   └── README.md
├── config/
│   ├── api.yaml             # Server host/port/log-level
│   ├── app.yaml             # CORS origins, session defaults
│   ├── box_detector.yaml    # Box model path and thresholds
│   ├── defect_detector.yaml # Defect model path, stability, tracking
│   ├── stream.yaml          # Camera resolution, ROI, throttle
│   ├── firebase.yaml        # Firebase credentials + DB URL (gitignored)
│   └── firebase.example.yaml
├── core/
│   ├── pipeline.py          # Per-frame inference, tracking, canvas rendering
│   ├── pipeline_manager.py  # Producer-consumer threading, MJPEG frame store
│   ├── pipeline_diagnostics.py
│   ├── session_manager.py   # Session lifecycle management
│   ├── session_worker.py    # Per-session camera feeder thread
│   ├── state.py             # Temporal voting, defect lock, track recovery
│   ├── stream.py            # OpenCV V4L2 camera reader with background thread
│   ├── model_loader.py      # Singleton YOLO model loader + warmup
│   ├── firebase_client.py   # Firebase Admin SDK wrapper
│   └── device_manager.py   # CUDA/MPS/CPU device resolution
├── detectors/
│   └── detector.py          # Thin YOLO inference wrapper
├── utils/
│   ├── geometry.py          # IoU + bbox exponential smoothing
│   └── visualizer.py        # Canvas drawing: ROI lines, boxes, stats panel
├── requirements/
│   └── requirements.txt
└── scripts/
    └── run_dev.sh
```

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements/requirements.txt
```

### 2. Configuration

```bash
# Firebase credentials (required)
cp config/firebase.example.yaml config/firebase.yaml
# Edit config/firebase.yaml: set service_account_path and database_url

# Tune ROI and detection thresholds for your environment
# Edit config/stream.yaml: roi_width, roi_center_offset, roi_top_y
# Edit config/box_detector.yaml / config/defect_detector.yaml
```

### 3. Execution

```bash
python main.py
```

## 📡 API Endpoints

See [`api/README.md`](api/README.md) for the full endpoint reference.

| Method | Path | Description |
| :----- | :--- | :---------- |
| `POST` | `/api/reports/open` | Start a detection session. |
| `POST` | `/api/reports/close` | Stop a session. |
| `GET`  | `/api/reports` | List active sessions. |
| `GET`  | `/api/health` | Health check. |
| `GET`  | `/video_feed?report_id=<id>` | MJPEG live stream. |
