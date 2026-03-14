# QC-SCM — Quality Control (Flow + Training)

QC-SCM is an AI-driven quality inspection system for manufacturing:

- **Training**: train two YOLO models (box detection + defect detection), export to ONNX, and (optionally) quantize.
- **Flow runtime**: run a FastAPI service that opens **reports** (`report_id`), streams annotated video over **WebRTC**, and publishes detection events to **Firebase Realtime Database**.

Detection events are minimal by design: under each `report_id`, every event contains only:

- `defect` (boolean)
- `timestamp` (ISO 8601 UTC string)

---

## Repository overview

| Part | Purpose |
|------|--------|
| `Boxes/flow/` | Runtime detection service (FastAPI + WebRTC + Firebase publishing). |
| `Boxes/training/` | Training pipelines (box-YOLO + defect-YOLO): train → export ONNX → quantize → metrics. |

---

## Quick start (Flow)

1. Configure Firebase + WebRTC secrets (see **Configuration (`Boxes/flow/config/`)** below).
2. Start the server:

```bash
cd Boxes/flow
python3 main.py
```

---

## Installation

### Prerequisites

- **Python**: 3.10+
- **Optional GPU**: CUDA-capable GPU (training/inference may still run on CPU)
- **Linux camera access**: if using `/dev/video*`, ensure permissions are correct

### Install options

- **Full dev install (repo root)**:

```bash
pip install -e .
```

- **Minimal Flow runtime install**:

```bash
pip install -r Boxes/flow/requirements/requirements.txt
```

---

## Repository structure (high level)

```
QC-SCM/
├── README.md
├── pyproject.toml
├── .gitignore
├── assets/
└── Boxes/
    ├── flow/                     # Runtime detection service
    └── training/                 # Training pipelines (box-YOLO + defect-YOLO)
```

---

## Boxes/flow — Runtime detection service

### What it does

- Opens/closes **reports** via REST (`report_id`)
- Runs a two-stage pipeline (box → defect) per report
- Streams annotated frames via **aiortc/WebRTC**
- Pushes detection events to **Firebase Realtime Database**

### Run

From repo root:

```bash
./Boxes/flow/scripts/run_dev.sh
```

Or directly:

```bash
cd Boxes/flow
python3 main.py
```

Endpoints:

- **API**: `http://localhost:8000`
- **OpenAPI docs**: `http://localhost:8000/docs`
- **Health**: `http://localhost:8000/api/health`

### Configuration (`Boxes/flow/config/`)

| File | Purpose |
|------|--------|
| `api.yaml` | Uvicorn host/port/log level (used by `main.py`). |
| `app.yaml` | Optional CORS configuration. |
| `webrtc.yaml` | STUN/TURN + TURN **secret** + `webrtc_mode` (**gitignored**). |
| `webrtc.example.yaml` | Template for `webrtc.yaml`. |
| `firebase.yaml` | Firebase `service_account_path` + `database_url`. |
| `firebase.example.yaml` | Template for `firebase.yaml`. |
| `firebase-service-account.json` | Firebase service account key (**gitignored**). |
| `firebase-service-account.example.json` | Template JSON structure. |
| `firebase_config.json` | Optional fallback for `database_url` (**gitignored**). |
| `firebase_config.json.example` | Template for the fallback file. |
| `box_detector.yaml` | box model path + thresholds. |
| `defect_detector.yaml` | defect model path + thresholds + stability settings. |
| `stream.yaml` | frame size + detection throttling. |

#### Secrets & gitignore

Configuration is loaded **only from files in `Boxes/flow/config/`** (no environment variables or `.env`). The repo ignores sensitive files:

- `Boxes/flow/config/firebase-service-account.json`
- `Boxes/flow/config/webrtc.yaml` (TURN secret)
- `Boxes/flow/config/firebase_config.json` (if used for `database_url`)

### Flow API (current contract)

#### Open a report

`POST /api/reports/open`

Body:

```json
{ "report_id": "r-123", "camera_source": 0, "production_line_id": "line-1" }
```

Notes:

- **Idempotent per `production_line_id`**: if a report is already open for the line, the API returns success (does not error).

Example:

```bash
curl -sS -X POST "http://localhost:8000/api/reports/open" \
  -H "content-type: application/json" \
  -d '{"report_id":"r-123","camera_source":0,"production_line_id":"line-1"}'
```

#### Close a report

`POST /api/reports/close`

Body:

```json
{ "report_id": "r-123" }
```

Notes:

- **Idempotent**: closing an already-closed report returns success.

Example:

```bash
curl -sS -X POST "http://localhost:8000/api/reports/close" \
  -H "content-type: application/json" \
  -d '{"report_id":"r-123"}'
```

#### List active reports

`GET /api/reports`

Response:

```json
[{ "report_id": "r-123", "viewers_count": 1 }]
```

Example:

```bash
curl -sS "http://localhost:8000/api/reports"
```

#### Health

`GET /api/health` → `{ "status": "healthy", "active_reports": 1 }`

#### WebRTC client config

`GET /api/config`

Returns:

- `webrtc.iceServers`: includes STUN and **temporary TURN credentials**
- `webrtc.webrtc_mode`: `auto | direct | stun | relay`

#### WebRTC offer/answer

`POST /webrtc/offer` with `{ sdp, type, report_id }`

### Firebase Realtime Database structure (current)

Events are written to:

```
{report_id}/{detection_id}
```

Payload:

```json
{ "defect": true, "timestamp": "2026-03-09T14:21:00Z" }
```

### Models

Flow reads model paths from:

- `Boxes/flow/config/box_detector.yaml`
- `Boxes/flow/config/defect_detector.yaml`

Artifacts typically live in `Boxes/flow/models/` and can be:

- `.engine` (TensorRT)
- `.onnx`
- `.pt` (Ultralytics)

---

## Boxes/training — Training pipelines (box-YOLO + defect-YOLO)

There are two separate projects:

- `Boxes/training/box-YOLO` → `PROJECT_NAME = detect_box`
- `Boxes/training/defect-YOLO` → `PROJECT_NAME = defect_box`

Both follow the same pipeline:

1. **Train** (`training/train.py`)
2. **Export ONNX** (`export/export_onnx.py`)
3. **Quantize** (`export/quantize_onnx.py`) using ONNX Runtime dynamic quantization
4. **Collect metrics** (`utils/utils.py`)

### Dataset layout (required)

Training expects:

```
data/
  data/
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
```

You must create `data/data/data.yaml` (it is not committed in this repo). Example:

```yaml
path: data/data
train: train/images
val: valid/images
test: test/images

names:
  0: box
```

### Merge datasets helper

Both projects include `scripts/merge_data.py` which merges multiple dataset folders like:

- `data/data 1`
- `data/data 2`
- `data/data 3`

into `data/data/` and can **force all labels to class 0** (see `CLASS_ID` inside the script).

### Run the full pipeline

Box model:

```bash
cd Boxes/training/box-YOLO
python3 scripts/run_all.py
```

Defect model:

```bash
cd Boxes/training/defect-YOLO
python3 scripts/run_all.py
```

### Outputs

After a successful run, artifacts are placed in:

- `models/exported/best.pt`
- `models/exported/<name>.onnx` (e.g. `detect_box.onnx`)
- `models/exported/<name>_int8.onnx` (e.g. `detect_box_int8.onnx`)

### Using trained models in Flow

Copy exported artifacts into the flow models folder and update flow configs:

```bash
cp Boxes/training/box-YOLO/models/exported/detect_box_int8.onnx   Boxes/flow/models/
cp Boxes/training/defect-YOLO/models/exported/defect_box_int8.onnx Boxes/flow/models/
```

Then set `model_path` in:

- `Boxes/flow/config/box_detector.yaml`
- `Boxes/flow/config/defect_detector.yaml`

---

## Troubleshooting

- **Firebase init fails**:
  - Verify `Boxes/flow/config/firebase-service-account.json` exists (and is valid).
  - Verify `Boxes/flow/config/firebase.yaml` has a correct `database_url`.
- **WebRTC can't connect**:
  - Configure `Boxes/flow/config/webrtc.yaml` (TURN secret required for many NATs).
  - Client reads ICE servers from `GET /api/config`.
- **Device selection**:
  - Flow: set `device` in `box_detector.yaml` / `defect_detector.yaml` (e.g. `auto`, `cuda`, `cpu`).
  - Training: set `DEVICE` in the training config (e.g. `configs/config.py`).

---

## Backend API & Integration Documentation

This section documents the backend (Flow) APIs, report lifecycle, WebRTC integration, Firebase writes, and configuration so that any developer can integrate a client without reading the code.

### 1. Project Overview

- **Purpose**: provide a headless detection service that:
  - opens *reports* for specific production lines and camera sources,
  - streams annotated video via WebRTC, and
  - writes minimal detection events to Firebase Realtime Database.
- **Main components**:
  - **API server** (`Boxes/flow/api/api_server.py`) — FastAPI app, REST endpoints, WebRTC offer handling.
  - **Pipeline** (`Boxes/flow/core/`) — camera stream, YOLO box + defect models, tracking and decision logic.
  - **Firebase integration** (`Boxes/flow/core/firebase_client.py`, `_firebase_worker` in `pipeline_manager.py`) — writes detection events.
  - **WebRTC streaming** (`Boxes/flow/core/webrtc_track.py`, `/webrtc/offer`) — delivers annotated frames to browsers.
- **High-level architecture**:

```
Client (any WebRTC-capable app)
→ Backend API (/api/reports/*, /api/health, /api/config, /webrtc/offer)
→ Pipeline (camera → box detector → defect detector)
→ WebRTC stream (annotated frames to client)
→ Firebase events ({report_id}/{detection_id} with defect, timestamp)
```

### 2. API Endpoints Documentation

All endpoints are defined in `Boxes/flow/api/api_server.py`.

#### 2.1 `POST /api/reports/open`

- **Description**:  
  Open (or reuse) a headless detection report for a given production line and camera source.
  - If a report for the given `production_line_id` is already open, returns **success** and reuses the existing report.
  - If the camera is already locked by another report, returns a 400 error.

- **Request body**:

```json
{
  "report_id": "string",
  "camera_source": "0",
  "production_line_id": "line-1"
}
```

`camera_source` can be a string (e.g. RTSP URL) or integer (device index).

- **Successful response (new report)**:

```json
{
  "status": "success",
  "report_id": "r-123",
  "message": "Report started with camera 0"
}
```

- **Successful response (already open for line)**:

```json
{
  "status": "success",
  "report_id": "r-123",
  "message": "Report is already open for this production line"
}
```

- **Error cases**:
  - **400 Bad Request** (ValueError raised in `SessionManager.create_session`):
    - Report already exists with the same `report_id`.
    - Camera is already in use by another report:
      - Message similar to:  
        `Camera {camera_source} in use by session {other_report_id}`.
  - **500 Internal Server Error**:
    - Any unexpected exception starting the pipeline.

- **Notes**:
  - Idempotency is enforced **per production line**: repeated opens for the same `production_line_id` are safe.
  - `report_id` is controlled by the client.

#### 2.2 `POST /api/reports/close`

- **Description**:  
  Close an active report identified by `report_id`. If the report is already closed or does not exist, the call is still treated as success.

- **Request body**:

```json
{
  "report_id": "r-123"
}
```

- **Successful response (closed now)**:

```json
{
  "status": "success",
  "report_id": "r-123",
  "message": "Report closed"
}
```

- **Successful response (already closed)**:

```json
{
  "status": "success",
  "report_id": "r-123",
  "message": "Report is already closed"
}
```

- **Error cases**:
  - **500 Internal Server Error**:
    - Any unexpected exception while stopping the worker or cleaning resources.

- **Notes**:
  - Explicitly idempotent: the client can safely call close multiple times without tracking local state.

#### 2.3 `GET /api/reports`

- **Description**:  
  List all currently active (open) reports.

- **Response**:

```json
[
  {
    "report_id": "r-123",
    "viewers_count": 1
  },
  {
    "report_id": "r-456",
    "viewers_count": 0
  }
]
```

- **Error cases**:
  - **500 Internal Server Error**:
    - Unexpected problems when querying the in-memory session registry.

- **Notes**:
  - `viewers_count` is the number of active WebRTC subscribers (video tracks) for that report.

#### 2.4 `GET /api/health`

- **Description**:  
  Lightweight health check and report count.

- **Response**:

```json
{
  "status": "healthy",
  "active_reports": 2
}
```

- **Error cases**:
  - **500 Internal Server Error**:
    - Any unexpected exception while reading the active sessions map.

#### 2.5 `GET /api/config`

- **Description**:  
  Returns client-side WebRTC configuration, including:
  - `webrtc.iceServers`: STUN/TURN servers.
  - Temporary TURN credentials (username/password) generated from the TURN secret.
  - `webrtc_mode`: connection strategy (`auto`, `direct`, `stun`, `relay`).

- **Response example**:

```json
{
  "webrtc": {
    "iceServers": [
      { "urls": "stun:20.51.117.96:3478" },
      {
        "urls": "turn:20.51.117.96:3478",
        "username": "1709999999:stream",
        "credential": "base64-hmac"
      }
    ],
    "webrtc_mode": "auto"
  }
}
```

- **Error cases**:
  - This endpoint is simple and normally does not error unless configuration is completely missing; in that case a generic 500 may be returned.

- **Notes**:
  - TURN credentials are short-lived (e.g. 5 minutes) and generated via `HMAC(secret, username)` in `_generate_turn_credentials`.

#### 2.6 `POST /webrtc/offer`

- **Description**:  
  Exchange WebRTC SDP offer/answer for a given report and attach a `VideoStreamTrack` that receives frames from the detection pipeline.

- **Request body**:

```json
{
  "sdp": "v=0...",
  "type": "offer",
  "report_id": "r-123"
}
```

- **Successful response**:

```json
{
  "sdp": "v=0...",
  "type": "answer"
}
```

- **Error cases**:
  - **404 Not Found**:
    - No active report with that `report_id`:
      - Detail: `"Report not found"`.
  - **500 Internal Server Error**:
    - Any unexpected aiortc error (e.g. SDP parsing, track wiring).

- **Notes**:
  - The client must first open a report (`/api/reports/open`) before calling `/webrtc/offer`.
  - The same endpoint is used for all reports; the selected pipeline is determined by `report_id`.

### 3. Report Lifecycle

The lifecycle of a report is:

1. **Open**: client calls `POST /api/reports/open` with a `report_id`, `camera_source`, and `production_line_id`.  
2. **Pipeline start**:
   - A `SessionWorker` thread starts.
   - Camera frames are pulled via `CamStream`.
   - Frames are passed into the shared `PipelineManager` (one inference thread).
3. **Streaming & detections**:
   - Client negotiates WebRTC via `POST /webrtc/offer`.
   - Annotated frames are pushed to each subscribed `VideoStreamTrack`.
   - When a box exits the ROI and a decision is made (defect / OK), an event is pushed into the Firebase worker queue.
4. **Close**: client calls `POST /api/reports/close`. The worker stops, the pipeline is unregistered, and subsequent WebRTC offers for that `report_id` fail with `"Report not found"`.

Informal flow:

```text
Client → POST /api/reports/open
       → POST /webrtc/offer  → WebRTC stream (video)
       → Firebase writes: {report_id}/{detection_id} {defect, timestamp}
       → POST /api/reports/close
```

### 4. WebRTC Configuration

`/api/config` builds its response from:

- `Boxes/flow/config/webrtc.yaml` (required; copy from `webrtc.example.yaml`), containing:

```yaml
stun:
  urls: "stun:HOST:3478"

turn:
  urls: "turn:HOST:3478"
  secret: "YOUR_TURN_SECRET"

webrtc_mode: auto   # auto | direct | stun | relay
```

The backend:

- Generates `webrtc.iceServers` based on `webrtc_mode`:
  - `auto`: host + STUN + TURN (with credentials).
  - `direct`: host only.
  - `stun`: host + STUN.
  - `relay`: host + TURN only.
- Uses TURN `secret` from `webrtc.yaml` to compute short-lived TURN username/credential.

#### WebRTC usage example

```javascript
// Load config once (or periodically)
const cfg = await fetch('/api/config').then(r => r.json());
const webrtc = cfg.webrtc || cfg;

const pc = new RTCPeerConnection({
  iceServers: webrtc.iceServers
});
```

### 5. Firebase Realtime Events

Firebase is initialized in `core/firebase_client.py` with:

- `credentials_path` → service account JSON (from `firebase.yaml` or env).
- `database_url` → Realtime Database URL (from `firebase.yaml` or `firebase_config.json`).

Events are written by `publish_detection(report_id, timestamp, defect)`:

```python
path = report_id
payload = {"defect": defect, "timestamp": timestamp}
db.reference(path).push(payload)
```

Structure in the database:

```text
{report_id}
  └── {detection_id}  # Firebase push key
        defect: true
        timestamp: "2026-03-09T14:21:00Z"
```

- `report_id`: identifier chosen by the client when opening the report.
- `detection_id`: auto-generated key from Firebase `push()`.
- `defect`:
  - `true` → defect detected.
  - `false` → explicitly non-defect event (depending on pipeline logic).
- `timestamp`: ISO 8601 UTC time string, generated by the Firebase worker when enqueueing the event.

### 6. Configuration System (backend)

All backend configuration is under `Boxes/flow/config/`:

- `api.yaml` — Uvicorn:
  - `host`, `port`, `log_level`.
- `app.yaml` — CORS:
  - `cors_origins` list (empty = allow all, good for local dev).
- `firebase.yaml` — Firebase project selection:

```yaml
service_account_path: "firebase-service-account.json"
database_url: "https://chainly-f4afa-default-rtdb.europe-west1.firebasedatabase.app"
```

- `webrtc.yaml` — STUN/TURN and TURN secret (gitignored).
- `firebase_config.json` (optional) — alternate place for `database_url` if not in `firebase.yaml`.

Resolution order for `database_url`: 1) `firebase.yaml` → `database_url`, 2) `firebase_config.json` → `database_url` or `FIREBASE_DATABASE_URL`.

### 7. Setup Instructions (backend)

1. **Clone the repository**:

   ```bash
   git clone <repo-url>
   cd QC-SCM
   ```

2. **Install dependencies** (full dev):

   ```bash
   pip install -e .
   ```

3. **Firebase service account JSON**:
   - Download a service account key JSON from the Firebase console.
   - Save it as:

   ```text
   Boxes/flow/config/firebase-service-account.json
   ```

4. **Configure `firebase.yaml`**:

   ```bash
   cp Boxes/flow/config/firebase.example.yaml Boxes/flow/config/firebase.yaml
   ```

   Then edit `firebase.yaml` to point to:

   - `service_account_path: "firebase-service-account.json"`
   - `database_url: "https://chainly-f4afa-default-rtdb.europe-west1.firebasedatabase.app"` (or your project).

5. **Configure WebRTC**:

   ```bash
   cp Boxes/flow/config/webrtc.example.yaml Boxes/flow/config/webrtc.yaml
   ```

   Set:

   - `stun.urls`
   - `turn.urls`
   - `turn.secret`

6. **Start the backend** (config is loaded from `Boxes/flow/config/` only):

   ```bash
   cd Boxes/flow
   python3 main.py
   ```

### 8. Security Notes

Sensitive files **must not** be committed:

- `Boxes/flow/config/firebase-service-account.json` (Firebase private key).
- `Boxes/flow/config/webrtc.yaml` (contains TURN secret).
- `Boxes/flow/config/firebase_config.json` (if used for `database_url`).

The repo's `.gitignore` is already configured to ignore these; only the `*.example` templates are tracked. When adding new secrets, keep them under `Boxes/flow/config/` and extend `.gitignore` as needed.

TURN credentials generated for WebRTC are time-limited and derived from a secret; treat the secret as sensitive as any password.

### 9. Repository Structure (backend focus)

Relevant folders for the backend:

```text
Boxes/
  flow/
    main.py          # Entry point (uvicorn)
    api/             # FastAPI app + schemas
    core/            # pipeline manager, session manager, Firebase, WebRTC track
    config/          # YAML + env configs
    detectors/       # YOLO wrapper
    models/          # Inference models for flow
    requirements/    # Flow runtime requirements
    scripts/         # run_dev.sh
    utils/           # Visualizer, geometry
```

Each of these has its own small `README.md` under `Boxes/flow/*/` for quick reference.

### 10. Example API Usage (curl)

#### Open a report

```bash
curl -sS -X POST "http://localhost:8000/api/reports/open" \
  -H "content-type: application/json" \
  -d '{"report_id":"r-123","camera_source":0,"production_line_id":"line-1"}'
```

#### Close a report

```bash
curl -sS -X POST "http://localhost:8000/api/reports/close" \
  -H "content-type: application/json" \
  -d '{"report_id":"r-123"}'
```

#### List active reports

```bash
curl -sS "http://localhost:8000/api/reports"
```

#### Get health

```bash
curl -sS "http://localhost:8000/api/health"
```

#### Get WebRTC config

```bash
curl -sS "http://localhost:8000/api/config"
```

Clients then use `/webrtc/offer` with an SDP offer plus `report_id` to establish a WebRTC stream for that report.

---

## 1. Project Overview

### 1.1 What the project does

QC-SCM is an AI-driven quality inspection system for manufacturing lines. It:

- Connects to camera sources on production lines.
- Runs a **two-stage YOLO-based detection pipeline**:
  - **Box detector** finds the box/package in a region of interest (ROI).
  - **Defect detector** examines that box for defects.
- Streams **annotated video frames** via **WebRTC** to external clients.
- Emits **minimal defect events** into **Firebase Realtime Database** for downstream analytics and traceability.

### 1.2 Goal of the system

- Provide a **headless detection service** that any client can integrate with via a simple HTTP + WebRTC API.
- Separate **training** (YOLO projects) from **runtime inference** (Flow service), so models can evolve without changing the serving code.

### 1.3 Main components

- **Flow runtime (`Boxes/flow/`)**
  - FastAPI backend.
  - WebRTC signaling and media streaming (via `aiortc`).
  - Box + defect detection pipeline.
  - Firebase Realtime Database event publishing.
- **Training pipelines (`Boxes/training/`)**
  - `box-YOLO`: trains the **box detector**.
  - `defect-YOLO`: trains the **defect detector**.
  - Exports ONNX and quantized INT8 ONNX models for deployment.

### 1.4 High-level architecture

```text
Camera → Flow backend:
  - CamStream (OpenCV capture)
  - PipelineManager (single inference thread)
  - Pipeline (box YOLO → defect YOLO → tracking + decision)
  - WebRTC track (annotated frames)
  - Firebase worker (defect events)

          ↓

External client (WebRTC viewer, dashboards, etc.)
```

With data paths:

```text
Camera → AI pipeline → WebRTC stream → external client
Camera → AI pipeline → detection event → Firebase Realtime Database
```

---

## 2. System Architecture

### 2.1 Backend services

- **FastAPI app (`Boxes/flow/api/api_server.py`)**
  - Creates the FastAPI instance.
  - Loads YAML/ENV configuration (`_load_configs`).
  - Initializes Firebase and models on startup (`@app.on_event("startup")`).
  - Cleans up `PipelineManager` and WebRTC peer connections on shutdown.
  - Exposes:
    - `/api/reports/open`, `/api/reports/close`, `/api/reports`
    - `/api/health`
    - `/api/config`
    - `/webrtc/offer`

- **Session management (`Boxes/flow/core/session_manager.py` + `session_worker.py`)**
  - `SessionManager`:
    - Ensures one active report per `production_line_id`.
    - Ensures a camera source is not shared across reports.
  - `SessionWorker` (per-report thread):
    - Creates a `Pipeline` configured with the correct detectors and stream settings.
    - Starts the camera capture (`CamStream`).
    - Registers the pipeline and WebRTC tracks with `PipelineManager`.
    - Spawns a camera feeder thread that pushes frames into `PipelineManager`.

- **Pipeline orchestration (`Boxes/flow/core/pipeline_manager.py`)**
  - Singleton `PipelineManager` with:
    - Bounded `frame_queue` for frames from all reports.
    - Bounded `result_queue` for annotated frames + exit events.
    - Bounded `firebase_queue` for final defect decisions.
  - Starts three threads:
    - **Inference thread**: serializes all GPU work, calling `Pipeline.run_step`.
    - **Result-consumer thread**: updates WebRTC tracks and enqueues Firebase events.
    - **Firebase worker thread**: commits events to Firebase Realtime Database.

- **Inference pipeline (`Boxes/flow/core/pipeline.py`)**
  - Holds:
    - Camera stream (`CamStream`).
    - Box + defect detectors.
    - `AppState` (tracking, stability, counts).
    - `Visualizer` (canvas layout and overlays).
  - Implements:
    - `run_step(frame, enqueue_time, camera_fps)` for frame-by-frame inference.
    - Throttled box and defect detection.
    - Single-box tracking and decision logic.

- **Camera capture (`Boxes/flow/core/stream.py`)**
  - `CamStream`:
    - Uses OpenCV with V4L2 backend and MJPG fourcc.
    - Runs a dedicated capture thread.
    - Stores only the latest frame in `deque(maxlen=1)`.
    - Provides non-blocking `get_latest_frame()`.

- **Model/device management (`Boxes/flow/core/device_manager.py`, `model_loader.py`)**
  - `select_device`:
    - Chooses `cuda` / `mps` / `cpu` from config (`box_detector.yaml` / `defect_detector.yaml` device).
  - `ModelLoader`:
    - Loads box and defect models once.
    - Provides references to `Pipeline` instances.
    - Optionally performs warmup passes.

### 2.2 WebRTC streaming

- Clients obtain ICE configuration from:

  ```http
  GET /api/config
  ```

  which returns:

  ```json
  {
    "webrtc": {
      "iceServers": [...],
      "webrtc_mode": "auto"
    }
  }
  ```

- Clients send SDP offers to:

  ```http
  POST /webrtc/offer
  {
    "sdp": "...",
    "type": "offer",
    "report_id": "r-123"
  }
  ```

- The backend:
  - Looks up `SessionWorker` for `report_id`.
  - Creates `RTCPeerConnection` with ICE servers derived from `webrtc.yaml`.
  - Adds a `VideoTransformTrack`:
    - `PipelineManager` feeds it annotated frames via `update_frame`.
    - aiortc pulls frames from `recv()` and sends them to the client.

### 2.3 Firebase event storage

- Initialization:
  - `_load_configs` reads `firebase.yaml` then `firebase_config.json` for `database_url`.
  - `firebase_client.initialize(credentials_path, database_url)` sets up `firebase_admin`.
- Publishing:
  - `PipelineManager`'s Firebase worker calls:

    ```python
    publish_detection(report_id, timestamp, defect)
    ```

  - Which writes to Realtime Database under path `report_id` with an auto-generated `detection_id`.

---

## 3. AI / ML Pipeline

### 3.1 Models and roles

- **Box detector (`Boxes/training/box-YOLO`)**
  - `PROJECT_NAME = "detect_box"`.
  - Detects the single box in the ROI per frame.
- **Defect detector (`Boxes/training/defect-YOLO`)**
  - `PROJECT_NAME = "defect_box"`.
  - Detects defects on the surface of the tracked box.

Both start from pretrained YOLO models (`yolo26n.pt`/`yolov8n.pt`) and use Ultralytics YOLO training APIs.

### 3.2 Training datasets

Both detectors use the same YOLO-style dataset structure:

```text
data/
  data/
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
```

`data/data/data.yaml` (user-provided) defines:

- `path`, `train`, `val`, `test` directories.
- Class `names` (e.g. `0: box` or `0: defect`).

Utility scripts `scripts/merge_data.py` help compose large datasets and can remap labels to a single class.

### 3.3 Training process

For each project (`training/train.py`):

1. Imports `configs.config` for:
   - Paths (`ROOT`, `DATA_YAML`, `PROJECT_DIR`, `PROJECT_NAME`).
   - Hyperparameters (`EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`, `DEVICE`).
2. Ensures output directories exist via `ensure_dirs`.
3. If `runs/train/<PROJECT_NAME>/weights/last.pt` exists:
   - Resumes training from `last.pt`.
   - Otherwise:
     - Ensures `BASE_MODEL` exists via `check_and_download_model`.
     - Starts from the base weights.
4. Selects training device:
   - `select_device(DEVICE, context="training")` (device from training config only).
5. Calls `YOLO(...).train(...)` with:
   - `data=DATA_YAML`, `epochs=100`, `imgsz=640`.
   - `batch=8` (box) or `16` (defect).
   - `patience=15`, `cache=True`, `cos_lr=True`, `mosaic=0.8`, `save=True`.
6. Cleans up intermediate checkpoints with `prune_weights`, keeping `best.pt` and `last.pt`.

### 3.4 Export and quantization

- **Export to ONNX**:
  - `export/export_onnx.py`:
    - Loads `best.pt` from `runs/train/<PROJECT_NAME>/weights/`.
    - Copies it to `models/exported/best.pt`.
    - Exports to ONNX with `opset=OPSET` and `dynamic=True`.
    - Renames `best.onnx` to `models/exported/<ONNX_NAME>`.

- **Quantize to INT8**:
  - `export/quantize_onnx.py`:
    - Loads `models/exported/<ONNX_NAME>`.
    - Runs `quantize_dynamic(..., weight_type=QuantType.QUInt8)`.
    - Produces `models/exported/<ONNX_INT8_NAME>`.

These ONNX artifacts are copied into `Boxes/flow/models/` for runtime use.

### 3.5 Inference flow per frame

Summarized from `core/pipeline.py`:

1. `CamStream` capture thread writes frames into `deque(maxlen=1)`.
2. `SessionWorker` feeder reads the newest frame and enqueues into `PipelineManager`.
3. Inference thread calls `Pipeline.run_step(frame, enqueue_time, camera_fps)`:
   - Builds or reuses a canvas with an info panel.
   - Extracts an ROI for the conveyor belt.
   - Runs box detection at a configurable rate (`box_detect_every_n_frames`).
   - Maintains a **single tracked box** using IoU-based matching and smoothed coordinates.
   - Runs defect detection on the cropped box at a separate rate (`defect_detect_every_n_frames`).
   - Uses `AppState` to:
     - Track entry/exit of the box through the ROI.
     - Perform a rolling vote over recent defect results.
     - Decide `"DEFECT"` vs `"OK"` once the box exits.
   - Returns:
     - `canvas`: annotated frame.
     - `exit_event`: `True`/`False` if a box has just exited; `None` otherwise.

---

## 4. Training Configuration

### 4.1 Hyperparameters (box-YOLO vs defect-YOLO)

Both `configs/config.py` files define:

- `EPOCHS = 100`
- `IMG_SIZE = 640`
- `DEVICE = "auto"` (from training config only)
- `BATCH_SIZE`:
  - Box detector: `8`
  - Defect detector: `16`
- Export:
  - `ONNX_NAME` / `ONNX_INT8_NAME`
  - `OPSET = 12`

### 4.2 Dataset config

- `DATA_YAML = DATA_DIR / "data" / "data.yaml"`:
  - Standard YOLO `data.yaml` with `train`, `val`, `test` and `names`.
- Data augmentations are those of Ultralytics YOLO, configured via training arguments:
  - `mosaic=0.8` enables mosaic augmentation most of the time.
  - Other augmentations (flip, HSV, scale) follow defaults.

---

## 5. Model Metrics

### 5.1 Metrics output

Each training run produces:

- `results.csv` – per-epoch metrics (precision, recall, mAP, losses).
- `results.png` – plotted curves.
- `confusion_matrix.png` and `confusion_matrix_normalized.png`.
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png`.

### 5.2 Metric collation

`utils/utils.py` in each project:

- `collect_final_metrics(run_dir, dest_dir)` copies the above into:

```text
Boxes/training/box-YOLO/runs/metrics/
Boxes/training/defect-YOLO/runs/metrics/
```

### 5.3 Using metrics

- Use **precision/recall/F1** and PR curves to:
  - Choose `conf_thres` and `iou_thres` for runtime configuration.
  - Compare model versions.
- Use **confusion matrices** to:
  - Inspect false positives/negatives.
  - Validate class separability (especially for multi-class defect detectors).

---

## 6. Runtime Detection Flow

End-to-end summary:

1. **Open report**
   - Client calls `POST /api/reports/open` with `{ report_id, camera_source, production_line_id }`.
   - Backend:
     - Starts `SessionWorker` if not already open for the production line.
     - Creates `Pipeline` and `CamStream`.
2. **Capture and enqueue frames**
   - `CamStream` capture thread reads from camera → updates `deque(maxlen=1)`.
   - Camera feeder thread reads from `CamStream` and pushes frames to `PipelineManager.put_frame(...)`.
3. **Single-threaded inference**
   - Inference thread:
     - Dequeues `(session_id, frame, enqueue_time, camera_fps)`.
     - Runs `Pipeline.run_step`.
     - Measures latency and updates diagnostics.
4. **Per-box decision**
   - `Pipeline.run_step`:
     - Updates tracking and defect votes.
     - Emits `exit_event` when a box leaves the ROI with a final decision.
5. **Distribute results**
   - Result-consumer thread:
     - Updates all WebRTC tracks for the report with the latest `canvas`.
     - If `exit_event` is present:
       - Pushes `(session_id, is_defect, firebase_meta)` into `firebase_queue`.
6. **Persist events**
   - Firebase worker thread:
     - Dequeues events and calls `publish_detection(report_id, timestamp, defect)`.
7. **Close report**
   - Client calls `POST /api/reports/close` with `{ report_id }`.
   - Backend:
     - Stops `SessionWorker`, unregisters from `PipelineManager`.

---

## 7. Firebase Event System

Each event is:

```json
{
  "defect": true,
  "timestamp": "2026-03-09T14:21:00Z"
}
```

Stored under:

```text
{report_id}/{detection_id}
```

- `detection_id` is a Firebase push key; ordering roughly follows write time.

---

## 8. Configuration System

### 8.1 Flow runtime

All config lives in `Boxes/flow/config/`:

- `api.yaml` – Uvicorn settings.
- `app.yaml` – CORS (origins).
- `webrtc.yaml` / `webrtc.example.yaml` – STUN/TURN and `webrtc_mode`.
- `firebase.yaml` / `firebase.example.yaml` – service account path and `database_url`.
- All configuration is in `Boxes/flow/config/` (no `.env` or environment variables).
- `firebase_config.json` / `.example` – JSON alternative for DB URL.
- `box_detector.yaml`, `defect_detector.yaml`, `stream.yaml` – detector and stream configuration.

### 8.2 Training

- `Boxes/training/*-YOLO/configs/config.py` – per-project training config.
- `data/data/data.yaml` – dataset config (user-provided).

---

## 9. Project Structure

High-level:

```text
QC-SCM/
  README.md
  pyproject.toml
  .gitignore
  Boxes/
    flow/       # Runtime detection service
    training/   # YOLO training projects
```

```text
Boxes/training/
  box-YOLO/
    configs/
    training/
    export/
    models/
    runs/
    scripts/
    utils/

  defect-YOLO/
    (same structure)
```

---

## 10. Deployment / Running the System

1. Clone repo and `pip install -e .` (or Flow-only requirements).
2. Configure Firebase (`firebase-service-account.json`, `firebase.yaml` or env).
3. Configure WebRTC (`webrtc.yaml`, TURN secret).
4. Train and export models (optional if you bring your own):
   - `Boxes/training/box-YOLO/scripts/run_all.py`
   - `Boxes/training/defect-YOLO/scripts/run_all.py`
5. Copy exported ONNX/INT8 artifacts to `Boxes/flow/models/` and update detector configs.
6. Start backend:

   ```bash
   cd Boxes/flow
   python3 main.py
   ```

7. Use `/api/reports/*` to manage reports and `/webrtc/offer` + `/api/config` to stream video.

---

## 11. Security Considerations

### 11.1 Sensitive files

Do **not** commit:

- `Boxes/flow/config/firebase-service-account.json`
- `Boxes/flow/config/webrtc.yaml`
- `Boxes/flow/config/firebase_config.json` (if used)
- Any proprietary datasets or raw production images.

### 11.2 Configuration (config directory only)

All configuration is loaded from files under `Boxes/flow/config/`. No environment variables or `.env` files are used. Sensitive values (e.g. `database_url`, TURN `secret`) belong in `firebase.yaml`, `webrtc.yaml`, or `firebase_config.json`; keep these files out of version control.

### 11.3 TURN and Firebase security

- Treat TURN `secret` as a password; it controls generation of short-lived TURN credentials.
- Restrict Firebase service account permissions and enforce strict Realtime Database rules so only the backend can write events.

### 11.4 Data & privacy

- Protect the backend API behind authentication or network controls where required.
- Use HTTPS in production for both API and WebRTC signaling.
- Define retention/anonymization policies for events stored in Firebase.