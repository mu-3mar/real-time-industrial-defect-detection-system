# QC-SCM: Quality Control — Supply Chain Management

AI-driven quality inspection system for manufacturing: **train** YOLO models (box + defect), then **run** a real-time detection server that streams video over WebRTC and writes detection events to Firebase Realtime Database. All factory and production-line metadata is provided by the client; the server does not store static configuration.

---

## Repository overview

| Part | Purpose |
|------|--------|
| **Boxes/flow** | Detection server: FastAPI, two-stage pipeline (box → defect), WebRTC streaming, Firebase Realtime Database publishing. |
| **Boxes/trainig** | Model training: box-YOLO (detect boxes) and defect-YOLO (classify defects). Train → export ONNX → quantize → use in flow. |
| **index.html** | Single-page HTML test client to open sessions, stream video, and test the API. |
| **pyproject.toml** | Python dependencies for the whole project. |

---

## Repository structure

```
QC-SCM/
├── README.md
├── index.html                    # HTML test client (open session, WebRTC stream)
├── pyproject.toml                # Dependencies
├── .gitignore
│
├── Boxes/
│   ├── flow/                     # Detection server (runtime)
│   │   ├── main.py               # Entry point: loads api.yaml, runs uvicorn
│   │   ├── api/                  # API layer
│   │   │   └── api_server.py     # FastAPI app: sessions, health, WebRTC, config
│   │   ├── config/               # YAML configs, .env.example, firebase_config.json.example
│   │   ├── core/                 # Pipeline, session worker, stream, state, Firebase client, device_manager
│   │   ├── docs/                 # Service docs (endpoints.md, README.md)
│   │   ├── scripts/              # run_dev.sh
│   │   ├── requirements/         # requirements.txt for flow runtime
│   │   ├── detectors/            # YOLO/ONNX detector wrapper
│   │   ├── utils/                # Visualizer, geometry
│   │   └── models/               # ONNX models (e.g. detect_box_int8.onnx, defect_box_int8.onnx)
│   │
│   └── trainig/                  # Model training (note: folder name is "trainig")
│       ├── box-YOLO/             # Box detection model
│       │   ├── configs/config.py    # Paths, epochs, batch, export names
│       │   ├── training/train.py    # Ultralytics YOLO training
│       │   ├── export/export_onnx.py
│       │   ├── export/quantize_onnx.py
│       │   ├── scripts/run_all.py   # Train → export ONNX → quantize → metrics
│       │   ├── scripts/merge_data.py
│       │   ├── inference/infer.py
│       │   ├── data/                # Dataset (data.yaml, train/valid/test)
│       │   ├── models/pretrained/   # e.g. yolo26n.pt
│       │   ├── models/exported/     # best.pt, *.onnx after export
│       │   └── runs/                # train outputs, metrics
│       │
│       └── defect-YOLO/          # Defect classification model
│           ├── configs/config.py
│           ├── training/train.py
│           ├── export/export_onnx.py
│           ├── export/quantize_onnx.py
│           ├── scripts/run_all.py
│           ├── scripts/merge_data.py
│           ├── inference/infer.py
│           ├── data/
│           ├── models/
│           └── runs/
```

---

## Boxes/flow — Detection server (full details)

The **flow** folder is the runtime: it loads ONNX models, runs the two-stage pipeline per session, streams video over WebRTC, and writes each box result to Firebase Realtime Database.

### Entry point and API

- **main.py**  
  Reads `config/api.yaml` (host, port, log_level), sets ONNX logging level, then runs `uvicorn` with `api_server:app`. Default: `http://0.0.0.0:8000`.

- **api_server.py**  
  - Loads configs from `config/` (app, webrtc, box, defect, stream, firebase).  
  - Initializes Firebase Realtime Database using credentials from `firebase.yaml` and database URL from `.env` or `config/firebase_config.json`.  
  - Endpoints:  
    - `POST /api/sessions/open` — open session (body: factory_id, factory_name, production_line_id, production_line_name, camera_id, camera_source, station_id, session_id).  
    - `POST /api/sessions/close` — close session (body: session_id, optional camera_source).  
    - `GET /api/sessions` — list active sessions with metadata.  
    - `GET /api/health` — health and active session count.  
    - `GET /api/config` — WebRTC ICE config for clients.  
    - `POST /webrtc/offer` — SDP offer/answer and attach video track (body: sdp, type, session_id).

### Core components

- **core/session_manager.py** — Singleton: creates/closes sessions by `session_id`, locks camera per session.  
- **core/session_worker.py** — Thread per session: runs the pipeline with the session’s metadata, calls Firebase on each box result.  
- **core/pipeline.py** — Camera → box detection → defect detection → entry/exit and voting; updates FPS/latency; calls `on_result_callback` when a box exits.  
- **core/stream.py** — `CamStream`: dedicated capture thread, deque(maxlen=1), non-blocking `get_latest_frame()`.  
- **core/state.py** — Per-track state: voting, defect lock, entry/exit, recovery.  
- **core/firebase_client.py** — Initializes Firebase from credentials path and database URL; `publish_detection(...)` pushes one record per detection to Realtime Database.  
- **core/model_loader.py** — Singleton: loads box and defect ONNX models from paths in config.  
- **core/webrtc_track.py** — Video track for WebRTC; receives frames from the pipeline.  
- **detectors/detector.py** — Wrapper around YOLO/ONNX inference.  
- **utils/visualizer.py**, **utils/geometry.py** — Drawing and IoU/smoothing helpers.

### Config (Boxes/flow/config)

| File | Purpose |
|------|--------|
| **api.yaml** | host, port, log_level (used by main.py / uvicorn). |
| **app.yaml** | Optional CORS origins. |
| **webrtc.yaml** | STUN/TURN URLs and **secret** (gitignored; use webrtc.example.yaml as template). |
| **firebase.yaml** | `credentials_path`: filename of Firebase service account JSON in this folder (JSON is gitignored). Database URL via `FIREBASE_DATABASE_URL` in `.env` or `firebase_config.json`. |
| **box_detector.yaml** | model_path, conf_thres, iou_thres, device. |
| **defect_detector.yaml** | model_path, model_version, conf_thres, iou_thres, device, tracking, stability, rendering. |
| **stream.yaml** | width, height; throttle (e.g. box/defect every N frames). |

See **Boxes/flow/config/README.md** for a full config reference.

### Firebase Realtime Database structure

Events are written under:

```
factories / {factory_id} / production_lines / {production_line_id} / sessions / {session_id} / insights / {auto_generated_event_id}
```

Each insight record: `timestamp`, `defect`, `camera_id`, `station_id`, `factory_name`, `production_line_name`, `model_version`, `confidence`.

### HTML test client (index.html)

Single-page UI at repo root: set API base URL, open session (factory + production line from dropdowns), start WebRTC stream, close session. Uses predefined test metadata and does not change API behavior.

---

## Boxes/trainig — Training folder (full details)

The **trainig** folder contains two YOLO projects: **box-YOLO** (detect boxes in the frame) and **defect-YOLO** (classify defect inside a box). Each follows the same layout and pipeline: train → export ONNX → quantize INT8 → (optionally) copy to `Boxes/flow/models/` for the detection server.

### Directory layout (each of box-YOLO and defect-YOLO)

- **configs/config.py** — Defines `ROOT`, `MODELS_DIR`, `PRETRAINED_DIR`, `EXPORTED_DIR`, `DATA_DIR`, `RUNS_DIR`, `PROJECT_NAME`, `PROJECT_DIR`, `BASE_MODEL`, `DATA_YAML`, epochs, batch, image size, device, ONNX names, opset, metrics dir.  
- **training/train.py** — Uses Ultralytics YOLO: loads base or last.pt, trains on `DATA_YAML`, saves to `runs/train/{PROJECT_NAME}/weights/` (best.pt, last.pt).  
- **export/export_onnx.py** — Exports `best.pt` to ONNX (dynamic, opset from config).  
- **export/quantize_onnx.py** — Quantizes ONNX to INT8 (e.g. detect_box_int8.onnx / defect_box_int8.onnx).  
- **scripts/run_all.py** — Full pipeline: (1) train, (2) export ONNX, (3) quantize, (4) collect metrics. Run from project root (e.g. `Boxes/trainig/box-YOLO`).  
- **scripts/merge_data.py** — Merges multiple dataset folders into one `data/data` with train/valid/test and unified class IDs.  
- **inference/infer.py** — Inference script (e.g. run trained model on images).  
- **data/** — Dataset: `data/data.yaml` and train/valid/test with images and labels.  
- **models/pretrained/** — Base weights (e.g. yolo26n.pt).  
- **models/exported/** — best.pt and ONNX after export/quantize.  
- **runs/train/** — Training runs; **runs/metrics/** — Collected metrics.

### Box-YOLO (box detection)

- **PROJECT_NAME**: `detect_box`.  
- **Output**: Box detections in the ROI; used by the flow pipeline as the first stage.  
- **Typical use**: Place dataset in `data/`, set `DATA_YAML` and paths in `configs/config.py`, run `scripts/run_all.py`. Copy `models/exported/detect_box_int8.onnx` to `Boxes/flow/models/` and set `box_detector.yaml` model_path.

### Defect-YOLO (defect classification)

- **PROJECT_NAME**: `defect_box`.  
- **Output**: Defect (e.g. hole) inside a cropped box; used by the flow pipeline as the second stage.  
- **Typical use**: Same as box-YOLO; copy `defect_box_int8.onnx` to `Boxes/flow/models/` and set `defect_detector.yaml` model_path.

### Running the training pipeline

From the **box-YOLO** or **defect-YOLO** directory:

```bash
cd Boxes/trainig/box-YOLO
python scripts/run_all.py
```

or

```bash
cd Boxes/trainig/defect-YOLO
python scripts/run_all.py
```

Ensure `data/data/data.yaml` and (if used) `models/pretrained/yolo26n.pt` exist. After quantize, copy the generated `*_int8.onnx` into `Boxes/flow/models/` and point the flow configs to them.

---

## Installation

**Requirements:** Python 3.10+, optional CUDA for GPU.

1. Clone the repository and go to the project root.

2. Install dependencies:

   ```bash
   pip install -e .
   ```

3. **Flow server:**  
   - Put Firebase service account JSON in `Boxes/flow/config/` and set `credentials_path` in `Boxes/flow/config/firebase.yaml`.  
   - Set `FIREBASE_DATABASE_URL` in `Boxes/flow/config/.env` (copy from `Boxes/flow/config/.env.example`) or in `Boxes/flow/config/firebase_config.json` (copy from `firebase_config.json.example`). Do not commit `.env`.  
   - Optionally copy `Boxes/flow/config/webrtc.example.yaml` to `webrtc.yaml` and set your TURN secret (or set `TURN_SECRET` env var).  
   - Ensure `Boxes/flow/config/box_detector.yaml` and `defect_detector.yaml` point to existing ONNX files under `Boxes/flow/models/`.

4. **Training:**  
   - Add dataset under `Boxes/trainig/box-YOLO/data/` and `Boxes/trainig/defect-YOLO/data/` as needed; adjust `configs/config.py` and (for merge) `scripts/merge_data.py`.

---

## Running the detection server

Activate the `qc` conda environment (lowercase), then run from `Boxes/flow`:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate qc && cd Boxes/flow && python main.py
```

Or from project root:

Or from repo root with conda:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate qc && cd Boxes/flow && python main.py
```

Or run the script:

```bash
./Boxes/flow/scripts/run_dev.sh
```

Or uvicorn directly from `Boxes/flow`:

```bash
cd Boxes/flow
python -m uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`  
- Docs: `http://localhost:8000/docs`  
- Health: `http://localhost:8000/api/health`

---

## Testing with the HTML client

1. Start the server (see above).  
2. Open `index.html` in a browser (or serve the repo root).  
3. Set API base URL if needed (e.g. `http://localhost:8000`).  
4. Open a session (factory + line), start stream, check video and (if configured) Realtime Database insights.

---

## API summary (flow)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/sessions/open` | Open session; body includes factory_id, factory_name, production_line_id, production_line_name, camera_id, camera_source, station_id, session_id. |
| POST | `/api/sessions/close` | Close session; body: session_id, optional camera_source. |
| GET | `/api/sessions` | List active sessions with metadata. |
| GET | `/api/health` | Health and active session count. |
| GET | `/api/config` | WebRTC ICE config. |
| POST | `/webrtc/offer` | WebRTC offer/answer; body: sdp, type, session_id. |

---

## Realtime Database insight record

Each detection event is one record (auto-generated key) in  
`factories/{factory_id}/production_lines/{production_line_id}/sessions/{session_id}/insights/`.

| Field | Type | Description |
|-------|------|-------------|
| timestamp | string | ISO 8601 UTC. |
| defect | boolean | True = defect, false = OK. |
| camera_id | string | From session. |
| station_id | string | From session. |
| factory_name | string | From session. |
| production_line_name | string | From session. |
| model_version | string | From defect_detector config. |
| confidence | number | 0.0–1.0. |

---

## Technology stack

- **Backend:** FastAPI, Uvicorn  
- **Detection:** OpenCV, Ultralytics YOLO (training), ONNX Runtime (inference)  
- **Streaming:** WebRTC (aiortc)  
- **Events:** Firebase Realtime Database (firebase-admin)
