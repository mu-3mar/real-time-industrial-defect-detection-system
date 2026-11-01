# QC-SCM

## Real-Time Industrial Defect Detection System

QC-SCM is a local, real-time quality-control platform for industrial camera
streams. It detects the product or box first, then analyzes the detected
product for defects and classifies each completed item as **OK** or
**DEFECT**.

The project contains:

- A two-stage YOLO inference pipeline.
- Multiple independent camera sessions.
- A FastAPI runtime service.
- A local operator dashboard with live MJPEG video.
- An interactive, session-scoped rectangular detection area.
- Temporary local result storage with automatic cleanup.
- Versioned training and inference pipelines.
- A service manager for starting, stopping, monitoring, and logging the
  runtime.

> **Current deployment model:** QC-SCM is local-first. Firebase was used in an
> earlier integration, but it is not required by the current runtime and no
> results are uploaded to Firebase.

---

## Contents

- [System workflow](#system-workflow)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the system](#running-the-system)
- [Using the dashboard](#using-the-dashboard)
- [Interactive detection area](#interactive-detection-area)
- [Runtime API](#runtime-api)
- [Temporary session storage](#temporary-session-storage)
- [Configuration](#configuration)
- [Training and model versions](#training-and-model-versions)
- [Architecture](#architecture)
- [Monitoring and troubleshooting](#monitoring-and-troubleshooting)
- [Development checks](#development-checks)
- [Repository hygiene](#repository-hygiene)

---

## System workflow

Each active session follows this flow:

```text
Camera
  │
  ▼
SessionWorker
  │  captures the latest frame without blocking inference
  ▼
PipelineManager
  │  bounded producer-consumer queues
  ▼
Single inference worker
  │
  ├─ 1. Crop the current session ROI
  ├─ 2. Box detector finds the product
  ├─ 3. Tracker stabilizes the product identity
  ├─ 4. Defect detector analyzes the product crop
  ├─ 5. Stability voting classifies OK / DEFECT
  └─ 6. Publish the annotated frame and result
       │
       ├─ Dashboard MJPEG stream
       └─ Temporary per-session JSON result file
```

Changing the ROI during inference changes the crop used by the detector in the
same active session. It does not modify the shared YAML configuration.

---

## Repository structure

```text
.
├── dashboard/
│   ├── index.html                 Main dashboard and service controls
│   ├── reports.html               Active session list
│   ├── session.html               Live inspection page and ROI editor
│   ├── app.js                     Dashboard polling and monitoring logic
│   ├── service_manager.py         Runtime lifecycle and system metrics
│   ├── brand-logo.html            Transparent animated QC-SCM logo
│   ├── theme.css                  Shared dashboard visual system
│   └── README.md                  Dashboard-specific operations
│
├── src/qc_scm/runtime/
│   ├── main.py                    Runtime entry point
│   ├── api/                       FastAPI application and HTTP routes
│   ├── config/                    Runtime YAML configuration
│   ├── core/                      Pipeline, sessions, state, and storage
│   ├── detectors/                 Model detector wrappers
│   ├── requirements/              Runtime dependencies
│   ├── scripts/                   Development and deployment helpers
│   └── utils/                     Geometry and visualization helpers
│
├── training/
│   └── versions/
│       ├── v1_0/                  Baseline training and inference
│       ├── v2_0/                  V2.0 training and exports
│       └── v2_1/                  V2.1 training and exports
│
├── pyproject.toml                 Package metadata and editable install
├── .gitignore                     Local state and generated-file policy
└── README.md                      Project-wide documentation
```

The old `Boxes/flow` layout is no longer the production source tree. Runtime
code belongs under `src/qc_scm/runtime`; training code belongs under its
version directory.

---

## Requirements

- Linux or another environment that supports the project camera and OpenCV.
- Python 3.10 or newer.
- Conda environment named `qc` (recommended).
- A working camera or configured video source.
- The model files referenced by:
  - `src/qc_scm/runtime/config/box_detector.yaml`
  - `src/qc_scm/runtime/config/defect_detector.yaml`

GPU acceleration is recommended but not required if the configured detector
device supports CPU execution.

---

## Installation

From the repository root:

```bash
conda activate qc
pip install -e .
```

To install runtime dependencies directly:

```bash
conda activate qc
pip install -r src/qc_scm/runtime/requirements/requirements.txt
```

The service manager automatically prefers the `qc` Conda environment when it
starts the runtime. Set `QC_SCM_CONDA_ENV` if a different environment is
required.

---

## Running the system

### Recommended: dashboard and service manager

Start the local service manager from the repository root:

```bash
conda activate qc
python dashboard/service_manager.py
```

Open the dashboard at:

```text
http://127.0.0.1:8080
```

Use **Start Service** in the dashboard to start the detection runtime. The
manager launches the runtime on port `8000`, monitors its health, and exposes
logs and host metrics.

### Run the runtime directly

```bash
conda activate qc
./src/qc_scm/runtime/scripts/run_dev.sh
```

Or:

```bash
cd src/qc_scm/runtime
python main.py
```

The runtime API is available at:

```text
http://127.0.0.1:8000
```

### Optional automatic startup on Linux

Install the user-level service manager once:

```bash
./dashboard/install_service_manager.sh
```

Then inspect it with:

```bash
systemctl --user status qc-scm-service-manager.service
systemctl --user restart qc-scm-service-manager.service
```

---

## Using the dashboard

1. Start `dashboard/service_manager.py`.
2. Open `http://127.0.0.1:8080`.
3. Confirm the runtime is healthy.
4. Open the **Reports** page.
5. Create a new report/session and select its camera source.
6. Open the active session.
7. Use the live stream and ROI editor while inference is running.
8. Close the session when inspection is complete.

The dashboard includes:

- Runtime health and service state.
- API and storage health.
- CPU, RAM, disk, GPU, and network metrics when available.
- Runtime and session-specific logs.
- Active report/session management.
- Live MJPEG detection stream.
- Live Total, Defect, OK, and processing-FPS counters.

Restarting the detection runtime closes all active sessions. A browser may
briefly retain the last MJPEG frame after a restart; the API session list is
the source of truth.

---

## Interactive detection area

The active session page provides a rectangular ROI editor:

- **Draw:** press and drag on the stream to create a rectangle.
- **Move:** drag inside the orange rectangle.
- **Resize:** drag one of the four rounded corner handles.
- **Active area:** the image inside the rectangle remains clear and is passed
  to the detector.
- **Disabled area:** everything outside the rectangle is dimmed with a gray
  overlay.

The rectangle is applied in memory to the current session only. Its coordinates
are sent to the runtime through:

```text
PATCH /api/reports/{report_id}/roi
```

When the ROI changes, the runtime:

1. Validates the rectangle against the camera dimensions.
2. Updates the crop used by the box detector.
3. Clears stale detections and tracking state.
4. Runs the next detection pass on the new crop.

The shared `stream.yaml` file is not modified by dragging. The ROI is
session-scoped and disappears when the session closes.

---

## Runtime API

The runtime listens on port `8000`.

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/reports/open` | Start a detection session |
| `POST` | `/api/reports/close` | Close a detection session |
| `GET` | `/api/reports` | List active sessions |
| `GET` | `/api/reports/{report_id}/roi` | Read the current ROI |
| `PATCH` | `/api/reports/{report_id}/roi` | Update the ROI in the active session |
| `GET` | `/api/reports/{report_id}/stats` | Read live counters and processing FPS |
| `GET` | `/api/health` | Runtime health and active-session count |
| `GET` | `/api/health/storage` | Temporary storage health |
| `GET` | `/video_feed?report_id=<id>` | MJPEG stream for a session |

Example health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Example response:

```json
{
  "status": "healthy",
  "active_reports": 1
}
```

The dashboard service manager listens on port `8080` and provides lifecycle
routes including `/status`, `/start`, `/stop`, `/restart`, `/logs`, and
`/logs/{session_id}`.

---

## Temporary session storage

Each active session creates a temporary JSON file under:

```text
src/qc_scm/runtime/.session_data/
```

The file contains:

- Session metadata.
- Detection records.
- Detection timestamps.
- The final OK/DEFECT decision for each completed item.

Writes are atomic and protected by per-session locks. The file is deleted when
the session worker exits. The directory is ignored by Git and must not be used
as permanent archival storage.

Firebase is intentionally not part of the current data path. If long-term
history, multi-station synchronization, or remote reporting is required later,
it should be added as a separate persistence adapter rather than coupled to the
inference loop.

---

## Configuration

Runtime configuration lives in
[`src/qc_scm/runtime/config/`](src/qc_scm/runtime/config/).

| File | Responsibility |
|---|---|
| `app.yaml` | CORS origins and session defaults |
| `stream.yaml` | Camera source, dimensions, ROI defaults, and frame throttling |
| `box_detector.yaml` | Box model path and box confidence/IoU thresholds |
| `defect_detector.yaml` | Defect model, stability voting, and tracking settings |
| `api.yaml` | API-related runtime settings |

The default ROI values in `stream.yaml` are only the initial rectangle for a
new session. Operators can adjust the active rectangle from the dashboard
without changing configuration files.

Before promoting a different model, verify:

1. The model file exists at the configured path.
2. The detector device is available.
3. Confidence and IoU thresholds match the production camera.
4. The selected model belongs to the intended training version.

See the detailed [configuration guide](src/qc_scm/runtime/config/README.md).

---

## Training and model versions

Training is organized by version and detector type:

```text
training/versions/v2_1/
├── box_detector/
└── defect_detector/
```

Each version is kept isolated so that training data, scripts, exports, and
experiments are not mixed with another model generation.

Typical workflow:

```bash
conda activate qc
cd training/versions/v2_1/box_detector
python scripts/run_all.py
```

Run the corresponding defect detector pipeline from:

```text
training/versions/v2_1/defect_detector
```

Do not copy checkpoints between version directories. When a model is promoted
to runtime, update the relevant runtime YAML deliberately and record the
promotion in the commit that changes the configuration.

---

## Architecture

### Session lifecycle

Each report ID maps to one active `SessionWorker`. The session owns its camera,
pipeline configuration, ROI, counters, and temporary result file.

Closing a session stops its worker, releases the camera lock, unregisters the
pipeline, and removes its temporary result file.

### Frame processing

Camera capture is decoupled from inference. A bounded queue keeps the system
focused on recent frames instead of allowing latency to grow without limit.
GPU-bound model execution is coordinated by the `PipelineManager` inference
worker.

### Detection and classification

The first detector localizes the product inside the current ROI. The second
detector analyzes the localized product. Stability voting and tracking reduce
transient frame-level noise before a product is counted as OK or DEFECT.

### Stream rendering

The runtime publishes the camera stream through MJPEG. Dashboard-only elements
such as ROI controls, disabled-area shading, counters, and session metadata are
rendered in HTML/CSS so they remain interactive and do not become permanently
baked into the camera frames.

---

## Monitoring and troubleshooting

### Check both services

```bash
curl http://127.0.0.1:8080/status
curl http://127.0.0.1:8000/api/health
```

### The dashboard says the session is inactive

Check:

```bash
curl http://127.0.0.1:8000/api/reports
```

If the report ID is missing, the session was closed or the runtime was
restarted. Open a new session.

### The ROI is visible but does not update

Confirm:

1. The session still appears in `/api/reports`.
2. The runtime is healthy.
3. The browser has loaded the current dashboard files.
4. `PATCH` is allowed by the runtime CORS configuration.
5. The ROI response changes after the drag.

Use the browser developer tools to inspect the PATCH response and status code.

### The stream looks stale

An MJPEG browser element can retain its last decoded frame after the backend
session closes. Verify the backend session list instead of relying on the
visible frame. Refresh the session page after starting a new report.

### The camera or model does not start

Review the service log and verify:

- Camera permissions and source index.
- Model paths in the detector YAML files.
- The selected Conda environment.
- CUDA/device availability if using GPU inference.
- Runtime dependencies.

---

## Development checks

Run these checks before committing runtime or dashboard changes:

```bash
conda run -n qc python -m compileall -q src/qc_scm/runtime
node --check dashboard/app.js
git diff --check
```

For inline JavaScript in `dashboard/session.html`, validate the script block
with a JavaScript parser such as Node's `vm.Script`.

Keep changes focused. Do not commit:

- `.session_data/`
- service state files
- generated logs
- Python bytecode
- local credentials
- temporary camera captures

---

## Related documentation

- [Dashboard operations](dashboard/README.md)
- [Runtime guide](src/qc_scm/runtime/README.md)
- [Configuration guide](src/qc_scm/runtime/config/README.md)
- [Runtime core notes](src/qc_scm/runtime/core/README.md)
