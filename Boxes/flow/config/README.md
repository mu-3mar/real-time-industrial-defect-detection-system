# `Boxes/flow/config`

Runtime configuration for the flow server (`Boxes/flow/main.py` + `api/api_server.py`).

## Files

- `api.yaml`: bind address + port + log level (used by `main.py`).
- `app.yaml`: optional CORS origins list.
- `webrtc.yaml`: STUN/TURN + TURN **secret** and `webrtc_mode` (gitignored).
- `webrtc.example.yaml`: template for `webrtc.yaml`.
- `firebase.yaml`: Firebase Realtime Database settings
  - `service_account_path`: filename of the service account JSON in this folder (JSON is gitignored)
  - `database_url`: Realtime Database URL
- `firebase-service-account.json`: service account key (gitignored).
- `firebase-service-account.example.json`: placeholder template for the service account JSON.
- `box_detector.yaml`, `defect_detector.yaml`: detector model paths + thresholds.
- `stream.yaml`: default frame size + detection throttling.

## Secrets

- Do not commit `webrtc.yaml` or `firebase-service-account.json`.
- Prefer `firebase.yaml` for `database_url`; server also supports `FIREBASE_DATABASE_URL` env and `firebase_config.json` fallback.
# Configuration Reference

## Setup (after cloning)

Copy the example files and fill in real values. Do not commit files that contain secrets.

| Example file | Copy to | Contains |
|--------------|---------|----------|
| `.env.example` | `.env` | `FIREBASE_DATABASE_URL`, optional overrides |
| `firebase.example.yaml` | `firebase.yaml` | `credentials_path` for Firebase service account JSON |
| `webrtc.example.yaml` | `webrtc.yaml` | STUN/TURN URLs and TURN `secret` |
| `firebase_config.json.example` | `firebase_config.json` | Alternative: `FIREBASE_DATABASE_URL` in JSON |

Place your Firebase service account JSON in this folder; it is gitignored.

---

## Config files and keys

## app.yaml (optional)

| Key | Purpose |
|-----|---------|
| `cors_origins` | List of allowed origins. Empty = allow all. Non-empty = restrict to these origins with credentials. |

## api.yaml (optional)

| Key | Purpose |
|-----|---------|
| `host` | Bind address (default: 0.0.0.0) |
| `port` | Listen port (default: 8000) |
| `log_level` | Uvicorn log level (default: info) |

## webrtc.yaml (optional, has defaults)

| Key | Purpose |
|-----|---------|
| `stun.urls` | STUN server for NAT traversal |
| `turn` | TURN server (urls + secret for credential generation) |
| `webrtc_mode` | `auto` (host+STUN+TURN), `direct` (host only), `stun` (host+STUN), `relay` (host+TURN) |

**Client config:** The `/api/config` endpoint returns STUN URLs, TURN URLs, and **temporary TURN credentials** (short-lived, e.g. 5 minutes) so the frontend can establish WebRTC connections. Credentials are time-limited for security.

## stream.yaml (required)

| Key | Purpose |
|-----|---------|
| `width` | Frame width |
| `height` | Frame height |

Source comes from `camera_source` in POST /api/reports/open.

## firebase.yaml (required)

| Key | Purpose |
|-----|---------|
| `service_account_path` | Filename of Firebase service account JSON in `config/` (gitignored). |
| `database_url` | Firebase Realtime Database URL (e.g. europe-west1 `*.firebasedatabase.app`). |

**Database URL (required):** Prefer `database_url` in `firebase.yaml`. As a fallback, `FIREBASE_DATABASE_URL` can be set in `.env` or in `firebase_config.json`. Do not commit `.env` or `firebase_config.json`.

Report open uses `report_id`, `camera_source`, and `production_line_id` from POST /api/reports/open. Detections are written to Firebase Realtime Database under the report only: root key is `report_id`; each detection is a child with an auto-generated key (`detection_id`), containing only `defect` (boolean) and `timestamp` (ISO 8601 string). No factory/line/station hierarchy.

## box_detector.yaml (required)

| Key | Purpose |
|-----|---------|
| `model_path` | Path to box detection model |
| `conf_thres` | Confidence threshold |
| `iou_thres` | IoU threshold for NMS |
| `device` | `auto` (CUDA→MPS→CPU), `cuda`, `mps`, `cpu`; override with `QC_SCM_FLOW_DEVICE` |

## defect_detector.yaml (required)

| Key | Purpose |
|-----|---------|
| `model_path` | Path to defect detection model |
| `model_version` | Optional; version string written to Realtime Database insights (default "1.0") |
| `conf_thres` | Confidence threshold |
| `iou_thres` | IoU threshold |
| `device` | Same as box_detector; both use one resolved device |
| `tracking` | IoU match, bbox smoothing |
| `stability` | Entry/exit, voting params |
| `rendering` | Visibility threshold |
