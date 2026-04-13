# `Boxes/flow/config`

Runtime configuration for the flow server (`Boxes/flow/main.py` + `api/api_server.py`).

## Files

- `api.yaml`: bind address + port + log level (used by `main.py`).
- `app.yaml`: optional CORS origins list and `session_defaults` (telemetry/control initialization placeholders).
- `webrtc.yaml`: **Required** runtime WebRTC config (STUN/TURN URLs, TURN secret, `webrtc_mode`). Not committed (gitignored); create by copying from the example.
- `webrtc.example.yaml`: Template committed to the repo (no secrets). Copy to `webrtc.yaml` and fill in real values.
- `firebase.yaml`: Firebase Realtime Database settings
  - `service_account_path`: filename of the service account JSON in this folder (JSON is gitignored)
  - `database_url`: Realtime Database URL
- `firebase-service-account.json`: service account key (gitignored).
- `firebase-service-account.example.json`: placeholder template for the service account JSON.
- `box_detector.yaml`, `defect_detector.yaml`: detector model paths + thresholds.
- `stream.yaml`: default frame size + detection throttling.

## Secrets

- Do not commit `webrtc.yaml` or `firebase-service-account.json`.
- Prefer `firebase.yaml` for `database_url`; optional fallback: `firebase_config.json` in this directory. No environment variables are used.

## Configuration Reference

### Setup (after cloning)

Copy the example files and fill in real values. Do not commit files that contain secrets.

| Example file | Copy to | Contains |
|--------------|---------|----------|
| `firebase.example.yaml` | `firebase.yaml` | `service_account_path` + `database_url` |
| `webrtc.example.yaml` | `webrtc.yaml` | STUN/TURN URLs and TURN `secret` |
| `firebase_config.json.example` | `firebase_config.json` | Alternative: `database_url` in JSON (optional) |

Place your Firebase service account JSON in this folder; it is gitignored.

---

## Config files and keys

## app.yaml (optional)

| Key | Purpose |
|-----|---------|
| `cors_origins` | List of allowed origins. Empty = allow all. Non-empty = restrict to these origins with credentials. |
| `session_defaults` | Initialization placeholders written ONCE to Firebase (`session_info` node) when a session starts. E.g. `telemetry` and `control`. |

## api.yaml (optional)

| Key | Purpose |
|-----|---------|
| `host` | Bind address (default: 0.0.0.0) |
| `port` | Listen port (default: 8000) |
| `log_level` | Uvicorn log level (default: info) |

## webrtc.yaml (required at runtime)

| Key | Purpose |
|-----|---------|
| `mode` or `webrtc_mode` | Connection mode (same meaning): `auto`, `direct`, `stun`, `relay` |
| `stun.urls` | STUN server for NAT traversal |
| `turn` | TURN server (`urls` + `secret` for credential generation) |

**Mode enforcement:** The backend builds `iceServers` from this file and returns them with the chosen `mode` from GET `/api/config`. The frontend creates `RTCPeerConnection` with those servers and sets `iceTransportPolicy: "relay"` when `mode: relay` so the connection uses TURN only; other modes use only the provided servers (direct = empty, stun = STUN only, auto = STUN + TURN).

**Client config:** GET `/api/config` returns `webrtc.iceServers` (with temporary TURN credentials when applicable), `webrtc.mode`, and when mode is `relay`, `webrtc.iceTransportPolicy: "relay"` so the frontend can enforce TURN-only connections.

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

**Database URL (required):** Set `database_url` in `firebase.yaml`, or optionally in `firebase_config.json`. Do not commit `firebase_config.json` if it contains secrets.

Report open uses `report_id`, `camera_source`, `production_line_id`, and session config (`target_speed`, `max_temp`, `max_amps`) from POST `/api/reports/open`. 

**Firebase Structure (Updated):**
All data is written under the `report_id` root key:
1. `session_info`: Written ONCE on session start. Combines POST request configs (`target_speed`, etc.) and `app.yaml` `session_defaults` (`rpm_actual`, `machine_status`, etc.).
2. `defect` and `non_defect`: Detections are grouped here. Each detection is a child with an auto-generated key (`detection_id`), containing only `timestamp` (ISO 8601 string). No factory/line/station hierarchy.

## box_detector.yaml (required)

| Key | Purpose |
|-----|---------|
| `model_path` | Path to box detection model |
| `conf_thres` | Confidence threshold |
| `iou_thres` | IoU threshold for NMS |
| `device` | `auto` (CUDA→MPS→CPU), `cuda`, `mps`, or `cpu` (from this config only) |

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
