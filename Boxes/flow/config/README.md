# Configuration Reference

Each config file and its keys:

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

## stream.yaml (required)

| Key | Purpose |
|-----|---------|
| `width` | Frame width |
| `height` | Frame height |

Source comes from `camera_source` in POST /api/sessions/open.

## firebase.yaml (required)

| Key | Purpose |
|-----|---------|
| `credentials_path` | Filename of Firebase service account JSON in `config/` |

**Database URL (required):** Set `FIREBASE_DATABASE_URL` in `.env` (copy from `.env.example`) or in `config/firebase_config.json` (e.g. `{"FIREBASE_DATABASE_URL": "https://..."}`). Do not hardcode in source.

All factory/line/session metadata is provided in POST /api/sessions/open; nothing is hardcoded. Insights are written to Firebase Realtime Database at:
`factories/{factory_id}/production_lines/{production_line_id}/sessions/{session_id}/insights/`.

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
