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
| `turn` | TURN server (backend only; not exposed to clients) |
| `debug_turn_only` | UNUSED (aiortc has no iceTransportPolicy) |

## stream.yaml (required)

| Key | Purpose |
|-----|---------|
| `width` | Frame width |
| `height` | Frame height |

Source comes from `camera_source` in POST /api/sessions/open.

## mqtt.yaml (required)

| Key | Purpose |
|-----|---------|
| `broker.host` | MQTT broker hostname |
| `broker.port` | MQTT broker port |
| `broker.username` | MQTT username |
| `broker.password` | MQTT password |
| `client.client_id_prefix` | Client ID prefix |
| `client.keepalive` | Keepalive seconds |
| `client.clean_session` | Clean session flag |
| `topics.insights_pattern` | Topic pattern for insights |

## box_detector.yaml (required)

| Key | Purpose |
|-----|---------|
| `model_path` | Path to box detection model |
| `conf_thres` | Confidence threshold |
| `iou_thres` | IoU threshold for NMS |
| `device` | Device (0 = GPU, cpu = CPU) |

## defect_detector.yaml (required)

| Key | Purpose |
|-----|---------|
| `model_path` | Path to defect detection model |
| `conf_thres` | Confidence threshold |
| `iou_thres` | IoU threshold |
| `device` | Device |
| `tracking` | IoU match, bbox smoothing |
| `stability` | Entry/exit, voting params |
| `rendering` | Visibility threshold |
