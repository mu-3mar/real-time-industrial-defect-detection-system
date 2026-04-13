# `Boxes/flow/api`

FastAPI layer for the flow (detection) service.

## Endpoints (surface)

- `POST /api/reports/open`: open a report (idempotent per `production_line_id`)
  - Body: `{ "report_id": "...", "camera_source": "...|0", "production_line_id": "...", "target_speed": 1500, "max_temp": 90, "max_amps": 40 }`
- `POST /api/reports/close`: close a report (idempotent)
  - Body: `{ "report_id": "..." }`
- `GET /api/reports`: list active (open) reports
  - Response: `[ { "report_id": "...", "viewers_count": 0 } ]`
- `GET /api/health`: health + active report count
- `GET /api/config`: client WebRTC config (`webrtc.iceServers` with **temporary** TURN creds)
- `POST /webrtc/offer`: WebRTC offer/answer for a specific `report_id`

## Key files

- `api_server.py`: FastAPI app, config loading, WebRTC offer handling.
