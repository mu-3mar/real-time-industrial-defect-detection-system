# `Boxes/flow/api`

FastAPI layer for the QC-SCM detection service.

## Endpoints

| Method | Path | Description |
| :----- | :--- | :---------- |
| `POST` | `/api/reports/open` | Start a new detection session. |
| `POST` | `/api/reports/close` | End an active session. |
| `GET`  | `/api/reports` | List active (open) sessions. |
| `GET`  | `/api/health` | Service health + active report count. |
| `GET`  | `/video_feed?report_id=<id>` | MJPEG annotated live stream for a session. |

### `POST /api/reports/open` body

```json
{
  "report_id": "my-report",
  "camera_source": "/dev/video0",
  "production_line_id": "line-1",
  "target_speed": 160,
  "max_temp": 80,
  "max_amps": 10,
  "command_state": "on",
  "emergency_state": "normal"
}
```

### `POST /api/reports/close` body

```json
{ "report_id": "my-report" }
```

## Key files

- `api_server.py`: FastAPI application, config loading, and all endpoint handlers.
