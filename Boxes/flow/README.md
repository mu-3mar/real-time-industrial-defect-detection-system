# QC-SCM Detection Service - API Documentation

## Overview

Multi-session quality control detection service with REST API for managing concurrent production line inspections. The server exposes **four core endpoints**: open session, close session, list sessions, and view running session.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Open Session

**POST** `/api/sessions/open`

Opens a new detection session (headless). The session runs in the background with the given camera source.

**Request:**

```json
{
  "report_id": "production_line_1",
  "camera_source": "4"
}
```

**Response:**

```json
{
  "status": "success",
  "report_id": "production_line_1",
  "message": "Session started with camera 4"
}
```

---

### 2. Close Session

**POST** `/api/sessions/close`

Closes an active detection session. Both `report_id` and `camera_source` must match the running session.

**Request:**

```json
{
  "report_id": "production_line_1",
  "camera_source": "4"
}
```

**Response:**

```json
{
  "status": "success",
  "report_id": "production_line_1",
  "message": "Session closed successfully"
}
```

---

### 3. List Sessions

**GET** `/api/sessions`

Returns all active sessions.

**Response:**

```json
{
  "sessions": [
    {
      "report_id": "production_line_1",
      "camera_source": "4",
      "status": "running",
      "started_at": "2026-02-06T05:30:00.000Z",
      "viewer_attached": false
    }
  ]
}
```

---

### 4. View Running Session

**POST** `/api/sessions/view`

Opens a viewer window for an **already running** session. The window shows the live camera feed with annotations (boxes, defects, stats). Use this to monitor a session that was started with **Open Session** (headless). If the session is not running, the API returns 404.

**Request:**

```json
{
  "report_id": "production_line_1",
  "camera_source": "4"
}
```

**Response:**

```json
{
  "status": "success",
  "report_id": "production_line_1",
  "message": "Viewer opened for session (camera 4)"
}
```

---

### Health Check

**GET** `/api/health`

Service health and active session count.

**Response:**

```json
{
  "status": "healthy",
  "active_sessions": 1
}
```

---

## Detection Results

When a box is detected and exits the frame, results are sent to the backend configured in `configs/backend.yaml`.

**Payload:**

```json
{
  "report_id": "production_line_1",
  "value": true
}
```

- `value: true` = Defect detected  
- `value: false` = Non-defect (OK)

---

## Configuration

### Backend

Edit `configs/backend.yaml`:

```yaml
base_url: "https://your-backend.example.com"
result_endpoint: "/report"
timeout: 5
max_retries: 3
```

### API Server

Edit `configs/api_server.yaml`:

```yaml
host: "0.0.0.0"
port: 8000
log_level: "info"
```

---

## Usage Examples

```bash
# Start server
python main.py

# 1. Open session (headless)
curl -X POST http://localhost:8000/api/sessions/open \
  -H "Content-Type: application/json" \
  -d '{"report_id": "line_1", "camera_source": "4"}'

# 2. List active sessions
curl http://localhost:8000/api/sessions

# 3. View running session (opens GUI window with annotations)
curl -X POST http://localhost:8000/api/sessions/view \
  -H "Content-Type: application/json" \
  -d '{"report_id": "line_1", "camera_source": "4"}'

# 4. Close session
curl -X POST http://localhost:8000/api/sessions/close \
  -H "Content-Type: application/json" \
  -d '{"report_id": "line_1", "camera_source": "4"}'
```

---

## Debugging

- Use **View Running Session** to open a live annotated window for an existing session and verify detections.
- Check server logs for detection values and errors.
- Verify defect model confidence in `configs/defect_detector.yaml`.
- Adjust voting threshold in defect stability config if defects appear only briefly.
