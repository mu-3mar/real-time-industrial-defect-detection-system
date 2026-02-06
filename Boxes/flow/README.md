# QC-SCM Detection Service - API Documentation

## Overview
Multi-session quality control detection service with REST API for managing concurrent production line inspections.

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Open Session (Headless)
**POST** `/api/sessions/open`

Opens a new detection session WITHOUT GUI window (headless mode).

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
  "message": "Headless session started with camera 4"
}
```

---

### 2. Open Session with GUI
**POST** `/api/sessions/open_gui`

Opens a new detection session WITH GUI window showing camera feed and annotations.

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
  "message": "GUI session started with camera 4"
}
```

---

### 3. Close Session
**POST** `/api/sessions/close`

Closes an active detection session.

**Request:**
```json
{
  "report_id": "production_line_1"
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

### 4. List Sessions
**GET** `/api/sessions`

Lists all active sessions.

**Response:**
```json
{
  "sessions": [
    {
      "report_id": "production_line_1",
      "camera_source": "4",
      "status": "running",
      "started_at": "2026-02-06T05:30:00"
    }
  ]
}
```

---

### 5. Health Check
**GET** `/api/health`

Service health status.

**Response:**
```json
{
  "status": "healthy",
  "active_sessions": 1
}
```

---

## Detection Results

When a box is detected and exits the frame, results are automatically sent to:

**Endpoint:** `https://e908-156-197-189-2.ngrok-free.app/report`

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

### Backend Endpoint
Edit `configs/backend.yaml`:
```yaml
base_url: "https://e908-156-197-189-2.ngrok-free.app"
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

# Open headless session (no GUI)
curl -X POST http://localhost:8000/api/sessions/open \
  -H "Content-Type: application/json" \
  -d '{"report_id": "line_1", "camera_source": "4"}'

# Open GUI session (shows window)
curl -X POST http://localhost:8000/api/sessions/open_gui \
  -H "Content-Type: application/json" \
  -d '{"report_id": "line_1", "camera_source": "4"}'

# List active sessions
curl http://localhost:8000/api/sessions

# Close session
curl -X POST http://localhost:8000/api/sessions/close \
  -H "Content-Type: application/json" \
  -d '{"report_id": "line_1"}'
```

---

## Debugging

If all detection results are False:

1. **Open GUI session** to visually verify defects are detected
2. **Check server logs** for detection values
3. **Verify defect model** confidence threshold in `configs/defect_detector.yaml`
4. **Check voting threshold** - may need to lower if defects appear briefly
