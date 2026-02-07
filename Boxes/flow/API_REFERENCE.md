# API Endpoints Reference

## Base URL
```
http://localhost:8000
```

## Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/sessions/open` | Start a new detection session |
| POST | `/api/sessions/close` | Stop a detection session |
| POST | `/api/sessions/view` | Attach viewer to running session |
| **POST** | **`/api/sessions/view/close`** | **Close viewer (NEW)** |
| GET | `/api/sessions` | List active sessions |
| GET | `/api/health` | Service health check |

---

## Detailed Endpoints

### 1. Open Session
**POST** `/api/sessions/open`

Start a new detection session for a specific camera.

**Request:**
```json
{
  "report_id": "session-123",
  "camera_source": 0
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "report_id": "session-123",
  "message": "Session started with camera 0"
}
```

**Error Responses:**
- `400 Bad Request` - Session already exists or camera in use
- `500 Internal Server Error` - Startup failure

---

### 2. Close Session
**POST** `/api/sessions/close`

Stop and cleanup a detection session.

**Request:**
```json
{
  "report_id": "session-123",
  "camera_source": 0
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "report_id": "session-123",
  "message": "Session closed successfully"
}
```

**Error Responses:**
- `404 Not Found` - Session not found or camera mismatch
- `500 Internal Server Error` - Cleanup failure

---

### 3. Open Viewer
**POST** `/api/sessions/view`

Attach a viewer window to a running session to visualize detection results.

**Request:**
```json
{
  "report_id": "session-123",
  "camera_source": 0
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "report_id": "session-123",
  "message": "Viewer opened for session (camera 0)"
}
```

**Error Responses:**
- `404 Not Found` - Session not running
- `500 Internal Server Error` - Viewer creation failed

---

### 4. Close Viewer ✨ NEW
**POST** `/api/sessions/view/close`

Detach viewer window from a session. The session continues running in background.

**Request:**
```json
{
  "report_id": "session-123",
  "camera_source": 0
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "report_id": "session-123",
  "message": "Viewer closed successfully"
}
```

**Error Responses:**
- `404 Not Found` - Session not found or no viewer attached
- `500 Internal Server Error` - Viewer cleanup failed

**Key Benefits:**
- Close viewer without stopping detection
- Free up display resources
- Switch between multiple session viewers
- Session continues running and collecting results

---

### 5. List Sessions
**GET** `/api/sessions`

Get list of all active detection sessions.

**Response (200 OK):**
```json
{
  "sessions": [
    {
      "report_id": "session-123",
      "camera_source": "0",
      "status": "running",
      "started_at": "2024-02-07T10:30:45Z",
      "viewer_attached": true
    },
    {
      "report_id": "session-456",
      "camera_source": "/dev/video0",
      "status": "running",
      "started_at": "2024-02-07T10:35:20Z",
      "viewer_attached": false
    }
  ]
}
```

---

### 6. Health Check
**GET** `/api/health`

Check service health and active session count.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "active_sessions": 2
}
```

---

## Usage Examples

### Start a detection session
```bash
curl -X POST "http://localhost:8000/api/sessions/open" \
  -H "Content-Type: application/json" \
  -d '{"report_id": "box-001", "camera_source": 0}'
```

### View the running session
```bash
curl -X POST "http://localhost:8000/api/sessions/view" \
  -H "Content-Type: application/json" \
  -d '{"report_id": "box-001", "camera_source": 0}'
```

### Close the viewer (detection continues)
```bash
curl -X POST "http://localhost:8000/api/sessions/view/close" \
  -H "Content-Type: application/json" \
  -d '{"report_id": "box-001", "camera_source": 0}'
```

### List all active sessions
```bash
curl "http://localhost:8000/api/sessions"
```

### Stop the session
```bash
curl -X POST "http://localhost:8000/api/sessions/close" \
  -H "Content-Type: application/json" \
  -d '{"report_id": "box-001", "camera_source": 0}'
```

### Check service health
```bash
curl "http://localhost:8000/api/health"
```

---

## Workflow Examples

### Single Camera Detection with Viewer
```
1. POST /api/sessions/open (start detection)
2. POST /api/sessions/view (open viewer window)
3. ... detection runs ...
4. POST /api/sessions/view/close (close window, keep detecting)
5. POST /api/sessions/close (stop session)
```

### Multi-Camera with Viewer Switching
```
1. POST /api/sessions/open (camera 0)
2. POST /api/sessions/open (camera 1)
3. POST /api/sessions/view (view camera 0)
4. ... monitor camera 0 ...
5. POST /api/sessions/view/close (close camera 0 viewer)
6. POST /api/sessions/view (view camera 1)
7. ... monitor camera 1 ...
8. POST /api/sessions/close (stop both)
```

### Background Detection with Periodic Viewing
```
1. POST /api/sessions/open (start detection)
2. ... detection running headless ...
3. POST /api/sessions/view (attach viewer for quick check)
4. ... see live results ...
5. POST /api/sessions/view/close (close viewer, continue detecting)
6. ... detection continues ...
7. POST /api/sessions/close (stop when done)
```

---

## Error Handling

All endpoints return standard HTTP status codes:

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Operation completed |
| 400 | Bad Request | Invalid input, check request body |
| 404 | Not Found | Session/resource doesn't exist |
| 500 | Server Error | Check logs and retry |

---

## Notes

- **Session ID:** Unique `report_id` for each detection session
- **Camera Source:** Can be integer (0, 1, ...) or path ("/dev/video0")
- **Headless Mode:** All sessions run headless; viewer is optional
- **Concurrent Sessions:** Multiple sessions can run simultaneously on different cameras
- **Result Callback:** Detection results are automatically sent to backend when boxes exit

