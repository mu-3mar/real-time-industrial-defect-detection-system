# QC-SCM API Endpoints — Backend Contract

This document describes all HTTP/WebRTC endpoints the **frontend (index.html)** calls against the **backend (Boxes/flow)**. Use it so the backend knows exactly what to **send** (request) and **receive** (response) for each endpoint.

**Base URL:** Configured by the client (e.g. `http://localhost:8000`). All paths below are relative to this base.

**Headers:** The client sends `Content-Type: application/json` for POST requests and expects JSON responses.

---

## 1. Health check

**Purpose:** Check if the service is up and how many sessions are active.

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/health` |
| **Request body** | None |
| **Query params** | None |

**Response (200 OK)** — JSON:

```json
{
  "status": "healthy",
  "active_sessions": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | e.g. `"healthy"` (client may display it uppercased). |
| `active_sessions` | integer | Number of active detection sessions. |

**Errors:**  
- `500` — Server error; client shows "UNREACHABLE" or similar.

---

## 2. List active sessions

**Purpose:** Get the list of all active detection sessions with metadata (for sidebar, grid, and session selection).

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/sessions` |
| **Request body** | None |
| **Query params** | None |

**Response (200 OK)** — JSON:

```json
{
  "sessions": [
    {
      "session_id": "string",
      "factory_id": "string",
      "factory_name": "string",
      "production_line_id": "string",
      "production_line_name": "string",
      "camera_id": "string",
      "camera_source": "string",
      "status": "active"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sessions` | array | List of session objects. |
| `sessions[].session_id` | string | Unique session identifier (client-generated). |
| `sessions[].factory_id` | string | Factory identifier. |
| `sessions[].factory_name` | string | Human-readable factory name. |
| `sessions[].production_line_id` | string | Production line identifier. |
| `sessions[].production_line_name` | string | Human-readable line name. |
| `sessions[].camera_id` | string | Camera identifier. |
| `sessions[].camera_source` | string | Camera source (URL, device index, or string ID). |
| `sessions[].status` | string | e.g. `"active"`. |

Optional per-session fields the client can use if present: `active_viewers`, `pipeline_fps`, `camera_fps_estimate`, `queue_latency_ms` (for metrics/UI).

**Errors:**  
- `500` — Server error; client may keep previous session list.

---

## 3. Open session

**Purpose:** Start a new headless detection session. All metadata is provided by the client; the backend does not store static factory/line config.

| | |
|---|---|
| **Method** | `POST` |
| **Path** | `/api/sessions/open` |
| **Request body** | JSON (see below) |
| **Query params** | None |

**Request body** — JSON:

```json
{
  "factory_id": "string",
  "factory_name": "string",
  "production_line_id": "string",
  "production_line_name": "string",
  "camera_id": "string",
  "camera_source": "string or number",
  "station_id": "string",
  "session_id": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `factory_id` | string | Yes | Factory identifier (e.g. normalized, lowercase, underscores). |
| `factory_name` | string | Yes | Human-readable factory name. |
| `production_line_id` | string | Yes | Production line identifier. |
| `production_line_name` | string | Yes | Human-readable line name. |
| `camera_id` | string | Yes | Camera identifier. |
| `camera_source` | string or number | Yes | Camera source: URL (rtsp/http), device index (e.g. 0), or string ID. |
| `station_id` | string | Yes | Station identifier. |
| `session_id` | string | Yes | Unique session ID (client-generated, e.g. UUID). |

**Response (200 OK)** — JSON:

```json
{
  "status": "success",
  "session_id": "string",
  "message": "Session started with camera <camera_source>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` on success. |
| `session_id` | string | Same as in request. |
| `message` | string | Human-readable message. |

**Errors:**  
- `400` — Validation or business error (e.g. camera in use, invalid payload); `detail` in response body.  
- `500` — Internal server error.

---

## 4. Close session

**Purpose:** Close an active detection session by `session_id`. Optional `camera_source` helps when the same camera is used by multiple sessions.

| | |
|---|---|
| **Method** | `POST` |
| **Path** | `/api/sessions/close` |
| **Request body** | JSON (see below) |
| **Query params** | None |

**Request body** — JSON:

```json
{
  "session_id": "string",
  "camera_source": "string or number (optional)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | Yes | Session to close. |
| `camera_source` | string or number | No | Optional; used for matching when closing. |

**Response (200 OK)** — JSON:

```json
{
  "status": "success",
  "session_id": "string",
  "message": "Session closed successfully"
}
```

**Errors:**  
- `404` — Session not found or camera mismatch; `detail` in response body.  
- `500` — Internal server error.

---

## 5. Metrics (observability)

**Purpose:** Optional metrics for dashboard (sessions count, viewers, WebRTC connections, FPS, latency). Does not affect core API contract.

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/metrics` |
| **Request body** | None |
| **Query params** | None |

**Response (200 OK)** — JSON:

```json
{
  "active_sessions": 0,
  "active_viewers": 0,
  "webrtc_connections": 0,
  "pipeline_fps": 0.0,
  "camera_fps_estimate": 0.0,
  "queue_latency_ms": 0.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `active_sessions` | integer | Number of active sessions. |
| `active_viewers` | integer | Total active viewers (e.g. WebRTC viewers). |
| `webrtc_connections` | integer | Number of active WebRTC peer connections. |
| `pipeline_fps` | number | Average pipeline FPS across sessions. |
| `camera_fps_estimate` | number | Average camera FPS estimate across sessions. |
| `queue_latency_ms` | number | Average queue latency in milliseconds. |

**Errors:**  
- `500` — Server error; client may show "Metrics unavailable".

---

## 6. Client config (WebRTC)

**Purpose:** Get WebRTC ICE server config (STUN/TURN) and mode so the client can create `RTCPeerConnection` and connect to the video stream.

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/config` |
| **Request body** | None |
| **Query params** | None |

**Response (200 OK)** — JSON:

```json
{
  "webrtc": {
    "iceServers": [
      { "urls": "stun:host:3478" },
      { "urls": "turn:host:3478", "username": "...", "credential": "..." }
    ],
    "webrtc_mode": "auto"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `webrtc` | object | WebRTC config. |
| `webrtc.iceServers` | array | List of ICE servers; each has `urls`, and optionally `username` and `credential` for TURN. |
| `webrtc.webrtc_mode` | string | One of: `"auto"`, `"direct"`, `"stun"`, `"relay"`. Client may set `iceTransportPolicy` from this. |

Client uses `iceServers` (or `ice_servers`) and `webrtc_mode` (or `mode`) from `response.webrtc` or from the root if `webrtc` is missing.

**Errors:**  
- Any error: client may fall back to empty ICE servers and mode `"auto"`.

---

## 7. WebRTC offer (SDP exchange)

**Purpose:** Exchange SDP offer/answer to establish a WebRTC connection and attach the detection video track for the given session.

| | |
|---|---|
| **Method** | `POST` |
| **Path** | `/webrtc/offer` |
| **Request body** | JSON (see below) |
| **Query params** | None |

**Request body** — JSON:

```json
{
  "sdp": "string (SDP text)",
  "type": "offer",
  "session_id": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sdp` | string | Yes | SDP string from `RTCPeerConnection.localDescription.sdp`. |
| `type` | string | Yes | Session description type, e.g. `"offer"`. |
| `session_id` | string | Yes | Active session ID (must exist from `POST /api/sessions/open`). |

**Response (200 OK)** — JSON:

```json
{
  "sdp": "string (SDP text)",
  "type": "answer"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sdp` | string | SDP string for the answer. |
| `type` | string | `"answer"`. |

Client sets this as `remoteDescription` and then receives the video track via `ontrack`. The backend attaches the detection stream as a video track to this peer connection.

**Errors:**  
- `404` — Session not found; `detail` in response body.  
- `500` — Internal server error (e.g. WebRTC/signaling failure).

---

## Summary table

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/health` | — | `{ status, active_sessions }` |
| GET | `/api/sessions` | — | `{ sessions: [...] }` |
| POST | `/api/sessions/open` | `factory_id`, `factory_name`, `production_line_id`, `production_line_name`, `camera_id`, `camera_source`, `station_id`, `session_id` | `{ status, session_id, message }` |
| POST | `/api/sessions/close` | `session_id`, optional `camera_source` | `{ status, session_id, message }` |
| GET | `/api/metrics` | — | `active_sessions`, `active_viewers`, `webrtc_connections`, `pipeline_fps`, `camera_fps_estimate`, `queue_latency_ms` |
| GET | `/api/config` | — | `{ webrtc: { iceServers, webrtc_mode } }` |
| POST | `/webrtc/offer` | `sdp`, `type`, `session_id` | `{ sdp, type: "answer" }` |

---

## Error response shape

On 4xx/5xx the client may receive a JSON body with a message, e.g.:

```json
{
  "detail": "Session not found"
}
```

or (for validation):

```json
{
  "detail": [ "field error messages" ]
}
```

The client uses `detail` or `message` when present to show an error to the user.
