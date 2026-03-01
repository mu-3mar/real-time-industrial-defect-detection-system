# Frontend Integration Guide

=== FRONTEND INTEGRATION GUIDE ===

This document provides everything a frontend developer needs to integrate with the QC-SCM Detection Service API.

---

## API Base URL

The API base URL is the root of the backend service. Use it as the prefix for all endpoints.

| Scenario | API Base URL |
|----------|--------------|
| **Local** (same machine) | `http://localhost:8000` |
| **LAN** (another machine on network) | `http://<server-ip>:8000` |
| **Server** (deployed) | `http://<domain>:8000` or `https://<domain>` |
| **ngrok** (optional, for remote dev) | `https://<your-subdomain>.ngrok-free.dev` |

**Important:** Replace the port if you changed it in `config/api.yaml` (default: 8000).

---

## Required Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check, active session count |
| POST | `/api/sessions/open` | Start a detection session |
| POST | `/api/sessions/close` | Stop a detection session |
| GET | `/api/sessions` | List active sessions |
| GET | `/api/config` | WebRTC ICE config (STUN servers) |
| POST | `/webrtc/offer` | WebRTC signaling (SDP exchange) |

---

## Example Fetch Configuration

```javascript
const API_BASE = "http://localhost:8000";  // Change for your environment

// Health check
const health = await fetch(`${API_BASE}/api/health`);
const { status, active_sessions } = await health.json();

// Open session
const openRes = await fetch(`${API_BASE}/api/sessions/open`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    report_id: "line1_abc123",
    production_line: "line1",
    camera_source: 0,  // or "/dev/video0" on Linux
  }),
});

// Close session
await fetch(`${API_BASE}/api/sessions/close`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    report_id: "line1_abc123",
    production_line: "line1",
    camera_source: 0,
  }),
});

// List sessions
const sessionsRes = await fetch(`${API_BASE}/api/sessions`);
const { sessions } = await sessionsRes.json();

// Get WebRTC config (STUN servers)
const configRes = await fetch(`${API_BASE}/api/config`);
const { webrtc } = await configRes.json();
const iceServers = webrtc.iceServers;
```

---

## WebRTC Signaling Usage

1. **Open a session** first via `POST /api/sessions/open`.
2. **Get ICE config** via `GET /api/config` (use `webrtc.iceServers` for `RTCPeerConnection`).
3. **Create offer** on the client, then send it to `POST /webrtc/offer`:

```javascript
const pc = new RTCPeerConnection({ iceServers: iceServers });

// Add transceiver to receive video
pc.addTransceiver("video", { direction: "recvonly" });

const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const offerRes = await fetch(`${API_BASE}/webrtc/offer`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    report_id: "line1_abc123",
  }),
});

const { sdp, type } = await offerRes.json();
await pc.setRemoteDescription(new RTCSessionDescription({ sdp, type }));
```

4. **Display video** from the `track` event on the peer connection.

---

## Environment Examples

### Local (same machine)

```javascript
const API_BASE = "http://localhost:8000";
```

### LAN (frontend on another machine)

```javascript
// Replace 192.168.1.100 with your backend server's IP
const API_BASE = "http://192.168.1.100:8000";
```

### Server deployment

```javascript
// Use your domain; HTTPS if behind reverse proxy
const API_BASE = "https://api.example.com";
```

### ngrok (optional, for remote development)

```javascript
// Only when using ngrok to expose local backend
const API_BASE = "https://abc123.ngrok-free.dev";
```

---

## CORS

The backend allows all origins by default. No CORS configuration is required for integration.

If you restrict CORS in `config/app.yaml` (production), add your frontend origin to `cors_origins`.

---

## API Documentation

Interactive Swagger docs: `http://<API_BASE>/docs`
