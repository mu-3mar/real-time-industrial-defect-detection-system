# Pipeline FPS Diagnostic Report

This document describes the **runtime diagnostics** added to identify the source of the observed FPS drop (~24 FPS) after the threaded pipeline migration. No architecture or detection logic was changed—only instrumentation.

---

## 1. Metrics Added

| Metric | Where measured | Log key / meaning |
|--------|----------------|-------------------|
| **camera_capture_fps** | `CamStream._capture_loop` (camera thread) | Actual FPS at which the camera is capturing. Logged every 10s in capture thread and passed into diagnostics from feeder. |
| **frame_enqueue_rate** | Frames successfully put into `frame_queue` per second | Rate at which the camera feeder is pushing frames into the shared frame queue (windowed over 10s). |
| **frame_queue_drops** | `put_frame()` returning False (queue full) | Total count of frames dropped because `frame_queue` was full. High drops → inference is the bottleneck. |
| **inference_fps** | Frames processed by `pipeline.run_step()` per second | Throughput of the single inference thread (windowed over 10s). This is the effective detection FPS. |
| **inference_latency_avg_ms** | Time inside `run_step()` per frame | Average time per frame in GPU inference + drawing. |
| **inference_latency_max_ms** | Max time inside `run_step()` in the window | Peak latency; spikes can indicate GPU stalls or CPU contention. |
| **result_queue_drops** | `result_queue.put_nowait()` Full | Frames dropped because result queue was full. High drops → result consumer or WebRTC is slow. |
| **webrtc_avg_ms** | Time in `track.update_frame(canvas)` | Average time per frame to push to WebRTC track. High values → encoding or lock contention. |
| **queue_sizes** | `frame_queue.qsize()`, `result_queue.qsize()`, `firebase_queue.qsize()` | Snapshot at log time. frame=5 or result=5 often → queue at maxsize (backpressure). |

---

## 2. Where to See the Numbers

- **Startup:** One line for GPU/device:
  - `[DIAG] inference device=cuda | torch.cuda.is_available()=True`
- **Every 10s in the camera capture thread:**
  - `[DIAG] camera_capture_fps=30.0 (capture thread, N frames total)`
- **Every 10s in the inference thread (combined pipeline stats):**
  - `[DIAG] camera_capture_fps=... | frame_enqueue_rate=... | frame_queue_drops=... | inference_fps=... | inference_latency_avg_ms=... | inference_latency_max_ms=... | result_queue_drops=... | webrtc_avg_ms=... | queue_sizes: frame=... result=... firebase=...`

Run the server, open a session, and let it run for at least 20–30 seconds to get several diagnostic lines.

---

## 3. How to Identify the Bottleneck

1. **Camera limited to ~24 FPS**
   - Check: `camera_capture_fps` in the capture-thread log and in the combined log.
   - If `camera_capture_fps ≈ 24` and never higher → camera/driver/hardware limit (e.g. 24 FPS cap). Check V4L2, resolution, and format (MJPG vs YUYV).

2. **Frame queue pressure (inference can’t keep up)**
   - Check: `frame_queue_drops` increasing, `queue_sizes: frame=5` (often at max).
   - If `frame_enqueue_rate` > `inference_fps` and `frame_queue_drops` is high → **inference is the bottleneck** (single GPU thread is the limiter).

3. **Inference throughput**
   - Check: `inference_fps` (effective detection FPS).
   - If `inference_fps ≈ 24` and `inference_latency_avg_ms ≈ 41` (1000/24) → each frame takes ~41 ms; GPU or run_step() is the bottleneck. Check GPU utilization and that ONNX/Ultralytics are using the GPU (device and torch.cuda.is_available() in startup log).

4. **Result queue pressure (WebRTC/consumer can’t keep up)**
   - Check: `result_queue_drops` increasing, `queue_sizes: result=5`.
   - If `inference_fps` is high but `result_queue_drops` is high → **result consumer or WebRTC is the bottleneck**. Then check `webrtc_avg_ms`.

5. **WebRTC encoding / update_frame**
   - Check: `webrtc_avg_ms`.
   - If `webrtc_avg_ms` is large (e.g. > 20 ms) → `track.update_frame()` or downstream encoding is slow (CPU-bound encoding or lock contention).

6. **Unnecessary sleeps**
   - `CamStream._capture_loop`: no sleep in the success path; only `time.sleep(0.05)` on read failure.
   - Camera feeder: `time.sleep(0.001)` only when no frame available; no sleep after a successful put. So no extra throttle in the hot path.

7. **GPU in use**
   - Startup log: `[DIAG] inference device=cuda | torch.cuda.is_available()=True`.
   - If device is `cuda` and torch reports True, models are configured for GPU; actual provider is Ultralytics/ONNX (device passed to `model(..., device=device)`).

8. **Frame copies**
   - No extra numpy copies were added by diagnostics. Only timers and counter updates.

---

## 4. Example Interpretation

- **Observed:** `inference_fps ≈ 24`, `camera_capture_fps ≈ 30`, `frame_queue_drops` high, `frame=5` often.
- **Conclusion:** Camera is fine (~30 FPS). Inference is doing ~24 FPS. Frame queue is full and dropping frames. **Bottleneck: single inference thread** (GPU or run_step() latency). Next step: profile `run_step()` (box + defect inference and drawing) and GPU utilization.

- **Observed:** `inference_fps ≈ 60`, `result_queue_drops` high, `webrtc_avg_ms` high.
- **Conclusion:** Inference is fast; result consumer or WebRTC is slow. **Bottleneck: WebRTC / update_frame**. Next step: reduce resolution, encoder settings, or lock contention.

---

## 5. Files Touched (instrumentation only)

| File | Change |
|------|--------|
| `core/pipeline_diagnostics.py` | **New.** Thread-safe counters and `maybe_log()` for all metrics. |
| `core/stream.py` | Log `camera_capture_fps` every 10s in capture loop. |
| `core/session_worker.py` | Feeder calls `get_diagnostics().set_camera_capture_fps()`, `record_frame_enqueue(ok)`. |
| `core/pipeline_manager.py` | Inference worker: time `run_step()`, `record_inference()`, `record_result_queue_drop()`, `maybe_log()` with queue sizes. Result consumer: time `track.update_frame()`, `record_webrtc_update()`. |
| `api/api_server.py` | After warmup, log `[DIAG] inference device=... | torch.cuda.is_available()=...`. |

---

## 6. Summary

- **Camera FPS:** `camera_capture_fps` (capture thread + combined log).
- **Inference FPS:** `inference_fps` (windowed frames per second in inference thread).
- **Queue utilization:** `queue_sizes: frame=X result=Y firebase=Z`; values at 5 (max) indicate backpressure.
- **Inference latency:** `inference_latency_avg_ms`, `inference_latency_max_ms`.
- **WebRTC cost:** `webrtc_avg_ms`.
- **Frame drops:** `frame_queue_drops` (camera feeder), `result_queue_drops` (inference worker).

Use these to pinpoint whether the limit is **camera**, **inference (GPU)**, or **WebRTC/result consumer**, then optimize that stage without refactoring the queue architecture.
