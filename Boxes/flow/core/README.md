# `Boxes/flow/core`

Core runtime for reports (internally still named “sessions”), inference, WebRTC frame fan-out, and Firebase publishing.

## Main pieces

- `session_manager.py`: in-memory report lifecycle (`report_id`), camera locks, line→report mapping.
- `session_worker.py`: per-report worker thread that feeds frames into the shared pipeline, and calls `publish_session_info` on session start.
- `pipeline_manager.py`: shared queues + workers (inference, result consumer, Firebase worker).
- `pipeline.py`: per-report pipeline (OpenCV stream, detectors, tracking/state, visualizer).
- `firebase_client.py`: Firebase Admin init + `publish_detection(report_id, detection_id, timestamp, defect)` + `publish_session_info(report_id, session_info)`.
- `webrtc_track.py`: aiortc `VideoStreamTrack` that receives latest annotated frames.
- `model_loader.py`: singleton model loader (Ultralytics YOLO).
- `device_manager.py`: resolves CPU/CUDA device selection.
