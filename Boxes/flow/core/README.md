# QC-SCM Core: The Detection Engine

The `core` directory contains the foundational logic for the QC-SCM detection pipeline.

## Key Components

| File | Responsibility |
| :--- | :--- |
| `pipeline.py` | Orchestrates dual-stage detection (Box → Defect), tracking, and canvas rendering. |
| `state.py` | Manages **Temporal Voting**, **Defect Lock**, track grace period, and recovery logic. |
| `pipeline_manager.py` | **Producer-Consumer** threading model: frame queue → inference → result/Firebase queues. |
| `pipeline_diagnostics.py` | Lightweight runtime counters for FPS, latency, and drop rates. |
| `session_manager.py` | Lifecycle management for active detection sessions. |
| `session_worker.py` | Per-session thread: starts the camera and feeds frames into the pipeline queue. |
| `model_loader.py` | Singleton that loads both YOLO models once and shares them across all sessions. |
| `firebase_client.py` | Thin wrapper around Firebase Admin SDK for pushing detection events. |
| `stream.py` | OpenCV V4L2 camera reader with a dedicated background capture thread. |
| `device_manager.py` | Resolves CUDA / MPS / CPU device from config or auto-detection. |

## Threading Model

```
SessionWorker (per camera)
      │  frames via get_latest_frame()
      ▼
  frame_queue  (bounded, FRAME_QUEUE_MAXSIZE=5)
      │
      ▼
InferenceThread  ← single GPU thread (pipeline_manager.py)
      │  (canvas, exit_event)
      ▼
  result_queue  (bounded, RESULT_QUEUE_MAXSIZE=5)
      │
      ▼
ResultConsumerThread
   ├─ stores latest frame for MJPEG
   └─ firebase_queue (bounded, FIREBASE_QUEUE_MAXSIZE=64)
             │
             ▼
        FirebaseWorkerThread
```

All queues are bounded to prevent latency buildup. Inference never blocks on Firebase or MJPEG consumers.
