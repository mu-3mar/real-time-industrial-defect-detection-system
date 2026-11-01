# QC-SCM Core: The Detection Engine

The `core` directory contains the foundational logic for the QC-SCM detection pipeline.

## 🔑 Key Components

- **pipeline.py**: Orchestrates box-to-defect detection and visualization.
- **state.py**: Temporal voting, defect lock, and track recovery.
- **pipeline_manager.py**: Producer-consumer processing for multiple sessions.
- **model_loader.py**: Shared YOLO model loading and warmup.
- **local_session_store.py**: Temporary per-session result files.
- **session_manager.py**: Active detection session lifecycle.
- **stream.py**: Camera I/O and frame buffering.

## 🧵 Threading Model
The core logic is designed to be thread-safe, allowing multiple `SessionWorker` instances to push frames to a single `InferenceThread`.
