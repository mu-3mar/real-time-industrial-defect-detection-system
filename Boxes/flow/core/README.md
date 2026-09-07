# QC-SCM Core: The Detection Engine

The `core` directory contains the foundational logic for the QC-SCM detection pipeline.

## 🔑 Key Components

- **[pipeline.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/pipeline.py)**: Orchestrates the dual-stage detection (Box -> Defect) and coordinates between tracking and visualization.
- **[state.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/state.py)**: Manages the **Temporal Voting Window**, **Defect Lock**, and **Track Recovery** logic.
- **[pipeline_manager.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/pipeline_manager.py)**: Implements the **Producer-Consumer** threading model to manage multi-session frame processing.
- **[model_loader.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/model_loader.py)**: A singleton class that handles efficient YOLO model loading and VRAM management.
- **[firebase_client.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/firebase_client.py)**: Handles asynchronous communication with Firebase Realtime Database.
- **[session_manager.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/session_manager.py)**: Manages the lifecycle and metadata of active detection sessions.
- **[stream.py](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/core/stream.py)**: Handles high-speed camera I/O and frame buffering.

## 🧵 Threading Model
The core logic is designed to be thread-safe, allowing multiple `SessionWorker` instances to push frames to a single `InferenceThread`.
