# QC-SCM: Real-Time Industrial Quality Control & Defect Detection System

QC-SCM is an AI-driven quality inspection platform designed for high-speed manufacturing environments. It uses a dual-stage deep learning pipeline and a multi-threaded asynchronous architecture to provide reliable, low-latency product localization and defect analysis.

## 🚀 Key Features

- **Dual-Stage AI Pipeline**: Product localization (Box Detection) followed by high-resolution anomaly analysis (Defect Detection).
- **Asynchronous Architecture**: Decoupled camera I/O and AI inference threads for maximum GPU utilization and minimal frame dropping.
- **Industrial-Grade Stability**: **Temporal Voting** and **Defect Lock** eliminate false positives and flickering decisions.
- **Cloud Integration**: Real-time reporting and telemetry synchronization via **Firebase Realtime Database**.
- **Multi-Session Support**: Monitor multiple production lines from a single centralized service.
- **Hardware Accelerated**: Optimized for **ONNX Runtime** and **NVIDIA TensorRT** for sub-30ms end-to-end latency.

---

## 📂 Repository Overview

| Module | Description |
| :--- | :--- |
| **`Boxes/flow/`** | High-performance runtime service (FastAPI + async pipeline + Firebase). |
| **`Boxes/training V2.0/`** | Training pipelines for Box and Defect detection models. |

---

## 🛠️ System Architecture

The system follows a **Producer-Consumer** pattern to handle high-speed video streams without blocking the inference engine.

- **Inference Thread**: A dedicated GPU-bound thread that processes one frame at a time.
- **Session Workers**: Independent threads per camera source for frame acquisition.
- **Reporting Workers**: Asynchronous threads for MJPEG streaming and Firebase cloud publishing.

```
Camera HW → SessionWorker (feeder thread)
                  ↓
           frame_queue (bounded)
                  ↓
           InferenceThread (PipelineManager)
                  ↓
           result_queue
                  ↓
     ┌────────────┴────────────┐
MJPEG stream           Firebase worker
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python**: 3.10+
- **CUDA/cuDNN**: Recommended for GPU-accelerated inference.
- **Firebase**: A service account JSON file and Realtime Database URL.

### 2. Installation

```bash
pip install -r Boxes/flow/requirements/requirements.txt
```

### 3. Configuration

Copy the example and fill in your credentials:

```bash
cp Boxes/flow/config/firebase.example.yaml Boxes/flow/config/firebase.yaml
# Edit firebase.yaml: set service_account_path and database_url
```

Tune detection thresholds in `Boxes/flow/config/box_detector.yaml` and `defect_detector.yaml`.

### 4. Launch

```bash
cd Boxes/flow
python3 main.py
```

The API will be available at `http://localhost:8000`. See `Boxes/flow/api/README.md` for endpoint documentation.

---

## 📊 Training (Version 2.0)

The V2.0 training pipeline introduces:
- **Horizontal Motion Blur**: Custom augmentation simulating conveyor belt speeds.
- **Mosaic Augmentation**: Improving detection for small-scale defects.
- **Optimized Metrics**: Achieving **99.2% mAP@50** for product localization.

```bash
cd "Boxes/training V2.0/box/training"
python3 train.py
```

---

**Developed by:** mu-3mar  
**Academic Year:** 2025–2026
