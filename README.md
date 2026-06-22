# QC-SCM: Real-Time Industrial Quality Control & Defect Detection System

QC-SCM is a state-of-the-art, AI-driven quality inspection platform designed for high-speed manufacturing environments. It leverages dual-stage deep learning models and a multi-threaded asynchronous architecture to provide reliable, low-latency product localization and defect analysis.

## 🚀 Key Features

- **Dual-Stage AI Pipeline**: Product localization (Box Detection) followed by high-resolution anomaly analysis (Defect Detection).
- **Asynchronous Architecture**: Decoupled camera I/O and AI inference threads for maximum GPU utilization and minimal frame dropping.
- **Industrial-Grade Stability**: Features like **Temporal Voting** and **Defect Lock** to eliminate false positives and flickering.
- **Cloud Integration**: Real-time reporting and telemetry synchronization via **Firebase Realtime Database**.
- **Scalable Design**: Multi-session support allows monitoring multiple production lines from a single centralized service.
- **Hardware Accelerated**: Optimized for **ONNX Runtime** and **NVIDIA TensorRT** for sub-30ms end-to-end latency.

---

## 📂 Repository Overview

The project is organized into two primary domains:

| Module | Description |
| :--- | :--- |
| **[Boxes/flow/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/flow/)** | High-performance runtime service (FastAPI + Asynchronous Pipeline + Firebase). |
| **[Boxes/training V2.0/](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Boxes/training%20V2.0/)** | Advanced training pipelines for Box and Defect detection models with industrial augmentations. |

---

## 🛠️ System Architecture

The system follows a **Producer-Consumer** pattern to handle high-speed video streams without blocking the inference engine.

- **Inference Thread**: A dedicated GPU-bound thread that processes one frame at a time.
- **Session Workers**: Independent threads for each camera source that handle frame acquisition and local telemetry.
- **Reporting Workers**: Asynchronous threads for MJPEG streaming and Firebase cloud publishing.

For a deep dive into the architecture, see the **[System Architecture Documentation](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/System_Architecture_Documentation.md)**.

---

## 🚀 Quick Start (Flow Runtime)

### 1. Prerequisites
- **Python**: 3.10+
- **CUDA/cuDNN**: (Highly Recommended) For GPU-accelerated inference.
- **Firebase**: A service account JSON file and Realtime Database URL.

### 2. Installation
Install the runtime dependencies:
```bash
pip install -r Boxes/flow/requirements/requirements.txt
```

### 3. Configuration
Set up your Firebase credentials in `Boxes/flow/config/firebase.yaml`.

### 4. Launch
Start the detection service:
```bash
cd Boxes/flow
python3 main.py
```

---

## 📊 Training (Version 2.0)

The V2.0 training pipeline introduces:
- **Horizontal Motion Blur**: Custom augmentation to simulate conveyor belt speeds.
- **Mosaic Augmentation**: Improving detection for small-scale defects.
- **Optimized Metrics**: Reaching **99.2% mAP@50** for product localization.

To start training, navigate to the specific model directory:
```bash
cd "Boxes/training V2.0/box/training"
python3 train.py
```

---

## 📖 Documentation
For comprehensive details, refer to the following documents:
- **[Full Project Thesis](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/Full_Project_Documentation.md)**: Academic-style documentation.
- **[Presentation Guide](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/AI_Presentation_Guide.md)**: Structured content for graduation project defense.
- **[System Architecture](file:///home/mu-3mar/projects/real-time-industrial-defect-detection-system/System_Architecture_Documentation.md)**: Detailed technical breakdown.

---
**Developed by:** [Your Name/Team Name]
**Academic Year:** 2025-2026
