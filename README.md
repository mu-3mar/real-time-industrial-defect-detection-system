# QC-SCM: Quality Control & Supply Chain Management System

A high-performance, AI-driven quality inspection system for production lines using Computer Vision (YOLOv8 + OpenCV) for real-time defect detection, counting, and quality assurance.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Installation & Setup](#-installation--setup)
3. [Running the Application](#-running-the-application)
4. [Box Inspection Flow](#-box-inspection-flow)
5. [Training Models](#-training-models)
6. [Configuration](#-configuration)
7. [Troubleshooting](#-troubleshooting)

---

##  Overview

This system provides automated quality control for box inspection on production lines. It uses a two-stage detection approach:

1. **Box Detection**: Locates boxes in the video stream
2. **Defect Detection**: Analyzes each box for defects (holes, tears, water damage)

### Key Features

- Real-time video processing with YOLO models
- Two-stage detection pipeline (box → defect)
- Configurable detection thresholds
- Support for ONNX and PyTorch models
- Modular architecture for easy extension

---

## 🛠 Installation & Setup

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+) / Windows 10+
- **GPU**: NVIDIA GPU with CUDA 12.x (recommended)
- **Python**: 3.10+
- **Anaconda**: For environment management

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mu-3mar/QC-SCM.git
   cd QC-SCM
   ```

2. **Create Conda Environment**:
   ```bash
   conda create -n qc python=3.10 -y
   conda activate qc
   ```

3. **Install Dependencies**:
   ```bash
   pip install .
   ```

   Key dependencies include:
   - `ultralytics` (YOLOv8)
   - `opencv-python`
   - `torch` (CUDA enabled)
   - `onnxruntime-gpu`
   - `PySide6` (GUI support)

---

## 🚀 Running the Application

### Run Box Inspection Flow

```bash
conda activate qc
python Boxes/flow/main.py
```

This will start the box inspection pipeline using the configuration files in `Boxes/flow/configs/`.

---

## 📦 Box Inspection Flow

### Architecture

The box inspection system is located in `Boxes/flow/` with the following structure:

```
Boxes/flow/
├── main.py           # Main entry point
├── configs/          # YAML configuration files
├── core/             # Core processing logic
├── detectors/        # YOLO detector wrappers
└── utils/            # Utility functions
```

### Detection Pipeline

1. **Input**: Video stream (camera/file/RTSP)
2. **Box Detection**: Primary YOLO model detects boxes
3. **Defect Detection**: Secondary YOLO model analyzes cropped boxes for defects
4. **Output**: Annotated video with defect counts

### Defect Types

The system currently detects:
- **Holes**: Physical damage/punctures
- **Water Damage**: Moisture damage
- **Tears**: Rips in packaging

---

## 🧠 Training Models

Training scripts are located in `Boxes/trainig/`:

### Box Detector Training

```bash
cd Boxes/trainig/box-YOLO
python train.py
```

### Defect Detector Training

```bash
cd Boxes/trainig/defect-YOLO
python train.py
```

### Training Configuration

Each training folder contains:
- `train.py`: Training script
- `data.yaml`: Dataset configuration
- `config.py`: Training parameters (epochs, batch size, etc.)

After training, copy the best weights to `Boxes/flow/models/` and update the detector config files.

---

## ⚙ Configuration

Configuration files are in `Boxes/flow/configs/`:

### Detector Configuration

Example structure:
```yaml
model_path: "path/to/model.onnx"
conf_thres: 0.85    # Confidence threshold
iou_thres: 0.5      # NMS threshold
device: "cuda:0"    # or "cpu"
```

### Stream Configuration

Configure video input source and resolution in the stream config file.

### Performance Tuning

Adjust these parameters in `main.py`:
- `PROCESS_EVERY_N_FRAMES`: Process every Nth frame (reduce for speed)
- `conf_thres`: Increase to reduce false positives
- `iou_thres`: Adjust NMS behavior

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Low FPS** | Increase `PROCESS_EVERY_N_FRAMES`, ensure CUDA is enabled |
| **False Detections** | Increase `conf_thres` in detector config |
| **Missed Defects** | Decrease `conf_thres`, process more frames |
| **CUDA Errors** | Verify CUDA installation, check GPU compatibility |

---

## 📢 Recent Updates (February 2026)

- **Refined Defect Visualization**: Corner bracket bounding boxes with transparent defect overlays
- **Modern Config**: Migrated to `pyproject.toml` for dependency management
- **Modular Architecture**: Separated configs, core logic, detectors, and utils

---

*QC-SCM Project | 2026*
