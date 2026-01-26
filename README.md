# QC-SCM: Advanced Quality Control & Supply Chain Management System

A high-performance, AI-driven quality inspection system for production lines. This project currently supports two distinct inspection flows: **Boxes** and **Bottles**. It leverages Computer Vision (YOLOv8 + OpenCV) to provide real-time defect detection, counting, and quality assurance.

---

## � Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Installation & Setup](#-installation--setup)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
3. [Running the Application](#-running-the-application)
    - [GUI Launch](#gui-launch)
    - [Headless Mode](#headless-mode)
4. [Module Details](#-module-details)
    - [📦 Boxes Line](#-boxes-line)
    - [🧴 Bottles Line](#-bottles-line)
5. [Configuration & Tuning](#-configuration--tuning)
    - [Detection Parameters](#detection-parameters)
    - [Stream Settings](#stream-settings)
    - [Performance Optimization](#performance-optimization)
6. [Training & Fine-Tuning](#-training--fine-tuning)
    - [Training Structure](#training-structure)
    - [How to Train](#how-to-train)
7. [Troubleshooting](#-troubleshooting)

---

## 🏗 Architecture Overview

The system is built on a modular "Flow" architecture. Each product line (Boxes, Bottles) is a self-contained module with its own:
- **Flow Engine**: `main.py` containing the inspection logic (detection, tracking, counting).
- **Detectors**: Wrappers around YOLOv8 models for inference.
- **Config**: YAML files defining models, thresholds, and input sources.
- **Models**: Quantized (INT8) or standard FP32 ONNX/PT models for inference.

### Data Flow
1.  **Input**: Video stream (Camera/RTSP/File) defined in `stream.yaml`.
2.  **Preprocessing**: Frame resizing and ROI (Region of Interest) cropping.
3.  **Inference**:
    *   **Level 1**: Primary object detection (find the Box or Bottle).
    *   **Level 2**: Secondary defect/part detection (find Holes, Caps, Labels) on the cropped object.
4.  **Logic Layer**:
    *   **Temporal Stability**: Using weighted voting or counters (`MIN_FRAMES`, `VOTE_WINDOW`) to reduce flicker.
    *   **State Machine**: Tracks objects entering and exiting the ROI to count them accurately.
5.  **Output**: Real-time visualization, local logging, and statistics.

---

## 🛠 Installation & Setup

### Prerequisites
-   **OS**: Linux (Ubunto 20.04+ recommended) / Windows 10+
-   **GPU**: NVIDIA GPU (RTX series recommended) with CUDA 12.x drivers.
-   **Anaconda**: For environment management.

### Environment Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/QC-SCM.git
    cd QC-SCM
    ```

2.  **Create Anaconda Environment**:
    We use a dedicated environment named `qc` with Python 3.10 to ensure compatibility.
    ```bash
    conda create -n qc python=3.10 -y
    conda activate qc
    ```

3.  **Install Dependencies**:
    All exact versions are pinned in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Key Dependencies:* `ultralytics` (YOLOv8), `opencv-python`, `PySide6` (GUI), `torch` (CUDA enabled).

---

## 🚀 Running the Application

### GUI Launch
The easiest way to start is using the central Graphical User Interface.
```bash
conda activate qc
python GUI/luncher.py
```
*   **Boxes Line**: Starts the Box Inspection flow.
*   **Bottles Line**: Starts the Bottle Inspection flow.

### Headless Mode
You can run individual flows directly from the terminal for debugging or server deployments.

**Run Boxes Flow:**
```bash
python Boxes/flow/main.py
```

**Run Bottles Flow:**
```bash
python Bottles/flow/main.py
```

---

## 📦 Module Details

### 📦 Boxes Line
**Objective**: Identify defective boxes (e.g., tears, holes) and count total throughput.

*   **Models**:
    *   `Box Detector`: Finds the bounding box of the carton.
    *   `Defect Detector`: Scans the *cropped* carton for defects (Class 0: Hole).
*   **Logic**:
    *   **Spatial Hashing**: Tracks unique boxes based on approximate location grid (`x//20`, `y//20`).
    *   **Voting System**: Stores the last `VOTE_WINDOW` defect predictions. Only flags a defect if `VOTE_THRESHOLD` is met.
    *   **Counting**: Increments counts only when a box *fully exits* the camera view to avoid double-counting.

### 🧴 Bottles Line
**Objective**: Ensure every bottle has a **Cap** and a **Label**.

*   **Models**:
    *   `Bottle Detector`: Finds the bottle.
    *   `Info Detector`: Detects parts (`Cap`, `Label`) inside the bottle crop.
*   **Logic**:
    *   **Session Persistence**: A bottle is marked as "Have Cap" or "Have Label" if these parts are seen for at least `MIN_CONFIRM_FRAMES` *at any point* during its passage.
    *   **Pass/Fail**:
        *   **OK**: Both Cap AND Label confirmed.
        *   **Defect**: Missing either Cap or Label upon exit.

---

## ⚙ Configuration & Tuning

Configuration is managed via YAML files located in `*/flow/config/`.

### Detection Parameters
**File**: `Boxes/flow/config/box_detector.yaml` (Example)
```yaml
model_path: "Boxes/flow/models/best_box_detector_int8.onnx" # Path to model
conf_thres: 0.85   # Minimum confidence to accept a detection (0.0 - 1.0)
iou_thres: 0.5     # Intersection Over Union threshold for NMS
device: "cpu"      # "cpu" or "cuda:0"
```
*   **Tip**: Lower `conf_thres` if valid objects are being missed. Raise it if you see "ghost" objects.

### Stream Settings
**File**: `*/flow/config/stream.yaml`
```yaml
source: "video.mp4" # Path to video file OR "0" for Webcam OR "rtsp://..."
width: 640          # Input width resize
height: 640         # Input height resize
```

### Performance Optimization
Tune these variables in `flow/main.py` to balance FPS vs Accuracy:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROCESS_EVERY_N_FRAMES` | `2` | Runs detection every N frames. `2` means 50% load. Increase to `5` for weak CPUs. |
| `MIN_FRAMES` | `3` | Frames required to confirm an object has "entered". Filter out noise. |
| `MAX_MISSED` | `5` | Frames allowed to miss detection before assuming object has "exited". |
| `SKIP_DEFECT_FRAMES` | `2` | (Boxes) Skip heavy defect model inference on some frames. |

---

## 🧠 Training & Fine-Tuning

The project includes training scripts to fine-tune models on new data.

### Training Structure
*   **Boxes**: `Boxes/fine-tuning/combine-fine-tuning/`
    *   `box-YOLO/train.py`: Trains the box detector.
    *   `defect-YOLO/train.py`: Trains the defect detector.
*   **Bottles**: `Bottles/fine-tuning/`
    *   `bottles/train.py`: Trains bottle detector.
    *   `bottle - cap and label- info/train.py`: Trains component detector.

### How to Train
1.  **Prepare Data**: Ensure your `data.yaml` points to your dataset (images/labels).
2.  **Configure**: Edit `config.py` in the respective training folder to set:
    *   `EPOCHS`: Number of training cycles (default often 100 or 300).
    *   `BATCH_SIZE`: Adjust based on VRAM (16, 32, 64).
    *   `BASE_MODEL`: e.g., `yolov8n.pt` (Nano) or `yolov8s.pt` (Small).
3.  **Run Training**:
    ```bash
    cd "Boxes/fine-tuning/combine-fine-tuning/box-YOLO"
    python train.py
    ```
4.  **Output**: Best weights will be saved to `runs/detect/train_name/weights/best.pt`.
5.  **Deploy**: Copy `best.pt` (or export to ONNX) to the `flow/models/` folder and update the `detector.yaml` config.

---

## 🔧 Troubleshooting

| Problem | Possible Cause | Solution |
| :--- | :--- | :--- |
| **"Pipeline already running"** | The previous process didn't close cleanly. | Check terminal for running python processes or restart the GUI. |
| **Low FPS / Lag** | Running on CPU or resolution too high. | 1. Ensure `device: cuda` in config.<br>2. Increase `PROCESS_EVERY_N_FRAMES`.<br>3. Resize input in `stream.yaml`. |
| **Flickering Detections** | Confidence threshold too low. | Increase `conf_thres` in detector yaml files. |
| **Missed Defects** | Object moving too fast. | Decrease `PROCESS_EVERY_N_FRAMES` to `1` (process every frame). |
| **ImportError: libGL.so.1** | Missing OS library. | Run `sudo apt-get install ffmpeg libsm6 libxext6`. |

---
*Generated for QC-SCM Project | 2026*
