# QC-SCM: Quality Control & Supply Chain Management System

> **A High-Performance AI-Driven Quality Assurance & Supply Chain Optimization Platform**
> 
> *Graduation Project 2026*

---

## 📋 Table of Contents

1.  [Overview](#-overview)
2.  [System Architecture](#-system-architecture)
3.  [Feature Modules](#-feature-modules)
    *   [Box Inspection](#-box-inspection)
    *   [Bottle Inspection](#-bottle-inspection)
    *   [Demand Forecasting](#-demand-forecasting)
4.  [Model Performance Metrics](#-model-performance-metrics)
5.  [Installation & Setup](#-installation--setup)
6.  [Usage Guide](#-usage-guide)
7.  [API Documentation](#-api-documentation)
8.  [Troubleshooting](#-troubleshooting)

---

## 🔥 Overview

**QC-SCM** is a comprehensive industrial automation solution designed to modernize production packaging lines. It integrates state-of-the-art Computer Vision (YOLOv8) and Machine Learning (LightGBM) to ensure product quality and optimize supply chain inventory.

The system operates in real-time to:
*   **Auto-Detect & Count** products (Boxes, Bottles).
*   **Identify Defects** (Holes, Tears, Water Damage, Missing Caps/Labels).
*   **Forecast Demand** to prevent stockouts and overstocking.
*   **Visualize Data** via a modern WebSocket-based GUI.

---

## 🏗 System Architecture

The project is divided into three core intelligent modules:

### 1. 📦 Box Inspection Module
Focuses on packaging integrity. It uses a **two-stage detection pipeline**:
1.  **Stage 1 (Global Detection)**: Identifies the presence and location of boxes on the conveyor belt.
2.  **Stage 2 (Defect Analysis)**: Crops the detected box and runs a high-resolution analysis to find specific defects like water damage, holes, or tears.

### 2. 🍾 Bottle Inspection Module
Ensures liquid product quality. It also employs a multi-stage approach:
1.  **Bottle Detection**: Counts distinct bottles in the frame.
2.  **Attribute Verification**: Checks for critical components:
    *   **Cap Presence**: Is the bottle sealed?
    *   **Label Integrity**: Is the branding label attached?
    *   **Water Level**: (Feature) Verifies fill level consistency.

### 3. 📈 Demand Forecasting Module
Optimizes the supply chain by predicting future product sales.
*   Uses historical sales data and external factors (seasonality, promotions).
*   Built on **FastAPI** and **LightGBM** for high-speed inference.

---

## 🚀 Feature Modules

### 📦 Box Inspection

*   **Core Tech**: YOLOv8 (PyTorch/ONNX)
*   **Defect Classes**:
    *   `Hole`
    *   `Tear`
    *   `Water Damage`
*   **Key Capabilities**:
    *   Real-time processing (30+ FPS on GPU).
    *   Dynamic defect visualization with transparent overlays.
    *   WebRTC streaming to remote dashboards.

### 🍾 Bottle Inspection

*   **Core Tech**: YOLOv8s (Medium) + YOLOv8n (Nano)
*   **Inspection Points**:
    *   Total Bottle Count
    *   Defect Count (Missing Cap/Label)
    *   Acceptable Product Count
*   **Workflow**:
    *   `YOLOv8s` detects all bottles.
    *   `YOLOv8n` classifies cropped regions for Caps and Labels.

### 📈 Demand Forecasting

*   **Core Tech**: LightGBM
*   **Features**:
    *   Stockout handling algorithms.
    *   Multi-day lag features for trend analysis.
    *   94%+ R² Accuracy on test data.

---

## 📊 Model Performance Metrics

We have rigorously trained and evaluated our models. Below are the key performance indicators (KPIs).

### 1. Box Detection Models

| Model Component | Architecture | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Box Locator** | YOLOv8 | **94.6%** | 89.6% | **95.8%** | 83.1% |
| **Defect Classifier** | YOLOv8 | 90.4% | 82.5% | 88.8% | 63.2% |

### 2. Bottle Inspection Models

| Model Component | Architecture | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bottle Detector** | YOLOv8s | **96.4%** | 93.2% | 96.8% | 89.7% |
| **Bottle Attributes** | YOLOv8n | **98.4%** | **99.7%** | **99.1%** | 97.5% |

### 3. Supply Chain Forecasting

| Metric | Value | Description |
| :--- | :--- | :--- |
| **R² Score** | **94.43%** | Explains 94% of sales variance. |
| **RMSE** | 0.423 | Root Mean Square Error. |
| **MAE** | 0.223 | Mean Absolute Error. |

---

## 🛠 Installation & Setup

### Prerequisites
*   **OS**: Linux (Ubuntu 20.04/22.04 recommended) or Windows 10/11
*   **Python**: 3.10+
*   **GPU**: NVIDIA GTX/RTX Series (CUDA 11.8+)
*   **Conda**: Recommended for environment isolation

### Step 1: Clone Repository
```bash
git clone https://github.com/mu-3mar/QC-SCM.git
cd QC-SCM
```

### Step 2: Create Environment
```bash
conda create -n qc python=3.10 -y
conda activate qc
```

### Step 3: Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎮 Usage Guide

### Running Box Inspection (Headless/API)
Starts the backend detection server.
```bash
conda activate qc
python Boxes/flow/main.py
```

### Running the Visualization GUI
Launch the WebSocket viewer to see live inferences.
1.  Open `index.html` in a modern web browser.
2.  Or use the Python viewer:
    ```bash
    python GUI/viewer.py
    ```

### Running Bottle Inspection Flow
```bash
# Update path to your specific flow script
python Bottles/flow/main.py
```

### Running Demand Forecasting API
```bash
cd DevIgnite-AI_DemandForeCasting/src
python app.py
```

---

## 📡 API Documentation

The system exposes a RESTful API (FastAPI) for integration.

### Base URL: `http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/sessions/open` | Start a new inspection session. Requires `report_id` and `camera_source`. |
| `POST` | `/api/sessions/close` | Stop an active session. |
| `GET` | `/api/sessions` | List all active inspection sessions. |
| `GET` | `/api/health` | Check system health status. |
| `POST` | `/webrtc/offer` | Initiate WebRTC stream for live video. |

**Example: Open a Session**
```bash
curl -X POST "http://localhost:8000/api/sessions/open" \
     -H "Content-Type: application/json" \
     -d '{"report_id": "session_01", "camera_source": 0}'
```

---

## 🔧 Troubleshooting

| **Issue** | **Possible Cause** | **Solution** |
| :--- | :--- | :--- |
| **CUDA OOM Error** | GPU memory full. | Reduce `batch_size` in config or `PROCESS_EVERY_N_FRAMES` in `main.py`. |
| **Low FPS** | Running on CPU. | Ensure `torch.cuda.is_available()` returns `True`. Install `onnxruntime-gpu`. |
| **No Detection** | Confidence threshold too high. | Lower `conf_thres` in `Boxes/flow/configs/box_detector.yaml`. |
| **WebRTC Fail** | Network firewall. | Ensure ports 8000 and UDP ports are open. Check `api_server.py` logs. |

---

### 📞 Contact & Support
*   **Lead Developer**: Muhammad Ammar
*   **Project Link**: [GitHub Repository](https://github.com/mu-3mar/QC-SCM)

> *"Quality is not an act, it is a habit."*
