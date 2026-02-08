# QC-SCM: Quality Control - Supply Chain Management

A high-performance, AI-driven quality inspection system for real-time defect detection in manufacturing production lines. The system uses advanced computer vision and deep learning models to automatically detect and classify defects in boxes during production.

## 🎯 Overview

QC-SCM is an intelligent quality control system designed for supply chain management that leverages state-of-the-art YOLO (You Only Look Once) models for real-time defect detection. The system provides:

- **Real-time Detection**: Live video stream processing with WebRTC support
- **Multi-Session Management**: Handle multiple production lines simultaneously
- **High Accuracy**: Optimized YOLO models with excellent performance metrics
- **Production-Ready**: RESTful API with FastAPI backend
- **Scalable Architecture**: Modular design for easy extension and maintenance

## 🏗️ Architecture

The system follows a modular architecture with the following components:

```
QC-SCM/
├── Boxes/                          # Box inspection module
│   ├── flow/                       # Runtime detection pipeline
│   │   ├── api_server.py          # FastAPI server with WebRTC
│   │   ├── main.py                # Entry point
│   │   ├── configs/               # Configuration files
│   │   ├── core/                  # Core detection logic
│   │   ├── detectors/             # Model inference
│   │   └── utils/                 # Utility functions
│   └── trainig/                   # Model training
│       ├── box-YOLO/              # Box detection model
│       └── defect-YOLO/           # Defect classification model
├── index.html                      # Production dashboard UI
└── pyproject.toml                 # Project dependencies
```

## 🤖 AI Models & Performance

### Box Detection Model (YOLOv8)

The box detection model identifies boxes in the production line with high precision.

**Model Details:**
- **Architecture**: YOLOv8n (Nano variant)
- **Training Epochs**: 31
- **Input Size**: Standard YOLO input
- **Model Path**: `Boxes/trainig/box-YOLO/models/exported/best.pt`

**Performance Metrics (Best Epoch - Epoch 31):**

| Metric | Value |
|--------|-------|
| **mAP@50** | **95.87%** |
| **mAP@50-95** | **83.08%** |
| **Precision** | **94.61%** |
| **Recall** | **89.62%** |
| **Confidence Threshold** | 0.7 |
| **IoU Threshold** | 0.6 |

### Defect Detection Model (YOLOv8)

The defect detection model classifies defects within detected boxes using a two-stage pipeline.

**Model Details:**
- **Architecture**: YOLOv8n (Nano variant)
- **Training Epochs**: 100
- **Input Size**: Standard YOLO input
- **Model Path**: `Boxes/trainig/defect-YOLO/models/exported/best.pt`

**Performance Metrics (Best Epoch - Epoch 100):**

| Metric | Value |
|--------|-------|
| **mAP@50** | **88.82%** |
| **mAP@50-95** | **63.18%** |
| **Precision** | **90.41%** |
| **Recall** | **82.49%** |
| **Confidence Threshold** | 0.3 |
| **IoU Threshold** | 0.6 |

**Defect Detection Features:**
- **Single-box tracking** with smooth, stable annotations
- **Defect voting system** for robust decision-making
- **Configurable stability parameters**:
  - Min frames before box is "inside": 4
  - Max missed frames before exit: 6
  - Vote window: 9 frames
  - Vote threshold: 5 defect votes (out of 9) to classify as DEFECT
- **Visibility-based rendering** (20% threshold)

## 📊 Detection Pipeline

The system uses a **two-stage detection pipeline**:

1. **Stage 1 - Box Detection**: 
   - Detects boxes in the video frame
   - Tracks boxes across frames with IoU-based tracking
   - Applies bounding box smoothing (alpha = 0.6)

2. **Stage 2 - Defect Classification**:
   - Crops detected boxes
   - Classifies defects within each box
   - Uses voting mechanism for stable defect decisions
   - Tracks defects across frames

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for real-time performance)
- Webcam or video input device

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd QC-SCM
```

2. **Install dependencies**:
```bash
pip install -e .
```

The project uses `pyproject.toml` for dependency management with the following key packages:
- `ultralytics==8.4.7` - YOLO model framework
- `torch==2.9.1` - Deep learning framework
- `opencv-python==4.13.0.90` - Computer vision
- `fastapi==0.115.6` - API framework
- `uvicorn[standard]==0.34.0` - ASGI server
- `onnxruntime-gpu==1.23.2` - Optimized inference
- `aiortc` - WebRTC support

## 💻 Usage

### Starting the Detection Service

1. **Start the API server**:
```bash
cd Boxes/flow
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

2. **Open the production dashboard**:
   - Open `index.html` in a web browser
   - The dashboard provides controls for multiple production lines

### Configuration

Edit the configuration files in `Boxes/flow/configs/`:

**Box Detector (`box_detector.yaml`):**
```yaml
model_path: Boxes/trainig/box-YOLO/models/exported/best.pt
conf_thres: 0.7
iou_thres: 0.6
device: 0
```

**Defect Detector (`defect_detector.yaml`):**
```yaml
model_path: Boxes/trainig/defect-YOLO/models/exported/best.pt
conf_thres: 0.3
iou_thres: 0.6
device: 0

tracking:
  iou_threshold: 0.35
  bbox_smooth_alpha: 0.6

stability:
  min_frames: 4
  max_missed: 6
  vote_window: 9
  vote_threshold: 5

rendering:
  visibility_threshold: 0.2
```

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Open Detection Session
**POST** `/api/sessions/open`

Opens a new headless detection session for a production line.

**Request Body:**
```json
{
  "report_id": "line1_abc123",
  "camera_source": "/dev/v4l/by-id/camera-device"
}
```

**Response:**
```json
{
  "status": "success",
  "report_id": "line1_abc123",
  "message": "Session started with camera /dev/v4l/by-id/camera-device"
}
```

#### 2. Close Detection Session
**POST** `/api/sessions/close`

Closes an active detection session.

**Request Body:**
```json
{
  "report_id": "line1_abc123",
  "camera_source": "/dev/v4l/by-id/camera-device"
}
```

**Response:**
```json
{
  "status": "success",
  "report_id": "line1_abc123",
  "message": "Session closed successfully"
}
```

#### 3. List Active Sessions
**GET** `/api/sessions`

Returns a list of all active detection sessions.

**Response:**
```json
{
  "sessions": [
    {
      "report_id": "line1_abc123",
      "camera_source": "/dev/v4l/by-id/camera-device",
      "status": "active"
    }
  ]
}
```

#### 4. Health Check
**GET** `/api/health`

Returns service health status and active session count.

**Response:**
```json
{
  "status": "healthy",
  "active_sessions": 2
}
```

#### 5. WebRTC Offer
**POST** `/webrtc/offer`

Establishes a WebRTC connection for real-time video streaming.

**Request Body:**
```json
{
  "sdp": "<SDP offer string>",
  "type": "offer",
  "report_id": "line1_abc123"
}
```

**Response:**
```json
{
  "sdp": "<SDP answer string>",
  "type": "answer"
}
```

## 🎨 Production Dashboard

The `index.html` file provides a web-based dashboard for managing multiple production lines:

**Features:**
- **Multi-line support**: Monitor multiple production lines simultaneously
- **Real-time video streaming**: WebRTC-based live video feed
- **Session management**: Open/close detection sessions
- **Status monitoring**: View connection and streaming status
- **Clean UI**: Modern, responsive interface

**Dashboard Controls:**
- **Open**: Start a new detection session
- **Close**: Stop the detection session
- **Start Stream**: Begin WebRTC video streaming
- **Stop Stream**: End video streaming
- **List Sessions**: View all active sessions

## 🔧 Training Models

### Box Detection Model

Navigate to the box detection training directory:
```bash
cd Boxes/trainig/box-YOLO
```

Run the complete training pipeline:
```bash
python scripts/run_all.py
```

This will:
1. Train the model
2. Export to ONNX format
3. Quantize the model

### Defect Detection Model

Navigate to the defect detection training directory:
```bash
cd Boxes/trainig/defect-YOLO
```

Run the complete training pipeline:
```bash
python scripts/run_all.py
```

### Data Management

Merge multiple datasets:
```bash
python scripts/merge_data.py
```

## 🌐 Network Configuration

The system supports both local and remote deployment:

- **Local Development**: Use `localhost` or `127.0.0.1`
- **LAN Deployment**: Configure `BASE_URL` in `index.html` to your server's IP
- **Tailscale VPN**: Supports peer-to-peer connections over Tailscale network

**Example Configuration (index.html):**
```javascript
const BASE_URL = "......" // Tailscale IP
```

## 📈 Performance Optimization

The system is optimized for real-time performance:

- **GPU Acceleration**: CUDA support for fast inference
- **Model Quantization**: INT8 quantization for reduced latency
- **Frame Skipping**: Configurable frame processing rate
- **Batch Processing**: Efficient batch inference
- **WebRTC Streaming**: Low-latency video transmission

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Ultralytics YOLO
- **Computer Vision**: OpenCV
- **Backend**: FastAPI, Uvicorn
- **Real-time Communication**: WebRTC (aiortc)
- **Model Optimization**: ONNX Runtime
- **Frontend**: HTML5, JavaScript (Vanilla)
