# QC-SCM
**Quality Control – Supply Chain Management**

An intelligent computer vision system for automated quality control in supply chain management using deep learning and real-time object detection.

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Components](#components)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Model Export & Quantization](#model-export--quantization)
- [Contributing](#contributing)

## Project Overview

QC-SCM is a comprehensive quality control system designed for supply chain management operations. It leverages advanced YOLO (You Only Look Once) object detection models to:

- **Detect cartons/boxes** on production lines in real-time
- **Identify defects** within detected cartons
- **Track items** across video frames using QR code identification
- **Monitor production status** (OK/Defect classification)
- **Integrate with backend APIs** for data logging and reporting

The system is built for production line monitoring and provides automated quality assessment with visual feedback and data persistence.

## System Architecture

The system consists of two main components:

### 1. **Fine-Tuning Module** (`fine-tuning/combine-fine-tuning/`)
Trains and exports two separate YOLO-based detection models:
- **Box/Carton Detector**: Identifies carton boxes on the production line
- **Defect Detector**: Detects defects within cropped carton regions

### 2. **Real-Time Pipeline** (`flow/pipeline/`)
Runs the inference pipeline that:
- Captures video from camera feed
- Performs real-time box detection
- Runs defect detection on detected boxes
- Reads QR codes for item tracking
- Maintains tracking state across frames
- Sends results to backend API

## Components

### Fine-Tuning Components

#### **box-YOLO** (`fine-tuning/combine-fine-tuning/box-YOLO/`)
Detects carton boxes/containers on production lines.

**Key Files:**
- `config.py` - Configuration for training parameters and paths
- `train.py` - Training script using YOLOv8
- `export_onnx.py` - Exports trained model to ONNX format
- `quantize_onnx.py` - Quantizes ONNX model to INT8 for optimization
- `export_tflite.py` - Exports to TensorFlow Lite format
- `run_all.py` - Executes entire pipeline (train → export → quantize)
- `data/data.yaml` - Dataset configuration
- `data/train/`, `data/valid/`, `data/test/` - Train, validation, and test datasets

**Configuration (`config.py`):**
```python
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "0"  # GPU device
BASE_MODEL = "yolov8n.pt"  # YOLOv8 Nano
ONNX_NAME = "best_box_detector.onnx"
TFLITE_NAME = "best_box_detector.tflite"
```

#### **defect-YOLO** (`fine-tuning/combine-fine-tuning/defect-YOLO/`)
Detects defects within cropped carton regions.

**Key Files:**
- Same structure as box-YOLO but focused on defect detection
- Processes cropped images from detected cartons
- Trained on defect-specific dataset

**Configuration (`config.py`):**
```python
NAME = "detect_defects"
ONNX_NAME = "best_defect_detector.onnx"
TFLITE_NAME = "best_defect_detector.tflite"
```

### Real-Time Pipeline Components

#### **main.py** - Main Processing Loop
Orchestrates the entire real-time quality control pipeline.

**Key Functions:**
- `process_frame()` - Main frame processing logic:
  1. Detects carton boxes
  2. Expands box regions for better context
  3. Runs defect detection on expanded regions
  4. Reads QR codes from cartons
  5. Updates tracking information
  6. Sends data to backend API

**Key Workflow:**
```
Input Frame
    ↓
Carton Detection
    ↓
For each detected carton:
    - Expand box region
    - Run Defect Detection
    - Read QR Code
    - Track across frames
    - Compute final status
    ↓
Output annotated frame + API updates
```

#### **config.py** - Pipeline Configuration
Central configuration for the inference pipeline.

**Key Parameters:**
```python
CARTON_MODEL = "path/to/best_box_detector_int8.onnx"
DEFECT_MODEL = "path/to/best_defect_detector_int8.onnx"
CAMERA_INDEX = 1  # Camera device index
CARTON_CONF = 0.5  # Confidence threshold for carton detection
DEFECT_CONF = 0.25  # Confidence threshold for defect detection
MAX_DISAPPEAR = 12  # Frames before track is removed
EXPAND_RATIO = 0.1  # Box expansion ratio for better context
API_URL = "https://chainly.azurewebsites.net/api/ProductionLines/sessions"
PRODUCTION_LINE_ID = 1
COMPANY_ID = 90
SESSION_ID = uuid.uuid4().hex  # Unique session identifier
```

#### **detector.py** - YOLO Inference Wrapper
Encapsulates YOLO model inference.

**Class: `Detector`**
```python
class Detector:
    def __init__(self, model_path: str)
        # Loads .onnx or .pt model
    
    def infer(frame, conf) -> (boxes, scores, classes)
        # Returns: boxes (Nx4 xyxy), scores (N,), classes (N,)
```

#### **tracker.py** - Multi-Object Tracking
Maintains tracking state across frames using QR code associations.

**Key Functions:**
- `create_new_track()` - Initializes tracking for a new QR code
- `update_track()` - Updates existing track with new frame data
- `finalize_disappeared()` - Handles disappeared tracks
- `finalize_all_and_send()` - Sends final status to API

#### **helpers.py** - Utility Functions
```python
compute_final_status_for_db(track_info) -> "ok" | "defect"
```

#### **utils.py** - Image Processing
```python
expand_box(box, w, h, ratio) -> expanded_box
read_qr(frame) -> qr_code_string
draw_box(frame, box, color, label)
highlight_box(frame, box)
```

#### **api_client.py** - Backend API Integration
Handles communication with backend API for data logging.

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- OpenCV
- PyTorch with CUDA support

### Setup Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd QC-SCM
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; import cv2; import ultralytics; print('✓ All dependencies installed')"
```

## Configuration

### Step 1: Update Model Paths
Edit `flow/pipeline/config.py` with your trained model paths:

```python
CARTON_MODEL = "path/to/your/best_box_detector_int8.onnx"
DEFECT_MODEL = "path/to/your/best_defect_detector_int8.onnx"
```

### Step 2: Configure Detection Thresholds
```python
CARTON_CONF = 0.5  # Lower = more detections, higher = more confidence
DEFECT_CONF = 0.25  # Defect detection confidence threshold
```

### Step 3: Set Camera Index
```python
CAMERA_INDEX = 1  # Change to 0, 1, 2, etc. based on your setup
```

### Step 4: Configure API Connection
```python
API_URL = "your-backend-api-url"
PRODUCTION_LINE_ID = 1
COMPANY_ID = 90
```

### Step 5: Adjust Tracking Parameters
```python
MAX_DISAPPEAR = 12  # Frames before track disappears
EXPAND_RATIO = 0.1  # Region expansion for context
```

## Usage

### Running the Real-Time Pipeline

**Basic usage (uses default config):**
```bash
python -m flow.pipeline.main
```

**With custom models:**
```bash
python -m flow.pipeline.main \
    --carton-model path/to/carton_model.onnx \
    --defect-model path/to/defect_model.onnx \
    --cam 0
```

**Command Line Arguments:**
- `--carton-model` - Path to carton detection model (default: from config)
- `--defect-model` - Path to defect detection model (default: from config)
- `--cam` - Camera index (default: from config)

**Exit the pipeline:**
Press `Q` to gracefully stop the pipeline.

### Training Models

**Train box detector:**
```bash
cd fine-tuning/combine-fine-tuning/box-YOLO
python train.py
```

**Train defect detector:**
```bash
cd fine-tuning/combine-fine-tuning/defect-YOLO
python train.py
```

**Run full pipeline (train → export → quantize):**
```bash
cd fine-tuning/combine-fine-tuning/box-YOLO
python run_all.py
```

## Project Structure

```
QC-SCM/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── fine-tuning/                       # Model training components
│   └── combine-fine-tuning/
│       ├── box-YOLO/                  # Carton detection model
│       │   ├── config.py              # Training config
│       │   ├── train.py               # Training script
│       │   ├── export_onnx.py         # Export to ONNX
│       │   ├── export_tflite.py       # Export to TFLite
│       │   ├── quantize_onnx.py       # INT8 quantization
│       │   ├── run_all.py             # Full pipeline
│       │   ├── utils.py               # Training utilities
│       │   ├── data/
│       │   │   ├── data.yaml          # Dataset config
│       │   │   ├── train/             # Training images & labels
│       │   │   ├── valid/             # Validation images & labels
│       │   │   └── test/              # Test images & labels
│       │   ├── runs/                  # Training outputs
│       │   │   ├── train/             # Checkpoints
│       │   │   └── metrics/           # Training metrics
│       │   └── *.pt                   # Pre-trained weights
│       │
│       └── defect-YOLO/               # Defect detection model
│           └── [Same structure as box-YOLO]
│
└── flow/                              # Real-time inference pipeline
    └── pipeline/
        ├── main.py                    # Main processing loop
        ├── config.py                  # Pipeline configuration
        ├── detector.py                # YOLO inference wrapper
        ├── tracker.py                 # Multi-object tracking
        ├── helpers.py                 # Status computation
        ├── utils.py                   # Image processing utilities
        ├── api_client.py              # Backend API client
        └── __pycache__/               # Python cache
```

## Technical Details

### Detection Pipeline

1. **Frame Capture**: Video stream from camera
2. **Carton Detection**: YOLO inference on full frame
3. **Region Expansion**: Expand detected box by `EXPAND_RATIO` for context
4. **Defect Detection**: Run defect detector on cropped region
5. **QR Reading**: Extract QR code from carton region
6. **Tracking**: Maintain tracking state per QR code
7. **Status Computation**: Determine final OK/Defect status
8. **Visualization**: Draw boxes and labels on frame
9. **API Logging**: Send results to backend

### Confidence Thresholds
- **CARTON_CONF = 0.5**: Detects main cartons with moderate confidence
- **DEFECT_CONF = 0.25**: Lower threshold for sensitive defect detection

### Tracking Strategy
- Uses QR codes as unique identifiers
- Maintains track state across frames
- Removes tracks after `MAX_DISAPPEAR` frames of no detection
- Aggregates defect information across multiple frames

### Output Format

The system outputs:
- **Annotated video frames** with bounding boxes and labels
- **Status indicators**: Green (OK), Red (Defect)
- **API POST requests** with:
  - QR code identifier
  - Detection status
  - Defect count
  - Timestamp
  - Session ID

## Model Export & Quantization

### Export to ONNX
```bash
cd fine-tuning/combine-fine-tuning/box-YOLO
python export_onnx.py
# Generates: best_box_detector.onnx
```

### INT8 Quantization
```bash
python quantize_onnx.py
# Generates: best_box_detector_int8.onnx
# Reduces model size ~4x, maintains accuracy
```

### Export to TensorFlow Lite
```bash
python export_tflite.py
# Generates: best_box_detector.tflite
# For mobile/edge deployment
```

### Model Sizes (Approximate)
- **PyTorch (.pt)**: 11 MB
- **ONNX (.onnx)**: 11 MB
- **ONNX INT8 (.onnx)**: 3 MB ⭐ Recommended for production
- **TFLite (.tflite)**: 3 MB

## Dependencies

Key packages (see `requirements.txt`):
- `torch==2.7.1+cu118` - Deep learning framework
- `ultralytics==8.3.229` - YOLO implementation
- `opencv-python==4.12.0` - Computer vision
- `pandas==2.3.3` - Data handling
- `numpy==1.26.4` - Numerical computing
- `onnxruntime==1.23.2` - ONNX model inference
- `onnxruntime-gpu==1.16.3` - GPU-accelerated inference

## Contributing

1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Make your changes
3. Commit (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License & Support

For issues, questions, or contributions, please contact the development team.
