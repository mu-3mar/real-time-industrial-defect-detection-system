# YOLO Defect Detection Project

## Overview
This project implements a defect detection system using YOLOv8. It includes training, inference, and model export (ONNX with quantization) capabilities.

## Structure
- `configs/`: Configuration files.
- `data/`: Data directory containing datasets.
  - `data` (merged), `data 1`, `data 2`.
- `models/`: Pretrained and exported models.
- `training/`: Training scripts.
- `inference/`: Inference scripts.
- `export/`: Scripts for exporting and quantizing models.
- `scripts/`: Utility scripts for running pipelines and data merging.
- `utils/`: Common utility functions.
- `runs/`: Training output directory (weights, logs, metrics).

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training & Export Pipeline
To run the full pipeline (train -> export -> quantize):
```bash
python scripts/run_all.py
```

### Inference
To run inference using the webcam:
```bash
python inference/infer.py
```
*Note: Ensure your webcam is connected and the index in `infer.py` is correct.*

### Data Merging
To merge `data 1` and `data 2` into `data`:
```bash
python scripts/merge_data.py
```
