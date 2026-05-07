Defect detector fine-tuning workspace.

This training setup fine-tunes a pretrained defect checkpoint using:
- `../data_preparation/datasets/defect_dataset/data.yaml`

Default pretrained checkpoint:
- `models/pretrained/defect_detector.pt`

## Folder layout

- `configs/`: central configuration.
- `training/`: training entrypoint.
- `export/`: ONNX export and quantization.
- `inference/`: simple inference script.
- `scripts/`: utility scripts.
- `models/pretrained/`: source checkpoints for fine-tuning.
- `models/exported/`: exported artifacts (`best.pt`, ONNX files).
- `runs/`: Ultralytics training runs and metrics.

## Run fine-tuning

From project root:

- `python3 defect_train/training/train.py`

Or full pipeline (train + export + quantize + metrics):

- `python3 defect_train/scripts/run_all.py`

Pipeline order:
- train
- export ONNX
- export TensorRT engine
- quantize ONNX (INT8)
- collect metrics

## Optional overrides

Environment variables:

- `DEFECT_TRAIN_BASE_MODEL`
- `DEFECT_TRAIN_DATA_YAML`
- `DEFECT_TRAIN_EPOCHS`
- `DEFECT_TRAIN_BATCH_SIZE`
- `DEFECT_TRAIN_IMG_SIZE`
- `DEFECT_TRAIN_DEVICE`
- `DEFECT_TRAIN_ENGINE_DEVICE`
- `DEFECT_TRAIN_ENGINE_HALF`
