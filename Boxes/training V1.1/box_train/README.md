Box detector fine-tuning workspace.

This training setup fine-tunes a pretrained detector checkpoint using:
- `../data_preparation/datasets/detector_dataset/data.yaml`

Default pretrained checkpoint:
- `models/pretrained/box_detector.pt`

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

- `python3 box_train/training/train.py`

Or full pipeline (train + export + quantize + metrics):

- `python3 box_train/scripts/run_all.py`

Pipeline order:
- train
- export ONNX
- export TensorRT engine
- quantize ONNX (INT8)
- collect metrics

## Optional overrides

Environment variables:

- `BOX_TRAIN_BASE_MODEL`
- `BOX_TRAIN_DATA_YAML`
- `BOX_TRAIN_EPOCHS`
- `BOX_TRAIN_BATCH_SIZE`
- `BOX_TRAIN_IMG_SIZE`
- `BOX_TRAIN_DEVICE`
- `BOX_TRAIN_ENGINE_DEVICE`
- `BOX_TRAIN_ENGINE_HALF`
