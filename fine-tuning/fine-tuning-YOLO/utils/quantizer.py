from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantise(onnx_path: Path, data_yaml: Path, save_path: Path) -> None:
    """Dynamic INT8 quantisation (no calibration images needed)."""
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(save_path),
        weight_type=QuantType.QUInt8
    )
    print(f"[INFO] INT8 ONNX saved → {save_path}")