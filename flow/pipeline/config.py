# pipeline/config.py
from dataclasses import dataclass
import uuid

@dataclass
class Config:
    # default model paths (edit to your actual paths)
    CARTON_MODEL: str = (
        "fine-tuning/combine-fine-tuning/box-YOLO/runs/train/detect_boxs/weights/"
        "best_box_detector_int8.onnx"
    )
    DEFECT_MODEL: str = (
        "fine-tuning/combine-fine-tuning/defect-YOLO/runs/train/defect/weights/"
        "best_defect_detector_int8.onnx"
    )

    # camera
    CAMERA_INDEX: int = 0

    # thresholds
    CARTON_CONF: float = 0.5
    DEFECT_CONF: float = 0.25

    # tracking behavior
    MAX_DISAPPEAR: int = 12
    EXPAND_RATIO: float = 0.1  # small expand

    # API
    API_URL: str = "https://chainly.azurewebsites.net/api/ProductionLines/sessions"
    PRODUCTION_LINE_ID: int = 1
    COMPANY_ID: int = 90

    # session identifier
    SESSION_ID: str = uuid.uuid4().hex

cfg = Config()
