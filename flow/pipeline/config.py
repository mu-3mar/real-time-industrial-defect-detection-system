from dataclasses import dataclass
import uuid


@dataclass
class Config:
    """Pipeline configuration."""
    CARTON_MODEL: str = (
        "fine-tuning/combine-fine-tuning/box-YOLO/runs/train/detect_boxs/weights/"
        "best_box_detector_int8.onnx"
    )
    DEFECT_MODEL: str = (
        "fine-tuning/combine-fine-tuning/defect-YOLO/runs/train/detect_defects/weights/"
        "best_defect_detector_int8.onnx"
    )
    CAMERA_INDEX: int = 0
    CARTON_CONF: float = 0.85
    DEFECT_CONF: float = 0.3
    MAX_DISAPPEAR: int = 12
    EXPAND_RATIO: float = 0.05
    API_URL: str = "https://chainly.azurewebsites.net/api/ProductionLines/sessions"
    PRODUCTION_LINE_ID: int = 1
    COMPANY_ID: int = 90
    SESSION_ID: str = uuid.uuid4().hex

cfg = Config()
