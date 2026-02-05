import os
import yaml
from pathlib import Path
from core.pipeline import Pipeline

# Suppress Logs
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["ORT_LOGGING_LEVEL"] = "3"

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    pass

def load_config(path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    BASE = Path(__file__).resolve().parent
    
    try:
        box_cfg = load_config(BASE / "configs/box_detector.yaml")
        defect_cfg = load_config(BASE / "configs/defect_detector.yaml")
        str_cfg = load_config(BASE / "configs/stream.yaml")
        
        # Inject constants that were previously hardcoded if missing from yaml
        if "stability" not in defect_cfg:
            defect_cfg["stability"] = {
                "min_frames": 3,
                "max_missed": 5,
                "vote_window": 7,
                "vote_threshold": 4
            }
            
        app = Pipeline(box_cfg, defect_cfg, str_cfg)
        app.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        raise e

if __name__ == "__main__":
    main()
