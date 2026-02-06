import os
import yaml
import uvicorn
from pathlib import Path

# Suppress Logs
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["ORT_LOGGING_LEVEL"] = "3"

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    pass

def main():
    """Launch FastAPI server."""
    BASE = Path(__file__).resolve().parent
    
    # Load API server config
    config_path = BASE / "configs/api_server.yaml"
    if config_path.exists():
        with open(config_path) as f:
            api_cfg = yaml.safe_load(f)
    else:
        # Default config
        api_cfg = {
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info"
        }
    
    print(f"Starting QC-SCM Detection Service on {api_cfg['host']}:{api_cfg['port']}")
    
    # Run uvicorn server
    uvicorn.run(
        "api_server:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        log_level=api_cfg["log_level"],
        reload=False
    )

if __name__ == "__main__":
    main()
