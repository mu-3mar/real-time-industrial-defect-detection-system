"""QC-SCM Detection Service entry point."""

import os
from pathlib import Path

import yaml
import uvicorn

# Suppress warnings and debug logging
os.environ["ORT_LOGGING_LEVEL"] = "3"

try:
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
except ImportError:
    pass


def main():
    """Launch FastAPI server."""
    base = Path(__file__).resolve().parent
    config_path = base / "configs/api_server.yaml"

    # Load API configuration
    if config_path.exists():
        with open(config_path) as f:
            api_cfg = yaml.safe_load(f)
    else:
        api_cfg = {
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
        }

    print(f"Starting QC-SCM Detection Service on {api_cfg['host']}:{api_cfg['port']}")

    uvicorn.run(
        "api_server:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        log_level=api_cfg["log_level"],
        reload=False,
    )


if __name__ == "__main__":
    main()
