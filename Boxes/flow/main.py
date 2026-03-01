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
    """Launch FastAPI server. Configuration from config/api.yaml."""
    base = Path(__file__).resolve().parent
    api_cfg_path = base / "config" / "api.yaml"
    if api_cfg_path.exists():
        with open(api_cfg_path) as f:
            api_cfg = yaml.safe_load(f) or {}
    else:
        api_cfg = {}

    api_cfg.setdefault("host", "0.0.0.0")
    api_cfg.setdefault("port", 8000)
    api_cfg.setdefault("log_level", "info")

    port = api_cfg["port"]
    print("=" * 60)
    print("QC-SCM Detection Service")
    print("=" * 60)
    print(f"API Base URL: http://localhost:{port}")
    print(f"  LAN:        http://<your-ip>:{port}")
    print(f"  Docs:       http://localhost:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "api_server:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        log_level=api_cfg["log_level"],
        reload=False,
    )


if __name__ == "__main__":
    main()
