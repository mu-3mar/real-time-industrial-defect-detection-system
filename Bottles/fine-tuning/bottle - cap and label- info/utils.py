import shutil
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def prune_weights(weights_dir: Path):
    for f in weights_dir.glob("*.pt"):
        if f.name not in ("best.pt", "last.pt"):
            f.unlink(missing_ok=True)

def collect_final_metrics(run_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "results.png",
        "confusion_matrix*.png",
        "*curve*.png",
        "results.csv"
    ]
    for pat in patterns:
        for f in run_dir.glob(pat):
            dst = dest_dir / f.name
            if not dst.exists():
                shutil.copy(f, dst)
