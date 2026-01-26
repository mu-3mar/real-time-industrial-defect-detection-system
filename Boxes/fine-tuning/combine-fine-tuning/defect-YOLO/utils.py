"""
Small utilities for directories, finding runs, copying metrics.
"""
import shutil
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_latest_run(project: str) -> Optional[Path]:
    p = Path(project)
    if not p.exists():
        return None
    dirs = [d for d in p.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda x: x.stat().st_mtime)[-1]


def find_best_pt(run_dir: Path) -> Optional[Path]:
    candidate = run_dir / "weights" / "best.pt"
    return candidate if candidate.exists() else None


def copy_raw_metrics_to_run(run_dir: Path, metrics_name: str = "metrics") -> Path:
    """
    Copy typical ultralytics output images from run root into run/metrics/.
    Keeps originals.
    """
    target = run_dir / metrics_name
    target.mkdir(parents=True, exist_ok=True)

    patterns = [
        "confusion_matrix*.png",
        "*curve*.png",
        "results.png",
        "results.csv",
        "labels*.jpg",
        "train_batch*.jpg",
        "val_batch*_pred.jpg",
        "val_batch*_labels.jpg"
    ]

    copied: List[str] = []
    for pat in patterns:
        for f in run_dir.glob(pat):
            try:
                shutil.copy(f, target / f.name)
                copied.append(f.name)
            except Exception:
                logger.debug("Could not copy %s", f)

    logger.info("Copied raw metrics: %s", copied)
    return target


def collect_final_metrics(run_dir: Path, dest_dir: Path) -> List[str]:
    """
    Copy final metrics (confusion, PR curves, results.csv/png) into a flattened dest_dir.
    No nested run folder inside dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "confusion_matrix*.png",
        "*curve*.png",
        "results.png",
        "results.csv"
    ]

    copied = []
    for location in [run_dir, run_dir / "metrics"]:
        for pat in patterns:
            for f in location.glob(pat):
                try:
                    shutil.copy(f, dest_dir / f.name)
                    copied.append(f.name)
                except Exception:
                    logger.debug("Failed to copy metric %s", f)

    logger.info("Collected final metrics into %s: %s", dest_dir, copied)
    return copied
