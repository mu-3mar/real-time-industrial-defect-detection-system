import os
import logging
from config import ROOT, PROJECT, FINAL_METRICS_DIR
from train import train
from export_onnx import export_onnx
from quantize_onnx import quantize
from utils import collect_final_metrics

os.chdir(str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    train()
    run_dir = PROJECT / "bottle_info"
    onnx_path = export_onnx(run_dir)
    quantize(run_dir, onnx_path)
    collect_final_metrics(run_dir, FINAL_METRICS_DIR)

if __name__ == "__main__":
    main()
