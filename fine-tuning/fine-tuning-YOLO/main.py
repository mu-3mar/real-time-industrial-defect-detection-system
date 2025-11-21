import shutil, sys, time
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------
#  single allowed folder
# ------------------------------------------------------------------
OUT = Path("output")
OUT.mkdir(exist_ok=True)
LAST_EPOCH_FILE = OUT / "epoch.txt"
# ------------------------------------------------------------------

def read_epoch() -> int:
    return int(LAST_EPOCH_FILE.read_text()) if LAST_EPOCH_FILE.exists() else 0

def write_epoch(e: int):
    LAST_EPOCH_FILE.write_text(str(e))

def train():
    # 1.  base config  –  WITHOUT 'epochs'
    cfg = {"data": "data/data.yaml",
           "imgsz": 640, "batch": 8, "device": "0",
           "project": str(OUT), "name": "train",
           "exist_ok": True, "plots": False, "val": False}

    start = read_epoch() + 1
    if start > 100:                      # 100 is our total
        print("Training already finished."); return

    best_map = 0.0
    for epoch in range(start, 101):      # 1 … 100
        print(f"\n----- epoch {epoch}/100 -----")

        model = YOLO(str(OUT / "last.pt")) if epoch > 1 else YOLO("yolov8n.pt")

        # 2.  pass epochs=1 explicitly, NOT inside cfg
        model.train(**cfg, epochs=1)

        # 3.  move weights into our single folder
        u_last = OUT / "train" / "weights" / "last.pt"
        u_best = OUT / "train" / "weights" / "best.pt"
        shutil.copy2(u_last, OUT / "last.pt")
        if u_best.exists():
            shutil.copy2(u_best, OUT / "best.pt")

        write_epoch(epoch)


    # ---------------- end of training ------------------------------
    print("\nFinal validation …")
    final_model = YOLO(str(OUT / "best.pt"))
    metrics = final_model.val(data=cfg["data"])
    print("mAP50-95 =", metrics.box.map)

    # export & quantise
    onnx_path = OUT / "best.onnx"
    final_model.export(format="onnx", imgsz=640, simplify=True, dynamic=True, opset=12)
    # ultralytics always writes best.onnx next to best.pt
    (OUT / "best.onnx").rename(onnx_path)

    # quantise
    from utils.quantizer import quantise
    quantise(onnx_path, Path(cfg["data"]), OUT / "best_int8.onnx")
    print("Done – files in", OUT.absolute())

if __name__ == "__main__":
    train()