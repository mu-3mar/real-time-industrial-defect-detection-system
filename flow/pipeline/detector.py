# pipeline/detector.py
from ultralytics import YOLO
import numpy as np
from typing import Tuple

class Detector:
    def __init__(self, model_path: str):
        # YOLO accepts .onnx and .pt - we use the provided path
        self.model = YOLO(str(model_path))

    def infer(self, frame: np.ndarray, conf: float):
        """
        Returns: (boxes, scores, classes)
        boxes: numpy array Nx4 with xyxy
        scores: numpy array N
        classes: numpy array N
        """
        res = self.model(frame, conf=conf, verbose=False)[0]
        if not hasattr(res, "boxes") or len(res.boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        return boxes, scores, classes
