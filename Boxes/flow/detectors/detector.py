from ultralytics import YOLO
import logging

class Detector:
    def __init__(self, model_path, conf, iou, device):
        self.model = YOLO(model_path, task='detect')
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame, verbose=False):
        """
        Runs inference on a frame/crop.
        Returns the first result object (Result).
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=verbose
        )
        return results[0]
