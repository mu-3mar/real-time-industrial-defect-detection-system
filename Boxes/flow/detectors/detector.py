from ultralytics import YOLO
import logging

class Detector:
    def __init__(self, model, conf, iou, device):
        """
        Initialize detector with pre-loaded model.
        
        Args:
            model: Pre-loaded YOLO model instance
            conf: Confidence threshold
            iou: IoU threshold
            device: Device to run inference on
        """
        self.model = model
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
