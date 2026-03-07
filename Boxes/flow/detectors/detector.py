"""YOLO-based object detector wrapper."""

import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class Detector:
    """Wrapper for YOLO inference with consistent parameters."""

    def __init__(self, model: YOLO, conf: float, iou: float, device: str):
        """
        Initialize detector with pre-loaded model.

        Args:
            model: Pre-loaded YOLO model instance
            conf: Confidence threshold (0-1)
            iou: IoU threshold for NMS (0-1)
            device: Device to run inference on (cpu, cuda, etc.)
        """
        self.model = model
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame, verbose: bool = False):
        """
        Run inference on a frame or crop.

        Args:
            frame: Input image array
            verbose: Whether to print inference details

        Returns:
            First result object from ultralytics Results
        """
        if _TORCH_AVAILABLE and hasattr(torch, "inference_mode"):
            with torch.inference_mode():
                results = self.model(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    verbose=verbose,
                )
        else:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=verbose,
            )
        return results[0]
