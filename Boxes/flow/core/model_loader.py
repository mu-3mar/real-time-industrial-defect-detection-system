"""Singleton loader for YOLO detection models."""

import logging
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class for loading and sharing YOLO models across sessions."""

    _instance: Optional["ModelLoader"] = None

    def __init__(self):
        self.box_model: Optional[YOLO] = None
        self.defect_model: Optional[YOLO] = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self, box_model_path: str, defect_model_path: str) -> None:
        """
        Load both detection models.

        Args:
            box_model_path: Path to box detection model
            defect_model_path: Path to defect detection model
        """
        if self._loaded:
            logger.info("Models already loaded")
            return

        logger.info("Loading box detection model from %s", box_model_path)
        self.box_model = YOLO(box_model_path, task="detect")

        logger.info("Loading defect detection model from %s", defect_model_path)
        self.defect_model = YOLO(defect_model_path, task="detect")

        self._loaded = True
        logger.info("Models loaded successfully")

    def warmup(self, device: str = "0") -> None:
        """Run dummy inference to warm up GPU/cache. Reduces first-frame latency."""
        if not self._loaded:
            return
        try:
            import numpy as np
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.box_model(dummy, verbose=False, device=device)
            self.defect_model(dummy, verbose=False, device=device)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning("Model warmup failed (non-fatal): %s", e)

    def get_box_model(self) -> YOLO:
        """Get box detection model."""
        if not self._loaded or self.box_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.box_model

    def get_defect_model(self) -> YOLO:
        """Get defect detection model."""
        if not self._loaded or self.defect_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.defect_model
