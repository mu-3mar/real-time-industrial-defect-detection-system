"""Singleton loader for YOLO detection models."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _silence_tensorrt() -> None:
    """Monkey-patch trt.Logger default to ERROR so TensorRT C++ runtime is quiet."""
    try:
        import tensorrt as trt

        _OrigLogger = trt.Logger
        _ERROR = trt.Logger.ERROR

        class _QuietLogger(_OrigLogger):
            def __init__(self, severity=None):
                super().__init__(_ERROR)

        trt.Logger = _QuietLogger
    except ImportError:
        pass


def _disable_ultralytics_autoupdate() -> None:
    """
    Prevent Ultralytics check_requirements() from running pip AutoUpdate.
    When onnxruntime-gpu is installed Ultralytics incorrectly sees 'onnxruntime'
    as missing and installs onnxruntime (CPU) over it, breaking GPU inference.
    """
    try:
        from ultralytics.utils import checks  # type: ignore

        _orig = checks.check_requirements

        def _safe_check_requirements(requirements=None, package="ultralytics", install=True, cmds=None, exclude=None):
            return _orig(requirements, package=package, install=False, cmds=cmds, exclude=exclude)

        checks.check_requirements = _safe_check_requirements
        logger.debug("Ultralytics check_requirements patched: AutoUpdate disabled")
    except Exception:
        pass


def _configure_onnx_providers() -> None:
    """
    Prioritize ONNX Runtime CUDAExecutionProvider and skip TensorrtExecutionProvider.
    Ensures ONNX models run via ORT CUDA (or CPU fallback), not via ORT's TRT EP.
    Patches onnxruntime.get_available_providers before Ultralytics builds its sessions.
    """
    try:
        import onnxruntime as ort  # type: ignore

        _original = ort.get_available_providers

        def _patched() -> list:
            providers = _original()
            preferred = []
            if "CUDAExecutionProvider" in providers:
                preferred.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in providers:
                preferred.append("CPUExecutionProvider")
            return preferred if preferred else providers

        ort.get_available_providers = _patched
        logger.debug("ONNX Runtime providers: CUDA first, CPU fallback, TensorRT skipped")
    except ImportError:
        pass


_silence_tensorrt()
_configure_onnx_providers()
_disable_ultralytics_autoupdate()

from ultralytics import YOLO


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
            logger.debug("Models already loaded")
            return

        self.box_model = YOLO(box_model_path, task="detect")
        self.defect_model = YOLO(defect_model_path, task="detect")
        self._loaded = True
        logger.info("[Service] models loaded")

    def warmup(self, device: str = "0") -> None:
        """Run dummy inference to warm up GPU/cache. Reduces first-frame latency."""
        if not self._loaded:
            return
        try:
            import numpy as np
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.box_model(dummy, verbose=False, device=device)
            self.defect_model(dummy, verbose=False, device=device)
            logger.debug("Model warmup complete")
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
