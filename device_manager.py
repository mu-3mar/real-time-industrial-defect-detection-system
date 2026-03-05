"""
Unified device selection for QC-SCM (training and flow/inference).

API:
    select_device(config_value=None, env_var="QC_SCM_DEVICE", context="") -> str

Supported device values: auto, cuda, mps, cpu.
Legacy numeric values (e.g. "0") are accepted and interpreted as CUDA.

Priority: config value (if set and not "auto") -> environment variable -> automatic.
Automatic order: CUDA -> MPS -> CPU.
If the requested device is unavailable, logs a warning and falls back to the best available.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_ALLOWED = frozenset({"auto", "cuda", "mps", "cpu"})
_LEGACY_CUDA = frozenset({"0", "1", "2", "3"})  # common GPU indices


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = (value or "").strip().lower()
    if not v:
        return None
    if v in _ALLOWED:
        return v
    if v in _LEGACY_CUDA:
        return "cuda"
    return v


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        return False


def _resolve_auto() -> str:
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return _cuda_available()
    if device == "mps":
        return _mps_available()
    return False


def select_device(
    config_value: Optional[str] = None,
    env_var: str = "QC_SCM_DEVICE",
    context: str = "",
) -> str:
    """
    Resolve runtime device: config (if not auto) -> env -> automatic (cuda -> mps -> cpu).

    Args:
        config_value: Value from config (e.g. "auto", "cuda", "cpu", or legacy "0").
        env_var: Environment variable to override when config is "auto".
        context: Optional label for logs (e.g. "flow", "training").

    Returns:
        Resolved device string: "cuda", "mps", or "cpu".
    """
    prefix = f"[{context}] " if context else ""
    cfg = _normalize(config_value)
    env = _normalize(os.environ.get(env_var))

    # Effective request: config wins unless it's "auto", then env, then "auto"
    if cfg and cfg != "auto":
        requested = cfg
    elif env and env != "auto":
        requested = env
    else:
        requested = "auto"

    if requested == "auto":
        resolved = _resolve_auto()
        logger.info("%sDevice: auto -> %s", prefix, resolved)
        return resolved

    if not _device_available(requested):
        fallback = _resolve_auto()
        logger.warning(
            "%sRequested device %s is unavailable; falling back to %s",
            prefix, requested, fallback,
        )
        return fallback

    logger.info("%sDevice: %s", prefix, requested)
    return requested
