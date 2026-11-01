"""
Single inference-thread pipeline manager (producer-consumer).

Architecture:
  Camera feeder(s)  -> frame_queue (bounded) -> Inference thread (single GPU)
                                                      -> result_queue (bounded)
                                                      -> Result consumer thread
                                                             -> MJPEG frame updates
                                                             -> local_store_queue -> local storage worker

Inference never waits for local storage or MJPEG consumers. All heavy I/O and publishing
run in separate daemon threads.
"""

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np

from core.local_session_store import publish_detection
from core.pipeline_diagnostics import get_diagnostics

logger = logging.getLogger(__name__)

# Bounded queues to prevent latency buildup
FRAME_QUEUE_MAXSIZE = 5
RESULT_QUEUE_MAXSIZE = 5
LOCAL_STORE_QUEUE_MAXSIZE = 64

# Sentinel for shutdown
_SHUTDOWN = object()


class PipelineManager:
    """
    Singleton: one inference thread, one result-consumer thread, one local storage worker.
    Sessions register/unregister; camera feeders push (session_id, frame, enqueue_time, camera_fps).
    """

    _instance: Optional["PipelineManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._result_queue: queue.Queue = queue.Queue(maxsize=RESULT_QUEUE_MAXSIZE)
        self._local_store_queue: queue.Queue = queue.Queue(maxsize=LOCAL_STORE_QUEUE_MAXSIZE)

        self._pipelines: Dict[str, Any] = {}  # session_id -> Pipeline
        self._latest_frames: Dict[str, np.ndarray] = {}  # session_id -> latest annotated frame
        self._local_store_meta: Dict[str, dict] = {}

        self._lock_registry = threading.Lock()
        self._stop_event = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._result_consumer_thread: Optional[threading.Thread] = None
        self._local_store_thread: Optional[threading.Thread] = None
        self._started = False
        self._inference_frame_count: int = 0

    @classmethod
    def get_instance(cls) -> "PipelineManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_workers(self) -> None:
        """Start inference, result-consumer, and local storage workers."""
        with self._lock_registry:
            if self._started:
                return
            self._stop_event.clear()
            self._inference_thread = threading.Thread(
                target=self._inference_worker,
                name="InferenceThread",
                daemon=True,
            )
            self._result_consumer_thread = threading.Thread(
                target=self._result_consumer_worker,
                name="ResultConsumerThread",
                daemon=True,
            )
            self._local_store_thread = threading.Thread(
                target=self._local_store_worker,
                name="LocalSessionStoreThread",
                daemon=True,
            )
            self._inference_thread.start()
            self._result_consumer_thread.start()
            self._local_store_thread.start()
            self._started = True
            logger.debug("PipelineManager workers started")

    def put_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        enqueue_time: float = 0.0,
        camera_fps: float = 0.0,
    ) -> bool:
        """
        Non-blocking put from camera feeder. Drops if full so inference is never blocked.
        Returns True if enqueued, False if queue full (frame dropped).
        """
        with self._lock_registry:
            pipeline = self._pipelines.get(session_id)
        strict_mode = bool(getattr(pipeline, "strict_debug_mode", False)) if pipeline is not None else False
        try:
            if strict_mode:
                self._frame_queue.put((session_id, frame, enqueue_time, camera_fps), timeout=0.2)
            else:
                self._frame_queue.put_nowait((session_id, frame, enqueue_time, camera_fps))
            return True
        except queue.Full:
            return False

    def register_session(
        self,
        session_id: str,
        pipeline: Any,
        session_meta: dict,
    ) -> None:
        """Register a session's pipeline and local-storage metadata."""
        with self._lock_registry:
            self._pipelines[session_id] = pipeline
            self._local_store_meta[session_id] = session_meta
        self.start_workers()
        logger.debug("Registered session %s with PipelineManager", session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister and cleanup pipeline (e.g. call pipeline.cleanup())."""
        with self._lock_registry:
            pipeline = self._pipelines.pop(session_id, None)
            self._latest_frames.pop(session_id, None)
            self._local_store_meta.pop(session_id, None)
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception as e:
                logger.warning("Pipeline cleanup error for session %s: %s", session_id, e)
        logger.debug("Unregistered session %s from PipelineManager", session_id)

    def get_latest_frame(self, session_id: str) -> Optional[np.ndarray]:
        """Get the latest annotated frame for a session."""
        with self._lock_registry:
            return self._latest_frames.get(session_id)

    def _inference_worker(self) -> None:
        """Single thread that runs all GPU inference (one frame at a time from frame_queue)."""
        logger.debug("Inference worker started")
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
                if item is _SHUTDOWN:
                    break
                session_id, frame, enqueue_time, camera_fps = item
                with self._lock_registry:
                    pipeline = self._pipelines.get(session_id)
                if pipeline is None:
                    continue
                t0 = time.perf_counter()
                try:
                    canvas, exit_event = pipeline.run_step(
                        frame,
                        enqueue_time=enqueue_time if enqueue_time > 0 else None,
                        camera_fps=camera_fps if camera_fps > 0 else None,
                    )
                except Exception as e:
                    logger.exception("Inference run_step error session=%s: %s", session_id, e)
                    continue
                latency_sec = time.perf_counter() - t0
                get_diagnostics().record_inference(latency_sec)
                try:
                    self._result_queue.put_nowait((session_id, canvas, exit_event))
                except queue.Full:
                    get_diagnostics().record_result_queue_drop()
                if self._inference_frame_count % 30 == 0:
                    get_diagnostics().maybe_log(
                        self._frame_queue.qsize(),
                        self._result_queue.qsize(),
                        self._local_store_queue.qsize(),
                    )
                self._inference_frame_count += 1
            except queue.Empty:
                continue
        logger.debug("Inference worker stopped")

    def _result_consumer_worker(self) -> None:
        """Consumes results: store latest frame and forward exit events to local storage."""
        logger.debug("Result consumer worker started")
        while not self._stop_event.is_set():
            try:
                item = self._result_queue.get(timeout=0.1)
                if item is _SHUTDOWN:
                    break
                session_id, canvas, exit_event = item

                with self._lock_registry:
                    pipeline = self._pipelines.get(session_id)
                if pipeline is None:
                    continue

                # The dashboard renders stats below the stream, not inside it.
                stream_frame = canvas[:, pipeline.visualizer.info_width:].copy()
                # Store the latest camera frame for MJPEG streaming.
                with self._lock_registry:
                    if session_id in self._pipelines:
                        self._latest_frames[session_id] = stream_frame
                        session_meta = self._local_store_meta.get(session_id)
                    else:
                        session_meta = None

                if exit_event is not None and session_meta is not None:
                    try:
                        self._local_store_queue.put_nowait((session_id, exit_event, session_meta))
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
        logger.debug("Result consumer worker stopped")

    def _local_store_worker(self) -> None:
        """Dedicated thread for local result writes; never blocks inference."""
        logger.debug("Local session store worker started")
        while not self._stop_event.is_set():
            try:
                item = self._local_store_queue.get(timeout=0.1)
                if item is _SHUTDOWN:
                    break
                session_id, is_defect, meta = item
                timestamp = datetime.utcnow().isoformat() + "Z"
                try:
                    local_key = publish_detection(
                        report_id=meta["report_id"],
                        timestamp=timestamp,
                        defect=is_defect,
                    )
                    if local_key:
                        logger.debug(
                            "Local detection key=%s report=%s",
                            local_key,
                            meta["report_id"],
                        )
                except Exception as e:
                    logger.error("[Error] local result write report=%s: %s", session_id, e)
            except queue.Empty:
                continue
        logger.debug("Local session store worker stopped")

    def shutdown(self) -> None:
        """Signal workers to stop (e.g. on app shutdown)."""
        self._stop_event.set()
        for q in (self._frame_queue, self._result_queue, self._local_store_queue):
            try:
                q.put_nowait(_SHUTDOWN)
            except queue.Full:
                pass
        self._started = False
        logger.debug("PipelineManager shutdown")
