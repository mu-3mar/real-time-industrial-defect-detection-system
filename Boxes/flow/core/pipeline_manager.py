"""
Single inference-thread pipeline manager (producer-consumer).

Architecture:
  Camera feeder(s)  -> frame_queue (bounded) -> Inference thread (single GPU)
                                                      -> result_queue (bounded)
                                                      -> Result consumer thread
                                                             -> WebRTC updates
                                                             -> firebase_queue -> Firebase worker thread

Inference never waits for Firebase or WebRTC. All heavy I/O and publishing
run in separate daemon threads.
"""

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np

from core.firebase_client import publish_detection
from core.pipeline_diagnostics import get_diagnostics

logger = logging.getLogger(__name__)

# Bounded queues to prevent latency buildup
FRAME_QUEUE_MAXSIZE = 5
RESULT_QUEUE_MAXSIZE = 5
FIREBASE_QUEUE_MAXSIZE = 64

# Sentinel for shutdown
_SHUTDOWN = object()


class PipelineManager:
    """
    Singleton: one inference thread, one result-consumer thread, one Firebase worker.
    Sessions register/unregister; camera feeders push (session_id, frame, enqueue_time, camera_fps).
    """

    _instance: Optional["PipelineManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._result_queue: queue.Queue = queue.Queue(maxsize=RESULT_QUEUE_MAXSIZE)
        self._firebase_queue: queue.Queue = queue.Queue(maxsize=FIREBASE_QUEUE_MAXSIZE)

        self._pipelines: Dict[str, Any] = {}  # session_id -> Pipeline
        self._tracks_registry: Dict[str, Tuple[Set[Any], threading.Lock]] = {}  # session_id -> (tracks, lock)
        self._firebase_meta: Dict[str, dict] = {}  # session_id -> meta for publish_detection

        self._lock_registry = threading.Lock()
        self._stop_event = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._result_consumer_thread: Optional[threading.Thread] = None
        self._firebase_thread: Optional[threading.Thread] = None
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
        """Start inference, result-consumer, and Firebase worker threads (idempotent)."""
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
            self._firebase_thread = threading.Thread(
                target=self._firebase_worker,
                name="FirebaseWorkerThread",
                daemon=True,
            )
            self._inference_thread.start()
            self._result_consumer_thread.start()
            self._firebase_thread.start()
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
        try:
            self._frame_queue.put_nowait((session_id, frame, enqueue_time, camera_fps))
            return True
        except queue.Full:
            return False

    def register_session(
        self,
        session_id: str,
        pipeline: Any,
        tracks_ref: Tuple[Set[Any], threading.Lock],
        firebase_meta: dict,
    ) -> None:
        """Register a session's pipeline, tracks, and Firebase metadata."""
        with self._lock_registry:
            self._pipelines[session_id] = pipeline
            self._tracks_registry[session_id] = tracks_ref
            self._firebase_meta[session_id] = firebase_meta
        self.start_workers()
        logger.debug("Registered session %s with PipelineManager", session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister and cleanup pipeline (e.g. call pipeline.cleanup())."""
        with self._lock_registry:
            pipeline = self._pipelines.pop(session_id, None)
            self._tracks_registry.pop(session_id, None)
            self._firebase_meta.pop(session_id, None)
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception as e:
                logger.warning("Pipeline cleanup error for session %s: %s", session_id, e)
        logger.debug("Unregistered session %s from PipelineManager", session_id)

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
                        self._firebase_queue.qsize(),
                    )
                self._inference_frame_count += 1
            except queue.Empty:
                continue
        logger.debug("Inference worker stopped")

    def _result_consumer_worker(self) -> None:
        """Consumes result_queue: update WebRTC tracks; forward exit events to Firebase queue."""
        logger.debug("Result consumer worker started")
        while not self._stop_event.is_set():
            try:
                item = self._result_queue.get(timeout=0.1)
                if item is _SHUTDOWN:
                    break
                session_id, canvas, exit_event = item
                # Update WebRTC tracks (non-blocking for inference)
                with self._lock_registry:
                    tracks_ref = self._tracks_registry.get(session_id)
                    firebase_meta = self._firebase_meta.get(session_id)
                if tracks_ref is not None:
                    tracks_set, tracks_lock = tracks_ref
                    with tracks_lock:
                        for track in list(tracks_set):
                            try:
                                t0 = time.perf_counter()
                                track.update_frame(canvas)
                                get_diagnostics().record_webrtc_update(time.perf_counter() - t0)
                            except Exception as e:
                                logger.error("[Error] track update session=%s: %s", session_id, e)
                if exit_event is not None and firebase_meta is not None:
                    try:
                        is_defect, detection_id = exit_event
                        self._firebase_queue.put_nowait((session_id, is_defect, detection_id, firebase_meta))
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
        logger.debug("Result consumer worker stopped")

    def _firebase_worker(self) -> None:
        """Dedicated thread for Firebase writes; never blocks inference or WebRTC."""
        logger.debug("Firebase worker started")
        while not self._stop_event.is_set():
            try:
                item = self._firebase_queue.get(timeout=0.1)
                if item is _SHUTDOWN:
                    break
                session_id, is_defect, detection_id, meta = item
                timestamp = datetime.utcnow().isoformat() + "Z"
                try:
                    publish_detection(
                        report_id=meta["report_id"],
                        detection_id=detection_id,
                        timestamp=timestamp,
                        defect=is_defect,
                    )
                except Exception as e:
                    logger.error("[Error] Firebase write report=%s: %s", session_id, e)
            except queue.Empty:
                continue
        logger.debug("Firebase worker stopped")

    def shutdown(self) -> None:
        """Signal workers to stop (e.g. on app shutdown)."""
        self._stop_event.set()
        for q in (self._frame_queue, self._result_queue, self._firebase_queue):
            try:
                q.put_nowait(_SHUTDOWN)
            except queue.Full:
                pass
        self._started = False
        logger.debug("PipelineManager shutdown")
