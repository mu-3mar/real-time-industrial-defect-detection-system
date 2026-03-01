from collections import deque

from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np


class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an external source.
    Uses a deque (maxlen=1) for thread-safe handoff between pipeline and WebRTC.
    """
    def __init__(self):
        super().__init__()
        self._frame_queue: deque = deque(maxlen=1)
        self._fallback_frame: np.ndarray | None = None

    def update_frame(self, frame: np.ndarray) -> None:
        """Push frame from pipeline thread. Thread-safe; latest frame wins."""
        self._frame_queue.append(frame)
        self._fallback_frame = frame

    async def recv(self):
        """
        Called by aiortc to get the next frame.
        """
        pts, time_base = await self.next_timestamp()

        frame_data = None
        if self._frame_queue:
            frame_data = self._frame_queue.popleft()
        elif self._fallback_frame is not None:
            frame_data = self._fallback_frame

        if frame_data is None:
            frame = VideoFrame(width=640, height=480)
            for p in frame.planes:
                p.update(bytes(p.buffer_size))
        else:
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")

        frame.pts = pts
        frame.time_base = time_base
        return frame
