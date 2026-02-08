from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np
import time

class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an external source.
    """
    def __init__(self):
        super().__init__()
        self._latest_frame = None

    def update_frame(self, frame: np.ndarray):
        """ Update the current frame to be served. """
        self._latest_frame = frame

    async def recv(self):
        """
        Called by aiortc to get the next frame.
        """
        pts, time_base = await self.next_timestamp()

        # If no frame is available, send a black dummy frame
        if self._latest_frame is None:
            frame = VideoFrame(width=640, height=480)
            for p in frame.planes:
                p.update(bytes(p.buffer_size))
        else:
            # Create VideoFrame from numpy array (BGR)
            frame = VideoFrame.from_ndarray(self._latest_frame, format="bgr24")

        frame.pts = pts
        frame.time_base = time_base
        return frame
