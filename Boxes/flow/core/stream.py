"""Video stream reader with OpenCV."""

import logging

import cv2

logger = logging.getLogger(__name__)


class CamStream:
    """Wrapper for OpenCV video capture."""

    def __init__(self, source, width, height):
        # Use V4L2 backend and force MJPG to avoid YUYV (10 FPS bottleneck on many webcams)
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            logger.error("Failed to open video source: %s", source)
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        """Read the next frame from the stream."""
        return self.cap.read()

    def release(self):
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
