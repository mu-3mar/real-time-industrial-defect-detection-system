import cv2

class CamStream:
    def __init__(self, source, width, height):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
             print(f"Error: Could not open video source {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
