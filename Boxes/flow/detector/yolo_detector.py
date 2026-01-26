from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf, iou, device):
        self.model = YOLO(model_path, task='detect')
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame):
        r = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        return r[0]
