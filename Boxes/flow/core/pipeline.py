import cv2
import time
from datetime import datetime
import numpy as np
from core.state import AppState
from core.stream import CamStream
from detectors.detector import Detector
from utils.visualizer import Visualizer
from utils.geometry import Geometry

class Pipeline:
    def __init__(self, box_cfg, defect_cfg, stream_cfg):
        # 1. Setup Stream
        self.stream = CamStream(
            stream_cfg["source"], 
            stream_cfg["width"], 
            stream_cfg["height"]
        )
        
        # 2. Setup Detectors
        self.box_detector = Detector(
            box_cfg["model_path"],
            box_cfg["conf_thres"],
            box_cfg["iou_thres"],
            box_cfg["device"]
        )
        
        self.defect_detector = Detector(
            defect_cfg["model_path"],
            defect_cfg["conf_thres"],
            defect_cfg["iou_thres"],
            defect_cfg["device"]
        )
        
        # 3. Setup State
        self.state = AppState(defect_cfg.get("stability", {}))
        
        # 4. Setup Visualizer
        # Constants from original code
        INFO_WIDTH = 300
        ROI_WIDTH = 400
        ROI_CENTER_OFFSET = 420
        self.LEFT_X = INFO_WIDTH + ROI_CENTER_OFFSET - ROI_WIDTH // 2
        self.RIGHT_X = INFO_WIDTH + ROI_CENTER_OFFSET + ROI_WIDTH // 2
        
        self.visualizer = Visualizer(
            stream_cfg["width"], 
            stream_cfg["height"], 
            INFO_WIDTH, 
            ROI_WIDTH
        )
        
        # Optimization
        self.SKIP_DEFECT_FRAMES = 2
        
        # Metrics
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0

    def run(self):
        cv2.namedWindow("Box Inspection System", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.stream.read()
                if not ret:
                    break
                
                self.frame_count += 1
                self.update_fps()
                
                h, w = frame.shape[:2]
                
                # Canvas Setup
                canvas = cv2.copyMakeBorder(
                    frame, 0, 0, self.visualizer.info_width, 0,
                    cv2.BORDER_CONSTANT, value=(235, 235, 235)
                )
                
                self.visualizer.draw_layout(canvas)
                self.visualizer.draw_stats(canvas, self.state, self.fps)
                
                # ROI Processing
                roi = frame[:, self.LEFT_X - self.visualizer.info_width : self.RIGHT_X - self.visualizer.info_width]
                
                # --- STAGE 1: Box Detection ---
                box_result = self.box_detector.detect(roi)
                detected = False
                
                if box_result.boxes is not None:
                    for box in box_result.boxes.xyxy.cpu().numpy():
                        detected = True
                        
                        # --- STAGE 2: Defect Detection ---
                        # Crop from original frame
                        # Note: box is relative to ROI. 
                        # x1 relative to ROI -> x1 + (LEFT_X - INFO_WIDTH) relative to frame
                        x1, y1, x2, y2 = map(int, box)
                        frame_x1 = x1 + (self.LEFT_X - self.visualizer.info_width)
                        frame_x2 = x2 + (self.LEFT_X - self.visualizer.info_width)
                        
                        crop = frame[y1:y2, frame_x1:frame_x2]
                        
                        # Optimization & Logic
                        box_id = (x1 // 20, y1 // 20) # Simple spatial ID
                        
                        is_defect, defect_boxes = self.check_defect(crop, box_id)
                        
                        # Update State
                        self.state.update_history(box_id, is_defect)
                        
                        # Get Status for UI
                        label, color, final_code = self.state.get_status(box_id)
                        
                        # Draw
                        # We pass the box relative to ROI, Visualizer adds offset
                        self.visualizer.draw_box(canvas, box, label, color)
                        
                        # Draw Specific Defects
                        if defect_boxes:
                            # Pass box origin (x1, y1) to help offset defect coords
                            self.visualizer.draw_defects(canvas, (x1, y1), defect_boxes)

                # --- State Update (Entry/Exit) ---
                just_exited = self.state.process_entry_exit(detected)
                if just_exited:
                     print(f"[{datetime.now().strftime('%H:%M:%S')}] Box Processed: {self.state.final_decision} | Total: {self.state.total_count}")

                cv2.imshow("Box Inspection System", canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.stream.release()
            cv2.destroyAllWindows()

    def check_defect(self, crop, box_id):
        """
        Determines if a crop has a defect. 
        Returns (is_defect_bool, list_of_defect_boxes).
        Uses caching and frame skipping optimization.
        """
        # Determine if we should run defect detection
        should_run = (box_id not in self.state.last_defect_results) or (self.frame_count % (self.SKIP_DEFECT_FRAMES + 1) == 0)
        
        hole_detected = False
        defect_boxes = []
        
        if should_run:
            if crop.size > 0:
                d_res = self.defect_detector.detect(crop)
                if d_res.boxes is not None:
                    # Filter for defect class (assuming 0 is Hole/Defect)
                    # We might want to pass all defects found
                    defects = d_res.boxes.data.cpu().numpy() # x1, y1, x2, y2, conf, cls
                    for d in defects:
                        cls = int(d[5])
                        if cls == 0:
                            hole_detected = True
                            defect_boxes.append(d[:4])
                            
            self.state.last_defect_results[box_id] = (hole_detected, defect_boxes)
        else:
            # Retrieve from cache
            result = self.state.last_defect_results.get(box_id, (False, []))
            # Handle backward compatibility if cache was old structure (unlikely in runtime but safe)
            if isinstance(result, tuple):
                hole_detected, defect_boxes = result
            else:
                hole_detected = result
                defect_boxes = []
            
        return hole_detected, defect_boxes

    def update_fps(self):
        if self.frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            if elapsed > 0:
                self.fps = 10 / elapsed
            self.last_time = current_time
