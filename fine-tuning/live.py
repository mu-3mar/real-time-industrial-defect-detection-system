import cv2
from ultralytics import YOLO
import time
import numpy as np

def run_realtime_quantized_models(camera_index=0):
    """
    Run both quantized models in real-time camera feed
    """
    
    # Load both quantized models
    models = {
        "Segmentation_INT8": YOLO("/home/muhammad-ammar/AI/SCM_8s/fine-tuning-seg/output/best_int8.onnx", task="segment"),
        "Detection_INT8": YOLO("/home/muhammad-ammar/AI/SCM_8s/fine-tuning/output/best_int8.onnx", task="detect")
    }
    
    # Different thresholds for each model
    thresholds = {
        "Segmentation_INT8": {0: 0.4, 1: 0.9},
        "Detection_INT8": {0: 0.95, 1: 0.95}
    }
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    print("🚀 Starting real-time inference with quantized models...")
    print("Press 'q' to quit, 's' to switch models")
    
    current_model = "Segmentation_INT8"  # Start with segmentation
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        # Start timing
        start_time = time.time()
        
        # Run inference with current model
        model = models[current_model]
        results = model(frame, conf=0.1, verbose=False)
        
        inference_time = time.time() - start_time
        
        # Process results
        result_frame = frame.copy()
        detections_count = 0
        
        if results and results[0].boxes is not None:
            current_thresholds = thresholds[current_model]
            
            for i, box in enumerate(results[0].boxes):
                product_type = int(box.cls)
                confidence = float(box.conf)
                
                # Apply model-specific threshold
                if confidence >= current_thresholds[product_type]:
                    detections_count += 1
                    
                    # Colors and labels
                    color = (0, 255, 0) if product_type == 0 else (0, 0, 255)
                    label = f"{'box' if product_type == 0 else 'defected'} {confidence:.1%}"
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw mask for segmentation
                    if current_model == "Segmentation_INT8" and results[0].masks is not None and i < len(results[0].masks):
                        mask = results[0].masks[i]
                        contours = mask.xy[0].astype(int)
                        mask_overlay = result_frame.copy()
                        cv2.fillPoly(mask_overlay, [contours], color)
                        cv2.addWeighted(mask_overlay, 0.3, result_frame, 0.7, 0, result_frame)
                    
                    # Add text
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result_frame, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
                    cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Add model info and performance metrics
        info_text = [
            f"Model: {current_model}",
            f"FPS: {1/inference_time:.1f}",
            f"Detections: {detections_count}",
            f"Time: {inference_time*1000:.1f}ms",
            "Press 's' to switch models"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display the frame
        cv2.imshow('Quantized Models - Real Time', result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Switch between models
            current_model = "Detection_INT8" if current_model == "Segmentation_INT8" else "Segmentation_INT8"
            print(f"🔄 Switched to: {current_model}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Real-time inference stopped")

def run_side_by_side_models(camera_index=0):
    """
    Run both models side by side for comparison
    """
    
    # Load both quantized models
    models = {
        "Segmentation": YOLO("fine-tuning/fine-tuning-seg/output/best_int8.onnx", task="segment"),
        "Detection": YOLO("fine-tuning/fine-tuning-YOlO-old/runs/train/train/weights/best_int8.onnx", task="detect")
    }
    
    thresholds = {
        "Segmentation": {0: 0.6, 1: 0.9},
        "Detection": {0: 0.8, 1: 0.8}
    }
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    print("🚀 Starting side-by-side comparison...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        # Create output frame (side by side)
        height, width = frame.shape[:2]
        combined_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        results_data = {}
        
        # Process both models
        for model_name, model in models.items():
            start_time = time.time()
            results = model(frame, conf=0.1, verbose=False)
            inference_time = time.time() - start_time
            
            # Create result frame
            result_frame = frame.copy()
            detections_count = 0
            
            if results and results[0].boxes is not None:
                current_thresholds = thresholds[model_name]
                
                for i, box in enumerate(results[0].boxes):
                    product_type = int(box.cls)
                    confidence = float(box.conf)
                    
                    if confidence >= current_thresholds[product_type]:
                        detections_count += 1
                        
                        color = (0, 255, 0) if product_type == 0 else (0, 0, 255)
                        label = f"{'box' if product_type == 0 else 'defected'} {confidence:.1%}"
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw mask for segmentation
                        if model_name == "Segmentation" and results[0].masks is not None and i < len(results[0].masks):
                            mask = results[0].masks[i]
                            contours = mask.xy[0].astype(int)
                            mask_overlay = result_frame.copy()
                            cv2.fillPoly(mask_overlay, [contours], color)
                            cv2.addWeighted(mask_overlay, 0.3, result_frame, 0.7, 0, result_frame)
                        
                        # Add text
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(result_frame, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
                        cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Add model info
            info_text = [
                f"Model: {model_name}",
                f"FPS: {1/inference_time:.1f}",
                f"Detections: {detections_count}",
                f"Time: {inference_time*1000:.1f}ms"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(result_frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(result_frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Store results
            results_data[model_name] = {
                'frame': result_frame,
                'inference_time': inference_time,
                'detections_count': detections_count
            }
        
        # Arrange side by side
        combined_frame[:, :width] = results_data["Segmentation"]['frame']
        combined_frame[:, width:width*2] = results_data["Detection"]['frame']
        
        # Add separator line
        cv2.line(combined_frame, (width, 0), (width, height), (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Side by Side: Segmentation (Left) vs Detection (Right)', combined_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Side-by-side comparison stopped")

if __name__ == "__main__":
    print("🎥 Choose camera mode:")
    print("1. Single model (switch with 's')")
    print("2. Side by side comparison")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_realtime_quantized_models(camera_index=0)
    elif choice == "2":
        run_side_by_side_models(camera_index=0)
    else:
        print("❌ Invalid choice")