import os
import time
import torch
import pandas as pd
from ultralytics import YOLO
from typing import Dict, Any, Optional


class Config:
    """Configuration class for YOLO training parameters"""
    
    def __init__(self):
        self.model_name = "yolov8n.pt"
        self.epochs = 50
        self.batch_size = 16
        self.img_size = 640
        self.yaml_path = "data/data.yaml"
        self.device = "0"  # "0" for GPU, "cpu" for CPU
        self.output_dir = "runs/train"
        self.resume = False
        self.onnx_opset = 12
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class YoloTrainer:
    """YOLO model trainer with validation, checkpointing, and export capabilities"""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.start_time = time.time()
        self.model = YOLO(self.cfg.model_name)
        self.metrics_log = []
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create necessary directories for training outputs"""
        os.makedirs(self.cfg.output_dir, exist_ok=True)
    
    def train_model(self) -> Any:
        """Execute training process with configured parameters"""
        print("[INFO] Starting training...")
        results = self.model.train(
            data=self.cfg.yaml_path,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.img_size,
            batch=self.cfg.batch_size,
            device=self.cfg.device,
            project=self.cfg.output_dir,
            resume=self.cfg.resume
        )
        return results
    
    def validate_model(self) -> Any:
        """Run model validation and log metrics"""
        print("[INFO] Running validation...")
        metrics = self.model.val()
        self.metrics_log.append(metrics.results_dict)
        return metrics
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint for specific epoch"""
        ckpt_path = os.path.join(self.cfg.output_dir, f"checkpoint_epoch{epoch}.pt")
        self.model.save(ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")
    
    def export_to_onnx(self, epoch: int, is_best: bool = False) -> None:
        """Export model to ONNX format"""
        name = f"best.onnx" if is_best else f"epoch{epoch}.onnx"
        export_path = os.path.join(self.cfg.output_dir, name)
        
        self.model.export(format="onnx", opset=self.cfg.onnx_opset, dynamic=True)
        
        # Handle ONNX file relocation
        weights_path = "weights/best.onnx"
        if os.path.exists(weights_path):
            os.rename(weights_path, export_path)
        
        print(f"[INFO] Exported ONNX model: {export_path}")
    
    def _is_best_precision(self, current_metrics: Dict[str, Any]) -> bool:
        """Check if current epoch has best precision score"""
        precision_key = "metrics/precision(B)"
        if precision_key not in current_metrics:
            return False
        
        current_precision = current_metrics[precision_key]
        best_precision = max(
            metrics.get(precision_key, 0) for metrics in self.metrics_log
        )
        return current_precision == best_precision
    
    def save_metrics_log(self) -> None:
        """Save training metrics to CSV file"""
        log_path = os.path.join(self.cfg.output_dir, "metrics_log.csv")
        pd.DataFrame(self.metrics_log).to_csv(log_path, index=False)
        print(f"[INFO] Metrics saved to {log_path}")
    
    def log_runtime(self) -> None:
        """Calculate and display total training runtime"""
        elapsed = time.time() - self.start_time
        print(f"[INFO] Total runtime: {elapsed/60:.2f} minutes")


class TrainingManager:
    """Main training orchestrator class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.trainer = YoloTrainer(config)
    
    def run_training_loop(self) -> None:
        """Execute complete training loop with validation and checkpointing"""
        for epoch in range(1, self.config.epochs + 1):
            self._train_epoch(epoch)
        
        # Save final results
        self._finalize_training()
    
    def _train_epoch(self, epoch: int) -> None:
        """Execute single epoch training and validation"""
        print(f"\n========== Epoch {epoch}/{self.config.epochs} ==========")
        
        # Train and validate
        self.trainer.train_model()
        metrics = self.trainer.validate_model()
        
        # Save checkpoint and export ONNX
        self.trainer.save_checkpoint(epoch)
        self.trainer.export_to_onnx(epoch)
        
        # Export best model if precision improved
        if self.trainer._is_best_precision(metrics.results_dict):
            self.trainer.export_to_onnx(epoch, is_best=True)
    
    def _finalize_training(self) -> None:
        """Save final training results and logs"""
        self.trainer.save_metrics_log()
        self.trainer.log_runtime()


def main():
    """Main entry point for YOLO training pipeline"""
    # Initialize configuration and start training
    config = Config()
    manager = TrainingManager(config)
    manager.run_training_loop()


if __name__ == "__main__":
    main()