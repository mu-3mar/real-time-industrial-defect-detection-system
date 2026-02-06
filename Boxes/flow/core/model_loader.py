import logging
from ultralytics import YOLO
from typing import Optional

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton class for loading and sharing YOLO models across sessions."""
    
    _instance: Optional['ModelLoader'] = None
    
    def __init__(self):
        self.box_model: Optional[YOLO] = None
        self.defect_model: Optional[YOLO] = None
        self._loaded = False
    
    @classmethod
    def get_instance(cls) -> 'ModelLoader':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_models(self, box_model_path: str, defect_model_path: str):
        """
        Load both detection models.
        
        Args:
            box_model_path: Path to box detection model
            defect_model_path: Path to defect detection model
        """
        if self._loaded:
            logger.info("Models already loaded, skipping...")
            return
        
        logger.info(f"Loading box detection model from {box_model_path}")
        self.box_model = YOLO(box_model_path, task='detect')
        
        logger.info(f"Loading defect detection model from {defect_model_path}")
        self.defect_model = YOLO(defect_model_path, task='detect')
        
        self._loaded = True
        logger.info("All models loaded successfully")
    
    def get_box_model(self) -> YOLO:
        """Get box detection model."""
        if not self._loaded or self.box_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.box_model
    
    def get_defect_model(self) -> YOLO:
        """Get defect detection model."""
        if not self._loaded or self.defect_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.defect_model
