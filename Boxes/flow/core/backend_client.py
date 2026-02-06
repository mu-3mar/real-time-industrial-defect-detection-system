import requests
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class BackendClient:
    """HTTP client for sending detection results to backend."""
    
    def __init__(self, base_url: str, result_endpoint: str, timeout: int = 5, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.result_endpoint = result_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.full_url = f"{self.base_url}{self.result_endpoint}"
    
    def send_result(self, report_id: str, is_defect: bool) -> bool:
        """
        Send detection result to backend.
        
        Args:
            report_id: Unique identifier for the production line/session
            is_defect: True if defect detected, False otherwise
        
        Returns:
            True if successfully sent, False otherwise
        """
        payload = {
            "report_id": report_id,
            "value": is_defect
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.full_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"Result sent successfully for {report_id}: defect={is_defect}")
                    return True
                else:
                    logger.warning(f"Backend returned status {response.status_code} for {report_id}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout sending result for {report_id} (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error sending result for {report_id} (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Unexpected error sending result for {report_id}: {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        logger.error(f"Failed to send result for {report_id} after {self.max_retries} attempts")
        return False
