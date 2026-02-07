"""HTTP client for sending detection results to backend."""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class BackendClient:
    """HTTP client for sending detection results to backend."""

    def __init__(
        self,
        base_url: str,
        result_endpoint: str,
        timeout: int = 5,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.result_endpoint = result_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.full_url = f"{self.base_url}{self.result_endpoint}"

    def send_result(self, report_id: str, is_defect: bool) -> bool:
        """
        Send detection result to backend with exponential backoff retry.

        Args:
            report_id: Unique identifier for the session/report
            is_defect: True if defect detected, False otherwise

        Returns:
            True if successfully sent, False otherwise
        """
        payload = {"report_id": report_id, "value": is_defect}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.full_url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    logger.info(
                        "Result sent for %s: defect=%s",
                        report_id,
                        is_defect,
                    )
                    return True

                logger.warning(
                    "Backend returned %d for %s",
                    response.status_code,
                    report_id,
                )

            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout sending result for %s (attempt %d/%d)",
                    report_id,
                    attempt + 1,
                    self.max_retries,
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Connection error for %s (attempt %d/%d)",
                    report_id,
                    attempt + 1,
                    self.max_retries,
                )
            except Exception as e:
                logger.error("Unexpected error sending result for %s: %s", report_id, e)

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logger.error(
            "Failed to send result for %s after %d attempts",
            report_id,
            self.max_retries,
        )
        return False
