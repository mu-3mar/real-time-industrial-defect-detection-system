# pipeline/api_client.py
import requests
from datetime import datetime
from typing import Dict
from .config import cfg

def send_product_to_api(product_id: str, info: Dict, final_status: str) -> bool:
    """
    Send product info to the external API.
    Fields sent:
        product_id, session_id, status, max_defects, timestamp, productionline_id, companyId
    """
    payload = {
        "product_id": product_id,
        "session_id": cfg.SESSION_ID,
        "status": final_status,
        "max_defects": int(info.get("max_defects", 0)),
        "timestamp": datetime.now().isoformat(),
        "productionline_id": cfg.PRODUCTION_LINE_ID,
        "companyId": cfg.COMPANY_ID,
    }

    try:
        resp = requests.post(cfg.API_URL, json=payload, timeout=5)
        if resp.status_code in (200, 201):
            print(f"[API] OK {product_id} -> {final_status}")
            return True
        else:
            print(f"[API] Error {resp.status_code} for {product_id}: {resp.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[API] Connection error for {product_id}: {e}")
        return False
