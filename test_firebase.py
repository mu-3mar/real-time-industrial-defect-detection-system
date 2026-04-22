import sys
import logging
logging.basicConfig(level=logging.INFO)

from Boxes.flow.core.firebase_client import initialize, publish_session_info

if not initialize("Boxes/flow/config/firebase-service-account.json", "https://qc-scm-default-rtdb.firebaseio.com"):
    print("Init failed")
    sys.exit(1)

publish_session_info("test_report_123", {"test": "data"})
