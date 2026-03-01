"""MQTT client for publishing detection events to HiveMQ Cloud."""

import json
import logging
import ssl
import threading
import time
import queue
from datetime import datetime
from typing import Optional
import uuid

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MqttClient:
    """Singleton MQTT client for publishing detection insights."""

    _instance: Optional["MqttClient"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        client_id_prefix: str = "qc-scm-edge",
        keepalive: int = 60,
        clean_session: bool = True,
        topic_pattern: str = "factory/{production_line}/{report_id}/insights",
    ):
        """
        Initialize MQTT client.

        Args:
            host: MQTT broker hostname
            port: MQTT broker port (8883 for TLS)
            username: MQTT username
            password: MQTT password
            client_id_prefix: Prefix for client ID (unique ID appended)
            keepalive: Keepalive interval in seconds
            clean_session: Clean session flag
            topic_pattern: Topic pattern with placeholders
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.topic_pattern = topic_pattern
        self.keepalive = keepalive

        # Generate unique client ID
        client_id = f"{client_id_prefix}-{uuid.uuid4().hex[:8]}"

        # Create MQTT client
        self.client = mqtt.Client(
            client_id=client_id,
            clean_session=clean_session,
            protocol=mqtt.MQTTv311,
        )

        # Set credentials
        self.client.username_pw_set(username, password)

        # Configure TLS — cert verification temporarily disabled to diagnose connection hang.
        # TODO: restore cert_reqs=ssl.CERT_REQUIRED once root cause is confirmed.
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)
        logger.debug("[MQTT] TLS configured: cert_reqs=CERT_NONE (insecure, debug mode)")

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_log = self._on_log  # paho internal logs (TLS/TCP errors)

        # Connection state
        self._connected = False
        self._connecting = False
        self._connection_lock = threading.Lock()
        
        # ── Task 5: Non-blocking Publish Queue ──
        self._publish_queue: queue.Queue = queue.Queue()
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @classmethod
    def get_instance(cls) -> Optional["MqttClient"]:
        """Get singleton instance (thread-safe)."""
        return cls._instance

    @classmethod
    def initialize(
        cls,
        host: str,
        port: int,
        username: str,
        password: str,
        client_id_prefix: str = "qc-scm-edge",
        keepalive: int = 60,
        clean_session: bool = True,
        topic_pattern: str = "factory/{production_line}/{report_id}/insights",
    ) -> "MqttClient":
        """
        Initialize singleton instance (thread-safe).

        Args:
            host: MQTT broker hostname
            port: MQTT broker port
            username: MQTT username
            password: MQTT password
            client_id_prefix: Prefix for client ID
            keepalive: Keepalive interval
            clean_session: Clean session flag
            topic_pattern: Topic pattern with placeholders

        Returns:
            MqttClient instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        host=host,
                        port=port,
                        username=username,
                        password=password,
                        client_id_prefix=client_id_prefix,
                        keepalive=keepalive,
                        clean_session=clean_session,
                        topic_pattern=topic_pattern,
                    )
        return cls._instance

    # Human-readable MQTT return codes for debugging
    _RC_MESSAGES = {
        0: "Connection accepted",
        1: "Refused – incorrect protocol version",
        2: "Refused – invalid client identifier",
        3: "Refused – server unavailable",
        4: "Refused – bad username or password",
        5: "Refused – not authorized",
    }

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker."""
        rc_msg = self._RC_MESSAGES.get(rc, f"Unknown error code {rc}")
        if rc == 0:
            self._connected = True
            logger.info(
                "[MQTT] on_connect fired: rc=%d (%s) — connected to %s:%d",
                rc, rc_msg, self.host, self.port,
            )
        else:
            self._connected = False
            logger.error(
                "[MQTT] on_connect fired: rc=%d (%s) — connection REFUSED",
                rc, rc_msg,
            )

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker."""
        self._connected = False
        if rc != 0:
            logger.warning("MQTT client disconnected unexpectedly (code %d), will auto-reconnect", rc)
        else:
            logger.info("MQTT client disconnected gracefully")

    def _on_publish(self, client, userdata, mid):
        """Callback when message is published."""
        logger.debug("MQTT message published (mid=%d)", mid)

    def _on_log(self, client, userdata, level, buf):
        """Paho internal log callback — captures TLS/TCP errors before on_connect fires."""
        # Map paho log levels to Python logging levels
        if level == mqtt.MQTT_LOG_ERR:
            logger.error("[MQTT-paho] %s", buf)
        elif level == mqtt.MQTT_LOG_WARNING:
            logger.warning("[MQTT-paho] %s", buf)
        elif level == mqtt.MQTT_LOG_NOTICE or level == mqtt.MQTT_LOG_INFO:
            logger.info("[MQTT-paho] %s", buf)
        else:
            logger.debug("[MQTT-paho] %s", buf)

    def _sender_loop(self):
        """
        Background thread that drains the publish queue and sends to MQTT.
        Ensures the pipeline is never blocked by network latency.
        """
        logger.info("[MQTT] Sender thread started")
        while not self._stop_event.is_set():
            try:
                # Wait for an item, but wake up periodically to check stop_event
                item = self._publish_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                topic, payload_str = item
                if self._connected:
                    result = self.client.publish(
                        topic=topic,
                        payload=payload_str,
                        qos=1,
                        retain=False,
                    )
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        logger.info("Published insight to %s (queued msg, mid=%d)", topic, result.mid)
                    else:
                        logger.error("Failed to publish insight (rc=%d)", result.rc)
                else:
                    logger.warning("[MQTT] Dropped insight; client not connected. Topic: %s", topic)
            except Exception as e:
                logger.error("[MQTT] Error in sender loop: %s", e)
            finally:
                self._publish_queue.task_done()
        logger.info("[MQTT] Sender thread stopped")

    def connect(self) -> bool:
        """
        Connect to MQTT broker (non-blocking) and start background sender thread.

        Returns:
            True if connection initiated, False otherwise
        """
        with self._connection_lock:
            if self._connected:
                logger.info("MQTT client already connected")
                return True

            if self._connecting:
                logger.info("MQTT connection already in progress")
                return True

            try:
                self._connecting = True
                logger.info(
                    "[MQTT] Connecting to broker: host=%s port=%d username=%s keepalive=%ds",
                    self.host, self.port, self.username, self.keepalive,
                )
                self.client.connect_async(self.host, self.port, self.keepalive)
                logger.debug("[MQTT] connect_async() called — starting network loop")
                self.client.loop_start()

                # Start the background sender thread for the queue
                self._stop_event.clear()
                if self._sender_thread is None or not self._sender_thread.is_alive():
                    self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
                    self._sender_thread.start()

                # HiveMQ Cloud TLS handshake can take several seconds.
                # Wait up to 15s for on_connect to fire before declaring "pending".
                timeout = 15
                start_time = time.time()
                while not self._connected and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                elapsed = time.time() - start_time
                if self._connected:
                    logger.info("[MQTT] Connection established in %.1fs", elapsed)
                    return True
                else:
                    logger.warning(
                        "[MQTT] Connection pending after %.1fs — "
                        "on_connect has not fired yet. "
                        "loop_start() is running; connection will complete in background.",
                        elapsed,
                    )
                    return True  # loop_start() keeps retrying in background

            except Exception as e:
                logger.error(
                    "[MQTT] Failed to initiate connection to %s:%d — %s: %s",
                    self.host, self.port, type(e).__name__, e,
                )
                self._connecting = False
                return False
            finally:
                self._connecting = False

    def disconnect(self) -> None:
        """Disconnect from MQTT broker and stop sender thread."""
        try:
            logger.info("Disconnecting MQTT client...")
            self._stop_event.set()
            if self._sender_thread is not None:
                self._sender_thread.join(timeout=2.0)
            
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            logger.info("MQTT client disconnected")
        except Exception as e:
            logger.error("Error disconnecting MQTT client: %s", e)

    def publish_insight(
        self,
        production_line: str,
        report_id: str,
        defect: bool,
    ) -> bool:
        """
        Enqueue detection insight for MQTT publishing (non-blocking).

        Args:
            production_line: Production line identifier
            report_id: Report/session identifier
            defect: True if defect detected, False otherwise

        Returns:
            True if enqueue succeeded, False otherwise
        """
        # We don't check self._connected here to allow queueing while reconnecting
        try:
            # Build topic
            topic = self.topic_pattern.format(
                production_line=production_line,
                report_id=report_id,
            )

            # Build payload
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "production_line": production_line,
                "report_id": report_id,
                "defect": defect,
            }
            
            payload_str = json.dumps(payload)
            
            # Enqueue to background thread
            # Fast, non-blocking call
            try:
                self._publish_queue.put_nowait((topic, payload_str))
                logger.debug("Enqueued insight to %s: defect=%s", topic, defect)
                return True
            except queue.Full:
                logger.error("MQTT publish queue full, dropping insight")
                return False

        except Exception as e:
            logger.error("Error queueing insight: %s", e)
            return False
