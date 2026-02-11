"""MQTT client for publishing detection events to HiveMQ Cloud."""

import json
import logging
import ssl
import threading
import time
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

        # Configure TLS
        self.client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        # Connection state
        self._connected = False
        self._connecting = False
        self._connection_lock = threading.Lock()

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

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker."""
        if rc == 0:
            self._connected = True
            logger.info("MQTT client connected successfully to %s:%d", self.host, self.port)
        else:
            self._connected = False
            logger.error("MQTT connection failed with code %d", rc)

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

    def connect(self) -> bool:
        """
        Connect to MQTT broker (non-blocking).

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
                logger.info("Connecting to MQTT broker %s:%d...", self.host, self.port)
                self.client.connect_async(self.host, self.port, self.keepalive)
                self.client.loop_start()

                # Wait briefly for connection (non-blocking)
                timeout = 5
                start_time = time.time()
                while not self._connected and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                if self._connected:
                    logger.info("MQTT connection established")
                    return True
                else:
                    logger.warning("MQTT connection pending (async)")
                    return True  # Connection in progress

            except Exception as e:
                logger.error("Failed to connect to MQTT broker: %s", e)
                self._connecting = False
                return False
            finally:
                self._connecting = False

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        try:
            logger.info("Disconnecting MQTT client...")
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
        Publish detection insight to MQTT (non-blocking).

        Args:
            production_line: Production line identifier
            report_id: Report/session identifier
            defect: True if defect detected, False otherwise

        Returns:
            True if publish initiated, False otherwise
        """
        if not self._connected:
            logger.warning("MQTT client not connected, cannot publish insight")
            return False

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

            # Publish (QoS 1 for at-least-once delivery)
            result = self.client.publish(
                topic=topic,
                payload=json.dumps(payload),
                qos=1,
                retain=False,
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(
                    "Published insight to %s: defect=%s (mid=%d)",
                    topic,
                    defect,
                    result.mid,
                )
                return True
            else:
                logger.error("Failed to publish insight (rc=%d)", result.rc)
                return False

        except Exception as e:
            logger.error("Error publishing insight: %s", e)
            return False

    def is_connected(self) -> bool:
        """Check if client is connected to broker."""
        return self._connected
