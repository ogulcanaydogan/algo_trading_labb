"""
Webhook Alerts - Push notifications to external services.

Sends trading alerts, signals, and system events to webhooks
for integration with external tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from enum import Enum
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Types of webhook events."""

    TRADE_SIGNAL = "trade_signal"
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_ALERT = "risk_alert"
    DRAWDOWN_ALERT = "drawdown_alert"
    SYSTEM_ERROR = "system_error"
    ML_PREDICTION = "ml_prediction"
    PRICE_ALERT = "price_alert"
    CUSTOM = "custom"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""

    endpoint_id: str
    name: str
    url: str
    secret: Optional[str] = None  # For signature verification
    events: List[WebhookEventType] = field(default_factory=list)
    is_active: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 3
    timeout_seconds: int = 10

    def to_dict(self) -> Dict:
        return {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "url": self.url[:50] + "..." if len(self.url) > 50 else self.url,
            "events": [e.value for e in self.events],
            "is_active": self.is_active,
            "retry_count": self.retry_count,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    delivery_id: str
    endpoint_id: str
    event_type: WebhookEventType
    payload: Dict
    status: Literal["pending", "success", "failed"]
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "delivery_id": self.delivery_id,
            "endpoint_id": self.endpoint_id,
            "event_type": self.event_type.value,
            "status": self.status,
            "response_code": self.response_code,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "error": self.error,
        }


@dataclass
class WebhookConfig:
    """Webhook system configuration."""

    # Delivery settings
    default_retry_count: int = 3
    retry_delay_seconds: float = 1.0
    max_retry_delay_seconds: float = 60.0
    default_timeout_seconds: int = 10

    # Rate limiting
    max_webhooks_per_minute: int = 60

    # History
    max_delivery_history: int = 1000

    # Signature
    signature_header: str = "X-Webhook-Signature"
    timestamp_header: str = "X-Webhook-Timestamp"


class WebhookManager:
    """
    Manage webhook endpoints and deliveries.

    Features:
    - Multi-endpoint support
    - Event filtering by type
    - Automatic retries
    - Signature verification
    - Delivery tracking
    """

    def __init__(self, config: Optional[WebhookConfig] = None):
        self.config = config or WebhookConfig()
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._delivery_history: List[WebhookDelivery] = []
        self._delivery_count = 0
        self._rate_limit_tokens = self.config.max_webhooks_per_minute
        self._last_refill = time.time()

    def register_endpoint(self, endpoint: WebhookEndpoint):
        """Register a webhook endpoint."""
        self._endpoints[endpoint.endpoint_id] = endpoint
        logger.info(f"Registered webhook endpoint: {endpoint.name} ({endpoint.endpoint_id})")

    def add_endpoint(
        self,
        name: str,
        url: str,
        events: Optional[List[WebhookEventType]] = None,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> WebhookEndpoint:
        """Add a new webhook endpoint."""
        endpoint_id = f"wh_{hashlib.md5(url.encode()).hexdigest()[:8]}"

        endpoint = WebhookEndpoint(
            endpoint_id=endpoint_id,
            name=name,
            url=url,
            secret=secret,
            events=events or list(WebhookEventType),
            headers=headers or {},
        )

        self.register_endpoint(endpoint)
        return endpoint

    def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove a webhook endpoint."""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            return True
        return False

    def toggle_endpoint(self, endpoint_id: str, active: bool) -> bool:
        """Enable or disable an endpoint."""
        if endpoint_id in self._endpoints:
            self._endpoints[endpoint_id].is_active = active
            return True
        return False

    async def send(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        endpoint_ids: Optional[List[str]] = None,
    ) -> List[WebhookDelivery]:
        """
        Send webhook to all matching endpoints.

        Args:
            event_type: Type of event
            payload: Event payload
            endpoint_ids: Specific endpoints (None = all matching)

        Returns:
            List of delivery records
        """
        deliveries = []

        for endpoint_id, endpoint in self._endpoints.items():
            if not endpoint.is_active:
                continue

            if endpoint_ids and endpoint_id not in endpoint_ids:
                continue

            if event_type not in endpoint.events:
                continue

            delivery = await self._deliver(endpoint, event_type, payload)
            deliveries.append(delivery)

        return deliveries

    async def _deliver(
        self,
        endpoint: WebhookEndpoint,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
    ) -> WebhookDelivery:
        """Deliver webhook to a single endpoint."""
        self._delivery_count += 1
        delivery_id = f"del_{self._delivery_count}_{int(time.time())}"

        # Check rate limit
        self._refill_rate_limit()
        if self._rate_limit_tokens <= 0:
            return WebhookDelivery(
                delivery_id=delivery_id,
                endpoint_id=endpoint.endpoint_id,
                event_type=event_type,
                payload=payload,
                status="failed",
                error="Rate limit exceeded",
            )

        self._rate_limit_tokens -= 1

        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            endpoint_id=endpoint.endpoint_id,
            event_type=event_type,
            payload=payload,
            status="pending",
        )

        # Prepare payload
        full_payload = {
            "event_type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "delivery_id": delivery_id,
            "data": payload,
        }

        # Calculate signature
        timestamp = str(int(time.time()))
        signature = self._calculate_signature(endpoint.secret, timestamp, full_payload)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            self.config.timestamp_header: timestamp,
            **endpoint.headers,
        }

        if signature:
            headers[self.config.signature_header] = signature

        # Attempt delivery with retries
        delay = self.config.retry_delay_seconds

        for attempt in range(endpoint.retry_count):
            delivery.attempts = attempt + 1

            try:
                response_code, response_body = await self._http_post(
                    endpoint.url,
                    full_payload,
                    headers,
                    endpoint.timeout_seconds,
                )

                delivery.response_code = response_code
                delivery.response_body = response_body[:500] if response_body else None

                if 200 <= response_code < 300:
                    delivery.status = "success"
                    delivery.delivered_at = datetime.now()
                    logger.debug(f"Webhook delivered: {delivery_id} to {endpoint.name}")
                    break
                else:
                    delivery.error = f"HTTP {response_code}"

            except Exception as e:
                delivery.error = str(e)
                logger.warning(f"Webhook delivery failed: {e}")

            # Exponential backoff
            if attempt < endpoint.retry_count - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.max_retry_delay_seconds)

        if delivery.status == "pending":
            delivery.status = "failed"

        # Store delivery record
        self._delivery_history.append(delivery)
        if len(self._delivery_history) > self.config.max_delivery_history:
            self._delivery_history = self._delivery_history[-self.config.max_delivery_history :]

        return delivery

    async def _http_post(
        self,
        url: str,
        payload: Dict,
        headers: Dict[str, str],
        timeout: int,
    ) -> tuple:
        """Make HTTP POST request."""
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return response.status, body
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8") if e.fp else ""
        except urllib.error.URLError as e:
            raise Exception(f"URL error: {e.reason}")

    def _calculate_signature(
        self,
        secret: Optional[str],
        timestamp: str,
        payload: Dict,
    ) -> Optional[str]:
        """Calculate HMAC signature for payload."""
        if not secret:
            return None

        message = f"{timestamp}.{json.dumps(payload, sort_keys=True)}"
        signature = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return f"sha256={signature}"

    def _refill_rate_limit(self):
        """Refill rate limit tokens."""
        now = time.time()
        elapsed = now - self._last_refill

        if elapsed >= 60:
            self._rate_limit_tokens = self.config.max_webhooks_per_minute
            self._last_refill = now
        else:
            refill = int((elapsed / 60) * self.config.max_webhooks_per_minute)
            self._rate_limit_tokens = min(
                self.config.max_webhooks_per_minute,
                self._rate_limit_tokens + refill,
            )

    def send_sync(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
    ) -> List[WebhookDelivery]:
        """Synchronous wrapper for send."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.send(event_type, payload))
        finally:
            loop.close()

    # Convenience methods for common events

    def send_trade_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        reasoning: str,
    ) -> List[WebhookDelivery]:
        """Send trade signal webhook."""
        return self.send_sync(
            WebhookEventType.TRADE_SIGNAL,
            {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "price": price,
                "reasoning": reasoning,
            },
        )

    def send_trade_executed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
    ) -> List[WebhookDelivery]:
        """Send trade executed webhook."""
        return self.send_sync(
            WebhookEventType.TRADE_EXECUTED,
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
            },
        )

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        details: Optional[Dict] = None,
    ) -> List[WebhookDelivery]:
        """Send risk alert webhook."""
        return self.send_sync(
            WebhookEventType.RISK_ALERT,
            {
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "details": details or {},
            },
        )

    def send_price_alert(
        self,
        symbol: str,
        current_price: float,
        alert_price: float,
        direction: str,
    ) -> List[WebhookDelivery]:
        """Send price alert webhook."""
        return self.send_sync(
            WebhookEventType.PRICE_ALERT,
            {
                "symbol": symbol,
                "current_price": current_price,
                "alert_price": alert_price,
                "direction": direction,
            },
        )

    def get_endpoints(self) -> List[Dict]:
        """Get all registered endpoints."""
        return [e.to_dict() for e in self._endpoints.values()]

    def get_delivery_history(
        self,
        endpoint_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get delivery history."""
        history = self._delivery_history

        if endpoint_id:
            history = [d for d in history if d.endpoint_id == endpoint_id]

        if status:
            history = [d for d in history if d.status == status]

        return [d.to_dict() for d in history[-limit:]]

    def get_stats(self) -> Dict:
        """Get webhook statistics."""
        total = len(self._delivery_history)
        success = sum(1 for d in self._delivery_history if d.status == "success")
        failed = sum(1 for d in self._delivery_history if d.status == "failed")

        return {
            "endpoints_registered": len(self._endpoints),
            "endpoints_active": sum(1 for e in self._endpoints.values() if e.is_active),
            "total_deliveries": total,
            "successful_deliveries": success,
            "failed_deliveries": failed,
            "success_rate": success / total if total > 0 else 0,
            "rate_limit_remaining": self._rate_limit_tokens,
        }


def create_webhook_manager(config: Optional[WebhookConfig] = None) -> WebhookManager:
    """Factory function to create webhook manager."""
    return WebhookManager(config=config)
