/**
 * WebSocket utilities for the trading dashboard.
 *
 * Provides reconnection logic, message handling, and connection state management.
 */

/**
 * WebSocket connection manager with automatic reconnection.
 */
class WebSocketManager {
    constructor(options = {}) {
        this.url = options.url || null;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        this.reconnectDelay = options.reconnectDelay || 1000;
        this.maxReconnectDelay = options.maxReconnectDelay || 30000;

        this.ws = null;
        this.reconnectAttempts = 0;
        this.reconnectTimeout = null;
        this.isConnecting = false;
        this.isIntentionallyClosed = false;

        // Callbacks
        this.onOpen = options.onOpen || (() => {});
        this.onMessage = options.onMessage || (() => {});
        this.onClose = options.onClose || (() => {});
        this.onError = options.onError || (() => {});
        this.onReconnecting = options.onReconnecting || (() => {});
    }

    /**
     * Connect to the WebSocket server.
     *
     * @param {string} url - Optional URL override
     */
    connect(url = null) {
        if (url) this.url = url;
        if (!this.url) {
            console.error('[WebSocketManager] No URL provided');
            return;
        }

        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.isConnecting = true;
        this.isIntentionallyClosed = false;

        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = (event) => {
                this.isConnecting = false;
                this.reconnectAttempts = 0;
                console.log('[WebSocketManager] Connected');
                this.onOpen(event);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.onMessage(data, event);
                } catch (e) {
                    console.warn('[WebSocketManager] Failed to parse message:', e);
                    this.onMessage(event.data, event);
                }
            };

            this.ws.onclose = (event) => {
                this.isConnecting = false;
                console.log('[WebSocketManager] Closed:', event.code, event.reason);
                this.onClose(event);

                if (!this.isIntentionallyClosed) {
                    this.scheduleReconnect();
                }
            };

            this.ws.onerror = (event) => {
                this.isConnecting = false;
                console.error('[WebSocketManager] Error:', event);
                this.onError(event);
            };

        } catch (e) {
            this.isConnecting = false;
            console.error('[WebSocketManager] Connection error:', e);
            this.scheduleReconnect();
        }
    }

    /**
     * Schedule a reconnection attempt with exponential backoff.
     */
    scheduleReconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('[WebSocketManager] Max reconnection attempts reached');
            return;
        }

        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );

        this.reconnectAttempts++;
        console.log(`[WebSocketManager] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        this.onReconnecting(this.reconnectAttempts, delay);

        this.reconnectTimeout = setTimeout(() => {
            this.connect();
        }, delay);
    }

    /**
     * Send a message through the WebSocket.
     *
     * @param {Object|string} data - Data to send
     * @returns {boolean} Whether the message was sent
     */
    send(data) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('[WebSocketManager] Cannot send - not connected');
            return false;
        }

        try {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.ws.send(message);
            return true;
        } catch (e) {
            console.error('[WebSocketManager] Send error:', e);
            return false;
        }
    }

    /**
     * Close the WebSocket connection.
     */
    close() {
        this.isIntentionallyClosed = true;

        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Get the current connection state.
     *
     * @returns {string} 'connecting', 'open', 'closing', 'closed', or 'unknown'
     */
    get state() {
        if (!this.ws) return 'closed';

        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'connecting';
            case WebSocket.OPEN: return 'open';
            case WebSocket.CLOSING: return 'closing';
            case WebSocket.CLOSED: return 'closed';
            default: return 'unknown';
        }
    }

    /**
     * Check if the WebSocket is connected.
     *
     * @returns {boolean}
     */
    get isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}

/**
 * Message buffer for batching updates.
 */
class MessageBuffer {
    constructor(options = {}) {
        this.bufferSize = options.bufferSize || 100;
        this.flushInterval = options.flushInterval || 1000;
        this.onFlush = options.onFlush || (() => {});

        this.buffer = [];
        this.flushTimeout = null;
    }

    /**
     * Add a message to the buffer.
     *
     * @param {any} message - Message to buffer
     */
    add(message) {
        this.buffer.push({
            message,
            timestamp: Date.now(),
        });

        // Trim buffer if too large
        if (this.buffer.length > this.bufferSize) {
            this.buffer = this.buffer.slice(-this.bufferSize);
        }

        // Schedule flush if not already scheduled
        if (!this.flushTimeout) {
            this.flushTimeout = setTimeout(() => {
                this.flush();
            }, this.flushInterval);
        }
    }

    /**
     * Flush the buffer.
     */
    flush() {
        if (this.flushTimeout) {
            clearTimeout(this.flushTimeout);
            this.flushTimeout = null;
        }

        if (this.buffer.length > 0) {
            const messages = [...this.buffer];
            this.buffer = [];
            this.onFlush(messages);
        }
    }

    /**
     * Clear the buffer without flushing.
     */
    clear() {
        if (this.flushTimeout) {
            clearTimeout(this.flushTimeout);
            this.flushTimeout = null;
        }
        this.buffer = [];
    }

    /**
     * Get current buffer size.
     *
     * @returns {number}
     */
    get size() {
        return this.buffer.length;
    }
}

/**
 * Rate limiter for outgoing messages.
 */
class RateLimiter {
    constructor(options = {}) {
        this.maxRequests = options.maxRequests || 10;
        this.windowMs = options.windowMs || 1000;

        this.requests = [];
    }

    /**
     * Check if a request can be made.
     *
     * @returns {boolean}
     */
    canRequest() {
        const now = Date.now();
        const windowStart = now - this.windowMs;

        // Remove old requests
        this.requests = this.requests.filter(t => t > windowStart);

        return this.requests.length < this.maxRequests;
    }

    /**
     * Record a request.
     *
     * @returns {boolean} Whether the request was allowed
     */
    request() {
        if (!this.canRequest()) {
            return false;
        }

        this.requests.push(Date.now());
        return true;
    }

    /**
     * Get remaining requests in current window.
     *
     * @returns {number}
     */
    get remaining() {
        const now = Date.now();
        const windowStart = now - this.windowMs;

        this.requests = this.requests.filter(t => t > windowStart);
        return Math.max(0, this.maxRequests - this.requests.length);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WebSocketManager,
        MessageBuffer,
        RateLimiter,
    };
}
