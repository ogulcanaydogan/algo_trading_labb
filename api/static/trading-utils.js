/**
 * Trading utilities for the dashboard.
 *
 * Common functions for trading calculations, formatting, and state management.
 */

/**
 * Calculate position P&L.
 *
 * @param {string} side - 'long' or 'short'
 * @param {number} entryPrice - Entry price
 * @param {number} currentPrice - Current market price
 * @param {number} quantity - Position size
 * @returns {Object} {pnl, pnlPercent}
 */
function calculatePositionPnL(side, entryPrice, currentPrice, quantity) {
    let pnl;
    if (side === 'long') {
        pnl = (currentPrice - entryPrice) * quantity;
    } else {
        pnl = (entryPrice - currentPrice) * quantity;
    }
    const pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100 * (side === 'long' ? 1 : -1);

    return { pnl, pnlPercent };
}

/**
 * Calculate risk metrics for a position.
 *
 * @param {Object} position - Position object
 * @param {number} accountBalance - Total account balance
 * @returns {Object} Risk metrics
 */
function calculatePositionRisk(position, accountBalance) {
    const positionValue = position.quantity * position.currentPrice;
    const exposure = (positionValue / accountBalance) * 100;

    const distanceToStop = position.stopLoss
        ? Math.abs((position.currentPrice - position.stopLoss) / position.currentPrice) * 100
        : null;

    const distanceToTakeProfit = position.takeProfit
        ? Math.abs((position.takeProfit - position.currentPrice) / position.currentPrice) * 100
        : null;

    const riskRewardRatio = (distanceToStop && distanceToTakeProfit)
        ? distanceToTakeProfit / distanceToStop
        : null;

    return {
        positionValue,
        exposure,
        distanceToStop,
        distanceToTakeProfit,
        riskRewardRatio,
    };
}

/**
 * Calculate portfolio metrics from positions.
 *
 * @param {Array} positions - Array of position objects
 * @param {number} balance - Cash balance
 * @returns {Object} Portfolio metrics
 */
function calculatePortfolioMetrics(positions, balance) {
    const totalPositionValue = positions.reduce((sum, p) => {
        return sum + (p.quantity * (p.currentPrice || p.entryPrice));
    }, 0);

    const totalUnrealizedPnL = positions.reduce((sum, p) => {
        return sum + (p.unrealizedPnl || 0);
    }, 0);

    const totalEquity = balance + totalUnrealizedPnL;

    const longExposure = positions
        .filter(p => p.side === 'long')
        .reduce((sum, p) => sum + p.quantity * (p.currentPrice || p.entryPrice), 0);

    const shortExposure = positions
        .filter(p => p.side === 'short')
        .reduce((sum, p) => sum + p.quantity * (p.currentPrice || p.entryPrice), 0);

    return {
        totalPositionValue,
        totalUnrealizedPnL,
        totalEquity,
        longExposure,
        shortExposure,
        netExposure: longExposure - shortExposure,
        positionCount: positions.length,
    };
}

/**
 * Format a timestamp for display.
 *
 * @param {string|Date} timestamp - Timestamp to format
 * @param {string} format - 'time', 'date', 'datetime', 'relative'
 * @returns {string} Formatted timestamp
 */
function formatTimestamp(timestamp, format = 'datetime') {
    const date = new Date(timestamp);
    const now = new Date();

    switch (format) {
        case 'time':
            return date.toLocaleTimeString(undefined, {
                hour: '2-digit',
                minute: '2-digit'
            });

        case 'date':
            return date.toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric'
            });

        case 'datetime':
            return date.toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });

        case 'relative':
            const diffMs = now - date;
            const diffSec = Math.floor(diffMs / 1000);
            const diffMin = Math.floor(diffSec / 60);
            const diffHour = Math.floor(diffMin / 60);
            const diffDay = Math.floor(diffHour / 24);

            if (diffSec < 60) return 'Just now';
            if (diffMin < 60) return `${diffMin}m ago`;
            if (diffHour < 24) return `${diffHour}h ago`;
            if (diffDay < 7) return `${diffDay}d ago`;
            return date.toLocaleDateString();

        default:
            return date.toISOString();
    }
}

/**
 * Get signal badge HTML based on signal type.
 *
 * @param {string} signal - 'LONG', 'SHORT', 'FLAT'
 * @param {number} confidence - Confidence level (0-1)
 * @returns {string} HTML string for badge
 */
function getSignalBadge(signal, confidence = null) {
    const colors = {
        'LONG': { bg: 'rgba(34, 197, 94, 0.2)', text: '#22c55e', border: '#22c55e' },
        'SHORT': { bg: 'rgba(239, 68, 68, 0.2)', text: '#ef4444', border: '#ef4444' },
        'FLAT': { bg: 'rgba(148, 163, 184, 0.2)', text: '#94a3b8', border: '#94a3b8' },
        'BUY': { bg: 'rgba(34, 197, 94, 0.2)', text: '#22c55e', border: '#22c55e' },
        'SELL': { bg: 'rgba(239, 68, 68, 0.2)', text: '#ef4444', border: '#ef4444' },
    };

    const style = colors[signal?.toUpperCase()] || colors['FLAT'];
    const confText = confidence !== null ? ` (${(confidence * 100).toFixed(0)}%)` : '';

    return `<span style="
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background: ${style.bg};
        color: ${style.text};
        border: 1px solid ${style.border};
    ">${signal}${confText}</span>`;
}

/**
 * Get regime badge HTML.
 *
 * @param {string} regime - Market regime
 * @returns {string} HTML string for badge
 */
function getRegimeBadge(regime) {
    const colors = {
        'trending': { bg: 'rgba(59, 130, 246, 0.2)', text: '#3b82f6' },
        'mean_reverting': { bg: 'rgba(139, 92, 246, 0.2)', text: '#8b5cf6' },
        'volatile': { bg: 'rgba(245, 158, 11, 0.2)', text: '#f59e0b' },
        'calm': { bg: 'rgba(34, 197, 94, 0.2)', text: '#22c55e' },
        'unknown': { bg: 'rgba(148, 163, 184, 0.2)', text: '#94a3b8' },
    };

    const style = colors[regime?.toLowerCase()] || colors['unknown'];
    const displayName = regime?.replace('_', ' ').toUpperCase() || 'UNKNOWN';

    return `<span style="
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        background: ${style.bg};
        color: ${style.text};
    ">${displayName}</span>`;
}

/**
 * Calculate win rate from trades.
 *
 * @param {Array} trades - Array of trade objects with 'pnl' property
 * @returns {Object} {winRate, totalWins, totalLosses}
 */
function calculateWinRate(trades) {
    if (!trades || trades.length === 0) {
        return { winRate: 0, totalWins: 0, totalLosses: 0 };
    }

    const totalWins = trades.filter(t => t.pnl > 0).length;
    const totalLosses = trades.filter(t => t.pnl <= 0).length;
    const winRate = (totalWins / trades.length) * 100;

    return { winRate, totalWins, totalLosses };
}

/**
 * Calculate Sharpe ratio from returns.
 *
 * @param {Array} returns - Array of period returns
 * @param {number} riskFreeRate - Annualized risk-free rate (default: 0.02)
 * @param {number} periodsPerYear - Trading periods per year (default: 252 for daily)
 * @returns {number} Sharpe ratio
 */
function calculateSharpeRatio(returns, riskFreeRate = 0.02, periodsPerYear = 252) {
    if (!returns || returns.length < 2) return 0;

    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return 0;

    const periodRiskFreeRate = riskFreeRate / periodsPerYear;
    const excessReturn = avgReturn - periodRiskFreeRate;

    return (excessReturn / stdDev) * Math.sqrt(periodsPerYear);
}

/**
 * Calculate maximum drawdown from equity curve.
 *
 * @param {Array} equityCurve - Array of equity values
 * @returns {Object} {maxDrawdown, maxDrawdownPercent, drawdownStart, drawdownEnd}
 */
function calculateMaxDrawdown(equityCurve) {
    if (!equityCurve || equityCurve.length < 2) {
        return { maxDrawdown: 0, maxDrawdownPercent: 0 };
    }

    let peak = equityCurve[0];
    let maxDrawdown = 0;
    let maxDrawdownPercent = 0;
    let drawdownStart = 0;
    let drawdownEnd = 0;
    let currentDrawdownStart = 0;

    for (let i = 0; i < equityCurve.length; i++) {
        const value = equityCurve[i];

        if (value > peak) {
            peak = value;
            currentDrawdownStart = i;
        }

        const drawdown = peak - value;
        const drawdownPercent = (drawdown / peak) * 100;

        if (drawdownPercent > maxDrawdownPercent) {
            maxDrawdown = drawdown;
            maxDrawdownPercent = drawdownPercent;
            drawdownStart = currentDrawdownStart;
            drawdownEnd = i;
        }
    }

    return {
        maxDrawdown,
        maxDrawdownPercent,
        drawdownStart,
        drawdownEnd,
    };
}

/**
 * Debounce function calls.
 *
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function calls.
 *
 * @param {Function} func - Function to throttle
 * @param {number} limit - Minimum time between calls in ms
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        calculatePositionPnL,
        calculatePositionRisk,
        calculatePortfolioMetrics,
        formatTimestamp,
        getSignalBadge,
        getRegimeBadge,
        calculateWinRate,
        calculateSharpeRatio,
        calculateMaxDrawdown,
        debounce,
        throttle,
    };
}
