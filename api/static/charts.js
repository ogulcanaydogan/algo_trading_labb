/**
 * Chart utilities for the trading dashboard.
 *
 * Provides reusable chart configurations and helpers for Chart.js.
 */

// Default chart colors
const CHART_COLORS = {
    profit: 'rgb(34, 197, 94)',      // Green
    loss: 'rgb(239, 68, 68)',        // Red
    neutral: 'rgb(148, 163, 184)',   // Gray
    primary: 'rgb(59, 130, 246)',    // Blue
    secondary: 'rgb(139, 92, 246)',  // Purple
    warning: 'rgb(245, 158, 11)',    // Orange

    // Semi-transparent versions
    profitAlpha: 'rgba(34, 197, 94, 0.2)',
    lossAlpha: 'rgba(239, 68, 68, 0.2)',
    primaryAlpha: 'rgba(59, 130, 246, 0.2)',
};

// Grid and axis styling
const CHART_GRID_CONFIG = {
    color: 'rgba(148, 163, 184, 0.1)',
    drawBorder: false,
};

const CHART_AXIS_CONFIG = {
    ticks: {
        color: 'rgb(148, 163, 184)',
        font: { size: 11 },
    },
    grid: CHART_GRID_CONFIG,
};

/**
 * Create an equity curve chart configuration.
 *
 * @param {Array} labels - X-axis labels (timestamps)
 * @param {Array} data - Equity values
 * @param {number} initialValue - Starting value for reference line
 * @returns {Object} Chart.js configuration
 */
function createEquityCurveConfig(labels, data, initialValue = null) {
    const datasets = [{
        label: 'Portfolio Value',
        data: data,
        borderColor: CHART_COLORS.primary,
        backgroundColor: CHART_COLORS.primaryAlpha,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHitRadius: 10,
    }];

    // Add reference line for initial value
    if (initialValue !== null) {
        datasets.push({
            label: 'Initial Capital',
            data: Array(data.length).fill(initialValue),
            borderColor: CHART_COLORS.neutral,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
        });
    }

    return {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: 'rgb(248, 250, 252)',
                    bodyColor: 'rgb(148, 163, 184)',
                    borderColor: 'rgba(148, 163, 184, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => `$${ctx.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`,
                    },
                },
            },
            scales: {
                x: {
                    ...CHART_AXIS_CONFIG,
                    display: true,
                    title: { display: false },
                },
                y: {
                    ...CHART_AXIS_CONFIG,
                    display: true,
                    ticks: {
                        ...CHART_AXIS_CONFIG.ticks,
                        callback: (value) => '$' + value.toLocaleString(),
                    },
                },
            },
        },
    };
}

/**
 * Create a PnL bar chart configuration.
 *
 * @param {Array} labels - X-axis labels (dates)
 * @param {Array} data - PnL values
 * @returns {Object} Chart.js configuration
 */
function createPnLBarConfig(labels, data) {
    return {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Daily P&L',
                data: data,
                backgroundColor: data.map(v => v >= 0 ? CHART_COLORS.profit : CHART_COLORS.loss),
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const value = ctx.parsed.y;
                            const sign = value >= 0 ? '+' : '';
                            return `${sign}$${value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                        },
                    },
                },
            },
            scales: {
                x: CHART_AXIS_CONFIG,
                y: {
                    ...CHART_AXIS_CONFIG,
                    ticks: {
                        ...CHART_AXIS_CONFIG.ticks,
                        callback: (value) => (value >= 0 ? '+' : '') + '$' + value.toLocaleString(),
                    },
                },
            },
        },
    };
}

/**
 * Create a drawdown chart configuration.
 *
 * @param {Array} labels - X-axis labels (timestamps)
 * @param {Array} data - Drawdown percentages (negative values)
 * @returns {Object} Chart.js configuration
 */
function createDrawdownConfig(labels, data) {
    return {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Drawdown',
                data: data,
                borderColor: CHART_COLORS.loss,
                backgroundColor: CHART_COLORS.lossAlpha,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y.toFixed(2)}%`,
                    },
                },
            },
            scales: {
                x: CHART_AXIS_CONFIG,
                y: {
                    ...CHART_AXIS_CONFIG,
                    max: 0,
                    ticks: {
                        ...CHART_AXIS_CONFIG.ticks,
                        callback: (value) => value.toFixed(1) + '%',
                    },
                },
            },
        },
    };
}

/**
 * Create a win rate donut chart configuration.
 *
 * @param {number} winRate - Win rate as percentage (0-100)
 * @returns {Object} Chart.js configuration
 */
function createWinRateDonutConfig(winRate) {
    const lossRate = 100 - winRate;

    return {
        type: 'doughnut',
        data: {
            labels: ['Wins', 'Losses'],
            datasets: [{
                data: [winRate, lossRate],
                backgroundColor: [CHART_COLORS.profit, CHART_COLORS.loss],
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '70%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
                    },
                },
            },
        },
    };
}

/**
 * Create a positions allocation pie chart configuration.
 *
 * @param {Array} positions - Array of {symbol, value} objects
 * @returns {Object} Chart.js configuration
 */
function createPositionAllocationConfig(positions) {
    const colors = [
        CHART_COLORS.primary,
        CHART_COLORS.profit,
        CHART_COLORS.warning,
        CHART_COLORS.secondary,
        CHART_COLORS.loss,
        CHART_COLORS.neutral,
    ];

    return {
        type: 'pie',
        data: {
            labels: positions.map(p => p.symbol),
            datasets: [{
                data: positions.map(p => p.value),
                backgroundColor: positions.map((_, i) => colors[i % colors.length]),
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: 'rgb(148, 163, 184)',
                        padding: 10,
                        font: { size: 11 },
                    },
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const value = ctx.parsed;
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((value / total) * 100).toFixed(1);
                            return `${ctx.label}: $${value.toLocaleString()} (${pct}%)`;
                        },
                    },
                },
            },
        },
    };
}

/**
 * Format a number as currency.
 *
 * @param {number} value - The value to format
 * @param {string} currency - Currency symbol (default: '$')
 * @returns {string} Formatted currency string
 */
function formatCurrency(value, currency = '$') {
    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';
    return sign + currency + absValue.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

/**
 * Format a number as percentage.
 *
 * @param {number} value - The value to format
 * @param {boolean} showSign - Whether to show + for positive values
 * @returns {string} Formatted percentage string
 */
function formatPercent(value, showSign = true) {
    const sign = showSign && value > 0 ? '+' : '';
    return sign + value.toFixed(2) + '%';
}

/**
 * Get color based on value (positive = green, negative = red).
 *
 * @param {number} value - The value to check
 * @returns {string} CSS color string
 */
function getPnLColor(value) {
    if (value > 0) return CHART_COLORS.profit;
    if (value < 0) return CHART_COLORS.loss;
    return CHART_COLORS.neutral;
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CHART_COLORS,
        createEquityCurveConfig,
        createPnLBarConfig,
        createDrawdownConfig,
        createWinRateDonutConfig,
        createPositionAllocationConfig,
        formatCurrency,
        formatPercent,
        getPnLColor,
    };
}
