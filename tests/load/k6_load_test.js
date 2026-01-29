/**
 * K6 Load Testing Script for Trading API.
 *
 * Provides comprehensive load testing with:
 * - Smoke tests (basic functionality)
 * - Load tests (typical load)
 * - Stress tests (high load)
 * - Spike tests (sudden bursts)
 * - Soak tests (extended duration)
 *
 * Usage:
 *   # Run smoke test
 *   k6 run --env SCENARIO=smoke tests/load/k6_load_test.js
 *
 *   # Run load test
 *   k6 run --env SCENARIO=load tests/load/k6_load_test.js
 *
 *   # Run stress test
 *   k6 run --env SCENARIO=stress tests/load/k6_load_test.js
 *
 *   # Run with custom host
 *   k6 run --env BASE_URL=http://localhost:8000 tests/load/k6_load_test.js
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const dashboardLatency = new Trend('dashboard_latency');
const statusLatency = new Trend('status_latency');
const positionsLatency = new Trend('positions_latency');
const metricsLatency = new Trend('metrics_latency');
const requestsTotal = new Counter('requests_total');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const SCENARIO = __ENV.SCENARIO || 'load';

// Test scenarios
const scenarios = {
  smoke: {
    executor: 'constant-vus',
    vus: 1,
    duration: '30s',
  },
  load: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: 20 },   // Ramp up to 20 users
      { duration: '3m', target: 20 },   // Stay at 20 users
      { duration: '1m', target: 50 },   // Ramp up to 50 users
      { duration: '3m', target: 50 },   // Stay at 50 users
      { duration: '1m', target: 0 },    // Ramp down
    ],
  },
  stress: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '2m', target: 50 },   // Ramp up
      { duration: '5m', target: 50 },   // Hold
      { duration: '2m', target: 100 },  // Stress point
      { duration: '5m', target: 100 },  // Hold at stress
      { duration: '2m', target: 150 },  // Breaking point
      { duration: '5m', target: 150 },  // Hold
      { duration: '2m', target: 0 },    // Ramp down
    ],
  },
  spike: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: 10 },   // Normal load
      { duration: '10s', target: 200 }, // Spike!
      { duration: '1m', target: 200 },  // Stay at spike
      { duration: '10s', target: 10 },  // Drop back
      { duration: '2m', target: 10 },   // Recovery
      { duration: '1m', target: 0 },    // Ramp down
    ],
  },
  soak: {
    executor: 'constant-vus',
    vus: 30,
    duration: '30m',
  },
};

export const options = {
  scenarios: {
    default: scenarios[SCENARIO] || scenarios.load,
  },
  thresholds: {
    // Response time thresholds
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    dashboard_latency: ['p(95)<300'],
    status_latency: ['p(95)<200'],

    // Error rate thresholds
    errors: ['rate<0.01'],  // Less than 1% errors
    http_req_failed: ['rate<0.01'],
  },
};

// Helper function to make requests
function request(method, path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...options.headers,
    },
    tags: { endpoint: path },
    ...options.params,
  };

  let response;
  if (method === 'GET') {
    response = http.get(url, params);
  } else if (method === 'POST') {
    response = http.post(url, JSON.stringify(options.body || {}), params);
  }

  requestsTotal.add(1);
  return response;
}

// Main test function
export default function() {
  // Dashboard state - most critical endpoint
  group('Dashboard', () => {
    const start = Date.now();
    const response = request('GET', '/api/dashboard/unified-state');
    dashboardLatency.add(Date.now() - start);

    const success = check(response, {
      'dashboard: status 200': (r) => r.status === 200,
      'dashboard: has version': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.version !== undefined;
        } catch {
          return false;
        }
      },
      'dashboard: has total': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.total !== undefined;
        } catch {
          return false;
        }
      },
      'dashboard: response time < 500ms': (r) => r.timings.duration < 500,
    });

    errorRate.add(!success);
  });

  sleep(0.5);

  // Unified status
  group('Status', () => {
    const start = Date.now();
    const response = request('GET', '/api/unified/status');
    statusLatency.add(Date.now() - start);

    const success = check(response, {
      'status: status 200': (r) => r.status === 200,
      'status: response time < 300ms': (r) => r.timings.duration < 300,
    });

    errorRate.add(!success);
  });

  sleep(0.3);

  // Positions
  group('Positions', () => {
    const start = Date.now();
    const response = request('GET', '/api/unified/positions');
    positionsLatency.add(Date.now() - start);

    const success = check(response, {
      'positions: status 200': (r) => r.status === 200,
      'positions: is array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body);
        } catch {
          return false;
        }
      },
    });

    errorRate.add(!success);
  });

  sleep(0.3);

  // Health check
  group('Health', () => {
    const response = request('GET', '/health');

    const success = check(response, {
      'health: status 200': (r) => r.status === 200,
      'health: is healthy': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'healthy';
        } catch {
          return false;
        }
      },
    });

    errorRate.add(!success);
  });

  sleep(0.2);

  // Metrics (less frequent)
  if (Math.random() < 0.2) {  // 20% of iterations
    group('Metrics', () => {
      const start = Date.now();
      const response = request('GET', '/metrics');
      metricsLatency.add(Date.now() - start);

      const success = check(response, {
        'metrics: status 200': (r) => r.status === 200,
        'metrics: prometheus format': (r) => {
          return r.body && r.body.includes('# HELP') && r.body.includes('# TYPE');
        },
      });

      errorRate.add(!success);
    });
  }

  sleep(0.5);

  // API versions (occasional)
  if (Math.random() < 0.1) {  // 10% of iterations
    group('Versions', () => {
      const response = request('GET', '/api/versions');

      check(response, {
        'versions: status 200': (r) => r.status === 200,
      });
    });
  }

  // Variable wait based on "user behavior"
  sleep(Math.random() * 2 + 0.5);  // 0.5 to 2.5 seconds
}

// Setup function - runs once before test
export function setup() {
  console.log(`Running ${SCENARIO} test against ${BASE_URL}`);

  // Verify API is up
  const response = http.get(`${BASE_URL}/health`);
  if (response.status !== 200) {
    throw new Error(`API is not healthy: ${response.status}`);
  }

  return { startTime: Date.now() };
}

// Teardown function - runs once after test
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration.toFixed(1)} seconds`);
}

// Handle summary
export function handleSummary(data) {
  // Calculate summary metrics
  const summary = {
    scenario: SCENARIO,
    timestamp: new Date().toISOString(),
    duration_seconds: data.metrics.iteration_duration?.values?.avg / 1000 || 0,
    total_requests: data.metrics.http_reqs?.values?.count || 0,
    requests_per_second: data.metrics.http_reqs?.values?.rate || 0,
    error_rate: data.metrics.http_req_failed?.values?.rate || 0,
    response_times: {
      avg: data.metrics.http_req_duration?.values?.avg || 0,
      p95: data.metrics.http_req_duration?.values['p(95)'] || 0,
      p99: data.metrics.http_req_duration?.values['p(99)'] || 0,
    },
    thresholds_passed: Object.values(data.thresholds || {}).every(t => t.ok),
  };

  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'tests/load/results/summary.json': JSON.stringify(summary, null, 2),
  };
}

// Text summary helper (since k6/summary module may not be available)
function textSummary(data, options) {
  const { metrics, thresholds } = data;

  let output = '\n========== LOAD TEST SUMMARY ==========\n\n';

  // Request metrics
  if (metrics.http_reqs) {
    output += `Total Requests: ${metrics.http_reqs.values.count}\n`;
    output += `Requests/sec: ${metrics.http_reqs.values.rate?.toFixed(2)}\n`;
  }

  // Response time
  if (metrics.http_req_duration) {
    output += `\nResponse Times:\n`;
    output += `  Average: ${metrics.http_req_duration.values.avg?.toFixed(2)}ms\n`;
    output += `  P95: ${metrics.http_req_duration.values['p(95)']?.toFixed(2)}ms\n`;
    output += `  P99: ${metrics.http_req_duration.values['p(99)']?.toFixed(2)}ms\n`;
  }

  // Errors
  if (metrics.http_req_failed) {
    output += `\nError Rate: ${(metrics.http_req_failed.values.rate * 100)?.toFixed(2)}%\n`;
  }

  // Thresholds
  output += '\nThresholds:\n';
  for (const [name, threshold] of Object.entries(thresholds || {})) {
    const status = threshold.ok ? 'PASS' : 'FAIL';
    output += `  ${name}: ${status}\n`;
  }

  output += '\n========================================\n';

  return output;
}
