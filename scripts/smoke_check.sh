#!/usr/bin/env bash
# Simple smoke checks for local services
set -euo pipefail

echo 'GET /'
curl -sS -w '\nHTTP_CODE:%{http_code}\n' http://localhost:8000/

echo
s="GET /dashboard"
curl -sS -w '\nHTTP_CODE:%{http_code}\n' http://localhost:8000/dashboard

echo
s="GET /status"
curl -sS -w '\nHTTP_CODE:%{http_code}\n' http://localhost:8000/status

echo
s="List containers"
docker compose ps --services --filter status=running
