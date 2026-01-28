# Multi-stage build for smaller image
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY bot/ ./bot/
COPY api/ ./api/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bot; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
