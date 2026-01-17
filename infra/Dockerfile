FROM python:3.11-slim

# Build arguments
ARG BUILD_DATE
ARG GIT_COMMIT
LABEL maintainer="algo-trading-lab"
LABEL build-date=$BUILD_DATE
LABEL git-commit=$GIT_COMMIT

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=trader:trader . .

# Create necessary directories
RUN mkdir -p /app/data/logs /app/data/models /app/data/cache \
    && chown -R trader:trader /app/data

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "bot.bot"]

