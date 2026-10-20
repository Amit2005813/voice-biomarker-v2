# ============================================================================
# Voice Biomarker Platform v2 - Production Dockerfile
# Multi-stage build for optimized image size and security
# ============================================================================

# ------------------------- Stage 1: Builder ----------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    liblapack-dev \
    libblas-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ------------------------- Stage 2: Runtime ----------------------------------
FROM python:3.11-slim AS runtime

# Runtime labels
LABEL maintainer="Voice Biomarker Team <team@voicebiomarker.com>" \
      version="2.0.0" \
      description="Voice Biomarker Platform v2 - Depression Detection via Voice Analysis"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    APP_HOST="0.0.0.0" \
    APP_PORT="8000" \
    APP_WORKERS="4" \
    APP_ENV="production" \
    LOG_LEVEL="info" \
    MODEL_PATH="/app/models" \
    DATA_PATH="/app/data" \
    UPLOAD_PATH="/app/uploads" \
    REDIS_URL="redis://redis:6379/0" \
    CORS_ORIGINS="*" \
    MAX_UPLOAD_SIZE=52428800 \
    HEALTH_CHECK_INTERVAL=30

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    liblapack3 \
    libblas3 \
    libgfortran5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Create application directories
RUN mkdir -p /app /app/models /app/data /app/uploads /app/logs \
    && chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY --chown=appuser:appuser ./src ./src
COPY --chown=appuser:appuser ./configs ./configs
COPY --chown=appuser:appuser ./requirements.txt .

# Copy entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info", \
     "--access-log"]
