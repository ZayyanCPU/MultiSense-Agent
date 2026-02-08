# ─────────────────────────────────────────────
# MultiSense Agent - Dockerfile
# Multi-stage build for production optimization
# ─────────────────────────────────────────────

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /tmp

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Metadata
LABEL maintainer="MultiSense Agent"
LABEL description="Multi-Modal AI Chatbot with WhatsApp Integration"
LABEL version="1.0.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/chroma /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); r.raise_for_status()"

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
