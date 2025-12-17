# Dockerfile for Local OCR service (CPU version)
# Uses Surya OCR (~300M params) - CPU and GPU friendly
# For GPU version with faster performance, use Dockerfile.gpu

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for PyMuPDF
    libmupdf-dev \
    mupdf-tools \
    # General build tools
    build-essential \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ocr_server.py .

# Set ownership
RUN chown -R appuser:appgroup /app

# Create cache directory for Hugging Face models
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appgroup /home/appuser

# Switch to non-root user
USER appuser

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0
ENV IDLE_TIMEOUT=300
# Note: Only Surya OCR is supported
# Hugging Face cache directory
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "--workers", "1", "ocr_server:app"]
