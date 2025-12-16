# Dockerfile for PaddleOCR-VL local OCR service
# Supports both CPU and GPU (NVIDIA CUDA) inference

ARG USE_GPU=false

# Use CUDA base image for GPU, slim for CPU
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu-base
FROM python:3.11-slim AS cpu-base

# Select the appropriate base
FROM ${USE_GPU:+gpu-base}${USE_GPU:-cpu-base} AS base

# Prevent Python from writing pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python (needed for CUDA image)
    python3 \
    python3-pip \
    python3-dev \
    # Required for PyMuPDF
    libmupdf-dev \
    mupdf-tools \
    # General build tools
    build-essential \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy requirements first for caching
COPY requirements.txt requirements-gpu.txt ./

# Install Python dependencies based on GPU flag
ARG USE_GPU=false
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir -r requirements-gpu.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

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
ENV MODEL_PATH=PaddlePaddle/PaddleOCR-VL
# Hugging Face cache directory
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with gunicorn for production
# High timeout for CPU inference, GPU is much faster
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "--workers", "1", "ocr_server:app"]
