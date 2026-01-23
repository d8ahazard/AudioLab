# AudioLab Docker Image
# =====================
# Multi-stage build for optimized container size
#
# Build:
#   docker build -t audiolab:latest .
#
# Run:
#   docker run -p 7860:7860 --gpus all audiolab:latest

# =============================================================================
# Stage 1: Base image with CUDA and system dependencies
# =============================================================================
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    espeak-ng \
    libespeak-ng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip wheel setuptools

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-core.txt requirements-cuda.txt requirements-wheels.txt ./
COPY wheels/ ./wheels/

# Install PyTorch with CUDA support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install custom wheels first
RUN pip install -r requirements-wheels.txt || true

# Install core dependencies
RUN pip install -r requirements-core.txt

# Install CUDA-specific dependencies
RUN pip install triton>=3.0.0 flash_attn>=2.7.4 || true

# =============================================================================
# Stage 3: Application
# =============================================================================
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/outputs /app/models /app/temp_uploads /app/logs

# Set up model cache directories
ENV HF_HOME=/app/models/hf
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV TTS_HOME=/app/models
ENV COQUI_TOS_AGREED=1

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "main.py", "--listen"]

# =============================================================================
# Stage 4: Development image (optional)
# =============================================================================
FROM application AS development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Set development environment
ENV AUDIOLAB_ENV=development
ENV GRADIO_DEBUG=1

# Override command for development
CMD ["python", "main.py", "--listen"]
