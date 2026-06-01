# Dockerfile for running platelet-seg with optional NVIDIA GPU support
# Base image contains CUDA runtime + cuDNN. If you do not have a GPU
# or do not want the CUDA image, use the CPU fallback note below.

# GPU base image
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Notes:
# - This image includes the CUDA runtime; it is intended for GPU hosts.
# - CPU-only fallback: replace the first line with:
#     FROM python:3.10-slim
#   and remove GPU-specific runtime and device configurations in docker-compose.

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies and Python 3.10 + pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       python3.10 \
       python3.10-venv \
       python3-pip \
       build-essential \
       git \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure /usr/bin/python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 || true

# Upgrade pip and install minimal tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache if file unchanged
COPY requirements.txt /app/requirements.txt

# Install Python dependencies if requirements.txt exists
RUN if [ -f /app/requirements.txt ]; then \
      python3 -m pip install -r /app/requirements.txt; \
    else \
      echo "No requirements.txt found, skipping pip install."; \
    fi

# Copy project files
COPY . /app

# Expose Flask port
EXPOSE 5000

# Environment
ENV PYTHONUNBUFFERED=1

# Default command: run Flask app entrypoint. This file (`src/api/base.py`) constructs
# and runs the Flask app when executed directly. Adjust if you use a WSGI server.
CMD ["python3", "src/api/base.py"]
