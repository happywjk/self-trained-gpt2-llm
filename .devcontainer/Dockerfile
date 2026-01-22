# Devcontainer base image with PyTorch + CUDA (works on CPU too)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # NCCL reasonable defaults for single-node
    NCCL_P2P_DISABLE=0 \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_DEBUG=WARN

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates openssh-client \
    build-essential cmake \
    vim nano htop \
    && rm -rf /var/lib/apt/lists/*

# Optional: ensure a clean, modern pip
RUN python -m pip install --upgrade pip

# Copy requirements if present and install (torch already in base image)
WORKDIR /workspace
COPY ../requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ]; then pip install -r /tmp/requirements.txt; fi

# Default workdir for Codespaces/Dev Containers
WORKDIR /workspaces/self-trained-gpt2-llm
