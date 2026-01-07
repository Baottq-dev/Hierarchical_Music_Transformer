FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ninja-build \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Mamba dependencies
RUN pip install packaging
RUN pip install causal-conv1d>=1.2.0
RUN pip install mamba-ssm

# Install ML dependencies
RUN pip install miditok>=3.0.0
RUN pip install transformers>=4.35.0
RUN pip install scipy numpy
RUN pip install hydra-core omegaconf
RUN pip install tensorboard wandb
RUN pip install tqdm

# Setup working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Install project in editable mode
RUN pip install -e .

CMD ["bash"]
