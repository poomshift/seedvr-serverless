# CUDA 12.8 base matches SeedVR README Apex wheel and common GPUs
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Symlink python (force overwrite if exists)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Torch pinned to cu121; comfy readme uses 2.6/cu126 but SeedVR main uses 2.4 era.
# 2.4.0/cu121 is a safe midpoint for serverless; comfy reqs tolerate it.
RUN pip install --upgrade pip
RUN pip install --no-cache torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# APEX: SeedVR README references apex 0.1 wheels for py310/torch2.4/cu12.1
# If apex fails, feel free to omit; the comfy CLI removed flash-attn dependency and does not strictly require apex.
# Keeping it optional.
# RUN pip install https://huggingface.co/bytedance-seed/SeedVR/resolve/main/apex-0.1-cp310-cp310-linux_x86_64.whl

# Worker deps
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Source
COPY handler.py /workspace/handler.py

# Create output dir
RUN mkdir -p /workspace/outputs

# Runpod serverless
CMD ["python", "-u", "/workspace/handler.py"]