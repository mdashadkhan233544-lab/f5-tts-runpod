FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    sox \
    libsox-fmt-all \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Install specific CUDA PyTorch & Torchaudio wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install RunPod and F5-TTS with all official requirements (preventing torch overwrite)
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir --no-deps f5-tts && \
    pip install --no-cache-dir \
    torchdiffeq \
    transformers \
    accelerate \
    soundfile \
    librosa \
    jieba \
    rjieba \
    pypinyin \
    anyascii \
    vocos \
    cached_path \
    ema-pytorch \
    wandb \
    tomli \
    hydra-core \
    omegaconf \
    matplotlib \
    scipy \
    tqdm \
    pydub \
    torchaudio-augmentations \
    x-transformers \
    click

COPY . /app

CMD ["python", "-u", "handler.py"]
