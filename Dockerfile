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

# Install matching Torch and Torchaudio with CUDA 12.1 support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install runpod, f5-tts without replacing torch, along with all inference requirements
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir --no-deps f5-tts && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    soundfile \
    librosa \
    jieba \
    pypinyin \
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
    x-transformers

COPY . /app

CMD ["python", "-u", "handler.py"]
