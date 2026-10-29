FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (git, ffmpeg, sox, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    sox \
    libsox-fmt-all \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install runpod, f5-tts without touching torch/torchaudio, and all required packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir runpod && \
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
    tqdm

COPY . /app

CMD ["python", "-u", "handler.py"]
