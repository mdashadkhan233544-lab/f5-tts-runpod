FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (git, ffmpeg, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Force install matching torch and torchaudio versions for CUDA 12.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install f5-tts and all inference dependencies
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
    tqdm

COPY . /app

CMD ["python", "-u", "handler.py"]
