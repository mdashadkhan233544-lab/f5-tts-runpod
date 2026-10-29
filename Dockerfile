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

# 1. Install specific CUDA PyTorch & Torchaudio
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install RunPod and F5-TTS with ALL required dependencies automatically
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir f5-tts --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install compatible Transformers, upgraded HuggingFace Hub & Kokoro
RUN pip install --no-cache-dir "transformers==4.46.3" "huggingface_hub>=0.26.0" --upgrade datasets scipy soundfile kokoro misaki[en]

COPY . /app

CMD ["python", "-u", "handler.py"]
