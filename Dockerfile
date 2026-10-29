FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch & matching Torchaudio install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# F5-TTS, RunPod & other dependencies
RUN pip install --no-cache-dir \
    f5-tts \
    runpod \
    soundfile

# Copy your handler file
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
